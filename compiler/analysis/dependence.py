"""Scalar lifetime checks and conservative ISL parallel-legality analysis.

The original execution order is retained. Ordered memory conflicts are sufficient
for proving independence; no value-based dependence removal or scheduling is done.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import islpy as isl

from compiler.ir import (
    ArrayAccess,
    Assignment,
    Binary,
    Block,
    CompilationError,
    Expr,
    If,
    IntrinsicCall,
    Literal,
    Loop,
    ParallelRegion,
    Reference,
    RegionReport,
    ScalarType,
    Size,
    SourceLocation,
    Symbol,
    Unary,
    block_writes,
    referenced_symbols,
    statement_reads,
    walk_expr,
)
from compiler.ir.integers import fits_integer

from .effects import (
    all_loops as _all_loops,
)
from .effects import (
    assignments as _assignments,
)
from .effects import (
    header_reads as _header_reads,
)
from .effects import (
    mapped_prefix as _mapped_prefix,
)
from .effects import (
    upward_reads as _upward_reads,
)
from .proof import ParallelizationError, ParallelProof


@dataclass(frozen=True)
class Affine:
    constant: int = 0
    terms: tuple[tuple[str, int], ...] = ()

    def add(self, other: Affine) -> Affine:
        terms = dict(self.terms)
        for name, coefficient in other.terms:
            terms[name] = terms.get(name, 0) + coefficient
        return Affine(self.constant + other.constant, tuple(sorted((k, v) for k, v in terms.items() if v)))

    def scale(self, factor: int) -> Affine:
        return Affine(self.constant * factor, tuple((k, v * factor) for k, v in self.terms if v * factor))

    def __str__(self) -> str:
        return " + ".join([str(self.constant), *(f"{value}*{name}" for name, value in self.terms)])


def affine_expression(
    expression: Expr,
    environment: dict[Symbol, Affine],
    iterators: dict[Symbol, Affine],
    location: SourceLocation,
) -> Affine:
    if isinstance(expression, Literal) and expression.dtype is ScalarType.INTEGER:
        return Affine(int(expression.value))
    if isinstance(expression, Reference):
        if expression.symbol.dtype is not ScalarType.INTEGER or expression.symbol.rank:
            raise CompilationError("Affine indices and bounds require integer scalar values", location)
        value = iterators.get(expression.symbol, environment.get(expression.symbol))
        if value is None:
            raise CompilationError(f"'{expression.symbol.name}' is not a defined invariant affine integer", location)
        return value
    if isinstance(expression, Size):
        return Affine(terms=((f"d{expression.symbol.id}_{expression.dimension}", 1),))
    if isinstance(expression, Unary) and expression.operator in {"+", "-"}:
        value = affine_expression(expression.operand, environment, iterators, location)
        return value.scale(-1 if expression.operator == "-" else 1)
    if isinstance(expression, Binary) and expression.operator in {"+", "-", "*", "/"}:
        left = affine_expression(expression.left, environment, iterators, location)
        right = affine_expression(expression.right, environment, iterators, location)
        if expression.operator == "+":
            return left.add(right)
        if expression.operator == "-":
            return left.add(right.scale(-1))
        if expression.operator == "/":
            if not left.terms and not right.terms and right.constant:
                quotient = abs(left.constant) // abs(right.constant)
                return Affine(-quotient if (left.constant < 0) != (right.constant < 0) else quotient)
        elif not left.terms:
            return right.scale(left.constant)
        elif not right.terms:
            return left.scale(right.constant)
    raise CompilationError("Unsupported non-affine array index or loop bound", location)


def ordered_conflicts(reads: isl.UnionMap, writes: isl.UnionMap, schedule: isl.UnionMap):
    """Relations map earlier statement instances to later conflicting instances.

    Only statements that access arrays can conflict, so only their order is
    worked out: a body with many scalar statements (inlined procedures, say)
    would otherwise order every pair of its statements.
    """
    schedule = schedule.intersect_domain(reads.domain().union(writes.domain()))
    before = schedule.lex_lt_union_map(schedule)
    return {
        "RAW": writes.apply_range(reads.reverse()).intersect(before),
        "WAR": reads.apply_range(writes.reverse()).intersect(before),
        "WAW": writes.apply_range(writes.reverse()).intersect(before),
    }


@dataclass(frozen=True)
class _Event:
    origin: Assignment | Loop | If
    coordinates: tuple[str, ...]
    conditions: tuple[str, ...]
    order: tuple[str, ...]
    order_conditions: tuple[str, ...]
    reads: tuple[ArrayAccess, ...]
    writes: tuple[ArrayAccess, ...]
    environment: dict[Symbol, Affine]
    expressions: dict[Symbol, Expr]
    active: dict[Symbol, Affine]


@dataclass(frozen=True)
class _AffineRegionModel:
    """Unaccepted original event model, never parallel-execution authority."""

    region: ParallelRegion
    events: tuple[_Event, ...]
    parallel: isl.UnionMap
    conflicts: tuple[tuple[str, isl.UnionMap], ...]
    # What a parameter of the relations stands for, where that is a value known when the region starts.
    known: tuple[tuple[str, Expr], ...] = ()

    def independence_condition(self):
        """A condition on the values the region starts with under which no iteration conflicts with another.

        The conflict relations say for which parameter values two iterations
        touch the same element. Outside those values the region is independent.
        A case in which one of the region's loops has a single iteration is
        left out: such a condition would only say that a loop is too short to
        conflict with itself.
        Returns the condition in ISL syntax and as an expression, or None when
        nothing is left, or when the condition needs a value that is not known
        when the region starts.
        """
        same_iteration = self.parallel.apply_range(self.parallel.reverse())
        conflicting = None
        for _kind, relation in self.conflicts:
            bad = relation.subtract(same_iteration)
            if not bad.is_empty():
                conflicting = bad.params() if conflicting is None else conflicting.union(bad.params())
        if conflicting is None:
            return None
        free = isl.Set.universe(conflicting.get_space()).subtract(conflicting).coalesce()
        if free.is_empty():
            return None
        known = dict(self.known)
        iterations = isl.Set.from_union_set(self.parallel.range())
        pairs = isl.Map.from_domain_and_range(iterations, iterations)
        # Per loop of the region: the pairs of iterations that differ along it.
        apart = [pairs.order_lt(isl.dim_type.in_, axis, isl.dim_type.out, axis).union(
                     pairs.order_gt(isl.dim_type.in_, axis, isl.dim_type.out, axis))
                 for axis in range(iterations.dim(isl.dim_type.set))]
        apart = [item for item in apart if not item.is_empty()]
        texts, expressions = [], []
        for basic in free.get_basic_sets():
            if any(item.intersect_params(isl.Set.from_basic_set(basic)).is_empty() for item in apart):
                continue
            names = [basic.get_dim_name(isl.dim_type.param, index) for index in range(basic.dim(isl.dim_type.param))]
            parts, terms = [], []
            for constraint in basic.get_constraints():
                if basic.dim(isl.dim_type.div) and constraint.involves_dims(isl.dim_type.div, 0, basic.dim(isl.dim_type.div)):
                    return None
                coefficients = [(name, constraint.get_coefficient_val(isl.dim_type.param, index).to_python())
                                for index, name in enumerate(names)]
                coefficients = [(name, value) for name, value in coefficients if value]
                if any(name not in known for name, _value in coefficients):
                    return None
                constant = constraint.get_constant_val().to_python()
                if not fits_integer(constant) or any(not fits_integer(value) for _name, value in coefficients):
                    return None
                relation = "=" if constraint.is_equality() else ">="
                parts.append("(" + " + ".join([str(constant), *(f"{value}*{name}" for name, value in coefficients)])
                             + f" {relation} 0)")
                total: Expr = Literal(str(constant), ScalarType.INTEGER)
                for name, value in coefficients:
                    total = Binary("+", total, Binary("*", Literal(str(value), ScalarType.INTEGER), known[name]))
                terms.append(Binary("==" if constraint.is_equality() else ">=", total, Literal("0", ScalarType.INTEGER)))
            if not terms:
                return None
            texts.append("(" + " and ".join(parts) + ")")
            expression = terms[0]
            for term in terms[1:]:
                expression = Binary(".and.", expression, term)
            expressions.append(expression)
        if not expressions:
            return None
        condition = expressions[0]
        for expression in expressions[1:]:
            condition = Binary(".or.", condition, expression)
        return "(" + " or ".join(texts) + ")", condition

    def require_unconditional(self):
        same_iteration = self.parallel.apply_range(self.parallel.reverse())
        for kind, relation in self.conflicts:
            bad = relation.subtract(same_iteration)
            if not bad.is_empty():
                mapping = bad.get_map_list().get_at(0)
                source = self.events[int(mapping.get_tuple_name(isl.dim_type.in_)[1:])]
                sink = self.events[int(mapping.get_tuple_name(isl.dim_type.out)[1:])]
                names = ", ".join(sorted({access.symbol.name for access in (
                    *source.reads, *source.writes, *sink.reads, *sink.writes)}))
                conservative = self.region.report.conservative
                raise ParallelizationError(
                    f"Cannot parallelize: {kind} {'possible ' if conservative else ''}loop-carried conflict involving {names}; "
                    + ("conservative dependence model; " if conservative else "")
                    + f"accesses at {source.origin.location} and {sink.origin.location}; "
                    f"relation {mapping}; witness {mapping.wrap().sample_point()}",
                    self.region.loops[0].location,
                    witness=str(mapping.wrap().sample_point()),
                    conservative=conservative,
                )
        return self.region


def _resolve(expression: Expr, definitions: dict[Symbol, Expr]) -> Expr:
    if isinstance(expression, Reference):
        return definitions.get(expression.symbol, expression)
    if isinstance(expression, Binary):
        return Binary(
            expression.operator, _resolve(expression.left, definitions), _resolve(expression.right, definitions)
        )
    if isinstance(expression, Unary):
        return Unary(expression.operator, _resolve(expression.operand, definitions))
    if isinstance(expression, IntrinsicCall):
        return IntrinsicCall(
            expression.name, tuple(_resolve(arg, definitions) for arg in expression.arguments), expression.dtype
        )
    if isinstance(expression, ArrayAccess):
        return ArrayAccess(expression.symbol, tuple(_resolve(index, definitions) for index in expression.indices))
    return expression


def _region_model(
    region_id: int,
    loop: Loop,
    environment: dict[Symbol, Affine],
    defined: set[Symbol],
    later: Block,
    *,
    injective=frozenset(),
    assume=(),
) -> _AffineRegionModel:
    context = isl.DEFAULT_CONTEXT
    loops, body = _mapped_prefix(loop)
    assignments = tuple(_assignments(body))
    sequential = tuple(_all_loops(body))
    mapped = {item.iterator for item in loops}
    scalar_writes = {stmt.target.symbol for stmt in assignments if isinstance(stmt.target, Reference)}
    private = scalar_writes | {item.iterator for item in sequential}
    if scalar_writes & mapped:
        raise CompilationError("Assignments to induction variables are unsupported", loop.location)
    for symbol in private:
        if symbol.parameter:
            raise CompilationError(f"Cannot privatize dummy argument '{symbol.name}'", loop.location)
    live_out = (private | mapped) & _upward_reads(later)
    if live_out:
        symbol = min(live_out, key=lambda s: s.id)
        raise ParallelizationError(f"Loop-written scalar '{symbol.name}' is live after the region", loop.location)

    captured: set[Symbol] = set()
    events: list[_Event] = []
    axis_number = 0
    conservative = False
    parameters: set[str] = set()
    iterators: dict[Symbol, Affine] = {}

    def check_reads(symbols, available, location):
        for symbol in sorted(symbols, key=lambda s: s.id):
            if symbol.rank:
                captured.add(symbol)
            elif symbol in private:
                if symbol not in available:
                    raise ParallelizationError(
                        f"Scalar '{symbol.name}' is read before its per-iteration definition (reduction or carried value)",
                        location,
                    )
            elif symbol not in available:
                if symbol not in defined:
                    raise CompilationError(f"Scalar '{symbol.name}' is read before definition", location)
                captured.add(symbol)

    def affine_or_unknown(expression, values, active, location):
        try:
            value = affine_expression(expression, values, active, location)
        except CompilationError:
            return None
        parameters.update(name for name, _ in value.terms if not name.startswith("q"))
        return value

    def affine_predicate(expression, values, active, location):
        """Exact integer predicates constrain the events in each branch.

        Array/real/unknown predicates retain the enclosing domain. Predicate
        reads themselves are always evaluated in that enclosing domain.
        """
        if isinstance(expression, Unary) and expression.operator.lower() in {".not.", "not"}:
            return "not (" + affine_predicate(expression.operand, values, active, location) + ")"
        if not isinstance(expression, Binary):
            raise CompilationError("Non-affine predicate", location)
        operator = expression.operator.lower()
        if operator in {".and.", "and", ".or.", "or"}:
            left = affine_predicate(expression.left, values, active, location)
            right = affine_predicate(expression.right, values, active, location)
            return "(" + left + ") " + ("and" if operator in {".and.", "and"} else "or") + " (" + right + ")"
        relation = {"==": "=", ".eq.": "=", "/=": "!=", "!=": "!=", ".ne.": "!=",
                    "<": "<", ".lt.": "<", "<=": "<=", ".le.": "<=",
                    ">": ">", ".gt.": ">", ">=": ">=", ".ge.": ">="}.get(operator)
        if relation is None:
            raise CompilationError("Non-affine predicate", location)
        left = affine_expression(expression.left, values, active, location)
        right = affine_expression(expression.right, values, active, location)
        parameters.update(name for value in (left, right) for name, _ in value.terms if not name.startswith("q"))
        return "(" + str(left) + ") " + relation + " (" + str(right) + ")"

    def constraints(item, values, active, is_mapped):
        nonlocal axis_number, conservative
        axis = f"q{axis_number}"
        axis_number += 1
        axes = []
        bounds = []
        for suffix, expression in (("lower", item.lower), ("upper", item.upper), ("step", item.step_expression())):
            value = affine_or_unknown(expression, values, active, item.location)
            if value is None and (is_mapped or not (referenced_symbols(expression) & (set(active) | private))):
                name = f"b{region_id}_{axis}_{suffix}"
                parameters.add(name)
                value = Affine(terms=((name, 1),))
            bounds.append(value)
        lower, upper, step = bounds
        conservative |= any(value is None for value in bounds)
        order_axis = f"r{axis_number - 1}"
        if step is not None and not step.terms:
            if step.constant == 0:
                raise CompilationError("DO stride cannot be zero", item.location)
            sign = 1 if step.constant > 0 else -1
            if lower is not None:
                axes.append(f"{axis} {'>=' if sign > 0 else '<='} ({lower})")
                if abs(step.constant) > 1:
                    axes.append(f"({axis} - ({lower})) mod {abs(step.constant)} = 0")
            if upper is not None:
                axes.append(f"{axis} {'<=' if sign > 0 else '>='} ({upper})")
            order = axis if sign > 0 else f"-{axis}"
            order_conditions = []
        elif step is not None:
            conservative = True
            positive = [f"({step}) > 0"]
            negative = [f"({step}) < 0"]
            if lower is not None:
                positive.append(f"{axis} >= ({lower})")
                negative.append(f"{axis} <= ({lower})")
            if upper is not None:
                positive.append(f"{axis} <= ({upper})")
                negative.append(f"{axis} >= ({upper})")
            axes.append(f"(({' and '.join(positive)}) or ({' and '.join(negative)}))")
            order = order_axis
            order_conditions = [
                f"((({step}) > 0 and {order_axis} = {axis}) or (({step}) < 0 and {order_axis} = -{axis}))"
            ]
        else:
            # An unknown sequential stride/bound only widens this thread's domain.
            # Across-thread order is determined by the mapped prefix before it.
            order = axis
            order_conditions = []
        return axis, tuple(axes), order, tuple(order_conditions)

    coordinates: tuple[str, ...] = ()
    conditions: tuple[str, ...] = ()
    order: tuple[str, ...] = ()
    order_conditions: tuple[str, ...] = ()
    available: set[Symbol] = set()
    for item in loops:
        if item is not loops[0] and _header_reads(item) & (mapped - available):
            raise ParallelizationError("Nonrectangular mapped loop prefix", item.location)
        for symbol in _header_reads(item):
            if not symbol.rank and symbol not in defined and symbol not in available:
                raise CompilationError(f"Scalar '{symbol.name}' is read before definition", item.location)
        captured.update(_header_reads(item))
        axis, extra, time, time_constraints = constraints(item, environment, iterators, True)
        coordinates += (axis,)
        conditions += extra
        order += (time,)
        order_conditions += time_constraints
        iterators[item.iterator] = Affine(terms=((axis, 1),))
        available.add(item.iterator)

    def visit(block, values, definitions, available, active, coords, domain, position, timing):
        nonlocal conservative
        values = dict(values)
        definitions = dict(definitions)
        available = set(available)
        for index, statement in enumerate(block.statements):
            path = (*position, str(index))
            if isinstance(statement, Assignment):
                check_reads(statement_reads(statement), available, statement.location)
                read_expressions = [statement.value]
                if isinstance(statement.target, ArrayAccess):
                    read_expressions.extend(statement.target.indices)
                    captured.add(statement.target.symbol)
                reads = tuple(
                    node for expr in read_expressions for node in walk_expr(expr) if isinstance(node, ArrayAccess)
                )
                writes = (statement.target,) if isinstance(statement.target, ArrayAccess) else ()
                events.append(
                    _Event(
                        statement,
                        coords,
                        domain,
                        path,
                        timing,
                        reads,
                        writes,
                        dict(values),
                        dict(definitions),
                        dict(active),
                    )
                )
                if isinstance(statement.target, Reference):
                    symbol = statement.target.symbol
                    available.add(symbol)
                    resolved = _resolve(statement.value, definitions)
                    for key in [k for k, expr in definitions.items() if symbol in referenced_symbols(expr)]:
                        definitions.pop(key, None)
                    definitions[symbol] = resolved
                    if symbol.dtype is ScalarType.INTEGER:
                        value = affine_or_unknown(statement.value, values, active, statement.location)
                        if value is None:
                            values.pop(symbol, None)
                        else:
                            values[symbol] = value
                else:
                    written = statement.target.symbol
                    for key in [k for k, expr in definitions.items() if written in referenced_symbols(expr)]:
                        definitions.pop(key, None)
            elif isinstance(statement, If):
                try:
                    predicate = affine_predicate(statement.condition, values, active, statement.location)
                except CompilationError:
                    predicate = None
                    conservative = True
                check_reads(referenced_symbols(statement.condition), available, statement.location)
                predicate_reads = tuple(
                    node for node in walk_expr(statement.condition) if isinstance(node, ArrayAccess)
                )
                if predicate_reads:
                    events.append(
                        _Event(
                            statement,
                            coords,
                            domain,
                            (*path, "0"),
                            timing,
                            predicate_reads,
                            (),
                            dict(values),
                            dict(definitions),
                            dict(active),
                        )
                    )
                then_available = visit(
                    statement.then_body, values, definitions, available, active, coords,
                    domain if predicate is None else (*domain, "(" + predicate + ")"), (*path, "1"), timing
                )
                else_available = visit(
                    statement.else_body, values, definitions, available, active, coords,
                    domain if predicate is None else (*domain, "not (" + predicate + ")"), (*path, "2"), timing
                )
                available.update(then_available & else_available)
                # Branch-local values cannot be substituted after the join.
                mutated = block_writes(statement.then_body) | block_writes(statement.else_body)
                for symbol in mutated:
                    values.pop(symbol, None)
                    definitions.pop(symbol, None)
                for key in [k for k, expr in definitions.items() if referenced_symbols(expr) & mutated]:
                    definitions.pop(key, None)
            else:
                check_reads(_header_reads(statement), available, statement.location)
                expressions = (statement.lower, statement.upper, statement.step_expression())
                reads = tuple(node for expr in expressions for node in walk_expr(expr) if isinstance(node, ArrayAccess))
                if reads:
                    events.append(
                        _Event(
                            statement,
                            coords,
                            domain,
                            (*path, "0"),
                            timing,
                            reads,
                            (),
                            dict(values),
                            dict(definitions),
                            dict(active),
                        )
                    )
                axis, extra, time, time_constraints = constraints(statement, values, active, False)
                nested = dict(active)
                nested[statement.iterator] = Affine(terms=((axis, 1),))
                # Scalar recurrences are serial within this mapped iteration,
                # but their values cannot be modeled as one affine substitution
                # reused for every sequential iteration.
                carried = block_writes(statement.body) & _upward_reads(statement.body)
                body_values = {symbol: value for symbol, value in values.items() if symbol not in carried}
                body_definitions = {
                    symbol: value
                    for symbol, value in definitions.items()
                    if symbol not in carried
                    and symbol != statement.iterator
                    and not (referenced_symbols(value) & (carried | {statement.iterator}))
                }
                visit(
                    statement.body,
                    body_values,
                    body_definitions,
                    available | {statement.iterator},
                    nested,
                    (*coords, axis),
                    (*domain, *extra),
                    (*path, "1", time),
                    (*timing, *time_constraints),
                )
                # A zero-trip body cannot define a scalar. Its iterator is still
                # assigned the initial value, but its final affine value is unknown.
                available.add(statement.iterator)
                mutated = block_writes(statement.body) | {statement.iterator}
                for symbol in mutated:
                    values.pop(symbol, None)
                    definitions.pop(symbol, None)
                for key in [k for k, expr in definitions.items() if referenced_symbols(expr) & mutated]:
                    definitions.pop(key, None)
        return available

    visit(body, environment, {}, available, iterators, coordinates, conditions, order, order_conditions)
    start_values = {symbol for symbol in environment if symbol.parameter and symbol.dtype is ScalarType.INTEGER
                    and environment[symbol] == Affine(terms=((f"p{symbol.id}", 1),))}
    unknown: dict[Expr, str] = {}

    def fixed(expression):
        """Built from nothing but total integer operations on values the region starts with."""
        nodes = tuple(walk_expr(expression))
        if any(isinstance(node, (ArrayAccess, Size)) for node in nodes):
            return False
        if any((isinstance(node, Binary) and node.operator == "/")
               or (isinstance(node, IntrinsicCall) and node.name.lower() in {"mod", "modulo"})
               for node in nodes):
            return False
        return bool(referenced_symbols(expression)) and referenced_symbols(expression) <= start_values

    def unknown_value(expression):
        name = unknown.setdefault(expression, f"v{region_id}_{len(unknown)}")
        parameters.add(name)
        return Affine(terms=((name, 1),))

    specifications = []
    for index, event in enumerate(events):
        active = event.active
        for access_kind, accesses in (("read", event.reads), ("write", event.writes)):
            for access in accesses:
                resolved = tuple(_resolve(expr, event.expressions) for expr in access.indices)
                indices = tuple(
                    affine_or_unknown(expr, event.environment, active, event.origin.location) for expr in access.indices
                )
                # A subscript that is not affine but does not change while the region runs is one
                # unknown value: accesses through the same expression meet, through another they may.
                indices = tuple(value if value is not None or not fixed(expression) else unknown_value(expression)
                                for value, expression in zip(indices, resolved, strict=True))
                # A later injectivity certificate may use affine operands of
                # this otherwise nonlinear expression. Collect their parameters
                # before constructing any ISL spaces.
                for expression in resolved:
                    for node in walk_expr(expression):
                        affine_or_unknown(node, event.environment, active, event.origin.location)
                specifications.append((index, access_kind, access.symbol, resolved, indices, active))

    parameter_text = f"[{','.join(sorted(parameters))}] -> " if parameters else ""
    extent_constraints = tuple(f"{name} >= 0" for name in sorted(parameters) if name.startswith("d"))

    def event_domain(event):
        return " and ".join((*event.conditions, *extent_constraints, *assume)) or "true"

    def instance(index):
        return f"S{index}[{','.join(events[index].coordinates)}]"

    def union_map(parts):
        return isl.UnionMap.read_from_str(context, parameter_text + "{ " + "; ".join(parts) + " }")

    # Non-affine coordinates are conservatively unconstrained. An identical
    # square of an affine expression is injective on a proven single-sign domain;
    # in that case equality of addresses is exactly equality of its input axis.
    replacements = {}
    groups = {}
    for spec in specifications:
        for dimension, expression in enumerate(spec[3]):
            groups.setdefault((spec[2], dimension), []).append((spec, expression))
    for key, entries in groups.items():
        if not all(entry[1] == entries[0][1] for entry in entries):
            continue
        expression = entries[0][1]
        if not isinstance(expression, Binary) or expression.operator != "*" or expression.left != expression.right:
            continue
        candidates = []
        for spec, _ in entries:
            index, _, _, _, _, active = spec
            value = affine_or_unknown(expression.left, events[index].environment, active, events[index].origin.location)
            if value is None:
                break
            axes = [(name, coefficient) for name, coefficient in value.terms if name.startswith("q")]
            if len(axes) != 1:
                break
            domain = isl.Set.read_from_str(
                context,
                parameter_text
                + "{ ["
                + ",".join(events[index].coordinates)
                + "] : "
                + event_domain(events[index])
                + " }",
            )
            nonnegative = isl.Set.read_from_str(
                context,
                parameter_text + "{ [" + ",".join(events[index].coordinates) + f"] : ({value}) >= 0 }}",
            )
            nonpositive = isl.Set.read_from_str(
                context,
                parameter_text + "{ [" + ",".join(events[index].coordinates) + f"] : ({value}) <= 0 }}",
            )
            candidates.append((axes[0][0], domain.is_subset(nonnegative), domain.is_subset(nonpositive)))
        if len(candidates) == len(entries) and (all(c[1] for c in candidates) or all(c[2] for c in candidates)):
            replacements[key] = {
                entry[0][0]: candidate[0] for entry, candidate in zip(entries, candidates, strict=True)
            }

    # A caller may check, before the region runs, that the subscript tuples an
    # array is written through address a different element in every iteration,
    # and different elements from one another. When every access to that array
    # in the region goes through such a tuple at the same loop depth, two
    # accesses touch the same element exactly when they use the same tuple in
    # the same iteration, whatever the index values are. A tuple that is only
    # read is not covered: nothing was checked about the elements it names.
    certified = {}
    accesses_by_symbol = {}
    for spec in specifications:
        accesses_by_symbol.setdefault(spec[2], []).append(spec)
    for symbol, entries in accesses_by_symbol.items():
        first = entries[0]
        if (symbol.name.lower() in injective and events[first[0]].coordinates
                and all(any(value is None for value in entry[4])
                        and any(isinstance(node, ArrayAccess) for expression in entry[3]
                                for node in walk_expr(expression))
                        and events[entry[0]].coordinates == events[first[0]].coordinates for entry in entries)):
            tuples = []
            for entry in entries:
                if entry[3] not in tuples:
                    tuples.append(entry[3])
            if all(any(entry[1] == "write" and entry[3] == item for entry in entries) for item in tuples):
                certified[symbol] = tuples

    def access_maps(kind):
        nonlocal conservative
        parts = []
        for position, (index, access_kind, symbol, _, indices, _) in enumerate(specifications):
            if kind != access_kind:
                continue
            if symbol in certified:
                # The coordinates and the tuple stand for the element; they are not its physical position.
                conservative = True
                which = certified[symbol].index(specifications[position][3])
                parts.append(f"{instance(index)} -> A{symbol.id}[{','.join((*events[index].coordinates, str(which)))}]"
                             f" : {event_domain(events[index])}")
                continue
            outputs = []
            for dimension, value in enumerate(indices):
                replacement = replacements.get((symbol, dimension), {}).get(index)
                conservative |= replacement is None and value is None
                outputs.append(
                    replacement if replacement is not None else str(value) if value is not None else f"u{dimension}"
                )
            parts.append(f"{instance(index)} -> A{symbol.id}[{','.join(outputs)}] : {event_domain(events[index])}")
        return union_map(parts)

    reads = access_maps("read")
    writes = access_maps("write")
    width = max((len(event.order) for event in events), default=len(order))
    schedule = union_map(
        [
            f"{instance(i)} -> T[{','.join((*event.order, *('0' for _ in range(width - len(event.order)))))}]"
            f" : {event_domain(event)}"
            + (" and " + " and ".join(event.order_conditions) if event.order_conditions else "")
            for i, event in enumerate(events)
        ]
    )
    parallel = union_map(
        [f"{instance(i)} -> P[{','.join(coordinates)}] : {event_domain(event)}" for i, event in enumerate(events)]
    )
    conflicts = ordered_conflicts(reads, writes, schedule)
    domains = parallel.domain()
    report = RegionReport(
        str(domains),
        str(reads),
        str(writes),
        str(schedule),
        *(str(conflicts[k]) for k in ("RAW", "WAR", "WAW")),
        conservative=conservative,
    )
    read_arrays = {access.symbol for event in events for access in event.reads}
    read_arrays.update(
        node.symbol
        for item in loops
        for expr in (item.lower, item.upper, item.step_expression())
        for node in walk_expr(expr)
        if isinstance(node, ArrayAccess)
    )
    write_arrays = {access.symbol for event in events for access in event.writes}
    region = ParallelRegion(
        region_id,
        loops,
        assignments,
        tuple(sorted(private, key=lambda s: s.id)),
        tuple(sorted(captured - private - mapped, key=lambda s: s.id)),
        report,
        body,
        tuple(sorted(read_arrays, key=lambda s: s.id)),
        tuple(sorted(write_arrays, key=lambda s: s.id)),
    )
    known = (*((f"p{symbol.id}", Reference(symbol)) for symbol in sorted(start_values, key=lambda s: s.id)),
             *((name, expression) for expression, name in unknown.items()))
    return _AffineRegionModel(region, tuple(events), parallel, tuple(conflicts.items()), known)


def _region(region_id, loop, environment, defined, later, injective=frozenset()):
    """The region when its iterations are independent, if need be under a condition on its starting values."""
    model = _region_model(region_id, loop, environment, defined, later, injective=injective)
    try:
        return model.require_unconditional()
    except ParallelizationError as error:
        condition = model.independence_condition()
        if condition is None:
            raise
        text, expression = condition
        try:
            region = _region_model(region_id, loop, environment, defined, later, injective=injective,
                                   assume=(text,)).require_unconditional()
        except ParallelizationError:
            raise error from None
        return replace(region, assumptions=(expression,))


def prove_region(region_id, loop, environment, defined, later, *, injective=frozenset()):
    """Query legality after semantic validation; carry failure evidence as data.

    ``injective`` names arrays whose subscript tuple the caller guarantees to
    address a different element in every iteration of a nest that uses it.
    """
    try:
        return ParallelProof(region=_region(region_id, loop, environment, defined, later, injective))
    except ParallelizationError as error:
        return ParallelProof(failure=error.failure)
