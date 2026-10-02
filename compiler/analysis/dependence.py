"""Scalar lifetime checks and conservative ISL parallel-legality analysis.

The original execution order is retained. Ordered memory conflicts are sufficient
for proving independence; no value-based dependence removal or scheduling is done.
"""

from __future__ import annotations

from dataclasses import dataclass

import islpy as isl

from compiler.ir import (
    ArrayAccess,
    Assignment,
    Binary,
    Block,
    CompilationError,
    ExecutionPlan,
    Expr,
    FunctionIR,
    HostBlock,
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


def _step(loop: Loop) -> Expr:
    return Literal(str(loop.step), ScalarType.INTEGER) if isinstance(loop.step, int) else loop.step


def _header_reads(loop: Loop) -> frozenset[Symbol]:
    return referenced_symbols(loop.lower) | referenced_symbols(loop.upper) | referenced_symbols(_step(loop))


def _all_loops(block: Block):
    for statement in block.statements:
        if isinstance(statement, Loop):
            yield statement
            yield from _all_loops(statement.body)


def _assignments(block: Block):
    for statement in block.statements:
        if isinstance(statement, Assignment):
            yield statement
        else:
            yield from _assignments(statement.body)


def _mapped_prefix(loop: Loop) -> tuple[tuple[Loop, ...], Block]:
    """Map a rectangular perfect prefix; retain the remaining ordered body."""
    loops = [loop]
    body = loop.body
    written = block_writes(body)
    while len(body.statements) == 1 and isinstance(body.statements[0], Loop):
        inner = body.statements[0]
        # SIZE reads immutable extents, not the array cells written by the body.
        header_values = frozenset(
            node.symbol
            for expression in (inner.lower, inner.upper, _step(inner))
            for node in walk_expr(expression)
            if isinstance(node, (Reference, ArrayAccess))
        )
        changing = header_values & (written | {item.iterator for item in loops})
        if changing:
            break
        loops.append(inner)
        body = inner.body
    return tuple(loops), body


def ordered_conflicts(reads: isl.UnionMap, writes: isl.UnionMap, schedule: isl.UnionMap):
    """Relations map earlier statement instances to later conflicting instances."""
    before = schedule.lex_lt_union_map(schedule)
    return {
        "RAW": writes.apply_range(reads.reverse()).intersect(before),
        "WAR": reads.apply_range(writes.reverse()).intersect(before),
        "WAW": writes.apply_range(writes.reverse()).intersect(before),
    }


def _upward_reads(block: Block) -> frozenset[Symbol]:
    """Reads needing an incoming value; loop writes do not define values on zero trips."""
    reads: set[Symbol] = set()
    defined: set[Symbol] = set()
    for statement in block.statements:
        if isinstance(statement, Assignment):
            reads.update(statement_reads(statement) - defined)
            if isinstance(statement.target, Reference):
                defined.add(statement.target.symbol)
        else:
            incoming = _header_reads(statement) | (_upward_reads(statement.body) - {statement.iterator})
            reads.update(incoming - defined)
    return frozenset(reads)


@dataclass(frozen=True)
class _Event:
    origin: Assignment | Loop
    coordinates: tuple[str, ...]
    conditions: tuple[str, ...]
    order: tuple[str, ...]
    order_conditions: tuple[str, ...]
    reads: tuple[ArrayAccess, ...]
    writes: tuple[ArrayAccess, ...]
    environment: dict[Symbol, Affine]
    expressions: dict[Symbol, Expr]
    active: dict[Symbol, Affine]


def _resolve(expression: Expr, definitions: dict[Symbol, Expr]) -> Expr:
    if isinstance(expression, Reference):
        return definitions.get(expression.symbol, expression)
    if isinstance(expression, Binary):
        return Binary(
            expression.operator, _resolve(expression.left, definitions), _resolve(expression.right, definitions)
        )
    if isinstance(expression, Unary):
        return Unary(expression.operator, _resolve(expression.operand, definitions))
    if isinstance(expression, ArrayAccess):
        return ArrayAccess(expression.symbol, tuple(_resolve(index, definitions) for index in expression.indices))
    return expression


def _region(
    region_id: int,
    loop: Loop,
    environment: dict[Symbol, Affine],
    defined: set[Symbol],
    later: Block,
) -> ParallelRegion:
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
        raise CompilationError(f"Loop-written scalar '{symbol.name}' is live after the region", loop.location)

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
                    raise CompilationError(
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

    def constraints(item, values, active, is_mapped):
        nonlocal axis_number, conservative
        axis = f"q{axis_number}"
        axis_number += 1
        axes = []
        bounds = []
        for suffix, expression in (("lower", item.lower), ("upper", item.upper), ("step", _step(item))):
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
            raise CompilationError("Nonrectangular mapped loop prefix", item.location)
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
                    definitions[symbol] = resolved
                    if symbol.dtype is ScalarType.INTEGER:
                        value = affine_or_unknown(statement.value, values, active, statement.location)
                        if value is None:
                            values.pop(symbol, None)
                        else:
                            values[symbol] = value
            else:
                check_reads(_header_reads(statement), available, statement.location)
                expressions = (statement.lower, statement.upper, _step(statement))
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
                    if symbol not in carried and symbol != statement.iterator
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
                for symbol in block_writes(statement.body) | {statement.iterator}:
                    values.pop(symbol, None)
                    definitions.pop(symbol, None)
        return available

    visit(body, environment, {}, available, iterators, coordinates, conditions, order, order_conditions)
    specifications = []
    for index, event in enumerate(events):
        active = event.active
        for access_kind, accesses in (("read", event.reads), ("write", event.writes)):
            for access in accesses:
                resolved = tuple(_resolve(expr, event.expressions) for expr in access.indices)
                indices = tuple(
                    affine_or_unknown(expr, event.environment, active, event.origin.location) for expr in access.indices
                )
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
        return " and ".join((*event.conditions, *extent_constraints)) or "true"

    def instance(index):
        return f"S{index}[{','.join(events[index].coordinates)}]"

    def union_map(parts):
        return isl.UnionMap.read_from_str(isl.DEFAULT_CONTEXT, parameter_text + "{ " + "; ".join(parts) + " }")

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
                isl.DEFAULT_CONTEXT,
                parameter_text
                + "{ ["
                + ",".join(events[index].coordinates)
                + "] : "
                + event_domain(events[index])
                + " }",
            )
            nonnegative = isl.Set.read_from_str(
                isl.DEFAULT_CONTEXT,
                parameter_text + "{ [" + ",".join(events[index].coordinates) + f"] : ({value}) >= 0 }}",
            )
            nonpositive = isl.Set.read_from_str(
                isl.DEFAULT_CONTEXT,
                parameter_text + "{ [" + ",".join(events[index].coordinates) + f"] : ({value}) <= 0 }}",
            )
            candidates.append((axes[0][0], domain.is_subset(nonnegative), domain.is_subset(nonpositive)))
        if len(candidates) == len(entries) and (all(c[1] for c in candidates) or all(c[2] for c in candidates)):
            replacements[key] = {
                entry[0][0]: candidate[0] for entry, candidate in zip(entries, candidates, strict=True)
            }

    def access_maps(kind):
        nonlocal conservative
        parts = []
        for index, access_kind, symbol, _, indices, _ in specifications:
            if kind != access_kind:
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
    same_iteration = parallel.apply_range(parallel.reverse())
    for kind, relation in conflicts.items():
        bad = relation.subtract(same_iteration)
        if not bad.is_empty():
            mapping = bad.get_map_list().get_at(0)
            source = events[int(mapping.get_tuple_name(isl.dim_type.in_)[1:])]
            sink = events[int(mapping.get_tuple_name(isl.dim_type.out)[1:])]
            names = ", ".join(
                sorted({access.symbol.name for access in (*source.reads, *source.writes, *sink.reads, *sink.writes)})
            )
            raise ParallelizationError(
                f"Cannot parallelize: {kind} {'possible ' if conservative else ''}loop-carried conflict involving {names}; "
                + ("conservative access/stride model; " if conservative else "")
                + f"accesses at {source.origin.location} and {sink.origin.location}; "
                f"relation {mapping}; witness {mapping.wrap().sample_point()}",
                loop.location,
                witness=str(mapping.wrap().sample_point()),
                conservative=conservative,
            )
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
        for expr in (item.lower, item.upper, _step(item))
        for node in walk_expr(expr)
        if isinstance(node, ArrayAccess)
    )
    write_arrays = {access.symbol for event in events for access in event.writes}
    return ParallelRegion(
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


def prove_region(region_id, loop, environment, defined, later):
    """Return modeled dependence failures; other compilation errors propagate."""
    try:
        return ParallelProof(region=_region(region_id, loop, environment, defined, later))
    except ParallelizationError as error:
        return ParallelProof(failure=error.failure)


def build_execution_plan(function: FunctionIR) -> ExecutionPlan:
    """Prove mapped iterations independent and retain all host/body source order."""
    environment = {
        symbol: Affine(terms=((f"p{symbol.id}", 1),))
        for symbol in function.parameters
        if not symbol.rank and symbol.dtype is ScalarType.INTEGER
    }
    defined = {symbol for symbol in function.parameters if not symbol.rank}
    steps: list[HostBlock | ParallelRegion] = []
    host: list[Assignment] = []
    region_id = 0

    def flush_host():
        if not host:
            return
        reads = {
            node.symbol
            for stmt in host
            for expr in (stmt.value, stmt.target)
            for node in walk_expr(expr)
            if isinstance(node, ArrayAccess)
        }
        writes = {stmt.target.symbol for stmt in host if isinstance(stmt.target, ArrayAccess)}
        steps.append(
            HostBlock(tuple(host), tuple(sorted(reads, key=lambda s: s.id)), tuple(sorted(writes, key=lambda s: s.id)))
        )
        host.clear()

    for index, statement in enumerate(function.body.statements):
        if isinstance(statement, Assignment):
            for symbol in statement_reads(statement):
                if not symbol.rank and symbol not in defined:
                    raise CompilationError(f"Scalar '{symbol.name}' is read before definition", statement.location)
            if isinstance(statement.target, Reference):
                symbol = statement.target.symbol
                if symbol.parameter:
                    raise CompilationError("Assignments to scalar dummy arguments are unsupported", statement.location)
                if symbol.dtype is ScalarType.INTEGER:
                    try:
                        environment[symbol] = affine_expression(statement.value, environment, {}, statement.location)
                    except CompilationError:
                        environment[symbol] = Affine(terms=((f"h{index}_{symbol.id}", 1),))
                defined.add(symbol)
            host.append(statement)
        else:
            flush_host()
            steps.append(
                _region(region_id, statement, environment, defined, Block(function.body.statements[index + 1 :]))
            )
            region_id += 1
    flush_host()
    return ExecutionPlan(tuple(steps))


def format_plan(plan: ExecutionPlan) -> str:
    lines = []
    for step in plan.steps:
        if isinstance(step, HostBlock):
            lines.append(f"host block: {len(step.assignments)} ordered assignment(s)")
            lines.append(f"  array reads: {', '.join(s.name for s in step.read_symbols) or '(none)'}")
            lines.append(f"  array writes: {', '.join(s.name for s in step.write_symbols) or '(none)'}")
        else:
            lines.append(f"region {step.id}: {len(step.loops)} mapped dimensions, parallel legality PROVEN")
            lines.append(f"  retained sequential loops: {len(tuple(_all_loops(step.body)))}")
            lines.append(f"  access model: {'conservative' if step.report.conservative else 'exact'}")
            lines.append(f"  private: {', '.join(s.cpp_name for s in step.private_symbols) or '(none)'}")
            lines.append(f"  domain: {step.report.domain}")
            lines.append(f"  schedule: {step.report.schedule}")
            lines.append(f"  RAW: {step.report.raw}")
            lines.append(f"  WAR: {step.report.war}")
            lines.append(f"  WAW: {step.report.waw}")
    return "\n".join(lines)
