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
    Expr,
    FunctionIR,
    Literal,
    Loop,
    Reference,
    ScalarType,
    Size,
    SourceLocation,
    Symbol,
    Unary,
    referenced_symbols,
    statement_reads,
    walk_expr,
)


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


@dataclass(frozen=True)
class RegionReport:
    domain: str
    reads: str
    writes: str
    schedule: str
    raw: str
    war: str
    waw: str


@dataclass(frozen=True)
class HostBlock:
    assignments: tuple[Assignment, ...]


@dataclass(frozen=True)
class ParallelRegion:
    id: int
    loops: tuple[Loop, ...]
    assignments: tuple[Assignment, ...]
    private_symbols: tuple[Symbol, ...]
    captured_symbols: tuple[Symbol, ...]
    report: RegionReport


@dataclass(frozen=True)
class ExecutionPlan:
    steps: tuple[HostBlock | ParallelRegion, ...]

    @property
    def regions(self) -> tuple[ParallelRegion, ...]:
        return tuple(step for step in self.steps if isinstance(step, ParallelRegion))


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
    if isinstance(expression, Binary) and expression.operator in {"+", "-", "*"}:
        left = affine_expression(expression.left, environment, iterators, location)
        right = affine_expression(expression.right, environment, iterators, location)
        if expression.operator == "+":
            return left.add(right)
        if expression.operator == "-":
            return left.add(right.scale(-1))
        if not left.terms:
            return right.scale(left.constant)
        if not right.terms:
            return left.scale(right.constant)
    raise CompilationError("Unsupported non-affine array index or loop bound", location)


def _nest(loop: Loop) -> tuple[tuple[Loop, ...], tuple[Assignment, ...]]:
    loops = []
    current = loop
    while True:
        loops.append(current)
        if current.step != 1:
            raise CompilationError("Only positive unit-stride loops are supported", current.location)
        body = current.body.statements
        if len(body) == 1 and isinstance(body[0], Loop):
            current = body[0]
            continue
        if any(isinstance(stmt, Loop) for stmt in body):
            raise CompilationError("Only perfect rectangular loop nests are supported", current.location)
        break
    if len(loops) > 3:
        raise CompilationError("Loop nests deeper than three are unsupported", loop.location)
    if len({item.iterator for item in loops}) != len(loops):
        raise CompilationError("A nested loop cannot reuse an active induction variable", loop.location)
    return tuple(loops), tuple(body)


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
            incoming = (
                referenced_symbols(statement.lower)
                | referenced_symbols(statement.upper)
                | (_upward_reads(statement.body) - {statement.iterator})
            )
            reads.update(incoming - defined)
    return frozenset(reads)


def _region(
    region_id: int,
    loop: Loop,
    environment: dict[Symbol, Affine],
    defined: set[Symbol],
    later: Block,
) -> ParallelRegion:
    loops, assignments = _nest(loop)
    iterators = {item.iterator: Affine(terms=((f"q{i}", 1),)) for i, item in enumerate(loops)}
    induction = set(iterators)
    scalar_writes = {stmt.target.symbol for stmt in assignments if isinstance(stmt.target, Reference)}
    for depth, item in enumerate(loops):
        bound_symbols = referenced_symbols(item.lower) | referenced_symbols(item.upper)
        changing_bounds = bound_symbols & (scalar_writes | (induction if depth else set()))
        if changing_bounds:
            symbol = min(changing_bounds, key=lambda s: s.id)
            raise CompilationError(
                f"Loop bound '{symbol.name}' changes within the nest; only invariant rectangular bounds are supported",
                item.location,
            )
    bounds = [
        (
            affine_expression(item.lower, environment, {}, item.location),
            affine_expression(item.upper, environment, {}, item.location),
        )
        for item in loops
    ]
    if scalar_writes & induction:
        raise CompilationError("Assignments to induction variables are unsupported", loop.location)
    for symbol in scalar_writes:
        if symbol.parameter:
            raise CompilationError(f"Cannot privatize dummy argument '{symbol.name}'", loop.location)
    live_out = (scalar_writes | induction) & _upward_reads(later)
    if live_out:
        symbol = min(live_out, key=lambda s: s.id)
        raise CompilationError(f"Loop-written scalar '{symbol.name}' is live after the region", loop.location)

    available = set(induction)
    captured: set[Symbol] = set()
    for stmt in assignments:
        for symbol in sorted(statement_reads(stmt), key=lambda s: s.id):
            if symbol.rank:
                captured.add(symbol)
            elif symbol in scalar_writes:
                if symbol not in available:
                    raise CompilationError(
                        f"Scalar '{symbol.name}' is read before its per-iteration definition (reduction or carried value)",
                        stmt.location,
                    )
            elif symbol not in available:
                if symbol not in defined:
                    raise CompilationError(f"Scalar '{symbol.name}' is read before definition", stmt.location)
                captured.add(symbol)
        if isinstance(stmt.target, Reference):
            available.add(stmt.target.symbol)
        else:
            captured.add(stmt.target.symbol)
    for item in loops:
        captured.update(referenced_symbols(item.lower) | referenced_symbols(item.upper))

    coordinates = ",".join(f"q{i}" for i in range(len(loops)))
    conditions = [f"({lower}) <= q{i} <= ({upper})" for i, (lower, upper) in enumerate(bounds)]
    affine_values = [value for pair in bounds for value in pair]
    read_specs = []
    write_specs = []
    index_environment = dict(environment)
    for index, stmt in enumerate(assignments):
        accesses = [node for node in walk_expr(stmt.value) if isinstance(node, ArrayAccess)]
        if isinstance(stmt.target, ArrayAccess):
            for subscript in stmt.target.indices:
                accesses.extend(node for node in walk_expr(subscript) if isinstance(node, ArrayAccess))
        for access in accesses:
            subscripts = tuple(
                affine_expression(e, index_environment, iterators, stmt.location) for e in access.indices
            )
            affine_values.extend(subscripts)
            read_specs.append((index, access.symbol, subscripts))
        if isinstance(stmt.target, ArrayAccess):
            subscripts = tuple(
                affine_expression(e, index_environment, iterators, stmt.location) for e in stmt.target.indices
            )
            affine_values.extend(subscripts)
            write_specs.append((index, stmt.target.symbol, subscripts))
        elif stmt.target.symbol.dtype is ScalarType.INTEGER:
            try:
                index_environment[stmt.target.symbol] = affine_expression(
                    stmt.value, index_environment, iterators, stmt.location
                )
            except CompilationError:
                index_environment.pop(stmt.target.symbol, None)

    parameters = sorted({name for value in affine_values for name, _ in value.terms if not name.startswith("q")})
    parameter_text = f"[{','.join(parameters)}] -> " if parameters else ""
    domain = " and ".join(conditions) or "true"
    extents = [f"{name} >= 0" for name in parameters if name.startswith("d")]
    if extents:
        domain += " and " + " and ".join(extents)

    def statement(index: int) -> str:
        return f"S{index}[{coordinates}]"

    def union_map(parts: list[str]) -> isl.UnionMap:
        return isl.UnionMap.read_from_str(isl.DEFAULT_CONTEXT, parameter_text + "{ " + "; ".join(parts) + " }")

    def access_maps(specifications):
        return union_map(
            [
                f"{statement(index)} -> A{symbol.id}[{','.join(str(e) for e in subscripts)}] : {domain}"
                for index, symbol, subscripts in specifications
            ]
        )

    reads = access_maps(read_specs)
    writes = access_maps(write_specs)
    schedule = union_map(
        [f"{statement(i)} -> T[{region_id},{coordinates},{i}] : {domain}" for i in range(len(assignments))]
    )
    parallel = union_map([f"{statement(i)} -> P[{coordinates}] : {domain}" for i in range(len(assignments))])
    domains = isl.UnionSet.read_from_str(
        isl.DEFAULT_CONTEXT,
        parameter_text + "{ " + "; ".join(f"{statement(i)} : {domain}" for i in range(len(assignments))) + " }",
    )
    conflicts = ordered_conflicts(reads, writes, schedule)
    same_iteration = parallel.apply_range(parallel.reverse())
    for kind, relation in conflicts.items():
        bad = relation.subtract(same_iteration)
        if not bad.is_empty():
            mapping = bad.get_map_list().get_at(0)
            point = mapping.wrap().sample_point()
            source_name = mapping.get_tuple_name(isl.dim_type.in_)
            sink_name = mapping.get_tuple_name(isl.dim_type.out)
            source_stmt = assignments[int(source_name[1:])]
            sink_stmt = assignments[int(sink_name[1:])]
            names = ", ".join(
                sorted({symbol.name for symbol in referenced_symbols(source_stmt.value) | {source_stmt.target.symbol}})
            )
            raise CompilationError(
                f"Cannot parallelize: {kind} loop-carried conflict involving {names}; "
                f"accesses at {source_stmt.location} and {sink_stmt.location}; "
                f"relation {mapping}; witness {point}",
                loop.location,
            )
    report = RegionReport(
        str(domains), str(reads), str(writes), str(schedule), *(str(conflicts[k]) for k in ("RAW", "WAR", "WAW"))
    )
    return ParallelRegion(
        region_id,
        loops,
        assignments,
        tuple(sorted(scalar_writes, key=lambda s: s.id)),
        tuple(sorted(captured - induction - scalar_writes, key=lambda s: s.id)),
        report,
    )


def build_execution_plan(function: FunctionIR) -> ExecutionPlan:
    """Check supported semantics, prove each region safe, and retain source order."""
    environment = {
        symbol: Affine(terms=((f"p{symbol.id}", 1),))
        for symbol in function.parameters
        if not symbol.rank and symbol.dtype is ScalarType.INTEGER
    }
    defined = {symbol for symbol in function.parameters if not symbol.rank}
    steps: list[HostBlock | ParallelRegion] = []
    host: list[Assignment] = []
    region_id = 0
    for index, statement in enumerate(function.body.statements):
        if isinstance(statement, Assignment):
            if not isinstance(statement.target, Reference):
                raise CompilationError("Array writes outside loop regions are unsupported", statement.location)
            if any(isinstance(node, ArrayAccess) for node in walk_expr(statement.value)):
                raise CompilationError("Array reads outside loop regions are unsupported", statement.location)
            for symbol in statement_reads(statement):
                if not symbol.rank and symbol not in defined:
                    raise CompilationError(f"Scalar '{symbol.name}' is read before definition", statement.location)
            if statement.target.symbol.parameter:
                raise CompilationError("Assignments to scalar dummy arguments are unsupported", statement.location)
            if statement.target.symbol.dtype is ScalarType.INTEGER:
                try:
                    environment[statement.target.symbol] = affine_expression(
                        statement.value, environment, {}, statement.location
                    )
                except CompilationError:
                    # A host-computed integer is invariant during this region even
                    # when its computation is not affine. Give each definition a
                    # fresh parameter, without assuming a relationship to inputs.
                    environment[statement.target.symbol] = Affine(
                        terms=((f"h{index}_{statement.target.symbol.id}", 1),)
                    )
            defined.add(statement.target.symbol)
            host.append(statement)
        else:
            if host:
                steps.append(HostBlock(tuple(host)))
                host = []
            later = Block(function.body.statements[index + 1 :])
            region = _region(region_id, statement, environment, defined, later)
            steps.append(region)
            region_id += 1
    if host:
        steps.append(HostBlock(tuple(host)))
    return ExecutionPlan(tuple(steps))


def format_plan(plan: ExecutionPlan) -> str:
    lines = []
    for step in plan.steps:
        if isinstance(step, HostBlock):
            lines.append(f"host block: {len(step.assignments)} scalar assignment(s)")
        else:
            lines.append(f"region {step.id}: {len(step.loops)} dimensions, parallel legality PROVEN")
            lines.append(f"  private: {', '.join(symbol.cpp_name for symbol in step.private_symbols) or '(none)'}")
            lines.append(f"  domain: {step.report.domain}")
            lines.append(f"  schedule: {step.report.schedule}")
            lines.append(f"  RAW: {step.report.raw}")
            lines.append(f"  WAR: {step.report.war}")
            lines.append(f"  WAW: {step.report.waw}")
    return "\n".join(lines)
