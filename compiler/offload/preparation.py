"""Bounded, side-effect-free preparation of structured ordinary-call policies.

Only control/footprint dependencies are evaluated by the query. Numerical host
assignments remain in source order, even when device storage spans them.
"""

from dataclasses import dataclass

from compiler.ir import (
    ArrayAccess,
    Binary,
    ConditionalRegion,
    HostBlock,
    IntrinsicCall,
    Literal,
    ParallelRegion,
    Reference,
    ScalarType,
    Size,
    Unary,
    block_reads,
    block_writes,
    referenced_symbols,
)
from compiler.offload.analysis import (
    Interval,
    OffloadAnalysis,
    Unit,
    _merge_footprints,
    _protected_scalar_inputs,
    _unit_footprints,
)


@dataclass(frozen=True)
class Preparation:
    analysis: OffloadAnalysis
    assignments: frozenset = frozenset()
    query_scalars: frozenset = frozenset()
    unused_scalars: frozenset = frozenset()
    live_scalars: frozenset = frozenset()


def steps(plan):
    for step in plan.steps:
        yield step
        if isinstance(step, ConditionalRegion):
            yield from steps(step.then_plan)
            yield from steps(step.else_plan)


def scalar_reads(expression):
    return {s for s in referenced_symbols(expression) if not s.rank and s.parameter}


def prepare_offload(function, plan):
    """Prove that all host work can safely execute with an open GPU interval."""

    def unavailable(reason):
        return Preparation(OffloadAnalysis(False, reason))

    all_steps = tuple(steps(plan))
    if not plan.regions:
        return unavailable("structured offload requires device regions")
    writes = block_writes(function.body)
    parameters = frozenset(function.parameters)
    immutable = {s for s in parameters if s.rank and s not in writes}
    assignments = []
    controls = []
    for step in all_steps:
        if isinstance(step, HostBlock):
            for assignment in step.assignments:
                if not isinstance(assignment.target, Reference) or assignment.target.symbol in parameters:
                    return unavailable("host preparation must write only local scalars")
                if any(s.rank and s not in immutable for s in referenced_symbols(assignment.value)):
                    return unavailable("host preparation reads array storage written by the entry")
                assignments.append(assignment)
        elif isinstance(step, ConditionalRegion):
            controls.append(step.condition)
        elif not isinstance(step, ParallelRegion):
            return unavailable("sequential host regions are outside structured offload")

    # Host integer locals are invariant for each mapped region. Their values
    # are prepared at that region's original position, not substituted globally.
    locals_ = {a.target.symbol for a in assignments}
    invariants = parameters | frozenset(s for s in locals_ if s.dtype is ScalarType.INTEGER)
    units = []
    needed = set()
    for region in plan.regions:
        if any(s not in parameters for s in (*region.read_symbols, *region.write_symbols)):
            return unavailable("array storage must be backed by entry parameters")
        if any(not s.rank and s not in parameters and s not in locals_ for s in region.captured_symbols):
            return unavailable("scalar state produced by a mapped region is unsupported")
        protected, _ = _protected_scalar_inputs(region, parameters)
        if protected:
            return unavailable(
                "scalar inputs may be protected by conditional or empty retained-loop control: "
                + ", ".join(s.name for s in sorted(protected, key=lambda s: s.id))
            )
        unit = _unit_footprints(Unit(len(units), region, (), None), invariants)
        units.append(unit)
        for loop in region.loops:
            controls.extend((loop.lower, loop.upper))
            if not isinstance(loop.step, int):
                controls.append(loop.step)
        for footprint in unit.footprints:
            for box in (*footprint.uploads, *footprint.downloads):
                controls.extend((*box.lower, *box.upper))

    def safe(expression):
        if isinstance(expression, Literal):
            return expression.dtype in {ScalarType.INTEGER, ScalarType.LOGICAL}
        if isinstance(expression, Reference):
            return (
                not expression.symbol.rank
                and expression.symbol.dtype in {ScalarType.INTEGER, ScalarType.LOGICAL}
                and expression.symbol in parameters | locals_
            )
        if isinstance(expression, Size):
            return expression.symbol in parameters
        if isinstance(expression, ArrayAccess):
            return (
                expression.symbol in immutable
                and expression.symbol.dtype is ScalarType.INTEGER
                and all(safe(i) for i in expression.indices)
            )
        if isinstance(expression, Unary):
            return expression.operator in {"+", "-", ".not."} and safe(expression.operand)
        if isinstance(expression, Binary):
            return (
                expression.operator
                in {
                    "+",
                    "-",
                    "*",
                    "/",
                    "<",
                    ">",
                    "<=",
                    ">=",
                    "==",
                    "/=",
                    ".eq.",
                    ".ne.",
                    ".lt.",
                    ".le.",
                    ".gt.",
                    ".ge.",
                    ".and.",
                    ".or.",
                    ".eqv.",
                    ".neqv.",
                }
                and safe(expression.left)
                and safe(expression.right)
            )
        return (
            isinstance(expression, IntrinsicCall)
            and expression.name.lower() in {"min", "max"}
            and expression.dtype is ScalarType.INTEGER
            and all(safe(a) for a in expression.arguments)
        )

    if not all(safe(e) for e in controls):
        return unavailable(
            "decision preparation requires checked INTEGER/LOGICAL expressions and immutable array reads"
        )
    for expression in controls:
        needed.update(referenced_symbols(expression))
    selected = set()
    changed = True
    while changed:
        changed = False
        for assignment in assignments:
            if assignment.target.symbol in needed and assignment not in selected:
                if not safe(assignment.value):
                    return unavailable("decision preparation depends on unsupported numerical scalar setup")
                selected.add(assignment)
                needed.update(referenced_symbols(assignment.value))
                changed = True

    # Fortran LOGICAL -> C_BOOL conversion reads its operand. Only permit such
    # query inputs when native execution also reads them before its first guard.
    unconditional = set()
    for step in plan.steps:
        if isinstance(step, HostBlock):
            for assignment in step.assignments:
                unconditional.update(scalar_reads(assignment.value))
        elif isinstance(step, ConditionalRegion):
            unconditional.update(scalar_reads(step.condition))
            break
        else:
            break
    query_scalars = frozenset(s for s in needed if s in parameters and not s.rank)
    if any(s.dtype is ScalarType.LOGICAL and s not in unconditional for s in query_scalars):
        return unavailable("conditional LOGICAL query inputs require native execution")
    live = frozenset(s for s in block_reads(function.body) if s in parameters and not s.rank)
    unused = frozenset(s for s in parameters if not s.rank and s not in live)
    units = tuple(units)
    spans = {(i, j) for i in range(len(units)) for j in range(i + 1, min(len(units), i + 4) + 1)}
    spans.add((0, len(units)))
    intervals = tuple(
        Interval(i, j, _merge_footprints(f for u in units[i:j] for f in u.footprints)) for i, j in sorted(spans)
    )
    analysis = OffloadAnalysis(
        True, None, units, intervals, None, "structured preparation is supported only by sections and auto"
    )
    return Preparation(analysis, frozenset(selected), query_scalars, unused, live)
