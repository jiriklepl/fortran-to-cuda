"""Fuse equal-domain independent nests after checked scalar setup motion.

For individually independent regions with equal domains, concatenating their
bodies preserves the order of every same-coordinate operation. A fresh proof
that the combined body has no cross-coordinate conflicts therefore certifies
both source-pass order preservation and its proposed parallel schedule.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace

from compiler.analysis import build_execution_plan
from compiler.analysis.semantics import constant_integer
from compiler.driver.options import CompilerOptions
from compiler.ir import (
    ArrayAccess,
    Assignment,
    Block,
    ExecutionPlan,
    FunctionIR,
    If,
    Loop,
    ParallelRegion,
    Reference,
    Symbol,
    block_reads,
    block_writes,
    referenced_symbols,
    statement_reads,
    walk_expr,
)


def _substitute(value, bindings: dict[Symbol, Symbol]):
    """Rename bound identities without changing expressions or provenance."""
    if isinstance(value, Symbol):
        return bindings.get(value, value)
    if isinstance(value, tuple):
        return tuple(_substitute(item, bindings) for item in value)
    if is_dataclass(value):
        return replace(
            value, **{field.name: _substitute(getattr(value, field.name), bindings) for field in fields(value)}
        )
    return value


def _constant_step(loop: Loop) -> int | None:
    return loop.step if isinstance(loop.step, int) else constant_integer(loop.step)


def _equal_domains(left: ParallelRegion, right: ParallelRegion) -> bool:
    if len(left.loops) != len(right.loops):
        return False
    bindings = {b.iterator: a.iterator for a, b in zip(left.loops, right.loops, strict=True)}
    mapped = {loop.iterator for loop in (*left.loops, *right.loops)}
    for a, b in zip(left.loops, right.loops, strict=True):
        stride = _constant_step(a)
        if stride is None or stride == 0 or stride != _constant_step(b):
            return False
        bounds = (a.lower, a.upper, b.lower, b.upper)
        # Header references denote values before iteration starts. Renaming a
        # self-referential lower/upper bound would conflate distinct snapshots.
        if any(referenced_symbols(bound) & mapped for bound in bounds):
            return False
        if any(isinstance(node, ArrayAccess) for bound in bounds for node in walk_expr(bound)):
            return False
        if a.lower != _substitute(b.lower, bindings) or a.upper != _substitute(b.upper, bindings):
            return False
    return True


def _can_hoist(assignments: tuple, crossed: Loop) -> bool:
    reads = statement_reads(crossed)
    writes = block_writes(Block((crossed,)))
    for assignment in assignments:
        if not isinstance(assignment, Assignment) or not isinstance(assignment.target, Reference):
            return False
        if assignment.target.symbol.parameter or assignment.target.symbol in reads | writes:
            return False
        if any(isinstance(node, ArrayAccess) for node in walk_expr(assignment.value)):
            return False
        if statement_reads(assignment) & writes:
            return False
    # Assignments keep their relative order. Semantic validation of the
    # candidate checks availability at the destination as well.
    return True


def _merge(left: ParallelRegion, right: ParallelRegion) -> Loop:
    bindings = {b.iterator: a.iterator for a, b in zip(left.loops, right.loops, strict=True)}
    body = Block((*left.body.statements, *_substitute(right.body, bindings).statements))
    for loop in reversed(left.loops):
        merged = replace(loop, body=body)
        body = Block((merged,))
    return merged


def _regions(plan: ExecutionPlan) -> dict[int, ParallelRegion]:
    return {id(region.loops[0]): region for region in plan.regions}


def _host_blocks(block: Block):
    """Visit host branch bodies, never the bodies of parallel loops."""
    yield block
    for statement in block.statements:
        if isinstance(statement, If):
            yield from _host_blocks(statement.then_body)
            yield from _host_blocks(statement.else_body)


def _candidate_pairs(block: Block, regions: dict[int, ParallelRegion]):
    for current in _host_blocks(block):
        statements = current.statements
        for index, statement in enumerate(statements):
            if id(statement) not in regions:
                continue
            next_index = index + 1
            while next_index < len(statements) and isinstance(statements[next_index], Assignment):
                next_index += 1
            if next_index < len(statements) and id(statements[next_index]) in regions:
                yield current, index, next_index


def _replace_host_block(block: Block, old: Block, new: Block) -> Block:
    """Replace one branch while retaining all unaffected source-loop identities."""
    if block is old:
        return new
    statements = []
    changed = False
    for statement in block.statements:
        if isinstance(statement, If):
            then_body = _replace_host_block(statement.then_body, old, new)
            else_body = _replace_host_block(statement.else_body, old, new)
            if then_body is not statement.then_body or else_body is not statement.else_body:
                statement = replace(statement, then_body=then_body, else_body=else_body)
                changed = True
        statements.append(statement)
    return replace(block, statements=tuple(statements)) if changed else block


def optimize_function(
    function: FunctionIR, *, options: CompilerOptions | None = None
) -> tuple[FunctionIR, ExecutionPlan]:
    """Return a transformed function and freshly checked execution plan."""
    options = options or CompilerOptions()
    plan = build_execution_plan(function)
    if not options.opt_level:
        return function, replace(plan, reports=(*plan.reports, "optimization: disabled (source regions retained)"))

    # Imported here to keep this module usable while public analysis facades
    # initialize. Only parallel-proof failures are optimization skip reasons.
    from compiler.analysis import ParallelizationError

    reports: list[str] = []
    while True:
        regions = _regions(plan)
        changed = False
        for block, index, next_index in _candidate_pairs(function.body, regions):
            statements = block.statements
            statement = statements[index]
            left, right = regions[id(statement)], regions[id(statements[next_index])]
            label = f"fusion at {statement.location} and {statements[next_index].location}"
            if not _equal_domains(left, right):
                reports.append(
                    f"{label}: skipped (different domains, runtime strides, or iterator/array-valued bounds)"
                )
                continue
            # Reusing the first nest's induction identities must not capture
            # a private scalar or retained serial induction in the second body.
            if {loop.iterator for loop in left.loops} & set(right.private_symbols):
                reports.append(f"{label}: skipped (mapped iterator conflicts with second-region private scalar)")
                continue
            setup = statements[index + 1 : next_index]
            if not _can_hoist(setup, statement):
                reports.append(f"{label}: skipped (intervening effects prevent scalar motion)")
                continue
            # A scalar value escaping either original body must not become a
            # per-point value merely because two source passes are combined.
            if block_writes(left.body) & block_reads(right.body) - {s for s in function.symbols if s.rank}:
                reports.append(f"{label}: skipped (scalar lifetime crosses source passes)")
                continue
            fused = _merge(left, right)
            candidate_block = Block((*statements[:index], *setup, fused, *statements[next_index + 1 :]))
            candidate = replace(function, body=_replace_host_block(function.body, block, candidate_block))
            try:
                checked = build_execution_plan(candidate)
            except ParallelizationError as error:
                reports.append(f"{label}: skipped ({error.message})")
                continue
            candidate_regions = _regions(checked)
            preserved = set(regions) - {id(statement), id(statements[next_index])}
            if id(fused) not in candidate_regions or not preserved <= set(candidate_regions):
                reports.append(f"{label}: skipped (combined parallel independence not proved)")
                continue
            reports.append(f"{label}: applied; moved {len(setup)} scalar setup assignment(s)")
            function, plan = candidate, checked
            changed = True
            break
        if not changed:
            return function, replace(plan, reports=(*plan.reports, *dict.fromkeys(reports)))
