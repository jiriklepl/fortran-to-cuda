"""Identify host storage and device captures in an execution plan."""

from __future__ import annotations

from typing import TYPE_CHECKING

from compiler.ir import (
    Assignment,
    Block,
    ConditionalRegion,
    FunctionIR,
    HostBlock,
    If,
    Loop,
    ParallelRegion,
    Symbol,
    referenced_symbols,
)

if TYPE_CHECKING:
    from compiler.ir import ExecutionPlan


def _step_symbols(loop: Loop) -> frozenset[Symbol]:
    return frozenset() if isinstance(loop.step, int) else referenced_symbols(loop.step)


def _block_symbols(block: Block) -> set[Symbol]:
    used: set[Symbol] = set()
    for statement in block.statements:
        if isinstance(statement, Assignment):
            used.update(referenced_symbols(statement.value))
            used.update(referenced_symbols(statement.target))
        elif isinstance(statement, If):
            used.update(referenced_symbols(statement.condition))
            used.update(_block_symbols(statement.then_body))
            used.update(_block_symbols(statement.else_body))
        elif isinstance(statement, Loop):
            used.update(referenced_symbols(statement.lower))
            used.update(referenced_symbols(statement.upper))
            used.update(_step_symbols(statement))
            used.add(statement.iterator)
            used.update(_block_symbols(statement.body))
        else:
            raise TypeError(f"Unknown statement: {type(statement).__name__}")
    return used


def host_symbols(function: FunctionIR, plan: ExecutionPlan) -> tuple[Symbol, ...]:
    """Only locals used by host blocks or captured by a launch need host storage."""
    used: set[Symbol] = set()
    for step in plan.steps:
        if isinstance(step, ParallelRegion):
            for loop in step.loops:
                used.update(referenced_symbols(loop.lower))
                used.update(referenced_symbols(loop.upper))
                used.update(_step_symbols(loop))
            used.update(symbol for symbol in step.captured_symbols if not symbol.rank)
        elif isinstance(step, ConditionalRegion):
            used.update(referenced_symbols(step.condition))
            used.update(host_symbols(function, step.then_plan))
            used.update(host_symbols(function, step.else_plan))
        elif isinstance(step, HostBlock):
            for assignment in step.assignments:
                used.update(referenced_symbols(assignment.value))
                used.update(referenced_symbols(assignment.target))
        else:
            raise TypeError(f"Unknown execution step: {type(step).__name__}")
    return tuple(symbol for symbol in function.symbols if symbol in used and not symbol.parameter)


def region_body(region: ParallelRegion) -> Block:
    return region.body if region.body is not None else Block(region.assignments)


def region_symbols(region: ParallelRegion) -> tuple[Symbol, ...]:
    used = set(region.captured_symbols) | _block_symbols(region_body(region))
    # Mapped bounds and strides are captured as separate scalar snapshots. Their
    # source expressions need not be evaluated again in the device kernel.
    used.difference_update(region.private_symbols)
    used.difference_update(loop.iterator for loop in region.loops)
    return tuple(sorted(used, key=lambda symbol: symbol.id))
