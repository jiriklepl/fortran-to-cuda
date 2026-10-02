"""Derive memory operations from effects without target-specific code generation.

Ensures are conditional transfers: runtime ownership state decides whether a copy
is needed. This keeps branch joins and repeated resident runs coherent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from compiler.ir import (
    ArrayAccess,
    ConditionalRegion,
    ExecutionPlan,
    HostBlock,
    ParallelRegion,
    SequentialRegion,
    Symbol,
    walk_expr,
)


@dataclass(frozen=True)
class MemoryOperation:
    kind: Literal["acquire", "host", "device", "host_write", "device_write", "execute", "branch", "sync", "release"]
    symbols: tuple[Symbol, ...] = ()
    step: HostBlock | ParallelRegion | SequentialRegion | ConditionalRegion | None = None
    then_ops: tuple[MemoryOperation, ...] = ()
    else_ops: tuple[MemoryOperation, ...] = ()


@dataclass(frozen=True)
class MemoryPlan:
    create: tuple[MemoryOperation, ...]
    run: tuple[MemoryOperation, ...]
    retrieve: tuple[MemoryOperation, ...]
    destroy: tuple[MemoryOperation, ...]


def plan_memory(plan: ExecutionPlan, parameters: tuple[Symbol, ...]) -> MemoryPlan:
    arrays = tuple(symbol for symbol in parameters if symbol.rank)

    def ordered(symbols):
        symbols = frozenset(symbols)
        return tuple(symbol for symbol in arrays if symbol in symbols)

    def operations(current):
        result = []
        for step in current.steps:
            if isinstance(step, ParallelRegion):
                bounds = [expr for loop in step.loops for expr in (loop.lower, loop.upper)]
                bounds.extend(loop.step for loop in step.loops if not isinstance(loop.step, int))
                bound_reads = {
                    node.symbol for expr in bounds for node in walk_expr(expr) if isinstance(node, ArrayAccess)
                }
                result.append(MemoryOperation("host", ordered(bound_reads)))
                # Partial writes preserve values outside the written domain.
                result.append(MemoryOperation("device", ordered(set(step.read_symbols) | set(step.write_symbols))))
                result.append(MemoryOperation("execute", step=step))
                result.append(MemoryOperation("device_write", ordered(step.write_symbols)))
            elif isinstance(step, ConditionalRegion):
                predicate_reads = {node.symbol for node in walk_expr(step.condition) if isinstance(node, ArrayAccess)}
                result.append(MemoryOperation("host", ordered(predicate_reads)))
                result.append(
                    MemoryOperation(
                        "branch", step=step, then_ops=operations(step.then_plan), else_ops=operations(step.else_plan)
                    )
                )
            elif isinstance(step, (HostBlock, SequentialRegion)):
                result.append(MemoryOperation("host", ordered(set(step.read_symbols) | set(step.write_symbols))))
                result.append(MemoryOperation("execute", step=step))
                result.append(MemoryOperation("host_write", ordered(step.write_symbols)))
            else:
                raise TypeError(f"Unknown execution step: {type(step).__name__}")
        return tuple(operation for operation in result if operation.symbols or operation.step is not None)

    return MemoryPlan(
        create=(
            MemoryOperation("acquire", arrays),
            MemoryOperation("device", ordered(s for s in arrays if s.intent != "out")),
        ),
        run=operations(plan),
        retrieve=(MemoryOperation("sync"), MemoryOperation("host", ordered(s for s in arrays if s.intent != "in"))),
        destroy=(MemoryOperation("sync"), MemoryOperation("release", arrays)),
    )


def format_memory(memory: MemoryPlan) -> str:
    """Explain ownership requirements, including transfers decided at runtime."""
    lines = []

    def describe(operations, depth):
        for operation in operations:
            detail = ", ".join(symbol.name for symbol in operation.symbols)
            if isinstance(operation.step, (ParallelRegion, SequentialRegion)):
                detail = f"region {operation.step.id}"
            elif isinstance(operation.step, HostBlock):
                detail = "host statements"
            elif isinstance(operation.step, ConditionalRegion):
                detail = str(operation.step.location)
            lines.append("  " * depth + operation.kind + (f": {detail}" if detail else ""))
            if operation.kind == "branch":
                lines.append("  " * (depth + 1) + "then:")
                describe(operation.then_ops, depth + 2)
                lines.append("  " * (depth + 1) + "else:")
                describe(operation.else_ops, depth + 2)

    for stage in ("create", "run", "retrieve", "destroy"):
        lines.append(stage + ":")
        describe(getattr(memory, stage), 1)
    return "\n".join(lines)
