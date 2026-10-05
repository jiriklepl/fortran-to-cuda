"""Derive memory operations from effects without target-specific code generation.

Upload/download operations copy caller arrays explicitly. Host/device ensures
are conditional transfers: runtime ownership state decides whether a copy is needed. This keeps branch joins and repeated resident runs coherent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from compiler.ir import (
    ArrayAccess,
    Block,
    ConditionalRegion,
    ExecutionPlan,
    HostBlock,
    ParallelRegion,
    SequentialRegion,
    Symbol,
    block_reads,
    block_writes,
    referenced_symbols,
    walk_expr,
)


@dataclass(frozen=True)
class MemoryOperation:
    kind: Literal[
        "acquire",
        "upload",
        "download",
        "host",
        "device",
        "host_write",
        "device_write",
        "execute",
        "branch",
        "sync",
        "release",
    ]
    symbols: tuple[Symbol, ...] = ()
    step: HostBlock | ParallelRegion | SequentialRegion | ConditionalRegion | None = None
    then_ops: tuple[MemoryOperation, ...] = ()
    else_ops: tuple[MemoryOperation, ...] = ()
    acquisition_policy: Literal["dedicated", "pooled"] | None = field(default=None, kw_only=True)


@dataclass(frozen=True)
class MemoryPlan:
    create: tuple[MemoryOperation, ...]
    run: tuple[MemoryOperation, ...]
    retrieve: tuple[MemoryOperation, ...]
    destroy: tuple[MemoryOperation, ...]


def plan_memory(
    plan: ExecutionPlan,
    parameters: tuple[Symbol, ...],
    *,
    acquisition_policy: Literal["dedicated", "pooled"] | None = None,
) -> MemoryPlan:
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
        return tuple(
            operation
            for operation in result
            if operation.kind not in {"host", "device", "host_write", "device_write"} or operation.symbols
        )

    memory = MemoryPlan(
        create=(
            MemoryOperation("acquire", arrays, acquisition_policy=acquisition_policy),
            MemoryOperation("upload", ordered(s for s in arrays if s.intent != "out")),
        ),
        run=operations(plan),
        retrieve=(MemoryOperation("sync"), MemoryOperation("download", ordered(s for s in arrays if s.intent != "in"))),
        destroy=(MemoryOperation("sync"), MemoryOperation("release", arrays)),
    )
    validate_memory(memory, parameters)
    return memory


def _execution_arrays(step) -> set[Symbol]:
    """Check storage identities in executable IR as well as declared effects."""
    if isinstance(step, HostBlock):
        body = Block(step.assignments)
    elif isinstance(step, ParallelRegion):
        body = Block((*step.loops, *step.body.statements, *step.assignments))
    elif isinstance(step, SequentialRegion):
        body = step.body
    elif isinstance(step, ConditionalRegion):
        used = set(referenced_symbols(step.condition))
        for plan in (step.then_plan, step.else_plan):
            for child in plan.steps:
                used.update(_execution_arrays(child))
        return {symbol for symbol in used | set(step.read_symbols) | set(step.write_symbols) if symbol.rank}
    else:
        raise TypeError(f"Unknown executable step: {type(step).__name__}")
    used = block_reads(body) | block_writes(body) | set(step.read_symbols) | set(step.write_symbols)
    if isinstance(step, ParallelRegion):
        used |= set(step.captured_symbols)
    return {symbol for symbol in used if symbol.rank}


def validate_memory(memory: MemoryPlan, parameters: tuple[Symbol, ...], *, allow_pooled: bool = True) -> None:
    """Reject malformed internal plans before emission or partial publication.

    This checks ownership and lifecycle structure; runtime coherence still decides
    whether a host/device ensure needs a transfer on the executed control path.
    """
    if not isinstance(memory, MemoryPlan):
        raise ValueError("Expected a MemoryPlan")
    arrays = {symbol for symbol in parameters if symbol.rank}
    if any(not symbol.parameter for symbol in arrays):
        raise ValueError("Memory parameters must be function array parameters")
    inputs = {symbol for symbol in arrays if symbol.intent != "out"}
    outputs = {symbol for symbol in arrays if symbol.intent != "in"}
    phases = {
        "acquire": "create",
        "upload": "create",
        "download": "retrieve",
        "host": "run",
        "device": "run",
        "host_write": "run",
        "device_write": "run",
        "execute": "run",
        "branch": "run",
        "release": "destroy",
    }

    def validate(phase, operations, acquired):
        if not isinstance(operations, tuple):
            raise ValueError(f"Memory {phase} operations must be a tuple")
        transferred = set()
        synchronized = False
        for operation in operations:
            if not isinstance(operation, MemoryOperation):
                raise ValueError("Expected a MemoryOperation")
            kind = operation.kind
            if not isinstance(kind, str) or kind not in {*phases, "sync"}:
                raise ValueError(f"Unknown memory operation: {kind!r}")
            if operation.acquisition_policy not in (None, "dedicated", "pooled"):
                raise ValueError(f"Unknown memory acquisition policy: {operation.acquisition_policy!r}")
            if kind != "acquire" and operation.acquisition_policy is not None:
                raise ValueError(f"Memory operation {kind} cannot contain an acquisition policy")
            if not allow_pooled and operation.acquisition_policy == "pooled":
                raise ValueError("Explicit sessions require dedicated allocations")
            if kind != "sync" and phases[kind] != phase:
                raise ValueError(f"Memory operation {kind} is invalid in {phase}")
            if not isinstance(operation.symbols, tuple) or any(
                not isinstance(symbol, Symbol) for symbol in operation.symbols
            ):
                raise ValueError(f"Memory operation {kind} requires a tuple of symbols")
            symbols = set(operation.symbols)
            if len(symbols) != len(operation.symbols):
                raise ValueError(f"Duplicate symbols in memory operation {kind}")
            if symbols - arrays or any(not symbol.parameter or not symbol.rank for symbol in symbols):
                raise ValueError(f"Memory operation {kind} refers to an unknown or nonarray parameter")
            if not isinstance(operation.then_ops, tuple) or not isinstance(operation.else_ops, tuple):
                raise ValueError("Memory branch operations must be tuples")
            if kind != "branch" and (operation.then_ops or operation.else_ops):
                raise ValueError(f"Memory operation {kind} cannot contain branch operations")
            if kind not in {"execute", "branch"} and operation.step is not None:
                raise ValueError(f"Memory operation {kind} cannot contain an execution step")
            if kind in {"execute", "branch", "sync"} and symbols:
                raise ValueError(f"Memory operation {kind} cannot contain symbols")
            if kind in {"host", "device", "host_write", "device_write"} and not symbols:
                raise ValueError(f"Memory operation {kind} requires array symbols")
            if kind == "acquire":
                if symbols & acquired:
                    raise ValueError("Duplicate array acquisition")
                acquired.update(symbols)
            elif symbols - acquired:
                raise ValueError(f"Memory operation {kind} uses an array before acquisition or after release")
            if kind == "sync":
                synchronized = True
            elif kind in {"upload", "download"}:
                expected = inputs if kind == "upload" else outputs
                if symbols - expected:
                    raise ValueError(f"Memory operation {kind} conflicts with array intent")
                if kind == "download" and not synchronized:
                    raise ValueError("Memory retrieval requires synchronization before download")
                transferred.update(symbols)
            elif kind == "release":
                if not synchronized:
                    raise ValueError("Memory destruction requires synchronization before release")
                acquired.difference_update(symbols)
            elif kind in {"execute", "branch"}:
                step = operation.step
                expected = (ConditionalRegion,) if kind == "branch" else (HostBlock, ParallelRegion, SequentialRegion)
                if not isinstance(step, expected):
                    raise TypeError(f"Invalid execution step for memory operation {kind}: {type(step).__name__}")
                used = _execution_arrays(step)
                if used - arrays or any(not symbol.parameter for symbol in used):
                    raise ValueError("Memory execution refers to an unknown array parameter")
                if used - acquired:
                    raise ValueError("Memory execution uses an array before acquisition")
                if kind == "branch":
                    validate("run", operation.then_ops, acquired.copy())
                    validate("run", operation.else_ops, acquired.copy())
        return acquired, transferred, synchronized

    acquired, uploaded, _ = validate("create", memory.create, set())
    if acquired != arrays:
        raise ValueError("Memory creation must acquire every array parameter")
    if uploaded != inputs:
        raise ValueError("Memory creation must upload every input array")
    validate("run", memory.run, acquired.copy())
    _, downloaded, synchronized = validate("retrieve", memory.retrieve, acquired.copy())
    if not synchronized or downloaded != outputs:
        raise ValueError("Memory retrieval must synchronize and download every output array")
    remaining, _, synchronized = validate("destroy", memory.destroy, acquired.copy())
    if not synchronized or remaining:
        raise ValueError("Memory destruction must synchronize and release every array")


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
            policy = f" [{operation.acquisition_policy or 'dedicated'}]" if operation.kind == "acquire" else ""
            lines.append("  " * depth + operation.kind + policy + (f": {detail}" if detail else ""))
            if operation.kind == "branch":
                lines.append("  " * (depth + 1) + "then:")
                describe(operation.then_ops, depth + 2)
                lines.append("  " * (depth + 1) + "else:")
                describe(operation.else_ops, depth + 2)

    for stage in ("create", "run", "retrieve", "destroy"):
        lines.append(stage + ":")
        describe(getattr(memory, stage), 1)
    return "\n".join(lines)
