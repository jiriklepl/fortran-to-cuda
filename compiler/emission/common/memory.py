"""Render explicit storage operations; target callbacks render executable steps."""

from __future__ import annotations

from collections.abc import Callable

from compiler.emission.common.abi import dimension_name
from compiler.emission.common.c_family import indent, render_expression
from compiler.ir import ConditionalRegion, HostBlock, ParallelRegion, SequentialRegion, Symbol
from compiler.memory import MemoryOperation

Executable = HostBlock | ParallelRegion | SequentialRegion


def buffer_slot(symbol: Symbol) -> str:
    return f"fort_internal_state.{symbol.cpp_name}"


def buffer_name(symbol: Symbol) -> str:
    return f"{buffer_slot(symbol)}.value()"


def array_dimensions(symbol: Symbol, *, stored: bool = False) -> str:
    prefix = "fort_internal_state." if stored else ""
    return "{" + ", ".join(prefix + dimension_name(symbol, axis) for axis in range(1, symbol.rank + 1)) + "}"


def render_memory(
    operations: tuple[MemoryOperation, ...], *, device: bool, execute: Callable[[Executable], list[str]] | None = None
) -> list[str]:
    """CPU rendering collapses both abstract address spaces onto host storage."""
    lines: list[str] = []
    for operation in operations:
        kind = operation.kind
        if kind == "sync":
            lines.append("storage::synchronize();")
        elif kind == "branch":
            step = operation.step
            if not isinstance(step, ConditionalRegion):
                raise TypeError("Memory branch requires a ConditionalRegion")
            lines.append(f"if ({render_expression(step.condition)}) {{")
            lines.extend(indent(render_memory(operation.then_ops, device=device, execute=execute)))
            lines.append("} else {")
            lines.extend(indent(render_memory(operation.else_ops, device=device, execute=execute)))
            lines.append("}")
        elif kind == "execute":
            if execute is None or not isinstance(operation.step, (HostBlock, ParallelRegion, SequentialRegion)):
                raise TypeError("Memory execution requires a supported step and target renderer")
            lines.extend(execute(operation.step))
        elif kind in {"acquire", "release", "upload", "download", "host", "device", "host_write", "device_write"}:
            for symbol in operation.symbols:
                buffer = buffer_name(symbol)
                if kind == "acquire":
                    lines.append(
                        f"{buffer_slot(symbol)}.emplace(std::initializer_list<std::size_t>{array_dimensions(symbol, stored=True)});"
                    )
                elif kind == "release":
                    lines.append(f"{buffer_slot(symbol)}.reset();")
                elif kind in {"upload", "download"}:
                    direction = "device" if kind == "upload" else "host"
                    lines.append(f"{buffer}.update_{direction}({symbol.cpp_name}, {array_dimensions(symbol)});")
                elif kind in {"host", "device"}:
                    on_device = device and kind == "device"
                    pointer = symbol.cpp_name + ("_device" if on_device else "")
                    lines.append(f"{pointer} = {buffer}.{'device' if on_device else 'host'}_data();")
                else:
                    side = "device" if device and kind == "device_write" else "host"
                    lines.append(f"{buffer}.{side}_written();")
        else:
            raise ValueError(f"Unsupported memory operation: {kind}")
    return lines
