"""Render Fortran loop bounds and iteration semantics for C++ and CUDA."""

from __future__ import annotations

from typing import TYPE_CHECKING

from compiler.emission.common.c_family import indent, render_assignment, render_expression
from compiler.ir import Assignment, Block, Loop

if TYPE_CHECKING:
    from compiler.ir import ParallelRegion


def _loop_step(loop: Loop) -> str:
    return str(loop.step) if isinstance(loop.step, int) else render_expression(loop.step)


def _loop_snapshot(loop: Loop, suffix: str, active: str | None = None, *, device: bool = False) -> list[str]:
    """Evaluate bounds once and count positive or negative Fortran DO trips."""
    lower = f"fort_internal_lower{suffix}"
    upper = f"fort_internal_upper{suffix}"
    stride = f"fort_internal_stride{suffix}"
    extent = f"fort_internal_extent{suffix}"

    def evaluate(expression: str) -> str:
        return f"({active}) ? ({expression}) : 0" if active else expression

    location = str(loop.location).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")
    lines = [
        f"const int {lower} = {evaluate(render_expression(loop.lower))};",
        f"const int {upper} = {evaluate(render_expression(loop.upper))};",
        f"const int {stride} = {evaluate(_loop_step(loop))};",
    ]
    condition = f"({active}) && {stride} == 0" if active else f"{stride} == 0"
    failure = (
        [f'    printf("%s\\n", "{location}: DO stride must be nonzero");', '    asm("trap;");']
        if device
        else [f'    std::cerr << "{location}: DO stride must be nonzero" << std::endl;', "    std::abort();"]
    )
    lines.extend(
        [
            f"if ({condition}) {{",
            *failure,
            "}",
            f"const std::size_t {extent} = {stride} > 0 && {upper} >= {lower}",
            f"    ? static_cast<std::size_t>((static_cast<long long>({upper}) - {lower}) / {stride} + 1)",
            f"    : {stride} < 0 && {lower} >= {upper}",
            f"        ? static_cast<std::size_t>((static_cast<long long>({lower}) - {upper})",
            f"            / -static_cast<long long>({stride}) + 1) : 0;",
        ]
    )
    return lines


def iteration_value(suffix: str, ordinal: str) -> str:
    return (
        f"static_cast<int>(static_cast<long long>(fort_internal_lower{suffix})"
        f" + static_cast<long long>({ordinal}) * fort_internal_stride{suffix})"
    )


def sequential_block(block: Block, depth: int, counter: list[int], *, device: bool = False) -> list[str]:
    lines: list[str] = []
    for statement in block.statements:
        if isinstance(statement, Assignment):
            lines.extend(indent([render_assignment(statement)], depth))
            continue
        suffix = f"_serial{counter[0]}"
        counter[0] += 1
        ordinal = f"fort_internal_ordinal{suffix}"
        lines.extend(indent(["{"], depth))
        lines.extend(indent(_loop_snapshot(statement, suffix, device=device), depth + 1))
        lines.extend(
            indent(
                [
                    f"for (std::size_t {ordinal} = 0; {ordinal} < fort_internal_extent{suffix}; ++{ordinal}) {{",
                    f"    {statement.iterator.cpp_name} = {iteration_value(suffix, ordinal)};",
                ],
                depth + 1,
            )
        )
        lines.extend(sequential_block(statement.body, depth + 2, counter, device=device))
        lines.extend(
            indent(
                [
                    "}",
                    f"{statement.iterator.cpp_name} = {iteration_value(suffix, f'fort_internal_extent{suffix}')};",
                ],
                depth + 1,
            )
        )
        lines.extend(indent(["}"], depth))
    return lines


def mapped_snapshots(region: ParallelRegion) -> list[str]:
    lines: list[str] = []
    for dimension, loop in enumerate(region.loops):
        active = " && ".join(f"fort_internal_extent{previous} > 0" for previous in range(dimension))
        lines.extend(_loop_snapshot(loop, str(dimension), active or None))
    return lines
