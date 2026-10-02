"""Emit C++ execution plans using schedules selected before emission."""

from __future__ import annotations

from typing import TYPE_CHECKING

from compiler.emission.c.declarations import cpp_declaration
from compiler.emission.common.abi import AbiArgument
from compiler.emission.common.c_family import cpp_type, indent, render_assignment, render_expression
from compiler.emission.common.loops import iteration_value, mapped_snapshots, sequential_block
from compiler.emission.common.schedules import checked_product, region_schedule, tile_counts
from compiler.emission.common.symbols import host_symbols, region_body
from compiler.ir import ConditionalRegion, FunctionIR, HostBlock, ParallelRegion

if TYPE_CHECKING:
    from compiler.ir import ExecutionPlan


def _point_body(region: ParallelRegion, depth: int) -> list[str]:
    lines = []
    for axis, loop in enumerate(region.loops):
        lines.extend(
            indent(
                [f"const int {loop.iterator.cpp_name} = {iteration_value(str(axis), f'fort_internal_ordinal{axis}')};"],
                depth,
            )
        )
    lines.extend(indent([f"{cpp_type(symbol)} {symbol.cpp_name};" for symbol in region.private_symbols], depth))
    lines.extend(sequential_block(region_body(region), depth, [0]))
    return lines


def _cpp_region(region: ParallelRegion) -> list[str]:
    schedule = region_schedule(region)
    lines = ["{", f"    // Verified parallel region {region.id}."]
    lines.extend(indent(mapped_snapshots(region)))
    depth = 1
    if schedule.tile_sizes:
        lines.extend(indent(tile_counts(region)))
        lines.extend(
            indent(
                checked_product(
                    "fort_internal_total_tiles", tuple(f"fort_internal_tiles{axis}" for axis in schedule.axis_order)
                )
            )
        )
        lines.extend(
            indent(
                [
                    "#pragma omp parallel for",
                    "for (std::size_t fort_internal_tile = 0;",
                    "     fort_internal_tile < fort_internal_total_tiles; ++fort_internal_tile) {",
                    "    std::size_t fort_internal_tile_index = fort_internal_tile;",
                ]
            )
        )
        depth += 1
        for axis in schedule.axis_order:
            size = schedule.tile_sizes[axis]
            lines.extend(
                indent(
                    [
                        f"const std::size_t fort_internal_begin{axis} = "
                        f"(fort_internal_tile_index % fort_internal_tiles{axis}) * {size}ULL;",
                        f"fort_internal_tile_index /= fort_internal_tiles{axis};",
                        f"const std::size_t fort_internal_remaining{axis} = "
                        f"fort_internal_extent{axis} - fort_internal_begin{axis};",
                        f"const std::size_t fort_internal_end{axis} = fort_internal_begin{axis} + "
                        f"(fort_internal_remaining{axis} < {size}ULL ? fort_internal_remaining{axis} : {size}ULL);",
                    ],
                    depth,
                )
            )
    else:
        lines.extend(
            indent(
                checked_product(
                    "fort_internal_total", tuple(f"fort_internal_extent{axis}" for axis in schedule.axis_order)
                )
            )
        )
        pragma = "#pragma omp parallel for"
        if len(region.loops) > 1:
            pragma += f" collapse({len(region.loops)})"
        lines.extend(indent([pragma]))
    for axis in reversed(schedule.axis_order):
        ordinal = f"fort_internal_ordinal{axis}"
        begin = f"fort_internal_begin{axis}" if schedule.tile_sizes else "0"
        end = f"fort_internal_end{axis}" if schedule.tile_sizes else f"fort_internal_extent{axis}"
        lines.extend(
            indent(
                [f"for (std::size_t {ordinal} = {begin};", f"     {ordinal} < {end}; ++{ordinal}) {{"],
                depth,
            )
        )
        depth += 1
    lines.extend(_point_body(region, depth))
    for _ in region.loops:
        depth -= 1
        lines.extend(indent(["}"], depth))
    if schedule.tile_sizes:
        lines.extend(indent(["}"]))
    lines.append("}")
    return lines


def cpp_plan_lines(plan: ExecutionPlan, depth: int = 0) -> list[str]:
    """Render nested host control flow and proved parallel regions."""
    lines: list[str] = []
    for step in plan.steps:
        if isinstance(step, ParallelRegion):
            lines.extend(indent(_cpp_region(step), depth))
        elif isinstance(step, HostBlock):
            lines.extend(indent([render_assignment(assignment) for assignment in step.assignments], depth))
        elif isinstance(step, ConditionalRegion):
            lines.extend(indent([f"if ({render_expression(step.condition)}) {{"], depth))
            lines.extend(cpp_plan_lines(step.then_plan, depth + 1))
            lines.extend(indent(["} else {"], depth))
            lines.extend(cpp_plan_lines(step.else_plan, depth + 1))
            lines.extend(indent(["}"], depth))
        else:
            raise TypeError(f"Unsupported execution-plan step: {type(step).__name__}")
    return lines


def generate_cpp(function: FunctionIR, plan: ExecutionPlan, abi: tuple[AbiArgument, ...], common_header: str) -> str:
    lines = [
        "#include <cstdlib>",
        f'#include "{common_header}"',
        "",
        "namespace generated_kernels {",
        "using namespace generated_kernels::indexing;",
        "",
        'extern "C" void cpp_start_hot() {}',
        'extern "C" void cpp_finish_hot() {}',
        "",
        f'extern "C" void cpp_{function.name}(',
    ]
    lines.extend(
        indent(
            [cpp_declaration(argument) + ("," if index < len(abi) - 1 else "") for index, argument in enumerate(abi)]
        )
    )
    lines.append(") {")
    lines.extend(indent([f"{cpp_type(symbol)} {symbol.cpp_name};" for symbol in host_symbols(function, plan)]))
    lines.extend(cpp_plan_lines(plan, 1))
    lines.extend(["}", "}", ""])
    return "\n".join(lines)
