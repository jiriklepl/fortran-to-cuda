"""Emit the C++ entry point and OpenMP parallel regions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from compiler.emission.c.declarations import cpp_declaration
from compiler.emission.common.abi import AbiArgument
from compiler.emission.common.c_family import cpp_type, indent, render_assignment
from compiler.emission.common.loops import iteration_value, mapped_snapshots, sequential_block
from compiler.emission.common.symbols import host_symbols, region_body
from compiler.ir import FunctionIR

if TYPE_CHECKING:
    from compiler.ir import ExecutionPlan, ParallelRegion


def _cpp_region(region: ParallelRegion) -> list[str]:
    lines = ["{", f"    // Verified parallel region {region.id}."]
    lines.extend(indent(mapped_snapshots(region)))
    pragma = "    #pragma omp parallel for"
    if len(region.loops) > 1:
        pragma += f" collapse({len(region.loops)})"
    lines.append(pragma)
    depth = 1
    for dimension in range(len(region.loops)):
        ordinal = f"fort_internal_ordinal{dimension}"
        lines.extend(
            indent(
                [
                    f"for (std::size_t {ordinal} = 0;",
                    f"     {ordinal} < fort_internal_extent{dimension}; ++{ordinal}) {{",
                ],
                depth,
            )
        )
        depth += 1
    for dimension, loop in enumerate(region.loops):
        lines.extend(
            indent(
                [
                    f"const int {loop.iterator.cpp_name} = {iteration_value(str(dimension), f'fort_internal_ordinal{dimension}')};"
                ],
                depth,
            )
        )
    lines.extend(indent([f"{cpp_type(symbol)} {symbol.cpp_name};" for symbol in region.private_symbols], depth))
    lines.extend(sequential_block(region_body(region), depth, [0]))
    for _ in region.loops:
        depth -= 1
        lines.extend(indent(["}"], depth))
    lines.append("}")
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
    for step in plan.steps:
        if hasattr(step, "loops"):
            lines.extend(indent(_cpp_region(step)))
        else:
            lines.extend(indent([render_assignment(assignment) for assignment in step.assignments]))
    lines.extend(["}", "}", ""])
    return "\n".join(lines)
