"""Emit CUDA region kernels and their source-order launches."""

from __future__ import annotations

from typing import TYPE_CHECKING

from compiler.emission.c.declarations import cpp_declaration
from compiler.emission.common.abi import abi_arguments
from compiler.emission.common.c_family import cpp_type, indent
from compiler.emission.common.loops import iteration_value, mapped_snapshots, sequential_block
from compiler.emission.common.symbols import region_body, region_symbols

if TYPE_CHECKING:
    from compiler.ir import ParallelRegion


def kernel_name(region: ParallelRegion) -> str:
    return f"kernel_region_{region.id}_device"


def generate_kernel(region: ParallelRegion) -> list[str]:
    symbols = region_symbols(region)
    parameters = [cpp_declaration(argument) for argument in abi_arguments(symbols)]
    for dimension in range(len(region.loops)):
        parameters.extend(
            [
                f"int fort_internal_lower{dimension}",
                f"int fort_internal_stride{dimension}",
                f"std::size_t fort_internal_extent{dimension}",
            ]
        )
    parameters.append("std::size_t fort_internal_total")
    lines = [f"__global__ void {kernel_name(region)}("]
    lines.extend(
        indent([parameter + ("," if index < len(parameters) - 1 else "") for index, parameter in enumerate(parameters)])
    )
    lines.extend(
        [
            ") {",
            "    std::size_t fort_internal_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;",
            "    if (fort_internal_index >= fort_internal_total) return;",
        ]
    )
    for dimension in reversed(range(len(region.loops))):
        name = region.loops[dimension].iterator.cpp_name
        lines.append(
            f"    const int {name} = "
            f"{iteration_value(str(dimension), f'fort_internal_index % fort_internal_extent{dimension}')};"
        )
        lines.append(f"    fort_internal_index /= fort_internal_extent{dimension};")
    lines.extend(indent([f"{cpp_type(symbol)} {symbol.cpp_name};" for symbol in region.private_symbols]))
    lines.extend(sequential_block(region_body(region), 1, [0], device=True))
    lines.extend(["}", ""])
    return lines


def generate_launch(region: ParallelRegion) -> list[str]:
    lines = ["{", f"    // Source-order parallel region {region.id}."]
    lines.extend(indent(mapped_snapshots(region)))
    total = " * ".join(f"fort_internal_extent{dimension}" for dimension in range(len(region.loops)))
    lines.extend(
        [
            f"    const std::size_t fort_internal_total = {total};",
            "    if (fort_internal_total > 0) {",
            "        constexpr unsigned int fort_internal_threads = 256;",
            "        const unsigned int fort_internal_blocks = static_cast<unsigned int>(",
            "            (fort_internal_total - 1) / fort_internal_threads + 1);",
        ]
    )
    arguments: list[str] = []
    for argument in abi_arguments(region_symbols(region)):
        arguments.append(
            argument.name + "_device" if argument.dimension is None and argument.symbol.rank else argument.name
        )
    for dimension in range(len(region.loops)):
        arguments.extend(
            [
                f"fort_internal_lower{dimension}",
                f"fort_internal_stride{dimension}",
                f"fort_internal_extent{dimension}",
            ]
        )
    arguments.append("fort_internal_total")
    lines.append(f"        {kernel_name(region)}<<<fort_internal_blocks, fort_internal_threads>>>(")
    lines.extend(
        indent([argument + ("," if index < len(arguments) - 1 else "") for index, argument in enumerate(arguments)], 3)
    )
    lines.extend(["        );", "        CUCH(cudaGetLastError());", "    }", "}"])
    return lines
