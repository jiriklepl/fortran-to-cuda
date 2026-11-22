"""Emit CUDA kernels and launches from explicit arbitrary-rank schedules."""

from __future__ import annotations

from typing import TYPE_CHECKING

from compiler.emission.c.declarations import cpp_declaration
from compiler.emission.common.abi import abi_arguments
from compiler.emission.common.c_family import cpp_type, indent
from compiler.emission.common.loops import mapped_coordinates, mapped_snapshots, sequential_block
from compiler.emission.common.schedules import checked_product, region_schedule, tile_counts
from compiler.emission.common.symbols import region_body, region_symbols

if TYPE_CHECKING:
    from compiler.ir import ParallelRegion


def kernel_name(region: ParallelRegion) -> str:
    return f"kernel_region_{region.id}_device"


def points_name(region: ParallelRegion) -> str:
    return f"kernel_region_{region.id}_points"


def group_kernel_name(regions) -> str:
    return "kernel_group_" + "_".join(str(region.id) for region in regions) + "_device"


def _point_body(region: ParallelRegion, depth: int) -> list[str]:
    lines = indent(mapped_coordinates(region), depth)
    lines.extend(indent([f"{cpp_type(symbol)} {symbol.cpp_name};" for symbol in region.private_symbols], depth))
    lines.extend(sequential_block(region_body(region), depth, [0], device=True, addressing=region.addressing))
    return lines


def _untiled_body(region: ParallelRegion) -> list[str]:
    schedule = region_schedule(region)
    lines = [
        "    const std::size_t fort_internal_grid_stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;",
        "    std::size_t fort_internal_point = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;",
        "    while (fort_internal_point < fort_internal_total) {",
        "        std::size_t fort_internal_index = fort_internal_point;",
    ]
    for position, axis in enumerate(schedule.axis_order):
        if position + 1 == len(schedule.axis_order):
            # The checked product and point < total prove that the remaining
            # quotient is already below this final extent. Its modulo is the
            # identity, and the subsequent division would be unused.
            lines.extend(indent([f"const std::size_t fort_internal_ordinal{axis} = fort_internal_index;"], 2))
            continue
        lines.extend(
            indent(
                [
                    f"const std::size_t fort_internal_ordinal{axis} = fort_internal_index % fort_internal_extent{axis};",
                    f"fort_internal_index /= fort_internal_extent{axis};",
                ],
                2,
            )
        )
    lines.extend(_point_body(region, 2))
    lines.extend(
        [
            "        if (fort_internal_total - fort_internal_point <= fort_internal_grid_stride) break;",
            "        fort_internal_point += fort_internal_grid_stride;",
            "    }",
        ]
    )
    return lines


def _tiled_body(region: ParallelRegion) -> list[str]:
    schedule = region_schedule(region)
    lines = indent(tile_counts(region))
    for axis, size in enumerate(schedule.tile_sizes):
        lines.extend(
            indent(
                [
                    f"const std::size_t fort_internal_width{axis} = "
                    f"fort_internal_extent{axis} < {size}ULL ? fort_internal_extent{axis} : {size}ULL;"
                ]
            )
        )
    lines.extend(
        [
            "    std::size_t fort_internal_tile = blockIdx.x;",
            "    while (fort_internal_tile < fort_internal_total) {",
            "        std::size_t fort_internal_tile_index = fort_internal_tile;",
        ]
    )
    for axis in schedule.axis_order:
        lines.extend(
            indent(
                [
                    f"const std::size_t fort_internal_begin{axis} = "
                    f"(fort_internal_tile_index % fort_internal_tiles{axis}) * fort_internal_width{axis};",
                    f"fort_internal_tile_index /= fort_internal_tiles{axis};",
                ],
                2,
            )
        )
    lines.extend(
        [
            "        std::size_t fort_internal_point = threadIdx.x;",
            "        while (fort_internal_point < fort_internal_tile_volume) {",
            "            std::size_t fort_internal_point_index = fort_internal_point;",
        ]
    )
    for axis in schedule.axis_order:
        lines.extend(
            indent(
                [
                    f"const std::size_t fort_internal_ordinal{axis} = fort_internal_begin{axis} + "
                    f"fort_internal_point_index % fort_internal_width{axis};",
                    f"fort_internal_point_index /= fort_internal_width{axis};",
                ],
                3,
            )
        )
    active = " && ".join(f"fort_internal_ordinal{axis} < fort_internal_extent{axis}" for axis in schedule.axis_order)
    lines.extend(indent([f"if ({active}) {{"], 3))
    lines.extend(_point_body(region, 4))
    lines.extend(
        [
            "            }",
            "            if (fort_internal_tile_volume - fort_internal_point <= blockDim.x) break;",
            "            fort_internal_point += blockDim.x;",
            "        }",
            "        if (fort_internal_total - fort_internal_tile <= gridDim.x) break;",
            "        fort_internal_tile += gridDim.x;",
            "    }",
        ]
    )
    return lines


def _kernel_parameters(region: ParallelRegion) -> list[str]:
    schedule = region_schedule(region)
    parameters = [cpp_declaration(argument) for argument in abi_arguments(region_symbols(region))]
    for axis in range(len(region.loops)):
        parameters.extend(
            [
                f"int fort_internal_lower{axis}",
                f"int fort_internal_stride{axis}",
                f"std::size_t fort_internal_extent{axis}",
            ]
        )
    parameters.append("std::size_t fort_internal_total")
    if schedule.tile_sizes:
        parameters.append("std::size_t fort_internal_tile_volume")
    return parameters


def _signature(prefix: str, parameters: list[str]) -> list[str]:
    lines = [prefix + "("]
    lines.extend(
        indent([parameter + ("," if index < len(parameters) - 1 else "") for index, parameter in enumerate(parameters)])
    )
    lines.append(") {")
    return lines


def generate_kernel(region: ParallelRegion) -> list[str]:
    schedule = region_schedule(region)
    lines = _signature(f"static __global__ void {kernel_name(region)}", _kernel_parameters(region))
    lines.extend(_tiled_body(region) if schedule.tile_sizes else _untiled_body(region))
    lines.extend(["}", ""])
    return lines


def generate_group_kernel(regions) -> list[str]:
    """One kernel that runs several untiled regions, for regions that do not depend on each other.

    Every thread takes its share of the points of each member in turn, so the
    members interleave across threads exactly as independent launches could.
    The parameters are those of the members' own kernels, one member after
    another; a member with no points is passed a zero total.
    """
    lines = []
    parameters, calls = [], []
    for position, region in enumerate(regions):
        if region_schedule(region).tile_sizes:
            raise ValueError("A group kernel takes untiled regions only")
        own = _kernel_parameters(region)
        lines.extend(_signature(
            f"static __device__ __forceinline__ void {points_name(region)}",
            [*own, "const std::size_t fort_internal_grid_stride", "std::size_t fort_internal_point"],
        ))
        # The region's point loop, without the two declarations these parameters replace.
        lines.extend(_untiled_body(region)[2:])
        lines.extend(["}", ""])
        names = []
        for index, parameter in enumerate(own):
            declared_type, _, _name = parameter.rpartition(" ")
            names.append(f"fort_member{position}_{index}")
            parameters.append(f"{declared_type} {names[-1]}")
        calls.append(f"    {points_name(region)}({', '.join(names)}, fort_internal_grid_stride, fort_internal_point);")
    lines.extend(_signature(f"static __global__ void {group_kernel_name(regions)}", parameters))
    lines.extend(
        [
            "    const std::size_t fort_internal_grid_stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;",
            "    const std::size_t fort_internal_point = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;",
            *calls,
            "}",
            "",
        ]
    )
    return lines


def launch_bounds(region: ParallelRegion) -> list[str]:
    """Capture protected bounds and checked work dimensions in source order."""
    schedule = region_schedule(region)
    lines = mapped_snapshots(region)
    if schedule.tile_sizes:
        lines.extend(tile_counts(region))
        total_factors = tuple(f"fort_internal_tiles{axis}" for axis in schedule.axis_order)
        volume_factors = tuple(
            f"(fort_internal_extent{axis} < {size}ULL ? fort_internal_extent{axis} : {size}ULL)"
            for axis, size in enumerate(schedule.tile_sizes)
        )
        lines.extend(checked_product("fort_internal_tile_volume", volume_factors))
    else:
        total_factors = tuple(f"fort_internal_extent{axis}" for axis in schedule.axis_order)
    lines.extend(checked_product("fort_internal_total", total_factors))
    return lines


def generate_launch(
    region: ParallelRegion, *, stream: str | None = None, profile: bool = True,
    prepared_bounds: bool = False, error_check: str | None = None, launch_record: str | None = None,
) -> list[str]:
    schedule = region_schedule(region)
    lines = ["{", f"    // Scheduled parallel region {region.id}."]
    if not prepared_bounds:
        lines.extend(indent(launch_bounds(region)))
    blocks = "fort_internal_total" if schedule.tile_sizes else "(fort_internal_total - 1) / fort_internal_threads + 1"
    lines.extend(
        [
            "    if (fort_internal_total > 0) {",
            f"        constexpr unsigned int fort_internal_threads = {schedule.cuda_threads};",
            f"        const std::size_t fort_internal_needed_blocks = {blocks};",
            "        const unsigned int fort_internal_blocks = static_cast<unsigned int>(",
            "            fort_internal_needed_blocks > 65535 ? 65535 : fort_internal_needed_blocks);",
        ]
    )
    arguments: list[str] = []
    for argument in abi_arguments(region_symbols(region)):
        arguments.append(
            argument.name + "_device" if argument.dimension is None and argument.symbol.rank else argument.name
        )
    for axis in range(len(region.loops)):
        arguments.extend(
            [
                f"fort_internal_lower{axis}",
                f"fort_internal_stride{axis}",
                f"fort_internal_extent{axis}",
            ]
        )
    arguments.append("fort_internal_total")
    if schedule.tile_sizes:
        arguments.append("fort_internal_tile_volume")
    if profile:
        lines.append("        timing::measure_kernel_executions([&]() {")
    configuration = "fort_internal_blocks, fort_internal_threads" + (f", 0, {stream}" if stream else "")
    lines.append(f"            {kernel_name(region)}<<<{configuration}>>>(")
    lines.extend(
        indent([argument + ("," if index < len(arguments) - 1 else "") for index, argument in enumerate(arguments)], 4)
    )
    lines.append("            );")
    if profile:
        lines.append("        });")
    lines.extend([
        "        " + (error_check or "CUCH(cudaGetLastError());"),
        "        " + (launch_record or 'storage::trace("kernel");'), "    }", "}",
    ])
    return lines
