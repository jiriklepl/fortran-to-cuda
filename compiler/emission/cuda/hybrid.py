"""Emit bounded asynchronous CUDA slabs and disjoint serial CPU windows.

The dependence/footprint analyzer owns partition legality. This emitter consumes
its explicit ChunkPlan; it does not infer safety from source text or recipes.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from compiler.emission.c.declarations import cpp_declaration
from compiler.emission.common.abi import abi_arguments, dimension_name
from compiler.emission.common.c_family import cpp_type, indent
from compiler.emission.common.loops import mapped_coordinates, mapped_snapshots, sequential_block
from compiler.emission.common.schedules import checked_product, region_schedule
from compiler.emission.common.symbols import region_body
from compiler.ir import ParallelRegion, ScalarType
from compiler.offload.codegen import profile_expression, query_expression


@dataclass(frozen=True)
class HybridEmission:
    available: bool
    reason: str | None
    helpers: str
    body: list[str]
    decision_body: list[str]
    entry_name: str
    cpu_entry_name: str


def hybrid_support(function, plan, analysis):
    if not analysis.available or analysis.chunk is None:
        return False, analysis.chunk_reason or analysis.reason or "no independent slab partition"
    if not plan.steps or any(not isinstance(step, ParallelRegion) for step in plan.steps):
        return False, "hybrid windows require a flat sequence of proved parallel regions"
    if not analysis.chunk.arrays:
        return False, "hybrid windows require physical array captures"
    parameters = set(function.parameters)
    if any(array.symbol not in parameters for array in analysis.chunk.arrays):
        return False, "hybrid array captures must belong to the ordinary entry ABI"
    return True, None


def _signature(name, parameters, result="void"):
    return [
        f"static {result} {name}(",
        *indent([p + ("," if i + 1 < len(parameters) else "") for i, p in enumerate(parameters)]),
        ") {",
    ]


def _point(region, depth, *, device=False):
    return [
        *indent(mapped_coordinates(region), depth),
        *indent([f"{cpp_type(symbol)} {symbol.cpp_name};" for symbol in region.private_symbols], depth),
        *sequential_block(region_body(region), depth, [0], device=device, addressing=region.addressing),
    ]


def _cpu_window(name, parameters, plan, axis):
    lines = _signature(name, [*parameters, "std::size_t fort_hybrid_begin", "std::size_t fort_hybrid_count"])
    for region in plan.regions:
        lines.extend(indent(["{", *indent(mapped_snapshots(region))]))
        depth = 2
        for dimension in reversed(region_schedule(region).axis_order):
            begin = "fort_hybrid_begin" if dimension == axis else "0"
            end = "fort_hybrid_begin + fort_hybrid_count" if dimension == axis else f"fort_internal_extent{dimension}"
            lines.extend(
                indent(
                    [
                        f"for (std::size_t fort_internal_ordinal{dimension} = {begin};",
                        f"     fort_internal_ordinal{dimension} < {end}; ++fort_internal_ordinal{dimension}) {{",
                    ],
                    depth,
                )
            )
            depth += 1
        lines.extend(_point(region, depth))
        for _ in region.loops:
            depth -= 1
            lines.extend(indent(["}"], depth))
        lines.extend(indent(["}"]))
    return [*lines, "}", ""]


def _kernel(name, function, region, chunk):
    arrays = {array.symbol for array in chunk.arrays}
    parameters = []
    for argument in abi_arguments(function.parameters):
        if argument.dimension is None and argument.symbol in arrays:
            symbol = argument.symbol
            const = "const " if symbol.intent == "in" else ""
            parameters.append(f"hybrid::View<{const}{cpp_type(symbol)}, {symbol.rank}> {argument.name}")
        else:
            parameters.append(cpp_declaration(argument))
    for axis in range(len(region.loops)):
        parameters.extend(
            [
                f"int fort_internal_lower{axis}",
                f"int fort_internal_stride{axis}",
                f"std::size_t fort_internal_extent{axis}",
            ]
        )
    parameters.extend(["std::size_t fort_hybrid_begin", "std::size_t fort_internal_total"])
    lines = [
        f"__global__ void {name}(",
        *indent([p + ("," if i + 1 < len(parameters) else "") for i, p in enumerate(parameters)]),
        ") {",
        "    const std::size_t fort_hybrid_stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;",
        "    std::size_t fort_hybrid_point = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;",
        "    while (fort_hybrid_point < fort_internal_total) {",
        "        std::size_t fort_hybrid_index = fort_hybrid_point;",
    ]
    for axis in region_schedule(region).axis_order:
        offset = " + fort_hybrid_begin" if axis == chunk.axis else ""
        lines.extend(
            indent(
                [
                    f"const std::size_t fort_internal_ordinal{axis} = fort_hybrid_index % fort_internal_extent{axis}{offset};",
                    f"fort_hybrid_index /= fort_internal_extent{axis};",
                ],
                2,
            )
        )
    lines.extend(_point(region, 2, device=True))
    lines.extend(
        [
            "        if (fort_internal_total - fort_hybrid_point <= fort_hybrid_stride) break;",
            "        fort_hybrid_point += fort_hybrid_stride;",
            "    }",
            "}",
            "",
        ]
    )
    return lines


def _arrays(chunk):
    lines = ["std::vector<hybrid::Array> fort_hybrid_arrays;", "bool fort_hybrid_valid = true;"]
    for array in chunk.arrays:
        symbol = array.symbol
        lower = [value for value in (array.read_lower_offset, array.write_lower_offset) if value is not None]
        upper = [value for value in (array.read_upper_offset, array.write_upper_offset) if value is not None]

        def bound(values, operation):
            rendered = [f"offload::index({query_expression(value)}, fort_hybrid_valid)" for value in values]
            return rendered[0] if len(rendered) == 1 else f"std::{operation}({', '.join(rendered)})"

        dimensions = ", ".join(dimension_name(symbol, axis + 1) for axis in range(symbol.rank))
        lines.extend(
            [
                "fort_hybrid_arrays.push_back({",
                f"    const_cast<{cpp_type(symbol)} *>({symbol.cpp_name}), sizeof({cpp_type(symbol)}),",
                f"    {{{dimensions}}}, {array.dimension}, {array.coefficient},",
                f"    {bound(lower, 'min')}, {bound(upper, 'max')},",
                f"    {'true' if array.write_lower_offset is not None else 'false'}" + "});",
            ]
        )
    return lines


def _query_snapshots(region):
    lines = ["bool fort_hybrid_bounds_valid = true;"]
    for axis, loop in enumerate(region.loops):
        active = " && ".join(f"fort_internal_extent{i} > 0" for i in range(axis)) or "true"
        for field, expression in (("lower", loop.lower), ("upper", loop.upper), ("stride", loop.step)):
            value = str(expression) if isinstance(expression, int) else query_expression(expression)
            lines.extend(
                [
                    f"const long long fort_hybrid_{field}{axis} = ({active})",
                    f"    ? offload::index({value}, fort_hybrid_bounds_valid) : 0;",
                    f"if (!fort_hybrid_bounds_valid || fort_hybrid_{field}{axis} < -2147483648LL ||",
                    f"    fort_hybrid_{field}{axis} > 2147483647LL) return {{}};",
                    f"const int fort_internal_{field}{axis} = static_cast<int>(fort_hybrid_{field}{axis});",
                ]
            )
        lines.extend(
            [
                f"if (({active}) && !fort_internal_stride{axis}) return {{}};",
                f"const std::size_t fort_internal_extent{axis} =",
                f"    fort_internal_stride{axis} > 0 && fort_internal_upper{axis} >= fort_internal_lower{axis}",
                f"    ? (fort_hybrid_upper{axis} - fort_hybrid_lower{axis}) / fort_internal_stride{axis} + 1",
                f"    : fort_internal_stride{axis} < 0 && fort_internal_lower{axis} >= fort_internal_upper{axis}",
                f"    ? (fort_hybrid_lower{axis} - fort_hybrid_upper{axis}) / -fort_hybrid_stride{axis} + 1 : 0;",
            ]
        )
    return lines


def generate_hybrid(function, plan, analysis, profile=None, policy="hybrid", *, host_threads=4, collective=False):
    """Return namespace-scope helpers and ordinary-entry/query body fragments.

    ``body`` synchronizes all CPU windows and both streams before returning.
    ``decision_body`` reads only shape/bound scalars and offline profile data.
    Unsupported plans retain a correct serial CPU fallback for direct ABI calls.
    """
    if policy not in {"hybrid", "chunked"}:
        raise ValueError("Hybrid policy must be hybrid or chunked")
    if not isinstance(host_threads, int) or host_threads < 1:
        raise ValueError("Hybrid host thread budget must be positive")
    suffix = sha256(f"{function.module}::{function.name}".encode()).hexdigest()[:12]
    name = f"fort_hybrid_{suffix}"
    abi = abi_arguments(function.parameters)
    parameters = [cpp_declaration(argument) for argument in abi]
    query_parameters = [
        f"const {cpp_type(argument.symbol)}& {argument.name}"
        if argument.dimension is None and not argument.symbol.rank
        else cpp_declaration(argument)
        for argument in abi
    ]
    arguments = ", ".join(argument.name for argument in abi)

    def call(entry):
        return f"{entry}({arguments});"

    available, reason = hybrid_support(function, plan, analysis)
    cpu_entry = name + "_cpu_entry"
    collective_literal = "true" if collective else "false"
    if not available:
        lines = _signature(name + "_cpu_serial", parameters)
        lines.extend(indent([f"{cpp_type(s)} {s.cpp_name};" for s in function.symbols if not s.parameter]))
        lines.extend(sequential_block(function.body, 1, [0]))
        lines.extend(["}", *_signature(cpu_entry, parameters)])
        lines.append(f'    offload::decision_trace("{function.name}", "native", 0, {len(plan.regions)});')
        if collective:
            lines.extend(
                [
                    "    if (offload::in_parallel()) {",
                    "        #pragma omp single",
                    "        { " + call(name + "_cpu_serial") + " }",
                    "    } else {",
                ]
            )
            lines.append("        " + call(name + "_cpu_serial"))
            lines.append("    }")
        else:
            lines.append("    " + call(name + "_cpu_serial"))
        lines.extend(["}", ""])
        return HybridEmission(
            False,
            reason,
            "\n".join(lines),
            [call(cpu_entry)],
            [f'offload::decision_trace("{function.name}", "native", 0, {len(plan.regions)});', "return 0;"],
            cpu_entry,
            cpu_entry,
        )

    chunk = analysis.chunk
    first = plan.regions[0]
    axis = chunk.axis
    snapshot = mapped_snapshots(first)
    snapshot += checked_product(
        "fort_hybrid_total_points", tuple(f"fort_internal_extent{i}" for i in range(len(first.loops)))
    )
    snapshot += ["if (!fort_hybrid_total_points) return {};"]
    lines = _cpu_window(name + "_cpu_window", parameters, plan, axis)
    for region in plan.regions:
        lines.extend(_kernel(name + f"_kernel_{region.id}", function, region, chunk))
    lines.extend(_signature(name + "_select", query_parameters, "hybrid::Choice"))
    lines.append("    offload::DecisionRange fort_hybrid_decision_range;")
    lines.append(f"    if (!offload::context_valid({host_threads}, {collective_literal})) return {{}};")
    # Queries must decline an unrepresentable domain rather than terminate the
    # application. Keep the normal dispatcher's checked-product diagnostics,
    # and alter only the selector's error branch of the shared arithmetic.
    query_product = [
        line.replace("std::abort();", "return {};")
        for line in snapshot[len(mapped_snapshots(first)) :]
        if "std::cerr" not in line
    ]
    lines.extend(indent([*_query_snapshots(first), *query_product]))
    lines.extend(
        [
            "    hybrid::Choice fort_hybrid_native;",
            f"    fort_hybrid_native.total_iterations = fort_internal_extent{axis};",
        ]
    )
    lines.append(f"    static const offload::Profile fort_hybrid_profile = {profile_expression(profile)};")
    precision = 64 if any(s.dtype is ScalarType.REAL for s in function.parameters) else 32
    lines.append(
        "    const bool fort_hybrid_compatible = fort_hybrid_profile.valid && "
        f"fort_hybrid_profile.threads == {host_threads} && fort_hybrid_profile.precision == {precision};"
    )
    if policy == "hybrid":
        lines.append("    if (!fort_hybrid_compatible) return fort_hybrid_native;")
        if any(unit.work_per_iteration is None or unit.work_is_upper_bound for unit in analysis.units):
            lines.append("    return fort_hybrid_native; // Uncertain work has no calibrated cost estimate.")
    lines.extend(indent(_arrays(chunk)))
    lines.append("    if (!fort_hybrid_valid) return fort_hybrid_native;")
    work = sum(unit.work_per_iteration or 0 for unit in analysis.units)
    traffic = sum(
        (4 if footprint.symbol.dtype in {ScalarType.INTEGER, ScalarType.REAL32} else 8)
        * (len(footprint.reads) + len(footprint.writes))
        for unit in analysis.units
        for footprint in unit.footprints
    )
    lines.extend(
        [
            "    auto fort_hybrid_selected = hybrid::select(fort_hybrid_arrays,",
            f"        fort_internal_extent{axis}, fort_internal_lower{axis}, fort_internal_stride{axis},",
            f"        static_cast<double>(fort_hybrid_total_points) * {work}.0,",
            f"        static_cast<double>(fort_hybrid_total_points) * {traffic}.0, {len(plan.regions)},",
            f"        fort_hybrid_profile, fort_hybrid_compatible, {host_threads}, "
            f"{'true' if policy == 'chunked' else 'false'}, "
            f"{int((profile or {}).get('hardware', {}).get('async_engine_count', 1))});",
            f"    fort_hybrid_selected.total_iterations = fort_internal_extent{axis};",
        ]
    )
    lines.append("    if (fort_hybrid_selected.gpu_iterations) {")
    if policy == "hybrid":
        lines.append(
            f"        if (!offload::compatible(fort_hybrid_profile, {host_threads}, {precision})) return fort_hybrid_native;"
        )
    elif profile is not None:
        lines.extend(
            [
                f"        if (fort_hybrid_compatible && !offload::compatible(fort_hybrid_profile, {host_threads}, {precision})) {{",
                "            fort_hybrid_selected = hybrid::select(fort_hybrid_arrays,",
                f"                fort_internal_extent{axis}, fort_internal_lower{axis}, fort_internal_stride{axis},",
                f"                0, 0, {len(plan.regions)}, fort_hybrid_profile, false, {host_threads}, true);",
                f"            fort_hybrid_selected.total_iterations = fort_internal_extent{axis};",
                "        }",
            ]
        )
    lines.extend(
        [
            "        if (cudaGetDevice(&fort_hybrid_selected.device) != cudaSuccess) return fort_hybrid_native;",
            "    }",
            "    return fort_hybrid_selected;",
            "}",
            "",
        ]
    )

    # Reuse this one dispatcher for the ordinary chosen path and explicit CPU
    # fallback. Its CPU callback never introduces an OpenMP team or workshare.
    lines.extend(_signature(name + "_dispatch", [*parameters, "hybrid::Choice fort_hybrid_choice"]))
    lines.extend(indent(snapshot[:-1]))
    lines.extend(
        [
            "    if (!fort_hybrid_total_points) {",
            f'        offload::decision_trace("{function.name}", "native", 0, 0, "slabs");',
            "        return;",
            "    }",
        ]
    )
    lines.extend(indent(["std::vector<hybrid::Array> fort_hybrid_arrays;", "if (fort_hybrid_choice.gpu_iterations) {"]))
    lines.extend(indent(_arrays(chunk)[1:], 2))
    lines.append("        if (!fort_hybrid_valid) { fort_hybrid_choice = {}; fort_hybrid_arrays.clear(); }")
    lines.append("    }")
    lines.extend(
        [
            "    auto fort_hybrid_cpu = [=](std::size_t begin, std::size_t count) {",
            f"        {name}_cpu_window({arguments}, begin, count);",
            "    };",
            "    auto fort_hybrid_gpu = [=](hybrid::Slot &slot, std::size_t begin, std::size_t count) {",
        ]
    )
    factors = tuple("count" if i == axis else f"fort_internal_extent{i}" for i in range(len(first.loops)))
    lines.extend(indent(checked_product("fort_hybrid_points", factors), 2))
    lines.extend(
        [
            "        if (!fort_hybrid_points) return;",
            "        const unsigned blocks = static_cast<unsigned>(std::min<std::size_t>(65535, (fort_hybrid_points - 1) / 256 + 1));",
        ]
    )
    physical = {array.symbol: index for index, array in enumerate(chunk.arrays)}
    kernel_arguments = []
    for argument in abi:
        if argument.dimension is None and argument.symbol in physical:
            symbol = argument.symbol
            const = "const " if symbol.intent == "in" else ""
            kernel_arguments.append(f"slot.view<{const}{cpp_type(symbol)}, {symbol.rank}>({physical[symbol]})")
        else:
            kernel_arguments.append(argument.name)
    for dimension in range(len(first.loops)):
        kernel_arguments.extend(
            [
                f"fort_internal_lower{dimension}",
                f"fort_internal_stride{dimension}",
                "count" if dimension == axis else f"fort_internal_extent{dimension}",
            ]
        )
    kernel_arguments += ["begin", "fort_hybrid_points"]
    for region in plan.regions:
        lines.extend(
            [
                f"        {name}_kernel_{region.id}<<<blocks, 256, 0, slot.stream>>>(",
                *indent(
                    [
                        argument + ("," if i + 1 < len(kernel_arguments) else "")
                        for i, argument in enumerate(kernel_arguments)
                    ],
                    3,
                ),
                "        );",
                "        CUCH(cudaGetLastError());",
                '        storage::trace("kernel");',
            ]
        )
    lines.extend(
        [
            "    };",
            "    hybrid::execute(std::move(fort_hybrid_arrays),",
            f"        fort_internal_extent{axis}, fort_internal_lower{axis}, fort_internal_stride{axis},",
            f"        fort_hybrid_choice, {host_threads}, {collective_literal}, fort_hybrid_cpu, fort_hybrid_gpu,",
            f'        "{function.name}", "{policy}");',
            "}",
            "",
            *_signature(cpu_entry, parameters),
            f"    {name}_dispatch({arguments}, hybrid::Choice{{}});",
            "}",
            "",
            *_signature(name, parameters),
        ]
    )
    if collective:
        lines.extend(
            [
                "    hybrid::Choice fort_hybrid_choice;",
                "    if (offload::in_parallel()) {",
                "        #pragma omp barrier",
                "        #pragma omp single copyprivate(fort_hybrid_choice)",
                f"        {{ fort_hybrid_choice = {name}_select({arguments}); }}",
                "    } else {",
                f"        fort_hybrid_choice = {name}_select({arguments});",
                "    }",
                f"    {name}_dispatch({arguments}, fort_hybrid_choice);",
            ]
        )
    else:
        lines.append(f"    {name}_dispatch({arguments}, {name}_select({arguments}));")
    lines.extend(["}", ""])
    return HybridEmission(
        True,
        None,
        "\n".join(lines),
        [call(name)],
        [
            f"const auto fort_hybrid_decision = {name}_select({arguments});",
            f'if (!fort_hybrid_decision.gpu_iterations) offload::decision_trace("{function.name}", "native", 0, '
            'fort_hybrid_decision.total_iterations, "slabs");',
            "return fort_hybrid_decision.gpu_iterations > 0 ? 1 : 0;",
        ],
        name,
        cpu_entry,
    )
