"""Submit consecutive checked device tasks through one coherent CUDA graph.

Effects remain per task: the runtime derives external reads in execution order,
so an internal producer can define storage consumed by a later kernel. No
combined INTENT-based overwrite promise is manufactured by emission.
"""

from __future__ import annotations

from compiler.emission.common.abi import abi_arguments
from compiler.emission.common.c_family import cpp_type, indent
from compiler.emission.common.schedules import region_schedule
from compiler.emission.common.symbols import region_symbols
from compiler.emission.cuda.kernels import group_kernel_name, kernel_name
from compiler.ir import ArrayAccess, referenced_symbols, walk_expr


def graph_group_eligible(regions, units, protected):
    """Host-dependent launch geometry stays at its original execution point."""
    for region in regions:
        if protected[region.id]:
            return False
        expressions = [
            expression
            for loop in region.loops
            for expression in (loop.lower, loop.upper, *((loop.step,) if not isinstance(loop.step, int) else ()))
        ]
        expressions += [
            expression
            for footprint in units[region.id].footprints
            for box in (*footprint.reads, *footprint.writes)
            for expression in (*box.lower, *box.upper)
        ]
        if any(isinstance(node, ArrayAccess) for expression in expressions for node in walk_expr(expression)):
            return False
        # Scalar values written by a mapped region must not become eager
        # arguments to a later node. Ordinary numerical proofs normally reject
        # such escaping state; keep the group boundary explicit as well.
        writes = {symbol for symbol in region.write_symbols if not symbol.rank}
        if any(writes & referenced_symbols(expression) for expression in expressions):
            return False
    return True


def _kernel_arguments(region):
    arguments = []
    for argument in abi_arguments(region_symbols(region)):
        value = argument.name + "_device" if argument.dimension is None and argument.symbol.rank else argument.name
        arguments.append(
            (
                f"std::decay_t<decltype({value})>",
                value,
                argument.symbol if argument.dimension is None and argument.symbol.rank else None,
            )
        )
    for axis in range(len(region.loops)):
        arguments.extend(
            (
                ("int", f"fort_internal_lower{axis}", None),
                ("int", f"fort_internal_stride{axis}", None),
                ("std::size_t", f"fort_internal_extent{axis}", None),
            )
        )
    arguments.append(("std::size_t", "fort_internal_total", None))
    if region_schedule(region).tile_sizes:
        arguments.append(("std::size_t", "fort_internal_tile_volume", None))
    return arguments


def generate_graph_group(
    regions,
    *,
    units,
    arrays,
    capacity,
    bounds,
    descriptors,
    array_argument,
    root_views,
    group_key,
    fallback,
    enqueue=False,
    accepted=(),
    condition="fort_gpu_requested",
    prepared_access=None,
    merged=False,
    prologue=(),
):
    """Emit a preexecution attempt with stable argument/effect storage.

    A rejected attempt falls through to the original per-task implementation.
    Graph submission errors after execution starts are terminal runtime errors.
    ``prepared_access`` names an access batch that the enclosing scope has
    already filled for the one queued task, next to that task's bounds. The
    attempt then exports those effects and computes neither again.
    ``merged`` queues all regions as the members of one launch of their group
    kernel. The runtime accepts it only when the members' actual sections are
    independent; ``accepted`` then runs only if something was queued.
    ``prologue`` lines open the attempt.
    """
    if enqueue and len(regions) != 1 and not merged:
        raise ValueError("Deferred enqueue accepts exactly one checked task")
    if prepared_access is not None and not enqueue:
        raise ValueError("Only a deferred enqueue shares its caller's access batch")
    if merged and (not enqueue or prepared_access is not None or len(regions) < 2
                   or len({region_schedule(region).cuda_threads for region in regions}) != 1
                   or any(region_schedule(region).tile_sizes for region in regions)):
        raise ValueError("A merged launch queues several untiled regions with one block size")
    declarations = [*prologue] + ([
        f"fort_scope_graph_member_v1 fort_graph_members[{len(regions)}]{{}};",
        "std::size_t fort_graph_count = 0, fort_graph_points = 0;",
    ] if merged else [
        f"fort_scope_graph_task_v1 fort_graph_tasks[{len(regions)}]{{}};",
        "std::size_t fort_graph_count = 0;",
    ])
    all_arguments = []
    preparations = []
    array_updates = []
    active_arrays = {symbol: [] for symbol in arrays}
    for ordinal, region in enumerate(regions):
        prefix = f"fort_graph_{ordinal}"
        access_name = prepared_access or prefix + "_access"
        local_declarations, local_code = [], []
        for line in descriptors(units[region.id]) if prepared_access is None else ():
            line = (
                line.replace("fort_access", access_name)
                .replace("fort_effect_", prefix + "_effect_")
                .replace("fort_valid", prefix + "_valid")
            )
            if line.startswith(("fort_scoped::", "fort_scope_access ", "std::size_t ", "fort_scope_section ", "bool ")):
                local_declarations.append(line)
            else:
                local_code.append(line)
        declarations += local_declarations
        declarations += [
            f"fort_scope_plan_binding {prefix}_bindings[{capacity}]{{}};",
            f"std::size_t {prefix}_binding_count = 0;",
            f"bool {prefix}_active = false;",
        ]
        arguments = _kernel_arguments(region)
        declarations += [f"struct {prefix}_arguments {{"]
        declarations += [f"    {dtype} value_{index}{{}};" for index, (dtype, _value, _array) in enumerate(arguments)]
        declarations += [
            f"}} {prefix}_values;",
            f"void *{prefix}_parameters[] = {{"
            + ", ".join(f"&{prefix}_values.value_{index}" for index in range(len(arguments)))
            + "};",
        ]
        if enqueue and not merged:
            declarations += [
                f"const std::size_t {prefix}_argument_sizes[] = {{"
                + ", ".join(f"sizeof({prefix}_values.value_{index})" for index in range(len(arguments)))
                + "};",
            ]
        all_arguments += [f"{prefix}_values.value_{index}" for index in range(len(arguments))]
        schedule = region_schedule(region)
        blocks = (
            "fort_internal_total"
            if schedule.tile_sizes
            else (f"(fort_internal_total - 1) / {schedule.cuda_threads} + 1")
        )
        code = [
            *(bounds(region) if prepared_access is None else ()),
            "if (fort_internal_total) {",
            *indent(local_code),
            f"    FORT_SHARED_CHECK({access_name}.export_bindings({prefix}_bindings, {capacity}, &{prefix}_binding_count));",
            f"    {prefix}_active = true;",
        ]
        code += [
            f"    fort_graph_members[fort_graph_count++] = {{{prefix}_bindings, {prefix}_binding_count}};",
            "    if (fort_internal_total > fort_graph_points) fort_graph_points = fort_internal_total;",
        ] if merged else [
            f"    auto &{prefix}_task = fort_graph_tasks[fort_graph_count++];",
            f"    {prefix}_task.bindings = {prefix}_bindings;",
            f"    {prefix}_task.binding_count = {prefix}_binding_count;",
            f"    {prefix}_task.kernel.version = FORT_SCOPE_GRAPH_ABI_VERSION;",
            f"    {prefix}_task.kernel.function = reinterpret_cast<const void *>({kernel_name(region)});",
            f"    const std::size_t {prefix}_blocks = {blocks};",
            f"    {prefix}_task.kernel.grid[0] = static_cast<unsigned int>({prefix}_blocks > 65535 ? 65535 : {prefix}_blocks);",
            f"    {prefix}_task.kernel.grid[1] = {prefix}_task.kernel.grid[2] = 1;",
            f"    {prefix}_task.kernel.block[0] = {schedule.cuda_threads};",
            f"    {prefix}_task.kernel.block[1] = {prefix}_task.kernel.block[2] = 1;",
            f"    {prefix}_task.kernel.arguments = {prefix}_parameters;",
            f"    {prefix}_task.kernel.argument_count = {len(arguments)};",
        ]
        for index, (_dtype, value, array) in enumerate(arguments):
            assignment = f"{prefix}_values.value_{index} = {value};"
            if array is None:
                code.append("    " + assignment)
            else:
                array_updates.append(f"if ({prefix}_active) {assignment}")
                active_arrays[array].append(prefix + "_active")
        preparations += ["{", *indent(code), "    }", "}"]
    requested = [(symbol, active) for symbol, active in active_arrays.items() if active]
    reserves, pointers = [], []
    if requested:
        reserves += [
            f"fort_buffer_t fort_graph_handles[{len(requested)}]{{}};",
            f"void *fort_graph_pointers[{len(requested)}]{{}};",
            "std::size_t fort_graph_reserve_count = 0;",
        ]
    for symbol, active in requested:
        slot = "fort_graph_slot_" + str(symbol.id)
        active_condition = " || ".join(dict.fromkeys(active))
        reserves += [
            f"std::size_t {slot} = 0;",
            f"if ({active_condition}) {{",
            f"    {slot} = fort_graph_reserve_count;",
            f"    fort_graph_handles[fort_graph_reserve_count++] = {symbol.cpp_name}_handle;",
            "}",
        ]
        pointer = f"fort_graph_pointers[{slot}]"
        if root_views:
            value_type = ("const " if symbol.intent == "in" else "") + cpp_type(symbol)
            value = (
                f"fort_scoped::RootView<{value_type}, {symbol.rank}>({symbol.cpp_name}_layout, "
                f"static_cast<{cpp_type(symbol)} *>({pointer}))"
            )
        else:
            value = f"static_cast<{cpp_type(symbol)} *>({pointer})"
        pointers += [f"if ({active_condition}) {symbol.cpp_name}_device = {value};"]
    if requested:
        reserves += [
            "FORT_SHARED_CHECK(fort_scope_device_reserve_many_v1("
            "fort_context, fort_graph_handles, fort_graph_reserve_count, fort_graph_pointers));",
            *pointers,
        ]
    submit = (
        f"if (!fort_graph_count) return FORT_SCOPE_OK;\n"
        f"return fort_scope_graph_enqueue_v1(fort_context, {group_key}ULL, fort_graph_tasks, fort_graph_0_argument_sizes);"
        if enqueue
        else f"return fort_scope_graph_execute_v1(fort_context, {group_key}ULL, fort_graph_tasks, fort_graph_count);"
    )
    if merged:
        threads = region_schedule(regions[0]).cuda_threads
        submit = "\n".join([
            "if (!fort_graph_count) return FORT_SCOPE_OK;",
            "void *fort_graph_parameters[] = {" + ", ".join("&" + value for value in all_arguments) + "};",
            "const std::size_t fort_graph_argument_sizes[] = {"
            + ", ".join(f"sizeof({value})" for value in all_arguments) + "};",
            "fort_scope_kernel_v1 fort_graph_kernel{};",
            "fort_graph_kernel.version = FORT_SCOPE_GRAPH_ABI_VERSION;",
            f"fort_graph_kernel.function = reinterpret_cast<const void *>({group_kernel_name(regions)});",
            # Every thread strides through each member, so the largest member sizes the grid.
            f"const std::size_t fort_graph_blocks = (fort_graph_points - 1) / {threads} + 1;",
            "fort_graph_kernel.grid[0] = static_cast<unsigned int>(fort_graph_blocks > 65535 ? 65535 : fort_graph_blocks);",
            "fort_graph_kernel.grid[1] = fort_graph_kernel.grid[2] = 1;",
            f"fort_graph_kernel.block[0] = {threads};",
            "fort_graph_kernel.block[1] = fort_graph_kernel.block[2] = 1;",
            "fort_graph_kernel.arguments = fort_graph_parameters;",
            f"fort_graph_kernel.argument_count = {len(all_arguments)};",
            f"FORT_SHARED_CHECK(fort_scope_graph_enqueue_members_v1(fort_context, {group_key}ULL, &fort_graph_kernel, "
            "fort_graph_argument_sizes, fort_graph_members, fort_graph_count));",
            "fort_graph_queued = true;",
            "return FORT_SCOPE_OK;",
        ])
    attempt = [
        *declarations,
        *preparations,
        *reserves,
        *array_updates,
        *submit.splitlines(),
    ]
    if merged:
        accepted = ["if (fort_graph_queued) {", *indent(accepted), "}"]
    return [
        "{",
        "    bool fort_graph_done = false;",
        *(["    bool fort_graph_queued = false;"] if merged else []),
        f"    if ({condition}) {{",
        "        const int fort_graph_status = [&]() -> int {",
        *indent(attempt, 3),
        "        }();",
        "        if (!fort_graph_status) {",
        "            fort_graph_done = true;",
        *indent(accepted, 3),
        "        }",
        "        else if (fort_graph_status != FORT_SCOPE_RESOURCE && fort_graph_status != FORT_SCOPE_BOUNDARY)",
        "            return fort_graph_status;",
        "    }",
        "    if (!fort_graph_done) {",
        *indent(fallback, 2),
        "    }",
        "}",
    ]
