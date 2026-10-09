"""Full-layout window workers for proved, ordered scoped GPU subchains."""

from __future__ import annotations

from compiler.emission.common.abi import dimension_name
from compiler.emission.common.c_family import cpp_type, indent
from compiler.emission.common.schedules import region_schedule, tile_counts
from compiler.emission.cuda.kernels import generate_launch
from compiler.ir import ScalarType
from compiler.offload.analysis import _add, _scale
from compiler.offload.codegen import profile_expression, query_expression


def base_cost_lines(profile, costs, *, available=True):
    result = ["fort_scope_plan_costs costs{}; costs.version = FORT_SCOPE_PLANNING_ABI_VERSION;"]
    if costs is None:
        return result
    rates = profile["rates"]
    fields = {"cpu_flops": rates["cpu_flops_per_second"], "cpu_bandwidth": rates["cpu_memory_bytes_per_second"],
              "gpu_flops": rates["gpu_flops_per_second"], "gpu_bandwidth": rates["gpu_memory_bytes_per_second"],
              "h2d_latency": rates["h2d_pageable"]["latency_seconds"],
              "h2d_bandwidth": rates["h2d_pageable"]["bandwidth_bytes_per_second"],
              "d2h_latency": rates["d2h_pageable"]["latency_seconds"],
              "d2h_bandwidth": rates["d2h_pageable"]["bandwidth_bytes_per_second"], **costs}
    return [*result, f"costs.valid = {int(available)};",
            f"costs.max_allocation_bytes = {profile['scoped']['max_allocation_bytes']}ULL;",
            *[f"costs.{field} = {float(value)!r};" for field, value in fields.items()]]


def transfer_cost_lines(profile, costs):
    result = ["fort_scope_batch_costs transfer_costs{}; transfer_costs.version = FORT_SCOPE_BATCH_ABI_VERSION;"]
    if costs is None:
        return result
    engines = profile["hardware"].get("async_engine_count")
    result += ["transfer_costs.valid = 1;", f"transfer_costs.async_engine_count = {engines or 0};",
               f"transfer_costs.max_slot_bytes = {profile['scoped']['transfers']['max_slot_bytes']}ULL;"]
    for field, value in costs.items():
        if isinstance(value, list):
            result += [f"transfer_costs.{field}[{index}] = {float(item)!r};" for index, item in enumerate(value)]
        else:
            result.append(f"transfer_costs.{field} = {float(value)!r};")
    for direction in ("h2d", "d2h"):
        rates = profile["rates"][direction + "_pinned"]
        result += [f"transfer_costs.pinned_{direction}_latency = {float(rates['latency_seconds'])!r};",
                   f"transfer_costs.pinned_{direction}_bandwidth = {float(rates['bandwidth_bytes_per_second'])!r};"]
    return result


def checked_snapshots(region, *, failure, prefix="fort_batch_original"):
    """Query bounds in source order without narrowing before the range check."""
    valid = prefix + "_valid"
    result = [f"bool {valid} = true;"]
    for axis, loop in enumerate(region.loops):
        active = " && ".join(f"{prefix}_extent{k} > 0" for k in range(axis)) or "true"
        for field, expression in (("lower", loop.lower), ("upper", loop.upper), ("stride", loop.step)):
            value = str(expression) if isinstance(expression, int) else query_expression(expression)
            result += [f"const long long {prefix}_{field}{axis} = ({active})",
                       f"    ? offload::index({value}, {valid}) : 0;",
                       f"if (!{valid} || {prefix}_{field}{axis} < -2147483648LL ||",
                       f"    {prefix}_{field}{axis} > 2147483647LL) {{ {failure} }}"]
        result += [f"if (({active}) && !{prefix}_stride{axis}) {{ {failure} }}",
                   f"const std::size_t {prefix}_extent{axis} =",
                   f"    {prefix}_stride{axis} > 0 && {prefix}_upper{axis} >= {prefix}_lower{axis}",
                   f"    ? ({prefix}_upper{axis} - {prefix}_lower{axis}) / {prefix}_stride{axis} + 1",
                   f"    : {prefix}_stride{axis} < 0 && {prefix}_lower{axis} >= {prefix}_upper{axis}",
                   f"    ? ({prefix}_lower{axis} - {prefix}_upper{axis}) / -{prefix}_stride{axis} + 1 : 0;"]
    return result


def safe_product(name, factors, *, failure):
    return [f"std::size_t {name} = 1;", *[line for factor in factors for line in (
        f"if ({factor} && {name} > std::numeric_limits<std::size_t>::max() / ({factor})) {{ {failure} }}",
        f"{name} *= ({factor});")]]


def layout_lines(arrays, *, window=False):
    result = []
    for symbol in arrays:
        name = symbol.cpp_name
        if window:
            result += [f"const fort_scope_batch_view *{name}_view = nullptr;",
                       "for (std::size_t k=0; k<fort_window->view_count; ++k)",
                       f"    if (fort_window->views[k].buffer == {name}_handle) {name}_view = &fort_window->views[k];",
                       f"if (!{name}_view) return FORT_SCOPE_ARGUMENT;",
                       f"const auto &{name}_layout = {name}_view->layout;",
                       f"auto *{name}_device = static_cast<{cpp_type(symbol)} *>({name}_view->device);"]
        else:
            result += [f"fort_scope_layout {name}_layout{{}};",
                       f"if (const int status = fort_scope_layout_get(fort_context, {name}_handle, &{name}_layout)) return status;"]
        dtype = {ScalarType.INTEGER: "FORT_SCOPE_INTEGER32", ScalarType.REAL: "FORT_SCOPE_REAL64",
                 ScalarType.REAL32: "FORT_SCOPE_REAL32", ScalarType.LOGICAL: "FORT_SCOPE_LOGICAL"}[symbol.dtype]
        result += [f"if ({name}_layout.rank != {symbol.rank} || {name}_layout.type != {dtype} || {name}_layout.element_bytes != sizeof({cpp_type(symbol)}))",
                   "    return FORT_SCOPE_ARGUMENT;"]
        result += [f"const std::size_t {dimension_name(symbol,k+1)} = {name}_layout.extents[{k}];"
                   for k in range(symbol.rank)]
    return result


def scalar_lines(scalars):
    return [line for symbol in scalars for line in (
        f"if (!fort_scalar_{symbol.cpp_name}) return FORT_SCOPE_ARGUMENT;",
        f"const {cpp_type(symbol)} &{symbol.cpp_name} = *fort_scalar_{symbol.cpp_name};")]


def window_signature(function, name):
    return ["const fort_scope_batch_window *fort_window", "std::size_t fort_first", "std::size_t fort_stop",
            "unsigned int fort_axis", "uint64_t *fort_launches",
            *[f"fort_buffer_t {symbol.cpp_name}_handle" for symbol in function.parameters if symbol.rank],
            *[f"const {cpp_type(symbol)} *fort_scalar_{symbol.cpp_name}" for symbol in function.parameters if not symbol.rank]]


def window_worker(function, plan, name):
    """Reuse each original kernel with an adjusted launch window and full views."""
    arrays = [symbol for symbol in function.parameters if symbol.rank]
    scalars = [symbol for symbol in function.parameters if not symbol.rank]
    regions = tuple(plan.regions)
    result = [f'extern "C" int {name}({", ".join(window_signature(function, name))}) {{',
              "    if (!fort_window || fort_window->version != FORT_SCOPE_BATCH_ABI_VERSION || !fort_launches ||",
              f"        fort_first >= fort_stop || fort_stop > {len(regions)}ULL) return FORT_SCOPE_ARGUMENT;",
              *indent(layout_lines(arrays, window=True)), *indent(scalar_lines(scalars))]
    for index, region in enumerate(regions):
        body = [*checked_snapshots(region, failure="return FORT_SCOPE_EXECUTION;"),
                f"if (fort_axis >= {len(region.loops)}U) return FORT_SCOPE_ARGUMENT;"]
        for axis in range(len(region.loops)):
            body += [f"if (fort_axis == {axis}U && (fort_window->begin > fort_batch_original_extent{axis} ||",
                     f"    fort_window->count > fort_batch_original_extent{axis} - fort_window->begin)) return FORT_SCOPE_ARGUMENT;",
                     f"const long long fort_batch_lower{axis} = fort_batch_original_lower{axis}",
                     f"    + (fort_axis == {axis}U ? static_cast<long long>(fort_window->begin) * fort_batch_original_stride{axis} : 0);",
                     f"if (fort_batch_lower{axis} < -2147483648LL || fort_batch_lower{axis} > 2147483647LL) return FORT_SCOPE_EXECUTION;",
                     f"const int fort_internal_lower{axis} = static_cast<int>(fort_batch_lower{axis});",
                     f"const int fort_internal_stride{axis} = static_cast<int>(fort_batch_original_stride{axis});",
                     f"const std::size_t fort_internal_extent{axis} = fort_axis == {axis}U",
                     f"    ? fort_window->count : fort_batch_original_extent{axis};"]
        schedule = region_schedule(region)
        if schedule.tile_sizes:
            body += tile_counts(region)
            body += safe_product("fort_internal_tile_volume", tuple(
                f"(fort_internal_extent{axis} < {size}ULL ? fort_internal_extent{axis} : {size}ULL)"
                for axis, size in enumerate(schedule.tile_sizes)), failure="return FORT_SCOPE_EXECUTION;")
            factors = tuple(f"fort_internal_tiles{axis}" for axis in schedule.axis_order)
        else:
            factors = tuple(f"fort_internal_extent{axis}" for axis in schedule.axis_order)
        body += safe_product("fort_internal_total", factors, failure="return FORT_SCOPE_EXECUTION;")
        body += generate_launch(region, stream="static_cast<cudaStream_t>(fort_window->stream)", profile=False,
                                prepared_bounds=True,
                                error_check="if (cudaGetLastError() != cudaSuccess) return FORT_SCOPE_EXECUTION;",
                                launch_record="++*fort_launches;")
        result += [f"    if (fort_first <= {index}ULL && {index}ULL < fort_stop) {{", *indent(body, 2), "    }"]
    return [*result, "    return FORT_SCOPE_OK;", "}"]


STORAGE = r"""struct ScopedBatchBoxes {
    struct Box { std::vector<std::size_t> lower, upper; };
    std::vector<Box> boxes;
    std::vector<fort_scope_section> sections;
    void finish() {
        sections.reserve(boxes.size());
        for (auto &box : boxes) sections.push_back({box.lower.data(),box.upper.data()});
    }
};
struct ScopedBatchBinding {
    fort_scope_batch_binding value{};
    ScopedBatchBoxes reads, writes;
    void finish() {
        reads.finish(); writes.finish();
        value.access.read_count=reads.sections.size(); value.access.reads=reads.sections.data();
        value.access.write_count=writes.sections.size(); value.access.writes=writes.sections.data();
        value.access.overwrite_count=writes.sections.size(); value.access.overwrites=writes.sections.data();
    }
};
struct ScopedBatchUnit {
    fort_scope_batch_unit value{};
    std::vector<ScopedBatchBinding> storage;
    std::vector<fort_scope_batch_binding> bindings;
    void finish() {
        bindings.reserve(storage.size());
        for (auto &binding : storage) { binding.finish(); bindings.push_back(binding.value); }
        value.bindings=bindings.data(); value.count=bindings.size();
    }
};
struct ScopedBatchModel {
    std::vector<ScopedBatchUnit> storage;
    std::vector<fort_scope_batch_unit> units;
    std::vector<ScopedBatchBinding> publication;
    std::vector<fort_scope_plan_binding> exports;
    fort_scope_batch descriptor{};
    void finish() {
        units.reserve(storage.size());
        for (auto &unit : storage) { unit.finish(); units.push_back(unit.value); }
        descriptor.version=FORT_SCOPE_BATCH_ABI_VERSION;
        descriptor.units=units.data(); descriptor.unit_count=units.size();
        exports.reserve(publication.size());
        for (auto &binding : publication) {
            binding.finish(); exports.push_back({binding.value.buffer,binding.value.access});
        }
        descriptor.exports=exports.data(); descriptor.export_count=exports.size();
    }
};""".splitlines()


def _rectangles(footprint, boxes, label, index, parameters, *, full=False):
    result = []
    for number, box in enumerate(boxes):
        name = f"fort_box_{index}_{label}_{number}"
        result += [f"ScopedBatchBoxes::Box {name};"]
        for dimension, (lower, upper, mapping) in enumerate(zip(box.lower, box.upper, box.axes, strict=True)):
            if mapping.axis is not None and not full:
                first = query_expression(_add(_scale(parameters[mapping.axis].lower, mapping.coefficient), mapping.offset))
                lo = f"(!fort_prefix[{index}] && fort_axis == {mapping.axis}U) ? ({first}) : ({query_expression(lower)})"
                hi = f"(!fort_prefix[{index}] && fort_axis == {mapping.axis}U) ? ({first}) : ({query_expression(upper)})"
            else:
                lo, hi = query_expression(lower), query_expression(upper)
            result += ["{", "    bool valid=true;",
                       f"    const auto lower=offload::index({lo},valid);",
                       f"    const auto upper=offload::index({hi},valid);",
                       f"    if (!valid || lower < 1 || upper < lower || static_cast<unsigned long long>(upper) > {dimension_name(footprint.symbol,dimension+1)}) return FORT_SCOPE_BOUNDARY;",
                       f"    {name}.lower.push_back(static_cast<std::size_t>(lower-1));",
                       f"    {name}.upper.push_back(static_cast<std::size_t>(upper));", "}"]
        result += [f"fort_binding.{label}.boxes.push_back(std::move({name}));"]
    return result


def model_preparation(function, units, name, unit_ids, *, definition_events=None, exports=()):
    """Build every candidate from one emitted copy of each numerical unit."""
    arrays = [symbol for symbol in function.parameters if symbol.rank]
    scalars = [symbol for symbol in function.parameters if not symbol.rank]
    array_indices = {symbol: index for index, symbol in enumerate(arrays)}
    parameters = ["fort_scope_t fort_context", "std::size_t fort_first", "std::size_t fort_stop", "unsigned int fort_axis",
                  "const bool *fort_prefix", "const bool *fort_used", "ScopedBatchModel &fort_model",
                  *[f"fort_buffer_t {symbol.cpp_name}_handle" for symbol in arrays],
                  *[f"const {cpp_type(symbol)} *fort_scalar_{symbol.cpp_name}" for symbol in scalars]]
    result = [f"static int {name}({', '.join(parameters)}) {{", *indent(layout_lines(arrays)), *indent(scalar_lines(scalars)),
              "    bool fort_first_worker=true;"]
    if definition_events is None:
        definition_events = {0: tuple(symbol for symbol in arrays if symbol.intent == "out")}
    if definition_events.get(0):
        result += ["    if (fort_first == 0) {"]
        for symbol in definition_events[0]:
            result += ["        fort_model.storage.emplace_back();", "        { auto &event=fort_model.storage.back();",
                       "          event.value.kind=FORT_SCOPE_PLAN_FORGET; event.storage.emplace_back();",
                       f"          event.storage.back().value.buffer={symbol.cpp_name}_handle;",
                       "          event.storage.back().value.axis=FORT_SCOPE_BATCH_FIXED_AXIS; }"]
        result += ["    }"]
    for position, unit in enumerate(units):
        region = unit.region
        body = []
        for symbol in definition_events.get(position, ()) if position else ():
            body += ["fort_model.storage.emplace_back();", "{ auto &event=fort_model.storage.back();",
                     "  event.value.kind=FORT_SCOPE_PLAN_FORGET; event.storage.emplace_back();",
                     f"  event.storage.back().value.buffer={symbol.cpp_name}_handle;",
                     "  event.storage.back().value.axis=FORT_SCOPE_BATCH_FIXED_AXIS; }"]
        body += [f"if (fort_axis >= {len(region.loops)}U) return FORT_SCOPE_BOUNDARY;",
                *checked_snapshots(region, failure="return FORT_SCOPE_BOUNDARY;"),
                *safe_product("fort_total", tuple(f"fort_batch_original_extent{axis}" for axis in range(len(region.loops))),
                              failure="return FORT_SCOPE_BOUNDARY;"),
                "if (!fort_total) return FORT_SCOPE_BOUNDARY;", "std::size_t fort_iterations=0;",
                *[f"if (fort_axis == {axis}U) fort_iterations=fort_batch_original_extent{axis};" for axis in range(len(region.loops))],
                "if (fort_first_worker) fort_model.descriptor.iterations=fort_iterations;",
                "else if (fort_model.descriptor.iterations != fort_iterations) return FORT_SCOPE_BOUNDARY;",
                "fort_model.storage.emplace_back(); auto &fort_unit=fort_model.storage.back();",
                "fort_unit.value.kind=FORT_SCOPE_PLAN_WORKER;",
                f"fort_unit.value.unit={unit_ids[region.id]}ULL;",
                f"fort_unit.value.flops=static_cast<double>(fort_total)*{unit.work_per_iteration or 0}.0;"]
        memory = sum((4 if fp.symbol.dtype.value in {"integer", "real32"} else 1 if fp.symbol.dtype.value == "logical" else 8)
                     * (len(fp.reads) + len(fp.writes)) for fp in unit.footprints)
        body += [f"fort_unit.value.memory_bytes=static_cast<double>(fort_total)*{memory}.0;"]
        for footprint in unit.footprints:
            index = array_indices[footprint.symbol]
            first_box = (*footprint.reads, *footprint.writes)[0]
            binding = ["fort_unit.storage.emplace_back(); auto &fort_binding=fort_unit.storage.back();",
                       f"fort_binding.value.buffer={footprint.symbol.cpp_name}_handle;",
                       "fort_binding.value.axis=FORT_SCOPE_BATCH_FIXED_AXIS;",
                       f"if (!fort_prefix[{index}]) {{"]
            for dimension, mapping in enumerate(first_box.axes):
                if mapping.axis is not None:
                    step = region.loops[mapping.axis].step
                    stride = str(step) if isinstance(step, int) else query_expression(step)
                    binding += [f"    if (fort_axis == {mapping.axis}U) {{ fort_binding.value.axis={dimension}U;",
                                f"        fort_binding.value.step={mapping.coefficient}LL*({stride}); }}"]
            binding += ["    if (fort_binding.value.axis == FORT_SCOPE_BATCH_FIXED_AXIS) return FORT_SCOPE_BOUNDARY;", "}",
                        *_rectangles(footprint, footprint.reads, "reads", index, region.loops),
                        *_rectangles(footprint, footprint.writes, "writes", index, region.loops)]
            body += ["{", *indent(binding), "}"]
        for index, symbol in enumerate(arrays):
            body += [f"if (fort_first_worker && !fort_used[{index}]) {{",
                     "    fort_unit.storage.emplace_back(); auto &metadata=fort_unit.storage.back();",
                     f"    metadata.value.buffer={symbol.cpp_name}_handle; metadata.value.axis=FORT_SCOPE_BATCH_FIXED_AXIS;", "}"]
        body += ["fort_first_worker=false;"]
        result += [f"    if (fort_first <= {position}ULL && {position}ULL < fort_stop) {{", *indent(body, 2), "    }"]
    for footprint in exports:
        index = array_indices[footprint.symbol]
        result += ["    { fort_model.publication.emplace_back(); auto &fort_binding=fort_model.publication.back();",
                   f"      fort_binding.value.buffer={footprint.symbol.cpp_name}_handle;",
                   *indent(_rectangles(footprint, footprint.writes, "reads", index, (), full=True), 2), "    }"]
    return [*result, "    fort_model.finish(); return FORT_SCOPE_OK;", "}"]


def worker_arguments(function):
    return [*[symbol.cpp_name + "_handle" for symbol in function.parameters if symbol.rank],
            *["fort_scalar_" + symbol.cpp_name for symbol in function.parameters if not symbol.rank]]


def callback_worker(function, name, window_name):
    arrays = [symbol for symbol in function.parameters if symbol.rank]
    scalars = [symbol for symbol in function.parameters if not symbol.rank]
    state = name + "_state"
    arguments = [*["state." + symbol.cpp_name + "_handle" for symbol in arrays],
                 *["state.fort_scalar_" + symbol.cpp_name for symbol in scalars]]
    return [f"struct {state} {{", "    std::size_t first,stop; unsigned int axis;",
            *[f"    fort_buffer_t {symbol.cpp_name}_handle;" for symbol in arrays],
            *[f"    const {cpp_type(symbol)} *fort_scalar_{symbol.cpp_name};" for symbol in scalars], "};",
            f"static int {name}(const fort_scope_batch_window *window, void *user, uint64_t *launches) {{",
            f"    auto &state=*static_cast<{state} *>(user);",
            f"    return {window_name}(window,state.first,state.stop,state.axis,launches,{','.join(arguments)});", "}"]


def attempt_helper(function, candidates, name, preparation_name, callback_name, profile, costs, transfer_costs,
                   host_threads, precision, *, compatibility="scoped_host_compatible(profile)"):
    """Preview bounded alternatives without CUDA, then execute one complete chain.

    Preparing descriptors is the only throwing phase. Once the runtime starts a
    callback its failure propagates; numerical work can never be replayed.
    """
    arrays = [symbol for symbol in function.parameters if symbol.rank]
    scalars = [symbol for symbol in function.parameters if not symbol.rank]
    signature = ["fort_scope_t fort_context", "int fort_mode", "std::size_t fort_first", "std::size_t *fort_stop",
                 *[f"fort_buffer_t {symbol.cpp_name}_handle" for symbol in arrays],
                 *[f"const {cpp_type(symbol)} *fort_scalar_{symbol.cpp_name}" for symbol in scalars]]
    arguments = worker_arguments(function)
    result = [f'extern "C" int {name}({", ".join(signature)}) {{',
              "    if (!fort_stop) return FORT_SCOPE_ARGUMENT; *fort_stop=fort_first;",
              "    if (fort_mode == FORT_SCOPE_NATIVE) return FORT_SCOPE_OK;",
              *indent(base_cost_lines(profile, costs)), *indent(transfer_cost_lines(profile, transfer_costs)),
              "    if (!costs.valid || !transfer_costs.valid) return FORT_SCOPE_OK;",
              "    const auto profile=" + profile_expression(profile if costs is not None and transfer_costs is not None else None) + ";",
              f"    if (!({compatibility})) return FORT_SCOPE_OK;",
              "    std::unique_ptr<ScopedBatchModel> best; std::size_t best_stop=fort_first; unsigned int best_axis=0;",
              "    double best_saving=-std::numeric_limits<double>::infinity();", "    try {"]
    for interval, slab, _reason in candidates:
        if slab is None:
            continue
        prefixes = {item.symbol: item.immutable_prefix for item in slab.arrays}
        table = ",".join("true" if prefixes.get(symbol, False) else "false" for symbol in arrays) or "false"
        used = ",".join("true" if symbol in prefixes else "false" for symbol in arrays) or "false"
        result += [f"        if (fort_first == {interval.start}ULL) {{",
                   f"            const bool prefix[]={{ {table} }}, used[]={{ {used} }};",
                   "            auto model=std::make_unique<ScopedBatchModel>();",
                   f"            const int prepared={preparation_name}(fort_context,{interval.start}ULL,{interval.stop}ULL,{slab.axis}U,prefix,used,*model,{','.join(arguments)});",
                   "            if (prepared != FORT_SCOPE_OK && prepared != FORT_SCOPE_BOUNDARY) return prepared;",
                   "            if (prepared == FORT_SCOPE_OK) {",
                   "                model->descriptor.execution_mode=fort_mode; fort_scope_batch_report preview{};",
                   "                const int status=fort_scope_batch_execute_v1(fort_context,&model->descriptor,&costs,&transfer_costs,-1,nullptr,nullptr,&preview);",
                   "                if (status) return status;",
                   "                const double saving=preview.baseline_seconds-preview.execution_seconds;",
                   "                if (preview.available && preview.selected_transfers != FORT_SCOPE_TRANSFERS_DIRECT && saving > best_saving) {",
                   f"                    best=std::move(model); best_stop={interval.stop}ULL; best_axis={slab.axis}U; best_saving=saving;",
                   "                }", "            }", "        }"]
    result += ["    } catch (const std::bad_alloc &) { return FORT_SCOPE_OK; }",
               "    if (!best) return FORT_SCOPE_OK;",
               "    int scope_device=-1,current_device=-1;",
               "    if (const int status=fort_scope_device_get(fort_context,&scope_device)) return status;",
               "    if (cudaGetDevice(&current_device) != cudaSuccess || current_device != scope_device ||",
               f"        !offload::compatible(profile,{host_threads},{precision})) return FORT_SCOPE_OK;",
               f"    {callback_name}_state state{{fort_first,best_stop,best_axis,{','.join(arguments)}}};",
               "    fort_scope_batch_report report{};",
               f"    const int status=fort_scope_batch_execute_v1(fort_context,&best->descriptor,&costs,&transfer_costs,1,{callback_name},&state,&report);",
               "    if (status) return status;",
               "    if (report.applied) *fort_stop=best_stop;",
               "    return FORT_SCOPE_OK;", "}"]
    return result


def numerical_batch_helpers(function, plan, candidates, units, name, unit_ids, profile, costs, transfer_costs, host_threads, precision):
    window = name + "_window_v1"
    prepare = name + "_prepare_batch"
    callback = name + "_batch_worker"
    attempt = name + "_batch_v1"
    return [*STORAGE, *window_worker(function, plan, window),
            *model_preparation(function, units, prepare, unit_ids),
            *callback_worker(function, callback, window),
            *attempt_helper(function, candidates, attempt, prepare, callback, profile, costs, transfer_costs,
                            host_threads, precision)]
