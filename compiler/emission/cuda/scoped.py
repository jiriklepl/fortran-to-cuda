"""Numerical entries borrowing common buffers through the versioned public ABI."""

import json
from dataclasses import dataclass, replace
from hashlib import sha256

from compiler.emission.c.declarations import cpp_declaration
from compiler.emission.common.abi import abi_arguments, dimension_name
from compiler.emission.common.c_family import cpp_type, indent, render_assignment, render_expression
from compiler.emission.common.loops import _loop_snapshot, sequential_block
from compiler.emission.common.schedules import checked_product, region_schedule, tile_counts
from compiler.emission.common.symbols import host_symbols, region_symbols
from compiler.emission.cuda.compute_memory import COMPUTE_MEMORY_INCLUDES, generate_compute_memory
from compiler.emission.cuda.kernels import generate_kernel, generate_launch
from compiler.emission.cuda.native_preflight import generate_native_preflight
from compiler.emission.cuda.offload import _cpu_worker, _host_profile_compatibility, _metadata, _precision
from compiler.emission.cuda.structured import _query_expression
from compiler.emission.fortran.formatting import _fortran_list
from compiler.ir import (
    ArrayAccess,
    CompilationError,
    ConditionalRegion,
    HostBlock,
    ParallelRegion,
    Reference,
    ScalarType,
    SequentialRegion,
    referenced_symbols,
    walk_expr,
)
from compiler.offload.analysis import (
    OffloadAnalysis,
    Unit,
    _protected_scalar_inputs,
    _unit_footprints,
    scope_slab_candidates,
)
from compiler.offload.codegen import profile_expression, query_expression
from compiler.offload.preparation import prepare_offload
from compiler.offload.profile import (
    ProfileError,
    compiler_identity,
    scoped_collective_costs,
    scoped_costs,
    scoped_transfer_costs,
    validate_profile,
)


@dataclass(frozen=True)
class ScopedEmission:
    cuda: str
    fortran: str
    report: dict


TYPES = {
    ScalarType.REAL: ("FORT_SCOPE_REAL64", "real(c_double)"),
    ScalarType.REAL32: ("FORT_SCOPE_REAL32", "real(c_float)"),
    ScalarType.INTEGER: ("FORT_SCOPE_INTEGER32", "integer(c_int)"),
    ScalarType.LOGICAL: ("FORT_SCOPE_LOGICAL", "logical(c_bool)"),
}

ENTRY_ABI_VERSION = 2


def generate_scoped(function, plan, config, common_header, *, runtime_id=None, root_views=False, root_view_abi=1):
    if root_view_abi not in {1, 2}:
        raise CompilationError("unsupported borrowed numerical view ABI")
    view_type = f"fort_scope_view_v{root_view_abi}"
    view_layout_type = f"fort_scope_view_layout_v{root_view_abi}"
    view_get = f"fort_scope_view_get_v{root_view_abi}"
    if root_views and config.collective:
        raise CompilationError("borrowed section workers require a serial host coordinator")
    arrays = tuple(s for s in function.parameters if s.rank)
    scalars = tuple(s for s in function.parameters if not s.rank)
    if any(s.intent != "in" for s in scalars):
        raise CompilationError("shared numerical entries require read-only scalar parameters")
    identity = f"entry-abi:{ENTRY_ABI_VERSION}:{function.module.lower()}::{function.name.lower()}:{plan!r}"
    if function.requires_numerical_environment:
        identity += ":numerical-environment:nearest-nontrapping-v1"
    digest = sha256(identity.encode()).hexdigest()[:12]
    if root_views:
        digest = sha256((digest + f":root-view-v{root_view_abi}").encode()).hexdigest()[:12]
    name = "fort_shared_" + digest
    c_name = "cpp_" + name
    configure_name = c_name + "_configure"
    transfer_modes = {"direct": "FORT_SCOPE_TRANSFERS_DIRECT", "pinned": "FORT_SCOPE_TRANSFERS_PINNED",
                      "pipelined": "FORT_SCOPE_TRANSFERS_PIPELINED", "auto": "FORT_SCOPE_TRANSFERS_AUTO"}
    transfer_reason = {"auto": "transfer_estimates_unavailable", "pipelined": "pipelined_not_available"}.get(config.scope_transfers)
    scalar_signature = [f"const {cpp_type(s)} *fort_scalar_{s.cpp_name}" for s in scalars]
    def array_parameter(symbol):
        return (f"const {view_type} *{symbol.cpp_name}_view" if root_views else
                f"fort_buffer_t {symbol.cpp_name}_handle")

    def array_argument(symbol):
        return symbol.cpp_name + ("_view" if root_views else "_handle")

    batch_type = "ViewAccessBatch" if root_views else "AccessBatch"
    signature = ["fort_scope_t fort_context", "int fort_mode",
                 *[array_parameter(s) for s in arrays], *scalar_signature]
    abi = abi_arguments(function.parameters)
    worker_arguments = ", ".join(a.name for a in abi)
    capacity = max(len(arrays), 1)
    lines = ['#include <cuda_runtime.h>', '#include <omp.h>', '#define FORT_OFFLOAD_ENABLED 1',
             f'#include "{common_header}"', '#include "scoped_entry.hpp"',
             '#define FORT_SHARED_CHECK(expr) do { int fort_check_result = (expr); if (fort_check_result) return fort_check_result; } while (false)',
             f"namespace generated_kernels::{name} {{", "using namespace indexing;"]
    if root_views:
        lines.insert(5, '#include "view_entry.hpp"')
    lines += _host_profile_compatibility(config.host_threads, _precision(function))
    invariants = set(function.parameters)

    def collect(current):
        for step in current.steps:
            if isinstance(step, HostBlock):
                invariants.update(a.target.symbol for a in step.assignments
                                  if isinstance(a.target, Reference) and a.target.symbol.dtype is ScalarType.INTEGER)
            elif isinstance(step, ConditionalRegion):
                collect(step.then_plan)
                collect(step.else_plan)

    collect(plan)
    prep = prepare_offload(function, plan)
    units = {r.id: _unit_footprints(Unit(i, r, (), None), frozenset(invariants))
             for i, r in enumerate(plan.regions)}
    protected = {r.id: _protected_scalar_inputs(r, frozenset(function.parameters))[0]
                 for r in plan.regions}
    protected_footprints = {
        r.id: any(referenced_symbols(e) & protected[r.id]
                  for fp in units[r.id].footprints for box in (*fp.reads, *fp.writes)
                  for e in (*box.lower, *box.upper))
        for r in plan.regions
    }
    if config.native_participation is not None:
        from compiler.offload.source_compute import apply_source_compute_costs
        numerical_analysis = apply_source_compute_costs(
            OffloadAnalysis(prep.analysis.available, prep.analysis.reason, tuple(units.values())),
            config.profile, config.native_participation, array_views=root_views)
        units = {unit.region.id: unit for unit in numerical_analysis.units}
    elif not config.collective:
        from compiler.offload.numerical_calibration import apply_numerical_costs
        numerical_analysis = apply_numerical_costs(
            OffloadAnalysis(prep.analysis.available, prep.analysis.reason, tuple(units.values())), config.profile)
        units = {unit.region.id: unit for unit in numerical_analysis.units}
    query_available = prep.analysis.available
    query_reason = prep.analysis.reason
    planning_reason = query_reason
    if prep.analysis.available and any(u.work_per_iteration is None or u.work_is_upper_bound for u in units.values()):
        planning_reason = "; ".join(dict.fromkeys(u.work_estimate_reason or "work estimate is unknown or conditional"
            for u in units.values() if u.work_per_iteration is None or u.work_is_upper_bound))
    if function.requires_numerical_environment:
        # The current protocol fixtures do not include flagged entry guards or
        # their collective barriers. A source hash alone cannot price them.
        planning_reason = "numerical_environment_protocol_calibration_unavailable"
    planning_available = prep.analysis.available and planning_reason is None
    compute_models = [unit.compute_model for unit in units.values() if unit.compute_model is not None]
    source_compute = config.native_participation is not None
    affinity = compute_models[0]["cpu_affinity"] if compute_models else None
    runtime_schedule = any(model.get("runtime_schedule") is not None for model in compute_models)
    if runtime_schedule:
        lines += ["static bool scoped_compute_schedule_compatible() {",
                  "    omp_sched_t schedule; int chunk = -1;",
                  "    omp_get_schedule(&schedule, &chunk);",
                  "    const auto kind = static_cast<unsigned int>(schedule) & 0x7fffffffU;",
                  "    return kind == static_cast<unsigned int>(omp_sched_static) && chunk == 0;", "}"]
    if affinity is not None:
        lines.insert(0, '#include <sched.h>')
        protocol_environment = compute_models[0].get("cpu_protocol_environment")
        environment_guard = ""
        if protocol_environment is not None:
            lines[0:0] = ['#include <cstdlib>', '#include <cstring>']
            for field, variable in (("omp_wait_policy", "OMP_WAIT_POLICY"), ("gomp_spincount", "GOMP_SPINCOUNT")):
                value = protocol_environment[field]
                environment_guard += (f' && !std::getenv("{variable}")' if value is None else
                    f' && std::getenv("{variable}") && !std::strcmp(std::getenv("{variable}"), {json.dumps(value)})')
        lines += ["static bool scoped_compute_placement_compatible() {",
                  "    cpu_set_t actual; CPU_ZERO(&actual);",
                  "    if (sched_getaffinity(0, sizeof(actual), &actual)) return false;",
                  f"    if (CPU_COUNT(&actual) != {len(affinity)}) return false;",
                  *[f"    if (!CPU_ISSET({cpu}, &actual)) return false;" for cpu in affinity],
                  f"    return omp_get_level() == 0 && !omp_get_dynamic() && omp_get_max_threads() == {config.host_threads} && omp_get_thread_limit() >= {config.host_threads} && omp_get_proc_bind() == omp_proc_bind_false{' && scoped_compute_schedule_compatible()' if runtime_schedule else ''}{environment_guard};", "}"]
    else:
        lines += ["static bool scoped_compute_placement_compatible() { return true; }"]
    profile_reason = config.profile_reason
    costs = None
    transfer_costs = None
    transfer_profile_reason = "hardware profile is missing"
    team_costs = None
    if config.profile is None:
        profile_reason = profile_reason or "hardware profile is missing"
    elif runtime_id is None:
        profile_reason = "shared runtime calibration identity is unavailable"
    else:
        try:
            validate_profile(config.profile, precision_bits=_precision(function), cpu_threads=config.host_threads,
                             scoped_runtime_id=runtime_id)
            compiler_identity(config.profile)
            costs = scoped_costs(config.profile, runtime_id)
            try:
                transfer_costs = scoped_transfer_costs(config.profile, runtime_id)
                transfer_profile_reason = None
            except ProfileError as error:
                transfer_profile_reason = str(error)
        except ProfileError as error:
            profile_reason = str(error)
            transfer_profile_reason = str(error)
    if config.collective:
        try:
            if costs is None:
                reason = "collective synchronization calibration is unavailable"
                if profile_reason:
                    reason += ": " + profile_reason
                raise ProfileError(reason)
            team_costs = scoped_collective_costs(config.profile, runtime_id, cpu_threads=config.host_threads)
        except ProfileError as error:
            costs = None
            profile_reason = str(error)
        if function.requires_numerical_environment:
            team_costs = None
    team_configure_name = c_name + "_configure_team_v1"
    if config.collective:
        lines.insert(5, '#include "scoped_team_observer.hpp"')
        lines += [f"static int {team_configure_name}(fort_scope_t fort_context) {{"]
        if team_costs is None:
            lines += ["    return fort_scope_set_team_costs_v1(fort_context, nullptr, 0);"]
        else:
            lines += ["    fort_scope_team_costs team{}; team.version=FORT_SCOPE_TEAM_ABI_VERSION; team.valid=1;",
                      f"    team.cpu_threads={config.host_threads}; team.expected_omp_level=1;",
                      "    team.protocol_id=FORT_SCOPE_TEAM_PROTOCOL_ID;"]
            lines += [f"    team.{field}={float(value)!r};" for field,value in team_costs.items()
                      if field not in {"cpu_flops", "cpu_bandwidth"}]
            lines += ["    static const auto profile=" + profile_expression(config.profile) + ";",
                      "    return fort_scope_set_team_costs_v1(fort_context, &team, scoped_host_compatible(profile));"]
        lines += ["}"]
    if config.scope_transfers == "pinned" and transfer_costs is None:
        # Pinned bandwidth alone omits staging preparation, packing and changed
        # copy geometry. Planning v1 has no complete calibrated model for it.
        costs = None
        profile_reason = "transfer_estimates_unavailable"
    if transfer_costs is None or config.scope_transfers == "direct":
        lines += [f'extern "C" int {configure_name}(fort_scope_t fort_context) {{',
                  *(([f"    FORT_SHARED_CHECK(fort_scope_set_transfers(fort_context, {transfer_modes[config.scope_transfers]}));",
                       f"    return {team_configure_name}(fort_context);"]) if config.collective else
                    [f"    return fort_scope_set_transfers(fort_context, {transfer_modes[config.scope_transfers]});"]), "}"]
    else:
        from compiler.emission.cuda.batch import transfer_cost_lines
        lines += [f'extern "C" int {configure_name}(fort_scope_t fort_context) {{',
                  f"    if (const int status = fort_scope_set_transfers(fort_context, {transfer_modes[config.scope_transfers]})) return status;",
                  *indent(transfer_cost_lines(config.profile, transfer_costs)),
                  "    const auto profile = " + profile_expression(config.profile) + ";",
                  *((["    FORT_SHARED_CHECK(fort_scope_set_transfer_costs_v1(fort_context, &transfer_costs, scoped_host_compatible(profile)));",
                       f"    return {team_configure_name}(fort_context);"]) if config.collective else
                    ["    return fort_scope_set_transfer_costs_v1(fort_context, &transfer_costs, scoped_host_compatible(profile));"]), "}"]
    unit_ids = {r.id: int(sha256(f"{name}:region:{r.id}".encode()).hexdigest()[:16], 16) for r in plan.regions}
    batch_analysis, batch_candidates = scope_slab_candidates(function, plan)
    batch_requested = config.scope_transfers in {"pipelined", "auto"}
    batch_eligible = batch_requested and not config.collective and not root_views and any(slab for _, slab, _ in batch_candidates)
    batch_enabled = batch_eligible and costs is not None and transfer_costs is not None
    batch_name = c_name + "_batch_v1"
    window_name = c_name + "_window_v1"
    captures = {}
    locals_ = host_symbols(function, plan)
    for region in plan.regions:
        used = set(region_symbols(region))
        for loop in region.loops:
            for expression in (loop.lower, loop.upper) + (() if isinstance(loop.step, int) else (loop.step,)):
                used.update(referenced_symbols(expression))
        captures[region.id] = tuple(s for s in locals_ if s in used)
        extra_signature = "".join(f", const {cpp_type(s)} &{s.cpp_name}" for s in captures[region.id])
        # An ordinary nonvolatile reference allows the host optimizer to hoist
        # a guarded invariant load. Protected inputs need observable reads at
        # their original use sites, including in outlined OpenMP workers.
        worker_signature = ", ".join(
            cpp_declaration(a, array_views=root_views) if a.symbol.rank else
            f"const {'volatile ' if a.symbol in protected[region.id] else ''}{cpp_type(a.symbol)} &{a.name}"
            for a in abi)
        lines.extend(generate_kernel(region, array_views=root_views))
        lines.extend(_cpu_worker(units[region.id], worker_signature + extra_signature, f"cpu_{region.id}"))
    if batch_eligible:
        from compiler.emission.cuda.batch import numerical_batch_helpers
        lines[0:0] = ["#include <memory>", "#include <vector>"]
        lines += numerical_batch_helpers(function, plan, batch_candidates, batch_analysis.units, c_name, unit_ids,
                                          config.profile, costs, transfer_costs, config.host_threads, _precision(function))
    lines += [f'extern "C" int {c_name}({", ".join(signature)}) {{',
              *(["    bool fort_entry_work_started=false;",
                 "    const int fort_entry_result=[&]() -> int {"] if root_views else []),
              "    if (fort_scope_abi_version() != FORT_SCOPE_ABI_VERSION)",
              '        return fort_scope_report_error(FORT_SCOPE_ARGUMENT, "incompatible shared runtime ABI");',
              "    if (fort_mode < 0 || fort_mode > 2)",
              '        return fort_scope_report_error(FORT_SCOPE_ARGUMENT, "unknown shared execution mode");']
    if function.requires_numerical_environment:
        lines += ["    if (!fort_runtime::numerical_environment_supported())",
                  '        return fort_scope_report_error(FORT_SCOPE_BOUNDARY, "unsupported numerical environment for real short reduction");']
    setup = []
    for s in scalars:
        setup += [f"    if (!fort_scalar_{s.cpp_name})",
                  '        return fort_scope_report_error(FORT_SCOPE_ARGUMENT, "null shared scalar argument");',
                  f"    const {cpp_type(s)} &{s.cpp_name} = *fort_scalar_{s.cpp_name};"]
    writes = set()

    def written(current):
        for step in current.steps:
            writes.update(step.write_symbols)
            if isinstance(step, ConditionalRegion):
                written(step.then_plan)
                written(step.else_plan)

    written(plan)
    layout_setup = []
    for i, a in enumerate(arrays):
        for b in arrays[:i]:
            if not root_views and (a in writes or b in writes):
                lines += [f"    if ({a.cpp_name}_handle == {b.cpp_name}_handle)",
                          '        return fort_scope_report_error(FORT_SCOPE_ALIAS, "writable entry arguments alias");']
        field = a.cpp_name + "_layout"
        if root_views:
            value_type = ("const " if a.intent == "in" else "") + cpp_type(a)
            layout_setup += [f"    {view_layout_type} {field}{{}};",
                             f"    FORT_SHARED_CHECK({view_get}(fort_context, {a.cpp_name}_view, &{field}));",
                             f"    const fort_buffer_t {a.cpp_name}_handle = {a.cpp_name}_view->buffer;",
                             f"    if ({field}.root.type != {TYPES[a.dtype][0]} || {field}.root.element_bytes != sizeof({cpp_type(a)}) ||",
                             f"        {field}{'.root' if root_view_abi == 1 else ''}.rank != {a.rank})",
                             '        return fort_scope_report_error(FORT_SCOPE_ARGUMENT, "entry view layout/type mismatch");',
                             f"    for (std::size_t axis=0; axis<{a.rank}; ++axis)",
                             f"        if ({field}.lower_bounds[axis] != 1)",
                             '            return fort_scope_report_error(FORT_SCOPE_BOUNDARY, "numerical view requires its prepared one-based dummy bounds");',
                             f"    fort_scoped::RootView<{value_type}, {a.rank}> {a.cpp_name}({field}, static_cast<{value_type} *>({field}.root.host));",
                             f"    fort_scoped::RootView<{value_type}, {a.rank}> {a.cpp_name}_device;"]
        else:
            layout_setup += [f"    fort_scope_layout {field}{{}};",
                  f"    FORT_SHARED_CHECK(fort_scope_layout_get(fort_context, {a.cpp_name}_handle, &{field}));",
                  f"    if ({field}.rank != {a.rank} || {field}.type != {TYPES[a.dtype][0]} ||",
                  f"        {field}.element_bytes != sizeof({cpp_type(a)}))",
                  '        return fort_scope_report_error(FORT_SCOPE_ARGUMENT, "entry layout/type mismatch");',
                  f"    auto *{a.cpp_name} = static_cast<{cpp_type(a)} *>({field}.host);",
                  f"    {cpp_type(a)} *{a.cpp_name}_device = nullptr;"]
        layout_setup += [f"    const std::size_t {dimension_name(a,k+1)} = {field}.extents[{k}];" for k in range(a.rank)]
    setup += layout_setup
    alias_setup = []
    if root_views:
        alias_setup += ["    {", f"        fort_scoped::ViewAccessBatch<{capacity}> fort_aliases(fort_context);"]
        for array in arrays:
            flags = "FORT_SCOPE_WRITE_ALL | FORT_SCOPE_OVERWRITE_ALL" if array in writes or array.intent == "out" else "FORT_SCOPE_READ_ALL"
            alias_setup += [f"        fort_scope_access fort_alias_{array.cpp_name}{{}}; fort_alias_{array.cpp_name}.flags={flags};",
                            f"        FORT_SHARED_CHECK(fort_aliases.add({array.cpp_name}_view, fort_alias_{array.cpp_name}));"]
        alias_setup += ["    }"]
        setup += alias_setup
    lines += setup
    lines += [f"    {cpp_type(s)} {s.cpp_name};" for s in host_symbols(function, plan)]
    if batch_enabled:
        from compiler.emission.cuda.batch import worker_arguments as batch_worker_arguments
        batch_arguments = batch_worker_arguments(function)
        lines += ["    std::size_t fort_batch_skip=0;",
                  f"    FORT_SHARED_CHECK({batch_name}(fort_context,fort_mode,0,&fort_batch_skip,{','.join(batch_arguments)}));"]
    for s in arrays:
        if s.intent == "out":
            forget = (f"fort_scoped::forget_view(fort_context, {s.cpp_name}_view, false)" if root_views else
                      f"fort_scope_forget_definition(fort_context, {s.cpp_name}_handle)")
            lines += [f"    {'if (!fort_batch_skip) ' if batch_enabled else ''}FORT_SHARED_CHECK({forget});"]
    lines += ["    const bool fort_gpu_requested = fort_mode == 1;"]

    def prefetch(symbols, *, planning=False, query_helper=False):
        ordered = [s for s in arrays if s in symbols]
        if not ordered:
            return []
        result = ["{", f"    fort_scoped::{batch_type}<{capacity}> fort_metadata(fort_context);",
                  "    fort_scope_access fort_read{}; fort_read.flags = FORT_SCOPE_READ_ALL;"]
        def check(expression):
            if query_helper:
                return f"if ((fort_query_status = {expression})) {{ d.valid = false; return d; }}"
            return f"FORT_SHARED_CHECK({expression});"
        for s in ordered:
            result += ["    " + check(f"fort_metadata.add({array_argument(s)}, fort_read)")]
        if planning:
            return [*result, "    " + check("fort_metadata.record(FORT_SCOPE_PLAN_NATIVE, 0, 0, 0, false)"), "}"]
        return [*result, "    FORT_SHARED_CHECK(fort_metadata.begin(false));",
                "    FORT_SHARED_CHECK(fort_metadata.finish());", "}"]

    def expression_reads(expressions):
        return {n.symbol for e in expressions for n in walk_expr(e) if isinstance(n, ArrayAccess)}

    def bounds(region):
        result = []
        for axis, loop in enumerate(region.loops):
            active = " && ".join(f"fort_internal_extent{k} > 0" for k in range(axis))
            expressions = (loop.lower, loop.upper) + (() if isinstance(loop.step, int) else (loop.step,))
            reads = prefetch(expression_reads(expressions))
            if reads:
                result += [f"if ({active or 'true'}) {{", *indent(reads), "}"]
            result += _loop_snapshot(loop, str(axis), active or None)
        schedule = region_schedule(region)
        if schedule.tile_sizes:
            result += tile_counts(region)
            result += checked_product("fort_internal_tile_volume", tuple(
                f"(fort_internal_extent{a} < {size}ULL ? fort_internal_extent{a} : {size}ULL)"
                for a, size in enumerate(schedule.tile_sizes)))
            factors = tuple(f"fort_internal_tiles{a}" for a in schedule.axis_order)
        else:
            factors = tuple(f"fort_internal_extent{a}" for a in schedule.axis_order)
        return [*result, *checked_product("fort_internal_total", factors)]

    def descriptors(unit, *, planning=False):
        result = [f"fort_scoped::{batch_type}<{capacity}> fort_access(fort_context);", "bool fort_valid = true;"]
        checked = planning or any(isinstance(node, ArrayAccess)
                                  for fp in unit.footprints for box in (*fp.reads, *fp.writes)
                                  for expression in (*box.lower, *box.upper) for node in walk_expr(expression))
        if checked and not planning:
            result += ["offload::Data d;"]
        value = (lambda expression: _query_expression(expression, {})) if checked else query_expression
        invalid = 'return fort_scope_report_error(FORT_SCOPE_BOUNDARY, "invalid checked planning footprint");' if planning else 'return fort_scope_report_error(FORT_SCOPE_ARGUMENT, "invalid physical entry footprint");'

        for fp in unit.footprints:
            s = fp.symbol
            prefix = "fort_effect_" + s.cpp_name
            result += [f"fort_scope_access {prefix}{{}};"]
            if protected_footprints[unit.region.id]:
                # Evaluating a section's affine offset can itself read the
                # protected scalar. Conservative native effects need no such
                # evaluation; preserve the payload and its original IF guard.
                flags = (1 if fp.reads or fp.full_read else 0) | (2 if fp.writes or fp.full_write else 0)
                result += [f"{prefix}.flags = {flags};",
                           f"FORT_SHARED_CHECK(fort_access.add({array_argument(s)}, {prefix}));"]
                continue
            for label, boxes, full, flag in (("read", fp.reads, fp.full_read, "FORT_SCOPE_READ_ALL"),
                                            ("write", fp.writes, fp.full_write, "FORT_SCOPE_WRITE_ALL")):
                if full:
                    result += [f"{prefix}.flags |= {flag};"]
                    continue
                if not boxes:
                    continue
                for i, box in enumerate(boxes):
                    base = f"{prefix}_{label}_{i}"
                    result += [f"std::size_t {base}_lower[{s.rank}]{{}}, {base}_upper[{s.rank}]{{}};"]
                    for axis, (lo, hi) in enumerate(zip(box.lower, box.upper, strict=True)):
                        result += ["{",
                                   f"    const auto lo = offload::index({value(lo)}, fort_valid);",
                                   f"    const auto hi = offload::index({value(hi)}, fort_valid);",
                                   f"    if ({'!d.valid || ' if checked else ''}!fort_valid || lo < 1 || hi < lo || static_cast<unsigned long long>(hi) > {dimension_name(s,axis+1)})",
                                   "        " + invalid,
                                   f"    {base}_lower[{axis}] = static_cast<std::size_t>(lo-1);",
                                   f"    {base}_upper[{axis}] = static_cast<std::size_t>(hi);", "}"]
                items = ", ".join(f"{{{prefix}_{label}_{i}_lower, {prefix}_{label}_{i}_upper}}" for i in range(len(boxes)))
                result += [f"fort_scope_section {prefix}_{label}s[] = {{{items}}};",
                           f"{prefix}.{label}_count = {len(boxes)}; {prefix}.{label}s = {prefix}_{label}s;"]
                if label == "write" and fp.exact:
                    result += [f"{prefix}.overwrite_count = {len(boxes)}; {prefix}.overwrites = {prefix}_writes;"]
            result += [f"FORT_SHARED_CHECK(fort_access.add({array_argument(s)}, {prefix}));"]
        return result

    def host(step, executable):
        touched = set(step.read_symbols) | set(step.write_symbols)
        result = ["{", f"    fort_scoped::{batch_type}<{capacity}> fort_host(fort_context);"]
        for s in arrays:
            if s in touched:
                flags = (1 if s in step.read_symbols else 0) | (2 if s in step.write_symbols else 0)
                result += [f"    fort_scope_access fort_effect_{s.id}{{}}; fort_effect_{s.id}.flags = {flags};",
                           f"    FORT_SHARED_CHECK(fort_host.add({array_argument(s)}, fort_effect_{s.id}));"]
        return [*result, "    FORT_SHARED_CHECK(fort_host.begin(false));", "    fort_host.executing();",
                *(["    fort_entry_work_started=true;"] if root_views else []),
                *indent(executable), "    FORT_SHARED_CHECK(fort_host.finish());", "}"]

    def execution(current):
        result = []
        for step in current.steps:
            if isinstance(step, ConditionalRegion):
                result += prefetch(expression_reads((step.condition,)))
                result += [f"if ({render_expression(step.condition)}) {{", *indent(execution(step.then_plan)),
                           "} else {", *indent(execution(step.else_plan)), "}"]
            elif isinstance(step, HostBlock):
                result += host(step, [render_assignment(a) for a in step.assignments])
            elif isinstance(step, SequentialRegion):
                result += host(step, sequential_block(step.body, 0, [0]))
            elif isinstance(step, ParallelRegion):
                unit = units[step.id]
                if batch_enabled:
                    position = next(index for index, region in enumerate(plan.regions) if region.id == step.id)
                    result += [f"if ({position}ULL >= fort_batch_skip) {{"]
                    if position:
                        result += [f"    FORT_SHARED_CHECK({batch_name}(fort_context,fort_mode,{position}ULL,&fort_batch_skip,{','.join(batch_arguments)}));"]
                    result += [f"if ({position}ULL >= fort_batch_skip) {{"]
                result += ["{", *indent(bounds(step)), "    if (fort_internal_total) {"]
                # A CUDA launch captures scalar values before its kernel body.
                # References preserve CPU guards, but cannot make that capture
                # lazy. Keep only the affected worker native; subsequent safe
                # workers can still use the same coherent device allocations.
                gpu_choice = "false" if protected[step.id] else "fort_gpu_requested"
                body = [*descriptors(unit), f"bool fort_gpu = {gpu_choice};"]
                body += [f"if (fort_mode == 2) FORT_SHARED_CHECK(fort_access.decision({unit_ids[step.id]}ULL, fort_gpu));"]
                if protected[step.id]:
                    body += [f'if (fort_gpu_requested) offload::decision_trace("{function.name}", "native-scoped-protected-scalar", 0, 1);']
                body += [
                        "int fort_status = fort_access.begin(fort_gpu);",
                        "if (fort_gpu && (fort_status == FORT_SCOPE_RESOURCE || fort_status == FORT_SCOPE_BOUNDARY)) {",
                        f'    offload::decision_trace("{function.name}", fort_status == FORT_SCOPE_RESOURCE ? "native-scoped-resource" : "native-scoped-boundary", 0, 1);',
                        "    fort_gpu = false; fort_status = fort_access.begin(false);", "}",
                        "FORT_SHARED_CHECK(fort_status);", "fort_scoped::GPUCall fort_gpu_call(fort_context);",
                        "if (fort_gpu) {", "    fort_status = fort_gpu_call.begin();",
                        "    if (fort_status == FORT_SCOPE_RESOURCE) {", "        fort_access.cancel(); fort_gpu = false;",
                        f'        offload::decision_trace("{function.name}", "native-scoped-gpu-setup", 0, 1);',
                        "        fort_status = fort_access.begin(false);", "    }", "    FORT_SHARED_CHECK(fort_status);", "}"]
                for s in arrays:
                    pointer = f"static_cast<{cpp_type(s)} *>(fort_access.device({s.cpp_name}_handle))"
                    value_type = ("const " if s.intent == "in" else "") + cpp_type(s)
                    device = (f"fort_scoped::RootView<{value_type}, {s.rank}>({s.cpp_name}_layout, {pointer})"
                              if root_views else pointer)
                    body += [f"if (fort_gpu) {s.cpp_name}_device = {device};"]
                body += ["fort_access.executing();", *(["fort_entry_work_started=true;"] if root_views else []), "if (fort_gpu) {", *indent(generate_launch(
                    step, stream="static_cast<cudaStream_t>(fort_gpu_call.stream)", profile=False, prepared_bounds=True,
                    error_check='if (const auto error = cudaGetLastError(); error != cudaSuccess) return fort_scope_execution_error(fort_context, cudaGetErrorString(error));',
                    launch_record="FORT_SHARED_CHECK(fort_scope_note_launch(fort_context));")), "} else {"]
                arguments = worker_arguments + "".join(f", {s.cpp_name}" for s in captures[step.id])
                cpu = ["#ifdef _OPENMP", "if (omp_in_parallel()) {",
                       f"    cpu_{step.id}({arguments}, 0, 1);", "} else {",
                       f"    #pragma omp parallel num_threads({config.host_threads})", "    {",
                       f"        cpu_{step.id}({arguments}, omp_get_thread_num(), omp_get_num_threads());",
                       "    }", "}", "#else", f"cpu_{step.id}({arguments}, 0, 1);", "#endif"]
                body += [*indent(cpu), "}", "FORT_SHARED_CHECK(fort_access.finish());"]
                result += [*indent(body, 2), "    }", "}"]
                if batch_enabled:
                    result += ["}", "}"]
            else:
                raise TypeError(step)
        return result

    lines += [*indent(execution(plan)), "    return FORT_SCOPE_OK;",
              *(["    }();", "    if (fort_entry_result && fort_entry_work_started)",
                 '        return fort_scope_execution_error(fort_context, "borrowed numerical entry failed after execution started; unsafe replay prohibited");',
                 "    return fort_entry_result;"] if root_views else []), "}"]
    team_name = c_name + "_team"
    if config.collective:
        conditions = {}

        def collect_conditions(current):
            for step in current.steps:
                if isinstance(step, ConditionalRegion):
                    conditions[id(step)] = "condition_" + str(len(conditions))
                    collect_conditions(step.then_plan)
                    collect_conditions(step.else_plan)

        collect_conditions(plan)
        state_name = name + "_team_state"
        lines += [f"struct {state_name} {{", "    int status = FORT_SCOPE_OK;",
                  "    bool cpu_active = false;", "    bool work_started = false;"]
        if function.requires_numerical_environment:
            lines += ["    bool environment_supported = true;"]
        lines += [f"    bool {field} = false;" for field in conditions.values()]
        for s in arrays:
            lines += [f"    {cpp_type(s)} *{s.cpp_name} = nullptr;"]
            lines += [f"    std::size_t {dimension_name(s, axis+1)} = 0;" for axis in range(s.rank)]
        lines += [f"    {cpp_type(s)} {s.cpp_name}{{}};" for s in locals_]
        lines += ["};", f'extern "C" int {team_name}({", ".join(signature)}) {{',
                  "#ifdef _OPENMP",
                  f"    if (omp_get_level() != 1 || omp_get_num_threads() != {config.host_threads})",
                  '        return fort_scope_report_error(FORT_SCOPE_BOUNDARY, "qualified numerical companion requires its fixed level-one team");',
                  "#else",
                  '    return fort_scope_report_error(FORT_SCOPE_BOUNDARY, "qualified numerical companion requires OpenMP support");',
                  "#endif",
                  "    fort_scoped::TeamObservation fort_entry_observe(FORT_SCOPE_TEAM_OBSERVE_ENTRY);",
                  f"    {state_name} local_state{{}};", f"    {state_name} *shared = nullptr;",
                  "    #pragma omp single copyprivate(shared)", "    { shared = &local_state; }"]
        if function.requires_numerical_environment:
            lines += ["    if (!fort_runtime::numerical_environment_supported()) {",
                      "        #pragma omp atomic write", "        shared->environment_supported = false;", "    }",
                      "    #pragma omp barrier"]
        lines += ["    #pragma omp master", "    {", "        shared->status = [&]() -> int {"]
        if function.requires_numerical_environment:
            lines += ["            if (!shared->environment_supported)",
                      '                return fort_scope_report_error(FORT_SCOPE_BOUNDARY, "unsupported team numerical environment for real short reduction");']
        lines += [
                  "            if (fort_scope_abi_version() != FORT_SCOPE_ABI_VERSION)",
                  '                return fort_scope_report_error(FORT_SCOPE_ARGUMENT, "incompatible shared runtime ABI");',
                  "            if (fort_mode < 0 || fort_mode > 2)",
                  '                return fort_scope_report_error(FORT_SCOPE_ARGUMENT, "unknown shared execution mode");']
        for s in scalars:
            lines += [f"            if (!fort_scalar_{s.cpp_name})",
                      '                return fort_scope_report_error(FORT_SCOPE_ARGUMENT, "null shared scalar argument");']
        for i, s in enumerate(arrays):
            for other in arrays[:i]:
                if s in writes or other in writes:
                    lines += [f"            if ({s.cpp_name}_handle == {other.cpp_name}_handle)",
                              '                return fort_scope_report_error(FORT_SCOPE_ALIAS, "writable entry arguments alias");']
            lines += ["            {", "                fort_scope_layout layout{};",
                      f"                FORT_SHARED_CHECK(fort_scope_layout_get(fort_context, {s.cpp_name}_handle, &layout));",
                      f"                if (layout.rank != {s.rank} || layout.type != {TYPES[s.dtype][0]} ||",
                      f"                    layout.element_bytes != sizeof({cpp_type(s)}))",
                      '                    return fort_scope_report_error(FORT_SCOPE_ARGUMENT, "entry layout/type mismatch");',
                      f"                shared->{s.cpp_name} = static_cast<{cpp_type(s)} *>(layout.host);"]
            lines += [f"                shared->{dimension_name(s, axis+1)} = layout.extents[{axis}];"
                      for axis in range(s.rank)]
            lines += ["            }"]
        for s in arrays:
            if s.intent == "out":
                lines += [f"            FORT_SHARED_CHECK(fort_scope_forget_definition(fort_context, {s.cpp_name}_handle));"]
        lines += ["            return FORT_SCOPE_OK;", "        }();", "    }", "    #pragma omp barrier",
                  "    if (shared->status == FORT_SCOPE_OK) {"]
        lines += [f"        const {cpp_type(s)} &{s.cpp_name} = *fort_scalar_{s.cpp_name};" for s in scalars]
        for s in arrays:
            lines += [f"        auto *{s.cpp_name} = shared->{s.cpp_name};",
                      f"        {cpp_type(s)} *{s.cpp_name}_device = nullptr;"]
            lines += [f"        const auto {dimension_name(s, axis+1)} = shared->{dimension_name(s, axis+1)};"
                      for axis in range(s.rank)]
        lines += [f"        auto &{s.cpp_name} = shared->{s.cpp_name};" for s in locals_]
        lines += ["        const bool fort_gpu_requested = fort_mode == 1;"]

        def coordinate(body):
            # All members must enter the uniform status guard before master
            # can publish a failure. Otherwise a late reader could skip the
            # operation's barriers after seeing the newly written status.
            return ["#pragma omp barrier", "#pragma omp master", "{", "    shared->status = [&]() -> int {",
                    *indent(body, 2), "        return FORT_SCOPE_OK;", "    }();",
                    "    if (shared->status == FORT_SCOPE_OK && shared->work_started) {",
                    "        int device = -1;",
                    "        shared->status = fort_scope_device_get(fort_context, &device);", "    }",
                    "    if (shared->status != FORT_SCOPE_OK && shared->work_started &&",
                    "        shared->status != FORT_SCOPE_EXECUTION)",
                    "        shared->status = fort_scope_execution_error(fort_context, fort_scope_error());",
                    "}", "#pragma omp barrier"]

        def team_execution(current):
            result = []
            for step in current.steps:
                result += ["if (shared->status == FORT_SCOPE_OK) {"]
                if isinstance(step, ConditionalRegion):
                    condition = "shared->" + conditions[id(step)]
                    preparation = [*prefetch(expression_reads((step.condition,))),
                                   f"{condition} = {render_expression(step.condition)};"]
                    result += indent(coordinate(preparation))
                    # Nested branches need distinct fields: a fast participant
                    # must not replace a choice before its peers observe it.
                    result += ["    if (shared->status == FORT_SCOPE_OK) {", f"        if ({condition}) {{",
                               *indent(team_execution(step.then_plan), 3), "        } else {",
                               *indent(team_execution(step.else_plan), 3), "        }", "    }"]
                elif isinstance(step, (HostBlock, SequentialRegion)):
                    executable = ([render_assignment(a) for a in step.assignments]
                                  if isinstance(step, HostBlock) else sequential_block(step.body, 0, [0]))
                    if set(step.write_symbols) & set(arrays):
                        executable = ["shared->work_started = true;", *executable]
                    result += indent(coordinate(host(step, executable)))
                elif isinstance(step, ParallelRegion):
                    # Only master's batch is populated. Its lifetime spans
                    # the CPU worker barrier; begin owns the materialized
                    # regions, so temporary descriptor arrays can expire.
                    result += ["    fort_scoped::TeamObservation fort_worker_observe(FORT_SCOPE_TEAM_OBSERVE_CPU_WORKER);",
                               f"    fort_scoped::{batch_type}<{capacity}> fort_access(fort_context);"]
                    body = ["shared->cpu_active = false;", *bounds(step),
                            "if (!fort_internal_total) return FORT_SCOPE_OK;",
                            *descriptors(units[step.id])[1:]]
                    gpu_choice = "false" if protected[step.id] else "fort_gpu_requested"
                    body += [f"bool fort_gpu = {gpu_choice};",
                             f"if (fort_mode == 2) FORT_SHARED_CHECK(fort_access.decision({unit_ids[step.id]}ULL, fort_gpu));"]
                    if protected[step.id]:
                        body += [f'if (fort_gpu_requested) offload::decision_trace("{function.name}", "native-scoped-protected-scalar", 0, 1);']
                    body += ["int fort_status = fort_access.begin(fort_gpu);",
                             "if (fort_gpu && (fort_status == FORT_SCOPE_RESOURCE || fort_status == FORT_SCOPE_BOUNDARY)) {",
                             f'    offload::decision_trace("{function.name}", fort_status == FORT_SCOPE_RESOURCE ? "native-scoped-resource" : "native-scoped-boundary", 0, 1);',
                             "    fort_gpu = false; fort_status = fort_access.begin(false);", "}",
                             "FORT_SHARED_CHECK(fort_status);", "fort_scoped::GPUCall fort_gpu_call(fort_context);",
                             "if (fort_gpu) {", "    fort_status = fort_gpu_call.begin();",
                             "    if (fort_status == FORT_SCOPE_RESOURCE) {",
                             "        fort_access.cancel(); fort_gpu = false;",
                             f'        offload::decision_trace("{function.name}", "native-scoped-gpu-setup", 0, 1);',
                             "        fort_status = fort_access.begin(false);", "    }",
                             "    FORT_SHARED_CHECK(fort_status);", "}"]
                    for s in arrays:
                        body += [f"if (fort_gpu) {s.cpp_name}_device = static_cast<{cpp_type(s)} *>(fort_access.device({s.cpp_name}_handle));"]
                    body += ["shared->work_started = true;", "fort_access.executing();", "if (fort_gpu) {",
                        "    {", "        fort_scoped::TeamObservation fort_compute_observe(FORT_SCOPE_TEAM_OBSERVE_COMPUTE);", *indent(generate_launch(
                        step, stream="static_cast<cudaStream_t>(fort_gpu_call.stream)", profile=False,
                        prepared_bounds=True,
                        error_check='if (const auto error = cudaGetLastError(); error != cudaSuccess) return fort_scope_execution_error(fort_context, cudaGetErrorString(error));',
                        launch_record="FORT_SHARED_CHECK(fort_scope_note_launch(fort_context));"), 2), "    }",
                        "    FORT_SHARED_CHECK(fort_access.finish());", "} else {",
                        "    shared->cpu_active = true;", "}"]
                    result += indent(coordinate(body))
                    arguments = worker_arguments + "".join(f", {s.cpp_name}" for s in captures[step.id])
                    result += ["    if (shared->status == FORT_SCOPE_OK && shared->cpu_active) {",
                               "        fort_scoped::TeamObservation fort_compute_observe(FORT_SCOPE_TEAM_OBSERVE_COMPUTE);",
                               f"        cpu_{step.id}({arguments}, offload::thread_id(), offload::team_size());",
                               "        #pragma omp barrier", "    } else {", "        #pragma omp barrier", "    }",
                               "    #pragma omp master", "    {",
                               "        if (shared->status == FORT_SCOPE_OK && shared->cpu_active)",
                               "            shared->status = fort_access.finish();", "    }", "    #pragma omp barrier",
                               "    fort_worker_observe.record_as(shared->cpu_active ? FORT_SCOPE_TEAM_OBSERVE_CPU_WORKER : FORT_SCOPE_TEAM_OBSERVE_GPU_WORKER);"]
                else:
                    raise TypeError(step)
                result += ["}"]
            return result

        lines += indent(team_execution(plan), 2)
        lines += ["    }", "    const int fort_result = shared->status;",
                  "    #pragma omp barrier", "    return fort_result;", "}"]
    plan_name, choose_name = c_name + "_plan", c_name + "_choose"
    planning_payload_arrays = set()

    def planning_inputs(current):
        for step in current.steps:
            if isinstance(step, HostBlock):
                expressions = [a.value for a in step.assignments if a in prep.assignments]
            elif isinstance(step, ConditionalRegion):
                expressions = [step.condition]
                planning_inputs(step.then_plan)
                planning_inputs(step.else_plan)
            elif isinstance(step, ParallelRegion):
                expressions = [e for loop in step.loops for e in
                               (loop.lower, loop.upper) + (() if isinstance(loop.step, int) else (loop.step,))]
                expressions += [e for fp in units[step.id].footprints for box in (*fp.reads, *fp.writes)
                                for e in (*box.lower, *box.upper)]
            else:
                expressions = []
            planning_payload_arrays.update(expression_reads(expressions))

    planning_inputs(plan)
    memory_helpers = {}
    for unit in units.values():
        if unit.compute_model is None:
            continue
        for role in ("native_fortran", "generated_cpu", "gpu"):
            memory_model = unit.compute_model[role].get("memory_cost_model", {})
            if memory_model.get("kind") == "piecewise_bandwidth_v1":
                helper = "fort_compute_memory_" + role + "_" + sha256(str(unit.region.id).encode()).hexdigest()[:12]
                memory_helpers[unit.region.id, role] = helper
                lines += generate_compute_memory(helper, memory_model)
    if memory_helpers:
        lines[0:0] = COMPUTE_MEMORY_INCLUDES
    query_scalars = tuple(s for s in scalars if s in prep.query_scalars)
    planning_parameters = ["fort_context", *[array_argument(s) for s in arrays], *[s.cpp_name for s in query_scalars]]
    planning_signature = [signature[0], *[array_parameter(s) for s in arrays],
                          *[f"const {cpp_type(s)} *fort_scalar_{s.cpp_name}" for s in query_scalars]]
    query_captures = {}
    query_arguments = {}
    query_helpers = []
    if query_available:
        for unit in units.values():
            control_symbols = set()
            for loop in unit.region.loops:
                for expression in (loop.lower, loop.upper) + (() if isinstance(loop.step, int) else (loop.step,)):
                    control_symbols.update(referenced_symbols(expression))
            for footprint in unit.footprints:
                for box in (*footprint.uploads, *footprint.downloads):
                    for expression in (*box.lower, *box.upper):
                        control_symbols.update(referenced_symbols(expression))
            query_captures[unit.region.id] = tuple(s for s in locals_ if s in control_symbols)
            query_signature = ("fort_scope_t fort_context, int &fort_query_status, "
                               + "".join(array_parameter(s) + ", " for s in arrays)) + ", ".join(
                cpp_declaration(a, array_views=root_views) if a.symbol.rank else f"const volatile {cpp_type(a.symbol)} &{a.name}"
                for a in abi)
            query_signature += "".join(f", const {cpp_type(s)} &{s.cpp_name}" for s in query_captures[unit.region.id])
            query_arguments[unit.region.id] = ("fort_context, fort_query_status, "
                                               + "".join(array_argument(s) + ", " for s in arrays)) + worker_arguments + "".join(
                f", {s.cpp_name}" for s in query_captures[unit.region.id])
            helper = _metadata(function, OffloadAnalysis(True, None, (replace(unit, index=0),)),
                               query_signature, f"plan_unit_{unit.region.id}",
                               lambda expression: _query_expression(expression, {}),
                               host_pointer=(lambda symbol: symbol.cpp_name + ".root") if root_views else None)
            axis = 0
            for line in helper:
                query_helpers.append(line)
                if line == "        if (active) {":
                    loop = unit.region.loops[axis]
                    expressions = (loop.lower, loop.upper) + (() if isinstance(loop.step, int) else (loop.step,))
                    query_helpers += indent(prefetch(expression_reads(expressions), planning=True, query_helper=True), 3)
                    axis += 1
            if axis != len(unit.region.loops):
                raise CompilationError("checked scoped metadata did not preserve per-axis bound guards")
        # Helpers must be defined before the public query, and never execute
        # numerical assignments. INTEGER/LOGICAL values form a checked slice.
        lines += query_helpers
    native_preflight_name = c_name + "_native_preflight_v1"
    native_preflight = generate_native_preflight(function, plan, tuple(units.values()), native_preflight_name,
        source_compute=source_compute, collective=config.collective,
        planning_available=planning_available, profile_available=costs is not None, protected=protected)
    lines += native_preflight.cpp
    lines += [f'extern "C" int {plan_name}({", ".join(planning_signature)}) {{']
    if function.requires_numerical_environment:
        lines += ["    if (!fort_runtime::numerical_environment_supported())",
                  '        return fort_scope_report_error(FORT_SCOPE_BOUNDARY, "unsupported planning numerical environment for real short reduction");']
    if not query_available:
        lines += [f"    return fort_scope_report_error(FORT_SCOPE_BOUNDARY, {json.dumps(query_reason)});"]
    else:
        query_setup = []
        for symbol in scalars:
            if symbol in query_scalars:
                query_setup += [f"    if (!fort_scalar_{symbol.cpp_name})",
                                '        return fort_scope_report_error(FORT_SCOPE_ARGUMENT, "null planning scalar argument");',
                                f"    const volatile {cpp_type(symbol)} &{symbol.cpp_name} = *fort_scalar_{symbol.cpp_name};"]
            else:
                # Internal metadata helpers share the worker's ABI. Their
                # numerical scalar arguments are deliberately not public query
                # inputs and the checked control slice never reads them.
                query_setup += [f"    const {cpp_type(symbol)} {symbol.cpp_name}{{}};"]
        query_setup += layout_setup
        query_setup += alias_setup
        lines += query_setup
        if config.collective:
            lines += ["    FORT_SHARED_CHECK(fort_scope_plan_team_entry_v1(fort_context));"]
        lines += [f"    FORT_SHARED_CHECK(fort_scope_plan_host_current(fort_context, {s.cpp_name}_handle));"
                  for s in arrays if s in planning_payload_arrays]
        lines += ["    offload::Data d; int fort_query_status = FORT_SCOPE_OK;", *indent([f"{cpp_type(s)} {s.cpp_name};" for s in locals_
                                                if s.dtype in {ScalarType.INTEGER, ScalarType.LOGICAL}])]
        for symbol in arrays:
            if symbol.intent == "out":
                if root_views:
                    lines += [f"    FORT_SHARED_CHECK(fort_scoped::forget_view(fort_context, {symbol.cpp_name}_view, true));"]
                else:
                    lines += ["    {", f"        fort_scoped::{batch_type}<{capacity}> fort_forget(fort_context);",
                          "        fort_scope_access fort_definition{};",
                          f"        FORT_SHARED_CHECK(fort_forget.add({array_argument(symbol)}, fort_definition));",
                          "        FORT_SHARED_CHECK(fort_forget.record(FORT_SCOPE_PLAN_FORGET, 0, 0, 0, false));", "    }"]

        def project(current):
            result = []
            for step in current.steps:
                if isinstance(step, HostBlock):
                    touched = set(step.read_symbols) | set(step.write_symbols)
                    record = ["{", f"    fort_scoped::{batch_type}<{capacity}> fort_host(fort_context);"]
                    for symbol in arrays:
                        if symbol in touched:
                            flags = (1 if symbol in step.read_symbols else 0) | (2 if symbol in step.write_symbols else 0)
                            record += [f"    fort_scope_access fort_effect_{symbol.id}{{}}; fort_effect_{symbol.id}.flags = {flags};",
                                       f"    FORT_SHARED_CHECK(fort_host.add({array_argument(symbol)}, fort_effect_{symbol.id}));"]
                    result += ["if (d.valid) {", *indent(record + ["    FORT_SHARED_CHECK(fort_host.record(FORT_SCOPE_PLAN_NATIVE, 0, 0, 0, false));", "}"]), "}"]
                    for assignment in step.assignments:
                        if assignment in prep.assignments:
                            result += [f"if (d.valid) {assignment.target.symbol.cpp_name} = {_query_expression(assignment.value, {})};"]
                elif isinstance(step, ConditionalRegion):
                    result += ["if (d.valid) {", *indent(prefetch(expression_reads((step.condition,)), planning=True)),
                               f"    const bool branch = {_query_expression(step.condition, {})};",
                               "    if (d.valid && branch) {", *indent(project(step.then_plan), 2),
                               "    } else if (d.valid) {", *indent(project(step.else_plan), 2), "    }", "}"]
                elif isinstance(step, ParallelRegion):
                    known_work = (units[step.id].work_per_iteration is not None
                                  and not units[step.id].work_is_upper_bound
                                  and not function.requires_numerical_environment)
                    flops = "unit.units[0].flops" if known_work else "0.0"
                    memory_bytes = "unit.units[0].memory_bytes" if known_work else "0.0"
                    model = units[step.id].compute_model if not function.requires_numerical_environment else None
                    record = [f"        FORT_SHARED_CHECK(fort_access.record(FORT_SCOPE_PLAN_WORKER, {unit_ids[step.id]}ULL, {flops}, {memory_bytes}, {'true' if known_work and not protected[step.id] else 'false'}, unit.units[0].cpu_numerical_seconds, unit.units[0].gpu_numerical_seconds));"]
                    if source_compute:
                        record = ["        fort_scope_compute_costs_v1 fort_compute{};",
                                  "        fort_compute.version = FORT_SCOPE_COMPUTE_ABI_VERSION;"]
                        if model is not None:
                            lower, upper = model["item_range"]
                            # Public queries can be called without this entry's
                            # chooser. Validate cost participation here too;
                            # mismatches retain effects with unknown costs.
                            schedule_guard = " && scoped_compute_placement_compatible()"
                            if model.get("runtime_schedule") is not None:
                                schedule_guard += " && scoped_compute_schedule_compatible()"
                            if model.get("require_unit_stride") and root_views:
                                requirement = units[step.id].memory_access_requirement
                                schedule_guard += "".join(f" && {access.symbol.cpp_name}.strides[0] == 1ULL"
                                                         for access in requirement.accesses)
                            record += [f"        if (unit.units[0].iterations >= {lower}ULL && unit.units[0].iterations <= {upper}ULL{schedule_guard}) {{"]
                            if any((step.id, role) in memory_helpers for role in ("native_fortran", "generated_cpu", "gpu")):
                                from compiler.emission.cuda.working_set import working_set_lines
                                exact = (not protected_footprints[step.id] and all(
                                    fp.exact and not fp.full_upload and not fp.full_write
                                    for fp in units[step.id].footprints))
                                record += indent(working_set_lines([s.cpp_name + "_handle" for s in arrays], exact=exact), 3)
                            for role, bit in (("native_fortran", "FORT_SCOPE_COMPUTE_NATIVE_FORTRAN"),
                                              ("generated_cpu", "FORT_SCOPE_COMPUTE_GENERATED_CPU"),
                                              ("gpu", "FORT_SCOPE_COMPUTE_GPU")):
                                coefficients = model[role]
                                arithmetic = units[step.id].compute_arithmetic_operations_per_iteration
                                seconds = coefficients.get("compute_seconds_per_item",
                                    arithmetic * coefficients["arithmetic_seconds_per_operation"] + coefficients["intrinsic_seconds_per_item"])
                                helper = memory_helpers.get((step.id, role))
                                if helper is not None:
                                    record += ["            {", "                double fort_memory_seconds = 0.0;",
                                               f"                if (fort_working_set_available && {helper}(unit.units[0].memory_bytes, fort_working_set_bytes, fort_memory_seconds)) {{",
                                               f"                    fort_compute.{role}_seconds = std::max(static_cast<double>(unit.units[0].iterations) * {seconds!r}, fort_memory_seconds) + {coefficients['fixed_seconds']!r};",
                                               f"                    fort_compute.known_backends |= {bit};", "                }", "            }"]
                                else:
                                    record += [f"            fort_compute.{role}_seconds = std::max(static_cast<double>(unit.units[0].iterations) * {seconds!r}, unit.units[0].memory_bytes * {coefficients['memory_seconds_per_byte']!r}) + {coefficients['fixed_seconds']!r};",
                                               f"            fort_compute.known_backends |= {bit};"]
                            record += ["        }"]
                        record += [f"        FORT_SHARED_CHECK(fort_access.record_compute(FORT_SCOPE_PLAN_WORKER, {unit_ids[step.id]}ULL, {flops}, {memory_bytes}, {'true' if known_work and not protected[step.id] else 'false'}, fort_compute));"]
                    result += ["if (d.valid) {",
                               f"    auto unit = plan_unit_{step.id}({query_arguments[step.id]});",
                               "    FORT_SHARED_CHECK(fort_query_status);", "    d.valid = unit.valid;", "    if (d.valid && unit.units[0].iterations) {",
                               *indent(descriptors(units[step.id], planning=True), 2),
                               *record,
                               "    }", "}"]
                else:
                    raise TypeError(step)
            return result

        lines += indent(project(plan))
        lines += ['    if (!d.valid) return fort_scope_report_error(FORT_SCOPE_BOUNDARY, "checked planning preparation is invalid or overflows");',
                  "    return FORT_SCOPE_OK;"]
    lines += ["}"]
    cost_lines = ["fort_scope_plan_costs costs{}; costs.version = FORT_SCOPE_PLANNING_ABI_VERSION;"]
    if costs is not None:
        rates = config.profile["rates"]
        fields = {"cpu_flops": rates["cpu_flops_per_second"], "cpu_bandwidth": rates["cpu_memory_bytes_per_second"],
                  "gpu_flops": rates["gpu_flops_per_second"], "gpu_bandwidth": rates["gpu_memory_bytes_per_second"],
                  "h2d_latency": rates["h2d_pageable"]["latency_seconds"],
                  "h2d_bandwidth": rates["h2d_pageable"]["bandwidth_bytes_per_second"],
                  "d2h_latency": rates["d2h_pageable"]["latency_seconds"],
                  "d2h_bandwidth": rates["d2h_pageable"]["bandwidth_bytes_per_second"], **costs}
        if team_costs is not None:
            fields.update(cpu_flops=team_costs["cpu_flops"], cpu_bandwidth=team_costs["cpu_bandwidth"])
        cost_lines += [f"costs.valid = {int(planning_available)};", f"costs.max_allocation_bytes = {config.profile['scoped']['max_allocation_bytes']}ULL;"]
        cost_lines += [f"costs.{field} = {float(value)!r};" for field, value in fields.items()]
    choose_team_guard = (["    int fort_team_ready=0;",
                         "    FORT_SHARED_CHECK(fort_scope_team_costs_ready_v1(fort_context, &fort_team_ready));",
                         "    if (!fort_team_ready) costs.valid=0;"] if config.collective else [])
    lines += [f'extern "C" int {choose_name}(fort_scope_t fort_context, fort_scope_plan_decision *decision) {{']
    if function.requires_numerical_environment:
        lines += ["    if (!fort_runtime::numerical_environment_supported())",
                  '        return fort_scope_report_error(FORT_SCOPE_BOUNDARY, "unsupported selection numerical environment for real short reduction");']
    lines += ['    if (!decision) return fort_scope_report_error(FORT_SCOPE_ARGUMENT, "null planning decision");',
              *indent(cost_lines), *choose_team_guard, f"    static const auto profile = {profile_expression(config.profile if costs is not None else None)};",
              "    int fort_scope_device = -1, fort_current_device = -1;",
              "    FORT_SHARED_CHECK(fort_scope_device_get(fort_context, &fort_scope_device));",
              "    fort_scope_plan_decision preview{};",
              "    FORT_SHARED_CHECK(fort_scope_plan_select(fort_context, &costs, -1, &preview));",
              f"    const bool compatible = !preview.available || (scoped_host_compatible(profile) && scoped_compute_placement_compatible() && (!preview.gpu_units || (cudaGetDevice(&fort_current_device) == cudaSuccess && fort_current_device == fort_scope_device && offload::compatible(profile, {config.host_threads}, {_precision(function)}))));",
              "    FORT_SHARED_CHECK(fort_scope_plan_select(fort_context, &costs, compatible ? 1 : 0, decision));",
              f'    offload::decision_trace("{function.name}", decision->gpu_units ? "scoped-scheduled" : "native-scoped-scheduled", decision->gpu_units, decision->cpu_units);',
              "    return FORT_SCOPE_OK;", "}", "}", "#undef FORT_SHARED_CHECK", ""]
    parameters = ["fort_context", "fort_mode", *[array_argument(s) for s in arrays], *[s.cpp_name for s in scalars]]
    public = "  public :: run, run_team, plan, choose, configure" if config.collective else "  public :: run, plan, choose, configure"
    if native_preflight.available:
        public += ", native_preflight"
    view_import = ", " + view_type if root_views else ""
    fortran = [f"module {name}", "  use iso_c_binding", "  use fort_scoped_memory, only: fort_scope_plan_decision" + view_import, "  implicit none", "  private", public, "  interface"]
    fortran += _fortran_list("function run(", parameters, f") bind(C, name='{c_name}') result(fort_status)", 4)
    fortran += ["      import :: c_int, c_int64_t, c_double, c_float, c_bool" + view_import, "      integer(c_int) :: fort_status",
                "      integer(c_int64_t), value :: fort_context", "      integer(c_int), value :: fort_mode"]
    fortran += [(f"      type({view_type}), intent(in) :: {s.cpp_name}_view" if root_views else
                 f"      integer(c_int64_t), value :: {s.cpp_name}_handle") for s in arrays]
    fortran += [f"      {TYPES[s.dtype][1]}, intent(in) :: {s.cpp_name}" for s in scalars]
    fortran += ["    end function"]
    if config.collective:
        fortran += _fortran_list("function run_team(", parameters,
                                 f") bind(C, name='{team_name}') result(fort_status)", 4)
        fortran += ["      import :: c_int, c_int64_t, c_double, c_float, c_bool",
                    "      integer(c_int) :: fort_status", "      integer(c_int64_t), value :: fort_context",
                    "      integer(c_int), value :: fort_mode"]
        fortran += [f"      integer(c_int64_t), value :: {s.cpp_name}_handle" for s in arrays]
        fortran += [f"      {TYPES[s.dtype][1]}, intent(in) :: {s.cpp_name}" for s in scalars]
        fortran += ["    end function"]
    fortran += _fortran_list("function plan(", planning_parameters, f") bind(C, name='{plan_name}') result(fort_status)", 4)
    fortran += ["      import :: c_int, c_int64_t, c_double, c_float, c_bool" + view_import, "      integer(c_int) :: fort_status",
                "      integer(c_int64_t), value :: fort_context"]
    fortran += [(f"      type({view_type}), intent(in) :: {s.cpp_name}_view" if root_views else
                 f"      integer(c_int64_t), value :: {s.cpp_name}_handle") for s in arrays]
    fortran += [f"      {TYPES[s.dtype][1]}, intent(in) :: {s.cpp_name}" for s in query_scalars]
    fortran += ["    end function", f"    function choose(fort_context, decision) bind(C, name='{choose_name}') result(fort_status)",
                "      import :: c_int, c_int64_t, fort_scope_plan_decision", "      integer(c_int) :: fort_status",
                "      integer(c_int64_t), value :: fort_context", "      type(fort_scope_plan_decision), intent(out) :: decision",
                "    end function", f"    function configure(fort_context) bind(C, name='{configure_name}') result(fort_status)",
                "      import :: c_int, c_int64_t", "      integer(c_int) :: fort_status",
                "      integer(c_int64_t), value :: fort_context", "    end function",
                *native_preflight.fortran, "  end interface", f"end module {name}", ""]
    report = {
        "schema_version": 1, "abi_version": 1, "entry_abi_version": ENTRY_ABI_VERSION,
        "entry": c_name, "fortran_module": name, "fortran_procedure": "run",
        "cuda_source": "shared_entry.cu", "fortran_source": "shared_interface.f90",
        "array_parameters": [{"name": s.name, "rank": s.rank, "dtype": s.dtype.value,
                              "written": s in writes} for s in arrays],
        "scalar_parameters": [{"name": s.name, "dtype": s.dtype.value, "passing": "reference"} for s in scalars],
        "argument_order": ["context", "mode", *[s.name for s in arrays], *[s.name for s in scalars]],
        "modes": {"native": 0, "gpu": 1, "automatic": 2},
        "numerical_environment": {
            "required": function.requires_numerical_environment,
            "check": "fort_scope_numerical_environment_supported",
            "rounding": "round to nearest", "exceptions": "host traps disabled",
            "caller_contract": "floating-point exception flags are not observed",
            "check_position": "before entry work and planning; every run, all collective participants",
            "unsupported": "pre-work boundary; original-source caller executes its coherent native operation",
            "generated_cpu_fallback": "does not replace the original source for unsupported environments",
        },
        "transfer_configuration": {"abi_version": 1, "requested": config.scope_transfers,
                                   "selected": "pinned" if config.scope_transfers == "pinned" else "direct",
                                   "reason": transfer_reason, "available_modes": ["direct", "pinned"],
                                   "entry": configure_name, "fortran_procedure": "configure",
                                   "argument_order": ["context"],
                                   "position": "once per owning context before registration and planning",
                                   "execution": "synchronous", "pinned_budget_bytes": 64 * 1024 * 1024,
                                   "budget_scope": "process-wide including cached staging",
                                   "resource_fallback": "direct before this copy is enqueued; numerical work is never replayed",
                                   "runtime_stats": "fort_scope_transfer_stats_get_v1",
                                   "placement_estimate_available": bool(planning_available and costs is not None),
                                   "placement_estimate_reason": planning_reason or profile_reason},
        "batch_execution": {"abi_version": 1, "runtime_api": "fort_scope_batch_execute_v1",
                            "runtime_report": "fort_scope_batch_report_get_v1",
                            "eligible": batch_eligible, "enabled": batch_enabled,
                            "reason": ("collective batch coordination is unavailable" if config.collective else
                                       batch_analysis.reason if not batch_analysis.available else
                                       transfer_profile_reason if transfer_costs is None else None),
                            "entry": batch_name if batch_eligible else None,
                            "window_entry": window_name if batch_eligible else None,
                            "window_argument_order": ["window", "first_unit", "stop_unit", "axis", "launches",
                                                      *[s.name for s in arrays], *[s.name for s in scalars]],
                            "candidates": [{"first_unit": interval.start, "stop_unit": interval.stop,
                                            "eligible": slab is not None, "reason": reason,
                                            "slab": slab.to_dict() if slab else None}
                                           for interval, slab, reason in batch_candidates],
                            "limits": {"adjacent_units": 4, "complete_legal_group": True},
                            "coordinates": "original logical coordinates and full-array pitches",
                            "kernels": "reuse existing numerical kernels; no partition-specific copies",
                            "publication": "retain device results until original native consumers or owner close",
                            "complete_owner_estimate": "unavailable after batch execution until full repricing",
                            "fallback": "ordinary synchronous execution before batch work; no replay after start",
                            "statistics": ["selected_transfers", "chunk_iterations", "slot_bytes", "prefix_upload_bytes",
                                           "batches", "actual_upload_bytes", "actual_download_bytes", "actual_launches",
                                           "preparation_operations", "execution_seconds", "terminal_delta_seconds"]},
        "automatic_estimate_available": bool(planning_available and costs is not None),
        "automatic_reason": planning_reason or profile_reason,
        "compute_estimates": {"abi_version": 1 if source_compute else 0,
                              "runtime_api": "fort_scope_plan_add_compute_costs_v3" if source_compute else "fort_scope_plan_add_costs_v2",
                              "native_participation": config.native_participation,
                              "cpu_affinity": affinity,
                              "runtime_item_range_enforced": source_compute,
                              "memory_working_set": "checked per-resource union of exact physical sections; repeated touched root aliases unavailable",
                              "memory_range_enforced": bool(memory_helpers),
                              "runtime_schedule_checked": runtime_schedule,
                              "decision_timing_coverage": "runtime query recording, validation and selection; emitted footprint preparation is outside those timers",
                              "native_backend": "original Fortran" if source_compute else "generated C++ worker"},
        "native_preflight": native_preflight.public(native_preflight_name),
        "automatic_scope_available": planning_available, "host_threads": config.host_threads,
        "planning": {"abi_version": 1, "available": planning_available, "reason": planning_reason,
                     "supported_endpoint_modes": ["complete", "continue"],
                     "continuation_abi_version": 2, "runtime_report": "fort_scope_plan_report_v2",
                     "query_available": query_available, "query_reason": query_reason,
                     "entry": plan_name, "fortran_procedure": "plan", "selector": choose_name,
                     "fortran_selector": "choose", "argument_order": ["context", *[s.name for s in arrays], *[s.name for s in query_scalars]],
                     "scalar_inputs": [s.name for s in scalars if s in prep.query_scalars],
                     "payload_arrays": [s.name for s in arrays if s in planning_payload_arrays],
                     "layout_arrays": [s.name for s in arrays],
                     "source_effects": False, "preparation": "checked INTEGER/LOGICAL control slice",
                     "units": [{"region": r.id, "id": unit_ids[r.id], "work_per_iteration": units[r.id].work_per_iteration,
                                "workload_class": units[r.id].workload_class,
                                "workload_features": units[r.id].workload_features.to_dict(),
                                "compute_model": units[r.id].compute_model,
                                "work_estimate_reason": units[r.id].work_estimate_reason,
                                "physical_working_set": {
                                    "source_sections_exact": all(fp.exact and not fp.full_upload and not fp.full_write
                                                                 for fp in units[r.id].footprints),
                                    "range_check_required": any((r.id, role) in memory_helpers
                                                                for role in ("native_fortran", "generated_cpu", "gpu")),
                                    "union": "checked bounded read/write section union per touched canonical root",
                                    "runtime_requirements": ["distinct touched canonical roots", "validated injective views",
                                                             "per-backend calibrated working-set range", "checked byte arithmetic"],
                                    "unavailable": "compute estimate only; ordered effects and forced execution retained"},
                                "compute_arithmetic_operations_per_iteration": units[r.id].compute_arithmetic_operations_per_iteration,
                                "compute_runtime_divisions_per_iteration": units[r.id].compute_runtime_divisions_per_iteration,
                                "compute_arithmetic_estimate_reason": units[r.id].compute_arithmetic_estimate_reason,
                                "compute_dependencies": (units[r.id].compute_dependencies.to_dict()
                                                         if units[r.id].compute_dependencies is not None else None),
                                "memory_access_requirement": (units[r.id].memory_access_requirement.to_dict()
                                                              if units[r.id].memory_access_requirement is not None else None),
                                **({"intrinsic_work_per_iteration": dict(units[r.id].intrinsic_work_per_iteration),
                                    "cpu_numerical_seconds_per_iteration": units[r.id].cpu_numerical_seconds_per_iteration,
                                    "gpu_numerical_seconds_per_iteration": units[r.id].gpu_numerical_seconds_per_iteration}
                                   if units[r.id].intrinsic_work_per_iteration else {})}
                               for r in plan.regions],
                     "profile_available": costs is not None, "profile_reason": profile_reason,
                     "runtime_id": runtime_id},
        "definition_changes": [s.name for s in arrays if s.intent == "out"],
        "resource_failure": "continue native at current worker before execution; never replay completed work",
        "transfer_volume": "missing physical read/preservation sections from shared runtime state",
        "parallel_regions": len(plan.regions), "source": function.source,
        "region_execution": [{"region": r.id, "gpu_available": not protected[r.id],
                              "protected_scalars": [s.name for s in sorted(protected[r.id], key=lambda s: s.id)],
                              "native_footprints": "whole-resource" if protected_footprints[r.id] else "physical-sections",
                              "reason": "CUDA value capture would evaluate a protected scalar" if protected[r.id] else None}
                             for r in plan.regions],
    }
    if batch_enabled:
        report["transfer_configuration"].update(selected="runtime", reason=None,
            available_modes=["direct", "pinned", "pipelined"], execution="synchronous return with two-stream batch overlap")
    if root_views:
        report["borrowed_views"] = {"abi_version": root_view_abi, "validation": view_get,
                                   "layout": "full-root byte pitches; formal logical dimensions and coordinates",
                                   "logical_bounds": "prepared one-based numerical ABI; original indices remain in source-proven normalized scalars",
                                   "definitions": "exact partial view discard at original entry",
                                   "aliases": "disjoint writable views; exact bounded read unions"}
        for parameter in report["array_parameters"]:
            parameter["passing"] = f"root_view_v{root_view_abi}"
        if root_view_abi == 2:
            report["borrowed_views"]["rank_reduction"] = "retained logical axes map to original full-layout pitches; fixed axes select one coordinate"
        if batch_requested:
            report["batch_execution"]["reason"] = "borrowed-view subchain batching is unavailable"
    if config.collective:
        report["team"] = {
            "available": True, "entry_abi_version": 1, "entry": team_name,
            "fortran_procedure": "run_team", "argument_order": report["argument_order"],
            "participation": "qualified_full_team", "expected_omp_level": 1,
            "host_threads": config.host_threads, "coordinator": "master",
            "captures": "same context, mode, whole shared buffers and immutable scalar bindings on all participants",
            "caller_preflight": "allocation and exact descriptor agreement before numerical association; checked controls and complete ordered definition validation before work",
            "cpu_workers": "all existing team members with tid/team; no nested team",
            "locals": "shared coordinator preparation; borrowed scalar references",
            "status": "uniform after matching barriers; no replay after execution starts",
            "planning": "plan and choose are coordinator-only",
            "automatic_estimate_available": bool(planning_available and team_costs is not None and costs is not None),
            "automatic_reason": planning_reason or (profile_reason if team_costs is None or costs is None else None),
            "calibration_protocol_id": "0x4654434f4c4c0001",
            "calibration": "offline original persistent team and emitted coordination; serial rates are not substituted",
        }
    return ScopedEmission("\n".join(lines), "\n".join(fortran), report)
