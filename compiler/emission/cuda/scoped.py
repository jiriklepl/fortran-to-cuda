"""Numerical entries borrowing common buffers through the versioned public ABI."""

from dataclasses import dataclass
from hashlib import sha256

from compiler.analysis.sections import must_overlap
from compiler.emission.c.declarations import cpp_declaration
from compiler.emission.common.abi import abi_arguments, dimension_name
from compiler.emission.common.c_family import cpp_type, indent, render_assignment, render_expression
from compiler.emission.common.loops import _loop_snapshot, sequential_block
from compiler.emission.common.schedules import checked_product, region_schedule, tile_counts
from compiler.emission.common.symbols import host_symbols, region_symbols
from compiler.emission.cuda.graphs import _kernel_arguments, generate_graph_group, graph_group_eligible
from compiler.emission.cuda.kernels import generate_group_kernel, generate_kernel, generate_launch
from compiler.emission.cuda.offload import _cpu_worker
from compiler.emission.cuda.structured import _query_expression
from compiler.emission.fortran.formatting import _fortran_list
from compiler.ir import (
    ArrayAccess,
    CompilationError,
    ConditionalRegion,
    HostBlock,
    IntrinsicCall,
    Literal,
    ParallelRegion,
    Reference,
    ScalarType,
    SequentialRegion,
    referenced_symbols,
    walk_expr,
)
from compiler.numerical_contract import numerical_build_contract, numerical_source_prologue
from compiler.offload.analysis import (
    Unit,
    _protected_scalar_inputs,
    _unit_footprints,
)
from compiler.offload.codegen import query_expression


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

ENTRY_ABI_VERSION = 3


def generate_scoped(function, plan, config, common_header, *, root_views=False, root_view_abi=1):
    if root_view_abi not in {1, 2}:
        raise CompilationError("unsupported borrowed numerical view ABI")
    view_type = f"fort_scope_view_v{root_view_abi}"
    view_layout_type = f"fort_scope_view_layout_v{root_view_abi}"
    view_get = f"fort_scope_view_get_v{root_view_abi}"
    arrays = tuple(s for s in function.parameters if s.rank)
    scalars = tuple(s for s in function.parameters if not s.rank)
    if any(s.intent != "in" for s in scalars):
        raise CompilationError("shared numerical entries require read-only scalar parameters")
    identity = f"entry-abi:{ENTRY_ABI_VERSION}:{function.module.lower()}::{function.name.lower()}:{plan!r}"
    identity += ":dependency-task-graph-v1"
    if function.requires_numerical_environment:
        identity += ":numerical-environment:nearest-nontrapping-v1"
    digest = sha256(identity.encode()).hexdigest()[:12]
    if root_views:
        digest = sha256((digest + f":root-view-v{root_view_abi}").encode()).hexdigest()[:12]
    name = "fort_shared_" + digest
    c_name = "cpp_" + name
    from compiler.scopes.provenance import Provenance
    provenance = Provenance()
    procedure_id = provenance.add("procedure", procedure=f"{function.module.lower()}::{function.name.lower()}",
                                  source=function.source, numerical_identity=sha256(identity.encode()).hexdigest())
    implementation_ids = {(region.id, backend): provenance.add("implementation", procedure_id=procedure_id,
        procedure=f"{function.module.lower()}::{function.name.lower()}", source=function.source,
        region=region.id, backend=backend, numerical_identity=sha256(identity.encode()).hexdigest())
        for region in plan.regions for backend in ("cuda", "generated_cpu")}

    def trace_worker(region):
        return (f'fort_scoped::TracePosition fort_worker_trace(fort_context, "{procedure_id}", '
                f'fort_gpu ? "{implementation_ids[region.id, "cuda"]}" : "{implementation_ids[region.id, "generated_cpu"]}");')

    def trace_cpu(region):
        return f'fort_worker_trace.implementation("{implementation_ids[region.id, "generated_cpu"]}");'
    configure_name = c_name + "_configure"
    transfer_modes = {"direct": "FORT_SCOPE_TRANSFERS_DIRECT", "pinned": "FORT_SCOPE_TRANSFERS_PINNED",
                      "pipelined": "FORT_SCOPE_TRANSFERS_PIPELINED", "auto": "FORT_SCOPE_TRANSFERS_AUTO",
                      "coalesced": "FORT_SCOPE_TRANSFERS_COALESCED"}
    transfer_reason = {"auto": "transfer_estimates_unavailable", "pipelined": "pipelined_not_available"}.get(config.scope_transfers)
    scalar_signature = [(f"const {cpp_type(s)} fort_value_{s.cpp_name}" if s.by_value else
                         f"const {cpp_type(s)} *fort_scalar_{s.cpp_name}") for s in scalars]
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
    lines.insert(0, "#include <type_traits>")
    if root_views:
        lines.insert(5, '#include "view_entry.hpp"')
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
    lines += [f'extern "C" int {configure_name}(fort_scope_t fort_context) {{',
              f"    return fort_scope_set_transfers(fort_context, {transfer_modes[config.scope_transfers]});", "}"]
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

    def launch_groups(current):
        """Runs of adjacent regions that one launch can execute concurrently.

        Members write no shared scalar, and where one writes an array that
        another touches, their sections are not known to meet: the two ghost
        layers of opposite faces, say. Sections that meet whatever the bounds
        are keep their regions apart. Anything short of that is left to the
        runtime, which rechecks the actual sections; the entry keeps its
        one-kernel-per-region path for a launch the runtime refuses.
        """
        groups, run = [], []

        def arrays_of(region):
            footprints = units[region.id].footprints
            return ({fp.symbol for fp in footprints if fp.reads or fp.full_read},
                    {fp.symbol for fp in footprints if fp.writes or fp.full_write})

        def groupable(region):
            schedule = region_schedule(region)
            return (not schedule.tile_sizes and graph_group_eligible((region,), units, protected)
                    and not any(not symbol.rank and symbol not in region.private_symbols
                                for symbol in region.write_symbols))

        def independent(first, second):
            if region_schedule(first).cuda_threads != region_schedule(second).cuda_threads:
                return False
            first_reads, first_writes = arrays_of(first)
            second_reads, second_writes = arrays_of(second)
            shared = (first_writes & (second_reads | second_writes)) | (second_writes & first_reads)
            if not shared:
                return True
            # Both members run, so each of their loops has an iteration.
            running = [(loop.lower, loop.upper) for region in (first, second) for loop in region.loops
                       if loop.step == 1]
            for mine in units[first.id].footprints:
                for theirs in units[second.id].footprints:
                    if mine.symbol is not theirs.symbol or mine.symbol not in shared:
                        continue
                    # An access under a condition has its section as an upper bound, which serves here:
                    # sections that do not meet as bounds do not meet as accessed either.
                    if mine.full_read or mine.full_write or theirs.full_read or theirs.full_write:
                        return False
                    pairs = [(written, other) for written in mine.writes for other in (*theirs.reads, *theirs.writes)]
                    pairs += [(read, written) for read in mine.reads for written in theirs.writes]
                    if any(must_overlap(one, other, running) for one, other in pairs):
                        return False
            return True

        def close():
            if len(run) > 1:
                groups.append(tuple(run))
            run.clear()

        for step in current.steps:
            if isinstance(step, ConditionalRegion):
                close()
                groups.extend(launch_groups(step.then_plan))
                groups.extend(launch_groups(step.else_plan))
                continue
            member = isinstance(step, ParallelRegion) and groupable(step)
            # One launch passes every member's arguments, at most eight bytes each; CUDA takes 4096 bytes.
            if (member and all(independent(earlier, step) for earlier in run)
                    and sum(len(_kernel_arguments(region)) for region in (*run, step)) <= 384):
                run.append(step)
                continue
            close()
            if member:
                run.append(step)
        close()
        return groups

    merged_launches = [] if root_views else launch_groups(plan)
    for group in merged_launches:
        lines.extend(generate_group_kernel(group, array_views=root_views))
    lines += [f'extern "C" int {c_name}({", ".join(signature)}) {{',
              f'    fort_scoped::TracePosition fort_entry_trace(fort_context, "{procedure_id}", "");',
              *(["    bool fort_entry_work_started=false;",
                 "    const int fort_entry_result=[&]() -> int {"]),
              "    if (fort_scope_abi_version() != FORT_SCOPE_ABI_VERSION)",
              '        return fort_scope_report_error(FORT_SCOPE_ARGUMENT, "incompatible shared runtime ABI");',
              "    if (fort_mode < 0 || fort_mode > 1)",
              '        return fort_scope_report_error(FORT_SCOPE_ARGUMENT, "unknown shared execution mode");']
    if function.requires_numerical_environment:
        lines += ["    if (!fort_runtime::numerical_environment_supported())",
                  '        return fort_scope_report_error(FORT_SCOPE_BOUNDARY, "unsupported numerical environment for real short reduction");']
    setup = []
    for s in scalars:
        if s.by_value:
            setup += [f"    const {cpp_type(s)} &{s.cpp_name} = fort_value_{s.cpp_name};"]
            continue
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
    # A region may be independent only for some values of the entry's scalars. For others the entry does
    # nothing and says so; its caller still has the original statements.
    assumptions = [expression for region in plan.regions for expression in region.assumptions]
    lines += [f"    if (!{render_expression(expression)}) return FORT_SCOPE_DECLINED;" for expression in assumptions]
    lines += [f"    {cpp_type(s)} {s.cpp_name};" for s in host_symbols(function, plan)]
    for s in arrays:
        if s.intent == "out":
            forget = (f"fort_scoped::forget_view(fort_context, {s.cpp_name}_view)" if root_views else
                      f"fort_scope_forget_definition(fort_context, {s.cpp_name}_handle)")
            lines += [f"    FORT_SHARED_CHECK({forget});"]
    lines += ["    const bool fort_gpu_requested = fort_mode == 1;"]

    def prefetch(symbols):
        ordered = [s for s in arrays if s in symbols]
        if not ordered:
            return []
        result = ["{", f"    fort_scoped::{batch_type}<{capacity}> fort_metadata(fort_context);",
                  "    fort_scope_access fort_read{}; fort_read.flags = FORT_SCOPE_READ_ALL;"]
        for s in ordered:
            result += [f"    FORT_SHARED_CHECK(fort_metadata.add({array_argument(s)}, fort_read));"]
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

    def descriptors(unit):
        result = [f"fort_scoped::{batch_type}<{capacity}> fort_access(fort_context);", "bool fort_valid = true;"]
        checked = any(isinstance(node, ArrayAccess)
                      for fp in unit.footprints for box in (*fp.reads, *fp.writes)
                      for expression in (*box.lower, *box.upper) for node in walk_expr(expression))
        if checked:
            result += ["offload::Data d;"]
        value = (lambda expression: _query_expression(expression, {})) if checked else query_expression
        invalid = 'return fort_scope_report_error(FORT_SCOPE_ARGUMENT, "invalid physical entry footprint");'

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
        # Layout acquisition above has already checked this array-bearing
        # entry's context. Pure local integer/logical setup has no coherent
        # storage access or floating-point effect to publish. In particular,
        # loop-header snapshots must not open empty host transactions.
        local_setup = arrays and isinstance(step, HostBlock) and not touched.intersection(arrays)
        if local_setup:
            for assignment in step.assignments:
                if (not isinstance(assignment.target, Reference) or assignment.target.symbol.parameter
                        or assignment.target.symbol.dtype not in {ScalarType.INTEGER, ScalarType.LOGICAL}):
                    local_setup = False
                    break
                if any((isinstance(node, Reference) and node.symbol.dtype not in {ScalarType.INTEGER, ScalarType.LOGICAL})
                       or (isinstance(node, (Literal, IntrinsicCall)) and node.dtype not in {ScalarType.INTEGER, ScalarType.LOGICAL})
                       for node in walk_expr(assignment.value)):
                    local_setup = False
                    break
        if local_setup:
            return ["fort_entry_work_started=true;", *executable]
        result = ["{", f"    fort_scoped::{batch_type}<{capacity}> fort_host(fort_context);"]
        for s in arrays:
            if s in touched:
                flags = (1 if s in step.read_symbols else 0) | (2 if s in step.write_symbols else 0)
                result += [f"    fort_scope_access fort_effect_{s.id}{{}}; fort_effect_{s.id}.flags = {flags};",
                           f"    FORT_SHARED_CHECK(fort_host.add({array_argument(s)}, fort_effect_{s.id}));"]
        return [*result, "    FORT_SHARED_CHECK(fort_host.begin(false));", "    fort_host.executing();",
                "    fort_entry_work_started=true;",
                *indent(executable), "    FORT_SHARED_CHECK(fort_host.finish());", "}"]

    graph_launch_groups = {}

    def queued_task(step, *, accepted):
        if not graph_group_eligible((step,), units, protected):
            return []
        key = int(sha256(f"{name}:graph:{step.id}".encode()).hexdigest()[:16], 16)
        graph_launch_groups[key] = {"key": key, "regions": [step.id],
            "ordering": "checked task order; runtime groups across numerical entries",
            "effects": "owned per-task effects and arguments; virtual ordered definitions",
            "empty_domains": "omitted before argument capture and resource acquisition",
            "fallback": "original task only before enqueue; accepted work is never replayed"}
        return generate_graph_group((step,), units=units, arrays=arrays, capacity=capacity,
            bounds=bounds, descriptors=descriptors, array_argument=array_argument,
            root_views=root_views, group_key=key, fallback=(), enqueue=True,
            accepted=accepted, condition="fort_gpu", prepared_access="fort_access")

    def merged_launch(group, fallback):
        key = int(sha256(f"{name}:group:{'-'.join(str(region.id) for region in group)}".encode()).hexdigest()[:16], 16)
        graph_launch_groups[key] = {"key": key, "regions": [region.id for region in group],
            "ordering": "members of one launch; no member touches an array another writes",
            "effects": "per-member effects; the runtime accepts only independent actual sections and queues their union",
            "empty_domains": "omitted before argument capture and resource acquisition",
            "fallback": "the members' own tasks in order, before enqueue; accepted work is never replayed"}
        position = (f'fort_scoped::TracePosition fort_worker_trace(fort_context, "{procedure_id}", '
                    f'"{implementation_ids[group[0].id, "cuda"]}");')
        return generate_graph_group(group, units=units, arrays=arrays, capacity=capacity,
            bounds=bounds, descriptors=descriptors, array_argument=array_argument,
            root_views=root_views, group_key=key, fallback=fallback, enqueue=True,
            accepted=("fort_entry_work_started = true;",), condition="fort_gpu_requested",
            merged=True, prologue=(position,))

    group_first = {group[0].id: group for group in merged_launches}
    group_last = {group[-1].id: group for group in merged_launches}

    def execution(current):
        result = []
        group_start = None
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
                if step.id in group_first:
                    group_start = len(result)
                result += ["{", *indent(bounds(step)), "    if (fort_internal_total) {"]
                # A CUDA launch captures scalar values before its kernel body.
                # References preserve CPU guards, but cannot make that capture
                # lazy. Keep only the affected worker native; subsequent safe
                # workers can still use the same coherent device allocations.
                gpu_choice = "false" if protected[step.id] else "fort_gpu_requested"
                body = [*descriptors(unit), f"bool fort_gpu = {gpu_choice};"]
                if protected[step.id]:
                    body += [f'if (fort_gpu_requested) offload::decision_trace("{function.name}", "native-scoped-protected-scalar", 0, 1);']
                body += [trace_worker(step)]
                enqueue = queued_task(step, accepted=("fort_task_queued = true;", "fort_entry_work_started = true;"))
                if enqueue:
                    body += ["bool fort_task_queued = false;", *enqueue, "if (!fort_task_queued) {"]
                body += ["int fort_status = fort_access.begin(fort_gpu);",
                        "if (fort_gpu && (fort_status == FORT_SCOPE_RESOURCE || fort_status == FORT_SCOPE_BOUNDARY)) {",
                        f'    offload::decision_trace("{function.name}", fort_status == FORT_SCOPE_RESOURCE ? "native-scoped-resource" : "native-scoped-boundary", 0, 1);',
                        "    fort_gpu = false;", "    " + trace_cpu(step), "    fort_status = fort_access.begin(false);", "}",
                        "FORT_SHARED_CHECK(fort_status);", "fort_scoped::GPUCall fort_gpu_call(fort_context);",
                        "if (fort_gpu) {", "    fort_status = fort_gpu_call.begin();",
                        "    if (fort_status == FORT_SCOPE_RESOURCE) {", "        fort_access.cancel(); fort_gpu = false;",
                        "        " + trace_cpu(step),
                        f'        offload::decision_trace("{function.name}", "native-scoped-gpu-setup", 0, 1);',
                        "        fort_status = fort_access.begin(false);", "    }", "    FORT_SHARED_CHECK(fort_status);", "}"]
                for s in arrays:
                    pointer = f"static_cast<{cpp_type(s)} *>(fort_access.device({s.cpp_name}_handle))"
                    value_type = ("const " if s.intent == "in" else "") + cpp_type(s)
                    device = (f"fort_scoped::RootView<{value_type}, {s.rank}>({s.cpp_name}_layout, {pointer})"
                              if root_views else pointer)
                    body += [f"if (fort_gpu) {s.cpp_name}_device = {device};"]
                body += ["fort_access.executing();", *(["fort_entry_work_started=true;"]), "if (fort_gpu) {", *indent(generate_launch(
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
                if enqueue:
                    body += ["}"]
                result += [*indent(body, 2), "    }", "}"]
                if step.id in group_last:
                    # The members' own code, emitted just above, is the fallback.
                    result[group_start:] = merged_launch(group_last[step.id], result[group_start:])
            else:
                raise TypeError(step)
        return result

    lines += [*indent(execution(plan)), "    return FORT_SCOPE_OK;",
              *(["    }();", "    if (fort_entry_result && fort_entry_work_started)",
                 '        return fort_scope_execution_error(fort_context, "borrowed numerical entry failed after execution started; unsafe replay prohibited");',
                 "    return fort_entry_result;"]), "}"]
    lines += ["}", "#undef FORT_SHARED_CHECK", ""]
    parameters = ["fort_context", "fort_mode", *[array_argument(s) for s in arrays], *[s.cpp_name for s in scalars]]
    view_import = ", " + view_type if root_views else ""
    fortran = [f"module {name}", "  use iso_c_binding",
               *(["  use fort_scoped_memory, only: " + view_type] if root_views else []),
               "  implicit none", "  private", "  public :: run, configure", "  interface"]
    fortran += _fortran_list("function run(", parameters, f") bind(C, name='{c_name}') result(fort_status)", 4)
    fortran += ["      import :: c_int, c_int64_t, c_double, c_float, c_bool" + view_import, "      integer(c_int) :: fort_status",
                "      integer(c_int64_t), value :: fort_context", "      integer(c_int), value :: fort_mode"]
    fortran += [(f"      type({view_type}), intent(in) :: {s.cpp_name}_view" if root_views else
                 f"      integer(c_int64_t), value :: {s.cpp_name}_handle") for s in arrays]
    fortran += [f"      {TYPES[s.dtype][1]}, intent(in)" + (", value" if s.by_value else "") + f" :: {s.cpp_name}" for s in scalars]
    fortran += ["    end function"]
    fortran += [f"    function configure(fort_context) bind(C, name='{configure_name}') result(fort_status)",
                "      import :: c_int, c_int64_t", "      integer(c_int) :: fort_status",
                "      integer(c_int64_t), value :: fort_context", "    end function",
                "  end interface", f"end module {name}", ""]
    report = {
        "schema_version": 1, "abi_version": 1, "entry_abi_version": ENTRY_ABI_VERSION,
        "numerical_contract": numerical_build_contract(),
        "entry": c_name, "fortran_module": name, "fortran_procedure": "run",
        "cuda_source": "shared_entry.cu", "fortran_source": "shared_interface.f90",
        "array_parameters": [{"name": s.name, "rank": s.rank, "dtype": s.dtype.value,
                              "written": s in writes} for s in arrays],
        "scalar_parameters": [{"name": s.name, "dtype": s.dtype.value,
                               "passing": "value" if s.by_value else "reference"} for s in scalars],
        "argument_order": ["context", "mode", *[s.name for s in arrays], *[s.name for s in scalars]],
        "modes": {"native": 0, "gpu": 1},
        # With conditions, the entry returns FORT_SCOPE_DECLINED where one does not hold.
        "conditions": [render_expression(expression) for expression in assumptions],
        "numerical_environment": {
            "required": function.requires_numerical_environment,
            "check": "fort_scope_numerical_environment_supported",
            "rounding": "round to nearest", "exceptions": "host traps disabled",
            "caller_contract": "floating-point exception flags are not observed",
            "check_position": "before entry work; every run",
            "unsupported": "pre-work boundary; original-source caller executes its coherent native operation",
            "generated_cpu_fallback": "does not replace the original source for unsupported environments",
        },
        "transfer_configuration": {"abi_version": 1, "requested": config.scope_transfers,
                                   "selected": ("runtime" if config.scope_transfers == "coalesced" else
                                                "pinned" if config.scope_transfers == "pinned" else "direct"),
                                   "reason": transfer_reason, "available_modes": ["direct", "pinned", "coalesced"],
                                   "entry": configure_name, "fortran_procedure": "configure",
                                   "argument_order": ["context"],
                                   "position": "once per owning context before registration",
                                   "execution": "synchronous", "pinned_budget_bytes": 64 * 1024 * 1024,
                                   "budget_scope": "process-wide including cached staging",
                                   "resource_fallback": "direct before this copy is enqueued; numerical work is never replayed",
                                   "runtime_stats": "fort_scope_transfer_stats_get_v1",
                                   "physical_copy_calibration": {
                                       "abi_version": 3, "available": False,
                                       "selection": "AUTO selects supported complete Compact predictions below half the direct prediction",
                                       "missing_evidence": "direct physical copies",
                                   },
                                   "coalesced": {
                                       "experimental": True, "direction": "upload only; outputs remain exact",
                                       "minimum_physical_copies": 2, "maximum_extra_bytes_fraction": 0.125,
                                       "source_requirement": "whole canonical source resource is defined and host current",
                                       "coordinates": "unchanged full-array layout and logical indices",
                                       "pinning": False,
                                       "initial_effective_mode": "direct",
                                       "selected": "runtime eligibility per upload",
                                       "fallback": "exact direct transfer",
                                       "fallback_reason": "coalesced_upload_ineligible",
                                       "trace_events": {"upload_coalesced": "actual physical span bytes",
                                                        "upload_requested": "required section bytes"}}},
        "runtime_provenance": provenance.public(),
        "host_threads": config.host_threads,
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
    report["graph_execution"] = {"abi_version": 1, "runtime_api": "fort_scope_graph_enqueue_v1",
        "groups": list(graph_launch_groups.values()),
        "selection": "device-mode tasks enqueue; host-mode tasks acquire host effects",
        "submission": "context queue spans numerical calls; conflicting host tasks drain dependencies",
        "parameter_lifetime": "runtime owns nonempty task values and effects until submission",
        "fusion": "fresh dependence proofs before graph launch planning"}
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
    return ScopedEmission(numerical_source_prologue() + "\n".join(lines), "\n".join(fortran), report)
