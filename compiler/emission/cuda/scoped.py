"""Numerical entries borrowing common buffers through the versioned public ABI."""

from dataclasses import dataclass
from hashlib import sha256

from compiler.emission.c.declarations import cpp_declaration
from compiler.emission.common.abi import abi_arguments, dimension_name
from compiler.emission.common.c_family import cpp_type, indent, render_assignment, render_expression
from compiler.emission.common.loops import _loop_snapshot, sequential_block
from compiler.emission.common.schedules import checked_product, region_schedule, tile_counts
from compiler.emission.common.symbols import host_symbols, region_symbols
from compiler.emission.cuda.kernels import generate_kernel, generate_launch
from compiler.emission.cuda.offload import _cpu_worker
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
from compiler.offload.analysis import Unit, _unit_footprints
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


def generate_scoped(function, plan, config, common_header):
    arrays = tuple(s for s in function.parameters if s.rank)
    scalars = tuple(s for s in function.parameters if not s.rank)
    if any(s.intent != "in" for s in scalars):
        raise CompilationError("shared numerical entries require read-only scalar parameters")
    if config.collective:
        raise CompilationError("shared numerical entry requires a serial coordinator; collective scope hooks are pending")
    digest = sha256(f"{function.module.lower()}::{function.name.lower()}:{plan!r}".encode()).hexdigest()[:12]
    name = "fort_shared_" + digest
    c_name = "cpp_" + name
    scalar_signature = [f"const {cpp_type(s)} &{s.cpp_name}" for s in scalars]
    signature = ["fort_scope_t fort_context", "int fort_mode",
                 *[f"fort_buffer_t {s.cpp_name}_handle" for s in arrays], *scalar_signature]
    abi = abi_arguments(function.parameters)
    worker_signature = ", ".join(cpp_declaration(a) if a.symbol.rank else
                                  f"const {cpp_type(a.symbol)} &{a.name}" for a in abi)
    worker_arguments = ", ".join(a.name for a in abi)
    capacity = max(len(arrays), 1)
    lines = ['#include <cuda_runtime.h>', '#include <omp.h>', '#define FORT_OFFLOAD_ENABLED 1',
             f'#include "{common_header}"', '#include "scoped_entry.hpp"',
             '#define FORT_SHARED_CHECK(expr) do { int fort_check_result = (expr); if (fort_check_result) return fort_check_result; } while (false)',
             f"namespace generated_kernels::{name} {{", "using namespace indexing;"]
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
    captures = {}
    locals_ = host_symbols(function, plan)
    for region in plan.regions:
        used = set(region_symbols(region))
        for loop in region.loops:
            for expression in (loop.lower, loop.upper) + (() if isinstance(loop.step, int) else (loop.step,)):
                used.update(referenced_symbols(expression))
        captures[region.id] = tuple(s for s in locals_ if s in used)
        extra_signature = "".join(f", const {cpp_type(s)} &{s.cpp_name}" for s in captures[region.id])
        lines.extend(generate_kernel(region))
        lines.extend(_cpu_worker(units[region.id], worker_signature + extra_signature, f"cpu_{region.id}"))
    lines += [f'extern "C" int {c_name}({", ".join(signature)}) {{',
              "    if (fort_scope_abi_version() != FORT_SCOPE_ABI_VERSION)",
              '        return fort_scope_report_error(FORT_SCOPE_ARGUMENT, "incompatible shared runtime ABI");',
              "    if (fort_mode < 0 || fort_mode > 2)",
              '        return fort_scope_report_error(FORT_SCOPE_ARGUMENT, "unknown shared execution mode");']
    writes = set()

    def written(current):
        for step in current.steps:
            writes.update(step.write_symbols)
            if isinstance(step, ConditionalRegion):
                written(step.then_plan)
                written(step.else_plan)

    written(plan)
    for i, a in enumerate(arrays):
        for b in arrays[:i]:
            if a in writes or b in writes:
                lines += [f"    if ({a.cpp_name}_handle == {b.cpp_name}_handle)",
                          '        return fort_scope_report_error(FORT_SCOPE_ALIAS, "writable entry arguments alias");']
        field = a.cpp_name + "_layout"
        lines += [f"    fort_scope_layout {field}{{}};",
                  f"    FORT_SHARED_CHECK(fort_scope_layout_get(fort_context, {a.cpp_name}_handle, &{field}));",
                  f"    if ({field}.rank != {a.rank} || {field}.type != {TYPES[a.dtype][0]} ||",
                  f"        {field}.element_bytes != sizeof({cpp_type(a)}))",
                  '        return fort_scope_report_error(FORT_SCOPE_ARGUMENT, "entry layout/type mismatch");',
                  f"    auto *{a.cpp_name} = static_cast<{cpp_type(a)} *>({field}.host);",
                  f"    {cpp_type(a)} *{a.cpp_name}_device = nullptr;"]
        lines += [f"    const std::size_t {dimension_name(a,k+1)} = {field}.extents[{k}];" for k in range(a.rank)]
    lines += [f"    {cpp_type(s)} {s.cpp_name};" for s in host_symbols(function, plan)]
    for s in arrays:
        if s.intent == "out":
            lines += [f"    FORT_SHARED_CHECK(fort_scope_forget_definition(fort_context, {s.cpp_name}_handle));"]
    # Until scope-wide calibrated placement is connected, automatic mode remains
    # a successful coherent native choice. Explicit mode 1 exercises GPU workers.
    lines += ["    const bool fort_gpu_requested = fort_mode == 1;",
              f'    if (fort_mode == 2) offload::decision_trace("{function.name}", "native-scoped-estimate-unavailable", 0, {len(units)});']

    def prefetch(symbols):
        ordered = [s for s in arrays if s in symbols]
        if not ordered:
            return []
        result = ["{", f"    fort_scoped::AccessBatch<{capacity}> fort_metadata(fort_context);",
                  "    fort_scope_access fort_read{}; fort_read.flags = FORT_SCOPE_READ_ALL;"]
        for s in ordered:
            result += [f"    FORT_SHARED_CHECK(fort_metadata.add({s.cpp_name}_handle, fort_read));"]
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
        result = [f"fort_scoped::AccessBatch<{capacity}> fort_access(fort_context);", "bool fort_valid = true;"]
        for fp in unit.footprints:
            s = fp.symbol
            prefix = "fort_effect_" + s.cpp_name
            result += [f"fort_scope_access {prefix}{{}};"]
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
                                   f"    const auto lo = offload::index({query_expression(lo)}, fort_valid);",
                                   f"    const auto hi = offload::index({query_expression(hi)}, fort_valid);",
                                   f"    if (!fort_valid || lo < 1 || hi < lo || static_cast<unsigned long long>(hi) > {dimension_name(s,axis+1)})",
                                   '        return fort_scope_report_error(FORT_SCOPE_ARGUMENT, "invalid physical entry footprint");',
                                   f"    {base}_lower[{axis}] = static_cast<std::size_t>(lo-1);",
                                   f"    {base}_upper[{axis}] = static_cast<std::size_t>(hi);", "}"]
                items = ", ".join(f"{{{prefix}_{label}_{i}_lower, {prefix}_{label}_{i}_upper}}" for i in range(len(boxes)))
                result += [f"fort_scope_section {prefix}_{label}s[] = {{{items}}};",
                           f"{prefix}.{label}_count = {len(boxes)}; {prefix}.{label}s = {prefix}_{label}s;"]
                if label == "write" and fp.exact:
                    result += [f"{prefix}.overwrite_count = {len(boxes)}; {prefix}.overwrites = {prefix}_writes;"]
            result += [f"FORT_SHARED_CHECK(fort_access.add({s.cpp_name}_handle, {prefix}));"]
        return result

    def host(step, executable):
        touched = set(step.read_symbols) | set(step.write_symbols)
        result = ["{", f"    fort_scoped::AccessBatch<{capacity}> fort_host(fort_context);"]
        for s in arrays:
            if s in touched:
                flags = (1 if s in step.read_symbols else 0) | (2 if s in step.write_symbols else 0)
                result += [f"    fort_scope_access fort_effect_{s.id}{{}}; fort_effect_{s.id}.flags = {flags};",
                           f"    FORT_SHARED_CHECK(fort_host.add({s.cpp_name}_handle, fort_effect_{s.id}));"]
        return [*result, "    FORT_SHARED_CHECK(fort_host.begin(false));", "    fort_host.executing();",
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
                result += ["{", *indent(bounds(step)), "    if (fort_internal_total) {"]
                body = [*descriptors(unit), "bool fort_gpu = fort_gpu_requested;",
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
                    body += [f"if (fort_gpu) {s.cpp_name}_device = static_cast<{cpp_type(s)} *>(fort_access.device({s.cpp_name}_handle));"]
                body += ["fort_access.executing();", "if (fort_gpu) {", *indent(generate_launch(
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
            else:
                raise TypeError(step)
        return result

    lines += [*indent(execution(plan)), "    return FORT_SCOPE_OK;", "}", "}", "#undef FORT_SHARED_CHECK", ""]
    parameters = ["fort_context", "fort_mode", *[s.cpp_name + "_handle" for s in arrays], *[s.cpp_name for s in scalars]]
    fortran = [f"module {name}", "  use iso_c_binding", "  implicit none", "  private", "  public :: run", "  interface"]
    fortran += _fortran_list("function run(", parameters, f") bind(C, name='{c_name}') result(fort_status)", 4)
    fortran += ["      import :: c_int, c_int64_t, c_double, c_float, c_bool", "      integer(c_int) :: fort_status",
                "      integer(c_int64_t), value :: fort_context", "      integer(c_int), value :: fort_mode"]
    fortran += [f"      integer(c_int64_t), value :: {s.cpp_name}_handle" for s in arrays]
    fortran += [f"      {TYPES[s.dtype][1]}, intent(in) :: {s.cpp_name}" for s in scalars]
    fortran += ["    end function", "  end interface", f"end module {name}", ""]
    report = {
        "schema_version": 1, "abi_version": 1, "entry": c_name, "fortran_module": name, "fortran_procedure": "run",
        "cuda_source": "shared_entry.cu", "fortran_source": "shared_interface.f90",
        "array_parameters": [{"name": s.name, "rank": s.rank, "dtype": s.dtype.value,
                              "written": s in writes} for s in arrays],
        "scalar_parameters": [{"name": s.name, "dtype": s.dtype.value, "passing": "reference"} for s in scalars],
        "argument_order": ["context", "mode", *[s.name for s in arrays], *[s.name for s in scalars]],
        "modes": {"native": 0, "gpu": 1, "automatic": 2},
        "automatic_estimate_available": False,
        "automatic_reason": "scope-wide coherent placement is not yet connected; automatic mode selects native",
        "automatic_scope_available": False, "host_threads": config.host_threads,
        "definition_changes": [s.name for s in arrays if s.intent == "out"],
        "resource_failure": "continue native at current worker before execution; never replay completed work",
        "transfer_volume": "missing physical read/preservation sections from shared runtime state",
        "parallel_regions": len(plan.regions), "source": function.source,
    }
    return ScopedEmission("\n".join(lines), "\n".join(fortran), report)
