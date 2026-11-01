"""One generated whole-team kernel on the existing deferred graph engine.

This transport is deliberately separate from SourceDevicePatch/source_execute.
Acceptance becomes an irrevocable queue ownership boundary; no original native
replay is allowed after that boundary, even when submission later fails.
"""
from __future__ import annotations

import json
from hashlib import sha256

from compiler.emission.cuda.tile_access import access_preflight


def queued_single_provider(*, team, entry, module, parameters, state_fields, kernels,
                           plan, checks, envelopes, bindings, values, kernel, mapping,
                           support, readable_scalars, launch_abi, target_bytes):
    """Internal rendering after the ordinary exact team/IR/ABI checks."""
    from compiler.emission.cuda.tile_task_provider import TileTaskProvider

    fields = len(team.fields)
    key = int(sha256(entry.encode()).hexdigest()[:16], 16) or 1
    ctype = {"c_int": "int", "c_float": "float", "c_double": "double"}
    arguments = ["fort_scope_t context", *[f"fort_buffer_t field{i}" for i in range(fields)],
                 *[f"const {ctype[p.kind]}* {p.name}" for p in parameters], "fort_tile_queue_report_v1* report"]
    blocks = "count" if mapping == "cooperative" else "count/256+(count%256!=0)"
    source = ["#include <cuda_runtime.h>", '#include "scoped_runtime.h"', '#include "common_functions.cuh"',
              '#include "numeric.hpp"', "#include <algorithm>", "#include <climits>", "#include <cstdint>",
              "#include <vector>", "#include <new>", "using generated_kernels::indexing::F_IDX;",
              "struct fort_tile_queue_report_v1 { std::uint32_t version; int status; std::uint32_t accepted,committed; };",
              "namespace {", support, "struct tile_state {", *state_fields, "};", *kernels,
              *access_preflight(team.fields, checks, envelopes, layouts_argument=True),
              "}", f'extern "C" int {entry}(' + ",".join(arguments) + ") {",
              "if (!report) return FORT_SCOPE_ARGUMENT;", "*report={1,FORT_SCOPE_ARGUMENT,0,0};",
              "auto fail=[&](int status) { report->status=status; return status; };",
              "if (!fort_scope_serial_caller() || !fort_scope_numerical_environment_supported()) return fail(FORT_SCOPE_BOUNDARY);",
              "tile_state state{};", *plan,
              "std::size_t scratch_bytes=0;",
              "if (!state.stride[0] || !state.total[0]) return fail(FORT_SCOPE_RESOURCE);",
              f"state.batch=std::min(state.total[0],std::max<std::size_t>(1,{target_bytes}/state.stride[0]));",
              "const std::size_t expected=state.total[0]/state.batch+(state.total[0]%state.batch!=0);",
              "if (expected!=1) return fail(FORT_SCOPE_RESOURCE);",
              "if (!multiply(state.stride[0],state.total[0],scratch_bytes)) return fail(FORT_SCOPE_RESOURCE);",
              "fort_scope_graph_queue_stats_v1 queue{};",
              "int status=fort_scope_graph_queue_stats_get_v1(context,&queue);",
              "if (status!=FORT_SCOPE_OK) return fail(status);",
              "if (!queue.queue_enabled || queue.pending_tasks) return fail(FORT_SCOPE_BOUNDARY);",
              "fort_scope_external_binding_v1 bindings[]={" + ",".join(bindings) + "};",
              f"OwnedTileAccess accesses[{fields}]; fort_scope_layout layouts[{fields}]{{}};",
              "status=prepare_tile_access(context,state,bindings,accesses,layouts);",
              "if (status!=FORT_SCOPE_OK) return fail(status);",
              "fort_scope_scratch_lease_v1 lease{};",
              "status=fort_scope_scratch_acquire_v1(context,scratch_bytes,&lease);",
              "if (status!=FORT_SCOPE_OK) return fail(status);",
              "state.scratch=static_cast<unsigned char*>(lease.device);",
              "auto decline=[&](int result) {",
              " const int released=fort_scope_scratch_release_v1(context,lease.token);",
              " if (released!=FORT_SCOPE_OK) {",
              '  fort_scope_execution_error(context,"queued source prework cleanup failed; abandon context");',
              "  return fail(FORT_SCOPE_EXECUTION);", " }", " return fail(result);", "};",
              "fort_buffer_t handles[]={" + ",".join(f"field{i}" for i in range(fields)) + "};",
              f"void* devices[{fields}]{{}};",
              f"status=fort_scope_device_reserve_many_v1(context,handles,{fields},devices);",
              "if (status!=FORT_SCOPE_OK) return decline(status);",
              "const std::size_t first=0,count=state.total[0];",
              f"const unsigned blocks=static_cast<unsigned>(std::min<std::size_t>({blocks},65535));"]
    source += [f"auto arg{i}={value};" for i, value in enumerate(values)]
    source += ["void* arguments[]={" + ",".join(f"&arg{i}" for i in range(len(values))) + "};",
               "const std::size_t sizes[]={" + ",".join(f"sizeof(arg{i})" for i in range(len(values))) + "};",
               f"const fort_scope_kernel_v1 kernel{{1,0,reinterpret_cast<const void*>({kernel}),{{blocks,1,1}},{{256,1,1}},0,arguments,{len(values)}}};",
               "fort_scope_plan_binding effects[]={" + ",".join(f"{{field{i},bindings[{i}].access}}" for i in range(fields)) + "};",
               f"const fort_scope_graph_task_v1 task{{kernel,effects,{fields}}};",
               f"status=fort_scope_graph_enqueue_v1(context,{key}ULL,&task,sizes);",
               "if (status!=FORT_SCOPE_OK) return decline(status);",
               "report->accepted=1;",  # Queue may run at any later conflicting boundary.
               "status=fort_scope_graph_flush_v1(context);",
               "if (status!=FORT_SCOPE_OK) {",
               ' fort_scope_execution_error(context,"accepted queued source failed; native replay prohibited; abandon context");',
               " return fail(FORT_SCOPE_EXECUTION);", "}",
               "report->committed=1;",  # Submission commit, not device completion.
               "status=fort_scope_scratch_release_v1(context,lease.token);",
               "if (status!=FORT_SCOPE_OK) {",
               ' fort_scope_execution_error(context,"submitted queued source cleanup failed; native replay prohibited; abandon context");',
               " return fail(FORT_SCOPE_EXECUTION);", "}",
               "return fail(FORT_SCOPE_OK);", "}", ""]
    iso_type = {"c_int": "integer(c_int)", "c_float": "real(c_float)", "c_double": "real(c_double)"}
    names = ["context", *[f"field{i}" for i in range(fields)], *[p.name for p in parameters], "report"]
    fortran = [f"module {module}", "use iso_c_binding", "implicit none",
               "type,bind(c) :: fort_tile_queue_report_v1", " integer(c_int32_t) :: version,status,accepted,committed",
               "end type", "interface", f"integer(c_int) function {entry}( &",
               " & " + ", &\n & ".join(names) + f') bind(c,name="{entry}")', "import",
               "integer(c_int64_t),value :: context", *[f"integer(c_int64_t),value :: field{i}" for i in range(fields)],
               *[f"{iso_type[p.kind]},intent(in) :: {p.name}" for p in parameters],
               "type(fort_tile_queue_report_v1),intent(out) :: report", "end function", "end interface", "end module", ""]
    cuda_source, fortran_source = "\n".join(source), "\n".join(fortran)
    record = {"version": 1, "kind": "queued_single_original_tile_private_team", "team_identity": team.identity,
              "scalar_readability_identity": readable_scalars.identity if readable_scalars is not None else None,
              "unit_proof_identities": [u.identity for u in team.units], "mapping": mapping,
              "entry": entry, "module": module, "parameters": [p.public() for p in parameters],
              "cuda_sha256": sha256(cuda_source.encode()).hexdigest(),
              "fortran_sha256": sha256(fortran_source.encode()).hexdigest(),
              "backend": "fort_scope_graph_enqueue_v1+fort_scope_graph_flush_v1",
              "code_key": str(key), "launch_abi": launch_abi,
              "effects": {"source": "existing exact whole-team ordinary access envelopes",
                          "atomic_commit": "existing queue graph effect simulator; one kernel covers every original tile",
                          "overwrites_authorized": False, "initialization_authorized": False},
              "report": {"type": "fort_tile_queue_report_v1", "version": 1,
                         "fields": ["version", "status", "accepted", "committed"],
                         "accepted": "irrevocable queue ownership; native replay prohibited",
                         "committed": "successful graph/direct submission effect commit; not device completion"},
              "restriction": {"units": 1, "expected_launches": 1, "queue_enabled": True,
                              "empty_queue_at_entry": True, "target_bytes": target_bytes,
                              "fallback": "original complete team only on recoverable status before acceptance",
                              "recoverable_statuses": [1, 4, 5, 7, 9],
                              "postaccept_failure": "poison and abandon; no native replay"},
              "scratch": "lease spans enqueue and flush; shared same-stream arena lifetime/reuse after release",
              "canonical_storage": "current registered handles and typed layouts; queue owns argument bytes and effect coordinates; no pointer cache",
              "definitions_authorized": False, "placement_authorized": False,
              "pricing": "unpriced; distinct queued backend; no source_execute/tile cost reuse",
              "consumer_wiring": False}
    return TileTaskProvider(entry, module, cuda_source, fortran_source, tuple(team.fields), parameters,
                            json.dumps(record, sort_keys=True, separators=(",", ":")))
