"""A complete original multi-kernel team on the existing atomic queue flush.

All plans, conservative effects and storage reservations precede acceptance.
No task prefix defines a new value and no accepted prefix is native-replayable.
"""
from __future__ import annotations

import json
from hashlib import sha256

from compiler.emission.cuda.tile_access import access_preflight

METADATA_BUDGET_BYTES = 16 * 1024 * 1024
KIND = "queued_original_tile_private_team"


def queued_team_provider(*, team, entry, module, parameters, state_fields, kernels,
                         plan, checks, envelopes, bindings, launches, mapping,
                         support, readable_scalars, target_bytes):
    from compiler.emission.cuda.tile_task_provider import TileTaskProvider

    fields = len(team.fields)
    ctype = {"c_int": "int", "c_float": "float", "c_double": "double"}
    arguments = ["fort_scope_t context", *[f"fort_buffer_t field{i}" for i in range(fields)],
                 *[f"const {ctype[p.kind]}* {p.name}" for p in parameters], "fort_tile_queue_report_v1* report"]
    keys = [int(sha256((entry + ":unit:" + str(i)).encode()).hexdigest()[:16], 16) or 1
            for i in range(len(launches))]
    source = ["#include <cuda_runtime.h>", '#include "scoped_runtime.h"', '#include "common_functions.cuh"',
              '#include "numeric.hpp"', "#include <algorithm>", "#include <climits>", "#include <cstdint>",
              "#include <vector>", "#include <new>", "#include <type_traits>", "using generated_kernels::indexing::F_IDX;",
              "struct fort_tile_queue_report_v1 { std::uint32_t version; int status; std::uint32_t accepted,committed; };",
              "namespace {", support, "struct tile_state {", *state_fields, "};", *kernels,
              *access_preflight(team.fields, checks, envelopes, layouts_argument=True),
              "struct PlannedTask { fort_scope_graph_task_v1 task; const std::size_t* sizes; std::uint64_t key; };",
              "static bool add_payload(std::size_t count,std::size_t width,std::size_t& total) {",
              " std::size_t bytes=0; if (!multiply(count,width,bytes) || bytes>SIZE_MAX-total) return false;",
              " total+=bytes; return true;", "}", "}",
              f'extern "C" int {entry}(' + ",".join(arguments) + ") {",
              "if (!report) return FORT_SCOPE_ARGUMENT;", "*report={1,FORT_SCOPE_ARGUMENT,0,0};",
              "auto fail=[&](int status) { report->status=status; return status; };",
              "fort_scope_scratch_lease_v1 lease{}; bool leased=false;",
              "auto fatal=[&]() {",
              ' fort_scope_execution_error(context,"accepted queued whole team failed; abandon; native replay prohibited");',
              " return fail(FORT_SCOPE_EXECUTION);", "};",
              "auto decline=[&](int status) {",
              " if (leased) { const int released=fort_scope_scratch_release_v1(context,lease.token);",
              "  if (released!=FORT_SCOPE_OK) return fatal(); leased=false; }",
              " return fail(status);", "};", "try {",
              "if (!fort_scope_serial_caller() || !fort_scope_numerical_environment_supported()) return fail(FORT_SCOPE_BOUNDARY);",
              "tile_state state{};", *plan,
              "std::size_t max_stride=0,max_tiles=0,scratch_bytes=0,expected=0;",
              f"std::size_t unit_launches[{len(launches)}]{{}};",
              f"for (std::size_t unit=0;unit<{len(launches)};++unit) {{",
              " max_stride=std::max(max_stride,state.stride[unit]); max_tiles=std::max(max_tiles,state.total[unit]);", "}",
              "if (!max_stride || !max_tiles) return fail(FORT_SCOPE_RESOURCE);",
              f"state.batch=std::min(max_tiles,std::max<std::size_t>(1,{target_bytes}/max_stride));",
              "if (!multiply(max_stride,state.batch,scratch_bytes)) return fail(FORT_SCOPE_RESOURCE);",
              f"for (std::size_t unit=0;unit<{len(launches)};++unit) {{",
              " unit_launches[unit]=state.total[unit]/state.batch+(state.total[unit]%state.batch!=0);",
              " if (unit_launches[unit]>SIZE_MAX-expected) return fail(FORT_SCOPE_RESOURCE);",
              " expected+=unit_launches[unit];", "}",
              "fort_scope_graph_queue_stats_v1 queue{};",
              "int status=fort_scope_graph_queue_stats_get_v1(context,&queue);",
              "if (status!=FORT_SCOPE_OK) return fail(status);",
              "if (!queue.queue_enabled || queue.pending_tasks) return fail(FORT_SCOPE_BOUNDARY);",
              "fort_scope_external_binding_v1 bindings[]={" + ",".join(bindings) + "};",
              f"OwnedTileAccess accesses[{fields}]; fort_scope_layout layouts[{fields}]{{}};",
              "status=prepare_tile_access(context,state,bindings,accesses,layouts);",
              "if (status!=FORT_SCOPE_OK) return fail(status);",
              f"void* devices[{fields}]{{}}; std::size_t first=0,count=0;",
              "std::size_t metadata_bytes=0,effect_payload=0;",
              f"if (!add_payload({fields},sizeof(fort_scope_plan_binding),effect_payload)) return fail(FORT_SCOPE_RESOURCE);",
              f"for (std::size_t field=0;field<{fields};++field) {{",
              " const auto& access=bindings[field].access;",
              " if (access.read_count>SIZE_MAX-access.write_count) return fail(FORT_SCOPE_RESOURCE);",
              " const auto boxes=access.read_count+access.write_count; std::size_t coordinates=0;",
              " if (!multiply(2,layouts[field].rank,coordinates) ||",
              "     !add_payload(boxes,sizeof(fort_scope_section),effect_payload) ||",
              "     !multiply(boxes,coordinates,coordinates) ||",
              "     !add_payload(coordinates,sizeof(std::size_t),effect_payload)) return fail(FORT_SCOPE_RESOURCE);", "}",
              "if (!add_payload(expected,sizeof(PlannedTask),metadata_bytes) ||",
              "    !add_payload(expected,effect_payload,metadata_bytes)) return fail(FORT_SCOPE_RESOURCE);"]
    for index, (_kernel, values, _) in enumerate(launches):
        source += [f"struct Arguments{index} {{"]
        source += [f" std::decay_t<decltype({value})> value{i};" for i, value in enumerate(values)]
        source += [f" void* pointers[{len(values)}];", "};",
                   f"const std::size_t sizes{index}[]={{" + ",".join(f"sizeof(std::decay_t<decltype({v})>)" for v in values) + "};",
                   f"std::size_t argument_bytes{index}=0;"]
        source += [f"if (!add_payload(1,sizeof(std::decay_t<decltype({v})>),argument_bytes{index})) return fail(FORT_SCOPE_RESOURCE);"
                   for v in values]
        source += [f"if (!add_payload(unit_launches[{index}],sizeof(Arguments{index}),metadata_bytes) ||",
                   f"    !add_payload(unit_launches[{index}],argument_bytes{index},metadata_bytes)) return fail(FORT_SCOPE_RESOURCE);"]
    source += [f"if (metadata_bytes>{METADATA_BUDGET_BYTES}ULL) return fail(FORT_SCOPE_RESOURCE);",
               "std::vector<PlannedTask> tasks; tasks.reserve(expected);"]
    for index in range(len(launches)):
        source += [f"std::vector<Arguments{index}> values{index}; values{index}.reserve(unit_launches[{index}]);"]
    source += ["// No new plan allocation or resource acquisition is allowed after the first accepted task.",
               "status=fort_scope_scratch_acquire_v1(context,scratch_bytes,&lease);",
               "if (status!=FORT_SCOPE_OK) return fail(status);", "leased=true; state.scratch=static_cast<unsigned char*>(lease.device);",
               "fort_buffer_t handles[]={" + ",".join(f"field{i}" for i in range(fields)) + "};",
               f"status=fort_scope_device_reserve_many_v1(context,handles,{fields},devices);",
               "if (status!=FORT_SCOPE_OK) return decline(status);",
               "fort_scope_plan_binding effects[]={" + ",".join(f"{{field{i},bindings[{i}].access}}" for i in range(fields)) + "};"]
    for index, (kernel, values, _) in enumerate(launches):
        blocks = "count" if mapping == "cooperative" else "count/256+(count%256!=0)"
        source += [f"for (first=0;first<state.total[{index}];) {{",
                   f" count=std::min(state.batch,state.total[{index}]-first);",
                   f" values{index}.push_back(Arguments{index}{{" + ",".join(values) + ",{}});",
                   f" auto& args=values{index}.back();"]
        source += [f" args.pointers[{i}]=&args.value{i};" for i in range(len(values))]
        source += [f" const unsigned blocks=static_cast<unsigned>(std::min<std::size_t>({blocks},65535));",
                   f" const fort_scope_kernel_v1 kernel{{1,0,reinterpret_cast<const void*>({kernel}),{{blocks,1,1}},{{256,1,1}},0,args.pointers,{len(values)}}};",
                   f" tasks.push_back(PlannedTask{{{{kernel,effects,{fields}}},sizes{index},{keys[index]}ULL}});",
                   " first+=count;", "}"]
    source += ["if (tasks.size()!=expected) return decline(FORT_SCOPE_PLAN_MISMATCH);",
               "for (const auto& task:tasks) {",
               " status=fort_scope_graph_enqueue_v1(context,task.key,&task.task,task.sizes);",
               " if (status!=FORT_SCOPE_OK) return report->accepted ? fatal() : decline(status);",
               " report->accepted=1;", "}",
               "status=fort_scope_graph_flush_v1(context);",
               "if (status!=FORT_SCOPE_OK) return fatal();", "report->committed=1;",
               "status=fort_scope_scratch_release_v1(context,lease.token);",
               "if (status!=FORT_SCOPE_OK) return fatal();", "leased=false; return fail(FORT_SCOPE_OK);",
               "} catch (const std::bad_alloc&) { return report->accepted ? fatal() : decline(FORT_SCOPE_RESOURCE); }",
               "catch (...) { return report->accepted ? fatal() : decline(FORT_SCOPE_EXECUTION); }", "}", ""]
    iso_type = {"c_int": "integer(c_int)", "c_float": "real(c_float)", "c_double": "real(c_double)"}
    names = ["context", *[f"field{i}" for i in range(fields)], *[p.name for p in parameters], "report"]
    fortran = [f"module {module}", "use iso_c_binding", "implicit none", "type,bind(c) :: fort_tile_queue_report_v1",
               " integer(c_int32_t) :: version,status,accepted,committed", "end type", "interface",
               f"integer(c_int) function {entry}( &", " & " + ", &\n & ".join(names) + f') bind(c,name="{entry}")',
               "import", "integer(c_int64_t),value :: context",
               *[f"integer(c_int64_t),value :: field{i}" for i in range(fields)],
               *[f"{iso_type[p.kind]},intent(in) :: {p.name}" for p in parameters],
               "type(fort_tile_queue_report_v1),intent(out) :: report", "end function", "end interface", "end module", ""]
    cuda_source, fortran_source = "\n".join(source), "\n".join(fortran)
    record = {"version": 1, "kind": KIND, "team_identity": team.identity,
              "scalar_readability_identity": readable_scalars.identity if readable_scalars else None,
              "unit_proof_identities": [unit.identity for unit in team.units], "mapping": mapping,
              "entry": entry, "module": module, "parameters": [p.public() for p in parameters],
              "cuda_sha256": sha256(cuda_source.encode()).hexdigest(), "fortran_sha256": sha256(fortran_source.encode()).hexdigest(),
              "backend": "fort_scope_graph_enqueue_v1+fort_scope_graph_flush_v1", "code_keys": [str(k) for k in keys],
              "launch_abi": {"version": 1, "units": [abi for _, _, abi in launches]},
              "effects": {"source": "existing exact whole-team ordinary access envelopes on every kernel",
                          "atomic_commit": "one final existing queue flush; no intervening resource or host boundary",
                          "overwrites_authorized": False, "initialization_authorized": False},
              "report": {"type": "fort_tile_queue_report_v1", "version": 1,
                         "fields": ["version", "status", "accepted", "committed"],
                         "accepted": "first successful enqueue is irrevocable; native replay prohibited",
                         "committed": "whole group successful submission; not device completion"},
              "restriction": {"queue_enabled": True, "empty_queue_at_entry": True, "target_bytes": target_bytes,
                              "planned_launches": "sum ceil(unit_tiles/batch_tiles)",
                              "metadata_budget_bytes": METADATA_BUDGET_BYTES,
                              "metadata_budget": "checked caller plan, owned arguments and effect-coordinate payload; no graph truncation",
                              "fallback": "original complete team only on recoverable status before first acceptance",
                              "recoverable_statuses": [1, 4, 5, 7, 9],
                              "postaccept_failure": "poison and abandon; no native replay"},
              "scratch": "one lease spans all planned enqueues and final flush; existing same-stream ordered reuse",
              "canonical_storage": "current registered handles and typed layouts; queue owns argument bytes and effects; no pointer cache",
              "definitions_authorized": False, "placement_authorized": False,
              "pricing": "unpriced queued whole-team backend; no direct or queued-single cost reuse", "consumer_wiring": False}
    return TileTaskProvider(entry, module, cuda_source, fortran_source, tuple(team.fields), parameters,
                            json.dumps(record, sort_keys=True, separators=(",", ":")))
