"""Quote glue on the existing checked provider; no extra coherence engine."""

from __future__ import annotations

from importlib.resources import files

from compiler.emission.cuda.offload import _host_profile_compatibility
from compiler.emission.cuda.tile_costs import generate_tile_work_function, tile_work_support
from compiler.ir import CompilationError
from compiler.offload.codegen import profile_expression
from compiler.offload.source_task_pricing import SourceTaskPricing


def source_pricing_code(pricing, *, team, entry, launch_abi=()):
    if type(pricing) is not SourceTaskPricing or pricing.team is not team:
        raise CompilationError("source quote emission requires this team's registered pricing attachment")
    pricing.validate()
    record, profile = pricing.public(), pricing.profile()
    rank = len(pricing.work.units[0].axes)
    domains = [(f"state.lo0_{axis}", f"state.hi0_{axis}", f"state.step0_{axis}") for axis in range(rank)]
    work = generate_tile_work_function(pricing.work, name="source_work",
        parameters=("const tile_state& state", "std::size_t scratch_bytes"),
        domains=domains, batch_tiles="state.batch", scratch_lease_bytes="scratch_bytes")
    # The byte count is already guarded in the provider; pass it independently
    # rather than re-reading or narrowing any original source descriptor.
    offload = files("compiler.runtime").joinpath("offload.hpp").read_text(encoding="utf-8")
    allocation = files("compiler.runtime").joinpath("allocation.hpp").read_text(encoding="utf-8")
    storage = files("compiler.runtime").joinpath("storage.hpp").read_text(encoding="utf-8").replace(
        '#include "allocation.hpp"', allocation)
    timing = files("compiler.runtime").joinpath("timing.hpp").read_text(encoding="utf-8")
    support = [timing, storage, offload, "namespace offload=generated_kernels::offload;", tile_work_support(),
               *_host_profile_compatibility(record["cpu_threads"], record["precision_bits"])]
    functions = [work]
    live = pricing.native_live
    lines = ["if (!fort_runtime::numerical_environment_supported()) return fail(FORT_SCOPE_BOUNDARY);",
             "fort_runtime::HostFloatingEnvironment quote_environment;",
             "if (!quote_environment.valid()) return fail(FORT_SCOPE_BOUNDARY);",
             "fort_tile_cost::totals work{};",
             "if (!source_work(state,scratch_bytes,work) || work.launches!=expected) return fail(FORT_SCOPE_BOUNDARY);",
             "std::uint64_t memory=0;",
             "if (!fort_tile_cost::add(work.external_read_bytes,work.external_write_bytes,memory) ||",
             "    !fort_tile_cost::add(memory,work.scratch_read_bytes,memory) ||",
             "    !fort_tile_cost::add(memory,work.scratch_write_bytes,memory)) return fail(FORT_SCOPE_BOUNDARY);",
             "fort_scope_source_work_v1 request{};request.version=" + ("FORT_SCOPE_SOURCE_QUOTE_LIVE_ABI_VERSION;" if live else "FORT_SCOPE_SOURCE_QUOTE_ABI_VERSION;"),
             f"request.precision_bits={record['precision_bits']};",
             "request.flags=" + ("FORT_SCOPE_SOURCE_QUOTE_NATIVE_LIVE" if live else "FORT_SCOPE_SOURCE_QUOTE_NATIVE_CLOSE_DOMAIN") + "|FORT_SCOPE_SOURCE_QUOTE_SCRATCH_LEASE|FORT_SCOPE_SOURCE_QUOTE_WAIT_BEFORE_LEASE;",
             "request.expected_launches=expected;request.real_work_proxy=work.real_work_proxy;request.memory_bytes=memory;",
             "request.scratch_bytes=scratch_bytes;request.minimum_relative_saving=.05;",
             "request.compute.version=FORT_SCOPE_COMPUTE_ABI_VERSION;",
             "request.compute.known_backends=FORT_SCOPE_COMPUTE_NATIVE_FORTRAN|FORT_SCOPE_COMPUTE_GPU;",
             f"request.compute.native_fortran_seconds=std::max(double(work.real_work_proxy)/{record['rates']['cpu_flops_per_second']!r},double(memory)/{record['rates']['cpu_memory_bytes_per_second']!r})+{record['native_protocol']['seconds']!r};",
             f"request.compute.gpu_seconds=std::max(double(work.real_work_proxy)/{record['rates']['gpu_flops_per_second']!r},double(memory)/{record['rates']['gpu_memory_bytes_per_second']!r});",
             "fort_scope_plan_costs base{};base.version=FORT_SCOPE_PLANNING_ABI_VERSION;base.valid=1;",
             f"base.max_allocation_bytes={record['base_costs']['max_allocation_bytes']}ULL;"]
    lines += [f"base.{key}={value!r};" for key, value in record["base_costs"].items() if key != "max_allocation_bytes"]
    costs = record["source_costs"]
    lines += ["fort_scope_source_costs_v1 source{};source.version=FORT_SCOPE_SOURCE_QUOTE_ABI_VERSION;source.valid=1;",
              f"source.precision_bits={record['precision_bits']};source.protocol_id=FORT_SCOPE_SOURCE_PROTOCOL_ID;",
              f"source.max_bindings={costs['max_bindings']};source.max_launches={costs['max_launches']};source.max_scratch_bytes={costs['max_scratch_bytes']}ULL;"]
    lines += [f"source.{key}={value!r};" for key, value in costs["costs"].items()]
    if record.get("gpu_protocol_available") is False:
        # Observed but unqualified GPU costs are retained as diagnostics in
        # the bound profile. They cannot enable a placement or become zeros.
        lines += ["source.valid=0; // GPU source protocol failed independent coverage"]
    if record["version"] == 4:
        model = record["source_submit_model"]
        coefficients = model["coefficients"]
        if not launch_abi:
            raise CompilationError("structural source quote requires the exact emitted launch ABI")
        parameters = max(unit["parameters"] for unit in launch_abi)
        argument_bytes = max(unit["bytes"] for unit in launch_abi)
        lines += ["static_assert(sizeof(void*)==8 && sizeof(std::size_t)==8,\"source launch ABI differs\");",
                  f"if ({parameters}>{model['kernel_argument_words']} || {argument_bytes}>{model['kernel_argument_bytes']} ||",
                  f"    {len(team.fields)}>{model['max_bindings']} || expected>{model['max_launches']}) return fail(FORT_SCOPE_BOUNDARY);",
                  f"source.source_submit_seconds={model['coverage_factor']!r}*({coefficients['intercept_seconds']!r}+",
                  f"    {len(team.fields)}*{coefficients['binding_seconds']!r}+double(expected)*{coefficients['launch_seconds']!r});",
                  "if (!std::isfinite(source.source_submit_seconds) || source.source_submit_seconds<=0) return fail(FORT_SCOPE_BOUNDARY);"]
    if live:
        native = record["native_transaction_costs"]
        lines += ["fort_scope_source_host_costs_v1 native{};native.version=1;native.valid=1;",
                  "native.protocol_id=FORT_SCOPE_SOURCE_HOST_PROTOCOL_ID;",
                  f"native.max_bindings={native['max_bindings']};native.prepare_start_complete_seconds={native['prepare_start_complete_seconds']!r};"]
    schedule = record["native_protocol"]["c_schedule"]
    lines += [f"static const auto profile={profile_expression(profile)};", "#ifdef _OPENMP",
              "omp_sched_t schedule{};int chunk=0;omp_get_schedule(&schedule,&chunk);",
              f"const bool host_ok=scoped_host_compatible(profile)&&omp_get_max_threads()=={record['cpu_threads']}&&!omp_get_dynamic()&&",
              f"omp_get_level()==0&&omp_get_thread_limit()>={record['cpu_threads']}&&omp_get_max_active_levels()>0&&",
              f"static_cast<int>(schedule)=={schedule['raw_kind']}&&chunk=={schedule['chunk']};",
              "#else", "const bool host_ok=false;", "#endif",
              "fort_scope_source_quote_result_v1 quote{};",
              ("status=fort_scope_source_quote_v2(context,&call,&request,&base,&source,&native,host_ok?1:0,&quote);" if live else
               "status=fort_scope_source_quote_v1(context,&call,&request,&base,&source,host_ok?1:0,&quote);"),
              "if(status!=FORT_SCOPE_OK)return fail(status);",
              "const char* trace=std::getenv(\"FORT_OFFLOAD_TRACE\");",
              f'if(trace&&std::strcmp(trace,"0"))std::fprintf(stderr,"FORT_SOURCE_QUOTE entry={entry} available=%u gpu=%u reason=%u native=%.17g gpu_seconds=%.17g launches=%llu scratch=%zu\\n",quote.available,quote.gpu_preferred,quote.reason,quote.native_seconds,quote.gpu_seconds,static_cast<unsigned long long>(expected),scratch_bytes);',
              "if(!quote.available||!quote.gpu_preferred)return fail(FORT_SCOPE_BOUNDARY);",
              "int selected=-1,current=-1;status=fort_scope_device_get(context,&selected);if(status!=FORT_SCOPE_OK)return fail(status);",
              f"if(cudaGetDevice(&current)!=cudaSuccess||current!=selected||!offload::compatible(profile,{record['cpu_threads']},{record['precision_bits']}))return fail(FORT_SCOPE_BOUNDARY);"]
    return support, functions, lines
