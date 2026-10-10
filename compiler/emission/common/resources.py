"""Assemble packaged runtime units into the existing shared-header artifact."""

from hashlib import sha256
from importlib.resources import files

from compiler.numerical_contract import numerical_build_contract, numerical_source_prologue


def read_common_header() -> str:
    """Keep a single distributable support header with separately maintained units."""
    base = files(__package__).joinpath("templates").joinpath("common_functions.cuh").read_text(encoding="utf-8")
    runtime = files("compiler.runtime")
    storage = runtime.joinpath("storage.hpp").read_text(encoding="utf-8")
    allocation = runtime.joinpath("allocation.hpp").read_text(encoding="utf-8")
    storage = storage.replace('#include "allocation.hpp"', allocation)
    units = "\n".join(
        [runtime.joinpath(name).read_text(encoding="utf-8") for name in ("numeric.hpp", "timing.hpp")] + [storage]
    )
    section_copy = runtime.joinpath("section_copy.hpp").read_text(encoding="utf-8")
    staging = runtime.joinpath("staging.hpp").read_text(encoding="utf-8")
    floating_environment = runtime.joinpath("floating_environment.hpp").read_text(encoding="utf-8")
    experimental = [runtime.joinpath(name).read_text(encoding="utf-8")
                    .replace('#include "section_copy.hpp"', section_copy)
                    .replace('#include "staging.hpp"', staging)
                    .replace('#include "floating_environment.hpp"', floating_environment)
                    for name in ("offload.hpp", "hybrid.hpp") if runtime.joinpath(name).is_file()]
    units += "\n" + floating_environment
    units += "\n#ifdef FORT_OFFLOAD_ENABLED\n" + "\n".join(experimental) + "\n#endif\n"
    return numerical_source_prologue() + base.replace("// FORT_RUNTIME_UNITS", units)


def read_scoped_runtime() -> tuple[dict[str, str], dict]:
    """Publish the common compiled runtime independently of numerical entries."""
    runtime = files("compiler.runtime")
    outputs = {
        "scoped_runtime.h": runtime.joinpath("scoped_runtime.h").read_text(encoding="utf-8"),
        "floating_environment.hpp": runtime.joinpath("floating_environment.hpp").read_text(encoding="utf-8"),
        "scoped_entry.hpp": runtime.joinpath("scoped_entry.hpp").read_text(encoding="utf-8"),
        "scoped_team_observer.hpp": runtime.joinpath("scoped_team_observer.hpp").read_text(encoding="utf-8"),
        "fort_scoped_team_observer.f90": runtime.joinpath("scoped_team_observer.f90").read_text(encoding="utf-8"),
        "view_entry.hpp": runtime.joinpath("view_entry.hpp").read_text(encoding="utf-8"),
        "section_copy.hpp": runtime.joinpath("section_copy.hpp").read_text(encoding="utf-8"),
        "staging.hpp": runtime.joinpath("staging.hpp").read_text(encoding="utf-8"),
        "scoped_regions.hpp": runtime.joinpath("scoped_regions.hpp").read_text(encoding="utf-8"),
        "scoped_planning.hpp": runtime.joinpath("scoped_planning.hpp").read_text(encoding="utf-8"),
        "scoped_runtime.cu": numerical_source_prologue() + runtime.joinpath("scoped_runtime.cu").read_text(encoding="utf-8"),
        "fort_scoped_memory.f90": runtime.joinpath("scoped_memory.f90").read_text(encoding="utf-8"),
    }
    hashes = {name: sha256(content.encode()).hexdigest() for name, content in outputs.items()}
    identity = sha256("\n".join(f"{name}:{digest}" for name, digest in sorted(hashes.items())).encode()).hexdigest()
    manifest = {
        "schema_version": 1,
        "abi_version": 1,
        "runtime_id": identity,
        "runtime_provenance": {
            "schema_version": 1, "activation": "FORT_RUNTIME_TRACE=1",
            "set": "fort_scope_trace_set_v1", "restore": "fort_scope_trace_restore_v1",
            "identity": "full compiler manifest SHA256; unknown fields are explicit",
            "events": "actual API submissions and coherence commits; no physical execution timestamps",
            "initial_context_creation": "unknown until the source coordinator binds its owner",
            "batch_worker_attribution": "unavailable; batch callbacks cannot call context APIs",
            "execution_error_attribution": "unavailable; existing numerical errors are unchanged",
            "disabled": "no context lookup, allocations, coherence changes or numerical errors",
        },
        "link_once": True,
        "numerical_contract": numerical_build_contract(),
        "headers": ["scoped_runtime.h", "floating_environment.hpp", "scoped_entry.hpp", "scoped_team_observer.hpp", "view_entry.hpp", "section_copy.hpp", "staging.hpp", "scoped_regions.hpp", "scoped_planning.hpp"],
        "borrowed_views": {
            "abi_version": 1, "validate": "fort_scope_view_get_v1",
            "partial_definition": "fort_scope_forget_sections_v1",
            "planning_definition": "fort_scope_plan_forget_sections_v1",
            "coordinates": "canonical root physical origins and pitches; child logical bounds and extents",
            "allocation": "borrow existing full-layout root; no independent registration or device buffer",
            "aliases": "bounded exact read unions; writable formal views must be disjoint",
            "bounds": "checked original descriptor generation and existing INTEGER ABI",
            "rank_reduced": {
                "abi_version": 2, "validate": "fort_scope_view_get_v2",
                "coordinates": "root-rank origins; logical-rank extents and distinct retained-axis mapping",
                "footprints": "exact rectangular planes; omitted axes select one original coordinate",
                "strides": "original root pitches; initial actual views retain unit logical strides",
                "compatibility": "version 1 interfaces retained",
            },
        },
        "planning_abi_version": 1,
        "planning_numerical_costs": {
            "abi_version": 2, "record": "fort_scope_plan_add_costs_v2",
            "basis": "static scalar intrinsic counts and compatible independent offline holdouts",
            "missing_or_incompatible": "unknown estimate; automatic placement remains native",
        },
        "planning_compute_costs": {
            "abi_version": 1, "record": "fort_scope_plan_add_compute_costs_v3",
            "payload": "fort_scope_compute_costs_v1",
            "backends": {"native_fortran": 1, "generated_cpu": 2, "gpu": 4},
            "basis": "complete compute seconds; original Fortran and generated workers calibrated independently",
            "native_operation_cpu": "actual coordinated host execution including candidate-only preparation",
            "separate_costs": "transfers, allocation, launches, synchronization and coordination",
            "applicability": "original compiler semantics, fixed CPU placement, workload class and runtime item range",
            "missing_or_incompatible": "definition validation retained; automatic estimate unavailable",
            "continuation": "coherent host counterfactual; no replay of earlier numerical work",
            "compatibility": "v1/v2 planning interfaces retain their original meaning",
        },
        "numerical_environment": {
            "check": "fort_scope_numerical_environment_supported",
            "requires": "original nontrapping round-to-nearest thread; native IEEE guards stay in place",
            "runtime_setup": "preserve caller flags, rounding and trap mask",
        },
        "collective_automatic_calibration": {
            "abi_version": 1, "protocol_id": "0x4654434f4c4c0001",
            "configure": "fort_scope_set_team_costs_v1", "ready": "fort_scope_team_costs_ready_v1",
            "entry_marker": "fort_scope_plan_team_entry_v1",
            "native_call_marker": "fort_scope_plan_team_native_call_v1",
            "requirement": "offline persistent-team rates and exact emitted coordination, fixed team and toolchain",
            "missing_or_incompatible": "uniform unchanged native execution before numerical work",
        },
        "planning_continuation": {
            "abi_version": 2,
            "reset": "fort_scope_plan_reset_mode",
            "report": "fort_scope_plan_report_v2",
            "endpoints": {"complete": 0, "continue": 1},
            "costs": "incremental execution plus change in eventual publication/teardown liability",
            "terminal_operations": "hypothetical until owner close; excluded from executed segment counters",
            "native_selection": "execute reached segment with coherence hooks; retain owning context",
        },
        "sources": [
            {"path": "scoped_runtime.cu", "language": "cuda", "standard": "c++17", "host_openmp": True,
             "numerical_contract": numerical_build_contract()},
            {"path": "fort_scoped_memory.f90", "language": "fortran", "module": "fort_scoped_memory"},
            {"path": "fort_scoped_team_observer.f90", "language": "fortran", "module": "fort_scoped_team_observer"},
        ],
        "source_sha256": hashes,
        "devices_per_context": 1,
        "streams_per_context": 1,
        "scratch": {
            "abi_version": 1,
            "acquire": "fort_scope_scratch_acquire_v1",
            "release": "fort_scope_scratch_release_v1",
            "statistics": "fort_scope_scratch_stats_get_v1",
            "leases_per_context": 1,
            "storage": "cached context-owned device arena; no host binding or publication",
            "ordering": "existing context stream; growth waits for earlier use before replacement",
            "budget": "cached capacity and registered field payload share the context device budget",
            "zero_bytes": "null pointer with a release-required token; no CUDA initialization",
            "automatic_costs": "unavailable until numerical integration prices scratch lifetime costs",
            "compatibility": "existing field statistics and runtime interfaces remain unchanged",
        },
        "scope_transfers": {
            "abi_version": 1,
            "configure": "fort_scope_set_transfers",
            "statistics": "fort_scope_transfer_stats_get_v1",
            "completed_statistics_event": "transfer_statistics",
            "default": "direct",
            "modes": {"direct": 0, "pinned": 1, "pipelined": 2, "auto": 3},
            "supported": ["direct", "pinned", "pipelined", "auto"],
            "pinned_execution": "synchronous exact-section packing, copy, completion and unpack",
            "staging_slots": 2,
            "staging_streams": "two reusable non-default streams in the process-wide pool",
            "pinned_budget_bytes": 64 * 1024 * 1024,
            "budget_scope": "active, allocating and cached staging for scoped and chunked/hybrid calls",
            "automatic_placement": "calibrated pinned synchronous pricing; pipelined/auto retain direct placement baseline and select transfers for approved GPU subchains",
            "fallback_reasons": {"missing_calibration": "transfer_estimates_unavailable",
                                 "unproved_partition": "unsupported_chain",
                                 "resource_failure": "pre_execution_resource_failure"},
            "batch": {
                "abi_version": 1,
                "execute": "fort_scope_batch_execute_v1",
                "configure_costs": "fort_scope_set_transfer_costs_v1",
                "report": "fort_scope_batch_report_get_v1",
                "completed_statistics_event": "batch_statistics",
                "statistics_version": 1,
                "preview": "compatible=-1; metadata only; no callback, CUDA, schedule consumption or coherence changes",
                "callback": "complete source-proven GPU subchain; full-layout views and original coordinates; no reentrant context calls",
                "concurrency": "one active batch per context; two non-default staging streams",
                "completion": "both slots complete before return or native boundary",
                "batch_payload_bytes": [256 * 1024, 1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024],
                "costs": "offline staging/row packing/events/preparation plus compute, launch and pinned transfer rates; fill/drain included",
                "cost_reporting": "execution and terminal delta separate; unrepriced complete-owner estimate is unavailable",
                "failure": "unapplied fallback preserves schedule; started execution errors poison the context and prohibit replay",
            },
        },
        "stream_ordered": True,
        "section_coordinates": "zero-based physical offsets, exclusive upper bounds",
        "host_storage": "borrowed stable contiguous whole arrays",
        "concurrent_access": "separate contexts with nonconflicting host storage",
        "rectangle_limit": 32,
        "intersection_limit": 1024,
        "capabilities": ["section_coherence", "partial_host_initialization", "source_definition_changes", "device_memory_budget", "coherent_runtime_placement", "ordered_definition_validation", "context_query_reuse", "segment_continuation", "pinned_section_transfers", "pipelined_subchain_transfers", "context_scratch_leases"],
    }
    return outputs, manifest
