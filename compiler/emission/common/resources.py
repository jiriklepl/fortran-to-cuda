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
    units += "\n" + runtime.joinpath("floating_environment.hpp").read_text(encoding="utf-8")
    offload = runtime.joinpath("offload.hpp").read_text(encoding="utf-8")
    units += "\n#ifdef FORT_OFFLOAD_ENABLED\n" + offload + "\n#endif\n"
    return numerical_source_prologue() + base.replace("// FORT_RUNTIME_UNITS", units)


def read_scoped_runtime() -> tuple[dict[str, str], dict]:
    """Publish the common compiled runtime independently of numerical entries."""
    runtime = files("compiler.runtime")
    outputs = {
        "scoped_runtime.h": runtime.joinpath("scoped_runtime.h").read_text(encoding="utf-8"),
        "floating_environment.hpp": runtime.joinpath("floating_environment.hpp").read_text(encoding="utf-8"),
        "scoped_entry.hpp": runtime.joinpath("scoped_entry.hpp").read_text(encoding="utf-8"),
        "section_copy.hpp": runtime.joinpath("section_copy.hpp").read_text(encoding="utf-8"),
        "scoped_cuda_resources.hpp": runtime.joinpath("scoped_cuda_resources.hpp").read_text(encoding="utf-8"),
        "scoped_graphs.hpp": runtime.joinpath("scoped_graphs.hpp").read_text(encoding="utf-8"),
        "scoped_regions.hpp": runtime.joinpath("scoped_regions.hpp").read_text(encoding="utf-8"),
        "scoped_runtime.cu": numerical_source_prologue() + runtime.joinpath("scoped_runtime.cu").read_text(encoding="utf-8"),
        "fort_scoped_memory.f90": runtime.joinpath("scoped_memory.f90").read_text(encoding="utf-8"),
    }
    hashes = {name: sha256(content.encode()).hexdigest() for name, content in outputs.items()}
    identity = sha256("\n".join(f"{name}:{digest}" for name, digest in sorted(hashes.items())).encode()).hexdigest()
    manifest = {
        "schema_version": 1,
        "abi_version": 1,
        "runtime_id": identity,
        "residency_domains": {
            "abi_version": 1,
            "ownership": "caller-owned canonical storage; one active borrowed context per domain",
            "create": "fort_scope_domain_create_v1", "borrow": "fort_scope_create_in_domain_v1",
            "close": "fort_scope_domain_close_v1", "abandon": "fort_scope_domain_abandon_v1",
            "checked_owner_create": "fort_scope_resident_create_v1",
            "checked_owner_retire": "fort_scope_resident_close_v1",
            "private_contexts": "retain original publication and cleanup semantics",
            "borrowed_close": "detach execution; preserve replicas until domain retirement",
            "statistics": "domain-aggregate physical counters for borrowed contexts",
        },
        "external_device_tasks": {
            "abi_version": 1, "execute": "fort_scope_external_execute_v1",
            "access": "typed canonical buffer views and exact read/write/overwrite regions",
            "workspace": "domain reservation leases include provider-owned device allocations",
            "ordering": "domain stream; committed effects retain completion dependencies",
            "failure": "explicit rejection before submission may fall back; attempted or unknown work poisons",
        },
        "runtime_provenance": {
            "schema_version": 1, "activation": "FORT_RUNTIME_TRACE=1",
            "set": "fort_scope_trace_set_v1", "restore": "fort_scope_trace_restore_v1",
            "identity": "full compiler manifest SHA256; unknown fields are explicit",
            "events": "actual API submissions and coherence commits; no physical execution timestamps",
            "initial_context_creation": "unknown until the source coordinator binds its owner",
            "execution_error_attribution": "unavailable; existing numerical errors are unchanged",
            "disabled": "no context lookup, allocations, coherence changes or numerical errors",
        },
        "link_once": True,
        "numerical_contract": numerical_build_contract(),
        "headers": sorted(name for name in outputs if name.endswith((".h", ".hpp", ".cuh"))),
        "task_acquisition": {
            "abi_version": 2, "begin": "fort_scope_access_begin_v1",
            "original_host_begin": "fort_scope_original_host_begin_v1",
            "original_host_effects": "synchronize tracked may-accesses; only must-overwrites add definitions",
            "validation": "all resource effects before transfers or numerical execution",
            "host_completion": "one completion for task downloads; independent and disjoint accesses do not wait",
            "upload_lifetime": "exact pending DMA read sections; completion events precede unrelated numerical work",
            "completion_statistics": "fort_scope_completion_stats_get_v1",
        },
        "specification_entry": {
            "abi_version": 3, "observe": "fort_scope_observe_v1", "fallback": "fort_scope_observe_all_v1",
            "handle_check": "fort_scope_buffer_valid_v1",
            "regions": "tracked physical metadata regions without definition authority; whole domain if mappings or live handles are unavailable",
            "replicas": "read-only publication preserves current device replicas and generations",
            "failure": "stop before original entry; never replay",
        },
        "graph_execution": {
            "abi_version": 1, "execute": "fort_scope_graph_execute_v1",
            "enqueue": "fort_scope_graph_enqueue_v1", "flush": "fort_scope_graph_flush_v1",
            "queue_statistics": "fort_scope_graph_queue_stats_get_v1",
            "queue_lifetime": "owned kernel argument bytes and effects span generated helper calls",
            "host_boundaries": "drain conflicting tasks and their dependencies; retain independent queued work",
            "queued_definitions": "pending task writes contribute virtual definitions without premature physical commits",
            "reserve": "fort_scope_device_reserve_v1", "reserve_many": "fort_scope_device_reserve_many_v1",
            "buffer_valid": "fort_scope_buffer_valid_v1", "statistics": "fort_scope_graph_stats_get_v1",
            "residency_statistics": "fort_scope_residency_stats_get_v1",
            "budget_eviction": "whole-task reservation pins its footprint; pressure flushes queued arguments before exact dirty publication and LRU retirement",
            "effects": "ordered original task effects with one atomic coherence commit",
            "residency": "canonical context buffers shared across submissions and iterations",
            "replay": "context-owned CUDA executable graph; current kernel parameters updated before launch",
            "submission_policy": "first matching topology uses direct launches; second builds a graph; later reaches replay; topology changes restart warmup",
            "warmup_evidence": "graph_warmup_direct plus actual direct_submissions/direct_kernels; never graph_resource_fallbacks",
            "observation": "fort_scope_observe_v1", "discard": "fort_scope_discard_v1",
            "retire": "fort_scope_retire_v1", "fallback": "before numerical submission only; graph resource failure uses ordered direct launches",
            "ablation": {"queue": "FORT_SCOPE_GRAPH_QUEUE=0", "cuda_graphs": "FORT_SCOPE_CUDA_GRAPHS=0"},
            "definition_kill": "semantic coverage changes retain physical freshness; explicit discard retires dead values",
        },
        "idle_execution_resources": {
            "schema_version": 1,
            "limit_per_device": 1,
            "retained_device_bytes_limit": 64 * 1024 * 1024,
            "contents": "completed stream and allocation-free CUDA memory pool only",
            "freshness": "all identities, definitions and coherence are invocation-local",
            "trace_reuse": "FORT_SCOPED initialize_reuse; bytes is retained pool capacity",
            "legacy_allocations": "logical field allocations, including reused physical pages",
            "failure": "poisoned or incomplete resources are never cached",
            "close_completion": "stream-ordered publication and field/scratch retirement share one wait",
        },
        "numerical_environment": {
            "check": "fort_scope_numerical_environment_supported",
            "requires": "original nontrapping round-to-nearest thread; native IEEE guards stay in place",
            "runtime_setup": "preserve caller flags, rounding and trap mask",
        },
        "sources": [
            {"path": "scoped_runtime.cu", "language": "cuda", "standard": "c++17", "host_openmp": True,
             "numerical_contract": numerical_build_contract()},
            {"path": "fort_scoped_memory.f90", "language": "fortran", "module": "fort_scoped_memory"},
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
            "compatibility": "existing field statistics and runtime interfaces remain unchanged",
        },
        "stream_ordered": True,
        "section_coordinates": "zero-based physical offsets, exclusive upper bounds",
        "host_storage": "borrowed stable contiguous whole arrays",
        "concurrent_access": "separate contexts with nonconflicting host storage",
        "rectangle_limit": 128,
        "intersection_limit": 8192,
        "capabilities": ["section_coherence", "partial_host_initialization", "source_definition_changes", "device_memory_budget", "context_scratch_leases"],
    }
    return outputs, manifest
