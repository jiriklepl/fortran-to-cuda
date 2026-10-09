"""Assemble packaged runtime units into the existing shared-header artifact."""

from hashlib import sha256
from importlib.resources import files


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
    experimental = [runtime.joinpath(name).read_text(encoding="utf-8")
                    .replace('#include "section_copy.hpp"', section_copy)
                    .replace('#include "staging.hpp"', staging)
                    for name in ("offload.hpp", "hybrid.hpp") if runtime.joinpath(name).is_file()]
    units += "\n#ifdef FORT_OFFLOAD_ENABLED\n" + "\n".join(experimental) + "\n#endif\n"
    return base.replace("// FORT_RUNTIME_UNITS", units)


def read_scoped_runtime() -> tuple[dict[str, str], dict]:
    """Publish the common compiled runtime independently of numerical entries."""
    runtime = files("compiler.runtime")
    outputs = {
        "scoped_runtime.h": runtime.joinpath("scoped_runtime.h").read_text(encoding="utf-8"),
        "scoped_entry.hpp": runtime.joinpath("scoped_entry.hpp").read_text(encoding="utf-8"),
        "section_copy.hpp": runtime.joinpath("section_copy.hpp").read_text(encoding="utf-8"),
        "staging.hpp": runtime.joinpath("staging.hpp").read_text(encoding="utf-8"),
        "scoped_regions.hpp": runtime.joinpath("scoped_regions.hpp").read_text(encoding="utf-8"),
        "scoped_planning.hpp": runtime.joinpath("scoped_planning.hpp").read_text(encoding="utf-8"),
        "scoped_runtime.cu": runtime.joinpath("scoped_runtime.cu").read_text(encoding="utf-8"),
        "fort_scoped_memory.f90": runtime.joinpath("scoped_memory.f90").read_text(encoding="utf-8"),
    }
    hashes = {name: sha256(content.encode()).hexdigest() for name, content in outputs.items()}
    identity = sha256("\n".join(f"{name}:{digest}" for name, digest in sorted(hashes.items())).encode()).hexdigest()
    manifest = {
        "schema_version": 1,
        "abi_version": 1,
        "runtime_id": identity,
        "link_once": True,
        "headers": ["scoped_runtime.h", "scoped_entry.hpp", "section_copy.hpp", "staging.hpp", "scoped_regions.hpp", "scoped_planning.hpp"],
        "planning_abi_version": 1,
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
            {"path": "scoped_runtime.cu", "language": "cuda", "standard": "c++17", "host_openmp": True},
            {"path": "fort_scoped_memory.f90", "language": "fortran", "module": "fort_scoped_memory"},
        ],
        "source_sha256": hashes,
        "devices_per_context": 1,
        "streams_per_context": 1,
        "scope_transfers": {
            "abi_version": 1,
            "configure": "fort_scope_set_transfers",
            "statistics": "fort_scope_transfer_stats_get_v1",
            "completed_statistics_event": "transfer_statistics",
            "default": "direct",
            "modes": {"direct": 0, "pinned": 1, "pipelined": 2, "auto": 3},
            "supported": ["direct", "pinned"],
            "pinned_execution": "synchronous exact-section packing, copy, completion and unpack",
            "staging_slots": 2,
            "staging_streams": "two reusable non-default streams in the process-wide pool",
            "pinned_budget_bytes": 64 * 1024 * 1024,
            "budget_scope": "active, allocating and cached staging for scoped and chunked/hybrid calls",
            "automatic_placement": "pinned costs unavailable; explicit pinned automatic placement remains native",
            "fallback_reasons": {"pipelined": "pipelined_not_available", "auto": "transfer_estimates_unavailable"},
        },
        "stream_ordered": True,
        "section_coordinates": "zero-based physical offsets, exclusive upper bounds",
        "host_storage": "borrowed stable contiguous whole arrays",
        "concurrent_access": "separate contexts with nonconflicting host storage",
        "rectangle_limit": 32,
        "intersection_limit": 1024,
        "capabilities": ["section_coherence", "partial_host_initialization", "source_definition_changes", "device_memory_budget", "coherent_runtime_placement", "ordered_definition_validation", "context_query_reuse", "segment_continuation", "pinned_section_transfers"],
    }
    return outputs, manifest
