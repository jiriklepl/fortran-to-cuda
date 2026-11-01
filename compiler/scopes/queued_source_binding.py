"""Checked binding for one original source kernel on the existing queue.

This is an emission contract, not a new source authority or placement policy.
"""
import json
from dataclasses import replace
from hashlib import sha256

from compiler.ir import CompilationError

KIND = "queued_single_original_tile_private_team"
SOURCE_ROLE = "original numerical work and scalar readability only"


def transport_record():
    return {
        "version": 1,
        "backend": "fort_scope_graph_enqueue_v1+fort_scope_graph_flush_v1",
        "report": "fort_tile_queue_report_v1",
        "accepted": "irrevocable queue ownership; native replay prohibited",
        "committed": "successful submission effect commit; not device completion",
        "recoverable_preaccept_statuses": [1, 4, 5, 7, 9],
        "queue": "enabled and empty at entry",
        "units": 1,
        "expected_launches": 1,
        "failure": "poison and abandon after acceptance; no native replay",
        "pricing": "unavailable; no direct source_execute/tile cost reuse",
    }


def checked_provider(provider):
    """Bind the unchanged explicit artifact to this distinct checked consumer."""
    from compiler.emission.cuda.tile_task_provider import TileTaskProvider

    if type(provider) is not TileTaskProvider:
        raise CompilationError("queued source binding requires its generated typed provider")
    metadata = provider.public()
    if (metadata.get("version") != 1 or metadata.get("kind") != KIND
            or metadata.get("consumer_wiring") is not False
            or metadata.get("backend") != transport_record()["backend"]
            or metadata.get("restriction", {}).get("units") != 1
            or len(metadata.get("unit_proof_identities", ())) != 1):
        raise CompilationError("queued source binding requires its standalone single-unit provider")
    metadata.update(version=2, consumer_wiring="checked_source_device_task_v1")
    return replace(provider, _record=json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False))


def execution_record(proof, binding):
    record = {"version": 1, "kind": KIND, "source_proof": proof.public(),
              "source_proof_role": SOURCE_ROLE, "execution_binding": binding,
              "transport": transport_record()}
    return {"identity": sha256(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
            **record}


def report_lines(member):
    """Classify every accepted/malformed/fatal result before native fallback."""
    return [
        "fort_source_fatal = fort_source_report%version /= 1 .or. &",
        "fort_source_report%status /= fort_source_status .or. &",
        "fort_source_report%accepted < 0 .or. fort_source_report%accepted > 1 .or. &",
        "fort_source_report%committed < 0 .or. fort_source_report%committed > 1",
        "if (.not. fort_source_fatal) then",
        "if (fort_source_status == FORT_SCOPE_OK) then",
        "fort_source_fatal = fort_source_report%accepted /= 1 .or. fort_source_report%committed /= 1",
        "if (.not. fort_source_fatal) fort_source_done = .true.",
        "else",
        "fort_source_fatal = fort_source_report%accepted /= 0 .or. fort_source_report%committed /= 0 .or. &",
        "(fort_source_status /= FORT_SCOPE_ARGUMENT .and. fort_source_status /= FORT_SCOPE_RESOURCE .and. &",
        "fort_source_status /= FORT_SCOPE_BOUNDARY .and. fort_source_status /= FORT_SCOPE_UNINITIALIZED .and. &",
        "fort_source_status /= FORT_SCOPE_PLAN_MISMATCH)",
        "endif",
        "endif",
        "if (fort_source_fatal) then",
        "fort_source_error_message = 'queued source transaction failed; native replay prohibited'//c_null_char",
        f"fort_source_cleanup = fort_scope_execution_error({member.context}, c_loc(fort_source_error_message))",
        f"fort_source_cleanup = fort_scope_abandon({member.context})",
        "if (fort_source_cleanup /= FORT_SCOPE_OK) error stop 'queued source abandonment failed; replay prohibited'",
        f"{member.context} = 0_c_int64_t",
        "error stop 'queued source transaction failed; native replay prohibited'",
        "endif",
    ]
