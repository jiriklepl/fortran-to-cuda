"""Checked emission binding for one complete original team queue transaction."""
import json
from dataclasses import replace
from hashlib import sha256

from compiler.emission.cuda.queued_team_task_provider import KIND, METADATA_BUDGET_BYTES
from compiler.ir import CompilationError
from compiler.scopes.queued_source_binding import SOURCE_ROLE


def transport_record(units):
    return {
        "version": 1,
        "backend": "fort_scope_graph_enqueue_v1+fort_scope_graph_flush_v1",
        "report": "fort_tile_queue_report_v1",
        "accepted": "first successful enqueue is irrevocable; native replay prohibited",
        "committed": "whole group successful submission; not device completion",
        "recoverable_preaccept_statuses": [1, 4, 5, 7, 9],
        "queue": "enabled and empty at entry",
        "original_units": units,
        "planned_launches": "sum ceil(unit_tiles/batch_tiles)",
        "metadata_budget_bytes": METADATA_BUDGET_BYTES,
        "failure": "poison and abandon after first acceptance; no native replay",
        "pricing": "unavailable; no direct source_execute/tile or queued-single cost reuse",
    }


def checked_provider(provider):
    from compiler.emission.cuda.tile_task_provider import TileTaskProvider

    if type(provider) is not TileTaskProvider:
        raise CompilationError("queued whole-team binding requires its generated typed provider")
    metadata = provider.public()
    units = metadata.get("unit_proof_identities", ())
    if (metadata.get("version") != 1 or metadata.get("kind") != KIND
            or metadata.get("consumer_wiring") is not False or not units
            or metadata.get("backend") != transport_record(len(units))["backend"]
            or len(metadata.get("launch_abi", {}).get("units", ())) != len(units)
            or len(metadata.get("code_keys", ())) != len(units)):
        raise CompilationError("queued whole-team binding requires its complete standalone provider")
    metadata.update(version=2, consumer_wiring="checked_source_device_task_queue_team_v1")
    return replace(provider, _record=json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False))


def execution_record(proof, binding):
    record = {"version": 1, "kind": KIND, "source_proof": proof.public(),
              "source_proof_role": SOURCE_ROLE, "execution_binding": binding,
              "transport": transport_record(len(proof.team.units))}
    return {"identity": sha256(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
            **record}
