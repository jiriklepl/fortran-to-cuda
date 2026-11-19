"""Closed integrity of one whole original team on one atomic queue flush."""
from hashlib import sha256

from .checked_execution_plan import _digest, _require, _sha256
from .source_task_record import (
    _IDENT,
    _SOURCE,
    _keys,
    _validate_parameters,
    _validate_source_authority,
)

KIND = "queued_original_tile_private_team"


def _transport(units):
    return {"version": 1, "backend": "fort_scope_graph_enqueue_v1+fort_scope_graph_flush_v1",
            "report": "fort_tile_queue_report_v1",
            "accepted": "first successful enqueue is irrevocable; native replay prohibited",
            "committed": "whole group successful submission; not device completion",
            "recoverable_preaccept_statuses": [1, 4, 5, 7, 9], "queue": "enabled and empty at entry",
            "original_units": units, "planned_launches": "sum ceil(unit_tiles/batch_tiles)",
            "metadata_budget_bytes": 16777216,
            "failure": "poison and abandon after first acceptance; no native replay",
            "pricing": "unavailable; no direct source_execute/tile or queued-single cost reuse"}


def validate_queued_team_source_task(record, member, procedure, inputs, resources, *, owner, manifest,
                                     capture_facts=None):
    _keys(record, {"identity", "version", "kind", "source_proof", "source_proof_role",
                   "execution_binding", "transport"})
    _require(type(record["version"]) is int and record["version"] == 1 and record["kind"] == KIND
             and _sha256(record["identity"])
             and record["identity"] == _digest({key: value for key, value in record.items() if key != "identity"})
             and record["source_proof_role"] == "original numerical work and scalar readability only",
             "queued whole-team source task requires its exact checked transport identity")
    source = record["source_proof"]
    _keys(source, _SOURCE - {"source_proof_identity", "execution_binding"})
    _require(_sha256(source["identity"])
             and source["identity"] == _digest({key: value for key, value in source.items() if key != "identity"}),
             "queued whole-team source task original numerical proof differs")
    # The unchanged source proof supplies numerical/scalar premises only.
    # No direct-runtime execution or transaction pricing is inferred from it.
    view = {**source, "source_proof_identity": source["identity"], "execution_binding": record["execution_binding"]}
    team, fields, binding = _validate_source_authority(view, member, procedure, inputs, resources,
                                                       owner=owner, capture_facts=capture_facts)
    units = len(team["unit_proof_identities"])
    _require(units > 0, "queued whole-team source task requires every original unit")
    transport = record["transport"]
    _keys(transport, set(_transport(units)))
    _require(transport == _transport(units)
             and all(type(transport[key]) is int for key in ("version", "original_units", "metadata_budget_bytes"))
             and all(type(status) is int for status in transport["recoverable_preaccept_statuses"]),
             "queued whole-team source task cannot infer completion, replay, truncation or old pricing")
    provider = binding["provider"]
    _keys(provider, {"entry", "module", "cuda_source_sha256", "fortran_source_sha256", "metadata"})
    metadata = provider["metadata"]
    _keys(metadata, {"version", "kind", "team_identity", "scalar_readability_identity", "unit_proof_identities",
                     "mapping", "entry", "module", "parameters", "cuda_sha256", "fortran_sha256", "backend",
                     "code_keys", "launch_abi", "effects", "report", "restriction", "scratch", "canonical_storage",
                     "definitions_authorized", "placement_authorized", "pricing", "consumer_wiring"})
    _require(type(metadata["version"]) is int and metadata["version"] == 2 and metadata["kind"] == KIND
             and metadata["consumer_wiring"] == "checked_source_device_task_queue_team_v1"
             and metadata["team_identity"] == team["identity"]
             and metadata["scalar_readability_identity"] == source["identity"]
             and metadata["unit_proof_identities"] == team["unit_proof_identities"]
             and type(metadata["mapping"]) is str and metadata["mapping"] in {"cooperative", "thread"},
             "queued whole-team source task provider differs from its complete checked team")
    suffix = team["identity"][:20] + ("_cta" if metadata["mapping"] == "cooperative" else "_thread") + "_batch_boxes_qt"
    _require(provider["entry"] == metadata["entry"] == "fort_tile_task_" + suffix
             and provider["module"] == metadata["module"] == "fort_tile_module_" + suffix
             and all(type(provider[key]) is str and _IDENT.fullmatch(provider[key]) for key in ("entry", "module"))
             and metadata["code_keys"] == [str(int(sha256((provider["entry"] + ":unit:" + str(index)).encode()).hexdigest()[:16], 16) or 1)
                                            for index in range(units)],
             "queued whole-team source task original ordered code identities differ")
    _require(metadata["backend"] == transport["backend"]
             and metadata["scratch"] == "one lease spans all planned enqueues and final flush; existing same-stream ordered reuse"
             and metadata["canonical_storage"] == "current registered handles and typed layouts; queue owns argument bytes and effects; no pointer cache"
             and metadata["definitions_authorized"] is metadata["placement_authorized"] is False
             and metadata["pricing"] == "unpriced queued whole-team backend; no direct or queued-single cost reuse",
             "queued whole-team source task cannot gain definitions, stale pointers, placement or old costs")
    _keys(metadata["effects"], {"source", "atomic_commit", "overwrites_authorized", "initialization_authorized"})
    _require(metadata["effects"] == {
        "source": "existing exact whole-team ordinary access envelopes on every kernel",
        "atomic_commit": "one final existing queue flush; no intervening resource or host boundary",
        "overwrites_authorized": False, "initialization_authorized": False}
        and metadata["effects"]["overwrites_authorized"] is metadata["effects"]["initialization_authorized"] is False,
        "queued whole-team source task cannot publish prefix initializedness or overwrite definitions")
    _keys(metadata["report"], {"type", "version", "fields", "accepted", "committed"})
    _require(metadata["report"] == {
        "type": "fort_tile_queue_report_v1", "version": 1,
        "fields": ["version", "status", "accepted", "committed"],
        "accepted": "first successful enqueue is irrevocable; native replay prohibited",
        "committed": "whole group successful submission; not device completion"}
        and type(metadata["report"]["version"]) is int,
        "queued whole-team source task report cannot alias started or completed execution")
    restriction = metadata["restriction"]
    _keys(restriction, {"queue_enabled", "empty_queue_at_entry", "target_bytes", "planned_launches",
                        "metadata_budget_bytes", "metadata_budget", "fallback", "recoverable_statuses", "postaccept_failure"})
    _require(restriction == {
        "queue_enabled": True, "empty_queue_at_entry": True, "target_bytes": 4194304,
        "planned_launches": "sum ceil(unit_tiles/batch_tiles)", "metadata_budget_bytes": 16777216,
        "metadata_budget": "checked caller plan, owned arguments and effect-coordinate payload; no graph truncation",
        "fallback": "original complete team only on recoverable status before first acceptance",
        "recoverable_statuses": [1, 4, 5, 7, 9], "postaccept_failure": "poison and abandon; no native replay"}
        and all(type(restriction[key]) is int for key in ("target_bytes", "metadata_budget_bytes"))
        and restriction["queue_enabled"] is restriction["empty_queue_at_entry"] is True
        and all(type(status) is int for status in restriction["recoverable_statuses"]),
        "queued whole-team source task requires complete upfront plans and preaccept-only recovery")
    abi = metadata["launch_abi"]
    _keys(abi, {"version", "units"})
    _require(type(abi["version"]) is int and abi["version"] == 1
             and type(abi["units"]) is list and len(abi["units"]) == units,
             "queued whole-team source task launch ABI must cover every original unit")
    for item in abi["units"]:
        _keys(item, {"parameters", "bytes"})
        _require(type(item["parameters"]) is type(item["bytes"]) is int and 1 <= item["parameters"] <= 4096
                 and 4 * item["parameters"] <= item["bytes"] <= 8 * item["parameters"] and item["bytes"] <= 32764,
                 "queued whole-team source task generated launch ABI differs")
    _validate_parameters(metadata, team, fields)
    _require(_sha256(provider["cuda_source_sha256"]) and _sha256(provider["fortran_source_sha256"])
             and metadata["cuda_sha256"] == provider["cuda_source_sha256"]
             and metadata["fortran_sha256"] == provider["fortran_source_sha256"],
             "queued whole-team source task artifact digest differs")
    artifacts = manifest.get("artifacts_sha256", {})
    _require(isinstance(artifacts, dict)
             and artifacts.get(provider["entry"] + ".cu") == provider["cuda_source_sha256"]
             and artifacts.get(provider["module"] + ".f90") == provider["fortran_source_sha256"],
             "queued whole-team source task compiled artifact inventory differs")
