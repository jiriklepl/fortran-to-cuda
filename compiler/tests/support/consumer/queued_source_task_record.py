"""Closed transport integrity for checked queued source work; no execution grant."""
from hashlib import sha256

from .checked_execution_plan import _digest, _require, _sha256
from .source_task_record import (
    _IDENT,
    _SOURCE,
    _keys,
    _validate_parameters,
    _validate_source_authority,
)

KIND = "queued_single_original_tile_private_team"


def _transport():
    return {"version": 1, "backend": "fort_scope_graph_enqueue_v1+fort_scope_graph_flush_v1",
            "report": "fort_tile_queue_report_v1",
            "accepted": "irrevocable queue ownership; native replay prohibited",
            "committed": "successful submission effect commit; not device completion",
            "recoverable_preaccept_statuses": [1, 4, 5, 7, 9], "queue": "enabled and empty at entry",
            "units": 1, "expected_launches": 1,
            "failure": "poison and abandon after acceptance; no native replay",
            "pricing": "unavailable; no direct source_execute/tile cost reuse"}


def validate_queued_source_task(record, member, procedure, inputs, resources, *, owner, manifest,
                                capture_facts=None):
    _keys(record, {"identity", "version", "kind", "source_proof", "source_proof_role",
                   "execution_binding", "transport"})
    _require(type(record["version"]) is int and record["version"] == 1 and record["kind"] == KIND
             and _sha256(record["identity"])
             and record["identity"] == _digest({key: value for key, value in record.items() if key != "identity"})
             and record["source_proof_role"] == "original numerical work and scalar readability only",
             "queued source task requires its exact checked transport identity")
    _keys(record["transport"], set(_transport()))
    _require(record["transport"] == _transport()
             and all(type(record["transport"][key]) is int for key in ("version", "units", "expected_launches"))
             and all(type(status) is int for status in record["transport"]["recoverable_preaccept_statuses"]),
             "queued source task cannot infer completion, replay or direct pricing")
    source = record["source_proof"]
    _keys(source, _SOURCE - {"source_proof_identity", "execution_binding"})
    _require(_sha256(source["identity"])
             and source["identity"] == _digest({key: value for key, value in source.items() if key != "identity"}),
             "queued source task original numerical proof differs")
    # This private validation view combines two separately checked records.
    # The direct source proof supplies only the unchanged original numerical
    # and scalar premise; no direct compiled provider or transaction is forged.
    view = {**source, "source_proof_identity": source["identity"], "execution_binding": record["execution_binding"]}
    team, fields, binding = _validate_source_authority(view, member, procedure, inputs, resources,
                                                       owner=owner, capture_facts=capture_facts)
    _require(len(team["unit_proof_identities"]) == 1, "queued source task requires one complete original unit")
    provider = binding["provider"]
    _keys(provider, {"entry", "module", "cuda_source_sha256", "fortran_source_sha256", "metadata"})
    metadata = provider["metadata"]
    _keys(metadata, {"version", "kind", "team_identity", "scalar_readability_identity", "unit_proof_identities",
                     "mapping", "entry", "module", "parameters", "cuda_sha256", "fortran_sha256", "backend",
                     "code_key", "launch_abi", "effects", "report", "restriction", "scratch", "canonical_storage",
                     "definitions_authorized", "placement_authorized", "pricing", "consumer_wiring"})
    _require(type(metadata["version"]) is int and metadata["version"] == 2 and metadata["kind"] == KIND
             and metadata["consumer_wiring"] == "checked_source_device_task_v1"
             and metadata["team_identity"] == team["identity"]
             and metadata["scalar_readability_identity"] == source["identity"]
             and metadata["unit_proof_identities"] == team["unit_proof_identities"]
             and type(metadata["mapping"]) is str and metadata["mapping"] in {"cooperative", "thread"},
             "queued source task provider differs from its original checked team")
    suffix = team["identity"][:20] + ("_cta" if metadata["mapping"] == "cooperative" else "_thread") + "_batch_boxes_q1"
    _require(provider["entry"] == metadata["entry"] == "fort_tile_task_" + suffix
             and provider["module"] == metadata["module"] == "fort_tile_module_" + suffix
             and all(type(provider[key]) is str and _IDENT.fullmatch(provider[key]) for key in ("entry", "module"))
             and type(metadata["code_key"]) is str
             and metadata["code_key"] == str(int(sha256(provider["entry"].encode()).hexdigest()[:16], 16) or 1),
             "queued source task stable code identity differs")
    _require(metadata["backend"] == record["transport"]["backend"]
             and metadata["scratch"] == "lease spans enqueue and flush; shared same-stream arena lifetime/reuse after release"
             and metadata["canonical_storage"] == "current registered handles and typed layouts; queue owns argument bytes and effect coordinates; no pointer cache"
             and metadata["definitions_authorized"] is metadata["placement_authorized"] is False
             and metadata["pricing"] == "unpriced; distinct queued backend; no source_execute/tile cost reuse",
             "queued source task cannot gain definitions, stale pointers, placement or direct costs")
    _keys(metadata["effects"], {"source", "atomic_commit", "overwrites_authorized", "initialization_authorized"})
    _require(metadata["effects"] == {
        "source": "existing exact whole-team ordinary access envelopes",
        "atomic_commit": "existing queue graph effect simulator; one kernel covers every original tile",
        "overwrites_authorized": False, "initialization_authorized": False}
        and metadata["effects"]["overwrites_authorized"] is metadata["effects"]["initialization_authorized"] is False,
        "queued source task cannot publish partial-team or overwrite definitions")
    _keys(metadata["report"], {"type", "version", "fields", "accepted", "committed"})
    _require(metadata["report"] == {
        "type": "fort_tile_queue_report_v1", "version": 1,
        "fields": ["version", "status", "accepted", "committed"],
        "accepted": "irrevocable queue ownership; native replay prohibited",
        "committed": "successful graph/direct submission effect commit; not device completion"}
        and type(metadata["report"]["version"]) is int,
        "queued source task report cannot alias started or completed execution")
    restriction = metadata["restriction"]
    _keys(restriction, {"units", "expected_launches", "queue_enabled", "empty_queue_at_entry", "target_bytes",
                        "fallback", "recoverable_statuses", "postaccept_failure"})
    _require(restriction == {
        "units": 1, "expected_launches": 1, "queue_enabled": True, "empty_queue_at_entry": True,
        "target_bytes": 4194304, "fallback": "original complete team only on recoverable status before acceptance",
        "recoverable_statuses": [1, 4, 5, 7, 9], "postaccept_failure": "poison and abandon; no native replay"}
        and all(type(restriction[key]) is int for key in ("units", "expected_launches", "target_bytes"))
        and restriction["queue_enabled"] is restriction["empty_queue_at_entry"] is True
        and all(type(status) is int for status in restriction["recoverable_statuses"]),
        "queued source task requires exact one-launch preaccept recovery")
    abi = metadata["launch_abi"]
    _keys(abi, {"parameters", "bytes"})
    _require(type(abi["parameters"]) is type(abi["bytes"]) is int and 1 <= abi["parameters"] <= 4096
             and 4 * abi["parameters"] <= abi["bytes"] <= 8 * abi["parameters"] and abi["bytes"] <= 32764,
             "queued source task generated launch ABI differs")
    _validate_parameters(metadata, team, fields)
    _require(_sha256(provider["cuda_source_sha256"]) and _sha256(provider["fortran_source_sha256"])
             and metadata["cuda_sha256"] == provider["cuda_source_sha256"]
             and metadata["fortran_sha256"] == provider["fortran_source_sha256"],
             "queued source task artifact digest differs")
    artifacts = manifest.get("artifacts_sha256", {})
    _require(isinstance(artifacts, dict)
             and artifacts.get(provider["entry"] + ".cu") == provider["cuda_source_sha256"]
             and artifacts.get(provider["module"] + ".f90") == provider["fortran_source_sha256"],
             "queued source task compiled artifact inventory differs")
