"""Check the transport of unchanged original CPU may-access transactions."""
import hashlib
import json

from .checked_execution_plan import _require, _sha256, _text


def validate_original_host_accesses(procedure, unit, manifest):
    runtime = (manifest.get("runtime") or {}).get("task_acquisition", {})
    for operation in unit["operations"].get("native_operations", ()):
        contract = operation.get("original_host_access")
        complete_call = operation.get("kind") == "complete original native CALL"
        complete_group = operation.get("kind") == "complete original native group"
        if contract is None:
            _require(not (runtime.get("abi_version") == 2 and (complete_call or complete_group)),
                     "original host CALL is missing its acquisition contract")
            continue
        _require(isinstance(contract, dict) and set(contract) == {
            "version", "begin", "plan_kind", "effects", "definitions", "proof"},
            "original host acquisition schema differs")
        _require(type(contract["version"]) is int and contract["version"] == 1
                 and type(contract["plan_kind"]) is int and contract["plan_kind"] == 6
                 and contract["begin"] == "fort_scope_original_host_begin_v1"
                 and contract["effects"] == "may-read and may-write"
                 and contract["definitions"] == "must-overwrites only",
                 "original host acquisition semantics differ")
        _require(runtime.get("abi_version") == 2 and runtime.get("original_host_begin") == contract["begin"]
                 and runtime.get("original_host_plan_kind") == contract["plan_kind"],
                 "original host acquisition runtime ABI differs")
        _require((complete_call or complete_group) and operation.get("preserves_original_source") is True
                 and operation.get("completion", {}).get("available") is True
                 and operation.get("completion", {}).get("internal_cuts_authorized") is False,
                 "original host acquisition requires one complete unchanged CALL")
        proof = contract["proof"]
        false_fields = {"alias_authorized", "capture_authorized", "definitions_authorized", "motion_authorized",
                        "internal_cuts_authorized"}
        _require(isinstance(proof, dict) and set(proof) == false_fields | {
            "schema_version", "identity", "procedure", "reachability_identity", "storage_identity", "resources", "execution",
            "selection_kind", "completion_identity", "sections_identity"},
            "original host CALL proof schema differs")
        _require(type(proof["schema_version"]) is int and proof["schema_version"] == 2
                 and all(proof[key] is False for key in false_fields)
                 and all(_sha256(proof[key]) for key in ("identity", "reachability_identity", "storage_identity"))
                 and proof["identity"] == operation.get("demand_identity")
                 and proof["procedure"] == procedure["procedure"]
                 and proof["selection_kind"] == ("call" if complete_call else "joined_group")
                 and (proof["completion_identity"] is None if complete_call else _sha256(proof["completion_identity"]))
                 and proof["sections_identity"] == hashlib.sha256(json.dumps(operation.get("sections"),
                     sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
                 and proof["execution"] == "unchanged original operation after coherent access and physical alias registration",
                 "original host CALL proof authority differs")
        sections = operation.get("sections", {})
        _require(isinstance(sections.get("available"), bool)
                 and (complete_group or sections["available"] is False)
                 and all(not item.get("overwrites") for item in sections.get("resources", [])),
                 "original host may-access cannot create unproved definitions")
        selected = operation.get("selected_original_nodes")
        _require(isinstance(selected, list) and bool(selected) and (not complete_call or len(selected) == 1)
                 and len(selected) == len(set(selected)) and all(_text(item) for item in selected)
                 and all(any(item == node or item.startswith((node + "/", node + ":"))
                             for node in unit["selected_node_ids"]) for item in selected)
                 and operation.get("structured_summary_identity") == procedure["structured_summary_identity"],
                 "original host CALL source selection differs")
        _require(isinstance(proof["resources"], list), "original host CALL resources differ")
        effects, roots = {}, set()
        for resource in proof["resources"]:
            _require(isinstance(resource, dict) and set(resource) == {"resource", "role", "actions"},
                     "original host CALL resource schema differs")
            root, role, actions = resource["resource"], resource["role"], resource["actions"]
            _require(_text(root) and root not in roots and role in {"managed", "constant", "reservation"}
                     and isinstance(actions, list) and all(_text(action) for action in actions)
                     and actions == sorted(set(actions))
                     and set(actions) <= {"read", "write", "undefine", "association", "descriptor"},
                     "original host CALL resource effects differ")
            roots.add(root)
            if role == "managed":
                _require("undefine" not in actions, "original host CALL cannot grant definition-kill authority")
                payload = sorted(set(actions) & {"read", "write"})
                if payload:
                    effects[root] = payload
            elif role == "constant":
                _require(set(actions) <= {"read", "association", "descriptor"}
                         and root in operation.get("source_constants", {}),
                         "original host CALL constant authority differs")
            elif set(actions) - {"association"}:
                _require(root in operation.get("host_metadata", {}),
                         "original host CALL storage reservation is missing")
        _require(effects == operation.get("resource_effects"),
                 "original host CALL managed effects differ from its complete proof")

