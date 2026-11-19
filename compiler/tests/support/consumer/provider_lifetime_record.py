"""Pure structural reader for compiler-issued provider-only lifetimes.

Original source/call classification remains the compiler issuer's authority.
This record never grants field retention, incoming definitions or native effects.
"""
from hashlib import sha256

from .checked_execution_plan import _ORIGINAL_CALL_FIELDS, _digest, _require, _sha256, _source_inputs_identity, _text

_POLICY = {
    "normal_exit": "single original END SUBROUTINE or END PROGRAM; explicit RETURN, STOP and branch labels rejected",
    "private_storage": "exact provider domain, plan, workspace, configuration words and generation only",
    "child_return": "successful completion, publish and unregister every field, then detach borrowed context",
    "native_work": "all original statements remain; zero retained canonical fields or host pointers",
    "unknown_dispatch": "original standalone entry; no inherited provider slot",
    "preaccept_failure": "publish/unregister any fields, retire provider/domain, disable this owner; original CALL once",
    "postsubmit_failure": "fatal; no source replay and no later domain reuse",
}
_FLAGS = {"field_retention_authorized", "incoming_definition_authority", "alias_authority",
          "native_effect_authority", "execution_authority"}
_SOURCE_FIELDS = {"version", "kind", "owner", "entry", "request", "source_inputs", "members", "edges", "budgets", *_POLICY, *_FLAGS}
_EXECUTION_FIELDS = {"identity", "execution_identity", "provider_slot", "provider_source_proof_identity", "provider_manifest",
    "canonical_resources", "provider_retained_state", "configuration_words", "helper", "checked_leaf_entry", "generation", "source_edits_identity"}


def _keys(record, keys):
    _require(type(record) is dict and set(record) == set(keys), "provider-only lifetime record fields differ")


def _span(value):
    return type(value) is list and len(value) == 2 and all(type(item) is int for item in value) and 1 <= value[0] <= value[1]


def validate_provider_only_lifetime(manifest):
    record = manifest["provider_only_lifetime"]
    _keys(record, _SOURCE_FIELDS | _EXECUTION_FIELDS)
    _require(type(record["version"]) is int and record["version"] == 1
             and record["kind"] == "original_provider_only_lifetime"
             and record["identity"] == _digest({key: record[key] for key in _SOURCE_FIELDS})
             and record["execution_identity"] == _digest({key: value for key, value in record.items() if key != "execution_identity"})
             and all(record[key] == value for key, value in _POLICY.items())
             and all(record[key] is False for key in _FLAGS), "provider-only lifetime identity or authority differs")
    _require(type(manifest.get("source_edits")) is list, "provider-only lifetime requires exact source edits")
    _require(_digest(record["source_inputs"]) == _source_inputs_identity(manifest)
             and record["source_edits_identity"] == _digest(manifest["source_edits"]),
             "provider-only lifetime source inputs or source edits differ")
    request, budgets = record["request"], record["budgets"]
    _keys(request, {"version", "call_path"})
    _keys(budgets, {"procedure_limit", "depth_limit", "operation_limit"})
    path = request["call_path"]
    _require(type(request["version"]) is int and request["version"] == 1
             and type(path) is list and all(_text(name) and name == name.lower() for name in path)
             and all(type(value) is int and value > 0 for value in budgets.values())
             and 2 <= len(path) == len(set(path)) <= budgets["procedure_limit"]
             and len(path)-1 <= budgets["depth_limit"]
             and path[0] == record["owner"] and path[-1] == record["entry"] == manifest["entry"],
             "provider-only lifetime original owner or call spine differs")
    members, edges = record["members"], record["edges"]
    _require(type(members) is list and len(members) == len(path)
             and type(edges) is list and len(edges) == len(path)-1, "provider-only lifetime complete members differ")
    for index, (name, member) in enumerate(zip(path, members, strict=True)):
        _keys(member, {"procedure", "source_kind", "source", "source_sha256", "body_sha256", "span"})
        _require(member["procedure"] == name
                 and member["source_kind"] == ("program" if index == 0 and name.startswith("$program::") else "module")
                 and (index == 0 or not name.startswith("$program::")) and _text(member["source"])
                 and _sha256(member["source_sha256"])
                 and manifest["source_inputs"].get(member["source"]) == member["source_sha256"]
                 and _sha256(member["body_sha256"]) and _span(member["span"]),
                 "provider-only lifetime original member linkage differs")
    for index, edge in enumerate(edges):
        _keys(edge, {"caller", "callee", "calls"})
        _require(edge["caller"] == path[index] and edge["callee"] == path[index+1]
                 and type(edge["calls"]) is list and bool(edge["calls"]), "provider-only lifetime original edge differs")
        ordinals = []
        for item in edge["calls"]:
            _keys(item, {"ordinal", "span", "call_sha256", "call"})
            _require(type(item["ordinal"]) is int and item["ordinal"] >= 0 and _span(item["span"])
                     and members[index]["span"][0] <= item["span"][0] <= item["span"][1] <= members[index]["span"][1]
                     and _sha256(item["call_sha256"]), "provider-only lifetime exact CALL position differs")
            call = item["call"]
            _keys(call, _ORIGINAL_CALL_FIELDS)
            _require(type(call["schema_version"]) is int and call["schema_version"] == 1
                     and call["procedure"] == edge["callee"] and call["source_kind"] == "module" and _text(call["target"])
                     and call["actual_evaluation"] == "unchanged original call position and guards"
                     and all(call[key] is False for key in ("effect_authority", "capture_authority", "alias_authority",
                         "completion_authority", "execution_authority")), "provider-only lifetime original CALL authority differs")
            formals, actuals, original, associations = (call[key] for key in (
                "formal_arguments", "actual_arguments", "original_arguments", "associations"))
            _require(type(formals) is list and all(_text(name) for name in formals) and len(formals) == len(set(formals))
                     and type(actuals) is list and len(actuals) == len(formals)
                     and all(value is None or _text(value) for value in actuals)
                     and type(original) is list and type(associations) is list and len(associations) == len(formals)
                     and [value.get("formal") for value in associations if type(value) is dict] == formals
                     and [value.get("actual") for value in associations] == actuals,
                     "provider-only lifetime original argument associations differ")
            _require(all(type(argument) is dict and _text(argument.get("formal")) for argument in original)
                     and len({argument["formal"] for argument in original}) == len(original)
                     and {argument["formal"] for argument in original} ==
                         {formal for formal, actual in zip(formals, actuals, strict=True) if actual is not None},
                     "provider-only lifetime supplied/omitted original arguments differ")
            for position, argument in enumerate(original):
                _keys(argument, {"position", "formal", "keyword", "actual"})
                _require(type(argument["position"]) is int and argument["position"] == position
                         and argument["formal"] in formals and _text(argument["actual"])
                         and argument["actual"] == actuals[formals.index(argument["formal"])]
                         and (argument["keyword"] is None or argument["keyword"] == argument["formal"]),
                         "provider-only lifetime original argument order differs")
            ordinals.append(item["ordinal"])
        _require(ordinals == sorted(set(ordinals)), "provider-only lifetime duplicate CALL occurrence")
    helper = record["helper"]
    _keys(helper, {"path", "sha256"})
    _require(helper["path"] == "provider_lifetimes/fort_provider_lifetime_" + record["identity"][:12] + ".f90"
             and _text(helper["path"]) and helper["path"].startswith("provider_lifetimes/")
             and helper["path"].endswith(".f90") and ".." not in helper["path"].split("/")
             and _sha256(helper["sha256"]) and manifest["artifacts_sha256"].get(helper["path"]) == helper["sha256"]
             and sum(item.get("path") == helper["path"] and item.get("language") == "fortran"
                     and item.get("role") == "shared_entry" for item in manifest["build_sources"]) == 1,
             "provider-only lifetime helper/build artifact differs")
    tasks = [task for scope in manifest["scopes"] for task in scope.get("external_device_tasks", [])
             if task.get("execution_binding", {}).get("provider_only_lifetime") == record["identity"]]
    _require(len(tasks) == 1, "provider-only lifetime requires one exact checked external task")
    task, = tasks
    state = validate_provider_retained_state(task["provider"])
    _keys(record["provider_retained_state"], {"qualification", "record"})
    _require(record["provider_retained_state"]["qualification"] == task["provider"]["retained_state_qualification"]
             and _digest(record["provider_retained_state"]["record"]) == _digest(state),
             "provider-only lifetime retained-state record differs")
    _require(type(record["provider_slot"]) is int
             and record["provider_slot"] == task["execution_binding"]["provider_slot"] == 1
             and type(record["configuration_words"]) is int and record["configuration_words"] == task["execution_binding"]["configuration_words"]
             and record["provider_source_proof_identity"] == task["source_proof_identity"]
             and record["provider_manifest"] == task["contract"]["provider_manifest"]
             and record["canonical_resources"] == task["canonical_resources"]
             and record["checked_leaf_entry"] == "fort_provider_entry_" +
                 sha256((record["identity"] + record["entry"]).encode()).hexdigest()[:12]
             and record["generation"] == "fresh monotonically increasing owner-call generation; every field is unregistered before return",
             "provider-only lifetime checked provider/slot/configuration linkage differs")
    return record


_PROVIDER_DUTIES = {'retained_state': 'configuration, compiled plan, domain-owned workspace, private stream and events only', 'fields_after_completion': 'no application field pointer, external view or registration handle is reused or dereferenced after successful wait and unregister', 'execute_bindings': 'each execute uses only its current supplied context/bindings', 'publication': 'complete all submitted application field uses before unregister and borrowed close', 'failure': 'post-submit failure is fatal; no reuse or replay', 'field_retention_authorized': False, 'host_payload_retention_authorized': False, 'incoming_definition_authority': False, 'execution_authority': False}


def validate_provider_retained_state(provider):
    import json

    from .device_task_record import _reference
    _require("retained_state_qualification" in provider,
             "provider-only lifetime requires an explicit retained-state qualification")
    state = json.loads(_reference(provider["retained_state_qualification"]).read_text())
    keys = {"version", "kind", "provider_binding", "duties", "evidence", "prerequisite"}
    _keys(state, keys)
    original = json.loads(_reference(state["prerequisite"]).read_text())
    _keys(original, keys - {"prerequisite"})
    binding = {key: provider[key] for key in ("fortran", "artifacts", "metadata_query", "execution", "pressure_backend")
               if key in provider}
    for record, kind in ((state, "external_provider_retained_plan_state"),
                         (original, "reviewed_external_provider_retention_prerequisite")):
        _require(type(record["version"]) is int and record["version"] == 1 and record["kind"] == kind
                 and _digest(record["provider_binding"]) == _digest(binding)
                 and type(record["duties"]) is dict and record["duties"] == _PROVIDER_DUTIES
                 and all(record["duties"].get(key) is False for key in (
                     "field_retention_authorized", "host_payload_retention_authorized",
                     "incoming_definition_authority", "execution_authority"))
                 and type(record["evidence"]) is list and bool(record["evidence"]),
                 "provider-only lifetime retained-state implementation or duties differ")
        for evidence in record["evidence"]:
            _reference(evidence)
    _require(all(_digest(state[key]) == _digest(original[key]) for key in ("provider_binding", "duties", "evidence")),
             "provider-only lifetime retained-state prerequisite differs")
    return state
