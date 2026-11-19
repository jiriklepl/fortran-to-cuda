"""Check bounded original call-group records without issuing lifetime authority."""
from __future__ import annotations

from hashlib import sha256

from .checked_execution_plan import _digest, _require, _sha256, _text
from .conditional_return_record import _SELECTION
from .sequence_exit_record import _span, validate_sequence_exit
from .source_placement_record import _tasks

_FIELDS = {
    "identity", "version", "kind", "procedure", "selected_node_ids", "sequence_exit", "descriptor_identity",
    "selection_identity", "resources", "descriptor_guard_resources", "calls", "allocation_presence",
    "repeated_activation", "entry_definition_authority", "execution_authorized", "cross_frontier_retention_authorized",
}


def _names(value):
    return (isinstance(value, list) and all(_text(item) for item in value)
            and len(set(value)) == len(value))


def _selection(record, epoch, plan, *, version=2, lifetime="owner_local_call_groups", extra_fields=frozenset()):
    _require(isinstance(record, dict) and set(record) == _SELECTION | {"lifetime"} | extra_fields
             and type(record["version"]) is int and record["version"] == version
             and record["kind"] == "original_native_ancestor_selection"
             and record["lifetime"] == lifetime and record["entry"] == plan["entry"]
             and record["source_inputs"] == epoch["source_inputs"]
             and _sha256(record["identity"])
             and record["identity"] == _digest({key: value for key, value in record.items() if key != "identity"})
             and _sha256(record["original_entry_body_sha256"]),
             "owner-local call group native ancestor selection differs")
    _require(record["unselected_native_once"] is True
             and record["selection_semantics"] == "only exact selected original module subtrees may undergo device discovery; all ancestors remain original native source"
             and record["call_authority"] == "independent exact original call association and checked companion admission"
             and record["native_execution"] == "existing precise native effects and coherence hooks or reached publication and full domain retirement"
             and all(record[key] is False for key in ("payload_access_authorized", "descriptor_stability_authorized",
                 "entry_definition_authority", "execution_authorized", "placement_authorized", "alias_authority")),
             "owner-local call group selector cannot grant access, definitions, lifetime or aliases")
    roots = record["device_roots"]
    _require(isinstance(roots, list) and bool(roots), "owner-local call group needs selected original roots")
    names = []
    for root in roots:
        _require(isinstance(root, dict) and set(root) == {"procedure", "source", "source_sha256", "original_body_sha256"}
                 and _text(root["procedure"]) and root["procedure"] != plan["entry"]
                 and _text(root["source"]) and root["source"] in plan["sources"]
                 and root["source_sha256"] == plan["sources"][root["source"]]
                 and _sha256(root["original_body_sha256"]),
                 "owner-local call group original selected root source differs")
        names.append(root["procedure"])
        for child in plan["procedures"]:
            if child["procedure"] == root["procedure"]:
                _require(child["source"] == root["source"] and child["source_sha256"] == root["source_sha256"],
                         "owner-local call group selected child source differs")
    _require(names == sorted(set(names)), "owner-local call group repeats or reorders selected roots")
    return set(names)


def validate_owner_local_call_group(value, epoch, plan, manifest, scope, *, capture_facts=None):
    version = value.get("version") if isinstance(value, dict) else None
    _require(type(version) is int and version in (2, 3),
             "owner-local call group requires its exact versioned identity")
    fields = _FIELDS if version == 2 else _FIELDS | {"entry_preservation"}
    _require(isinstance(value, dict) and set(value) == fields
             and value["kind"] == "owner_local_original_call_group"
             and value["procedure"] == plan["entry"] == epoch["entry"]
             and _sha256(value["identity"])
             and value["identity"] == _digest({key: item for key, item in value.items() if key != "identity"})
             and _sha256(value["descriptor_identity"]),
             "owner-local call group requires its exact versioned identity")
    _require(scope.get("graph_root_kind") == "owner-local original CALL group"
             and _digest(scope.get("owner_local_call_group")) == _digest(value)
             and plan["policy"].get("lifetime") == "owner-local original call group; close at original sequence exit",
             "owner-local call group differs from its checked scope or lifetime")
    activation = ("fresh local context and NONE at every reached group; close and clear handles at original sequence exit"
                  if version == 2 else
                  "fresh local context; only independently preserved invocation-entry definitions; NONE after retirement; close and clear handles at original sequence exit")
    _require(value["allocation_presence"] == "lazy original guards at the first reached managed access"
             and value["repeated_activation"] == activation
             and all(value[key] is False for key in ("entry_definition_authority", "execution_authorized",
                                                    "cross_frontier_retention_authorized")),
             "owner-local call group cannot grant definitions, execution or cross-exit retention")
    selected_roots = _selection(manifest.get("native_ancestor_selection"), epoch, plan)
    _require(value["selection_identity"] == manifest["native_ancestor_selection"]["identity"],
             "owner-local call group selector identity differs")
    exit_record = validate_sequence_exit(value["sequence_exit"], epoch, plan)
    selected = value["selected_node_ids"]
    _require(selected == exit_record["selected_node_ids"], "owner-local call group selection differs from its exit")
    resources, guards = value["resources"], value["descriptor_guard_resources"]
    canonical = {item["resource"]: item for item in epoch["resources"]}
    _require(_names(resources) and bool(resources) and _names(guards) and guards == sorted(guards)
             and resources == sorted(root for root, item in canonical.items() if item.get("device_replica_authorized") is True)
             and not set(resources).intersection(guards)
             and all(root in canonical and canonical[root].get("device_replica_authorized") is True
                     and canonical[root].get("lifetime") == "selected_window" for root in resources),
             "owner-local call group managed and descriptor-only storage inventories differ")
    available = set(resources) | set(guards)
    execution = scope.get("graph_execution")
    _require(isinstance(execution, dict) and execution.get("source_order_executor") is True
             and isinstance(execution.get("procedures"), list), "owner-local call group needs its original graph")
    graphs = [row for row in execution["procedures"] if isinstance(row, dict) and row.get("procedure") == plan["entry"]]
    _require(len(graphs) == 1, "owner-local call group original graph procedure differs")
    tasks = _tasks(graphs[0].get("task_graph"))
    graph_ids = [task.get("id") for task in tasks]
    # Existing source CALL tasks are opaque barriers. Their separate generic
    # ordered flag stays at its default false; opacity already preserves all
    # inter-call dependencies. This does not grant source motion.
    _require(all(graph_ids.count(node) == 1 for node in selected)
             and all(task.get("opaque") is True and task.get("ordered") is False
                     for task in tasks if task.get("id") in selected),
             "owner-local call group original opaque CALL task differs")
    if version == 3:
        from .entry_preservation_record import validate_entry_preservation

        validate_entry_preservation(value["entry_preservation"], value, epoch, plan,
                                    original_tasks=tasks, capture_facts=capture_facts)
    calls = value["calls"]
    _require(isinstance(calls, list) and len(calls) == len(selected)
             and all(isinstance(call, dict) for call in calls)
             and [call.get("node_id") for call in calls] == selected,
             "owner-local call group complete ordered CALL inventory differs")
    procedures = {row["procedure"]: row for row in plan["procedures"]}
    coordinators = scope.get("module_coordinators")
    _require(isinstance(coordinators, list) and all(isinstance(row, dict) for row in coordinators),
             "owner-local call group requires checked child ENTRY records")
    names = [row.get("procedure") for row in coordinators]
    _require(all(_text(name) for name in names) and len(set(names)) == len(names)
             and set(names) == set(procedures)-{plan["entry"]},
             "owner-local call group checked companion inventory differs")
    for row in coordinators:
        _require(type(row.get("control_version")) is int and row["control_version"] == 4
                 and row.get("interface") == "original_module_entry_v1"
                 and row.get("native_abi") == "original entry; compiler-control arguments are never referenced"
                 and row.get("structured_summary_identity") == procedures[row["procedure"]]["structured_summary_identity"]
                 and row.get("parameterized_handles") is True
                 and row.get("storage_identity") == "caller canonical allocation identity forwarded separately from helper-local slots",
                 "owner-local call group original checked companion ABI differs")
    previous = exit_record["source_span"][0]-1
    for call in calls:
        _require(set(call) == {"node_id", "procedure", "source_span", "canonical_resources"}
                 and _text(call["procedure"]) and call["procedure"] in selected_roots
                 and call["procedure"] in procedures and _span(call["source_span"])
                 and previous < call["source_span"][0] <= call["source_span"][1] <= exit_record["source_span"][1],
                 "owner-local call group original checked child or edit span differs")
        previous = call["source_span"][1]
        mappings = call["canonical_resources"]
        _require(isinstance(mappings, dict) and all(_text(local) and _text(root) and root in available
                                                  for local, root in mappings.items())
                 and len(set(mappings.values())) == len(mappings),
                 "owner-local call group canonical CALL associations differ")
        child = procedures[call["procedure"]]
        sites = [site for site in child["calls"] if site.get("caller") == plan["entry"]
                 and (site.get("first_line"), site.get("last_line")) == tuple(call["source_span"])]
        _require(len(sites) == 1 and sites[0].get("resource_mappings") == mappings,
                 "owner-local call group exact original CALL site differs")
        handles = sites[0].get("handle_bindings")
        _require(isinstance(handles, dict) and all(_text(slot) and slot.isascii() and slot.isdecimal()
                 and str(int(slot)) == slot and int(slot) > 0 and type(number) is int and number > 0
                 and number in {item["registration_identity"] for item in canonical.values()}
                 for slot, number in handles.items()), "owner-local call group forwarded storage handles differ")
        used = {root for unit in child["units"] for root in unit["resources"]}
        for formal, root in mappings.items():
            if formal in used:
                slot = str(int(sha256((child["procedure"]+"|"+formal).encode()).hexdigest()[:8], 16)+1)
                _require(root in canonical and handles.get(slot) == canonical[root]["registration_identity"],
                         "owner-local call group original child canonical registration differs")
    _require(calls[0]["source_span"][0] == exit_record["source_span"][0]
             and calls[-1]["source_span"][1] == exit_record["source_span"][1]
             and not procedures[plan["entry"]]["boundaries"],
             "owner-local call group changed its exact closed source span")
