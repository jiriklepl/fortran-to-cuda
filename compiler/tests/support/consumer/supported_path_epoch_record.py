"""Transport a source-owned exclusion gate; grant no compiler or runtime authority."""
from __future__ import annotations

from hashlib import sha256

from .checked_execution_plan import _digest, _require, _sha256, _text
from .owner_local_call_group_record import _names, _selection
from .sequence_exit_record import _span
from .source_placement_record import _tasks

_FLAGS = {"entry_definition_authority", "payload_access_authorized", "alias_authority",
          "execution_authorized", "cross_frontier_retention_authorized"}
_FIELDS = _FLAGS | {
    "identity", "version", "kind", "selection_identity", "procedure", "source_graph", "selected_node_ids",
    "path_identity", "gate_identity", "descriptor_identity", "resources", "descriptor_guard_resources",
    "calls", "retained_edges", "outside_interval_edges", "eligibility", "sequence_exit", "activation", "allocation_presence",
    "ordering", "additional_unproved_edge",
}
_ELIGIBILITY = {
    "descriptor", "owner_control", "storage", "initial", "arm_source_span", "arm_source_sha256",
    "barred_calls", "controls", "propagation", "future_activation", "rearm_after_excluded_edge",
}
_BARRED = {"node_id", "procedure", "source_span", "original_source_sha256", "clear_source_span", "clear_source_sha256",
           "exposure_resources", "exposure_descriptor_identity"}
_ANCHOR = {"span", "sha256", "callee"}
_GATE_METADATA = {"node_id", "source_span", "original_source_sha256", "clock_identity", "descriptor_identity"}


def _keys(value, fields, reason):
    _require(isinstance(value, dict) and set(value) == fields, "supported path " + reason)


def _anchor(value):
    _keys(value, _ANCHOR, "requires exact original CALL anchors")
    _require(_span(value["span"]) and _sha256(value["sha256"]) and _text(value["callee"]),
             "supported path original CALL anchor differs")


def _request(record, value, plan, available):
    request = record["path_request"]
    _keys(request, {"version", "source", "first", "last", "eligibility"}, "requires its exact source path request")
    _require(type(request["version"]) is int and request["version"] == 1,
             "supported path source request version differs")
    source = request["source"]
    _keys(source, {"path", "sha256"}, "requires its exact original source reference")
    _require(source["path"] == plan["procedures"][0]["source"]
             and source["sha256"] == plan["procedures"][0]["source_sha256"],
             "supported path requested original source differs")
    for key, call in (("first", value["calls"][0]), ("last", value["calls"][-1])):
        anchor = request[key]
        _anchor(anchor)
        _require(anchor["span"] == call["source_span"] and anchor["callee"] == call["procedure"],
                 "supported path original endpoint differs")
    eligibility = request["eligibility"]
    _keys(eligibility, {"binding", "resources", "barred_calls"}, "requires its exact original eligibility request")
    _require(eligibility["binding"] == value["eligibility"]["descriptor"]["name"]
             and _names(eligibility["resources"]) and bool(eligibility["resources"]),
             "supported path requested persistent gate or allocation names differ")
    _require(set(eligibility["resources"]) <= available,
             "supported path requested canonical allocation lies outside its descriptor inventory")
    barred = eligibility["barred_calls"]
    _require(isinstance(barred, list) and len(barred) == len(value["eligibility"]["barred_calls"]),
             "supported path excluded CALL request inventory differs")
    for anchor, call in zip(barred, value["eligibility"]["barred_calls"], strict=True):
        _anchor(anchor)
        _require(anchor["span"] == call["source_span"] and anchor["callee"] == call["procedure"]
                 and anchor["sha256"] == call["original_source_sha256"],
                 "supported path requested excluded source CALL differs")


def _eligibility(value, plan, scope):
    control = value["eligibility"]
    _keys(control, _ELIGIBILITY, "requires its exact persistent eligibility controls")
    descriptor = control["descriptor"]
    _keys(descriptor, {"name", "resource", "type", "kind", "rank", "intent", "attributes", "lower_bounds", "shape"},
          "requires its original local SAVE flag descriptor")
    attributes = descriptor["attributes"]
    _require(_text(descriptor["name"]) and descriptor["resource"] == plan["entry"]+"::"+descriptor["name"]
             and type(descriptor["kind"]) is int and descriptor["kind"] == 4
             and descriptor["type"] == "logical" and type(descriptor["rank"]) is int and descriptor["rank"] == 0
             and descriptor["intent"] is None and descriptor["shape"] == [] and descriptor["lower_bounds"] == []
             and isinstance(attributes, list) and all(_text(item) for item in attributes)
             and attributes == sorted(set(attributes)) and "save" in attributes
             and not set(attributes).intersection({"pointer", "target", "optional", "volatile", "asynchronous", "value", "parameter", "protected"})
             and control["owner_control"] == descriptor["name"],
             "supported path flag must remain the original private local SAVE LOGICAL")
    _require(control["initial"] is False and control["rearm_after_excluded_edge"] is False
             and control["storage"] == "original private local SAVE LOGICAL"
             and control["propagation"] == "same original owner SAVE gate forwarded INOUT through every checked ENTRY"
             and control["future_activation"] == "false vetoes every managed registration, fresh context, child and device dispatch"
             and _span(control["arm_source_span"]) and _sha256(control["arm_source_sha256"])
             and control["arm_source_span"][1] < value["sequence_exit"]["source_span"][0],
             "supported path cannot rearm or bypass the original allocation-lifetime exclusion")
    members = plan["procedures"]
    controls = control["controls"]
    _require(isinstance(controls, list) and len(controls) == len(members),
             "supported path requires every checked member's activation veto")
    expected = []
    for index, member in enumerate(members):
        argument = (descriptor["name"] if index == 0 else
                    "fort_child_eligibility_" + sha256(member["procedure"].encode()).hexdigest()[:12])
        expected.append({"procedure": member["procedure"], "argument": argument})
    _require(controls == expected, "supported path original gate must reach every checked child ENTRY")
    rows = scope.get("module_coordinators")
    _require(isinstance(rows, list) and all(isinstance(row, dict) for row in rows),
             "supported path requires checked module ENTRY records")
    names = [row.get("procedure") for row in rows]
    _require(all(_text(name) for name in names) and len(set(names)) == len(names)
             and set(names) == {member["procedure"] for member in members[1:]},
             "supported path checked child inventory differs")
    by_name = {member["procedure"]: member for member in members}
    arguments = {item["procedure"]: item["argument"] for item in expected}
    for row in rows:
        _require(type(row.get("control_version")) is int and row["control_version"] == 6
                 and row.get("interface") == "original_module_entry_v1"
                 and row.get("structured_summary_identity") == by_name[row["procedure"]]["structured_summary_identity"]
                 and row.get("parameterized_handles") is True
                 and row.get("storage_identity") == "caller canonical allocation identity forwarded separately from helper-local slots"
                 and row.get("native_abi") == "original entry; compiler-control arguments are never referenced"
                 and row.get("storage") == "one original body and declarations; no persistent-state copy",
                 "supported path requires checked v6 ENTRY storage and activation controls")
        _require(row.get("eligibility_control") == {
            "argument": arguments[row["procedure"]],
            "association": "same original owner SAVE exclusion gate forwarded INOUT",
            "activation": "required before every registration, fresh context and child device dispatch"},
            "supported path child cannot replace or bypass the original saved eligibility gate")
    return control


def _checked_calls(value, plan, scope, canonical):
    """Cross-link original sites to the checked ENTRY and canonical handles."""
    members = {row["procedure"]: row for row in plan["procedures"]}
    coordinators = {row["procedure"]: row for row in scope["module_coordinators"]}
    for name, row in coordinators.items():
        _require(row.get("calls") == members[name]["calls"] and not row.get("boundaries")
                 and not members[name]["boundaries"],
                 "supported path checked child changed its sites or acquired an unproved retirement edge")
    for call in value["calls"]:
        child = members.get(call["procedure"])
        if call["barred"]:
            _require(child is None, "supported path excluded CALL cannot also be a checked child")
            continue
        if child is None:
            # Native retained edges have independent descriptor receipts. They
            # do not borrow handles merely because the source path retains them.
            continue
        _require(child["procedure"] != plan["entry"], "supported path cannot recurse into its storage owner")
        sites = [site for site in child["calls"] if site.get("caller") == plan["entry"]
                 and (site.get("first_line"), site.get("last_line")) == tuple(call["source_span"])]
        mappings = call["canonical_resources"]
        _require(len(sites) == 1 and sites[0].get("resource_mappings") == mappings,
                 "supported path checked child lost its exact original CALL association")
        handles = sites[0].get("handle_bindings")
        _require(isinstance(handles, dict) and all(
            _text(slot) and slot.isascii() and slot.isdecimal() and str(int(slot)) == slot and int(slot) > 0
            and type(number) is int and number > 0
            and number in {row["registration_identity"] for row in canonical.values()}
            for slot, number in handles.items()),
            "supported path checked child forwarded invalid canonical handles")
        used = {root for unit in child["units"] for root in unit["resources"]}
        for formal, root in mappings.items():
            if formal in used:
                slot = str(int(sha256((child["procedure"]+"|"+formal).encode()).hexdigest()[:8], 16)+1)
                _require(root in canonical and handles.get(slot) == canonical[root]["registration_identity"],
                         "supported path checked child canonical registration differs")
    expected = [{"first_line": row["source_span"][0], "last_line": row["source_span"][1],
                 "reason": "exact source-bound sticky exclusion",
                 "execution": "publish and close when reached; original continuation once",
                 "conditional_close": None, "reopen": True}
                for row in value["calls"] if row["barred"]]
    _require(members[plan["entry"]]["boundaries"] == expected and scope.get("boundaries") == expected,
             "supported path retirement inventory differs from its exact excluded original CALLs")


def validate_supported_path_epoch(value, epoch, plan, manifest, scope, *, capture_facts=None):
    version = value.get("version") if isinstance(value, dict) else None
    _require(type(version) is int and version in (1, 2, 3), "supported path epoch version differs")
    fields = _FIELDS | ({"gate_metadata_edges"} if version >= 2 else set())
    if version == 3:
        fields |= {"reached_position", "entry_preservation"}
    _keys(value, fields, "requires its exact epoch schema")
    _require(type(value["version"]) is int
             and value["kind"] == "supported_original_source_path"
             and _sha256(value["identity"])
             and value["identity"] == _digest({key: item for key, item in value.items() if key != "identity"})
             and all(_sha256(value[key]) for key in ("selection_identity", "source_graph", "path_identity", "gate_identity", "descriptor_identity")),
             "supported path requires its exact versioned content identity")
    _require(value["procedure"] == plan["entry"] == epoch["entry"]
             and value["source_graph"] == plan["procedures"][0]["structured_summary_identity"]
             and scope.get("graph_root_kind") == "supported original source path"
             and _digest(scope.get("supported_path_epoch")) == _digest(value)
             and plan["policy"].get("lifetime") == "supported original source path; persistent eligibility veto; close at original sequence exit",
             "supported path source, checked scope or lifetime differs")
    activation = ("original reached interval; source gate required before every activation; only independently preserved invocation-entry definitions; NONE after retirement"
                  if version == 3 else "original reached interval; source gate required before every activation; NONE")
    _require(all(value[flag] is False for flag in _FLAGS)
             and value["activation"] == activation
             and value["allocation_presence"] == "lazy original guards at first reached managed access"
             and value["ordering"] == "original source order; all checked graph tasks conservatively ordered"
             and value["additional_unproved_edge"] == "reject entire supported path",
             "supported path cannot grant definitions, payload, aliases or execution")
    selection = manifest.get("native_ancestor_selection")
    _selection(selection, epoch, plan, version=4, lifetime="supported_original_path", extra_fields={"path_request"})
    _require(value["selection_identity"] == selection["identity"], "supported path selector identity differs")
    selected = value["selected_node_ids"]
    _require(_names(selected) and len(selected) >= 2 and all(node.startswith(plan["entry"] + "/") for node in selected),
             "supported path requires its distinct original source interval")
    canonical = {item["resource"]: item for item in epoch["resources"]}
    resources, guards = value["resources"], value["descriptor_guard_resources"]
    _require(_names(resources) and bool(resources) and _names(guards) and guards == sorted(guards)
             and resources == sorted(root for root, row in canonical.items() if row.get("device_replica_authorized") is True)
             and not set(resources).intersection(guards)
             and all(canonical[root].get("lifetime") == "selected_window" for root in resources),
             "supported path canonical or descriptor-only storage inventory differs")
    available = set(resources) | set(guards)
    calls = value["calls"]
    _require(isinstance(calls, list) and len(calls) >= 2, "supported path requires its complete ordinary CALL inventory")
    seen, last_line = set(), 0
    for call in calls:
        _keys(call, {"node_id", "procedure", "source_span", "canonical_resources", "barred"}, "requires exact original CALL records")
        mappings = call["canonical_resources"]
        _require(_text(call["node_id"]) and call["node_id"].startswith(plan["entry"] + "/")
                 and call["node_id"] not in seen and _text(call["procedure"])
                 and _span(call["source_span"]) and call["source_span"][0] > last_line
                 and type(call["barred"]) is bool and isinstance(mappings, dict)
                 and all(_text(formal) and _text(root) and root in available for formal, root in mappings.items())
                 and len(set(mappings.values())) == len(mappings),
                 "supported path original CALL order, associations or exclusion differs")
        seen.add(call["node_id"])
        last_line = call["source_span"][1]
    exit_record = value["sequence_exit"]
    _keys(exit_record, {"last_call_node_id", "source_span", "after_line", "cleanup"}, "requires its mandatory original exit")
    _require(exit_record["last_call_node_id"] == selected[-1] == calls[-1]["node_id"]
             and calls[0]["node_id"] == selected[0] and _span(exit_record["source_span"])
             and exit_record["source_span"] == [calls[0]["source_span"][0], calls[-1]["source_span"][1]]
             and type(exit_record["after_line"]) is int and exit_record["after_line"] == exit_record["source_span"][1]
             and exit_record["cleanup"] == "publish and close immediately after final original CALL; clear handles",
             "supported path changed its original complete source interval or cleanup")
    control = _eligibility(value, plan, scope)
    _checked_calls(value, plan, scope, canonical)
    barred = control["barred_calls"]
    expected_barred = [call for call in calls if call["barred"]]
    _require(isinstance(barred, list) and bool(barred) and len(barred) == len(expected_barred),
             "supported path requires every explicit excluded CALL and original clearing")
    for row, call in zip(barred, expected_barred, strict=True):
        _keys(row, _BARRED, "requires exact excluded original CALL and clearing anchors")
        _require(all(row[key] == call[key] for key in ("node_id", "procedure", "source_span"))
                 and _sha256(row["original_source_sha256"]) and _sha256(row["clear_source_sha256"])
                 and _span(row["clear_source_span"]) and row["clear_source_span"][1] < row["source_span"][0],
                 "supported path excluded CALL must keep its preceding persistent clearing")
    _request(selection, value, plan, available)
    outside_private = sorted(available - set(selection["path_request"]["eligibility"]["resources"]))
    for row in barred:
        _require(row["exposure_resources"] == outside_private
                 and (_sha256(row["exposure_descriptor_identity"]) if outside_private
                      else row["exposure_descriptor_identity"] is None),
                 "supported path excluded edge needs its independent full outside-private descriptor receipt")
    retained = value["retained_edges"]
    _require(isinstance(retained, list) and bool(retained), "supported path requires every retained descriptor edge")
    edges = []
    for row in retained:
        _keys(row, {"node_id", "descriptor_identity"}, "requires exact independently checked descriptor components")
        _require(_text(row["node_id"]) and row["node_id"].startswith(plan["entry"] + "/")
                 and _sha256(row["descriptor_identity"]), "supported path retained descriptor identity differs")
        edges.append(row["node_id"])
    barred_ids = {call["node_id"] for call in expected_barred}
    _require(len(set(edges)) == len(edges) and not barred_ids.intersection(edges)
             and {call["node_id"] for call in calls if not call["barred"]} <= set(edges),
             "supported path cannot omit an ordinary retained CALL or retain an excluded edge")
    execution = scope.get("graph_execution")
    _require(isinstance(execution, dict) and execution.get("source_order_executor") is True
             and execution.get("source_motion") == [] and isinstance(execution.get("procedures"), list),
             "supported path must preserve original source order")
    graphs = execution["procedures"]
    _require([row.get("procedure") for row in graphs if isinstance(row, dict)] == [row["procedure"] for row in plan["procedures"]],
             "supported path complete original task graph inventory differs")
    for row in graphs:
        _require(all(task.get("ordered") is True for task in _tasks(row.get("task_graph"))),
                 "supported path cannot reorder persistent gate controls or source work")
    outside = value["outside_interval_edges"]
    _require(isinstance(outside, list) and bool(outside),
             "supported path requires its original prefix and suffix lifetime receipts")
    outside_ids = []
    for row in outside:
        _keys(row, {"node_id", "descriptor_identity"}, "requires exact outside-interval descriptor receipts")
        _require(_text(row["node_id"]) and row["node_id"].startswith(plan["entry"] + "/body/")
                 and _sha256(row["descriptor_identity"]),
                 "supported path outside-interval source or lifetime identity differs")
        outside_ids.append(row["node_id"])
    from .entry_preservation_record import _direct_children

    root_ids = {task["id"] for task in _tasks(graphs[0]["task_graph"])}
    sequence = plan["entry"] + "/body"
    children = [sequence + "/" + str(number) for number in _direct_children(root_ids, sequence)]
    _require(selected[0] in children and selected[-1] in children
             and selected == children[children.index(selected[0]):children.index(selected[-1])+1]
             and all(call["node_id"] in root_ids and any(
                 call["node_id"] == node or call["node_id"].startswith(node+"/") for node in selected)
                     for call in calls),
             "supported path selected interval or original CALL graph occurrence differs")
    _require(bool(children) and outside_ids == [node for node in children[1:] if node not in selected],
             "supported path omitted or reordered original outside-interval lifetime checks")
    if version >= 2:
        metadata = value["gate_metadata_edges"]
        _require(isinstance(metadata, list) and (version == 3 or bool(metadata)),
                 "supported path requires its separately proved allocation-guard metadata")
        seen_metadata, previous_line = set(), 0
        for row in metadata:
            _keys(row, _GATE_METADATA, "requires exact original allocation-guard metadata receipts")
            node = row["node_id"]
            _require(_text(node) and node.startswith(children[0]+"/body0/") and node not in seen_metadata
                     and any(task == node or task.startswith((node+"/", node+":alternative:"))
                             for task in root_ids)
                     and _span(row["source_span"]) and previous_line < row["source_span"][0]
                     <= row["source_span"][1] < control["arm_source_span"][0]
                     and all(_sha256(row[key]) for key in ("original_source_sha256", "clock_identity", "descriptor_identity")),
                     "supported path gate metadata lost its original first-guard source, clock or full-inventory descriptor receipt")
            seen_metadata.add(node)
            previous_line = row["source_span"][1]
    if version == 3:
        from .reached_entry_preservation_record import (
            validate_reached_entry_preservation,
        )

        _require(isinstance(value["entry_preservation"], dict)
                 and type(value["entry_preservation"].get("version")) is int
                 and value["entry_preservation"]["version"] == 4,
                 "supported path v3 requires its exact projected original entry preservation")
        validate_reached_entry_preservation(value["entry_preservation"], value["reached_position"], value,
            epoch, plan, original_tasks=_tasks(graphs[0]["task_graph"]), capture_facts=capture_facts,
            gate_identity=value["gate_identity"], private_resources=selection["path_request"]["eligibility"]["resources"])
    return value
