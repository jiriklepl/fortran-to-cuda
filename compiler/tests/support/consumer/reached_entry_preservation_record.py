"""Transport root-entry preservation without creating source or replica authority."""
from __future__ import annotations

import re

from .checked_execution_plan import _digest, _require, _sha256, _text
from .entry_preservation_record import (
    _FLAGS,
    _FORBIDDEN,
    _direct_children,
    _segment_ids,
)
from .source_position_record import validate_source_position

_BASE = _FLAGS | {
    "identity", "version", "kind", "procedure", "source_inputs", "source_graph",
    "reached_position_identity", "segments", "entry_capture_facts_identity", "preserved_resources",
    "definition_source", "prefix", "semantics",
}
_PROJECTION = _FLAGS | {
    "identity", "version", "kind", "procedure", "source_inputs", "source_graph",
    "allocation_gate_identity", "private_resources", "local_declaration_identities", "original_segments", "steps",
    "semantics", "ordinary_native_effects_require_independent_preservation",
    "private_resources_cannot_receive_preserved_entry_facts",
}
_STEP_COUNTS = {"ordinary_native": 0, "fresh_private_allocation": 1,
                "fresh_private_literal_overwrite": 0, "original_scalar_clock": 2,
                "original_optional_scalar_control": 1, "original_comment": 0, "original_end_if": 0}
_STRUCTURAL = {"original_comment", "original_end_if"}
_NUMBER = r"(?:0|[1-9][0-9]*)"


def _record(value, fields, version, kind, label):
    _require(isinstance(value, dict) and set(value) == fields
             and type(value["version"]) is int and value["version"] == version
             and value["kind"] == kind and _sha256(value["identity"])
             and value["identity"] == _digest({key: item for key, item in value.items() if key != "identity"}),
             label+" requires its exact versioned content identity")


def _source_order(identity, entry):
    """Order the existing root/body/predicate grammar, without a numeric window."""
    suffix = identity.removeprefix(entry+"/body/")
    parts, result = suffix.split("/"), []
    _require(re.fullmatch(_NUMBER, parts[0]) is not None, "entry projection source path differs")
    result.append(int(parts[0]))
    index = 1
    while index < len(parts):
        part = parts[index]
        condition = re.fullmatch("condition("+_NUMBER+")", part)
        if condition:
            _require(index == len(parts)-1, "entry projection predicate source path differs")
            result.extend((0, int(condition[1])))
            break
        body = re.fullmatch("body("+_NUMBER+")", part)
        _require(body is not None and index+1 < len(parts)
                 and re.fullmatch(_NUMBER, parts[index+1]) is not None,
                 "entry projection nested original source path differs")
        result.extend((1, int(body[1]), int(parts[index+1])))
        index += 2
    return tuple(result)


def _projection(value, entry_value, group, epoch, plan, prefix, task_ids, *, gate_identity, private_resources):
    _record(value, _PROJECTION, 3, "original_private_allocation_prefix_projection", "entry projection")
    _require(value["procedure"] == plan["entry"] and value["source_inputs"] == epoch["source_inputs"]
             and value["source_graph"] == group["source_graph"]
             and _sha256(value["allocation_gate_identity"])
             and (gate_identity is None or value["allocation_gate_identity"] == gate_identity)
             and value["original_segments"] == entry_value["original_prefix_segments"] == [prefix],
             "entry projection lost its complete original prefix or allocation gate")
    _require(all(value[flag] is False for flag in _FLAGS)
             and value["ordinary_native_effects_require_independent_preservation"] is True
             and value["private_resources_cannot_receive_preserved_entry_facts"] is True
             and value["semantics"] == "filter existing incoming definitions across unchanged source; successful fresh private storage only",
             "entry projection cannot grant definitions, payloads, lifetime, aliases or execution")
    private, declarations = value["private_resources"], value["local_declaration_identities"]
    _require(isinstance(private, list) and bool(private) and all(_text(root) for root in private)
             and len(set(private)) == len(private)
             and (private_resources is None or private == list(private_resources))
             and not set(private).intersection(entry_value["preserved_resources"])
             and isinstance(declarations, list) and len(declarations) == len(private)
             and all(_sha256(identity) for identity in declarations),
             "entry projection cannot preserve private allocations as incoming definitions")
    steps = value["steps"]
    _require(isinstance(steps, list) and bool(steps), "entry projection needs complete original source steps")
    _require(bool(prefix), "entry projection requires its original first allocation guard")
    first_guard = prefix[0]
    guard_body = first_guard+"/body0"
    guard_children = [guard_body+"/"+str(number) for number in _direct_children(task_ids, guard_body)]
    semantic, native, allocations, allocation_nodes, guard_steps, end_positions = [], [], [], [], [], []
    for index, step in enumerate(steps):
        _require(isinstance(step, dict)
                 and set(step) == {"kind", "source_node_id", "original_source_sha256", "proof_identities"}
                 and _text(step["kind"]) and step["kind"] in _STEP_COUNTS
                 and _sha256(step["original_source_sha256"])
                 and isinstance(step["proof_identities"], list)
                 and len(step["proof_identities"]) == _STEP_COUNTS[step["kind"]]
                 and all(_sha256(identity) for identity in step["proof_identities"]),
                 "entry projection original step or declaration link differs")
        node = step["source_node_id"]
        if step["kind"] in _STRUCTURAL:
            _require(node is None, "entry projection documentation cannot invent a graph node")
            if step["kind"] == "original_end_if":
                end_positions.append(index)
            continue
        _require(_text(node) and any(node == root or node.startswith(root+"/") for root in prefix)
                 and (node in task_ids or any(task.startswith(node+":alternative:") for task in task_ids)),
                 "entry projection step lost its existing original graph occurrence")
        semantic.append(node)
        if node.startswith(first_guard+"/"):
            guard_steps.append((index, step))
        if step["kind"] in {"fresh_private_allocation", "fresh_private_literal_overwrite"}:
            _require(node in guard_children, "entry projection private exemption escaped its original allocation guard")
        if step["kind"] == "ordinary_native":
            native.append([node])
        elif step["kind"] == "fresh_private_allocation":
            allocations.extend(step["proof_identities"])
            allocation_nodes.append(node)
    _require(bool(guard_children) and len(end_positions) == 1 and bool(guard_steps)
             and guard_steps[0][1]["source_node_id"] == first_guard+"/condition0"
             and guard_steps[0][1]["kind"] == "ordinary_native"
             and guard_steps[-1][1]["source_node_id"] == guard_children[-1]
             and guard_steps[-1][1]["kind"] == "ordinary_native"
             and guard_steps[-1][0] < end_positions[0]
             and all(index < end_positions[0] if node.startswith(first_guard+"/")
                     else index > end_positions[0]
                     for index, step in enumerate(steps) if (node := step["source_node_id"]) is not None)
             and allocation_nodes == guard_children[:len(private)],
             "entry projection changed its original allocation, predicate, arm or guard boundary")
    _require(allocations == declarations and len(set(semantic)) == len(semantic)
             and semantic == sorted(semantic, key=lambda node: _source_order(node, plan["entry"]))
             and not any(a != b and b.startswith(a+"/") for a in semantic for b in semantic),
             "entry projection duplicated, reordered or overlapped original semantic steps")
    leaves = {task for task in task_ids if ":" not in task[len(plan["entry"]):]
              and any(task == root or task.startswith(root+"/") for root in prefix)}
    _require(all(any(task == node or task.startswith(node+"/") for node in semantic) for task in leaves)
             and [segment["selected_node_ids"] for segment in entry_value["segments"]] == native,
             "entry projection omitted original work or its independent native preservation")
    return value


def validate_reached_entry_preservation(value, position, group, epoch, plan, *, original_tasks,
                                        capture_facts=None, gate_identity=None, private_resources=None):
    """Check emitted links only; initializedness still comes from independent facts."""
    version = value.get("version") if isinstance(value, dict) else None
    _require(type(version) is int and version in (3, 4), "reached entry preservation version differs")
    fields = _BASE | ({"prefix_projection", "original_prefix_segments"} if version == 4 else set())
    _record(value, fields, version, "original_invocation_entry_definition_preservation", "reached entry preservation")
    validate_source_position(position, group, epoch, plan, original_tasks=original_tasks)
    _require(value["procedure"] == plan["entry"] and value["source_inputs"] == epoch["source_inputs"]
             and value["source_graph"] == group["source_graph"]
             and value["reached_position_identity"] == position["identity"]
             and value["entry_capture_facts_identity"] == plan["capture_facts_identity"]
             and (capture_facts is None or _digest(capture_facts) == plan["capture_facts_identity"]),
             "reached entry preservation original position, source or independent facts differ")
    _require(all(value[flag] is False for flag in _FLAGS)
             and value["definition_source"] == "independent original invocation-entry WHOLE facts after original entry undefinition"
             and value["prefix"] == "exact prior original statements and reached predicates; no enclosing loop or association"
             and value["semantics"] == "preserve existing definedness; no stale replica or unchanged-value assumption",
             "reached entry preservation cannot grant new definitions or current replicas")
    task_ids = {task["id"] for task in original_tasks}
    sequence = plan["entry"]+"/body"
    children = [sequence+"/"+str(number) for number in _direct_children(task_ids, sequence)]
    prefix = children[:children.index(position["activation_node_id"])]
    segments = value["segments"]
    _require(isinstance(segments, list), "reached entry preservation requires original native segments")
    for segment in segments:
        _require(isinstance(segment, dict) and set(segment) == {"selected_node_ids", "preservation_identity"}
                 and _sha256(segment["preservation_identity"]), "reached entry preservation native proof link differs")
        _segment_ids(segment["selected_node_ids"], plan["entry"])
    if version == 3:
        _require([segment["selected_node_ids"] for segment in segments] == ([prefix] if prefix else []),
                 "reached entry preservation omitted or reordered original root prefix")
    else:
        _projection(value["prefix_projection"], value, group, epoch, plan, prefix, task_ids,
                    gate_identity=gate_identity, private_resources=private_resources)
    roots = value["preserved_resources"]
    available = set(group["resources"]) | set(group["descriptor_guard_resources"])
    _require(isinstance(roots, list) and bool(roots) and all(_text(root) for root in roots)
             and len(set(roots)) == len(roots) and set(roots) <= available,
             "reached entry preservation independent resource inventory differs")
    canonical = {row["resource"]: row for row in epoch["resources"]}
    for root in roots:
        if root in canonical:
            descriptor = canonical[root]["descriptor"]
            attributes = descriptor.get("attributes")
            _require(type(descriptor.get("rank")) is int and 1 <= descriptor["rank"] <= 4
                     and descriptor.get("type") in {"real", "integer", "logical"}
                     and descriptor.get("intent") != "out" and isinstance(attributes, list)
                     and all(_text(item) for item in attributes) and not _FORBIDDEN.intersection(attributes),
                     "reached entry preservation cannot carry OUT or exposed descriptor definitions")
        if capture_facts is not None:
            fact = capture_facts.get("captures", {}).get(root)
            _require(isinstance(fact, dict) and fact.get("initialized") == "whole"
                     and fact.get("lifetime", "invocation") == "invocation"
                     and fact.get("storage") == "stable" and fact.get("escapes") is False
                     and fact.get("allocation_changes") is False
                     and not any(key in fact for key in ("source_sections", "requires_registered_handle")),
                     "reached entry preservation requires independent original invocation-entry WHOLE facts")
    return value
