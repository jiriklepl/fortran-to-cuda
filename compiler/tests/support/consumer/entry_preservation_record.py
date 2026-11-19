"""Transport definedness preservation; never recreate compiler source authority."""
from __future__ import annotations

import re

from .checked_execution_plan import _digest, _require, _sha256, _text

_FLAGS = {
    "new_definition_authority", "execution_authorized", "payload_access_authorized",
    "descriptor_lifetime_authority", "alias_authority", "cross_exit_retention_authorized",
}
_FIELDS = _FLAGS | {
    "identity", "version", "kind", "procedure", "source_inputs", "source_graph",
    "sequence_exit_identity", "segments", "entry_capture_facts_identity", "preserved_resources",
    "definition_source", "prefix", "semantics",
}
_NUMBER = r"(?:0|[1-9][0-9]*)"
_FORBIDDEN = {"pointer", "target", "optional", "volatile", "asynchronous", "value"}


def _segment_ids(value, entry):
    _require(isinstance(value, list) and bool(value) and all(_text(node) for node in value)
             and len(set(value)) == len(value), "entry preservation requires distinct original segment nodes")
    grammar = re.escape(entry) + r"/body(?:/" + _NUMBER + r"(?:/body" + _NUMBER + r")?)*(?:/condition" + _NUMBER + r")?"
    _require(all(re.fullmatch(grammar, node) for node in value),
             "entry preservation segment original node grammar differs")
    positions = [node.rpartition("/") for node in value]
    _require(len({parent for parent, _separator, _position in positions}) == 1,
             "entry preservation segment crosses original source sequences")
    numbers = []
    kinds = set()
    for _parent, _separator, position in positions:
        condition = position.startswith("condition")
        kinds.add(condition)
        number = position[len("condition"):] if condition else position
        _require(re.fullmatch(_NUMBER, number) is not None,
                 "entry preservation segment has no original statement or predicate position")
        numbers.append(int(number))
    _require(len(kinds) == 1 and numbers == sorted(set(numbers)),
             "entry preservation segment original order differs")


def _direct_children(task_ids, sequence):
    """Collapse diagnostic IF alternatives to their original source child ID.

    Source positions can skip ordinary comments. The original graph inventory,
    rather than a numeric range, identifies every preceding semantic child.
    """
    prefix, positions = sequence + "/", set()
    for node in task_ids:
        if not node.startswith(prefix):
            continue
        part = node[len(prefix):].partition("/")[0]
        match = re.fullmatch("(" + _NUMBER + r")(?::alternative:" + _NUMBER + ")?", part)
        if match:
            positions.add(int(match[1]))
    return sorted(positions)


def _expected_segments(group, entry, task_ids, *, loop_invariant=False, loop_ids=frozenset()):
    """Cross-check the complete reached IF prefix in the existing graph grammar."""
    first = group["selected_node_ids"][0]
    prefix = entry + "/body/"
    _require(first.startswith(prefix), "entry preservation selected CALL source path differs")
    path = first[len(prefix):].split("/")
    sequence, expected, cursor = entry + "/body", [], 0
    while cursor < len(path):
        position = path[cursor]
        _require(re.fullmatch(_NUMBER, position) is not None,
                 "entry preservation selected CALL has no original sequence position")
        index = int(position)
        children = _direct_children(task_ids, sequence)
        _require(index in children, "entry preservation prefix differs from its original graph inventory")
        preceding = [sequence + "/" + str(number) for number in children if number < index]
        if preceding:
            expected.append(preceding)
        cursor += 1
        if cursor == len(path):
            break
        body = path[cursor]
        if body == "body" and loop_invariant:
            # The compiler proves a complete enclosing-loop invariant. The
            # portable reader requires that exact outermost container instead
            # of accepting a first-iteration prefix or a partial loop body.
            loop = sequence + "/" + position
            _require(loop in loop_ids and any(node.startswith(loop + "/body/") for node in task_ids),
                     "entry preservation lost its original loop body inventory")
            expected.append([loop])
            return expected
        _require(re.fullmatch("body" + _NUMBER, body) is not None,
                 "entry preservation cannot cross an unproved loop or association")
        ordinal = int(body[4:])
        # A forged ordinal cannot allocate a range larger than the supplied
        # complete graph. Source numeric positions themselves may have gaps.
        _require(ordinal <= len(task_ids), "entry preservation branch exceeds its original graph inventory")
        branch = sequence + "/" + position
        predicates = [branch + "/condition" + str(number) for number in range(max(1, ordinal))]
        _require(all(node in task_ids for node in predicates),
                 "entry preservation lost an evaluated original branch predicate")
        current = branch + "/condition" + str(ordinal)
        if ordinal and current in task_ids:
            predicates.append(current)
        expected.extend([node] for node in predicates)
        sequence = branch + "/" + body
        cursor += 1
    _require(not loop_invariant, "entry preservation loop invariant has no enclosing loop")
    return expected


def validate_entry_preservation(value, group, epoch, plan, *, original_tasks, capture_facts=None):
    """Check the compiler's exact links to independently supplied entry facts.

    Registered original parser objects and NativePreservationProof instances
    establish completeness in the compiler. This portable record cannot grant
    initializedness, payload access, unchanged bytes or a valid device replica.
    """
    _require(isinstance(value, dict) and set(value) == _FIELDS
             and type(value["version"]) is int and value["version"] in (2, 5)
             and value["kind"] == "original_invocation_entry_definition_preservation"
             and _sha256(value["identity"])
             and value["identity"] == _digest({key: item for key, item in value.items() if key != "identity"}),
             "entry preservation requires its exact versioned content identity")
    _require(value["procedure"] == plan["entry"] == epoch["entry"]
             and value["source_inputs"] == epoch["source_inputs"]
             and value["source_graph"] == group["sequence_exit"]["source_graph"]
             and value["sequence_exit_identity"] == group["sequence_exit"]["identity"]
             and value["entry_capture_facts_identity"] == plan["capture_facts_identity"],
             "entry preservation original source, exit or independent facts differ")
    _require(all(value[flag] is False for flag in _FLAGS)
             and value["definition_source"] == "independent original invocation-entry WHOLE facts after original entry undefinition"
             and value["prefix"] == (
                 "exact outer prefix and complete enclosing-loop preservation invariant; no association"
                 if value["version"] == 5 else
                 "exact prior original statements and reached predicates; no enclosing loop or association")
             and value["semantics"] == "preserve existing definedness; no stale replica or unchanged-value assumption",
             "entry preservation cannot grant new definitions, payload, lifetime, aliases or execution")
    segments = value["segments"]
    _require(isinstance(segments, list) and bool(segments),
             "entry preservation requires its complete ordered original prefix segments")
    seen = set()
    for segment in segments:
        _require(isinstance(segment, dict) and set(segment) == {"selected_node_ids", "preservation_identity"}
                 and _sha256(segment["preservation_identity"]),
                 "entry preservation requires exact native segment proof links")
        nodes = segment["selected_node_ids"]
        _segment_ids(nodes, plan["entry"])
        _require(not seen.intersection(nodes) and not set(group["selected_node_ids"]).intersection(nodes),
                 "entry preservation repeats a prefix node or includes a selected CALL")
        seen.update(nodes)
    _require(isinstance(original_tasks, list) and all(isinstance(task, dict) and _text(task.get("id"))
                                                    for task in original_tasks),
             "entry preservation needs its existing checked original task graph")
    task_ids = {task["id"] for task in original_tasks}
    loop_ids = {task["id"] for task in original_tasks if isinstance(task.get("loop"), dict)
                and task["loop"].get("name") == task["id"] + "/body"}
    _require([segment["selected_node_ids"] for segment in segments]
             == _expected_segments(group, plan["entry"], task_ids,
                                   loop_invariant=value["version"] == 5, loop_ids=loop_ids),
             "entry preservation omitted or reordered original prefix statements or predicates")
    roots = value["preserved_resources"]
    available = set(group["resources"]) | set(group["descriptor_guard_resources"])
    _require(isinstance(roots, list) and bool(roots) and all(_text(root) for root in roots)
             and len(set(roots)) == len(roots) and set(roots) <= available,
             "entry preservation independent resource inventory differs")
    canonical = {item["resource"]: item for item in epoch["resources"]}
    for root in roots:
        if root in canonical:
            descriptor = canonical[root]["descriptor"]
            attributes = descriptor.get("attributes")
            _require(type(descriptor.get("rank")) is int and 1 <= descriptor["rank"] <= 4
                     and descriptor.get("type") in {"real", "integer", "logical"}
                     and descriptor.get("intent") != "out"
                     and isinstance(attributes, list) and all(_text(item) for item in attributes)
                     and not _FORBIDDEN.intersection(attributes),
                     "entry preservation cannot carry original OUT or exposed descriptor definitions")
        if capture_facts is not None:
            fact = capture_facts.get("captures", {}).get(root)
            _require(isinstance(fact, dict) and fact.get("initialized") == "whole"
                     and fact.get("lifetime", "invocation") == "invocation"
                     and fact.get("storage") == "stable" and fact.get("escapes") is False
                     and fact.get("allocation_changes") is False
                     and not any(key in fact for key in ("source_sections", "requires_registered_handle")),
                     "entry preservation requires independent original invocation-entry WHOLE facts")
    return value
