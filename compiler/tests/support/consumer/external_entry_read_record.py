"""Integrity of explicit library entry-read prerequisites; no source authority."""
from __future__ import annotations

import json

from .checked_execution_plan import _digest, _require
from .device_task_record import _expression, _keys, _reference

_FLAGS = {"may_read_permissions_are_entry_requirements": False,
          "unreached_definition_authority": False, "halo_definition_authority": False,
          "alias_authority": False, "lifetime_authority": False, "execution_authority": False}
_CLAIM = "unconditional original entry reads before any same-root write under accepted query and layout predicates"


def validate_external_entry_reads(record, task, ranks, associations):
    _keys(record, ("version", "qualification", "qualification_record", "canonical_resources", "admission",
                   "existing_device_values_preserved", *_FLAGS))
    contract = task["contract"]
    _require(record["version"] == 1 and type(record["version"]) is int
             and record["qualification"] == contract["entry_read_requirements"]
             and all(record[key] is value for key, value in _FLAGS.items())
             and record["existing_device_values_preserved"] is True
             and record["admission"] == "only missing original entry reads after guards and successful provider creation at original CALL",
             "external entry-read authority or reached admission differs")
    path = _reference(record["qualification"])
    q = json.loads(path.read_text())
    _require(_digest(q) == _digest(record["qualification_record"]), "external entry-read qualification differs")
    _keys(q, ("version", "kind", "claim", "authority", "source_inputs", "task_scope", "metadata_query",
              "provider_binding", "required_reads", "layout_equalities", "evidence"))
    _require(type(q["version"]) is int and q["version"] == 1
             and q["kind"] == "qualified_original_external_entry_reads" and q["claim"] == _CLAIM
             and isinstance(q["authority"], dict) and set(q["authority"]) == set(_FLAGS)
             and all(q["authority"][key] is value for key, value in _FLAGS.items())
             and _digest(q["source_inputs"]) == _digest(task["source_inputs"])
             and q["task_scope"] == _digest({key: contract[key] for key in (
                 "source", "resources", "configuration", "opaque_arguments", "completion")})
             and _digest(q["metadata_query"]) == _digest(task["provider"]["metadata_query"])
             and _digest(q["provider_binding"]) == _digest({key: task["provider"][key] for key in (
                 "fortran", "artifacts", "metadata_query", "execution", "pressure_backend") if key in task["provider"]}),
             "external entry reads differ from exact source/task/query or unconditional prerequisite")
    _require(isinstance(q["evidence"], list) and bool(q["evidence"]), "external entry reads lack reviewed evidence")
    for evidence in q["evidence"]:
        _reference(evidence)
    resources = {item["formal"]: item for item in contract["resources"]}
    mapping = {item["formal"]: item for item in associations}
    seen = set()
    _require(isinstance(q["required_reads"], list) and bool(q["required_reads"]), "external entry reads lack regions")
    for item in q["required_reads"]:
        _keys(item, ("formal", "lower", "upper"))
        name = item["formal"]
        _require(isinstance(name, str) and name in resources and name not in seen
                 and mapping[name]["formal_descriptor"]["intent"] == "in"
                 and {effect["kind"] for effect in resources[name]["effects"]} == {"read"}
                 and isinstance(item["lower"], list) and isinstance(item["upper"], list)
                 and len(item["lower"]) == len(item["upper"]) == ranks[name]
                 and {"kind": "read", "lower": item["lower"], "upper": item["upper"]} in resources[name]["effects"],
                 "external entry requirement differs from exact readonly region")
        for value in (*item["lower"], *item["upper"]):
            _expression(value, ranks)
        seen.add(name)
    _require(record["canonical_resources"] == {name: task["canonical_resources"][name] for name in sorted(seen)},
             "external entry-read canonical mapping differs")
    components = {item["component"]: item for item in q["metadata_query"]["configuration_components"]}
    _require(isinstance(q["layout_equalities"], list) and bool(q["layout_equalities"]), "external entry reads lack layout guards")
    guards, covered = set(), set()
    for guard in q["layout_equalities"]:
        _keys(guard, ("component", "index", "descriptor"))
        component, descriptor = guard["component"], guard["descriptor"]
        entry = components.get(component) if isinstance(component, str) else None
        _require(entry is not None and entry["kind"] in ("int32", "int64", "size")
                 and type(guard["index"]) is int and 1 <= guard["index"] <= entry["count"]
                 and isinstance(descriptor, dict) and descriptor.get("descriptor") == "size"
                 and isinstance(descriptor.get("formal"), str) and descriptor.get("formal") in resources and _digest(guard) not in guards,
                 "external entry layout predicate differs")
        _expression(descriptor, ranks)
        covered.add((descriptor["formal"], descriptor["axis"]))
        guards.add(_digest(guard))
    _require(covered == {(name, axis) for name, rank in ranks.items() for axis in range(1, rank + 1)},
             "external entry layout guards must cover every physical array axis")
