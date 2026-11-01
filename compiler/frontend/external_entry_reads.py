"""Explicit qualified entry preconditions for one original external CALL.

Provider access permissions never establish these requirements. A reviewed
library contract supplies unconditional reads-before-writes under the exact
accepted readonly query and physical-layout predicates. No incoming definition,
entry/lifetime, numerical or alias authority is created by this module.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from compiler.ir import CompilationError

EXTERNAL_ENTRY_READ_REQUIREMENTS_VERSION = 1
_CLAIM = "unconditional original entry reads before any same-root write under accepted query and layout predicates"
_FLAGS = {"may_read_permissions_are_entry_requirements": False,
          "unreached_definition_authority": False, "halo_definition_authority": False,
          "alias_authority": False, "lifetime_authority": False, "execution_authority": False}


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def requirement_scope(contract):
    keys = ("source", "resources", "configuration", "opaque_arguments", "completion")
    return sha256(_encoded({key: contract[key] for key in keys}).encode()).hexdigest()


def provider_read_binding(provider):
    """Pin the query implementation artifacts, not only its Fortran spelling."""
    return {key: provider[key] for key in ("fortran", "artifacts", "metadata_query", "execution", "pressure_backend")
            if key in provider}


def _keys(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        raise CompilationError("external entry-read qualification has missing or unknown fields")


def derive_entry_reads(analysis, resolved, contract, provider):
    """Bind a distinct trusted library precondition, never a may-read summary."""
    from compiler.frontend.device_tasks import _expression, _reference

    reference = contract.get("entry_read_requirements")
    if reference is None:
        return None
    if contract["version"] != 2:
        raise CompilationError("external entry reads require the exact opaque v2 CALL")
    path = _reference(reference)
    try:
        record = json.loads(Path(path).read_text())
        _encoded(record)
    except (OSError, TypeError, ValueError) as error:
        raise CompilationError("external entry-read qualification requires finite JSON") from error
    _keys(record, ("version", "kind", "claim", "authority", "source_inputs", "task_scope",
                   "metadata_query", "provider_binding", "required_reads", "layout_equalities", "evidence"))
    if (type(record["version"]) is not int or record["version"] != 1
            or record["kind"] != "qualified_original_external_entry_reads"
            or record["claim"] != _CLAIM or type(record["authority"]) is not dict
            or set(record["authority"]) != set(_FLAGS)
            or any(record["authority"][key] is not value for key, value in _FLAGS.items())
            or _encoded(record["source_inputs"]) != _encoded(analysis.inputs.identity())
            or record["task_scope"] != requirement_scope(contract)
            or _encoded(record["metadata_query"]) != _encoded(provider["metadata_query"])
            or _encoded(record["provider_binding"]) != _encoded(provider_read_binding(provider))):
        raise CompilationError("external entry reads differ from their exact source/task/query or unconditional prerequisite")
    if not isinstance(record["evidence"], list) or not record["evidence"]:
        raise CompilationError("external entry reads need explicit reviewed source/library evidence")
    for evidence in record["evidence"]:
        _reference(evidence)
    formals = {item.formal: item for item in resolved.associations}
    readable = {item["formal"]: item for item in contract["resources"]}
    requirements = record["required_reads"]
    if not isinstance(requirements, list) or not requirements:
        raise CompilationError("external entry reads require explicit nonempty regions")
    seen = set()
    for item in requirements:
        _keys(item, ("formal", "lower", "upper"))
        formal = item["formal"]
        mapping = formals.get(formal) if isinstance(formal, str) else None
        if (mapping is None or mapping.binding is None or mapping.formal_binding.intent != "in"
                or not mapping.binding.rank or formal in seen or formal not in readable
                or {effect["kind"] for effect in readable[formal]["effects"]} != {"read"}
                or not isinstance(item["lower"], list) or not isinstance(item["upper"], list)
                or len(item["lower"]) != len(item["upper"]) or len(item["lower"]) != mapping.binding.rank
                or {"kind": "read", "lower": item["lower"], "upper": item["upper"]}
                not in readable[formal]["effects"]):
            raise CompilationError("external entry prerequisite requires an exact readonly original region, not a permission superset or entry-killed formal")
        seen.add(formal)
        bindings = {name: association.binding for name, association in formals.items() if association.binding is not None}
        for value in (*item["lower"], *item["upper"]):
            _expression(value, bindings)
    components = {item["component"]: item for item in provider["metadata_query"]["configuration_components"]}
    guards = record["layout_equalities"]
    if not isinstance(guards, list) or not guards:
        raise CompilationError("external entry reads require exact query/layout predicates")
    guard_keys, covered = set(), set()
    for guard in guards:
        _keys(guard, ("component", "index", "descriptor"))
        component = guard["component"]
        entry = components.get(component) if isinstance(component, str) else None
        expression = guard["descriptor"]
        if (entry is None or entry["kind"] not in {"int32", "int64", "size"}
                or type(guard["index"]) is not int or not 1 <= guard["index"] <= entry["count"]
                or type(expression) is not dict or expression.get("descriptor") != "size"
                or not isinstance(expression.get("formal"), str) or expression.get("formal") not in readable
                or _encoded(guard) in guard_keys):
            raise CompilationError("external entry layout predicate differs from exact typed query descriptors")
        _expression(expression, {name: value.binding for name, value in formals.items() if value.binding is not None})
        guard_keys.add(_encoded(guard))
        covered.add((expression["formal"], expression["axis"]))
    expected_axes = {(formal, axis) for formal in readable for axis in range(1, formals[formal].binding.rank + 1)}
    if covered != expected_axes:
        raise CompilationError("external entry layout guards must cover every physical array axis")
    analysis.inputs.verify_live()
    return {"version": 1, "qualification": reference, "qualification_record": record,
            "canonical_resources": {formal: formals[formal].binding.root for formal in sorted(seen)},
            "admission": "only missing original entry reads after guards and successful provider creation at original CALL",
            "existing_device_values_preserved": True, **_FLAGS}


@dataclass(frozen=True, slots=True)
class ExternalEntryReadRequirements:
    """A view of an exact existing issued ExternalDeviceTaskProof."""

    _task: object
    _record: str

    def validate(self, analysis, routine, node):
        from compiler.frontend.device_tasks import ExternalDeviceTaskProof

        if type(self._task) is not ExternalDeviceTaskProof:
            raise CompilationError("external entry reads require their exact external CALL proof")
        self._task.validate(analysis, routine, node)
        current = self._task.public()["source_effects"].get("entry_read_requirements")
        if current is None or _encoded(current) != self._record:
            raise CompilationError("external entry-read prerequisite changed")
        return self

    def public(self):
        return json.loads(self._record)


def external_entry_read_requirements(task):
    from compiler.frontend.device_tasks import ExternalDeviceTaskProof

    if type(task) is not ExternalDeviceTaskProof:
        raise CompilationError("external entry reads require an issued external CALL")
    task.validate(task._analysis, task._routine, task._node)
    record = task.public()["source_effects"].get("entry_read_requirements")
    return None if record is None else ExternalEntryReadRequirements(task, _encoded(record))
