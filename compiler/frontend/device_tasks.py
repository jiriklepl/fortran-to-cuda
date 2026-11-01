"""Source-bound admission for an explicitly qualified external device call.

The sidecar supplies numerical equivalence and an adapter ABI, never storage
lifetime, definitions, placement estimates, or permission to move a call. The
compiler retains the exact original association and rechecks every artifact.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path

from compiler.frontend.call_bindings import resolve_original_call
from compiler.frontend.source_effects import _kind
from compiler.ir import CompilationError

_ISSUER = object()
_IDENTIFIER = re.compile(r"[a-z][a-z0-9_]{0,62}\Z")
_COMPONENT = re.compile(r"[a-z][a-z0-9_]{0,62}(?:%[a-z][a-z0-9_]{0,62})*\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_KINDS = {"int32": (-(1 << 31), (1 << 31) - 1), "int64": (-(1 << 63), (1 << 63) - 1),
          "size": (0, (1 << 63) - 1), "real64": None}


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _keys(value, required, optional=()):
    if not isinstance(value, dict) or set(value) - set(required) - set(optional) or set(required) - set(value):
        raise CompilationError("external device contract has missing or unknown fields")


def _reference(value, *, path_key="path"):
    _keys(value, (path_key, "sha256"))
    path, digest = value[path_key], value["sha256"]
    if not isinstance(path, str) or not Path(path).is_absolute() or not isinstance(digest, str) or not _SHA.fullmatch(digest):
        raise CompilationError("external device artifacts require absolute paths and SHA256 identities")
    try:
        if sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise CompilationError("external device artifact changed: " + path)
    except OSError as error:
        raise CompilationError("external device artifact is unavailable: " + path) from error
    return path


def task_scope_identity(contract):
    """Bind qualified adapter regions and configuration to one original body."""
    record = {"original_body": contract["source"]["callee_body_sha256"],
              "resources": contract["resources"], "configuration": contract["configuration"]}
    if contract.get("version") == 2:
        record.update(opaque_arguments=contract["opaque_arguments"], completion=contract["completion"])
    if "entry_read_requirements" in contract:
        record["entry_read_requirements"] = contract["entry_read_requirements"]
    return sha256(_encoded(record).encode()).hexdigest()


def _provider(reference, body_digest, task_scope, version=1):
    path = _reference(reference)
    try:
        record = json.loads(Path(path).read_text())
        _encoded(record)
    except (OSError, TypeError, ValueError) as error:
        raise CompilationError("external device provider requires finite JSON") from error
    _keys(record, ("version", "kind", "fortran", "artifacts", "numerical_contract", "execution",
                   *(("metadata_query",) if version == 2 else ())),
          ("pressure_backend", "retained_state_qualification"))
    if type(record["version"]) is not int or record["version"] != version or record["kind"] != "external_device_provider":
        raise CompilationError("unsupported external device provider version")
    interface = record["fortran"]
    _keys(interface, ("module", "config_type", "create", "execute", "destroy"))
    if any(not isinstance(name, str) or not _IDENTIFIER.fullmatch(name) for name in interface.values()):
        raise CompilationError("external device Fortran interface requires ordinary identifiers")
    artifacts = record["artifacts"]
    if not isinstance(artifacts, dict) or not artifacts:
        raise CompilationError("external device provider requires immutable implementation artifacts")
    for artifact, digest in artifacts.items():
        _reference({"path": artifact, "sha256": digest})
    numeric = record["numerical_contract"]
    _keys(numeric, ("identity", "qualification", "original_bodies", "task_contracts"))
    if (not isinstance(numeric["identity"], str) or not _SHA.fullmatch(numeric["identity"])
            or not isinstance(numeric["original_bodies"], list) or body_digest not in numeric["original_bodies"]
            or any(not isinstance(digest, str) or not _SHA.fullmatch(digest) for digest in numeric["original_bodies"])):
        raise CompilationError("external device numerical contract does not bind the exact original body")
    if (not isinstance(numeric["task_contracts"], list) or task_scope not in numeric["task_contracts"]
            or any(not isinstance(digest, str) or not _SHA.fullmatch(digest) for digest in numeric["task_contracts"])):
        raise CompilationError("external device regions/configuration are outside the explicitly qualified adapter scope")
    _reference(numeric["qualification"])
    if record["execution"] != {"stream": "supplied", "completion": "same_stream", "workspace": "domain_claim",
                                "fallback": "prework_only"}:
        raise CompilationError("external device provider lacks the checked stream/workspace/error contract")
    if "pressure_backend" in record:
        _reference(record["pressure_backend"], path_key="manifest")
    if version == 2:
        query = record["metadata_query"]
        _keys(query, ("module", "procedure", "opaque_formals", "configuration_argument", "read_only",
                      "configuration_components"))
        if (any(not isinstance(query[key], str) or not _IDENTIFIER.fullmatch(query[key])
                for key in ("module", "procedure")) or query["configuration_argument"] != "config"
                or query["read_only"] is not True or not isinstance(query["opaque_formals"], list)
                or not query["opaque_formals"]
                or any(not isinstance(name, str) or not _IDENTIFIER.fullmatch(name)
                       for name in query["opaque_formals"])
                or len(set(query["opaque_formals"])) != len(query["opaque_formals"])):
            raise CompilationError("external metadata query requires exact opaque formals and a typed readonly interface")
        components = query["configuration_components"]
        if not isinstance(components, list) or not components:
            raise CompilationError("external metadata query requires explicit typed configuration components")
        seen = set()
        for component in components:
            _keys(component, ("component", "kind", "count"))
            if (not isinstance(component["component"], str) or not _COMPONENT.fullmatch(component["component"])
                    or component["component"] in seen or not isinstance(component["kind"], str)
                    or component["kind"] not in _KINDS
                    or type(component["count"]) is not int or not 1 <= component["count"] <= 64):
                raise CompilationError("external metadata query configuration is not a bounded typed component list")
            seen.add(component["component"])
    if "retained_state_qualification" in record:
        if version != 2:
            raise CompilationError("provider-only retained state requires external provider version2")
        from compiler.frontend.provider_lifetime import retained_state_qualification
        retained_state_qualification(record)
    return record


def _expression(value, formals):
    if type(value) is int and -(1 << 63) <= value < (1 << 63):
        return
    _keys(value, ("descriptor", "formal", "axis", "offset"))
    formal = formals.get(value["formal"]) if isinstance(value["formal"], str) else None
    if (not isinstance(value["descriptor"], str) or value["descriptor"] not in {"size", "lbound"}
            or formal is None or type(value["axis"]) is not int
            or not 1 <= value["axis"] <= formal.rank or type(value["offset"]) is not int
            or not -(1 << 31) <= value["offset"] < (1 << 31)):
        raise CompilationError("external device expression requires an exact reached descriptor")


def _source_effects(analysis, callee, resources):
    summary = analysis.summarize(callee.qualified)
    if (not summary["complete"] or summary.get("persistent_state") or summary.get("definition_changes")
            or summary.get("openmp_directives") or summary.get("capture_lifetime_requirements")):
        raise CompilationError("external device first slice cannot erase persistent, parallel or unknown original effects")
    actions = {item["formal"]: set() for item in resources}
    local = {binding.root: binding for binding in callee.scope.bindings.values() if binding.name not in callee.arguments}
    for operation in summary["operations"]:
        kind, root = operation["kind"], operation.get("resource")
        if kind not in {"read", "write", "overwrite", "descriptor_read"}:
            raise CompilationError("external device first slice cannot erase original host or call effects")
        if root in {"argument::" + name for name in actions}:
            if kind != "descriptor_read":
                actions[root.removeprefix("argument::")].add(kind)
        else:
            binding = local.get(root)
            if (binding is None or binding.rank or binding.attributes & {"save", "pointer", "allocatable", "volatile",
                                                                        "asynchronous"}):
                raise CompilationError("external device first slice cannot hide an original host storage effect")
    for resource in resources:
        supplied = {effect["kind"] for effect in resource["effects"]}
        original = actions[resource["formal"]]
        if ("read" in original and "read" not in supplied
                or bool(original & {"write", "overwrite"}) != bool(supplied & {"write", "overwrite"})):
            raise CompilationError("external device resource effects omit or invent original payload actions")
    return {"summary_identity": summary["summary_identity"], "summary_version": summary["summary_version"],
            "complete": True, "effects": {formal: sorted(kinds) for formal, kinds in actions.items()},
            "host_effects": False, "persistent_state": False}


def _derive(analysis, routine, node, resolved, contract):
    scope = analysis.source_scope_for(node, routine.scope)
    resolved.validate(analysis, scope, node)
    analysis.verify_source()
    version = contract.get("version") if isinstance(contract, dict) else None
    _keys(contract, ("version", "source", "provider_manifest", "resources", "configuration",
                     *(("opaque_arguments", "completion") if version == 2 else ())),
          ("entry_read_requirements",) if version == 2 else ())
    if type(version) is not int or version not in {1, 2}:
        raise CompilationError("unsupported external device task version")
    source = contract["source"]
    _keys(source, ("sha256", "span", "call_sha256", "callee", "callee_body_sha256"))
    span = tuple(getattr(node.item, "fort_original_span", node.item.span))
    body_digest = sha256(str(resolved.callee.scope.node).encode()).hexdigest()
    if source != {"sha256": analysis.sources[str(routine.scope.path)], "span": list(span),
                  "call_sha256": sha256(str(node).encode()).hexdigest(), "callee": resolved.callee.qualified,
                  "callee_body_sha256": body_digest}:
        raise CompilationError("external device task lost its exact original CALL/body identity")
    provider = _provider(contract["provider_manifest"], body_digest, task_scope_identity(contract), version)
    opaque = ()
    if version == 2:
        from compiler.frontend.opaque_library_state import original_opaque_state
        contracts = contract["opaque_arguments"]
        if (contract["completion"] != "wait_without_export" or not isinstance(contracts, list) or not contracts
                or any(not isinstance(item, dict) or not isinstance(item.get("formal"), str) for item in contracts)
                or [item["formal"] for item in contracts] != provider["metadata_query"]["opaque_formals"]):
            raise CompilationError("external device v2 requires exact original opaque arguments and a completion frontier")
        opaque = tuple(original_opaque_state(analysis, routine, node, resolved, item["formal"], item)
                       for item in contracts)
        if len({item.binding.root for item in opaque}) != len(opaque):
            raise CompilationError("external opaque arguments must be distinct original objects")
    opaque_formals = {item.public()["formal"] for item in opaque}
    associations = tuple(item for item in resolved.associations if item.formal not in opaque_formals)
    if (not associations or any(item.binding is None or _kind(item.actual) != "Name" or item.binding.rank < 1
            or item.binding.signature()[:2] not in {("real", 4), ("real", 8), ("integer", 4)}
            or item.binding.attributes & {"optional", "pointer", "volatile", "asynchronous", "value"}
            or item.formal_binding.attributes & {"optional", "pointer", "allocatable", "value"}
            or item.formal_binding.intent not in ({"in", "inout", "out"} if version == 2 else {"in", "inout"})
            or item.formal_binding.signature() != item.binding.signature() for item in associations)):
        raise CompilationError("external device first slice requires exact ordinary whole-array arguments")
    bindings = tuple(item.binding for item in associations)
    if len({binding.root for binding in bindings}) != len(bindings):
        raise CompilationError("external device arguments require distinct canonical storage roots")
    formals = {item.formal: item.binding for item in associations}
    resources = contract["resources"]
    if (not isinstance(resources, list) or [item.get("formal") for item in resources if isinstance(item, dict)]
            != [item.formal for item in associations]):
        raise CompilationError("external device resources must follow the exact original formal order")
    for resource in resources:
        _keys(resource, ("formal", "effects"))
        effects, binding = resource["effects"], formals[resource["formal"]]
        if not isinstance(effects, list) or not effects:
            raise CompilationError("external device resource requires explicit exact effects")
        for effect in effects:
            _keys(effect, ("kind", "lower", "upper"))
            if (not isinstance(effect["kind"], str) or effect["kind"] not in {"read", "write", "overwrite"}
                    or not isinstance(effect["lower"], list) or not isinstance(effect["upper"], list)
                    or len(effect["lower"]) != binding.rank or len(effect["upper"]) != binding.rank):
                raise CompilationError("external device effect requires a typed physical half-open box")
            for expression in (*effect["lower"], *effect["upper"]):
                _expression(expression, formals)
    if version == 1:
        source_effects = _source_effects(analysis, resolved.callee, resources)
    else:
        resource_effects = {item["formal"]: sorted({effect["kind"] for effect in item["effects"]})
                            for item in resources}
        for item in associations:
            effects = set(resource_effects[item.formal])
            if (item.formal_binding.intent == "in" and effects != {"read"}
                    or item.formal_binding.intent == "out" and ("read" in effects or not effects & {"write", "overwrite"})):
                raise CompilationError("qualified external effects contradict the original formal INTENT")
        source_effects = {"version": 2, "original_body_sha256": body_digest,
            "resource_effects": resource_effects,
            "entry_definition_kills": [item.formal for item in associations if item.formal_binding.intent == "out"],
            "opaque_state": [item.public() for item in opaque],
            "qualification": provider["numerical_contract"]["qualification"],
            "wrapper_effects": "unchanged original source; only exact inner CALL replaced"}
    configuration = contract["configuration"]
    if not isinstance(configuration, list):
        raise CompilationError("external device configuration requires explicit component assignments")
    seen = set()
    for assignment in configuration:
        _keys(assignment, ("component", "kind", "values"))
        component, kind, values = assignment["component"], assignment["kind"], assignment["values"]
        if (not isinstance(component, str) or not _COMPONENT.fullmatch(component) or component in seen
                or not isinstance(kind, str) or kind not in _KINDS or not isinstance(values, list) or not values):
            raise CompilationError("external device configuration has an invalid component/kind")
        seen.add(component)
        for value in values:
            if kind == "real64":
                try:
                    valid = type(value) in (int, float) and math.isfinite(value)
                except OverflowError:
                    valid = False
                if not valid:
                    raise CompilationError("external device REAL configuration requires finite literals")
            else:
                _expression(value, formals)
                if type(value) is int and not _KINDS[kind][0] <= value <= _KINDS[kind][1]:
                    raise CompilationError("external device INTEGER configuration is out of range")
    if version == 2 and seen & {item["component"] for item in provider["metadata_query"]["configuration_components"]}:
        raise CompilationError("external metadata query and explicit configuration cannot overwrite the same component")
    if "entry_read_requirements" in contract:
        from compiler.frontend.external_entry_reads import derive_entry_reads

        source_effects["entry_read_requirements"] = derive_entry_reads(analysis, resolved, contract, provider)
    graph, selected = analysis._selected_source(routine.qualified, (node,))
    return {"version": version, "procedure": routine.qualified, "source_graph": graph.identity,
            "selected_node_ids": list(selected), "source_inputs": analysis.inputs.identity(),
            "call": resolved.public(), "contract": contract, "provider": provider,
            "source_effects": source_effects,
            "canonical_resources": {item.formal: item.binding.root for item in associations},
            "source_association_authorized": True, "definitions_authorized": False,
            "placement_authorized": False, "numerical_equivalence": "explicit supplied adapter qualification",
            "lifetime": "existing checked capture and current caller-owned domain only",
            "position": "exact original reached CALL; no motion or eager descriptor reads"}


@dataclass(frozen=True, slots=True)
class ExternalDeviceTaskProof:
    identity: str
    bindings: tuple
    _analysis: object = field(repr=False, compare=False)
    _routine: object = field(repr=False, compare=False)
    _node: object = field(repr=False, compare=False)
    _resolved: object = field(repr=False, compare=False)
    _contract: str = field(repr=False, compare=False)
    _record: str = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)

    def validate(self, analysis, routine, node):
        if (self._issuer is not _ISSUER or analysis is not self._analysis or routine is not self._routine
                or node is not self._node or getattr(analysis, "_external_device_task_proofs", {}).get(self.identity) is not self):
            raise CompilationError("external device task requires its exact compiler-issued original proof")
        current = _task_contract(analysis, routine, node)
        if current is None or _encoded(current) != self._contract:
            raise CompilationError("external device task contract changed after source admission")
        if _encoded(_derive(analysis, routine, node, self._resolved, current)) != self._record:
            raise CompilationError("external device task source, binding or provider authority changed")
        associations = tuple(item for item in self._resolved.associations if item.binding is not None and item.binding.rank)
        if any(binding is not item.binding for binding, item in zip(self.bindings, associations, strict=True)):
            raise CompilationError("external device original physical binding changed")
        return self

    def public(self):
        return {"identity": self.identity, **json.loads(self._record)}


def _task_contract(analysis, routine, node):
    caller = analysis.contracts.get(routine.qualified, {})
    if not isinstance(caller, dict) or "device_tasks" not in caller:
        return None
    tasks = caller["device_tasks"]
    if not isinstance(tasks, list):
        raise CompilationError("external device tasks must be an explicit caller list")
    span = list(getattr(node.item, "fort_original_span", node.item.span))
    matches = [task for task in tasks if isinstance(task, dict) and isinstance(task.get("source"), dict)
               and task["source"].get("span") == span]
    if len(matches) > 1:
        raise CompilationError("external device CALL has duplicate supplied contracts")
    return matches[0] if matches else None


def external_device_task(analysis, routine, node):
    if _kind(node) != "Call_Stmt":
        return None
    contract = _task_contract(analysis, routine, node)
    if contract is None:
        return None
    for proof in getattr(analysis, "_external_device_task_proofs", {}).values():
        if proof._node is node:
            return proof.validate(analysis, routine, node)
    resolved = resolve_original_call(analysis, analysis.source_scope_for(node, routine.scope), node)
    encoded = _encoded(_derive(analysis, routine, node, resolved, contract))
    proof = ExternalDeviceTaskProof(sha256(encoded.encode()).hexdigest(), tuple(item.binding for item in resolved.associations
                                                                           if item.binding is not None and item.binding.rank),
        analysis, routine, node, resolved, _encoded(contract), encoded, _ISSUER)
    registry = getattr(analysis, "_external_device_task_proofs", None)
    if registry is None:
        registry = analysis._external_device_task_proofs = {}
    registry[proof.identity] = proof
    return proof.validate(analysis, routine, node)
