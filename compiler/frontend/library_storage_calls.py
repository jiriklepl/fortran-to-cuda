"""Explicit pinned library storage effects, independent of device-task authority.

The supplied qualification is a scoped library/source effect contract. Fresh
original receiver, constructor and argument associations remain compiler-issued.
Nothing here permits numerical execution, infers payload regions or definitions,
or establishes separation of caller arrays.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from hashlib import sha256
from pathlib import Path

from fparser.two.utils import walk

from compiler.frontend.call_bindings import resolve_original_call
from compiler.frontend.device_tasks import _keys, _reference
from compiler.frontend.metadata_constructor import original_metadata_constructor
from compiler.frontend.opaque_library_state import original_opaque_state
from compiler.frontend.private_library_state import (
    PrivateLibraryStateDeclaration,
    _attached,
    _attachments,
    _consumed_call,
    private_library_state,
)
from compiler.frontend.source_effects import _kind
from compiler.ir import CompilationError

LIBRARY_STORAGE_CALL_VERSION = 1
DESCRIPTOR_WINDOW_LIBRARY_STORAGE_VERSION = 14
_ISSUER = object()


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _span(node):
    item = getattr(node, "item", None)
    return list(getattr(item, "fort_original_span", item.span)) if item is not None else None


def _qualification(analysis, reference, resolved):
    path = _reference(reference)
    try:
        value = json.loads(Path(path).read_text())
        _encode(value)
    except (OSError, ValueError, TypeError) as error:
        raise CompilationError("library storage qualification requires finite JSON") from error
    _keys(value, ("schema_version", "kind", "operation", "receiver_formal", "array_formals", "storage",
        "library_sources", "configured_export", "external_abi", "evidence", "premises", "authority",
        "qualification_scope", "status"))
    if (type(value["schema_version"]) is not int or value["schema_version"] != 1
            or value["kind"] != "explicit_pinned_synchronous_private_library_storage"
            or value["status"] != "qualified"):
        raise CompilationError("library storage requires its explicit qualified source/ABI contract")
    operation = value["operation"]
    _keys(operation, ("procedure", "body_sha256", "source"))
    _reference(operation["source"])
    callee = resolved.callee
    if (operation["procedure"] != callee.qualified or operation["source"]["path"] != str(callee.scope.path)
            or operation["source"]["sha256"] != analysis.sources.get(str(callee.scope.path))
            or operation["body_sha256"] != sha256(str(callee.scope.node).encode()).hexdigest()):
        raise CompilationError("library storage qualification changed its original operation")
    expected = {"caller_descriptor_mutations": False, "caller_address_retention": False,
        "caller_pointer_association": False, "outstanding_caller_borrower_at_return": False,
        "private_receiver_scratch_may_change": True,
        "receiver_scratch_origin": "fresh library allocator; no alias to live caller allocations",
        "hidden_managed_resources": []}
    # This first slice explicitly qualifies a pinned library implementation.
    # Library-global state can change; caller-storage facts remain separate.
    storage = value["storage"]
    if (not isinstance(storage, dict) or set(storage) != {*expected, "library_global_effects"}
            or {key: storage[key] for key in expected} != expected
            or not isinstance(storage["library_global_effects"], str) or not storage["library_global_effects"]
            or any(type(storage.get(key)) is not bool for key in
            ("caller_descriptor_mutations", "caller_address_retention", "caller_pointer_association",
             "outstanding_caller_borrower_at_return", "private_receiver_scratch_may_change"))):
        raise CompilationError("library storage qualification has unbounded descriptor, alias or escape effects")
    authority = {"caller_descriptor_preservation": True, "caller_noescape": True,
        "caller_alias_separation": False, "payload_access": False, "initializedness": False,
        "numerical_equivalence": False, "execution": False, "placement": False}
    if value["authority"] != authority or any(type(v) is not bool for v in value["authority"].values()):
        raise CompilationError("library storage contract exceeds independent storage-only authority")
    sources = {**analysis.inputs.sources, **analysis.inputs.dependencies}
    if not isinstance(value["library_sources"], list) or not value["library_sources"]:
        raise CompilationError("library storage requires its complete pinned source inventory")
    names = []
    for item in value["library_sources"]:
        _reference(item)
        if sources.get(item["path"]) != item["sha256"]:
            raise CompilationError("library storage source is outside the configured original inventory")
        names.append(item["path"])
    if len(set(names)) != len(names) or str(callee.scope.path) not in names:
        raise CompilationError("library storage source inventory is duplicate or misses its operation")
    export = _reference(value["configured_export"])
    if json.loads(Path(export).read_text()) != analysis.inputs.public():
        raise CompilationError("library storage configured source qualification changed")
    if not isinstance(value["external_abi"], list) or not value["external_abi"]:
        raise CompilationError("library storage requires pinned external ABI artifacts")
    for item in value["external_abi"]:
        _keys(item, ("soname", "artifact"))
        if not isinstance(item["soname"], str) or not item["soname"]:
            raise CompilationError("library storage ABI name is unavailable")
        _reference(item["artifact"])
    evidence = value["evidence"]
    _keys(evidence, ("source_audit", "independent_static_review", "abi_observation", "actual_receiver_declaration"))
    for item in evidence.values():
        _reference(item)
    if (not isinstance(value["premises"], list) or not value["premises"]
            or any(not isinstance(item, str) or not item for item in value["premises"])
            or not isinstance(value["qualification_scope"], str) or not value["qualification_scope"]):
        raise CompilationError("library storage qualification must retain its independent premises")
    return value


def _derive(analysis, routine, node, resolved, contract):
    if analysis._require_original(routine.qualified) is not routine or _kind(node) != "Call_Stmt":
        raise CompilationError("library storage requires its exact original CALL owner")
    _keys(contract, ("version", "source", "receiver", "array_formals", "qualification"))
    supplied = analysis.contracts.get(routine.qualified, {}).get("library_storage_calls", ())
    if not isinstance(supplied, list) or _encode(contract) not in [_encode(item) for item in supplied]:
        raise CompilationError("library storage lost its explicit original caller contract")
    supplied_identity = _encode(supplied)
    if type(contract["version"]) is not int or contract["version"] != LIBRARY_STORAGE_CALL_VERSION:
        raise CompilationError("unsupported library storage call version")
    source = contract["source"]
    _keys(source, ("sha256", "span", "call_sha256", "callee", "callee_body_sha256"))
    association = _consumed_call(analysis, routine, node, resolved)
    if (not isinstance(source["span"], list) or len(source["span"]) != 2
            or any(type(line) is not int for line in source["span"])
            or source["sha256"] != analysis.sources.get(str(routine.scope.path)) or source["span"] != _span(node)
            or source["call_sha256"] != sha256(str(node).encode()).hexdigest()
            or source["callee"] != resolved.procedure
            or source["callee_body_sha256"] != sha256(str(resolved.callee.scope.node).encode()).hexdigest()):
        raise CompilationError("library storage CALL changed its exact source or resolved body")
    qualification = _qualification(analysis, contract["qualification"], resolved)
    receiver = contract["receiver"]
    _keys(receiver, ("formal", "constructor", "qualification"))
    if (not isinstance(receiver["formal"], str) or receiver["formal"] not in resolved.formals
            or receiver["qualification"] != contract["qualification"]
            or receiver["formal"] != qualification["receiver_formal"]):
        raise CompilationError("library storage receiver lost independent qualification")
    arrays = contract["array_formals"]
    if (not isinstance(arrays, list) or not arrays or any(not isinstance(x, str) for x in arrays)
            or len(set(arrays)) != len(arrays) or receiver["formal"] in arrays
            or set(resolved.formals) != {receiver["formal"], *arrays}):
        raise CompilationError("library storage requires exact receiver and ordinary array formals")
    array_records = []
    for formal in arrays:
        item = next((item for item in resolved.associations if item.formal == formal), None)
        if (item is None or item.actual is None or item.binding is None or _kind(item.actual) != "Name"
                or item.formal_binding.dtype not in {"real", "integer"} or item.formal_binding.kind not in {4, 8}
                or not 1 <= item.formal_binding.rank <= 4 or item.formal_binding.intent not in {"in", "out", "inout"}
                or item.formal_binding.attributes & {"pointer", "allocatable", "target", "optional", "save", "value", "volatile", "asynchronous"}
                or item.binding.attributes & {"pointer", "target", "optional", "value", "volatile", "asynchronous"}):
            raise CompilationError("library storage requires exact ordinary nonescaping array associations")
        binding = item.formal_binding
        array_records.append({"name": formal, "type": binding.dtype, "kind": binding.kind,
                              "rank": binding.rank, "intent": binding.intent})
    qualified_arrays = qualification["array_formals"]
    if (array_records != qualified_arrays or any(not isinstance(item, dict)
            or type(item.get("kind")) is not int or type(item.get("rank")) is not int for item in qualified_arrays)):
        raise CompilationError("library storage qualification differs from original typed array formals")
    actuals = [item.binding.root for item in resolved.associations if item.formal in arrays]
    if len(set(actuals)) != len(actuals):
        raise CompilationError("library storage requires independent caller alias guards")
    opaque = original_opaque_state(analysis, routine, node, resolved, receiver["formal"], receiver)
    assignments = [item for item in walk(routine.execution) if _kind(item) == "Assignment_Stmt"
                   and _span(item) == receiver["constructor"]["span"]]
    if len(assignments) != 1:
        raise CompilationError("library storage lost its exact once-only metadata constructor")
    constructor = original_metadata_constructor(analysis, routine, assignments[0], opaque)
    declaration = private_library_state(analysis, routine, opaque.binding, opaque, constructor)
    record = {"version": 1, "kind": "qualified_original_private_library_storage_call",
        "procedure": routine.qualified, "source_inputs": analysis.inputs.identity(),
        "original_call": source, "association": association, "private_declaration": declaration.public(),
        "qualification": contract["qualification"], "array_formals": array_records,
        "descriptor_preservation": "explicit independently qualified pinned library storage contract",
        "caller_noescape": True, "caller_alias_authority": False,
        "entry_definition_authority": False, "payload_access_authorized": False,
        "execution_authorized": False, "numerical_equivalence_authorized": False}
    analysis.verify_publication()
    declaration.validate()
    if (_consumed_call(analysis, routine, node, resolved) != association
            or _qualification(analysis, contract["qualification"], resolved) != qualification
            or _encode(analysis.contracts.get(routine.qualified, {}).get("library_storage_calls")) != supplied_identity):
        raise CompilationError("library storage original association or qualification changed at publication")
    return _encode(record), declaration


@dataclass(frozen=True, slots=True)
class QualifiedLibraryStorageCall:
    identity: str
    _analysis: object = field(repr=False)
    _routine: object = field(repr=False)
    _node: object = field(repr=False)
    _resolved: object = field(repr=False)
    _contract: str = field(repr=False)
    _declaration: object = field(repr=False)
    _record: str = field(repr=False)
    _issuer: object = field(repr=False)

    def validate(self):
        if (type(self) is not QualifiedLibraryStorageCall or self._issuer is not _ISSUER
                or type(self._declaration) is not PrivateLibraryStateDeclaration
                or type(self._record) is not str or type(self._contract) is not str):
            raise CompilationError("library storage requires its exact compiler-issued original token")
        retained = tuple((item.name, getattr(self, item.name)) for item in fields(self))
        declaration_attachments = _attachments(self._declaration)
        analysis = self._analysis
        if (type(self) is not QualifiedLibraryStorageCall or self._issuer is not _ISSUER
                or getattr(analysis, "_library_storage_calls_owner", None) is not analysis
                or getattr(analysis, "_library_storage_calls", {}).get(self.identity) is not self
                or sha256(self._record.encode()).hexdigest() != self.identity):
            raise CompilationError("library storage requires its exact compiler-issued original token")
        self._declaration.validate()
        record, declaration = _derive(analysis, self._routine, self._node, self._resolved, json.loads(self._contract))
        # The declaration composition is ephemeral; its underlying existing
        # registered original receiver/constructor tokens are exact.
        if (record != self._record or declaration.public() != self._declaration.public()
                or not _attached(self._declaration, declaration_attachments)
                or declaration._opaque is not self._declaration._opaque
                or declaration._constructor is not self._declaration._constructor
                or any(getattr(self, name) is not item for name, item in retained)
                or getattr(analysis, "_library_storage_calls_owner", None) is not analysis
                or getattr(analysis, "_library_storage_calls", {}).get(self.identity) is not self):
            raise CompilationError("library storage source or original issuer attachment changed")
        return self

    def public(self):
        return {"identity": self.identity, **json.loads(self._record)}


def library_storage_call(analysis, routine, node, resolved=None):
    contracts = analysis.contracts.get(routine.qualified, {}).get("library_storage_calls", [])
    if not isinstance(contracts, list):
        raise CompilationError("library storage contracts require an explicit original caller list")
    candidates = [item for item in contracts if isinstance(item, dict)
                  and isinstance(item.get("source"), dict) and item["source"].get("span") == _span(node)]
    if not candidates:
        return None
    if len(candidates) != 1:
        raise CompilationError("library storage CALL has ambiguous supplied qualifications")
    resolved = resolve_original_call(analysis, analysis.source_scope_for(node, routine.scope), node) if resolved is None else resolved
    encoded_contract = _encode(candidates[0])
    record, declaration = _derive(analysis, routine, node, resolved, candidates[0])
    if getattr(analysis, "_library_storage_calls_owner", None) is not analysis:
        analysis._library_storage_calls_owner, analysis._library_storage_calls = analysis, {}
    identity = sha256(record.encode()).hexdigest()
    registry = analysis._library_storage_calls
    if type(registry) is not dict:
        raise CompilationError("library storage requires its private source issuer registry")
    if identity not in registry:
        registry[identity] = QualifiedLibraryStorageCall(identity, analysis, routine, node, resolved,
            encoded_contract, declaration, record, _ISSUER)
    return registry[identity].validate()


def library_storage_declaration(analysis, routine, binding):
    if not analysis.contracts.get(routine.qualified, {}).get("library_storage_calls"):
        return None
    found = []
    for node in walk(routine.execution):
        if _kind(node) == "Call_Stmt":
            proof = library_storage_call(analysis, routine, node)
            if proof is not None and proof._declaration.binding is binding:
                found.append(proof)
    if len(found) > 1:
        raise CompilationError("private library receiver requires one exact qualified operation")
    return found[0] if found else None
