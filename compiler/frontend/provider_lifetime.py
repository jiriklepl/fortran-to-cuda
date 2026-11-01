"""Explicit original call-spine ownership of private provider storage only.

No array registration, host pointer, payload definition or replica survives a
selected child return. Unselected original CALLs keep their standalone ABI.
This is not a native-effect, noescape, field-lifetime or numerical proof.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, fields
from hashlib import sha256
from pathlib import Path

from fparser.two.utils import walk

from compiler.frontend.authority_context import CONTEXT_FIELDS
from compiler.frontend.call_bindings import _arrange, resolve_original_call
from compiler.frontend.descriptor_intrinsics import (
    intrinsic_scope_attachments,
    intrinsic_source_attachments,
    same_intrinsic_scope_attachments,
    same_intrinsic_source_attachments,
)
from compiler.frontend.source_effects import Routine, _children, _part
from compiler.frontend.source_functions import _original
from compiler.ir import CompilationError


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return sha256(encoded(value).encode()).hexdigest()


def span(node):
    item = getattr(node, "item", None)
    if item is None and type(getattr(node, "parent", None)).__name__ == "If_Stmt":
        item = node.parent.item
    value = getattr(item, "fort_original_span", getattr(item, "span", None))
    if value is None:
        raise CompilationError("provider-only lifetime requires original statement locations")
    return list(value)


def program_roots(request):
    """Opt in to exact source metadata; this does not select a program entry."""
    if (type(request) is dict and type(request.get("call_path")) is list and request["call_path"]
            and type(request["call_path"][0]) is str and request["call_path"][0].startswith("$program::")):
        return (request["call_path"][0],)
    return ()


def header(routine):
    return _part(routine.scope.node, "Program_Stmt" if routine.source_kind == "program" else "Subroutine_Stmt")


def ending(routine):
    return _part(routine.scope.node, "End_Program_Stmt" if routine.source_kind == "program" else "End_Subroutine_Stmt")


def _derive(analysis, entry, request):
    absent = object()
    fields_at_entry = tuple((name, getattr(analysis, name, absent)) for name in CONTEXT_FIELDS)
    if (type(request) is not dict or set(request) != {"version", "call_path"}
            or type(request["version"]) is not int or request["version"] != 1
            or type(request["call_path"]) is not list or len(request["call_path"]) < 2
            or len(request["call_path"]) > analysis.procedure_limit
            or len(request["call_path"]) - 1 > analysis.depth_limit
            or any(type(name) is not str or name != name.lower() for name in request["call_path"])
            or len(set(request["call_path"])) != len(request["call_path"])
            or request["call_path"][-1] != entry):
        raise CompilationError("provider-only lifetime requires one bounded exact original call spine")
    originals = tuple(analysis.routines.get(name) for name in request["call_path"])
    if not all(isinstance(value, Routine) for value in originals):
        raise CompilationError("provider-only lifetime requires available original source members")
    routine_fields = tuple((routine, tuple((part.name, getattr(routine, part.name)) for part in fields(routine)))
                           for routine in originals)
    routines = tuple(_original(analysis, name) for name in request["call_path"])
    if any(current is not original for current, original in zip(routines, originals, strict=True)):
        raise CompilationError("provider-only lifetime original routine attachments changed")
    for index, routine in enumerate(routines):
        start = header(routine)
        if start is not None and re.search(r"\b(?:PURE|ELEMENTAL)\b|\bBIND\s*\(", str(start), re.IGNORECASE):
            raise CompilationError("provider-only lifetime rejects PURE, ELEMENTAL and BIND(C) procedures")
        if start is not None:
            first, last = span(start)[0], span(ending(routine))[1]
            original_lines = analysis.source_text(routine.scope.path).splitlines()[first-1:last]
            if any(re.match(r"\s*[!c*]\$omp\b", line, re.IGNORECASE) for line in original_lines):
                raise CompilationError("provider-only lifetime first slice rejects OpenMP in selected original bodies")
            if any(";" in line for line in original_lines):
                raise CompilationError("provider-only lifetime rejects semicolon/co-spanning physical source edits")
        if ((routine.source_kind != "module" and not (index == 0 and routine.source_kind == "program"))
                or start is None or ending(routine) is None
                or any(type(node).__name__ == "Name" and str(node).lower().startswith("fort_")
                       for node in walk(routine.scope.node))
                or any(type(node).__name__ in {"Return_Stmt", "Entry_Stmt", "Goto_Stmt", "Computed_Goto_Stmt",
                    "Assigned_Goto_Stmt", "Alternate_Return_Spec", "Stop_Stmt"} or getattr(getattr(node, "item", None), "label", None)
                    for node in walk(routine.execution))):
            raise CompilationError("provider-only lifetime requires original module subroutines or an explicit program owner with one normal end and no private-name collision")
    # Save supported source/namespace attachments before resolution/serialization
    # callbacks. Sealed syntax uses existing original membership, not a new walk.
    scopes = {}
    for routine in routines:
        current = routine.scope
        while current is not None and id(current) not in scopes:
            scopes[id(current)] = current
            current = current.parent
    call_inventory = {}
    for routine in routines:
        rows = []
        for ordinal, node in enumerate(node for node in walk(routine.execution) if type(node).__name__ == "Call_Stmt"):
            scope = analysis.source_scope_for(node, routine.scope)
            rows.append((ordinal, node, scope))
            current = scope
            while current is not None and id(current) not in scopes:
                scopes[id(current)] = current
                current = current.parent
        call_inventory[routine.qualified] = rows
    source = intrinsic_source_attachments(analysis)
    raw = tuple(intrinsic_scope_attachments(scope, syntax=False) for scope in scopes.values())
    bindings = tuple((binding, tuple((part.name, getattr(binding, part.name)) for part in fields(binding)))
                     for scope in scopes.values() for binding in scope.bindings.values())
    call_fields = []
    snapshot = analysis._source_syntax
    membership = snapshot.checkpoint()
    if not snapshot.immutable or not all(membership.unchanged(routine.scope.node) for routine in routines):
        raise CompilationError("provider-only lifetime requires its sealed original source session")
    context = analysis.source_authority_context()
    context.require(analysis)
    calls, retained = [], []
    for caller, callee in zip(routines[:-1], routines[1:], strict=True):
        selected = []
        for ordinal, node, scope in call_inventory[caller.qualified]:
            if callee.qualified not in analysis._candidates(scope, str(node.items[0]).lower()):
                continue
            proof = resolve_original_call(analysis, scope, node)
            objects = (proof, *proof.original_arguments, *proof.associations)
            call_fields.extend((value, tuple((part.name, getattr(value, part.name)) for part in fields(value)))
                               for value in objects)
            if proof.callee is not callee:
                raise CompilationError("provider-only lifetime original CALL is ambiguous")
            actuals, arguments = _arrange(callee, tuple(_children(node.items[1])))
            if (proof.formals != callee.arguments or len(actuals) != len(proof.actuals)
                    or any(left is not right for left, right in zip(actuals, proof.actuals, strict=True))
                    or len(arguments) != len(proof.original_arguments)
                    or any((left.position, left.formal, left.keyword) != (right.position, right.formal, right.keyword)
                           or left.actual is not right.actual
                           for left, right in zip(arguments, proof.original_arguments, strict=True))):
                raise CompilationError("provider-only lifetime cached CALL arguments differ from exact original syntax")
            retained.append((caller, node, proof))
            selected.append({"ordinal": ordinal, "span": span(node), "call_sha256": sha256(str(node).encode()).hexdigest(),
                             "call": proof.public()})
        if not selected:
            raise CompilationError("provider-only lifetime is missing a selected original CALL edge")
        calls.append({"caller": caller.qualified, "callee": callee.qualified, "calls": selected})
    record = {"version": 1, "kind": "original_provider_only_lifetime", "owner": routines[0].qualified,
        "entry": entry, "request": request, "source_inputs": analysis.inputs.identity(),
        "members": [{"procedure": routine.qualified, "source_kind": routine.source_kind, "source": str(routine.scope.path),
                     "source_sha256": analysis.sources[str(routine.scope.path)],
                     "body_sha256": sha256(str(routine.scope.node).encode()).hexdigest(),
                     "span": [span(header(routine))[0], span(ending(routine))[1]]} for routine in routines],
        "edges": calls, "budgets": {name: getattr(analysis, name) for name in
            ("procedure_limit", "depth_limit", "operation_limit")}, "normal_exit": "single original END SUBROUTINE or END PROGRAM; explicit RETURN, STOP and branch labels rejected",
        "private_storage": "exact provider domain, plan, workspace, configuration words and generation only",
        "child_return": "successful completion, publish and unregister every field, then detach borrowed context",
        "native_work": "all original statements remain; zero retained canonical fields or host pointers",
        "unknown_dispatch": "original standalone entry; no inherited provider slot",
        "preaccept_failure": "publish/unregister any fields, retire provider/domain, disable this owner; original CALL once",
        "postsubmit_failure": "fatal; no source replay and no later domain reuse",
        "field_retention_authorized": False, "incoming_definition_authority": False,
        "alias_authority": False, "native_effect_authority": False, "execution_authority": False}
    result = encoded(record)
    context.require(analysis)
    analysis.verify_publication()
    analysis.inputs.verify_live()
    if (getattr(analysis, "_source_authority_context", None) is not context
            or any(getattr(analysis, name, absent) is not value for name, value in fields_at_entry)
            or not same_intrinsic_source_attachments(analysis, source)
            or not all(same_intrinsic_scope_attachments(state) for state in raw)
            or any(getattr(binding, name) is not value for binding, state in bindings for name, value in state)
            or any(getattr(value, name) is not original for value, state in (*routine_fields, *call_fields)
                   for name, original in state)
            or analysis._source_syntax is not snapshot
            or not all(membership.unchanged(routine.scope.node) for routine in routines)):
        raise CompilationError("provider-only lifetime original source or namespace changed")
    return result, routines, tuple(retained), context


class _Issue:
    def __init__(self, analysis):
        self.analysis, self.proof = analysis, None


@dataclass(frozen=True, slots=True)
class ProviderOnlyLifetime:
    identity: str
    _analysis: object
    _entry: str
    _request: str
    _record: str
    _routines: tuple
    _calls: tuple
    _context: object
    _issue: object

    def public(self):
        return {"identity": self.identity, **json.loads(self._record)}

    def validate(self):
        if self._issue.analysis is not self._analysis or self._issue.proof is not self:
            raise CompilationError("provider-only lifetime requires its exact issued original selection")
        issue = self._issue
        self._context.require(self._analysis)
        current, routines, calls, context = _derive(self._analysis, self._entry, json.loads(self._request))
        if (self._issue is not issue or issue.analysis is not self._analysis or issue.proof is not self
                or current != self._record or any(a is not b for a, b in zip(routines, self._routines, strict=True))
                or any(a[0] is not b[0] or a[1] is not b[1] or a[2] is not b[2]
                       for a, b in zip(calls, self._calls, strict=True))
                or context is not self._context):
            raise CompilationError("provider-only lifetime original call spine changed")
        return self


def provider_only_lifetime(analysis, entry, request):
    if request is None:
        return None
    request = encoded(request)
    record, routines, calls, context = _derive(analysis, entry, json.loads(request))
    issue = _Issue(analysis)
    proof = ProviderOnlyLifetime(sha256(record.encode()).hexdigest(), analysis, entry, request, record,
                                 routines, calls, context, issue)
    issue.proof = proof
    return proof


# This is an explicit reviewed external-provider duty, separate from numerical
# equivalence and source may-read permissions. No new effect engine is involved.
PROVIDER_RETENTION_DUTIES = {'retained_state': 'configuration, compiled plan, domain-owned workspace, private stream and events only', 'fields_after_completion': 'no application field pointer, external view or registration handle is reused or dereferenced after successful wait and unregister', 'execute_bindings': 'each execute uses only its current supplied context/bindings', 'publication': 'complete all submitted application field uses before unregister and borrowed close', 'failure': 'post-submit failure is fatal; no reuse or replay', 'field_retention_authorized': False, 'host_payload_retention_authorized': False, 'incoming_definition_authority': False, 'execution_authority': False}


def validate_retained_state_record(record, provider, *, prerequisite=False):
    from compiler.frontend.device_tasks import _reference
    from compiler.frontend.external_entry_reads import provider_read_binding
    expected = {"version", "kind", "provider_binding", "duties", "evidence"}
    if not prerequisite:
        expected.add("prerequisite")
    kind = ("reviewed_external_provider_retention_prerequisite" if prerequisite
            else "external_provider_retained_plan_state")
    if (type(record) is not dict or set(record) != expected or type(record["version"]) is not int
            or record["version"] != 1 or record["kind"] != kind
            or encoded(record["provider_binding"]) != encoded(provider_read_binding(provider))
            or type(record["duties"]) is not dict or record["duties"] != PROVIDER_RETENTION_DUTIES
            or any(record["duties"].get(key) is not False for key in (
                "field_retention_authorized", "host_payload_retention_authorized",
                "incoming_definition_authority", "execution_authority"))
            or type(record["evidence"]) is not list or not record["evidence"]):
        raise CompilationError("provider-only retained-state qualification or exact implementation binding differs")
    for evidence in record["evidence"]:
        _reference(evidence)
    if not prerequisite:
        reference = record["prerequisite"]
        original = json.loads(Path(_reference(reference)).read_text())
        validate_retained_state_record(original, provider, prerequisite=True)
        if any(encoded(record[key]) != encoded(original[key]) for key in ("provider_binding", "duties", "evidence")):
            raise CompilationError("provider-only retained-state qualification differs from reviewed prerequisite")
    return record


def retained_state_qualification(provider):
    from compiler.frontend.device_tasks import _reference
    reference = provider.get("retained_state_qualification")
    if reference is None:
        raise CompilationError("provider-only lifetime requires an explicit retained-state provider qualification")
    record = json.loads(Path(_reference(reference)).read_text())
    return validate_retained_state_record(record, provider)
