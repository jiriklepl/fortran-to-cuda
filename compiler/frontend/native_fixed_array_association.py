"""Exact whole fixed-array association for an unchanged original native CALL.

This token maps declared element coordinates only. It proves no payload effect,
definition, alias freedom, lifetime, completion, ownership or device eligibility.
Numerical and companion consumers retain their existing shape requirements.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from hashlib import sha256

from compiler.frontend.authority_context import context_for
from compiler.frontend.call_bindings import (
    _fixed_array_bounds,
    _function_array_mapping,
    _original_call_control,
    resolve_original_call,
)
from compiler.ir import CompilationError

NATIVE_FIXED_ARRAY_ASSOCIATION_VERSION = 1
_ISSUER = object()
_UNSAFE = {"pointer", "allocatable", "optional", "target", "value", "volatile", "asynchronous"}


def _attachments(proof):
    return tuple((part.name, object.__getattribute__(proof, part.name))
                 for part in fields(NativeFixedArrayAssociation))


def _require_attachments(proof, attached, analysis, registry, context, session, arena, identity):
    # Every retained value is immutable source state, an immutable value, or an
    # exact authority attachment. Finish publication with this callback-free check.
    if (any(object.__getattribute__(proof, name) is not value for name, value in attached)
            or getattr(analysis, "_native_fixed_array_associations", None) is not registry
            or registry.get(identity) is not proof
            or getattr(analysis, "_source_authority_context", None) is not context
            or getattr(analysis, "_source_session", None) is not session or context.session is not session
            or getattr(analysis, "_source_derivation_arena", None) is not arena):
        raise CompilationError("native fixed-array association attachments changed before publication")


def _derive(analysis, procedure, selection, formal_name):
    context = context_for(analysis)
    if type(procedure) is not str or type(formal_name) is not str or formal_name != formal_name.lower():
        raise CompilationError("native fixed-array association requires exact source procedure and formal names")
    original_call = _original_call_control(selection)
    if original_call is None:
        raise CompilationError("native fixed-array association requires one complete original CALL control")
    caller = analysis._require_original(procedure)
    if analysis.routines.get(procedure) is not caller:
        raise CompilationError("native fixed-array association requires its original caller")
    parent = context.session.lexical_parent(original_call)
    if type(parent).__name__ == "If_Stmt" and selection[0] is not parent:
        raise CompilationError("native fixed-array association must retain the complete original IF predicate")
    graph, selected = analysis._selected_source(procedure, selection)
    scope = analysis.source_scope_for(original_call, caller.scope)
    resolved = resolve_original_call(analysis, scope, original_call)
    association = next((item for item in resolved.associations if item.formal == formal_name), None)
    if association is None:
        raise CompilationError("native fixed-array association requires an original present array formal")
    formal, actual = association.formal_binding, association.binding
    if (type(association.actual).__name__ != "Name" or actual is None
            or formal.rank != 1 or actual.rank != 1
            or formal.dtype not in {"real", "integer", "logical"}
            or type(formal.kind) is not int or formal.kind < 1
            or formal.attributes & _UNSAFE or actual.attributes & _UNSAFE
            or (formal.intent != "in" and "parameter" in actual.attributes)):
        raise CompilationError("native fixed-array association requires ordinary whole rank-one primitive storage")
    # Original-call resolution already matches declared type, kind and rank.
    # Reuse the existing bounded constant-shape rule, never sections or temps.
    _function_array_mapping(analysis, formal, actual, None, element_limit=None)
    formal_bounds, = _fixed_array_bounds(analysis, formal, element_limit=None)
    actual_bounds, = _fixed_array_bounds(analysis, actual, element_limit=None)
    element_count = max(0, formal_bounds[1] - formal_bounds[0] + 1)
    item = selection[0].item
    span = getattr(item, "fort_original_span", item.span)
    record = {
        "schema_version": NATIVE_FIXED_ARRAY_ASSOCIATION_VERSION,
        "proof_role": "native_whole_fixed_array_call_association",
        "source_session": context.session.identity,
        "source_context": context._intern_key,
        "policy": {"version": NATIVE_FIXED_ARRAY_ASSOCIATION_VERSION,
                   "depth_limit": analysis.depth_limit, "procedure_limit": analysis.procedure_limit,
                   "operation_limit": analysis.operation_limit, "element_limit": None,
                   "bounds_abi": "default INTEGER bounds and extent"},
        "caller": procedure, "callee": resolved.procedure, "graph_identity": graph.identity,
        "selected_node_ids": list(selected), "source": str(caller.scope.path),
        "first_line": span[0], "last_line": span[1],
        "original_selection": str(selection[0]), "original_call": str(original_call),
        "source_call": resolved.public(), "formal": formal_name,
        "formal_descriptor": formal.public(), "actual_descriptor": actual.public(),
        "formal_bounds": list(formal_bounds), "actual_bounds": list(actual_bounds),
        "element_count": element_count,
        "actual_index_minus_formal_index": actual_bounds[0] - formal_bounds[0],
        "association": "original whole-array element order; declared lower bounds retained",
        "requires_original_execution": True,
        "payload_access_authorized": False, "payload_definition_authority": False,
        "entry_definition_authority": False, "initialization_authorized": False,
        "capture_authorized": False, "alias_authority": False, "ownership_authority": False,
        "lifetime_authority": False, "completion_authority": False, "execution_authorized": False,
        "native_write_authority": False, "gpu_legality_established": False,
        "numerical_lowering_authorized": False, "motion_authorized": False,
    }
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return context, caller, resolved, formal, actual, formal_bounds, actual_bounds, element_count, encoded


@dataclass(frozen=True)
class NativeFixedArrayAssociation:
    identity: str
    formal: object = field(repr=False, compare=False)
    actual: object = field(repr=False, compare=False)
    formal_bounds: tuple[int, int]
    actual_bounds: tuple[int, int]
    element_count: int
    _analysis: object = field(repr=False, compare=False)
    _context: object = field(repr=False, compare=False)
    _arena: object = field(repr=False, compare=False)
    _caller: object = field(repr=False, compare=False)
    _resolved: object = field(repr=False, compare=False)
    _selection: tuple = field(repr=False, compare=False)
    _formal_name: str = field(repr=False, compare=False)
    _record: str = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)

    @property
    def index_offset(self):
        """Actual index = formal index + this offset; no access is authorized."""
        return self.actual_bounds[0] - self.formal_bounds[0]

    def _require_handle(self, analysis):
        self._context.require(analysis)
        arena = self._context.derivation_arena(analysis)
        if not arena.accepts(self._arena):
            raise CompilationError("native fixed-array association belongs to a discarded generation arena")
        arena.require_handle(("native_fixed_array_association", id(analysis), self.identity), self)

    def validate(self, analysis, procedure, selection, formal_name):
        self._require_handle(analysis)
        attached = _attachments(self)
        registry = analysis._native_fixed_array_associations
        context, session, arena, identity = self._context, self._context.session, analysis._source_derivation_arena, self.identity
        self._validate_source(analysis, procedure, selection, formal_name)
        _require_attachments(self, attached, analysis, registry, context, session, arena, identity)
        self._require_handle(analysis)
        _require_attachments(self, attached, analysis, registry, context, session, arena, identity)
        return self

    def _validate_source(self, analysis, procedure, selection, formal_name):
        if (type(self) is not NativeFixedArrayAssociation or self._issuer is not _ISSUER
                or analysis is not self._analysis or procedure != self._caller.qualified
                or formal_name != self._formal_name or type(selection) is not tuple
                or len(selection) != len(self._selection)
                or any(left is not right for left, right in zip(selection, self._selection, strict=True))
                or getattr(analysis, "_native_fixed_array_associations", {}).get(self.identity) is not self):
            raise CompilationError("native fixed-array association requires its exact issued source selection")
        self._context.require(analysis)
        context, caller, resolved, formal, actual, formal_bounds, actual_bounds, count, record = _derive(
            analysis, procedure, selection, formal_name)
        if (context is not self._context or caller is not self._caller or resolved is not self._resolved
                or formal is not self.formal or actual is not self.actual
                or formal_bounds != self.formal_bounds or actual_bounds != self.actual_bounds
                or count != self.element_count or record != self._record
                or sha256(record.encode()).hexdigest() != self.identity):
            raise CompilationError("native fixed-array association source, policy or declared coordinates changed")
        analysis.verify_publication()
        if getattr(analysis, "_native_fixed_array_associations", {}).get(self.identity) is not self:
            raise CompilationError("native fixed-array association registry changed before publication")
        return self

    def public(self):
        return {"identity": self.identity, **json.loads(self._record)}


def prove_native_fixed_array_association(analysis, procedure, selection, formal_name):
    """Issue association-only evidence from an already sealed original session."""
    context, caller, resolved, formal, actual, formal_bounds, actual_bounds, count, record = _derive(
        analysis, procedure, selection, formal_name)
    identity = sha256(record.encode()).hexdigest()
    registry = getattr(analysis, "_native_fixed_array_associations", None)
    if registry is None:
        registry = analysis._native_fixed_array_associations = {}
    if identity in registry:
        proof = registry[identity]
        if type(proof) is not NativeFixedArrayAssociation:
            raise CompilationError("native fixed-array association registry contains foreign evidence")
        return proof.validate(analysis, procedure, selection, formal_name)
    arena = context.derivation_arena(analysis)
    proof = NativeFixedArrayAssociation(identity, formal, actual, formal_bounds, actual_bounds, count,
        analysis, context, arena, caller, resolved, selection, formal_name, record, _ISSUER)
    attached, session = _attachments(proof), context.session
    registry[identity] = proof
    try:
        proof._validate_source(analysis, procedure, selection, formal_name)
        if context.derivation_arena(analysis) is not arena:
            raise CompilationError("native fixed-array association issuing generation arena changed")
        _require_attachments(proof, attached, analysis, registry, context, session, arena, identity)
        arena.register_handle(("native_fixed_array_association", id(analysis), identity), proof)
        return proof
    except BaseException:
        if registry.get(identity) is proof:
            del registry[identity]
        raise
