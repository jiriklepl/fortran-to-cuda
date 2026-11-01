"""Compose exact private receiver/constructor evidence, without array lifetime.

This is a declaration/metadata service. The contracted operation's ordinary
array descriptor and noescape effects remain independent obligations. In
particular an external task's requirement for an existing domain cannot be
used to bootstrap that domain's lifetime.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from hashlib import sha256

from compiler.frontend.call_bindings import _OriginalTypes, _resolve_candidates, resolve_original_function
from compiler.frontend.metadata_constructor import OriginalMetadataConstructorProof
from compiler.frontend.opaque_library_state import OriginalOpaqueStateProof
from compiler.ir import CompilationError

_ISSUER = object()


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _attachments(value):
    return tuple((item.name, getattr(value, item.name)) for item in fields(value))


def _attached(value, retained):
    return all(getattr(value, name) is original for name, original in retained)


def _consumed_call(analysis, routine, original, resolved, *, function=False):
    """Bind retained resolver fields to fresh original candidate association."""
    scope = analysis.source_scope_for(original, routine.scope)
    resolved.validate(analysis, scope, original)
    types = _OriginalTypes(analysis)
    types.caller_scope = scope
    callee, actuals, associations, arguments = _resolve_candidates(
        analysis, scope, original, types.associate, function=function)
    if (resolved.callee is not callee or resolved._caller is not routine
            or resolved.procedure != callee.qualified or resolved.formals != callee.arguments
            or resolved.target != str(original.items[0]) or resolved.source_kind != callee.source_kind
            or len(resolved.actuals) != len(actuals)
            or any(a is not b for a, b in zip(resolved.actuals, actuals, strict=True))
            or len(resolved.associations) != len(associations)
            or any(a.actual is not b.actual or a.formal_binding is not b.formal_binding or a.binding is not b.binding
                   or a.public() != b.public() for a, b in zip(resolved.associations, associations, strict=True))
            or len(resolved.original_arguments) != len(arguments)
            or any(a.actual is not b.actual or a.public() != b.public()
                   for a, b in zip(resolved.original_arguments, arguments, strict=True))):
        raise CompilationError("private library state lost its exact original consumed call association")
    return resolved.public()


def _derive(analysis, routine, binding, opaque, constructor):
    if (type(opaque) is not OriginalOpaqueStateProof or type(constructor) is not OriginalMetadataConstructorProof
            or opaque._analysis is not analysis or constructor._analysis is not analysis
            or opaque._routine is not routine or constructor._routine is not routine
            or opaque.binding is not binding or constructor._opaque is not opaque
            or analysis._require_original(routine.qualified) is not routine
            or routine.scope.bindings.get(binding.name) is not binding
            or binding.declaring_scope is not routine.scope):
        raise CompilationError("private library state requires its exact receiver and constructor issuers")
    for proof in (opaque, constructor):
        if sha256(proof._record.encode()).hexdigest() != proof.identity:
            raise CompilationError("private library state changed an independent proof's content identity")
    opaque_attachments, constructor_attachments = _attachments(opaque), _attachments(constructor)
    opaque.validate(analysis, routine, opaque._node, opaque._resolved)
    constructor.validate(analysis, routine, constructor._assignment)
    assignment = constructor._assignment
    call = _consumed_call(analysis, routine, opaque._node, opaque._resolved)
    expression = constructor._assignment.items[2]
    resolved = resolve_original_function(analysis, analysis.source_scope_for(expression, routine.scope), expression)
    factory = _consumed_call(analysis, routine, expression, resolved, function=True)
    record = {"version": 1, "kind": "original_private_library_state_declaration",
        "procedure": routine.qualified, "binding": binding.public(),
        "source_inputs": analysis.inputs.identity(), "original_body_sha256": sha256(str(routine.scope.node).encode()).hexdigest(),
        "opaque_state": opaque.public(), "metadata_constructor": constructor.public(),
        "operation_association": call, "constructor_association": factory,
        "declaration_policy": "only exact private receiver and once-only fixed metadata construction; complete operation effects remain independent",
        "caller_array_descriptor_authorized": False, "caller_array_noescape_authorized": False,
        "entry_definition_authority": False, "payload_access_authorized": False,
        "alias_authority": False, "execution_authorized": False}
    analysis.verify_publication()
    # Publication hooks cannot detach an existing authority after its first
    # validation. These are identity/content fences, never a new effect claim.
    if (getattr(analysis, "_opaque_library_state_proofs", {}).get(opaque.identity) is not opaque
            or getattr(analysis, "_metadata_constructor_proofs", {}).get(constructor.identity) is not constructor
            or not _attached(opaque, opaque_attachments) or not _attached(constructor, constructor_attachments)
            or constructor._opaque is not opaque
            or constructor._assignment is not assignment
            or routine.scope.bindings.get(binding.name) is not binding
            or sha256(opaque._record.encode()).hexdigest() != opaque.identity
            or sha256(constructor._record.encode()).hexdigest() != constructor.identity
            or _consumed_call(analysis, routine, opaque._node, opaque._resolved) != call
            or _consumed_call(analysis, routine, expression, resolved, function=True) != factory):
        raise CompilationError("private library state authority changed at source publication")
    return _encode(record)


@dataclass(frozen=True, slots=True)
class PrivateLibraryStateDeclaration:
    """Ephemeral composition of registered source tokens, never array facts."""
    identity: str
    _analysis: object = field(repr=False)
    _routine: object = field(repr=False)
    binding: object = field(repr=False)
    _opaque: object = field(repr=False)
    _constructor: object = field(repr=False)
    _record: str = field(repr=False)
    _issuer: object = field(repr=False)

    def validate(self):
        attachments = _attachments(self)
        if (type(self) is not PrivateLibraryStateDeclaration or self._issuer is not _ISSUER
                or sha256(self._record.encode()).hexdigest() != self.identity
                or _derive(self._analysis, self._routine, self.binding, self._opaque, self._constructor) != self._record):
            raise CompilationError("private library state requires its exact original declaration composition")
        if not _attached(self, attachments):
            raise CompilationError("private library state declaration attachment changed at publication")
        return self

    def public(self):
        return {"identity": self.identity, **json.loads(self._record)}


def private_library_state(analysis, routine, binding, opaque, constructor):
    record = _derive(analysis, routine, binding, opaque, constructor)
    return PrivateLibraryStateDeclaration(sha256(record.encode()).hexdigest(), analysis, routine, binding,
                                         opaque, constructor, record, _ISSUER).validate()
