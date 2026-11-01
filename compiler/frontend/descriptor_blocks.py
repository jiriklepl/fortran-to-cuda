"""Original BLOCK declarations for a complete descriptor-only body walk.

The lexical index owns ordinary source Scope/Binding objects. This receipt
authenticates those declarations; its consumer must retain every original body
statement and independently check all storage effects and source calls.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from hashlib import sha256

from compiler.frontend.descriptor_associations import _same_syntax, _syntax_attachments
from compiler.frontend.optional_scalar_control import _same_scope_state, _scope_state
from compiler.frontend.source_effects import _children, _kind
from compiler.frontend.source_functions import _scope_roles, _validate_scope, _validate_scope_roles
from compiler.ir import CompilationError

DESCRIPTOR_WINDOW_BLOCK_VERSION = 16
_ISSUER = object()
_UNSAFE = {"pointer", "allocatable", "target", "optional", "value", "volatile", "asynchronous",
           "save", "external", "intrinsic", "bind"}


class DescriptorBlockUnsupported(CompilationError):
    """An original declaration support limit, never source/budget authority."""


def _attached(value):
    return tuple((item.name, getattr(value, item.name)) for item in fields(value))


def _same(value, saved):
    return all(getattr(value, name) is item for name, item in saved)


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _registry(analysis):
    registry = getattr(analysis, "_descriptor_blocks", None)
    owner = getattr(analysis, "_descriptor_blocks_owner", None)
    if registry is None or owner is None and not registry:
        registry = analysis._descriptor_blocks = {}
        analysis._descriptor_blocks_owner = analysis
    elif owner is not analysis or not isinstance(registry, dict):
        raise CompilationError("descriptor BLOCK requires its own exact analysis issuer")
    return registry


def _same_declaration(analysis, saved):
    scope, original, scope_fields, qualified, state, issues, bindings, syntax, occurrences, role = saved
    return (getattr(analysis, "_block_scopes", {}).get(id(original)) is scope
        and getattr(analysis, "_block_scope_roles", {}).get(id(original)) is role
        and _same(scope, scope_fields) and scope.qualified == qualified and _same_scope_state(state)
        and _encode({"issues": scope.issues, "unresolved_imports": scope.unresolved_imports}) == issues
        and _same_syntax(syntax)
        and all(scope.bindings.get(name) is binding and _same(binding, attached) for name, binding, attached in bindings)
        and all(analysis.source_scope_for(node) is owner for node, owner in occurrences))


def _original_member(analysis, routine, original):
    session = getattr(analysis, "_source_session", None)
    if session is not None:
        return session.execution_nodes(analysis, routine).get(id(original)) is original
    pending = [routine.execution]
    while pending:
        node = pending.pop()
        if node is original:
            return True
        pending.extend(node if isinstance(node, (tuple, list)) else _children(node))
    return False


def _derive(analysis, routine, original, scope_validator=None, tick=None):
    if (analysis._require_original(routine.qualified) is not routine
            or _kind(original) != "Block_Construct"):
        raise CompilationError("descriptor BLOCK requires its exact original routine and construct")
    # A BLOCK may be an original child inside one deferred whole-team unit.
    # Its source membership is independent of whether it has a numerical IR.
    source = analysis._require_original_global()
    if not source.unchanged(original) or not _original_member(analysis, routine, original):
        raise CompilationError("descriptor BLOCK requires unchanged original syntax")
    scope = getattr(analysis, "_block_scopes", {}).get(id(original))
    role = getattr(analysis, "_block_scope_roles", {}).get(id(original))
    children = tuple(_children(original))
    semantic = tuple(child for child in children if _kind(child) != "Comment")
    if (scope is None or role is None or role[0] is not scope or role[1] is not original
            or role[2] is not scope.parent or any(analysis.source_scope_for(item) is not owner for item, owner in role[3])
            or scope.node is not original or scope.parent is None
            or len(semantic) < 2 or _kind(semantic[0]) != "Block_Stmt"
            or _kind(semantic[-1]) != "End_Block_Stmt"
            or analysis.source_scope_for(original, routine.scope) is not scope.parent
            or analysis.source_scope_for(semantic[0], routine.scope) is not scope.parent):
        raise CompilationError("descriptor BLOCK requires its indexed original lexical scope")
    current, seen = scope.parent, set()
    lexical = {id(item): item for item in analysis._block_scopes.values()}
    lexical.update((id(item.scope), item.scope) for item in analysis._associate_scopes.values() if item.available)
    while current is not routine.scope:
        if current is None or id(current) in seen or lexical.get(id(current)) is not current:
            raise CompilationError("descriptor BLOCK requires its exact original owning procedure")
        seen.add(id(current))
        current = current.parent
    syntax, state, locals_ = _syntax_attachments(original), _scope_state(scope), scope.bindings
    scope_fields, qualified = _attached(scope), scope.qualified
    issues = _encode({"issues": scope.issues, "unresolved_imports": scope.unresolved_imports})
    bindings = tuple((name, binding, _attached(binding)) for name, binding in locals_.items())
    occurrences = tuple((node, analysis.source_scope_for(node)) for node, _children_ in syntax)
    declaration = (scope, original, scope_fields, qualified, state, issues, bindings, syntax, occurrences, role)
    roles, scopes = _scope_roles(analysis), {}

    def require_scope(current):
        if scope_validator is not None:
            scope_validator(current)
        _validate_scope(analysis, current)
        scopes[id(current)] = current

    require_scope(scope.parent)
    require_scope(scope)
    if scope.issues or scope.unresolved_imports:
        raise DescriptorBlockUnsupported("descriptor BLOCK declarations require complete original namespace resolution")
    specification = next((child for child in children if _kind(child) == "Specification_Part"), None)
    for child in _children(specification):
        if tick is not None:
            tick()
        if _kind(child) not in {"Use_Stmt", "Implicit_Part", "Type_Declaration_Stmt", "Comment"}:
            raise DescriptorBlockUnsupported("descriptor BLOCK declaration lifecycle or association requires separate authority")
        if _kind(child) == "Comment" and str(child).lstrip().startswith("!$"):
            raise DescriptorBlockUnsupported("descriptor BLOCK declaration concurrency requires separate authority")
        if _kind(child) == "Type_Declaration_Stmt":
            names = [str(entity.items[0]).lower() for entity in child.items[2].items]
            if len(names) != len(set(names)):
                raise DescriptorBlockUnsupported("descriptor BLOCK requires distinct original local declarations")
    for name, binding, _saved in bindings:
        if tick is not None:
            tick()
        if (binding.declaring_scope is not scope or binding.name != name
                or binding.root != scope.qualified + "::" + name
                or binding.dtype not in {"real", "integer", "logical"} or binding.kind not in {4, 8}
                or binding.rank or binding.shape_nodes or binding.attributes & _UNSAFE):
            raise DescriptorBlockUnsupported("descriptor BLOCK requires independent fixed numeric scalar locals")
    record = {"version": 1, "kind": "original_fixed_scalar_block_scope", "procedure": routine.qualified,
        "scope_id": scope.qualified, "original_source_sha256": sha256(str(original).encode()).hexdigest(),
        "original_specification_sha256": sha256(str(specification).encode()).hexdigest(),
        "local_bindings": {name: binding.public() for name, binding, _saved in bindings},
        "requires_complete_original_body": True, "descriptor_evidence_only": True,
        "entry_definition_authority": False, "payload_access_authorized": False,
        "alias_authority": False, "descriptor_lifetime_authorized": False,
        "execution_authorized": False, "motion_authorized": False}
    analysis.verify_publication()
    current = analysis._require_original_global()
    if analysis._require_original_routine(routine.qualified, _syntax=current) is not routine:
        raise CompilationError("descriptor BLOCK changed its original routine at publication")
    _validate_scope_roles(analysis, roles, scopes)
    for value in scopes.values():
        _validate_scope(analysis, value)
    if not _same_declaration(analysis, declaration):
        raise CompilationError("descriptor BLOCK attachment changed at original source publication")
    return _encode(record), scope, declaration


@dataclass(frozen=True, slots=True)
class FixedScalarBlock:
    identity: str
    _analysis: object = field(repr=False)
    _routine: object = field(repr=False)
    _original: object = field(repr=False)
    _scope: object = field(repr=False)
    _declaration: object = field(repr=False)
    _record: str = field(repr=False)
    _issuer: object = field(repr=False)

    def validate(self, *, scope_validator=None, tick=None):
        saved = _attached(self)
        registry = _registry(self._analysis)
        if (type(self) is not FixedScalarBlock or self._issuer is not _ISSUER
                or registry.get(self.identity) is not self or sha256(self._record.encode()).hexdigest() != self.identity
                or not _same_declaration(self._analysis, self._declaration)):
            raise CompilationError("descriptor BLOCK requires its source-issued declaration composition")
        record, scope, _declaration = _derive(self._analysis, self._routine, self._original, scope_validator, tick)
        if (scope is not self._scope or record != self._record or not _same(self, saved)
                or getattr(self._analysis, "_descriptor_blocks", None) is not registry
                or getattr(self._analysis, "_descriptor_blocks_owner", None) is not self._analysis
                or registry.get(self.identity) is not self or not _same_declaration(self._analysis, self._declaration)):
            raise CompilationError("descriptor BLOCK changed its exact original declaration composition")
        return self

    def public(self):
        return {"identity": self.identity, **json.loads(self._record)}


def original_fixed_block(analysis, routine, original, *, scope_validator=None, tick=None):
    registry = _registry(analysis)
    retained = []
    for identity, proof in registry.items():
        if getattr(proof, "_original", None) is original:
            if type(proof) is not FixedScalarBlock:
                raise CompilationError("descriptor BLOCK issuer has an untyped retained query token")
            retained.append((identity, proof, _attached(proof)))

    def require_registry():
        if (getattr(analysis, "_descriptor_blocks", None) is not registry
                or getattr(analysis, "_descriptor_blocks_owner", None) is not analysis
                or any(registry.get(identity) is not proof or not _same(proof, saved)
                       for identity, proof, saved in retained)):
            raise CompilationError("descriptor BLOCK issuer attachment changed before source publication")

    record, scope, declaration = _derive(analysis, routine, original, scope_validator, tick)
    require_registry()
    identity = sha256(record.encode()).hexdigest()
    if identity not in registry:
        registry[identity] = FixedScalarBlock(identity, analysis, routine, original, scope, declaration, record, _ISSUER)
    proof = registry[identity]
    if (type(proof) is not FixedScalarBlock or proof._analysis is not analysis or proof._routine is not routine
            or proof._original is not original or proof._scope is not scope or proof._issuer is not _ISSUER):
        raise CompilationError("descriptor BLOCK issuer requires its exact consuming analysis and original query")
    attached = _attached(proof)
    result = proof.validate(scope_validator=scope_validator, tick=tick)
    require_registry()
    if result is not proof or registry.get(identity) is not proof or not _same(proof, attached):
        raise CompilationError("descriptor BLOCK issuer changed its validated token at publication")
    return proof
