"""Read effects of unchanged, source-backed scalar-result functions.

This deliberately grants no numerical lowering or callback authority. Functions
keep their original Fortran implementation, argument evaluation and IEEE effects.
Only a complete, bounded, read-only source closure may supply these data effects.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256

from fparser.two.utils import Base

from compiler.frontend.call_bindings import _fixed_array_bounds, resolve_source_function
from compiler.frontend.source_objects import _kind_state
from compiler.frontend.structured_effects import _freeze, _thaw
from compiler.frontend.summary_cache import SummaryCache
from compiler.ir import CompilationError

SOURCE_FUNCTION_VERSION = 2


def _walk(node):
    stack, visits = [iter((node,))], 0
    while stack:
        try:
            item = next(stack[-1])
        except StopIteration:
            stack.pop()
            continue
        visits += 1
        if visits > 65536:
            raise CompilationError("native source function original-source scan budget exhausted")
        if isinstance(item, (tuple, list)):
            stack.append(iter(item))
        elif isinstance(item, Base):
            # Name/Type_Name leaves have .string rather than .items. Include
            # them so resolution authentication cannot silently skip names.
            yield item
            stack.append(iter(getattr(item, "content", getattr(item, "items", ())) or ()))


def _hash(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _resolution(scope):
    return _hash({"imports": scope.imports, "wildcards": scope.wildcards,
                  "exclusions": {key: sorted(value) for key, value in scope.wildcard_exclusions.items()},
                  "ambiguous": sorted(scope.ambiguous_imports), "access": scope.access,
                  "default_public": scope.default_public, "generics": scope.generics,
                  "procedures": scope.procedures, "externals": sorted(scope.externals),
                  "procedure_arguments": sorted(scope.procedure_arguments),
                  "intrinsics": scope.intrinsic_procedures, "intrinsic_wildcards": sorted(scope.intrinsic_wildcards),
                  "intrinsic_exclusions": {key: sorted(value) for key, value in scope.intrinsic_exclusions.items()},
                  "bindings": {key: value.public() for key, value in scope.bindings.items()}}), _kind_state(scope)


def register_source_function_authority(analysis):
    scopes = {id(scope): scope for scope in analysis.modules.values()}
    scopes.update((id(routine.scope), routine.scope) for routine in analysis.numerical_helpers.values())
    scopes.update((id(record.scope), record.scope) for record in analysis._associate_scopes.values()
                  if record.available)
    scopes.update((id(scope), scope) for scope in analysis._block_scopes.values())
    analysis._source_function_scopes = {key: (scope, scope.node, scope.parent, _resolution(scope),
                                            {name: (binding, binding.declaring_scope)
                                             for name, binding in scope.bindings.items()})
                                        for key, scope in scopes.items()}
    analysis._source_function_used_scopes = set()


def resolution_changes(analysis):
    if getattr(analysis, "_source_session", None) is not None:
        analysis._source_session.verify(analysis)
        return []
    changes = []
    for key in sorted(analysis._source_function_used_scopes):
        scope, node, parent, state, bindings = analysis._source_function_scopes[key]
        current = _resolution(scope)
        changed_bindings = any(scope.bindings.get(name) is not binding or binding.declaring_scope is not owner
                               for name, (binding, owner) in bindings.items())
        if scope.node is not node or scope.parent is not parent or current != state or changed_bindings:
            changes.append({"module": scope.module, "scope": getattr(scope, "qualified", scope.module),
                            "resolution": current, "original_binding_authority_changed": changed_bindings})
    return changes


def _validate_scope(analysis, scope, *, track=True):
    session = getattr(analysis, "_source_session", None)
    if session is not None:
        session.verify(analysis)
        if not session.contains(scope) or id(scope) not in analysis._source_function_scopes:
            raise CompilationError("native source function original resolution authority changed")
        if track:
            analysis._source_function_used_scopes.add(id(scope))
        return
    original = analysis._source_function_scopes.get(id(scope))
    if (original is None or original[0] is not scope or original[1] is not scope.node
            or original[2] is not scope.parent or original[3] != _resolution(scope)
            or any(scope.bindings.get(name) is not binding or binding.declaring_scope is not owner
                   for name, (binding, owner) in original[4].items())):
        raise CompilationError("native source function original resolution authority changed")
    if track:
        analysis._source_function_used_scopes.add(id(scope))


@dataclass(frozen=True, slots=True)
class _SessionScopeRoles:
    session: object
    overrides: tuple


def _scope_roles(analysis):
    """Retain original authority and only the explicitly projected role changes."""
    session = getattr(analysis, "_source_session", None)
    if session is not None:
        session.verify(analysis)
        from compiler.frontend.source_session import ProjectedRoutines
        overrides = (tuple((name, routine.scope) for name, routine in sorted(analysis.routines.overrides.items()))
                     if isinstance(analysis.routines, ProjectedRoutines) else ())
        return _SessionScopeRoles(session, overrides)
    roles = {("module", name): scope for name, scope in analysis.modules.items()}
    roles.update((("routine", name), routine.scope) for name, routine in analysis.routines.items())
    roles.update((("numerical_helper", name), routine.scope)
                 for name, routine in analysis.numerical_helpers.items())
    roles.update((("associate", identity), record.scope)
                 for identity, record in analysis._associate_scopes.items() if record.available)
    return roles


def _validate_scope_roles(analysis, roles, consulted):
    # Authenticating a retained scope alone does not bind a registry that was
    # replaced by an equal-looking scope while this demand was running.
    current = _scope_roles(analysis)
    if isinstance(roles, _SessionScopeRoles):
        if (not isinstance(current, _SessionScopeRoles) or current.session is not roles.session
                or len(current.overrides) != len(roles.overrides)
                or any(left[0] != right[0] or left[1] is not right[1]
                       for left, right in zip(current.overrides, roles.overrides, strict=True))):
            raise CompilationError("native source function original resolution authority changed")
        return
    if isinstance(current, _SessionScopeRoles):
        raise CompilationError("native source function original resolution authority changed")
    if any(current.get(role) is not scope for role, scope in roles.items() if id(scope) in consulted):
        raise CompilationError("native source function original resolution authority changed")


def original_name_absent(analysis, scope, name, *, _scope_validator=None, _generic_inapplicable=None):
    """Check a closed original name lookup, including host and USE reexports.

    An unavailable namespace cannot prove absence. This is an internal lookup
    result, not a reusable authority token; the consuming proof must retain
    these same named resolution scopes through its publication boundary.
    A consuming type proof may exclude exact generic definitions; all remaining
    host/USE namespaces must still be closed by this same bounded lookup.
    """
    pending, seen = [(scope, name, 0)], set()
    while pending:
        current, spelling, depth = pending.pop()
        if current is None or (id(current), spelling) in seen:
            continue
        if depth >= analysis.depth_limit or len(seen) >= analysis.operation_limit:
            raise CompilationError("original closed-name lookup budget exhausted")
        seen.add((id(current), spelling))
        if _scope_validator is None:
            _validate_scope(analysis, current)
        else:
            _scope_validator(current)
        if (spelling in current.bindings or spelling in current.procedures
                or spelling in current.procedure_arguments or spelling in current.externals
                or spelling in current.intrinsic_procedures):
            return False
        if (spelling in current.generics and (_generic_inapplicable is None
                or not _generic_inapplicable(current, spelling))):
            return False
        choices = analysis.use_sources(current, spelling)
        if choices is None:
            raise CompilationError("original closed-name namespace is unavailable")
        for module, remote in choices:
            owner = analysis.modules.get(module)
            if owner is None:
                raise CompilationError("original closed-name namespace is unavailable")
            if analysis._exported(owner, remote):
                pending.append((owner, remote, depth + 1))
            elif spelling in current.imports:
                raise CompilationError("original closed-name import is private")
        pending.append((current.parent, spelling, depth + 1))
    return True


def _validate_resolution(analysis, scope, nodes, *, tick=None, names=(), _scope_validator=None):
    visited, checked = set(), set()

    def resolve(current, name, depth=0):
        if current is None or (id(current), name) in visited:
            return
        visited.add((id(current), name))
        if tick is not None:
            tick("lookup")
        if len(visited) > analysis.operation_limit or depth >= analysis.depth_limit:
            raise CompilationError("native source function resolution budget exhausted")
        if id(current) not in checked:
            if _scope_validator is None:
                _validate_scope(analysis, current)
            else:
                _scope_validator(current)
            checked.add(id(current))
        if (name in current.bindings or name in current.procedures or name in current.procedure_arguments
                or name in current.intrinsic_procedures or name in current.generics):
            return
        sources = analysis.use_sources(current, name)
        for module, remote in sources or ():
            imported = analysis.modules.get(module)
            if imported is not None:
                resolve(imported, remote, depth + 1)
        resolve(current.parent, name, depth + 1)

    # Explicit names are semantic lookup obligations such as registering a
    # visible module allocation. They never stand in for executable source or
    # grant expression authority; the same original namespace checks apply.
    for name in names:
        resolve(scope, name)
    for node in nodes:
        if tick is not None:
            tick("syntax")
        if type(node).__name__ in {"Name", "Type_Name", "Intrinsic_Name"}:
            resolve(analysis.source_scope_for(node, scope), str(node).lower())


def _origin(analysis):
    seen = set()
    while getattr(analysis, "_descriptor_source", None) is not None:
        if id(analysis) in seen:
            raise CompilationError("native source function projection is cyclic")
        seen.add(id(analysis))
        analysis = analysis._descriptor_source
    return analysis


def _original(analysis, procedure):
    if procedure in analysis.routines:
        return analysis._require_original(procedure)
    session = getattr(analysis, "_source_session", None)
    if session is not None:
        routine = session.routine(analysis, procedure, function=True)
        if routine.source_kind not in {"function", "internal_function"}:
            raise CompilationError("native source function requires original lexical source authority")
        return routine
    analysis.verify_source()
    syntax = analysis._source_syntax.checkpoint()
    routine = analysis.numerical_helpers.get(procedure)
    role = analysis._numerical_roles.get(procedure)
    if (routine is None or routine.source_kind not in {"function", "internal_function"}
            or role is None or routine.scope.node is not role[0]
            or analysis._routine_signature(routine) != role[1] or not syntax.unchanged(routine.scope.node)):
        raise CompilationError("native source function requires original lexical source authority")
    # Authenticate original module and span authority too. Functions remain
    # absent from public routine/structured-entry registration.
    if any(analysis.modules.get(name) is None or analysis.modules[name].node is not node or not syntax.unchanged(node)
           for name, (node, _text) in analysis._source_module_roles.items()):
        raise CompilationError("native source function module authority changed")
    if any(getattr(node, "item", None) is not item or item.span != span
           or getattr(item, "fort_original_span", None) != original_span
           for node, item, span, original_span in analysis._source_provenance.values()):
        raise CompilationError("native source function span authority changed")
    return routine


def _source_identity(analysis, routine):
    session = getattr(analysis, "_source_session", None)
    if session is not None:
        session.verify(analysis)
        if not session.contains(routine):
            raise CompilationError("native source function requires original lexical source authority")
        return _hash({"version": SOURCE_FUNCTION_VERSION, "session": session.identity,
                      "procedure": routine.qualified})
    return _hash({"version": SOURCE_FUNCTION_VERSION, "sources": analysis.inputs.identity(),
                  "procedure": routine.qualified, "signature": analysis._routine_signature(routine),
                  "source": str(routine.scope.node)})


def _contract(analysis, routine, *, require_declared_purity=True):
    statement = next(node for node in routine.scope.node.content if type(node).__name__ == "Function_Stmt")
    prefix = {str(item).lower() for item in getattr(statement.items[0], "items", ()) or ()}
    if require_declared_purity and ("impure" in prefix or not prefix.intersection({"pure", "elemental"})):
        raise CompilationError("native source function requires PURE or implicitly pure ELEMENTAL source")
    if not require_declared_purity:
        if "impure" in prefix:
            raise CompilationError("descriptor source function explicit IMPURE requires separate authority")
        specification = next((part for part in routine.scope.node.content
                              if type(part).__name__ == "Specification_Part"), None)
        if (any(type(node).__name__ in {"Common_Stmt", "Equivalence_Stmt", "Namelist_Stmt", "Cray_Pointer_Stmt",
                                      "Bind_Stmt", "Codimension_Attr_Spec"} for node in _walk(specification))
                or any("target" in item.attributes or item.intent == "out" for item in routine.scope.bindings.values())):
            raise CompilationError("descriptor source function storage association or entry definition is unproved")
    result = str(statement.items[3].items[0] if statement.items[3] is not None else statement.items[1]).lower()
    binding = routine.scope.bindings.get(result)
    if (binding is None or binding.rank or binding.dtype not in {"real", "integer", "logical"}
            or binding.kind not in {4, 8}):
        raise CompilationError("native source function requires a fixed numeric or logical scalar result")
    forbidden = {"pointer", "allocatable", "optional", "value", "save", "volatile", "asynchronous"}
    if any(item.attributes & forbidden
           or item.rank and "target" in item.attributes
           or item.dtype not in {"real", "integer", "logical"} or item.kind not in {4, 8}
           for item in routine.scope.bindings.values()):
        raise CompilationError("native source function requires fixed private scalars and stable required arguments")
    arrays = []
    for item in routine.scope.bindings.values():
        if item.rank:
            bounds = _fixed_array_bounds(analysis, item)
            arrays.append({"resource": item.root, "rank": item.rank, "bounds": [list(axis) for axis in bounds],
                           "role": "readonly_argument" if item.name in routine.arguments else "private_local"})
    purity = "pure" if "pure" in prefix else "implicitly pure elemental" if "elemental" in prefix else "not declared PURE in configured source"
    return binding.public(), purity, arrays


def _projection(analysis, routine):
    # Use the established private native-analysis isolation, including proof
    # registries, then add exactly this original helper to that private view.
    # No synthetic CALL, original entry role or persistent cache is created.
    from compiler.scopes.segments import _native_analysis

    projected = _native_analysis(analysis)
    projected = projected.private_projection(routines=True)
    projected.project_routine(routine.qualified, routine)
    projected.summaries, projected._closures, projected._native_sections = {}, {}, {}
    projected._summary_cache = SummaryCache(max_entries=0)
    projected._descriptor_source = _origin(analysis)
    return projected


def _descriptor_declarations(analysis, routine):
    """Conservative fixed-storage contract for every original closure member."""
    specification = next((part for part in routine.scope.node.content
                          if type(part).__name__ == "Specification_Part"), None)
    if any(type(node).__name__ in {"Common_Stmt", "Equivalence_Stmt", "Namelist_Stmt", "Cray_Pointer_Stmt",
                                  "Bind_Stmt", "Codimension_Attr_Spec"} for node in _walk(specification)):
        raise CompilationError("descriptor source function closure storage association is unproved")
    unsafe = {"pointer", "allocatable", "optional", "value", "save", "volatile", "asynchronous", "target"}
    for binding in routine.scope.bindings.values():
        if (binding.attributes & unsafe or binding.intent == "out"
                or binding.dtype not in {"real", "integer", "logical"} or binding.kind not in {4, 8}):
            raise CompilationError("descriptor source function closure requires fixed nonassociated numeric storage")
        if binding.rank:
            _fixed_array_bounds(analysis, binding)


@dataclass(frozen=True)
class NativeSourceFunctionProof:
    procedure: str
    callee: str
    identity: str
    source_identity: str
    callee_source_identity: str
    resolved: object = field(repr=False, compare=False)
    _expression: object = field(repr=False, compare=False)
    _record: object = field(repr=False, compare=False)
    _effects: object = field(repr=False, compare=False)

    @property
    def effects(self):
        return _thaw(self._effects)

    def public(self):
        return {"schema_version": SOURCE_FUNCTION_VERSION, "proof_role": "native_readonly_source_function",
                "proof_identity": self.identity, "procedure": self.procedure, "callee": self.callee,
                "source_identity": self.source_identity, "callee_source_identity": self.callee_source_identity,
                **_thaw(self._record)}

    def validate(self, analysis, procedure, expression, *, resolution_tick=None):
        origin = _origin(analysis)
        caller, callee = _original(origin, procedure), _original(origin, self.callee)
        _validate_resolution(origin, caller.scope, _walk(expression), tick=resolution_tick)
        _validate_resolution(origin, callee.scope, _walk(callee.execution), tick=resolution_tick)
        if (procedure != self.procedure or expression is not self._expression
                or origin._native_functions.get(self.identity) is not self
                or _source_identity(origin, caller) != self.source_identity
                or _source_identity(origin, callee) != self.callee_source_identity
                or not any(node is expression for node in _walk(caller.execution))):
            raise CompilationError("native source function lacks registered original expression authority")
        resolved = resolve_source_function(origin, origin.source_scope_for(expression, caller.scope), expression)
        if _hash(resolved.public()) != _hash(_thaw(self._record)["source_call"]):
            raise CompilationError("native source function original argument mapping changed")
        return self


def prove_native_source_function(analysis, procedure, expression, active, closure):
    origin = _origin(analysis)
    caller = _original(origin, procedure)
    position = next((index for index, node in enumerate(_walk(caller.execution)) if node is expression), None)
    if position is None:
        raise CompilationError("native source function requires an exact original expression")
    _validate_resolution(origin, caller.scope, _walk(expression))
    resolved = resolve_source_function(origin, origin.source_scope_for(expression, caller.scope), expression)
    callee = _original(origin, resolved.procedure)
    result, purity, arrays = _contract(origin, callee)
    _validate_resolution(origin, callee.scope, _walk(callee.execution))
    syntax = [node for part in callee.scope.node.content
              if type(part).__name__ in {"Specification_Part", "Execution_Part"}
              for node in _walk(part)]
    if len(syntax) > origin.operation_limit:
        raise CompilationError("native source function syntax budget exhausted")
    projected = _projection(analysis, callee)
    child = projected.summarize(callee.qualified, (*active, procedure), _closure=closure)
    if not child["complete"]:
        raise CompilationError("native source function effects incomplete: " + callee.qualified + ": "
                               + "; ".join(child["reasons"]))
    completion = child.get("native_completion", {})
    if (not completion.get("available") or completion.get("has_openmp_in_closure") is not False
            or completion.get("has_opaque_calls_in_closure") is not False):
        raise CompilationError("native source function completion is unproved: " + callee.qualified)
    if (child.get("definition_changes") or child.get("persistent_state")
            or any(effect["kind"] not in {"read", "descriptor_read"} for effect in child["ordered_effects"])):
        raise CompilationError("native source function transitive effects are not read-only: " + callee.qualified)
    record = {"source_expression": str(expression), "source_position": position,
              "source_call": resolved.public(), "summary_identity": child["summary_identity"],
              "result": result, "purity": purity, "fixed_arrays": arrays, "complete_transitive_effects": True,
              "native_only": True, "requires_original_execution": True,
              "argument_evaluation": "original expression position, keyword order and guards",
              "gpu_legality_established": False, "numerical_lowering_authorized": False,
              "native_callback_authorized": False, "exception_observers_authorized": False}
    source, target_source = _source_identity(origin, caller), _source_identity(origin, callee)
    identity = _hash({"source": source, "callee_source": target_source, "record": record})
    existing = origin._native_functions.get(identity)
    if existing is not None:
        return existing.validate(origin, procedure, expression), child
    if len(origin._native_functions) >= origin.operation_limit:
        raise CompilationError("native source function proof budget exhausted")
    proof = NativeSourceFunctionProof(procedure, callee.qualified, identity, source, target_source,
                                      resolved, expression, _freeze(record), _freeze(deepcopy(child["ordered_effects"])))
    origin._native_functions[identity] = proof
    return proof, child


@dataclass(frozen=True)
class DescriptorSourceFunctionProof:
    """Original native storage evidence, never a PURE/executable function token."""

    procedure: str
    callee: str
    identity: str
    source_identity: str
    callee_source_identity: str
    limits: tuple
    resolved: object = field(repr=False, compare=False)
    _expression: object = field(repr=False, compare=False)
    _record: object = field(repr=False, compare=False)

    def public(self):
        return {"schema_version": 1, "proof_role": "descriptor_readonly_source_function",
                "proof_identity": self.identity, "procedure": self.procedure, "callee": self.callee,
                "source_identity": self.source_identity, "callee_source_identity": self.callee_source_identity,
                **_thaw(self._record)}

    def validate(self, analysis, procedure, expression, *, resolution_tick=None):
        origin = _origin(analysis)
        caller, callee = _original(origin, procedure), _original(origin, self.callee)
        _validate_resolution(origin, caller.scope, _walk(expression), tick=resolution_tick)
        _validate_resolution(origin, callee.scope, _walk(callee.execution), tick=resolution_tick)
        if (procedure != self.procedure or expression is not self._expression
                or origin._descriptor_functions.get(self.identity) is not self
                or self.limits != (origin.operation_limit, origin.procedure_limit, origin.depth_limit)
                or _source_identity(origin, caller) != self.source_identity
                or _source_identity(origin, callee) != self.callee_source_identity
                or not any(node is expression for node in _walk(caller.execution))):
            raise CompilationError("descriptor source function lacks registered original expression authority")
        resolved = resolve_source_function(origin, origin.source_scope_for(expression, caller.scope), expression)
        if _hash(resolved.public()) != _hash(_thaw(self._record)["source_call"]):
            raise CompilationError("descriptor source function original argument mapping changed")
        return self


def prove_descriptor_source_function(analysis, procedure, expression):
    """Issue a distinct, source-derived no-mutation token without assuming PURE."""
    from compiler.frontend.source_effects import _Closure

    origin = _origin(analysis)
    caller = _original(origin, procedure)
    position = next((index for index, node in enumerate(_walk(caller.execution)) if node is expression), None)
    if position is None:
        raise CompilationError("descriptor source function requires an exact original expression")
    _validate_resolution(origin, caller.scope, _walk(expression))
    resolved = resolve_source_function(origin, origin.source_scope_for(expression, caller.scope), expression)
    callee = _original(origin, resolved.procedure)
    result, purity, arrays = _contract(origin, callee, require_declared_purity=False)
    _validate_resolution(origin, callee.scope, _walk(callee.execution))
    syntax = [node for part in callee.scope.node.content
              if type(part).__name__ in {"Specification_Part", "Execution_Part"} for node in _walk(part)]
    if len(syntax) > origin.operation_limit:
        raise CompilationError("descriptor source function syntax budget exhausted")
    closure = _Closure()
    child = _projection(origin, callee).summarize(callee.qualified, (procedure,), _closure=closure)
    completion = child.get("native_completion", {})
    if (not child["complete"] or not completion.get("available")
            or completion.get("has_openmp_in_closure") is not False
            or completion.get("has_opaque_calls_in_closure") is not False
            or child.get("definition_changes") or child.get("persistent_state")
            or any(effect["kind"] not in {"read", "descriptor_read"} for effect in child["ordered_effects"])):
        raise CompilationError("descriptor source function complete read-only closure is unproved: "
                               + callee.qualified + ": " + "; ".join(child["reasons"]))
    # A diagnostic/consumer hook cannot change an earlier authenticated body or
    # namespace while a later child is summarized, then publish its old effects.
    for name in closure.summaries:
        original = _original(origin, name)
        _descriptor_declarations(origin, original)
        _validate_resolution(origin, original.scope, _walk(original.execution))
    _original(origin, procedure)
    origin.verify_publication()
    record = {"source_expression": str(expression), "source_position": position,
              "source_call": resolved.public(), "summary_identity": child["summary_identity"],
              "result": result, "configured_purity": purity, "fixed_arrays": arrays,
              "complete_transitive_effects": True, "requires_original_execution": True,
              "argument_evaluation": "unchanged original expression, order and control",
              "descriptor_evidence_only": True, "execution_authorized": False,
              "payload_access_authorized": False, "entry_definition_authority": False,
              "capture_authorized": False, "pure_scalar_authorized": False,
              "gpu_legality_established": False, "numerical_lowering_authorized": False,
              "native_callback_authorized": False, "exception_observers_authorized": False}
    source, target = _source_identity(origin, caller), _source_identity(origin, callee)
    identity = _hash({"source": source, "callee_source": target, "record": record})
    registry = origin._descriptor_functions
    if identity in registry:
        return registry[identity].validate(origin, procedure, expression)
    if len(registry) >= origin.operation_limit:
        raise CompilationError("descriptor source function proof budget exhausted")
    proof = DescriptorSourceFunctionProof(procedure, callee.qualified, identity, source, target,
        (origin.operation_limit, origin.procedure_limit, origin.depth_limit), resolved, expression, _freeze(record))
    registry[identity] = proof
    return proof


def descriptor_source_function_call(analysis, procedure, expression):
    """Frontend admission for one exact unmet descriptor demand, not PURE."""
    origin = _origin(analysis)
    caller = _original(origin, procedure)
    if not any(node is expression for node in _walk(caller.execution)):
        raise CompilationError("descriptor function demand is not an exact original expression")
    _validate_resolution(origin, caller.scope, _walk(expression))
    scope = origin.source_scope_for(expression, caller.scope)
    try:
        resolved = resolve_source_function(origin, scope, expression)
    except CompilationError:
        # A separate descriptor-only issuer may use original type association
        # for readonly assumed-shape formals. It never lends the old physical
        # mapping, PURE-v2 token, or scalar numerical capture authority.
        from compiler.frontend.call_bindings import resolve_original_function
        from compiler.frontend.descriptor_array_functions import prove_descriptor_array_function

        original = resolve_original_function(origin, scope, expression)
        if not any(binding.rank and any(type(axis).__name__ == "Assumed_Shape_Spec"
                                        for axis in binding.shape_nodes)
                   for name, binding in original.callee.scope.bindings.items()
                   if name in original.callee.arguments):
            raise
        return prove_descriptor_array_function(origin, procedure, expression)
    callee = _original(origin, resolved.procedure)
    statement = next(part for part in callee.scope.node.content if type(part).__name__ == "Function_Stmt")
    prefix = {str(item).lower() for item in getattr(statement.items[0], "items", ()) or ()}
    if prefix.intersection({"pure", "elemental"}):
        # The historical PURE-v2 admission path and its registry are unchanged.
        return None
    return prove_descriptor_source_function(origin, procedure, expression)
