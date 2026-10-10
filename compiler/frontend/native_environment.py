"""Authenticated original IEEE operations and strictly native local state.

These proofs retain the Fortran intrinsic, representation, lexical storage and
calling threads. They grant no device access, numerical legality or team cuts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256

from compiler.frontend.structured_effects import _freeze, _thaw
from compiler.ir import CompilationError

NATIVE_ENVIRONMENT_VERSION = 1
NATIVE_STATE_VERSION = 1
_SCAN_LIMIT = 65536
_EXCEPTIONS = {
    "ieee_get_halting_mode", "ieee_set_halting_mode", "ieee_all", "ieee_usual",
    "ieee_invalid", "ieee_overflow", "ieee_divide_by_zero", "ieee_underflow", "ieee_inexact",
}
_EXPORTS = {module: {name: "$intrinsic::ieee_exceptions::" + name for name in _EXCEPTIONS}
            for module in ("ieee_exceptions", "ieee_arithmetic")}
_CALLS = {"$intrinsic::ieee_exceptions::ieee_get_halting_mode",
          "$intrinsic::ieee_exceptions::ieee_set_halting_mode"}
_SCALAR_FLAGS = {"$intrinsic::ieee_exceptions::" + name for name in _EXCEPTIONS
                 if name not in {"ieee_all", "ieee_usual", "ieee_get_halting_mode", "ieee_set_halting_mode"}}
_ALL = "$intrinsic::ieee_exceptions::ieee_all"
_FORBIDDEN = {"save", "target", "pointer", "allocatable", "optional", "volatile", "asynchronous", "value", "parameter"}


def _kind(node):
    return type(node).__name__


def _children(node):
    return getattr(node, "content", getattr(node, "items", ())) or ()


def _walk(node):
    stack, visits = [iter((node,))], 0
    while stack:
        try:
            item = next(stack[-1])
        except StopIteration:
            stack.pop()
            continue
        visits += 1
        if visits > _SCAN_LIMIT:
            raise CompilationError("native environment original-source scan budget exhausted")
        if isinstance(item, (tuple, list)):
            stack.append(iter(item))
        elif hasattr(item, "items") or hasattr(item, "content"):
            yield item
            stack.append(iter(_children(item)))


def _hash(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def canonical_intrinsic_export(identity):
    """Recognize only the finite reviewed standard export set."""
    if not isinstance(identity, str):
        return None
    pieces = identity.split("::")
    return _EXPORTS.get(pieces[1], {}).get(pieces[2]) if len(pieces) == 3 and pieces[0] == "$intrinsic" else None


def intrinsic_export(analysis, scope, name):
    """Resolve one standard export through actual USE/host associations.

    Unknown wildcard modules, local procedures/storage and ambiguous exports
    cannot acquire intrinsic authority merely from an IEEE-looking spelling.
    """
    return _resolve_intrinsic_export(analysis, scope, name, _EXPORTS)


def _resolve_intrinsic_export(analysis, scope, name, reviewed_exports):
    """Share resolution mechanics; each native proof owns its finite exports."""
    name = str(name).lower()

    def canonical(identity):
        pieces = identity.split("::") if isinstance(identity, str) else ()
        return (reviewed_exports.get(pieces[1], {}).get(pieces[2])
                if len(pieces) == 3 and pieces[0] == "$intrinsic" else None)

    def resolve(owner, spelling, active):
        if owner is None:
            return set()
        key = id(owner), spelling
        if key in active or len(active) > analysis.depth_limit + analysis.procedure_limit:
            return {"$unresolved"}
        active = active | {key}
        if (spelling in owner.bindings or spelling in owner.procedure_arguments or spelling in owner.procedures
                or spelling in owner.externals or spelling in owner.generics or spelling in owner.ambiguous_imports):
            return {"$shadowed"}
        found = set()
        if spelling in owner.intrinsic_procedures:
            found.add(canonical(owner.intrinsic_procedures[spelling]) or "$unreviewed")
        for module in owner.intrinsic_wildcards:
            if spelling not in getattr(owner, "intrinsic_exclusions", {}).get(module, ()):
                value = reviewed_exports.get(module, {}).get(spelling)
                if value is not None:
                    found.add(value)
        sources = analysis.use_sources(owner, spelling)
        if sources is None:
            return {"$ambiguous"}
        for module, remote in sources:
            target = analysis.modules.get(module)
            if target is None:
                found.add("$unresolved")
            elif analysis._exported(target, remote):
                found.update(resolve(target, remote, active))
        return found if found else resolve(owner.parent, spelling, active)

    choices = resolve(scope, name, frozenset())
    return next(iter(choices)) if len(choices) == 1 and next(iter(choices)).startswith("$intrinsic::") else None


def _origin(analysis, procedure):
    seen = set()
    while getattr(analysis, "_descriptor_source", None) is not None:
        if id(analysis) in seen:
            raise CompilationError("native environment projection source is cyclic")
        seen.add(id(analysis))
        analysis = analysis._descriptor_source
    routine = analysis._require_original(procedure)
    return analysis, routine


def _source_identity(analysis, routine):
    return _hash({"schema_version": NATIVE_ENVIRONMENT_VERSION, "sources": analysis.inputs.identity(),
                  "procedure": routine.qualified, "signature": analysis._routine_signature(routine),
                  "source": str(routine.scope.node)})


def _associate(analysis, routine, call):
    if _kind(call) != "Call_Stmt" or _kind(call.items[0]) != "Name":
        raise CompilationError("native environment requires an original direct intrinsic CALL")
    scope = analysis.source_scope_for(call, routine.scope)
    intrinsic = intrinsic_export(analysis, scope, call.items[0])
    if intrinsic not in _CALLS:
        raise CompilationError("native environment intrinsic resolution is unproved")
    actuals, position, keyword = {}, 0, False
    for item in _children(call.items[1]):
        if _kind(item) == "Actual_Arg_Spec":
            keyword = True
            name, value = str(item.items[0]).lower(), item.items[1]
        else:
            if keyword or position >= 2:
                raise CompilationError("native environment argument association is unproved")
            name, value = ("flag", "halting")[position], item
            position += 1
        if name not in {"flag", "halting"} or name in actuals:
            raise CompilationError("native environment argument association is unproved")
        actuals[name] = value
    if set(actuals) != {"flag", "halting"} or _kind(actuals["flag"]) != "Name":
        raise CompilationError("native environment requires reviewed original FLAG and HALTING actuals")
    flag = intrinsic_export(analysis, scope, actuals["flag"])
    if flag not in _SCALAR_FLAGS | {_ALL}:
        raise CompilationError("native environment FLAG is not an authenticated scalar exception or IEEE_ALL")
    return scope, intrinsic, flag, actuals["halting"]


def _state_record(analysis, routine, binding):
    if (binding.declaring_scope is not routine.scope or binding.name in routine.arguments
            or not binding.root.startswith(routine.qualified + "::")
            or binding.dtype != "logical" or binding.kind != 4 or binding.rank not in {0, 1}
            or binding.attributes & _FORBIDDEN):
        raise CompilationError("native environment state requires original unsaved local default LOGICAL storage")
    shape = "scalar"
    if binding.rank:
        if len(binding.shape_nodes) != 1 or _kind(binding.shape_nodes[0]) != "Explicit_Shape_Spec":
            raise CompilationError("native environment state requires original SIZE(IEEE_ALL) bounds")
        lower, upper = binding.shape_nodes[0].items
        if lower is not None and str(lower) != "1":
            raise CompilationError("native environment state requires original SIZE(IEEE_ALL) bounds")
        if _kind(upper) != "Intrinsic_Function_Reference" or str(upper.items[0]).lower() != "size":
            raise CompilationError("native environment state requires original SIZE(IEEE_ALL) bounds")
        if (analysis._binding(routine.scope, "size") or analysis._candidates(routine.scope, "size")
                or analysis._unknown_exports(routine.scope)):
            raise CompilationError("native environment state SIZE intrinsic resolution is unproved")
        args = tuple(_children(upper.items[1]))
        if (len(args) != 1 or _kind(args[0]) != "Name"
                or intrinsic_export(analysis, routine.scope, args[0]) != _ALL):
            raise CompilationError("native environment state requires original SIZE(IEEE_ALL) bounds")
        shape = "original SIZE(intrinsic IEEE_ALL); no extent or payload evaluation"
    for node in _walk(routine.scope.node):
        if _kind(node) in {"Common_Stmt", "Equivalence_Stmt"}:
            raise CompilationError("native environment state storage association is unproved")
        if _kind(node) == "Comment" and str(node).lstrip().lower().startswith("!$omp threadprivate"):
            raise CompilationError("native environment state THREADPRIVATE ownership is unproved")
    allowed = set()
    for node in _walk(routine.execution):
        if _kind(node) != "Call_Stmt":
            continue
        try:
            _scope, _intrinsic, flag, halting = _associate(analysis, routine, node)
        except CompilationError:
            continue
        if _kind(halting) == "Name" and analysis._binding(_scope, halting) is binding:
            if ((binding.rank == 1 and flag != _ALL)
                    or (_intrinsic.endswith("::ieee_get_halting_mode") and binding.rank == 0 and flag == _ALL)):
                raise CompilationError("native environment FLAG/HALTING shape is unproved")
            allowed.add(id(halting))
    from compiler.frontend.component_bindings import references
    owners = {id(owner): owner for owner in (*analysis.routines.values(), *analysis.numerical_helpers.values())}
    for owner in owners.values():
        lexical = owner.scope
        while lexical is not None and lexical is not routine.scope:
            lexical = lexical.parent
        if lexical is None:
            continue
        specification = next((part for part in _children(owner.scope.node)
                              if _kind(part) == "Specification_Part"), None)
        # A declaration name is not a read. Its bounds/initializers and every
        # descendant specification expression still participate in the all-use
        # proof, including an uncalled child's automatic array bounds.
        declarations = {id(node.items[0]) for node in _walk(specification)
                        if owner is routine and _kind(node) == "Entity_Decl"
                        and str(node.items[0]).lower() == binding.name}
        for selected in (specification, owner.execution):
            for resolved, reference in references(analysis, owner.scope, selected):
                if (resolved.root == binding.root and id(reference) not in declarations
                        and (owner is not routine or id(reference) not in allowed)):
                    raise CompilationError("native environment local state has an unproved source use or escape")
    if not allowed:
        raise CompilationError("native environment local state lacks original intrinsic uses")
    return {"schema_version": NATIVE_STATE_VERSION, "resource": binding.root, "type": "logical",
            "kind": binding.kind, "rank": binding.rank, "shape_authority": shape,
            "storage": "unchanged original invocation-local Fortran storage",
            "all_uses": "original reviewed IEEE HALTING actuals in this lexical owner",
            "native_only": True, "device_capture": False, "managed_definitions": False,
            "allocation_or_escape": False, "representation_conversion": False}


@dataclass(frozen=True)
class NativeLocalStateProof:
    procedure: str
    source_identity: str
    identity: str
    binding: object = field(repr=False, compare=False)
    _record: object = field(repr=False, compare=False)

    def public(self):
        return {"proof_identity": self.identity, "source_identity": self.source_identity, **_thaw(self._record)}

    def validate(self, analysis, procedure, binding=None):
        origin, routine = _origin(analysis, procedure)
        if (procedure != self.procedure or origin._native_environment_states.get(self.identity) is not self
                or _source_identity(origin, routine) != self.source_identity
                or (binding is not None and (binding.root != self.binding.root or binding.public() != self.binding.public()))
                or _state_record(origin, routine, self.binding) != _thaw(self._record)):
            raise CompilationError("native environment local state lacks registered original source authority")
        return self


@dataclass(frozen=True)
class NativeEnvironmentProof:
    procedure: str
    source_identity: str
    identity: str
    intrinsic: str
    native_states: tuple[NativeLocalStateProof, ...]
    _call: object = field(repr=False, compare=False)
    _record: object = field(repr=False, compare=False)

    @property
    def native_state_bindings(self):
        return tuple(state.binding for state in self.native_states)

    @property
    def state_roots(self):
        return tuple(state.binding.root for state in self.native_states)

    @property
    def written_roots(self):
        return self.state_roots if self.intrinsic.endswith("::ieee_get_halting_mode") else ()

    def public(self):
        return {"schema_version": NATIVE_ENVIRONMENT_VERSION, "proof_role": "native_environment_operation",
                "proof_identity": self.identity, "source_identity": self.source_identity,
                "procedure": self.procedure, "intrinsic": self.intrinsic,
                "native_states": [state.public() for state in self.native_states], **_thaw(self._record)}

    def validate(self, analysis, procedure, original_call):
        origin, routine = _origin(analysis, procedure)
        if (procedure != self.procedure or original_call is not self._call
                or origin._native_environments.get(self.identity) is not self
                or _source_identity(origin, routine) != self.source_identity
                or not any(node is original_call for node in _walk(routine.execution))):
            raise CompilationError("native environment operation lacks registered original source authority")
        for state in self.native_states:
            state.validate(origin, procedure)
        return self


def prove_native_state(analysis, procedure, binding):
    origin, routine = _origin(analysis, procedure)
    original = routine.scope.bindings.get(binding.name)
    if original is None or original.root != binding.root or original.public() != binding.public():
        raise CompilationError("native environment local state requires original declaration authority")
    record, source = _state_record(origin, routine, original), _source_identity(origin, routine)
    identity = _hash({"source": source, "state": record})
    existing = origin._native_environment_states.get(identity)
    if existing is not None:
        return existing
    if len(origin._native_environment_states) >= origin.operation_limit:
        raise CompilationError("native environment local state proof budget exhausted")
    proof = NativeLocalStateProof(procedure, source, identity, original, _freeze(record))
    origin._native_environment_states[identity] = proof
    return proof


def prove_native_environment(analysis, procedure, original_call):
    origin, routine = _origin(analysis, procedure)
    if not any(node is original_call for node in _walk(routine.execution)):
        raise CompilationError("native environment operation requires an exact original CALL")
    scope, intrinsic, flag, halting = _associate(origin, routine, original_call)
    states = ()
    if _kind(halting) == "Name":
        binding = origin._binding(scope, halting)
        if binding is None:
            raise CompilationError("native environment HALTING storage is unresolved")
        states = (prove_native_state(origin, procedure, binding),)
        if ((binding.rank == 1 and flag != _ALL)
                or (intrinsic.endswith("::ieee_get_halting_mode") and binding.rank == 0 and flag == _ALL)):
            raise CompilationError("native environment FLAG/HALTING shape is unproved")
    elif not (intrinsic.endswith("::ieee_set_halting_mode") and _kind(halting) == "Logical_Literal_Constant"
              and halting.items[1] is None):
        raise CompilationError("native environment HALTING requires original local state or a SET logical literal")
    get = intrinsic.endswith("::ieee_get_halting_mode")
    effects = [{"kind": "environment_read" if get else "environment_write",
                "resource": "$native_environment::thread_fenv", "domain": "ieee_halting_mode",
                "thread_scope": "original calling thread", "source_access": str(original_call)}]
    for state in states:
        effects.append({"kind": "write" if get else "read", "resource": state.binding.root,
                        "rank": state.binding.rank, "type": "logical", "kind_width": state.binding.kind,
                        "section": "whole", "native_local_state": True, "source_access": str(halting)})
    record = {"source_call": str(original_call), "flag_identity": flag, "effects": effects,
              "requires_original_execution": True, "requires_original_thread_participation": True,
              "requires_completed_device_work": True, "native_only": True,
              "internal_cuts_authorized": False, "gpu_legality_established": False,
              "allocation_or_escape": False, "logical_representation_conversion": False}
    source = _source_identity(origin, routine)
    position = next(index for index, node in enumerate(_walk(routine.execution)) if node is original_call)
    identity = _hash({"source": source, "position": position, "intrinsic": intrinsic, "record": record,
                      "states": [state.identity for state in states]})
    existing = origin._native_environments.get(identity)
    if existing is not None:
        return existing
    if len(origin._native_environments) >= origin.operation_limit:
        raise CompilationError("native environment operation proof budget exhausted")
    proof = NativeEnvironmentProof(procedure, source, identity, intrinsic, states, original_call, _freeze(record))
    origin._native_environments[identity] = proof
    return proof
