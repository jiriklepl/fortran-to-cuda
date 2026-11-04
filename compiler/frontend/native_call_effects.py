"""Complete effects for one unchanged native CALL, independent of future work.

Numeric arrays require ordinary coherent host transactions. All other external
storage requires an opaque physical reservation, including scalar state and
fixed fields of record arrays. Neither different source names nor this token
prove disjointness. Runtime registration must reject overlapping roles before
the CALL executes. No definitions, device work or task motion are authorized.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields

from compiler.frontend.native_completion import native_call_storage_structures
from compiler.frontend.native_opaque_writes import _prerequisites, _require_prerequisites
from compiler.frontend.resource_reachability import _authenticate_binding, _digest, prove_resource_reachability
from compiler.frontend.storage_mutation_effects import prove_storage_mutation_effects, source_registry_budget
from compiler.ir import CompilationError

VERSION = 1
_ISSUER = object()
_REGISTRY = "_native_call_effects"
_UNSTABLE = {"pointer", "optional", "volatile", "asynchronous", "value", "codimension"}


def _require_complete(reachability):
    graph = reachability.native_graph
    if reachability.barriers or not graph.completion_eligible or graph.lifetime_obligations:
        raise CompilationError("native CALL requires complete synchronous original effects: " + "; ".join(
            dict.fromkeys(item.reason for item in reachability.barriers)))


def _project(analysis, procedure, reachability, storage):
    _require_complete(reachability)
    graph = reachability.native_graph
    record = storage.public()
    if not record["complete"] or record["frontiers"] or record["mutations"] or record["escapes"]:
        raise CompilationError("native CALL requires stable storage without escapes")
    owner = analysis._require_original(procedure).scope
    callees = {id(template._routine.scope): template._routine for template in graph.templates
               if template._routine.scope is not owner}
    paths, bindings = {}, {}
    for access in reachability.accesses:
        binding = access.storage.binding
        callee = callees.get(id(binding.declaring_scope))
        if (callee is not None and callee.scope.bindings.get(binding.name) is binding
                and binding.name not in callee.arguments and "save" not in binding.attributes):
            continue  # Substituted callee-private storage; ordered accesses remain in the graph.
        _authenticate_binding(analysis, binding)
        if binding.attributes & _UNSTABLE or any(part.attributes & _UNSTABLE for part in access.storage.components):
            raise CompilationError("native CALL has unreserved indirect storage: " + access.storage.resource)
        if any("allocatable" in part.attributes for part in access.storage.components):
            raise CompilationError("native CALL component needs a separate allocation mapping: " + access.storage.resource)
        bindings[id(binding)] = binding
        paths.setdefault(id(binding), []).append(access)
    result = []
    if not storage.descriptors_unchanged(tuple(binding.root for binding in bindings.values())):
        raise CompilationError("native CALL changes a reserved descriptor")
    for key, binding in bindings.items():
        accesses = paths[key]
        actions = frozenset(item.kind for item in accesses)
        if actions - {"read", "write", "undefine", "association", "descriptor"}:
            raise CompilationError("native CALL contains an unsupported storage effect: " + binding.root)
        if "parameter" in binding.attributes:
            if actions - {"read", "association", "descriptor"}:
                raise CompilationError("native CALL writes original constant storage: " + binding.root)
            role = "constant"
        elif binding.dtype.startswith("type("):
            if not binding.rank or any(not item.storage.components and item.kind in {"read", "write", "undefine"}
                                       for item in accesses):
                raise CompilationError("native CALL requires fixed fields of an original record array: " + binding.root)
            role = "reservation"
        elif binding.dtype in {"real", "integer", "logical"} and binding.kind in {1, 4, 8}:
            if any(item.storage.components for item in accesses):
                raise CompilationError("native CALL numeric storage has an unmapped component: " + binding.root)
            role = "managed" if binding.rank else "reservation"
            if role == "managed" and ("undefine" in actions or binding.dtype == "logical" or binding.kind == 1):
                raise CompilationError("native CALL managed array needs separate definition or layout authority: " + binding.root)
        else:
            raise CompilationError("native CALL storage has no physical reservation contract: " + binding.root)
        result.append((binding, role, actions))
    return tuple(result)


@dataclass(frozen=True)
class NativeCallEffects:
    identity: str
    procedure: str
    resources: tuple = field(repr=False, compare=False)
    reachability: object = field(repr=False, compare=False)
    storage: object = field(repr=False, compare=False)
    _analysis: object = field(repr=False, compare=False)
    _selection: tuple = field(repr=False, compare=False)
    _prerequisites: tuple = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)

    def validate(self, analysis, procedure, selection):
        attached = tuple((part.name, object.__getattribute__(self, part.name)) for part in fields(self))
        registry = getattr(analysis, _REGISTRY, None)
        if (type(self) is not NativeCallEffects or self._issuer is not _ISSUER
                or analysis is not self._analysis or procedure != self.procedure
                or type(selection) is not tuple or len(selection) != len(self._selection)
                or any(a is not b for a, b in zip(selection, self._selection, strict=True))
                or getattr(analysis, _REGISTRY, {}).get(self.identity) is not self):
            raise CompilationError("native CALL requires its exact registered source effects")
        # Share the existing complete native graph/storage receipt fence. Its
        # immutable attachments cover every ordered access and binding, not
        # only the resources selected for coherent payload transactions.
        _require_prerequisites(analysis, self._prerequisites)
        self.reachability.validate(analysis, procedure, selection, ())
        if (self.storage._native_graph is not self.reachability.native_graph
                or len(self.storage._selection) != len(selection)
                or any(a is not b for a, b in zip(self.storage._selection, selection, strict=True))):
            raise CompilationError("native CALL storage and payload graphs differ")
        self.storage.validate(analysis, procedure, selection=self.storage._selection, joined_storage=self.storage._joined_storage)
        current = _project(analysis, procedure, self.reachability, self.storage)
        if (len(current) != len(self.resources) or any(a is not b or (role, actions) != (old_role, old_actions)
                for (a, role, actions), (b, old_role, old_actions) in zip(current, self.resources, strict=True))):
            raise CompilationError("native CALL original storage projection changed")
        _require_prerequisites(analysis, self._prerequisites)
        if (getattr(analysis, _REGISTRY, None) is not registry or registry.get(self.identity) is not self
                or any(object.__getattribute__(self, name) is not value for name, value in attached)):
            raise CompilationError("native CALL retained source effects changed during validation")
        return self

    def public(self):
        return {"schema_version": VERSION, "identity": self.identity, "procedure": self.procedure,
                "reachability_identity": self.reachability.identity, "storage_identity": self.storage.identity,
                "resources": [{"resource": binding.root, "role": role, "actions": sorted(actions)}
                              for binding, role, actions in self.resources],
                "alias_authorized": False, "capture_authorized": False, "definitions_authorized": False,
                "motion_authorized": False, "internal_cuts_authorized": False,
                "execution": "unchanged original CALL after coherent access and physical alias registration"}


def prove_native_call_effects(analysis, procedure, selection):
    if type(selection) is not tuple or len(selection) != 1 or type(selection[0]).__name__ != "Call_Stmt":
        raise CompilationError("native CALL effects require one unchanged original CALL")
    reachability = prove_resource_reachability(analysis, procedure, selection, ())
    _require_complete(reachability)
    structures = native_call_storage_structures(analysis, reachability.native_graph)
    storage = prove_storage_mutation_effects(analysis, procedure, selection=selection,
        joined_storage=structures, native_graph=reachability.native_graph)
    prerequisites = _prerequisites(analysis, reachability, storage)
    resources = _project(analysis, procedure, reachability, storage)
    identity = _digest({"version": VERSION, "reachability": reachability.identity, "storage": storage.identity,
                        "resources": [(id(binding), role, sorted(actions)) for binding, role, actions in resources]})
    registry = getattr(analysis, _REGISTRY, None)
    if registry is None:
        registry = {}
        setattr(analysis, _REGISTRY, registry)
    if identity in registry:
        return registry[identity].validate(analysis, procedure, selection)
    if len(registry) >= source_registry_budget(analysis):
        raise CompilationError("native CALL effect registry budget exhausted")
    proof = NativeCallEffects(identity, procedure, resources, reachability, storage, analysis, selection, prerequisites, _ISSUER)
    registry[identity] = proof
    try:
        return proof.validate(analysis, procedure, selection)
    except BaseException:
        if registry.get(identity) is proof:
            del registry[identity]
        raise
