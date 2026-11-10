"""Complete effects for unchanged native CALLs and whole joined groups.

Numeric arrays require ordinary coherent host transactions. All other external
storage requires an opaque physical reservation, including scalar state and
fixed fields of record arrays. Neither different source names nor this token
prove disjointness. Runtime registration must reject overlapping roles before
the CALL executes. No definitions, device work or task motion are authorized.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, replace

from compiler.frontend.native_completion import native_call_storage_structures
from compiler.frontend.native_opaque_writes import _prerequisites, _require_prerequisites
from compiler.frontend.resource_reachability import _authenticate_binding, _digest, prove_resource_reachability
from compiler.frontend.storage_mutation_effects import prove_storage_mutation_effects, source_registry_budget
from compiler.ir import CompilationError

VERSION = 2
_ISSUER = object()
_REGISTRY = "_native_call_effects"
_UNSTABLE = {"pointer", "optional", "volatile", "asynchronous", "value", "codimension"}


def _require_complete(reachability):
    graph = reachability.native_graph
    if reachability.barriers or not graph.completion_eligible or graph.lifetime_obligations:
        raise CompilationError("native CALL requires complete synchronous original effects: " + "; ".join(
            dict.fromkeys(item.reason for item in reachability.barriers)))


def _project(analysis, procedure, reachability, storage, *, completion=None):
    _require_complete(reachability)
    graph = reachability.native_graph
    record = storage.public()
    if not record["complete"] or record["frontiers"] or record["mutations"] or record["escapes"]:
        raise CompilationError("native CALL requires stable storage without escapes")
    owner = analysis._require_original(procedure).scope
    private = set()
    if completion is not None:
        from compiler.frontend.descriptor_teams import DescriptorTeamProof

        selected = analysis._selected_source(procedure, reachability._selection)[1]
        completion.validate(analysis, procedure, selected)
        structures = storage._joined_storage if type(storage._joined_storage) is tuple else (storage._joined_storage,)
        teams = [team for team in structures if type(team) is DescriptorTeamProof
                 and team.procedure == procedure and team.selected_node_ids == selected]
        if completion._graph is not graph or len(teams) != 1:
            raise CompilationError("native host group requires its original joined completion and storage")
        teams[0].validate(analysis, procedure, selected)
        private.update(teams[0].public()["private_bindings"])
    callees = {id(template._routine.scope): template._routine for template in graph.templates
               if template._routine.scope is not owner}
    paths, bindings = {}, {}
    for access in reachability.accesses:
        binding = access.storage.binding
        if binding.root in private:
            continue  # Thread-private storage cannot become a caller reservation.
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
    # A joined body can name fixed record fields without passing or querying
    # the enclosing array. The storage scanner then records those exact paths,
    # whereas the runtime reserves their common physical allocation. Consume
    # every recorded field path; do not invent a whole-object descriptor fact.
    storage_resources = set(record["resources"])
    descriptor_resources = tuple(dict.fromkeys(
        resource for key, binding in bindings.items()
        for resource in ((binding.root,) if binding.root in storage_resources else
                         tuple(access.storage.resource for access in paths[key]))))
    if not storage.descriptors_unchanged(descriptor_resources):
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
class NativeHostEffects:
    identity: str
    procedure: str
    resources: tuple = field(repr=False, compare=False)
    reachability: object = field(repr=False, compare=False)
    storage: object = field(repr=False, compare=False)
    _analysis: object = field(repr=False, compare=False)
    _selection: tuple = field(repr=False, compare=False)
    _prerequisites: tuple = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)
    completion: object = field(default=None, repr=False, compare=False)
    sections: object = field(default=None, repr=False, compare=False)

    @property
    def storage_structures(self):
        storage = self.storage._joined_storage
        return storage if type(storage) is tuple else (storage,)

    def validate(self, analysis, procedure, selection):
        from compiler.frontend.native_sections import NativeSections

        attached = tuple((part.name, object.__getattribute__(self, part.name)) for part in fields(self))
        registry = getattr(analysis, _REGISTRY, None)
        if (type(self) is not NativeHostEffects or self._issuer is not _ISSUER or not isinstance(self.sections, NativeSections)
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
        if self.completion is None and (len(selection) != 1 or type(selection[0]).__name__ != "Call_Stmt"):
            raise CompilationError("native host operation lost its original CALL or joined completion")
        current = _project(analysis, procedure, self.reachability, self.storage, completion=self.completion)
        if (len(current) != len(self.resources) or any(a is not b or (role, actions) != (old_role, old_actions)
                for (a, role, actions), (b, old_role, old_actions) in zip(current, self.resources, strict=True))):
            raise CompilationError("native CALL original storage projection changed")
        if self.identity != _identity(self.reachability, self.storage, current, self.completion, self.sections):
            raise CompilationError("native host source regions changed after issuance")
        _require_prerequisites(analysis, self._prerequisites)
        if (getattr(analysis, _REGISTRY, None) is not registry or registry.get(self.identity) is not self
                or any(object.__getattribute__(self, name) is not value for name, value in attached)):
            raise CompilationError("native CALL retained source effects changed during validation")
        return self

    def public(self):
        return {"schema_version": VERSION, "identity": self.identity, "procedure": self.procedure,
                "reachability_identity": self.reachability.identity, "storage_identity": self.storage.identity,
                "selection_kind": "joined_group" if self.completion is not None else "call",
                "completion_identity": self.completion.identity if self.completion is not None else None,
                "sections_identity": _digest(self.sections.public()),
                "resources": [{"resource": binding.root, "role": role, "actions": sorted(actions)}
                              for binding, role, actions in self.resources],
                "alias_authorized": False, "capture_authorized": False, "definitions_authorized": False,
                "motion_authorized": False, "internal_cuts_authorized": False,
                "execution": "unchanged original operation after coherent access and physical alias registration"}


def prove_native_call_effects(analysis, procedure, selection):
    if type(selection) is not tuple or len(selection) != 1 or type(selection[0]).__name__ != "Call_Stmt":
        raise CompilationError("native CALL effects require one unchanged original CALL")
    reachability = prove_resource_reachability(analysis, procedure, selection, ())
    _require_complete(reachability)
    structures = native_call_storage_structures(analysis, reachability.native_graph)
    storage = prove_storage_mutation_effects(analysis, procedure, selection=selection,
        joined_storage=structures, native_graph=reachability.native_graph)
    return _issue(analysis, procedure, selection, reachability, storage)


def prove_native_group_effects(analysis, procedure, selection, *, descriptor_window=None, whole=False):
    """Reuse original host acquisition for one complete synchronous team."""
    from compiler.frontend.descriptor_teams import prove_descriptor_team
    from compiler.frontend.native_completion import prove_graph_joined_completion

    if type(selection) is not tuple or not selection:
        raise CompilationError("native host group requires its complete original source selection")
    reachability = prove_resource_reachability(analysis, procedure, selection, ())
    _require_complete(reachability)
    completion = prove_graph_joined_completion(analysis, procedure, selection, reachability.native_graph)
    team = prove_descriptor_team(analysis, procedure, selection)
    if completion.selected_node_ids != team.selected_node_ids or completion._ordinary_join is None:
        raise CompilationError("native host group requires the same complete original ordinary join")
    storage = prove_storage_mutation_effects(analysis, procedure, selection=selection,
        joined_storage=team, native_graph=reachability.native_graph)
    sections = None
    if not whole:
        sections = analysis.native_sections_for_nodes(procedure, selection, capture_locals=True,
            completion=completion._ordinary_join, descriptor_window=descriptor_window)
    return _issue(analysis, procedure, selection, reachability, storage, completion=completion, sections=sections)


def _identity(reachability, storage, resources, completion, sections):
    return _digest({"version": VERSION, "reachability": reachability.identity, "storage": storage.identity,
                    "completion": completion.identity if completion is not None else None,
                    "sections": sections.public(),
                    "resources": [(id(binding), role, sorted(actions)) for binding, role, actions in resources]})


def _issue(analysis, procedure, selection, reachability, storage, *, completion=None, sections=None):
    from compiler.frontend.native_sections import NativeSections

    prerequisites = _prerequisites(analysis, reachability, storage)
    resources = _project(analysis, procedure, reachability, storage, completion=completion)
    if sections is None:
        sections = NativeSections(False, "complete original host may-access; no overwrite authority")
    elif sections.available:
        managed = {binding.root for binding, role, actions in resources
                   if role == "managed" and actions & {"read", "write"}}
        sections = replace(sections, resources=tuple(replace(item, overwrites=()) for item in sections.resources
                                                     if item.resource in managed))
        if {item.resource for item in sections.resources} != managed:
            sections = NativeSections(False, "native host sections omit a managed may-access")
    identity = _identity(reachability, storage, resources, completion, sections)
    registry = getattr(analysis, _REGISTRY, None)
    if registry is None:
        registry = {}
        setattr(analysis, _REGISTRY, registry)
    if identity in registry:
        return registry[identity].validate(analysis, procedure, selection)
    if len(registry) >= source_registry_budget(analysis):
        raise CompilationError("native CALL effect registry budget exhausted")
    proof = NativeHostEffects(identity, procedure, resources, reachability, storage, analysis, selection,
                              prerequisites, _ISSUER, completion, sections)
    registry[identity] = proof
    try:
        return proof.validate(analysis, procedure, selection)
    except BaseException:
        if registry.get(identity) is proof:
            del registry[identity]
        raise
