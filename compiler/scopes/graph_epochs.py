"""Conditional allocation lifetime at an exact original graph position.

The descriptor issuer checks the selected source and direct call closure. The
graph owner retains the close frontier and physical CALL associations. Neither
proof supplies entry payload definitions or permits crossing the frontier.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, replace
from hashlib import sha256

from compiler.frontend.descriptor_windows import prove_descriptor_window
from compiler.ir import CompilationError

_ISSUER = object()


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _same_source(left, right):
    return (left.inputs is right.inputs and left.routines.keys() == right.routines.keys()
            and all(left.routines[name] is value for name, value in right.routines.items())
            and left.modules.keys() == right.modules.keys()
            and all(left.modules[name] is value for name, value in right.modules.items()))


def _binding_receipt(binding):
    """Small exact declaration receipt used only within one synchronous batch."""
    scope = binding.declaring_scope
    return (binding.root, binding.name, id(scope), id(scope.bindings.get(binding.name)),
            _encoded(binding.public()), tuple(id(node) for node in binding.shape_nodes),
            tuple(id(node) for node in binding.lower_bound_nodes))


def _covered(graph, selected):
    result, pending = set(), list(selected)
    while pending:
        identity = pending.pop()
        if identity in result:
            continue
        result.add(identity)
        node = graph.nodes[identity]
        pending.extend(node.children)
        for condition, body in node.alternatives:
            pending.extend((body,) if condition is None else (condition, body))
    return result


def _call_nodes(owner, selected):
    """Collect original IF descendants without selecting either path.

    Counted loops already discover their own immediate body sequence. This
    does not add a lifetime across a new loop, association or OpenMP group.
    """
    graph = owner.graph
    if any(graph.nodes[identity].kind not in {"call", "branch"} for identity in selected):
        raise CompilationError("graph allocation epoch requires complete original CALL or IF nodes")
    calls, visited, pending = [], set(), list(reversed(selected))
    while pending:
        identity = pending.pop()
        if identity in visited:
            raise CompilationError("graph allocation epoch requires distinct original structured descendants")
        visited.add(identity)
        if len(visited) > graph.node_limit:
            raise CompilationError("graph allocation epoch original descendant budget exhausted")
        node = graph.nodes[identity]
        if node.kind in {"loop", "associate"} or identity in graph.native_groups:
            raise CompilationError("graph allocation epoch nested loop, association or OpenMP needs separate authority")
        if node.kind == "call":
            calls.append(identity)
        children = (*node.children, *(child for condition, body in node.alternatives
                                      for child in (condition, body) if child is not None))
        pending.extend(reversed(children))
    return tuple(calls)


def _calls(owner, selected):
    result = []
    for identity in _call_nodes(owner, selected):
        original, = owner.graph.source_nodes(identity)
        resolved = owner.builder.execution_source_call(owner.routine, original)
        mappings = []
        for mapping in resolved.mappings:
            if not mapping.formal_binding.rank:
                continue
            if (mapping.binding is None or mapping.section is not None or mapping.presence != "supplied"
                    or mapping.binding.attributes & {"pointer", "optional", "volatile", "asynchronous", "target"}):
                raise CompilationError("graph allocation epoch requires whole nonescaping original actuals")
            mappings.append((mapping.formal_binding.root, owner.canonical(mapping.binding)))
        if len({binding.root for _formal, binding in mappings}) != len(mappings):
            raise CompilationError("graph allocation epoch CALL requires distinct original allocation slots")
        result.append((identity, resolved.procedure, tuple(mappings)))
    return tuple(result)


def _validate_call_bindings(epoch, plan, *, require_all=False):
    from compiler.scopes.segments import statement_span

    owner = epoch._owner
    members = {member.routine.qualified: member for member in plan.members}
    for identity, procedure, mappings in epoch._calls:
        child = members.get(procedure)
        if child is None:
            if require_all:
                raise CompilationError("owner-local group requires every original CALL to have a checked child")
            continue
        first, last = statement_span(owner.graph.source_nodes(identity)[0])
        sites = [site for site in child.call_sites if site["caller"] == owner.routine.qualified
                 and (site["first_line"], site["last_line"]) == (first, last)]
        expected = {formal: binding.root for formal, binding in mappings}
        if len(sites) != 1 or sites[0]["resource_mappings"] != expected:
            raise CompilationError("checked allocation epoch changed its mapped child association")
        for formal, binding in mappings:
            if formal in child.resources:
                local = child.resources[formal][1]
                if sites[0]["handle_bindings"].get(str(local)) != owner.resources[binding.root][1]:
                    raise CompilationError("checked allocation epoch changed its physical registration slots")


def _mapped_child_bindings(epoch, builder, analysis, procedure, closure):
    owner, parent_analysis = epoch._owner, epoch._owner.builder.analysis
    routine = analysis.routines.get(procedure)
    member = getattr(builder, "_native_host_owner", None)
    if (not _same_source(analysis, parent_analysis) or routine is None or builder.entry is not routine
            or procedure not in closure or parent_analysis.structure(procedure).identity != closure[procedure]
            or member is None or member.builder is not builder or member.routine is not routine
            or member.owner is not owner or member.parent is None
            or not hasattr(builder, "_native_host_array_actuals")):
        raise CompilationError("graph allocation epoch child needs its original mapped CALL authority")
    registered = owner.members.get(procedure)
    if registered is not member and (registered is not None or procedure not in owner.active):
        raise CompilationError("graph allocation epoch child lost its exact admitted companion")
    parent, original = member.parent, member.original_source_call
    if (original is None or parent.builder.analysis is not parent_analysis
            or parent.routine is not parent_analysis.routines.get(parent.routine.qualified)
            or parent.builder.entry is not parent.routine):
        raise CompilationError("graph allocation epoch child lost its original CALL association")
    graph, selected_call = parent_analysis._selected_source(parent.routine.qualified, (original,))
    if (not selected_call or len(selected_call) != 1
            or graph.nodes[selected_call[0]].kind != "call"
            or (parent is owner and not epoch.covers(parent_analysis, parent.routine.qualified, selected_call))
            or (parent is not owner and (parent.routine.qualified not in closure
                or graph.identity != closure[parent.routine.qualified]))):
        raise CompilationError("graph allocation epoch child CALL escapes its proved original closure")
    resolved = parent.builder.execution_source_call(parent.routine, original)
    if resolved.procedure != procedure:
        raise CompilationError("graph allocation epoch child changed its original CALL target")
    expected = {}
    for mapping in resolved.mappings:
        if not mapping.formal_binding.rank:
            continue
        if (mapping.binding is None or mapping.section is not None or mapping.presence != "supplied"
                or mapping.binding.attributes & {"pointer", "optional", "volatile", "asynchronous", "target"}):
            raise CompilationError("graph allocation epoch child needs whole original CALL actuals")
        expected[mapping.formal_binding.root] = parent.canonical(mapping.binding)
    actuals = builder._native_host_array_actuals
    if (set(actuals) != set(expected) or set(member.canonical_bindings) != set(expected)
            or any(actuals[root] is not binding or member.canonical_bindings[root] is not binding
                   for root, binding in expected.items())
            or len({id(binding) for binding in expected.values()}) != len(expected)):
        raise CompilationError("graph allocation epoch child changed its canonical original CALL actuals")
    return routine, expected


def mapped_child_descriptor(epoch, builder, analysis, procedure, selection, closure, *, readonly_allocatable=False):
    """Authenticate the original CALL, then prove this consumer's selection.

    Structured identities include summary policy. The retained parent's graph
    authenticates the original closure; an independent proof in the consuming
    analysis checks its own policy, budgets and exact original selected nodes.
    Neither the source association nor descriptor proof grants payload effects.
    """
    routine, expected = _mapped_child_bindings(epoch, builder, analysis, procedure, closure)
    for name in routine.arguments:
        binding = routine.scope.bindings[name]
        if binding.rank and "allocatable" in binding.attributes:
            if not readonly_allocatable:
                raise CompilationError("graph allocation epoch child allocatable formals need separate mapping authority")
            from compiler.frontend.readonly_allocatable_input import prove_readonly_allocatable_input
            prove_readonly_allocatable_input(analysis, routine, binding)
    _graph, selected = analysis._selected_source(procedure, selection)
    modules = []
    for root in epoch.module_roots:
        try:
            binding = builder.visible_binding(routine, root)
        except CompilationError:
            continue  # A parent's unused module guard need not be child-visible.
        if binding.root == root:
            modules.append(root)
    if not expected and not modules:
        # Scalar-only children carry no borrowed array descriptor obligation.
        return None
    proof = prove_descriptor_window(analysis, procedure, selected, tuple(expected), module_roots=tuple(modules))
    # Issuer callbacks and namespace inspection must not detach the consumer
    # or change its association between derivation and publication.
    _routine, current = _mapped_child_bindings(epoch, builder, analysis, procedure, closure)
    _current_graph, current_selected = analysis._selected_source(procedure, selection)
    if (tuple(current) != tuple(expected) or any(current[root] is not value for root, value in expected.items())
            or current_selected != selected):
        raise CompilationError("graph allocation epoch child changed during descriptor publication")
    return proof


@dataclass(frozen=True, slots=True)
class GraphStorageEpoch:
    identity: str
    selected: tuple
    roots: tuple
    module_roots: tuple
    cut_before: str
    _owner: object = field(repr=False, compare=False)
    _descriptor: object = field(repr=False, compare=False)
    _boundary: object = field(repr=False, compare=False)
    _calls: tuple = field(repr=False, compare=False)
    _bindings: tuple = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)

    def validate(self, builder=None):
        owner = self._owner
        builder = owner.builder if builder is None else builder
        if (self._issuer is not _ISSUER or getattr(owner, "position_epoch", None) is not self
                or builder is not owner.builder or builder.entry is not owner.routine
                or builder.config.scope_execution != "graph"):
            raise CompilationError("graph allocation epoch requires its exact issuing source owner")
        from compiler.frontend.descriptor_windows import DescriptorWindowProof
        from compiler.scopes.graph_regions import ControlLocalDescriptorRegions
        if type(self._descriptor) not in {DescriptorWindowProof, ControlLocalDescriptorRegions}:
            raise CompilationError("graph allocation epoch requires its original issued descriptor authority")
        record = self.public()
        record.pop("identity")
        if sha256(_encoded(record).encode()).hexdigest() != self.identity:
            raise CompilationError("graph allocation epoch changed its issued public identity")
        self._descriptor.validate(builder.analysis, owner.routine.qualified, self.selected)
        self._boundary.validate(owner, require_source=True)
        if (self._boundary.conditional_guard is not None
                or not any(patch is self._boundary for _first, _last, patch in owner.patches)):
            raise CompilationError("graph allocation epoch lost its mandatory unconditional retirement")
        graph, cut = builder.analysis._selected_source(owner.routine.qualified, self._boundary.originals)
        parent = graph.nodes[self._descriptor.public()["parent_sequence"]]
        start = parent.children.index(self.selected[0])
        if (graph is not owner.graph or tuple(parent.children[start:start+len(self.selected)]) != self.selected
                or start+len(self.selected) >= len(parent.children)
                or parent.children[start+len(self.selected)] != self.cut_before or not cut or cut[0] != self.cut_before):
            raise CompilationError("graph allocation epoch changed its exact original source frontier")
        call_selection = (self._descriptor.call_selection if type(self._descriptor) is ControlLocalDescriptorRegions
                          else self.selected)
        current = _calls(owner, call_selection)
        if (len(current) != len(self._calls)
                or any(a[:2] != b[:2] or len(a[2]) != len(b[2])
                       or any(x != y or p is not q for (x, p), (y, q) in zip(a[2], b[2], strict=True))
                       for a, b in zip(current, self._calls, strict=True))):
            raise CompilationError("graph allocation epoch changed its original CALL bindings")
        for binding in self._bindings:
            if builder.visible_binding(owner.routine, binding.root) is not binding:
                raise CompilationError("graph allocation epoch lost its original captured binding")
        return self

    def covers(self, analysis, procedure, selection):
        if procedure != self._owner.routine.qualified or not _same_source(analysis, self._owner.builder.analysis):
            return False
        graph, selected = analysis._selected_source(procedure, selection)
        from compiler.scopes.graph_regions import ControlLocalDescriptorRegions
        if type(self._descriptor) is ControlLocalDescriptorRegions:
            return self._descriptor.covers(selected)
        return bool(selected) and set(selected) <= _covered(graph, self.selected)

    def boundaries(self):
        from compiler.scopes.graph_regions import ControlLocalDescriptorRegions
        return ((*(patch for _identity, patch, _proof in self._descriptor._cuts), self._boundary)
                if type(self._descriptor) is ControlLocalDescriptorRegions else (self._boundary,))

    def captures(self, binding):
        return any(current is binding for current in self._bindings)

    def capture(self, builder, binding, fact):
        self.validate(builder)
        self._descriptor.resource_state(binding.root)
        return self._capture_result(binding, fact)

    def _capture_result(self, binding, fact):
        """Pure fact filtering after the caller authenticates this epoch."""
        if (not self.captures(binding) or fact.get("lifetime") != "selected_window"
                or fact.get("storage") != "stable" or fact.get("escapes") is not False
                or fact.get("allocation_changes") is not False):
            raise CompilationError("graph allocation epoch lacks independent selected-position storage facts")
        return {**{key: value for key, value in fact.items()
                   if key not in {"sections", "source_sections", "requires_registered_handle"}}, "initialized": "none"}

    def capture_many(self, builder, bindings, *, _consumer_check=None):
        """One synchronous, fully fenced capture batch; no validation cache.

        Validate the complete epoch on both sides. Between those checks only
        exact supplied fact objects and pure copies of their initial values
        may be filtered. Standalone capture/validate retain their full scans.
        """
        self.validate(builder)
        if (not isinstance(bindings, tuple) or len({id(binding) for binding in bindings}) != len(bindings)
                or len({binding.root for binding in bindings}) != len(bindings)):
            raise CompilationError("allocation epoch capture batch requires distinct original bindings")
        attachments = tuple((item.name, getattr(self, item.name)) for item in fields(self))
        analysis, routine, config = builder.analysis, builder.entry, builder.config
        original_graph = self._owner.graph
        facts, capture_facts = builder.facts, builder.facts["captures"]

        def encoded(value):
            try:
                return _encoded(value)
            except (TypeError, ValueError) as error:
                raise CompilationError("allocation epoch capture facts changed during batch") from error

        facts_record = encoded(facts)
        originals = tuple((binding, binding.root, _binding_receipt(binding),
                           capture_facts.get(binding.root), encoded(capture_facts.get(binding.root)))
                          for binding in bindings)
        descriptor = self._descriptor
        descriptor_attachments = tuple((item.name, getattr(descriptor, item.name)) for item in fields(descriptor))
        descriptor_registry = builder.analysis._descriptor_window_proofs
        from compiler.scopes.storage_epoch import descriptor_windows
        descriptor_components = tuple((proof, tuple((part.name, getattr(proof, part.name)) for part in fields(proof)))
                                      for proof in descriptor_windows(descriptor))
        entry = getattr(self, "_entry_preservation", None)
        entry_attachments = (tuple((item.name, getattr(entry, item.name)) for item in fields(entry))
                             if entry is not None else ())
        entry_registry = getattr(builder.analysis, "_entry_preservations", None)
        resources = {state["resource"] for state in descriptor.public()["resources"]}

        def checkpoint():
            if _consumer_check is not None:
                _consumer_check()
            if (any(getattr(self, name) is not value for name, value in attachments)
                    or self._owner.builder is not builder or self._owner.position_epoch is not self
                    or self._owner.routine is not routine or self._owner.graph is not original_graph
                    or builder.analysis is not analysis or builder.entry is not routine or builder.config is not config
                    or builder.facts is not facts or facts.get("captures") is not capture_facts
                    or builder.analysis._descriptor_window_proofs is not descriptor_registry
                    or any(descriptor_registry.get(proof.identity) is not proof
                           or any(getattr(proof, name) is not value for name, value in saved)
                           for proof, saved in descriptor_components)
                    or any(getattr(descriptor, name) is not value for name, value in descriptor_attachments)
                    or entry is not None and (getattr(builder.analysis, "_entry_preservations", None) is not entry_registry
                        or entry_registry.get(entry.identity) is not entry
                        or any(getattr(entry, name) is not value for name, value in entry_attachments))
                    or any(_binding_receipt(binding) != receipt for binding, _root, receipt, _fact, _record in originals)):
                raise CompilationError("allocation epoch capture attachment or registry changed during batch")

        results = []
        for binding, root, _receipt, supplied, record in originals:
            checked = builder.capture_fact(binding)
            checkpoint()
            if (checked is not supplied or capture_facts.get(root) is not supplied
                    or encoded(checked) != record or root not in resources):
                raise CompilationError("allocation epoch capture substituted an original supplied fact")
            # Never retain a mutable fact alias across another inspection hook.
            results.append(self._capture_result(binding, json.loads(record)))
        self.validate(builder)
        builder.analysis.verify_publication()
        checkpoint()
        if encoded(facts) != facts_record:
            raise CompilationError("allocation epoch capture facts changed at batch publication")
        return tuple(results)

    def descriptor_for(self, builder, analysis, procedure, selection):
        self.validate()
        if not _same_source(analysis, self._owner.builder.analysis):
            raise CompilationError("graph allocation epoch descriptor consumer changed original source objects")
        if procedure == self._owner.routine.qualified:
            if not self.covers(analysis, procedure, selection):
                return None
            _graph, selected = analysis._selected_source(procedure, selection)
            return prove_descriptor_window(analysis, procedure, selected, self.roots, module_roots=self.module_roots)
        return mapped_child_descriptor(self, builder, analysis, procedure, selection,
                                       self._descriptor.public()["source_call_closure"])

    def validate_plan(self, plan):
        self.validate()
        owner = self._owner
        if plan._owner is not owner or not any(patch is self._boundary for _a, _b, patch in plan.patches(owner)):
            raise CompilationError("checked allocation epoch lost its retained owner or retirement patch")
        if any(not any(current is patch for _a, _b, current in plan.patches(owner)) for patch in self.boundaries()):
            raise CompilationError("checked allocation epoch lost a control-local retirement patch")
        _validate_call_bindings(self, plan)
        return self

    def public(self):
        from compiler.scopes.graph_regions import ControlLocalDescriptorRegions
        partitioned = type(self._descriptor) is ControlLocalDescriptorRegions
        return {"identity": self.identity, "version": 2 if partitioned else 1,
                "procedure": self._owner.routine.qualified,
                "selected_node_ids": list(self.selected), "cut_before_node_id": self.cut_before,
                "descriptor_identity": self._descriptor.identity,
                "resources": list((*self.roots, *self.module_roots)),
                "calls": [{"node_id": node, "procedure": procedure,
                           "canonical_resources": {formal: binding.root for formal, binding in mappings}}
                          for node, procedure, mappings in self._calls],
                "allocation_presence": "lazy original guards at the first reached managed access",
                "repeated_activation": "same checked source position; NONE after retirement",
                "entry_definition_authority": False, "execution_authorized": False,
                "cross_frontier_retention_authorized": False,
                **({"control_local_regions": self._descriptor.public()} if partitioned else {})}


@dataclass(frozen=True, slots=True)
class CallGroupStorageEpoch:
    identity: str
    selected: tuple
    roots: tuple
    module_roots: tuple
    _owner: object = field(repr=False, compare=False)
    _descriptor: object = field(repr=False, compare=False)
    _exit: object = field(repr=False, compare=False)
    _calls: tuple = field(repr=False, compare=False)
    _bindings: tuple = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)
    _managed: tuple | None = field(default=None, repr=False, compare=False)
    _entry_preservation: object = field(default=None, repr=False, compare=False)

    # The SAME selected-storage capture and original child descriptor rules
    # apply. Only the exact mandatory cleanup representation is different.
    covers = GraphStorageEpoch.covers
    captures = GraphStorageEpoch.captures
    descriptor_for = GraphStorageEpoch.descriptor_for
    capture_many = GraphStorageEpoch.capture_many

    def capture(self, builder, binding, fact):
        self.validate(builder)
        self._descriptor.resource_state(binding.root)
        return self._capture_result(binding, fact)

    def _capture_result(self, binding, fact):
        if (not self.captures(binding) or fact.get("lifetime", "invocation") not in {"invocation", "selected_window"}
                or fact.get("storage") != "stable" or fact.get("escapes") is not False
                or fact.get("allocation_changes") is not False):
            raise CompilationError("owner-local call group lacks independent original allocation facts")
        preserved = (self._entry_preservation is not None and self._entry_preservation.permits(binding)
                     and fact.get("initialized") == "whole" and fact.get("lifetime", "invocation") == "invocation")
        return {**{key: value for key, value in fact.items()
                   if key not in {"sections", "source_sections", "requires_registered_handle"}},
                "lifetime": "selected_window", "initialized": "whole" if preserved else "none"}

    def validate(self, builder=None):
        owner = self._owner
        builder = owner.builder if builder is None else builder
        if (self._issuer is not _ISSUER or getattr(owner, "position_epoch", None) is not self
                or getattr(owner, "local_exit", None) is not self._exit or builder is not owner.builder
                or builder.entry is not owner.routine or builder.config.scope_execution != "graph"):
            raise CompilationError("owner-local call group requires its exact issuing source owner")
        selection = builder._native_ancestor_selection
        if (selection is None or selection.validate(builder.analysis).public().get("lifetime") != "owner_local_call_groups"
                or selection is not owner.native_ancestor_selection):
            raise CompilationError("owner-local call group lost its exact source selector")
        if self._managed is not None:
            from compiler.scopes.storage_epoch import managed_resources
            if self._managed != tuple(sorted(managed_resources(owner))):
                raise CompilationError("owner-local call group changed its admitted managed resources")
        elif getattr(owner, "_local_group_prepared", False):
            raise CompilationError("owner-local call group lost its final admitted storage inventory")
        record = self.public()
        record.pop("identity")
        if sha256(_encoded(record).encode()).hexdigest() != self.identity:
            raise CompilationError("owner-local call group changed its issued public identity")
        self._exit.validate(builder.analysis, owner.routine.qualified, owner.closed_selection)
        self._descriptor.validate(builder.analysis, owner.routine.qualified, self.selected)
        if (self._exit.public()["selected_node_ids"] != list(self.selected)
                or self._descriptor.public()["parent_sequence"] != self._exit.public()["parent_sequence"]):
            raise CompilationError("owner-local call group changed its original sequence exit")
        current = _calls(owner, self.selected)
        if (len(current) != len(self._calls)
                or any(a[:2] != b[:2] or len(a[2]) != len(b[2])
                       or any(x != y or p is not q for (x, p), (y, q) in zip(a[2], b[2], strict=True))
                       for a, b in zip(current, self._calls, strict=True))):
            raise CompilationError("owner-local call group changed its original CALL bindings")
        for binding in self._bindings:
            if builder.visible_binding(owner.routine, binding.root) is not binding:
                raise CompilationError("owner-local call group lost its independent captured binding")
        if self._entry_preservation is not None:
            from compiler.frontend.entry_preservation import PreservedEntryDefinitions
            if type(self._entry_preservation) is not PreservedEntryDefinitions:
                raise CompilationError("owner-local entry definitions require their exact registered source proof")
            self._entry_preservation.validate(builder.analysis, owner.routine.qualified, owner.closed_selection,
                                              self._bindings, builder.facts)
        return self

    def validate_plan(self, plan):
        self.validate()
        if plan._owner is not self._owner or self._managed is None:
            raise CompilationError("owner-local call group lost its checked execution owner")
        _validate_call_bindings(self, plan, require_all=True)
        return self

    def public(self):
        return {"identity": self.identity, "version": 3 if self._entry_preservation is not None else 2,
                "kind": "owner_local_original_call_group",
                "selection_identity": self._owner.builder._native_ancestor_selection.identity,
                "procedure": self._owner.routine.qualified, "selected_node_ids": list(self.selected),
                "sequence_exit": self._exit.public(), "descriptor_identity": self._descriptor.identity,
                "resources": list((*self.roots, *self.module_roots) if self._managed is None else self._managed),
                "descriptor_guard_resources": ([] if self._managed is None else
                    sorted(set((*self.roots, *self.module_roots))-set(self._managed))),
                "calls": [{"node_id": node, "procedure": procedure,
                           "source_span": list(self._owner.graph.nodes[node].span),
                           "canonical_resources": {formal: binding.root for formal, binding in mappings}}
                          for node, procedure, mappings in self._calls],
                "allocation_presence": "lazy original guards at the first reached managed access",
                "repeated_activation": ("fresh local context; only independently preserved invocation-entry definitions; NONE after retirement; close and clear handles at original sequence exit"
                    if self._entry_preservation is not None else
                    "fresh local context and NONE at every reached group; close and clear handles at original sequence exit"),
                **({"entry_preservation": self._entry_preservation.public()} if self._entry_preservation is not None else {}),
                "entry_definition_authority": False, "execution_authorized": False,
                "cross_frontier_retention_authorized": False}


def _used_fixed_module_captures(owner, descriptor, selected):
    """Discover candidates; the existing complete window issuer grants proof.

    Only original executable references in this selection and its already
    resolved complete call closure count. Unused broad facts and unrelated
    specifications never require a module descriptor or a registration.
    """
    from fparser.two.utils import Base

    from compiler.frontend.descriptor_budget import DEFAULT_SOURCE_SCAN_LIMIT, _syntax_inventory
    from compiler.frontend.source_effects import _children, _kind

    analysis = owner.builder.analysis
    candidates = {}
    for root in owner.builder.facts["captures"]:
        binding = owner.builder.visible_binding(owner.routine, root)
        module = analysis.modules.get(root.partition("::")[0])
        if (binding.rank and module is binding.declaring_scope and "allocatable" not in binding.attributes
                and len(binding.shape_nodes) == binding.rank
                and all(_kind(axis) == "Explicit_Shape_Spec" for axis in binding.shape_nodes)):
            candidates[id(binding)] = binding
    if not candidates:
        return ()
    members = [(owner.routine, tuple(node for identity in selected for node in owner.graph.source_nodes(identity)))]
    for procedure, identity in descriptor.public()["source_call_closure"].items():
        routine = analysis._require_original(procedure)
        if analysis.structure(procedure).identity != identity:
            raise CompilationError("owner-local fixed module discovery changed its original call closure")
        members.append((routine, (routine.execution,)))
    used = {}
    for routine, originals in members:
        syntax, _declarations = _syntax_inventory(originals)
        limit = max(DEFAULT_SOURCE_SCAN_LIMIT, 8*syntax)
        pending, seen, visits = list(originals), set(), 0
        while pending:
            node = pending.pop()
            if not isinstance(node, (Base, tuple, list)) or id(node) in seen:
                continue
            seen.add(id(node))
            visits += 1
            if visits > limit:
                raise CompilationError("owner-local fixed module original-source discovery budget exhausted")
            if _kind(node) == "Name":
                binding = analysis._binding(analysis.source_scope_for(node, routine.scope), str(node).lower())
                if candidates.get(id(binding)) is binding and binding is not None:
                    used[binding.root] = binding
            pending.extend(node if isinstance(node, (tuple, list)) else _children(node))
    return tuple(used.values())


def prepare_call_group_epoch(owner):
    from compiler.frontend.entry_preservation import EntryPreservationUnsupported, prove_preserved_entry_definitions
    from compiler.frontend.source_span_exit import prove_original_sequence_exit

    owner.local_exit = prove_original_sequence_exit(owner.builder.analysis, owner.routine.qualified,
                                                    owner.closed_selection)
    selected = tuple(owner.local_exit.public()["selected_node_ids"])
    calls = _calls(owner, selected)
    bindings = {binding.root: binding for _node, _procedure, mappings in calls for _formal, binding in mappings}
    # Optional broad capture guards must have an allocation/borrowed-descriptor
    # kind understood by the issuer. Exact CALL actuals above remain mandatory.
    # Unused fixed module metadata needs no allocation proof or registration;
    # any later managed field still must belong to this inventory at finalize.
    # Facts are never inferred from source INTENT or reads.
    for root in owner.builder.facts["captures"]:
        binding = owner.builder.visible_binding(owner.routine, root)
        if binding.rank and ("allocatable" in binding.attributes
                or binding.declaring_scope is owner.routine.scope and binding.name in owner.routine.arguments):
            bindings.setdefault(root, binding)
    roots, modules = [], []
    for binding in bindings.values():
        fact = owner.builder.capture_fact(binding)
        if fact.get("lifetime", "invocation") not in {"invocation", "selected_window"}:
            raise CompilationError("owner-local call group requires independently scoped allocation facts")
        (roots if binding.declaring_scope is owner.routine.scope else modules).append(binding.root)
    descriptor = prove_descriptor_window(owner.builder.analysis, owner.routine.qualified, selected,
                                         tuple(roots), module_roots=tuple(modules))
    fixed = _used_fixed_module_captures(owner, descriptor, selected)
    if fixed:
        for binding in fixed:
            # Independently supplied storage/entry facts remain mandatory.
            owner.builder.capture_fact(binding)
            bindings.setdefault(binding.root, binding)
            if binding.root not in modules:
                modules.append(binding.root)
        descriptor = prove_descriptor_window(owner.builder.analysis, owner.routine.qualified, selected,
                                             tuple(roots), module_roots=tuple(modules))
    try:
        entry_preservation = prove_preserved_entry_definitions(owner.builder.analysis, owner.routine.qualified,
            owner.closed_selection, tuple(bindings.values()), owner.builder.facts)
    except EntryPreservationUnsupported:
        entry_preservation = None
    epoch = CallGroupStorageEpoch("", selected, tuple(roots), tuple(modules), owner, descriptor,
                                 owner.local_exit, calls, tuple(bindings.values()), _ISSUER,
                                 _entry_preservation=entry_preservation)
    record = epoch.public()
    record.pop("identity")
    epoch = replace(epoch, identity=sha256(_encoded(record).encode()).hexdigest())
    owner.position_epoch = epoch
    return epoch.validate()


def initial_entry_snapshot(owner):
    """Only a typed local-span proof can seed the first reached registration."""
    epoch = getattr(owner, "position_epoch", None)
    if type(epoch) is CallGroupStorageEpoch and epoch._entry_preservation is not None:
        epoch.validate()
        return True
    return False


def finalize_call_group_epoch(owner):
    """Seal the SAME attachment only after child admission is complete.

    Private child builders borrow this object while being prepared. The issuer
    finalizes it before any execution plan exists; fresh validation then binds
    the exact managed inventory. No published execution authority is mutated.
    """
    from compiler.scopes.storage_epoch import managed_resources

    epoch = owner.position_epoch.validate()
    if owner.execution_plan is not None or epoch._managed is not None or getattr(owner, "_local_group_prepared", False):
        raise CompilationError("owner-local call group storage inventory was already sealed")
    managed = tuple(sorted(managed_resources(owner)))
    if not set(managed) <= {binding.root for binding in epoch._bindings}:
        raise CompilationError("owner-local call group has unproved managed storage outside its descriptor inventory")
    object.__setattr__(epoch, "_managed", managed)
    record = epoch.public()
    record.pop("identity")
    object.__setattr__(epoch, "identity", sha256(_encoded(record).encode()).hexdigest())
    owner._local_group_prepared = True
    return epoch.validate()


def discover_graph_storage_epoch(owner):
    """One contiguous original CALL/IF interval and its original cut.

    No alternate analysis or foreign owner executes. Failed candidate source
    proofs publish no capture capability and insert no retirement patch.
    """
    bindings, roots, modules = [], [], []
    for root, fact in owner.builder.facts["captures"].items():
        if fact.get("lifetime") != "selected_window":
            continue
        binding = owner.builder.visible_binding(owner.routine, root)
        if not binding.rank:
            continue
        owner.builder.capture_fact(binding)
        bindings.append(binding)
        # Local SAVE allocations need an original local declaration/privacy
        # proof, not module visibility. The descriptor issuer checks that
        # token before admitting the same selected CALL closure.
        (roots if binding.declaring_scope is owner.routine.scope else modules).append(root)
    if not bindings:
        return None
    analysis, graph = owner.builder.analysis, owner.graph
    from compiler.scopes.graph_epoch_candidates import candidates

    declines = []
    for selected, cut, priority in candidates(owner, (*roots, *modules)):
        spans = [graph.nodes[identity].span for identity in selected if graph.nodes[identity].span]
        attempt = {**priority, "selected_node_ids": list(selected), "cut_before": cut,
                   "span": ([min(span[0] for span in spans), max(span[1] for span in spans)] if spans else None),
                   "outcome": "proving"}
        owner.position_epoch_search["attempts"].append(attempt)
        try:
            from compiler.frontend.descriptor_windows import DescriptorWindowUnsupported
            from compiler.scopes.graph_regions import discover_control_local_epoch
            try:
                partitioned = discover_control_local_epoch(owner, selected, cut,
                                                           tuple(roots), tuple(modules), tuple(bindings))
            except DescriptorWindowUnsupported:
                # The bounded control-local route may decline a source shape
                # that the complete ordinary interval can prove.
                partitioned = None
            if partitioned is not None:
                attempt["outcome"] = "proved control-local epoch"
                return partitioned
            # A single IF can contain both producer and consumer. Site counts
            # grant no reachability or definition facts.
            if len(_call_nodes(owner, selected)) < 2:
                attempt["outcome"] = "fewer than two original calls"
                continue
            descriptor = prove_descriptor_window(analysis, owner.routine.qualified, selected,
                                                 tuple(roots), module_roots=tuple(modules))
            calls = _calls(owner, selected)
        except CompilationError as error:
            attempt.update(outcome="declined", reason=str(error))
            declines.append({"selected_node_ids": list(selected), "reason": str(error)})
            continue
        originals = graph.source_nodes(cut)
        owner.boundary(originals, "checked selected-position allocation epoch retirement", force_unconditional=True)
        boundary = owner.patches[-1][2]
        epoch = GraphStorageEpoch("", selected, tuple(roots), tuple(modules),
            cut, owner, descriptor, boundary, calls, tuple(bindings), _ISSUER)
        record = epoch.public()
        record.pop("identity")
        epoch = replace(epoch, identity=sha256(_encoded(record).encode()).hexdigest())
        owner.position_epoch = epoch
        epoch.validate()
        attempt["outcome"] = "proved complete epoch"
        return epoch
    owner.position_epoch_declines = declines
    return None
