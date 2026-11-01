"""An opt-in original structured path through the existing graph/coherence engine.

This module is staged independently of the public routing hook. Its source
gate is an activation veto, separate from returned-domain retention. Only the
four (or otherwise explicitly source-bound) barred CALLs may retire unknown
storage; an additional unproved edge rejects the entire checked path.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, replace
from hashlib import sha256

from compiler.frontend.source_effects import _children, _kind
from compiler.ir import CompilationError

_ISSUER = object()


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _span(node):
    item = getattr(node, "item", None)
    value = getattr(item, "fort_original_span", getattr(item, "span", None))
    if value is None:
        raise CompilationError("supported path requires an exact original edit anchor")
    return list(value)


def _clearing(routine, call):
    from compiler.frontend.supported_paths import _syntax
    for node, _path in _syntax((routine.execution,)):
        if hasattr(node, "content"):
            children = tuple(child for child in _children(node) if _kind(child) != "Comment")
            for index, child in enumerate(children):
                if child is call and index:
                    return children[index-1]
    raise CompilationError("supported path lost its original sticky clearing")


@dataclass(frozen=True, slots=True)
class SupportedPathStorageEpoch:
    identity: str
    selected: tuple
    roots: tuple
    module_roots: tuple
    _owner: object = field(repr=False, compare=False)
    _path: object = field(repr=False, compare=False)
    _gate: object = field(repr=False, compare=False)
    _descriptor: object = field(repr=False, compare=False)
    _calls: tuple = field(repr=False, compare=False)
    _bindings: tuple = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)
    _managed: tuple | None = field(default=None, repr=False, compare=False)
    _entry_preservation: object = field(default=None, repr=False, compare=False)
    _reached_position: object = field(default=None, repr=False, compare=False)

    def public(self):
        owner = self._owner
        gate = self._gate.public()
        flag = self._gate._flag
        original, = owner.graph.source_nodes(self.selected[-1])
        barred = {id(node): node for node in self._gate._barriers}
        exposures = {item["node_id"]: item for item in self._descriptor.public()["barred_calls"]}
        metadata = self._descriptor.public().get("gate_metadata_edges", [])
        calls = []
        barriers = []
        for node_id, procedure, mappings in self._calls:
            node, = owner.graph.source_nodes(node_id)
            is_barred = id(node) in barred
            calls.append({"node_id": node_id, "procedure": procedure, "source_span": _span(node),
                          "canonical_resources": {formal: binding.root for formal, binding in mappings},
                          "barred": is_barred})
            if is_barred:
                clearing = _clearing(owner.routine, node)
                barriers.append({"node_id": node_id, "procedure": procedure, "source_span": _span(node),
                    "original_source_sha256": sha256(str(node).encode()).hexdigest(),
                    "clear_source_span": _span(clearing), "clear_source_sha256": sha256(str(clearing).encode()).hexdigest(),
                    "exposure_resources": exposures[node_id]["exposure_resources"],
                    "exposure_descriptor_identity": exposures[node_id]["exposure_descriptor_identity"]})
        preserved = self._entry_preservation is not None
        return {"identity": self.identity, "version": 3 if preserved else 2 if metadata else 1, "kind": "supported_original_source_path",
                "selection_identity": owner.builder._native_ancestor_selection.identity,
                "procedure": owner.routine.qualified, "source_graph": owner.graph.identity,
                "selected_node_ids": list(self.selected), "path_identity": self._path.identity,
                "gate_identity": self._gate.identity, "descriptor_identity": self._descriptor.identity,
                "resources": list((*self.roots, *self.module_roots) if self._managed is None else self._managed),
                "descriptor_guard_resources": ([] if self._managed is None else
                    sorted(set((*self.roots, *self.module_roots))-set(self._managed))),
                "calls": calls,
                "retained_edges": self._descriptor.public()["components"],
                "outside_interval_edges": self._descriptor.public()["outside_interval_components"],
                **({"gate_metadata_edges": metadata} if metadata or preserved else {}),
                **({"reached_position": self._reached_position.public(),
                    "entry_preservation": self._entry_preservation.public()} if preserved else {}),
                "eligibility": {"descriptor": json.loads(_encoded(flag.public())), "owner_control": owner.eligibility_allowed,
                    "storage": "original private local SAVE LOGICAL", "initial": False,
                    "arm_source_span": gate["arm"]["original_span"],
                    "arm_source_sha256": gate["arm"]["syntax_sha256"], "barred_calls": barriers,
                    "controls": [{"procedure": member.routine.qualified, "argument": member.eligibility_allowed}
                                 for member in ((owner,) if self._managed is None else owner.members.values())],
                    "propagation": "same original owner SAVE gate forwarded INOUT through every checked ENTRY",
                    "future_activation": "false vetoes every managed registration, fresh context, child and device dispatch",
                    "rearm_after_excluded_edge": False},
                "sequence_exit": {"last_call_node_id": self.selected[-1],
                    "source_span": [_span(self._path._originals[0])[0], _span(original)[1]],
                    "after_line": _span(original)[1],
                    "cleanup": "publish and close immediately after final original CALL; clear handles"},
                "activation": ("original reached interval; source gate required before every activation; only independently preserved invocation-entry definitions; NONE after retirement"
                               if preserved else "original reached interval; source gate required before every activation; NONE"),
                "allocation_presence": "lazy original guards at first reached managed access",
                "ordering": "original source order; all checked graph tasks conservatively ordered",
                "additional_unproved_edge": "reject entire supported path",
                "entry_definition_authority": False, "payload_access_authorized": False,
                "alias_authority": False, "execution_authorized": False,
                "cross_frontier_retention_authorized": False}

    def validate(self, builder=None):
        attachments = tuple((part.name, getattr(self, part.name)) for part in fields(self))
        consumed, registered, registry_owners, binding_fields, projection_scopes = (), (), (), (), ()

        def require_attachments():
            if (any(getattr(self, name) is not value for name, value in attachments)
                    or any(any(getattr(proof, name) is not value for name, value in saved)
                           for proof, saved in consumed)
                    or any(getattr(builder.analysis, name, None) is not registry
                           or registry.get(identity) is not proof for name, registry, identity, proof in registered)
                    or any(getattr(builder.analysis, name, None) is not value for name, value in registry_owners)
                    or projection_scopes and not _same_source_scope_states(self._entry_preservation._projection, projection_scopes)
                    or any(any(getattr(binding, name) is not value for name, value in saved)
                           for binding, saved in binding_fields)):
                raise CompilationError("supported path epoch source attachment changed during validation")

        owner = self._owner
        builder = owner.builder if builder is None else builder
        if (self._issuer is not _ISSUER or builder is not owner.builder
                or getattr(owner, "position_epoch", None) is not self or owner.routine is not self._path._routine
                or owner.eligibility_allowed != self._gate._flag.name):
            raise CompilationError("supported path epoch lost its exact source owner or persistent gate alias")
        if (self._entry_preservation is None) != (self._reached_position is None):
            raise CompilationError("supported path epoch changed its original entry proof pair")
        if self._entry_preservation is not None:
            from compiler.frontend.entry_preservation import (
                PreservedEntryDefinitions,
                _native_attachments,
                _projection_attachments,
            )
            from compiler.frontend.private_entry_prefix import (
                PrivateEntryPrefixProjection,
                _same_source_scope_states,
                _source_scope_states,
            )
            from compiler.frontend.source_position import OriginalReachedPositionProof

            if (type(self._entry_preservation) is not PreservedEntryDefinitions
                    or type(self._reached_position) is not OriginalReachedPositionProof
                    or type(self._entry_preservation._projection) is not PrivateEntryPrefixProjection):
                raise CompilationError("supported path epoch requires its typed original entry proofs")
        retained = [(self._path, "_supported_path_selections"), (self._gate, "_allocation_gates"),
                    (self._descriptor, "_supported_path_storage")]
        if self._entry_preservation is not None:
            retained.extend(((self._entry_preservation, "_entry_preservations"),
                             (self._reached_position, "_original_reached_positions"),
                             (self._entry_preservation._projection, "_private_entry_prefixes")))
            retained.extend((proof, name) for proof, _attached_fields, name, _registry, _identity in (
                *_native_attachments(builder.analysis, self._entry_preservation._preservations),
                *_projection_attachments(builder.analysis, self._entry_preservation._projection)))
            projection_scopes = _source_scope_states(self._entry_preservation._projection)
        retained = tuple({id(proof): (proof, name) for proof, name in retained}.values())
        consumed = tuple((proof, tuple((part.name, getattr(proof, part.name)) for part in fields(proof)))
                         for proof, _registry in retained)
        registered = tuple((name, getattr(builder.analysis, name, None), proof.identity, proof)
                           for proof, name in retained)
        if any(not isinstance(registry, dict) or registry.get(identity) is not proof
               for _name, registry, identity, proof in registered):
            raise CompilationError("supported path epoch lost an exact registered source attachment")
        registry_owners = tuple((name, value) for name, value in builder.analysis.__dict__.items() if name.endswith("_owner"))
        retained_bindings = {id(binding): binding for binding in self._bindings}
        retained_bindings.update((id(binding), binding) for state in projection_scopes for _name, binding in state[4])
        binding_fields = tuple((binding, tuple((part.name, getattr(binding, part.name)) for part in fields(binding)))
                               for binding in retained_bindings.values())
        self._path.validate()
        require_attachments()
        self._gate.validate()
        require_attachments()
        self._descriptor.validate()
        require_attachments()
        private = {binding.root for binding in self._gate._resources}
        exposed = sorted(set((*self.roots, *self.module_roots))-private)
        if (any(binding.root not in {*self.roots, *self.module_roots} for binding in self._bindings)
                or any(item["exposure_resources"] != exposed
                       or (item["exposure_descriptor_identity"] is None) != (not exposed)
                       for item in self._descriptor.public()["barred_calls"])):
            raise CompilationError("supported path excluded edges lost the complete independent allocation footprint")
        selection = builder._native_ancestor_selection
        if (selection is None or selection is not owner.native_ancestor_selection
                or selection.validate(builder.analysis).public().get("lifetime") != "supported_original_path"):
            raise CompilationError("supported path epoch lost its exact explicit source selector")
        graph, selected = builder.analysis._selected_source(owner.routine.qualified, owner.closed_selection)
        if graph is not owner.graph or selected != self.selected:
            raise CompilationError("supported path epoch changed its complete original interval")
        require_attachments()
        record = self.public()
        require_attachments()
        record.pop("identity")
        if sha256(_encoded(record).encode()).hexdigest() != self.identity:
            raise CompilationError("supported path epoch changed its checked public identity")
        if self._managed is not None:
            from compiler.scopes.storage_epoch import managed_resources
            if self._managed != tuple(sorted(managed_resources(owner))):
                raise CompilationError("supported path epoch changed its complete admitted storage inventory")
        if any(builder.visible_binding(owner.routine, binding.root) is not binding for binding in self._bindings):
            raise CompilationError("supported path epoch changed an original allocation binding")
        if any(not hasattr(member, "eligibility_allowed") for member in owner.members.values()):
            raise CompilationError("supported path epoch omitted a checked child activation veto")
        require_attachments()
        if (builder.config.policy != "sections" or builder.config.scope_execution != "graph"
                or (self._entry_preservation is None) != (self._reached_position is None)):
            raise CompilationError("supported path epoch changed its explicit policy or original entry proof")
        if self._entry_preservation is not None:
            from compiler.frontend.entry_preservation import PreservedEntryDefinitions
            from compiler.frontend.private_entry_prefix import PrivateEntryPrefixProjection
            from compiler.frontend.source_position import OriginalReachedPositionProof

            entry, position = self._entry_preservation, self._reached_position
            if (type(entry) is not PreservedEntryDefinitions or type(position) is not OriginalReachedPositionProof
                    or entry._position is not position or type(entry._projection) is not PrivateEntryPrefixProjection
                    or entry._projection._gate is not self._gate
                    or any(binding.root in private for binding in entry.preserved)):
                raise CompilationError("supported path entry preservation lost its independent original private exclusion")
            position.validate(builder.analysis, owner.routine.qualified, self._path._originals)
            require_attachments()
            entry.validate(builder.analysis, owner.routine.qualified, self._path._originals, self._bindings, builder.facts)
        require_attachments()
        return self

    def covers(self, analysis, procedure, selection):
        from compiler.scopes.graph_epochs import _covered, _same_source
        if procedure != self._owner.routine.qualified or not _same_source(analysis, self._owner.builder.analysis):
            return False
        graph, selected = analysis._selected_source(procedure, selection)
        return bool(selected) and set(selected) <= _covered(graph, self.selected)

    def captures(self, binding):
        return any(binding is original for original in self._bindings)

    def capture(self, builder, binding, fact):
        self.validate(builder)
        if (not self.captures(binding) or fact.get("lifetime", "invocation") not in {"invocation", "selected_window"}
                or fact.get("storage") != "stable" or fact.get("escapes") is not False
                or fact.get("allocation_changes") is not False):
            raise CompilationError("supported path requires independent original storage/alias facts")
        if binding.root in (*self.roots, *self.module_roots):
            self._descriptor.resource_state(binding.root)
        return self._capture_result(binding, fact)

    def _capture_result(self, binding, fact):
        if (not self.captures(binding) or fact.get("lifetime", "invocation") not in {"invocation", "selected_window"}
                or fact.get("storage") != "stable" or fact.get("escapes") is not False
                or fact.get("allocation_changes") is not False):
            raise CompilationError("supported path requires independent original storage/alias facts")
        private = any(binding.root == original.root for original in self._gate._resources)
        preserved = (not private and self._entry_preservation is not None
                     and self._entry_preservation.permits(binding) and fact.get("initialized") == "whole"
                     and fact.get("lifetime", "invocation") == "invocation")
        return {**{key: value for key, value in fact.items()
                   if key not in {"sections", "source_sections", "requires_registered_handle"}},
                "lifetime": "selected_window", "initialized": "whole" if preserved else "none"}

    def capture_many(self, builder, bindings, *, _consumer_check=None):
        from compiler.scopes.supported_path_capture import capture_many

        return capture_many(self, builder, bindings, _consumer_check=_consumer_check)

    def descriptor_for(self, builder, analysis, procedure, selection):
        from compiler.scopes.graph_epochs import _same_source, mapped_child_descriptor
        self.validate()
        if not _same_source(analysis, self._owner.builder.analysis):
            raise CompilationError("supported path descriptor query changed its original source objects")
        if procedure == self._owner.routine.qualified:
            return self._descriptor.descriptor_for(selection, analysis=analysis) if self.covers(analysis, procedure, selection) else None
        closure = self._descriptor.public()["source_call_closure"]
        return mapped_child_descriptor(self, builder, analysis, procedure, selection, closure,
                                       readonly_allocatable=True)

    def validate_plan(self, plan):
        from compiler.scopes.graph_epochs import _validate_call_bindings
        self.validate()
        if plan._owner is not self._owner or self._managed is None:
            raise CompilationError("supported path epoch requires its final checked managed inventory")
        _validate_call_bindings(self, plan)
        return self


def finalize_supported_path_epoch(owner):
    from compiler.scopes.storage_epoch import managed_resources
    epoch = owner.position_epoch
    if epoch._managed is not None or owner.execution_plan is not None:
        raise CompilationError("supported path epoch cannot be retagged after execution planning")
    managed = tuple(sorted(managed_resources(owner)))
    if not set(managed) <= {binding.root for binding in epoch._bindings}:
        raise CompilationError("supported path admitted storage outside its independent capture inventory")
    staged = replace(epoch, identity="", _managed=managed)
    record = staged.public()
    record.pop("identity")
    identity = sha256(_encoded(record).encode()).hexdigest()
    if owner.position_epoch is not epoch or epoch._managed is not None or owner.execution_plan is not None:
        raise CompilationError("supported path epoch changed during private inventory finalization")
    # Child builders already borrow this exact private object. Finalization is
    # the one pre-publication issuer transition, never replacement/retagging of
    # a checked plan or a public token imported from JSON.
    object.__setattr__(epoch, "_managed", managed)
    object.__setattr__(epoch, "identity", identity)
    return epoch.validate()


def prepare_supported_path_epoch(owner, path, gate):
    from compiler.frontend.supported_path_storage import prove_supported_path_storage
    from compiler.frontend.supported_paths import _syntax
    from compiler.scopes.segments import statement_span

    builder = owner.builder
    path.validate()
    gate.validate()
    if (path._analysis is not builder.analysis or gate._analysis is not builder.analysis
            or path._routine is not owner.routine or gate._routine is not owner.routine):
        raise CompilationError("supported path epoch requires exact source-issued owner tokens")
    graph, selected = builder.analysis._selected_source(owner.routine.qualified, owner.closed_selection)
    if graph is not owner.graph or not selected or graph.nodes[selected[-1]].kind != "call":
        raise CompilationError("supported path epoch requires its unconditional original final CALL")
    calls = []
    bindings = {}
    for node, _location in _syntax(path._originals):
        if _kind(node) != "Call_Stmt":
            continue
        try:
            resolved = builder.execution_source_call(owner.routine, node)
        except CompilationError:
            # The descriptor composition below must independently admit this
            # exact retained intrinsic/native edge. Skipping a CALL association
            # never supplies its storage or scalar-clock authority.
            continue
        mappings = []
        for mapping in resolved.mappings:
            if not mapping.formal_binding.rank:
                continue
            if (mapping.binding is None or mapping.section is not None or mapping.presence != "supplied"
                    or mapping.binding.attributes & {"pointer", "optional", "volatile", "asynchronous", "target"}):
                raise CompilationError("supported path epoch needs independent whole original array associations")
            binding = owner.canonical(mapping.binding)
            mappings.append((mapping.formal_binding.root, binding))
            bindings[binding.root] = binding
        if len({binding.root for _formal, binding in mappings}) != len(mappings):
            raise CompilationError("supported path CALL has uncertain canonical array aliases")
        _source, identities = builder.analysis._selected_source(owner.routine.qualified, (node,))
        if len(identities) != 1:
            raise CompilationError("supported path CALL requires an exact original graph node")
        calls.append((identities[0], resolved.procedure, tuple(mappings)))
    for root in builder.facts["captures"]:
        binding = builder.visible_binding(owner.routine, root)
        if binding.rank:
            bindings.setdefault(root, binding)
    roots, modules = [], []
    for binding in bindings.values():
        fact = builder.capture_fact(binding)
        if fact.get("lifetime", "invocation") not in {"invocation", "selected_window"}:
            raise CompilationError("supported path requires independently scoped original capture facts")
        # The unchanged descriptor issuer decides whether this exact original
        # allocation or ordinary array descriptor is admissible. Discovery
        # never treats nonallocatable spelling as a lifetime/alias proof.
        (roots if binding.declaring_scope is owner.routine.scope else modules).append(binding.root)
    descriptor = prove_supported_path_storage(builder.analysis, path, gate, tuple(roots), module_roots=tuple(modules))
    # Both anchors are original statements. The selected path proof excludes
    # unstructured exits/concurrency and requires this final top-level CALL.
    if statement_span(path._originals[-1])[1] < statement_span(path._originals[0])[0]:
        raise CompilationError("supported path has no original unconditional cleanup insertion point")
    entry_preservation, position = None, None
    private = {binding.root for binding in gate._resources}
    if any(binding.root not in private and builder.capture_fact(binding).get("initialized") == "whole"
           and builder.capture_fact(binding).get("lifetime", "invocation") == "invocation"
           for binding in bindings.values()):
        from compiler.frontend.entry_preservation import (
            EntryPreservationUnsupported,
            _prefix_segments,
            prove_preserved_entry_definitions,
        )
        from compiler.frontend.private_entry_prefix import prove_private_entry_prefix
        from compiler.frontend.source_position import prove_original_reached_position

        try:
            position = prove_original_reached_position(builder.analysis, owner.routine.qualified, path._originals)
            _anchor, segments = _prefix_segments(builder.analysis, owner.routine.qualified, path._originals, position=position)
            projection = prove_private_entry_prefix(builder.analysis, owner.routine.qualified, segments, gate)
            entry_preservation = prove_preserved_entry_definitions(builder.analysis, owner.routine.qualified,
                path._originals, tuple(bindings.values()), builder.facts, position=position, prefix_projection=projection)
        except EntryPreservationUnsupported:
            entry_preservation = None
        if entry_preservation is None:
            position = None
    epoch = SupportedPathStorageEpoch("", selected, tuple(roots), tuple(modules), owner, path, gate,
        descriptor, tuple(calls), tuple(bindings.values()), _ISSUER,
        _entry_preservation=entry_preservation, _reached_position=position)
    record = epoch.public()
    record.pop("identity")
    object.__setattr__(epoch, "identity", sha256(_encoded(record).encode()).hexdigest())
    owner.position_epoch = epoch
    return epoch.validate()


def _source_order_graph(graph):
    """Add ordering edges without changing effects or source/numerical payloads."""
    from compiler.analysis.task_dependencies import build_task_graph
    from compiler.ir.task_graph import BranchRegion, LoopRegion

    tasks = []
    for task in graph.tasks:
        structure = task.structure
        if isinstance(structure, BranchRegion):
            structure = replace(structure, then_graph=_source_order_graph(structure.then_graph),
                                else_graph=_source_order_graph(structure.else_graph))
        elif isinstance(structure, LoopRegion):
            structure = replace(structure, body=_source_order_graph(structure.body))
        tasks.append(replace(task, effects=replace(task.effects, ordered=True),
                             header_effects=replace(task.header_effects, ordered=True), structure=structure))
    return build_task_graph(tuple(tasks), name=graph.name)


from compiler.scopes.closed_selection import ClosedIntervalOwner, OriginalSelectionOwner  # noqa: E402
from compiler.scopes.graph import GraphOwner  # noqa: E402


class OwnerLocalSupportedPath(GraphOwner, OriginalSelectionOwner):
    def __init__(self, builder, path, gate):
        ClosedIntervalOwner._validate_edit_boundaries(builder, path._originals)
        super().__init__(builder, path._originals)
        if builder.config.policy != "sections" or builder.config.scope_execution != "graph":
            raise CompilationError("supported source path first slice requires explicit unpriced sections graph")
        self.eligibility_allowed = gate._flag.name
        self.supported_path = path
        self.supported_gate = gate
        self.supported_path_mode = True
        self._barred_originals = tuple(gate._barriers)
        prepare_supported_path_epoch(self, path, gate)
        self.selection_context = {**self.selection_context,
            "activation": "original reached interval; persistent source eligibility required; NONE",
            "lifetime": "publish and close at original sequence exit or before exact excluded CALL",
            "fallback": "remaining original source once; false eligibility forbids fresh child/domain activation"}

    def scan(self, nodes, depth=0):
        if depth == 0 and (len(nodes) != len(self.closed_selection) or any(
                left is not right for left, right in zip(nodes, self.closed_selection, strict=True))):
            raise CompilationError("supported source path changed its exact original interval")
        return super().scan(nodes, depth)

    def _scan_group(self, group, depth):
        originals = self.builder.inline.original_selection(group if isinstance(group, tuple) else (group,))
        if len(originals) == 1 and any(originals[0] is node for node in self._barred_originals):
            self.boundary(originals, "exact source-bound sticky exclusion", force_unconditional=True)
            return
        return super()._scan_group(group, depth)

    def boundary(self, nodes, reason, *, force_unconditional=False):
        originals = self.builder.inline.original_selection(tuple(nodes))
        if len(originals) != 1 or not any(originals[0] is node for node in self._barred_originals):
            raise CompilationError("supported source path has an additional unproved edge: " + str(reason))
        self.supported_gate.validate()
        return super().boundary(nodes, reason, force_unconditional=True)

    def validate_plan(self, nodes):
        from compiler.ir.task_graph import ExecutionTarget
        from compiler.scheduling.task_graph import schedule_tasks, verify_schedule
        from compiler.scopes.graph import _member_graph

        self.supported_path.validate()
        self.supported_gate.validate()
        for member in self.members.values():
            if member is not self and member.boundaries:
                raise CompilationError("supported source path child has an additional unproved edge")
        if len(self.boundaries) != len(self._barred_originals):
            raise CompilationError("supported source path omitted an exact sticky retirement edge")
        graphs, schedules, unit_ids, spans = {}, {}, {}, {}
        for name, member in self.members.items():
            graph, identities, task_spans = _member_graph(member)
            graph = _source_order_graph(graph)
            required = {identities[str(index)]: ExecutionTarget.GPU if unit.leaves else ExecutionTarget.CPU
                        for index, unit in enumerate(member.units)}
            schedule = schedule_tasks(graph, required_targets=required, target_switch_cost=0.25)
            verify_schedule(graph, schedule)
            graphs[name], schedules[name], unit_ids[name], spans[name] = graph, schedule, identities, task_spans
        self.task_graphs, self.task_schedules, self.graph_unit_ids, self.graph_task_spans = graphs, schedules, unit_ids, spans
        from compiler.scopes.query_regions import discover_query_regions
        discover_query_regions(self)
        finalize_supported_path_epoch(self)

    def emit_control(self, nodes):
        from compiler.scopes.device_tasks import control
        from compiler.scopes.segments import fortran_lines, statement_span

        if self.execution_plan is None:
            raise CompilationError("supported source path requires its final checked execution plan")
        self.execution_plan.validate()
        self.position_epoch.validate()
        imports, provider_spec, provider_initialize = control(self)
        before = ["block", f"use iso_c_binding, only: {self.kind} => c_int64_t",
                  "use fort_scoped_memory, only: fort_scope_serial_caller", *imports,
                  f"integer({self.kind}) :: {self.context}", f"logical :: {self.enabled}",
                  f"logical :: {self.epoch_pristine}", f"logical :: {self.query_plan}",
                  *[f"integer({self.kind}) :: fort_buffer_{number}" for _binding, number in self.resources.values()],
                  *provider_spec, f"{self.context} = 0",
                  *[f"fort_buffer_{number} = 0" for _binding, number in self.resources.values()],
                  f"{self.enabled} = fort_scope_serial_caller() /= 0 .and. {self.eligibility_allowed}",
                  f"{self.epoch_pristine} = .false.", f"{self.query_plan} = .false.", *provider_initialize]
        first, last = statement_span(nodes[0])[0], statement_span(nodes[-1])[1]
        self.builder.add_edit(self.routine.scope.path, first, first-1,
                              "\n".join(fortran_lines(before))+"\n", prepend=True)
        after = [*self.close(disable=False, reason="supported original source sequence exit", span=(last, last)), "end block"]
        self.builder.add_edit(self.routine.scope.path, last+1, last, "\n".join(fortran_lines(after))+"\n")


def run_supported_path(builder):
    from compiler.frontend.supported_paths import requested_supported_path
    from compiler.scopes.generation_transaction import GenerationTransaction

    selection = builder._native_ancestor_selection.validate(builder.analysis)
    public = selection.public()
    if public["version"] != 4 or public["lifetime"] != "supported_original_path":
        raise CompilationError("supported source path requires its explicit versioned original selector")
    try:
        with GenerationTransaction(builder) as transaction:
            candidate = transaction.builder
            path, gate = requested_supported_path(candidate.analysis, candidate.entry, public["path_request"],
                tuple(item["procedure"] for item in public["device_roots"]))
            owner = OwnerLocalSupportedPath(candidate, path, gate)
            if owner.run(owner.closed_selection):
                record = candidate.scopes[-1]
                record["graph_root_kind"] = "supported original source path"
                record["supported_path_epoch"] = owner.position_epoch.public()
                record["estimate_available"] = False
                record["planning_reason"] = "explicit supported source path and provider are unpriced"
                transaction.adopt(candidate)
    except CompilationError as error:
        builder.boundaries.append({"procedure": builder.entry.qualified,
            "reason": "supported original source path unavailable: " + str(error),
            "execution": "unchanged original source path; no partial publication"})
    return builder.finish(materialize_native_effects=False)
