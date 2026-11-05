"""Typed native BYTE reservations; no numeric capture or definition authority."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from hashlib import sha256

from compiler.frontend.immutable_graph import original_type
from compiler.ir import CompilationError

_ISSUER = object()
_REGISTRY = "_native_opaque_read_permits"


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return sha256(_encoded(value).encode()).hexdigest()


def _readonly(summary, root):
    if not summary["complete"] or not summary["native_completion"]["available"]:
        raise CompilationError("native opaque READ requires complete original synchronous effects")
    if any(item.get("resource") == root and item["kind"] not in {"read", "descriptor_read"}
           for item in summary["operations"]):
        raise CompilationError("host-only native storage requires read-only complete effects: " + root)
    if root in summary["guaranteed_whole_overwrites"]:
        raise CompilationError("host-only native storage cannot acquire a managed write: " + root)


def _readonly_version(binding):
    return 2 if binding.root.startswith("argument::") and "allocatable" in binding.attributes else 1


@dataclass(frozen=True)
class NativeReadonlyPermit:
    identity: str
    binding: object
    _builder: object = field(repr=False, compare=False)
    _analysis: object = field(repr=False, compare=False)
    _routine: object = field(repr=False, compare=False)
    _selection: tuple = field(repr=False, compare=False)
    _graph: object = field(repr=False, compare=False)
    _consumer_graph: object = field(repr=False, compare=False)
    _selected: tuple = field(repr=False, compare=False)
    _actual: object = field(repr=False, compare=False)
    _actual_record: str | None = field(repr=False, compare=False)
    _summary: object = field(repr=False, compare=False)
    _record: str = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)
    _context: object = field(repr=False, compare=False)
    _arena: object = field(repr=False, compare=False)

    def _state(self, operation):
        """Only exact consumed attachments; this fence calls no serializers."""
        analysis = self._analysis
        binding = self.binding
        registry = getattr(analysis, _REGISTRY, None)
        arena = getattr(analysis, "_source_derivation_arena", None)
        from compiler.frontend.proof_arena import DerivationArena

        handle = (DerivationArena._find_handle(arena, (_REGISTRY, id(analysis), self.identity))
                  if type(arena) is DerivationArena else None)
        ancestry = (DerivationArena.accepts(arena, self._arena)
                    if type(arena) is DerivationArena and self._arena is not None else None)
        actual = self._actual
        member = getattr(self._builder, "_native_host_owner", None)
        owner = getattr(member, "owner", None)
        parent = getattr(member, "parent", None)
        association = (() if member is None else
                       (id(member.original_source_call), id(parent), id(parent.builder), id(parent.routine),
                        id(parent.builder.analysis), id(parent.builder.entry),
                        id(member.canonical_bindings),
                        tuple((root, id(value)) for root, value in member.canonical_bindings.items()),
                        id(parent.canonical_bindings),
                        tuple((root, id(value)) for root, value in parent.canonical_bindings.items())))
        ownership = (() if member is None else
                     (id(member), id(member.builder), id(member.routine), id(owner),
                      id(owner.builder), id(owner.routine), id(owner.builder.analysis), id(owner.builder.entry),
                      id(owner.members.get(member.routine.qualified)), tuple(owner.active), association))
        actual_state = (() if actual is None else
            (tuple((part.name, id(getattr(actual, part.name))) for part in fields(actual)),
             id(actual.declaring_scope.bindings.get(actual.name))))
        captures = self._builder.facts.get("captures", {})
        role = None
        if operation is not None:
            role = (tuple((part.name, id(getattr(operation, part.name))) for part in fields(operation)),
                    tuple((root, id(access), id(access.binding), access.mode, id(access.permit))
                          for root, access in operation.native_opaque_accesses.items()),
                    tuple((root, tuple(actions)) for root, actions in operation.effects.items()),
                    tuple(operation.overwrites), tuple(operation.bindings.items()),
                    self._summary_state(operation.summary))
        return (tuple((part.name, id(getattr(self, part.name))) for part in fields(self)),
                id(registry), id(dict.get(registry, self.identity)) if type(registry) is dict else None,
                id(getattr(analysis, _REGISTRY + "_owner", None)),
                id(getattr(analysis, "_source_authority_context", None)),
                id(getattr(analysis, "_source_session", None)), id(arena), id(handle), ancestry,
                id(self._builder.entry), id(self._builder.analysis),
                id(getattr(self._builder, "_native_host_array_actuals", {}).get(binding.root)),
                tuple((part.name, id(getattr(binding, part.name))) for part in fields(binding)),
                id(binding.declaring_scope.bindings.get(binding.name)),
                tuple(id(node) for node in self._selection), self._summary_state(self._summary), actual_state, ownership,
                id(captures.get(binding.root)), id(captures.get(actual.root)) if actual is not None else None, role)

    def _summary_state(self, summary):
        root = self.binding.root
        return (id(summary), summary.get("summary_identity"), summary.get("analysis_identity"),
                summary.get("complete"), summary.get("native_completion", {}).get("available"),
                tuple(summary.get("selected_node_ids", ())),
                tuple((item.get("resource"), item.get("kind")) for item in summary.get("operations", ())
                      if item.get("resource") == root), root in summary.get("guaranteed_whole_overwrites", ()))

    def _validate(self, operation=None, *, issued=True):
        from compiler.scopes.segments import _host_read_candidate

        analysis = self._analysis
        registry = getattr(analysis, _REGISTRY, None)
        if (type(self) is not NativeReadonlyPermit or self._issuer is not _ISSUER
                or type(registry) is not dict or getattr(analysis, _REGISTRY + "_owner", None) is not analysis
                or (registry.get(self.identity) is not self if issued else self.identity in registry)):
            raise CompilationError("native opaque READ requires its exact registered source permit")
        before = self._state(operation)
        if self._context is not None:
            self._context.require(analysis)
            arena = self._context.derivation_arena(analysis)
            if not arena.accepts(self._arena):
                raise CompilationError("native opaque READ belongs to a discarded generation")
            if issued:
                arena.require_handle((_REGISTRY, id(analysis), self.identity), self)
        if (self._builder.entry is not self._routine
                or self.binding.declaring_scope.bindings.get(self.binding.name) is not self.binding
                or getattr(self._builder, "_native_host_array_actuals", {}).get(self.binding.root) is not self._actual
                or not _host_read_candidate(self._builder, self.binding)):
            raise CompilationError("native opaque READ declaration or original actual association changed")
        graph, selected = analysis._selected_source(self._routine.qualified, self._selection)
        if graph is not self._graph or tuple(selected) != self._selected:
            raise CompilationError("native opaque READ changed its original operation selection")
        consumer, current = self._builder.analysis._selected_source(self._routine.qualified, self._selection)
        if consumer is not self._consumer_graph or tuple(current) != self._selected:
            raise CompilationError("native opaque READ changed its original consumer graph")
        _readonly(self._summary, self.binding.root)
        record = json.loads(self._record)
        if (sha256(self._record.encode()).hexdigest() != self.identity
                or record["version"] != _readonly_version(self.binding)
                or _encoded(self.binding.public()) != _encoded(record["binding"])
                or (None if self._actual is None else _encoded(self._actual.public())) != self._actual_record
                or self._summary["summary_identity"] != record["summary_identity"]
                or self._summary["analysis_identity"] != record["analysis_identity"]):
            raise CompilationError("native opaque READ source evidence changed")
        if operation is not None:
            access = operation.native_opaque_accesses.get(self.binding.root)
            if (original_type(access) is not NativeOpaqueAccess or access.permit is not self
                    or access.binding is not self.binding or access.mode != "R"
                    or operation.bindings.get(self.binding.root) is not self.binding
                    or operation.summary["summary_identity"] != record["summary_identity"]
                    or "write" in operation.effects.get(self.binding.root, ())
                    or self.binding.root in operation.overwrites):
                raise CompilationError("native opaque READ differs from the original operation role")
            _readonly(operation.summary, self.binding.root)
        analysis.inputs.verify_live()
        if self._state(operation) != before:
            raise CompilationError("native opaque READ attachments changed during source validation")
        return self

    def validate(self, operation=None):
        return self._validate(operation)

    def public(self):
        self.validate()
        return {"identity": self.identity, **json.loads(self._record)}


@dataclass(frozen=True)
class NativeOpaqueAccess:
    binding: object
    mode: str
    permit: object

    def validate(self, operation=None):
        from compiler.frontend.native_opaque_writes import NativeOnlyArrayWrite

        if (original_type(self) is not NativeOpaqueAccess
                or type(self.permit) not in {NativeReadonlyPermit, NativeOnlyArrayWrite}
                or self.permit.binding is not self.binding):
            raise CompilationError("native opaque access has no supported typed source permit")
        binding, mode, permit = self.binding, self.mode, self.permit
        if type(permit) is NativeReadonlyPermit and mode == "R":
            permit.validate(operation)
        elif type(permit) is NativeOnlyArrayWrite and mode == permit.mode and mode in {"W", "RW"}:
            _validate_write(self, operation)
        else:
            raise CompilationError("native opaque access has no supported typed source permit")
        if self.binding is not binding or self.mode is not mode or self.permit is not permit:
            raise CompilationError("native opaque access attachments changed during source validation")
        return self

    def public(self, operation=None):
        self.validate(operation)
        # The same operation-aware fence owns this serialization. Calling the
        # permit's public() again would discard that operation context.
        permit = {"identity": self.permit.identity, **json.loads(self.permit._record)}
        result = {"resource": self.binding.root, "mode": self.mode,
                  "permit_kind": "original_readonly", "permit": permit}
        if type(self.permit) is not NativeReadonlyPermit:
            result["permit_kind"] = "original_fixed_array_write"
            associations = {proof.identity: proof for _, _, completion in self.permit._reachability._completion
                            for team in completion._reduction_teams for proof in team._fixed_array_associations
                            if proof.identity in permit["association_identities"]}
            result["association_receipts"] = [{"identity": key, **json.loads(proof._record)}
                                               for key, proof in sorted(associations.items())]
            if operation is None:
                raise CompilationError("native opaque WRITE transport requires its exact original operation")
            result["execution_control"] = {
                "version": 1, "source_graph": operation.opaque_consumer_graph.identity,
                "selected_node_ids": operation.summary["selected_node_ids"],
                "original_guard": (str(self.permit._selection[0].items[0])
                                   if type(self.permit._selection[0]).__name__ == "If_Stmt" else None),
                "source_call": json.loads(operation.opaque_source_call)}
        return result


def write_effects(reachability):
    """Project original external payload effects; retain all R and W hazards."""
    effects, bindings = {}, {}
    for item in reachability.accesses:
        if not item.external:
            continue
        if item.storage.components:
            raise CompilationError("native opaque CALL leaves a component without physical mapping")
        binding = item.storage.binding
        bindings[binding.root] = binding
        if binding.rank and item.kind in {"read", "write"}:
            effects.setdefault(binding.root, set()).add(item.kind)
    return effects, bindings


def _write_state(permit, operation):
    from compiler.scopes.storage_admission import _Identity

    analysis = permit._analysis
    prerequisites = permit._prerequisites
    attachments, registries, _syntax, _nodes, namespaces = prerequisites
    state = (tuple((part.name, _Identity(getattr(permit, part.name))) for part in fields(permit)),
             tuple((_Identity(owner), name, _Identity(object.__getattribute__(owner, name))) for owner, name, _ in attachments),
             tuple((name, _Identity(getattr(analysis, name, None)), _Identity(getattr(analysis, name, {}).get(key)))
                   for name, _, key, _ in registries),
             tuple((_Identity(owner), name, key, _Identity(getattr(owner, name).get(key))) for owner, name, key, _ in namespaces),
             _Identity(getattr(analysis, "_native_opaque_write_proofs", None)),
             _Identity(getattr(analysis, "_native_opaque_write_proofs", {}).get(permit.identity)),
             _Identity(analysis._source_session), _Identity(analysis._source_authority_context),
             _Identity(analysis._source_derivation_arena))
    if operation is None:
        return state
    guard = operation.opaque_guard
    unit = operation.opaque_unit
    call = operation.opaque_call_mapping
    return (*state, () if unit is None else tuple((part.name, _Identity(getattr(unit, part.name))) for part in fields(unit)),
            () if unit is None else unit.scope._opaque_route_state(),
            () if call is None else tuple((part.name, _Identity(getattr(value, part.name)))
                for value in (call, *call.mappings) for part in fields(value)),
            tuple((part.name, _Identity(getattr(operation, part.name))) for part in fields(operation)),
            tuple((root, _Identity(access), _Identity(access.binding), access.mode, _Identity(access.permit))
                  for root, access in operation.native_opaque_accesses.items()),
            tuple((root, tuple(sorted(actions))) for root, actions in operation.effects.items()),
            tuple((root, _Identity(binding)) for root, binding in operation.bindings.items()),
            tuple((root, _Identity(binding)) for root, binding in operation.host_metadata.items()),
            tuple(operation.overwrites), tuple(map(_Identity, operation.nodes)), tuple(map(_Identity, operation.opaque_control)),
            _encoded(operation.summary), _Identity(guard),
            () if guard is None else ((_Identity(guard.items), tuple(map(_Identity, guard.items)))
                if type(guard).__name__ == "If_Stmt" else
                tuple((part.name, _Identity(getattr(guard, part.name))) for part in fields(guard))))


def access_state(access, operation):
    """Closed retained-proof snapshot dispatch for synchronous plan admission."""
    from compiler.frontend.native_opaque_writes import NativeOnlyArrayWrite

    if type(access.permit) is NativeReadonlyPermit:
        return NativeReadonlyPermit._state(access.permit, operation)
    if type(access.permit) is NativeOnlyArrayWrite:
        return _write_state(access.permit, operation)
    raise CompilationError("native opaque storage has no registered typed permit")


def _validate_write(access, operation):
    from compiler.frontend.call_bindings import ResolvedSourceCall, _original_call_control

    permit = access.permit
    before = _write_state(permit, operation)
    permit.validate(permit._analysis, permit._procedure, permit._selection, access.binding)
    if operation is not None:
        action = _original_call_control(permit._selection)
        record = json.loads(permit._record)
        effects, bindings = write_effects(permit._reachability)
        graph, selected = permit._analysis._selected_source(permit._procedure, (action,))
        builder = operation.opaque_builder
        consumer, consumer_selected = builder.analysis._selected_source(permit._procedure, (action,))
        guard = operation.opaque_guard
        guarded = type(permit._selection[0]).__name__ == "If_Stmt"
        if (operation.proof_analysis is not permit._analysis or operation.opaque_control != permit._selection
                or len(operation.nodes) != 1 or operation.nodes[0] is not action
                or (guarded and guard is not permit._selection[0]
                    and (guard is None or guard.nodes != (permit._selection[0].items[0],)))
                or (not guarded and guard is not None)
                or operation.native_opaque_accesses.get(access.binding.root) is not access
                or operation.effects != effects or operation.bindings != bindings or operation.overwrites
                or operation.host_metadata.get(access.binding.root) is not access.binding
                or operation.summary.get("summary_identity") != permit._reachability.identity
                or operation.summary.get("original_effect_inventory") != record["original_effect_inventory"]
                or operation.summary.get("selected_node_ids") != list(selected)
                or operation.summary.get("structured_identity") != graph.identity
                or consumer is not operation.opaque_consumer_graph or selected != consumer_selected
                or builder.analysis._source_session is not permit._analysis._source_session
                or operation.summary.get("guaranteed_whole_overwrites") != []):
            raise CompilationError("native opaque write differs from its exact original operation")
        unit = operation.opaque_unit
        if unit is not None and (not any(item is operation for item in unit.scope.native)
                    or (guarded and guard is permit._selection[0]
                        and (unit.scope.opaque_lexical_if is not guard
                             or unit.source_guard != str(guard.items[0])
                             or unit.native_source != str(action) or unit.publication_only))
                    or (not guarded and unit.source_guard is not None)):
            raise CompilationError("native opaque WRITE changed its reached original unit")
        public_call = json.loads(operation.opaque_source_call)
        associations = {proof.identity: proof for _, _, completion in permit._reachability._completion
                         for team in completion._reduction_teams for proof in team._fixed_array_associations
                         if proof.identity in record["association_identities"]}
        if not associations:
            raise CompilationError("native opaque WRITE lost its original array association")
        originals = [json.loads(proof._record)["source_call"] for proof in associations.values()]
        original = originals[0]
        if any(item != original for item in originals):
            raise CompilationError("native opaque WRITE mixes different original CALLs")
        if (public_call.get("caller") != permit._procedure or public_call.get("source") != record["source"]
                or [public_call.get("first_line"), public_call.get("last_line")] != [record["first_line"], record["last_line"]]
                or any(public_call.get(key) != original[key] for key in
                       ("procedure", "target", "source_kind", "formal_arguments", "actual_arguments", "original_arguments"))):
            raise CompilationError("native opaque write lost its original CALL transport")
        mapped = operation.opaque_call_mapping
        if (original_type(mapped) is not ResolvedSourceCall
                or _encoded(mapped.public()) != _encoded({key: value for key, value in public_call.items()
                    if key not in {"caller", "source", "first_line", "last_line"}})):
            raise CompilationError("native opaque write changed its complete CALL mapping")
        resolved = next(iter(associations.values()))._resolved
        if len(mapped.mappings) != len(resolved.associations):
            raise CompilationError("native opaque write lost an original managed argument")
        for mapping, argument in zip(mapped.mappings, resolved.associations, strict=True):
            actual_binding = argument.actual_type.binding if argument.actual_type is not None else None
            if (mapping.formal != argument.formal or mapping.actual is not argument.actual
                    or mapping.formal_binding is not argument.formal_binding or mapping.binding is not actual_binding
                    or mapping.section is not None or mapping.scalar_element is not None or mapping.source_object is not None
                    or mapping.presence != ("omitted" if argument.actual is None else "supplied")
                    or mapping.resource != (actual_binding.root if actual_binding is not None else None)):
                raise CompilationError("native opaque write changed an original managed argument mapping")
    if before != _write_state(permit, operation):
        raise CompilationError("native opaque WRITE attachments changed during source validation")


def readonly_accesses(builder, analysis, nodes, summary, bindings):
    """Wrap the existing complete READ checks; never derive WRITE permission."""
    from compiler.scopes.segments import _host_read_candidate

    graph, selected = analysis._selected_source(builder.entry.qualified, nodes)
    consumer_graph, consumer_selected = builder.analysis._selected_source(builder.entry.qualified, nodes)
    if tuple(selected) != tuple(consumer_selected):
        raise CompilationError("native opaque READ differs from its original consumer selection")
    registry = getattr(analysis, _REGISTRY, None)
    if registry is None:
        registry = {}
        setattr(analysis, _REGISTRY, registry)
        setattr(analysis, _REGISTRY + "_owner", analysis)
    if type(registry) is not dict or getattr(analysis, _REGISTRY + "_owner", None) is not analysis:
        raise CompilationError("native opaque READ issuer belongs to another analysis")
    result = {}
    for root, binding in sorted(bindings.items()):
        if root != binding.root or not _host_read_candidate(builder, binding):
            raise CompilationError("native opaque READ requires its original readonly storage candidate")
        _readonly(summary, root)
        record = {"version": _readonly_version(binding), "authority": "complete original reached native operation",
                  "procedure": builder.entry.qualified, "source_graph": consumer_graph.identity,
                  "selected_node_ids": list(selected), "binding": binding.public(),
                  "analysis_identity": summary["analysis_identity"], "summary_identity": summary["summary_identity"],
                  "structured_identity": summary.get("structured_identity"),
                  "demand_identity": summary.get("demand_identity"),
                  "allocation_changes": False, "calls_or_escapes": False,
                  "device_capture": False, "managed_definitions": False,
                  "lifetime": "original stable native operation; allocation changes close ownership",
                  "range_validation": "allocated guard then undefined BYTE reservation; managed aliases close before native work"}
        identity = _digest(record)
        context = analysis.source_authority_context() if analysis._source_session is not None else None
        arena = context.derivation_arena(analysis) if context is not None else None
        permit = registry.get(identity)
        actual = getattr(builder, "_native_host_array_actuals", {}).get(root)
        if permit is None:
            permit = NativeReadonlyPermit(identity, binding, builder, analysis, builder.entry, tuple(nodes),
                graph, consumer_graph, tuple(selected), actual,
                None if actual is None else _encoded(actual.public()), summary, _encoded(record), _ISSUER, context, arena)
            permit._validate(issued=False)
            # No callback-bearing validation after this publication fence.
            if getattr(analysis, _REGISTRY, None) is not registry or identity in registry:
                raise CompilationError("native opaque READ provisional registration changed")
            registry[identity] = permit
            try:
                if arena is not None:
                    arena.register_handle((_REGISTRY, id(analysis), identity), permit)
            except BaseException:
                if registry.get(identity) is permit:
                    del registry[identity]
                key = (_REGISTRY, id(analysis), identity)
                if arena is not None and arena._handles.get(key) is permit:
                    del arena._handles[key]
                raise
        else:
            if (type(permit) is not NativeReadonlyPermit or permit._builder is not builder
                    or permit.binding is not binding or permit._actual is not actual
                    or permit._routine is not builder.entry or permit._analysis is not analysis
                    or permit._graph is not graph or permit._consumer_graph is not consumer_graph
                    or permit._selected != tuple(selected)
                    or len(permit._selection) != len(nodes)
                    or any(a is not b for a, b in zip(permit._selection, nodes, strict=True))):
                raise CompilationError("native opaque READ reuse changed its original consumer or actual")
            permit.validate()
        result[root] = NativeOpaqueAccess(binding, "R", permit)
    return result


def public_accesses(operation):
    if not operation.native_opaque_accesses:
        return None
    record = {"version": 2, "accesses": [access.public(operation)
              for _root, access in sorted(operation.native_opaque_accesses.items())],
              "device_capture": False, "initialization_authorized": False,
              "managed_definitions": False, "alias_authorized": False}
    return {"identity": _digest(record), **record}


def epoch_inventory(owner, managed):
    """Final whole-owner role union, including each original borrowed actual.

    This transports existing source permits and original call associations. It
    does not create data, layout, alias, or returned-lifetime authority.
    """
    from compiler.scopes.source_task_bindings import _field_roots, checked_call_inventory

    rows, opaque = [], set()
    for member in owner.members.values():
        for ordinal, unit in enumerate(member.units):
            if any(type(access.permit) is not NativeReadonlyPermit for operation in unit.scope.native
                   for access in operation.native_opaque_accesses.values()):
                unit.scope.validate_opaque_control(unit)
            for operation_id, operation in enumerate(unit.scope.native):
                access_record = public_accesses(operation)
                if access_record is None:
                    continue
                nodes, edges, _changed = checked_call_inventory(member)
                field_slots = {root: member.resources[root][1] for root in sorted(operation.native_opaque_accesses)}
                field_roots = _field_roots(nodes, edges.values(), member.routine.qualified, field_slots)
                projection = {"members": [nodes[name] for name in sorted(nodes)],
                    "calls": [edges[key] for key in sorted(edges)], "field_slots": field_slots,
                    "field_roots": field_roots}
                resources = []
                for root, access in sorted(operation.native_opaque_accesses.items()):
                    canonical = field_roots[root]
                    opaque.update(canonical)
                    resources.append({"resource": root, "canonical_resources": sorted(canonical), "mode": access.mode,
                        "permit_identity": access.permit.identity})
                rows.append({"procedure": member.routine.qualified, "unit_ordinal": ordinal,
                    "operation_id": operation_id, "access_identity": access_record["identity"], "resources": resources,
                    "call_projection": projection})
    # A template's canonical representative may describe only its first call.
    # Every reserved original native actual must occur in the complete union.
    native_actuals = {member.canonical(mapping.binding).root for member in owner.members.values()
        for arrays in getattr(member, "original_native_actual_registrations", {}).values() for mapping in arrays}
    if not native_actuals <= opaque:
        raise CompilationError("native opaque inventory lost an original native actual")
    scalar_backing = {member.canonical(element.binding).root for member in owner.members.values()
        for elements in getattr(member, "original_scalar_element_reservations", {}).values() for element in elements}
    scalar_backing.update(getattr(owner, "scalar_element_resources", {}))
    if opaque & (set(managed) | scalar_backing):
        raise CompilationError("native opaque storage overlaps a canonical numeric or scalar backing role")
    if not opaque <= owner.resources.keys():
        raise CompilationError("native opaque inventory lost its canonical reservation")
    return rows
