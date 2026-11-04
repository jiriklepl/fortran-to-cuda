"""One synchronous inspection of a final checked storage inventory.

The explicit emission scope can answer only membership and
canonical storage names already admitted by an independently issued plan. It
does not return entry facts, geometry, code, or an execution permission.
"""
from __future__ import annotations

import json
from contextlib import nullcontext
from contextvars import ContextVar
from dataclasses import fields, is_dataclass
from types import MappingProxyType

from compiler.frontend.immutable_graph import FrozenDict, FrozenList
from compiler.ir import CompilationError
from compiler.scopes.graph_epochs import _encoded

_ACTIVE = ContextVar("checked_storage_admission", default=None)


class _Identity(tuple):
    """Retain an exact object for this scope; numeric IDs can be recycled."""
    __slots__ = ()

    def __new__(cls, value):
        return tuple.__new__(cls, (value,))

    @property
    def value(self):
        return self[0]

    def __eq__(self, other):
        return type(other) is _Identity and self.value is other.value

    def __ne__(self, other):
        return not self == other


def _attachments(value):
    return tuple((part.name, _Identity(getattr(value, part.name))) for part in fields(value))


def _mapping(value):
    return _Identity(value), tuple((key, _Identity(item)) for key, item in value.items())


def _retained_state(value):
    if is_dataclass(value):
        return _attachments(value)
    if isinstance(value, (dict, FrozenDict, MappingProxyType)):
        return _mapping(value)
    return tuple(_Identity(item) for item in value)


def _binding(binding):
    scope = binding.declaring_scope
    return (_Identity(binding), binding.root, binding.name, _Identity(scope),
            _Identity(scope.bindings.get(binding.name)), _encoded(binding.public()),
            tuple(_Identity(node) for node in binding.shape_nodes),
            tuple(_Identity(node) for node in binding.lower_bound_nodes))


def _optional_attachments(value):
    if value is None:
        return ()
    if not is_dataclass(value):
        raise CompilationError("storage admission lost a retained proof attachment")
    return _attachments(value)


def _scope_storage_roles(scope):
    from compiler.scopes.native_opaque_accesses import access_state

    native = tuple((_Identity(operation), _attachments(operation),
                    _mapping(operation.native_opaque_accesses), tuple((_attachments(access), access_state(access, operation))
                          for access in operation.native_opaque_accesses.values()), _mapping(operation.host_metadata))
                   for operation in scope.native)
    return (_Identity(scope.native), native, _Identity(scope.nodes),
            tuple(_Identity(node) for node in scope.nodes), _Identity(scope.tree),
            tuple(_Identity(item) for item in scope.tree),
            _mapping(scope.guarded), _mapping(scope.close_guards))


def _unit_storage_roles(unit):
    return (_Identity(unit.host_metadata),
            unit.source_guard, unit.native_source, unit.publication_only,
            None if unit.host_metadata is None else _mapping(unit.host_metadata),
            _scope_storage_roles(unit.scope))


def _storage_topology(member):
    """Owned storage/control edges; emission results deliberately stay separate."""
    maps = ("original_actual_registrations", "original_native_actual_registrations",
            "original_scalar_element_reservations", "scalar_element_resources",
            "resource_slot_roots", "parameterized_native_bindings", "_issued_lexical_calls")
    sequences = ("units", "patches", "internal_calls", "call_sites", "counted_controls",
                 "return_calls", "return_conditions", "closed_suffixes", "source_lexical_calls",
                 "source_companion_calls", "graph_native_tasks")
    controls = ("context", "enabled", "epoch_pristine", "query_plan", "control_guard",
                "retention_allowed", "eligibility_allowed", "resource_slot_namespace")
    return (tuple((name, None if getattr(member, name, None) is None else
                    _mapping(getattr(member, name))) for name in maps),
            tuple((name, _Identity(getattr(member, name, None)),
                   tuple(_Identity(item) for item in getattr(member, name, ()))) for name in sequences),
            tuple((name, _Identity(getattr(member, name, None))) for name in controls),
            _encoded([receipt.record for receipt in member.source_lexical_calls]),
            _encoded([getattr(member, name, ()) for name in ("internal_calls", "call_sites", "counted_controls")]),
            tuple((_Identity(nodes), tuple(_Identity(node) for node in nodes),
                   _Identity(scope), _scope_storage_roles(scope)) for nodes, scope in member.graph_native_tasks))


def _returned_topology(owner):
    """Retain selected-return ownership without reissuing its source proofs."""
    draft = getattr(owner.builder, "_conditional_return_draft", None)
    selection = getattr(owner, "hierarchical_selection", None)
    returned = getattr(owner, "return_contract", None)
    sequences = ("closed_selection", "closed_prefix", "closed_controls", "closed_initializations")
    context = getattr(owner, "selection_context", None)
    return (_Identity(returned), _optional_attachments(returned),
            _Identity(selection), _optional_attachments(selection),
            _Identity(draft), None if draft is None else (
                tuple((name, _Identity(getattr(draft, name))) for name in
                      ("owner", "_issuer", "selected_root", "_selected_root_receipt")),
                _Identity(draft._consumers),
                tuple((_Identity(row), tuple(_Identity(item) for item in row)) for row in draft._consumers),
                _mapping(draft._native_preservations)),
            tuple((name, _Identity(getattr(owner, name, None)),
                   tuple(_Identity(item) for item in getattr(owner, name, ()))) for name in sequences),
            _Identity(getattr(owner, "selection_bundle", None)),
            _Identity(context), None if context is None else _encoded(context),
            _Identity(owner.emission.selection_prefix))


class StorageAdmission:
    """A detached record bounded by fresh full checks, never a reusable cache."""

    def __init__(self, plan):
        self._plan = plan
        self._proof = plan.storage_epoch()
        self._owner = plan._owner
        self._used = False
        self._live = False
        self._token = None

    def _state(self):
        from compiler.scopes.execution_plan import _configuration

        plan, owner = self._plan, self._owner
        members = []
        bindings = {}
        for member in plan.members:
            units = []
            for unit in plan.units(member):
                for binding in unit.arrays.values():
                    bindings[id(binding)] = binding
                    actual = member.canonical_bindings.get(binding.root, binding)
                    bindings[id(actual)] = actual
                units.append((_Identity(unit), _Identity(unit.scope), _mapping(unit.arrays),
                              tuple(_Identity(node) for node in unit.nodes), _unit_storage_roles(unit)))
            for binding, _number in member.resources.values():
                bindings[id(binding)] = binding
            for binding in member.canonical_bindings.values():
                bindings[id(binding)] = binding
            members.append((_Identity(member), _Identity(member.builder), _Identity(member.builder.analysis),
                _Identity(member.routine), _Identity(member.graph), _Identity(member.owner), _Identity(member.parent),
                _mapping(member.resources), _mapping(member.canonical_bindings),
                tuple(units), tuple(_Identity(task) for task in member.device_tasks),
                tuple(_Identity(task) for task in member.source_device_tasks),
                tuple(_Identity(transfer) for transfer in member.storage_transfers),
                _encoded(member.boundaries), _storage_topology(member)))
        facts, policies, seen = [], [], set()
        for builder, analysis, _routine, _saved_configuration, _facts in plan._contexts:
            session = analysis._source_session
            if session is None:
                raise CompilationError("storage admission requires a sealed original source session")
            session.verify(analysis)
            captures = builder.facts.get("captures", {})
            facts.append((_Identity(builder), _Identity(builder.analysis), _Identity(builder.entry), _Identity(builder.config),
                          _Identity(builder.options), _Identity(builder.facts), _Identity(captures),
                          tuple((root, _Identity(fact)) for root, fact in captures.items()),
                          _encoded(builder.facts), _configuration(builder)))
            if id(analysis) not in seen:
                seen.add(id(analysis))
                policies.append((_Identity(analysis), _Identity(session), _attachments(session),
                                 _encoded(analysis._summary_authority())))
        epoch = getattr(owner, "position_epoch", None)
        return (_attachments(plan), _attachments(self._proof), _Identity(owner.execution_plan),
                _mapping(owner.members), _mapping(owner.resources), _Identity(epoch),
                _optional_attachments(epoch), tuple(members), tuple(facts), tuple(policies),
                tuple(_binding(binding) for binding in bindings.values()),
                _returned_topology(owner))

    def _authorities(self):
        """Snapshot retained proof attachments without calling their validators."""
        from fparser.two.utils import Base

        from compiler.frontend.source_effects import SourceEffects
        from compiler.frontend.source_facts import SourceFact
        from compiler.frontend.source_session import SourceSession
        from compiler.scopes.lexical import LexicalOwner, OwnerEmission
        from compiler.scopes.source import ScopeBuilder

        sessions = {id(analysis._source_session): analysis._source_session
                    for _builder, analysis, _routine, _configuration, _facts in self._plan._contexts}
        pending = [getattr(self._owner, "position_epoch", None),
                   getattr(self._owner, "return_contract", None),
                   getattr(self._owner, "hierarchical_selection", None),
                   self._owner.emission.selection_prefix,
                   *self._plan._procedures, *self._plan._numerical_authorities,
                   *self._plan._proof_handoffs, *self._plan._selection_prefixes]
        draft = getattr(self._owner.builder, "_conditional_return_draft", None)
        if draft is not None:
            pending.extend(draft._native_preservations.values())
        pending.extend(member.graph for member in self._plan.members)
        pending.extend(receipt for member in self._plan.members for name in
                       ("source_lexical_calls", "source_companion_calls", "return_calls", "return_conditions")
                       for receipt in getattr(member, name, ()))
        scopes = [unit.scope for member in self._plan.members for unit in self._plan.units(member)]
        scopes.extend(scope for member in self._plan.members for _nodes, scope in member.graph_native_tasks)
        pending.extend(getattr(scope, name) for scope in scopes for name in
                       ("native", "calls", "tree", "guarded", "close_guards"))
        pending.extend(access for member in self._plan.members for unit in self._plan.units(member)
                       for operation in unit.scope.native for access in operation.native_opaque_accesses.values())
        pending.extend(task for member in self._plan.members for task in (*member.device_tasks, *member.source_device_tasks))
        seen, retained = set(), []
        while pending:
            value = pending.pop()
            if id(value) in seen:
                continue
            seen.add(id(value))
            if (value is None or isinstance(value, (Base, SourceEffects, ScopeBuilder, LexicalOwner, OwnerEmission))
                    or any(session.contains(value) for session in sessions.values())):
                continue
            if is_dataclass(value):
                analysis = getattr(value, "_analysis", None)
                registry = []
                if isinstance(analysis, SourceEffects):
                    for name, container in vars(analysis).items():
                        if isinstance(container, dict):
                            for key in (getattr(value, "identity", None), id(value)):
                                if container.get(key) is value:
                                    registry.append((analysis, name, container, key))
                retained.append((value, _attachments(value), tuple(registry)))
                # Source services and arena dependencies retain their own
                # ownership checks; their indexes are not execution payload.
                if type(value) not in {SourceFact, SourceSession}:
                    pending.extend(getattr(value, part.name) for part in fields(value))
            elif isinstance(value, (dict, FrozenDict, MappingProxyType)):
                retained.append((value, _retained_state(value), ()))
                pending.extend(value.values())
            elif isinstance(value, (tuple, list, FrozenList, frozenset, set)):
                # Exact immutable containers cannot change membership. Their
                # owning fields keep strong identity references; still visit
                # every nested mutable object and registry attachment.
                if type(value) is not tuple and type(value) is not frozenset:
                    retained.append((value, _retained_state(value), ()))
                pending.extend(value)
        return tuple(retained)

    def _registry_state(self):
        """Callback-free identities of the consumed registrations and handles."""
        from compiler.frontend.proof_arena import DerivationArena
        from compiler.frontend.source_facts import _ABSENT
        from compiler.scopes.conditional_return import ConditionalUnitDescriptorProof

        result = []
        for dependency in self._dependencies:
            owner, token = dependency.owner, dependency.token
            arena = getattr(owner, "_source_derivation_arena", None)
            handles = []
            fact = dependency.fact
            issued = dependency.issued
            origins = (() if fact is None else ((fact.arena, fact.derived, fact.query),))
            if issued:
                origins += ((issued[1], issued[2], issued[2].query),)
            for origin, derived, query in origins:
                if type(arena) is not DerivationArena:
                    raise CompilationError("storage admission lost its retained derivation arena")
                handles.append((_Identity(origin), _Identity(derived),
                    DerivationArena.accepts(arena, origin),
                    _Identity(DerivationArena._find(arena, (derived.family, derived.query))),
                    _Identity(DerivationArena._find_handle(arena, (derived.family.name, id(owner), query)))))
            registry = getattr(owner, dependency.name, None)
            result.append((_attachments(dependency), _Identity(registry),
                _Identity(dict.get(registry, dependency.key) if type(registry) is dict else None),
                _Identity(getattr(owner, dependency.name + "_owner", _ABSENT)),
                _Identity(getattr(owner, "_atomic_worksharing_owner", _ABSENT)
                          if dependency.name == "_atomic_worksharing_teams" else _ABSENT),
                _Identity(getattr(owner, "_source_authority_context", None)),
                _Identity(getattr(owner, "_source_session", None)), _Identity(arena),
                _Identity(getattr(token, "_record", _ABSENT)),
                _Identity(getattr(token, "identity", _ABSENT)),
                _Identity(getattr(token, "_source_fact", None)),
                _optional_attachments(fact), tuple(handles)))
        for proof, _attached, _registries in self._retained:
            if type(proof) is ConditionalUnitDescriptorProof:
                registry = getattr(proof._analysis, "_conditional_descriptor_consumers", None)
                result.append((_Identity(proof), _Identity(registry),
                               _Identity(dict.get(registry, proof.identity) if type(registry) is dict else None)))
        for handoff in self._plan._proof_handoffs:
            registry = getattr(handoff.analysis, handoff.registry, None)
            result.append((_Identity(handoff), _Identity(registry),
                           _Identity(dict.get(registry, handoff.key) if type(registry) is dict else None),
                           _Identity(handoff.analysis._source_session)))
        return tuple(result)

    def _check_retained(self):
        for proof, attached, registries in self._retained:
            if _retained_state(proof) != attached or any(
                    getattr(analysis, name, None) is not registry or dict.get(registry, key) is not proof
                    for analysis, name, registry, key in registries):
                raise CompilationError("storage admission lost a retained exact proof attachment")
        if self._registry_state() != self._registrations:
            raise CompilationError("storage admission changed retained prerequisite ownership")

    def _check(self):
        active = _ACTIVE.get()
        if not self._live or active is None or active[0] is not self:
            raise CompilationError("storage admission requires its active synchronous emission scope")
        if self._anchors() != active[1] or _mapping(self._admitted) != active[2]:
            raise CompilationError("storage admission changed its detached canonical record")
        if self._state() != self._initial:
            raise CompilationError("storage admission final inventory, facts or source policy changed")
        self._check_retained()
        for dependency in self._dependencies:
            dependency.require()
        # Source publication remains fresh before returning any membership.
        # This reads source inputs, not numerical IR or provider implementations.
        for analysis in self._analyses:
            analysis.inputs.verify_live()
        if self._state() != self._initial:
            raise CompilationError("storage admission changed during source/dependency validation")
        # No source, provider, or public serializer callback follows this fence.
        self._check_retained()

    def _anchors(self):
        return tuple(_Identity(getattr(self, name)) for name in
                     ("_plan", "_proof", "_owner", "_initial", "_retained", "_record", "_admitted",
                      "_dependencies", "_analyses", "_registrations"))

    def _release(self):
        if self._token is not None:
            _ACTIVE.reset(self._token)
        self._token = None
        self._live = False
        # No receipt or detached membership map survives the emission scope.
        for name in ("_plan", "_proof", "_owner", "_initial", "_retained", "_record", "_admitted",
                     "_dependencies", "_analyses", "_registrations"):
            setattr(self, name, None)

    def __enter__(self):
        if self._used or _ACTIVE.get() is not None:
            raise CompilationError("storage admission scope cannot be reused or nested")
        self._used = True
        try:
            return self._enter()
        except BaseException:
            self._release()
            raise

    def _enter(self):
        if not _supported(self._plan):
            raise CompilationError("storage admission requires a supported final sections storage inventory")
        self._initial = self._state()
        self._retained = self._authorities()
        from compiler.frontend.source_facts import dependencies

        self._dependencies = dependencies((analysis, name, registry, key, proof)
            for proof, _attached, registries in self._retained for analysis, name, registry, key in registries)
        self._analyses = tuple({id(analysis): analysis for _builder, analysis, _routine, _config, _facts
                                in self._plan._contexts}.values())
        self._registrations = self._registry_state()
        self._record = self._proof._record
        self._proof.validate(self._plan)
        record = json.loads(self._record)
        recorded_members = {item["procedure"]: item for item in record["members"]}
        if len(recorded_members) != len(record["members"]) or len(recorded_members) != len(self._plan.members):
            raise CompilationError("storage admission changed its sealed procedure inventory")
        admitted = {}
        for member in self._plan.members:
            stored = recorded_members.get(member.routine.qualified)
            units = self._plan.units(member)
            if stored is None or len(stored["units"]) != len(units):
                raise CompilationError("storage admission changed its sealed original units")
            for ordinal, (unit, issued) in enumerate(zip(units, stored["units"], strict=True)):
                canonical = issued["canonical_resources"]
                if issued["ordinal"] != ordinal or canonical.keys() != unit.arrays.keys():
                    raise CompilationError("storage admission changed its sealed canonical resource inventory")
                for root, binding in unit.arrays.items():
                    actual = member.canonical_bindings.get(binding.root, binding)
                    frozen_root = canonical[root]
                    registered = self._owner.resources.get(frozen_root)
                    if actual.root != frozen_root or registered is None or registered[0] is not actual:
                        raise CompilationError("storage admission changed an issued canonical binding association")
                    admitted[id(member), id(binding)] = member, binding, frozen_root
        self._admitted = FrozenDict(admitted)
        self._live = True
        self._token = _ACTIVE.set((self, self._anchors(), _mapping(self._admitted)))
        self._check()
        return self

    def validate(self, proof, plan):
        self._check()
        if proof is not self._proof or plan is not self._plan:
            raise CompilationError("storage admission cannot consume another checked plan or lifetime proof")
        return proof

    def admits_many(self, proof, plan, member, bindings):
        active = _ACTIVE.get()
        if (type(self) is not StorageAdmission or self._live is not True
                or type(active) is not tuple or len(active) != 3 or active[0] is not self):
            raise CompilationError("storage admission requires its active synchronous emission scope")
        if proof is not self._proof or plan is not self._plan:
            raise CompilationError("storage admission cannot consume another checked plan or lifetime proof")
        # Iteration can invoke arbitrary caller code. Fence it once, using the
        # unchanged complete source, dependency and retained-attachment checks.
        bindings = tuple(bindings)
        self._check()
        if _ACTIVE.get() is not active or self._live is not True:
            raise CompilationError("storage admission changed its active synchronous emission scope")
        anchors, snapshot = active[1], active[2]
        if (type(anchors) is not tuple
                or any(type(item) is not _Identity or len(item) != 1 for item in anchors)
                or self._anchors() != anchors):
            raise CompilationError("storage admission changed its detached canonical record")
        admitted = self._admitted
        # Exact builtin shapes make both the final table comparison and every
        # following lookup callback-free, including equal/hash-spoofed keys.
        if (type(admitted) is not FrozenDict
                or any(type(key) is not tuple or len(key) != 2
                       or any(type(part) is not int for part in key)
                       or type(entry) is not tuple or len(entry) != 3 or type(entry[2]) is not str
                       for key, entry in dict.items(admitted))):
            raise CompilationError("storage admission changed its builtin canonical membership table")
        if (type(snapshot) is not tuple or len(snapshot) != 2
                or type(snapshot[0]) is not _Identity or len(snapshot[0]) != 1
                or type(snapshot[1]) is not tuple
                or any(type(row) is not tuple or len(row) != 2
                       or type(row[0]) is not tuple or len(row[0]) != 2
                       or any(type(part) is not int for part in row[0])
                       or type(row[1]) is not _Identity or len(row[1]) != 1
                       for row in snapshot[1])
                or _mapping(admitted) != snapshot):
            raise CompilationError("storage admission changed its detached canonical membership")
        roots = []
        for binding in bindings:
            entry = dict.get(admitted, (id(member), id(binding)))
            if entry is None or entry[0] is not member or entry[1] is not binding:
                raise CompilationError("storage admission requires its exact admitted original binding")
            roots.append(entry[2])
        return tuple(roots)

    def __exit__(self, kind, value, traceback):
        try:
            if kind is None:
                self._check()
                # The ordinary public validator remains full and unmodified.
                self._proof.validate(self._plan)
                for _builder, analysis, _routine, _configuration, _facts in self._plan._contexts:
                    analysis.verify_publication()
                    analysis.inputs.verify_live()
                self._check()
        finally:
            self._release()


def active_admission():
    """Only the explicit emission routing may consume the current scope."""
    active = _ACTIVE.get()
    return active[0] if active is not None else None


def _supported(plan):
    from compiler.frontend.immutable_graph import original_type, sealed
    from compiler.scopes.execution_plan import CheckedExecutionPlan
    from compiler.scopes.graph import GraphOwner
    from compiler.scopes.graph_epochs import CallGroupStorageEpoch
    from compiler.scopes.hierarchical_regions import HierarchicalRegionOwner, SelectedRoot
    from compiler.scopes.lexical import LexicalOwner

    if type(plan) is not CheckedExecutionPlan:
        return False
    plan._require_owner()
    owner = plan._owner
    epoch = getattr(owner, "position_epoch", None)
    if (not sealed(owner) or owner.builder.config.scope_execution != "graph"
            or owner.builder.config.policy != "sections" or owner.builder.config.scope_measurement is not None):
        return False
    if type(epoch) is CallGroupStorageEpoch:
        return epoch._managed is not None
    if epoch is not None:
        return False
    if (owner.interval_draft is not None or getattr(owner, "local_exit", None) is not None
            or getattr(owner.builder, "_native_ancestor_selection", None) is not None
            or getattr(owner, "supported_path_mode", False)):
        return False
    record = json.loads(plan.storage_epoch()._record)
    if any(name in record for name in ("selected_position_epoch",
            "supported_path_epoch", "private_allocation_obligations", "original_worksharing_storage_obligations")):
        return False
    returned = getattr(owner, "return_contract", None)
    draft = getattr(owner.builder, "_conditional_return_draft", None)
    if original_type(owner) is HierarchicalRegionOwner:
        from compiler.scopes.conditional_return import ConditionalReturnDraft, ConditionalReturnProof

        selection = getattr(owner, "hierarchical_selection", None)
        return (type(returned) is ConditionalReturnProof and returned._owner is owner
                and type(draft) is ConditionalReturnDraft and draft.owner is owner
                and type(selection) is SelectedRoot and selection._owner is owner
                and draft.selected_root is selection and draft._selected_root_receipt is selection
                and "return_contract" in record)
    return (original_type(owner) in {LexicalOwner, GraphOwner} and returned is None and draft is None
            and "return_contract" not in record)


def emission_admission(plan):
    """Opt in after final plan issuance; all other owners retain full checks."""
    return StorageAdmission(plan) if _supported(plan) else nullcontext()


def validate_emitted_storage(proof, plan):
    admission = active_admission()
    return proof.validate(plan) if admission is None else admission.validate(proof, plan)


def admit_emitted_storage(proof, plan, member, bindings):
    admission = active_admission()
    return (proof.admits_many(plan, member, bindings) if admission is None
            else admission.admits_many(proof, plan, member, bindings))
