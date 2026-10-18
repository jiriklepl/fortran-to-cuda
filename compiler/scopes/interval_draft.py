"""Source-position inventory shared by admission, emission and placement.

This is the construction state of a checked execution plan, not another effect
analyser. Original call associations and numerical preparations are retained
before native admission. Native effects, lifetime, sections and completion must
still be proved by their existing issuers; discovery never grants them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from fparser.two.utils import walk

from compiler.frontend.call_bindings import resolve_original_call
from compiler.frontend.native_group_structure import (
    NATIVE_GROUP_SCAN_LIMIT,
    NativeGroupCandidate,
    grouped_original_sequence,
)
from compiler.frontend.numerical_preparation import NumericalPreparationProvider
from compiler.frontend.source_effects import _children, _kind
from compiler.ir import CompilationError
from compiler.scopes.execution_plan import _configuration, _digest, _json
from compiler.scopes.segments import statement_span

_ISSUER = object()


@dataclass(frozen=True)
class IntervalMember:
    ordinal: int
    procedure: str
    originals: tuple = field(repr=False)
    selected: tuple
    graph: object = field(repr=False)
    parent: int | None
    association: object = field(repr=False)
    mappings: tuple = field(repr=False)
    chain: tuple = field(repr=False)

    def canonical(self, binding):
        return next((target for source, target in self.mappings if source is binding), binding)

    def public(self):
        return {"ordinal": self.ordinal, "procedure": self.procedure, "parent": self.parent,
                "selected_node_ids": list(self.selected),
                "association": self.association.public() if self.association is not None else None,
                "canonical_bindings": [{"local": local.public(), "canonical": target.public()}
                                       for local, target in self.mappings]}


@dataclass(frozen=True)
class NumericalAnchor:
    procedure: str
    originals: tuple = field(repr=False)
    prepared: object = field(repr=False)
    reason: str | None

    def public(self):
        return {"procedure": self.procedure,
                "span": [statement_span(self.originals[0])[0], statement_span(self.originals[-1])[1]],
                "preparation": self.prepared.public() if self.prepared is not None else None,
                "reason": self.reason}


@dataclass(frozen=True)
class NativeIntervalContext:
    demand: object
    local_bindings: tuple
    canonical_bindings: tuple
    associations: tuple
    _draft: object = field(repr=False)
    _member: object = field(repr=False)
    _issuer: object = field(repr=False)

    @property
    def identity(self):
        return _digest({"interval": self._draft.identity, "member": self._member.ordinal})

    def canonical(self, binding):
        return next((target for local, target in self.canonical_bindings if local is binding), binding)

    def validate(self, analysis, procedure, originals):
        from compiler.scopes.resource_demand import _covered_nodes

        draft = self._draft.validate()
        builder = draft._owner.builder
        if (type(self) is not NativeIntervalContext or self._issuer is not _ISSUER
                or analysis is not builder.analysis or procedure != self._member.procedure
                or getattr(builder, "_interval_native_contexts", {}).get(
                    (draft.identity, self._member.ordinal)) is not self):
            raise CompilationError("native interval context requires exact issued source authority")
        graph, selected = analysis._selected_source(procedure, tuple(originals))
        if graph is not self._member.graph or not set(selected) <= _covered_nodes(graph, self._member.selected):
            raise CompilationError("native operation escapes its source interval member")
        return self


@dataclass(frozen=True)
class SourceIntervalDraft:
    identity: str
    originals: tuple = field(repr=False)
    members: tuple[IntervalMember, ...]
    numerical_anchors: tuple[NumericalAnchor, ...]
    demand: object = field(repr=False)
    _owner: object = field(repr=False)
    _configuration: str
    _facts: str
    _context: str
    _record: str
    _issuer: object = field(repr=False)

    @property
    def selection_context(self):
        return json.loads(self._context)

    def public(self):
        return json.loads(self._record)

    def validate(self, plan=None):
        owner, builder = self._owner, self._owner.builder
        if (type(self) is not SourceIntervalDraft or self._issuer is not _ISSUER
                or getattr(builder, "_interval_drafts", {}).get(self.identity) is not self
                or owner.interval_draft is not self
                or _configuration(builder) != self._configuration or _digest(builder.facts) != self._facts):
            raise CompilationError("source interval requires its exact unchanged issued inventory")
        builder.analysis.verify_source()
        for member in self.members:
            graph, selected = builder.analysis._selected_source(member.procedure, member.originals)
            if graph is not member.graph or selected != member.selected:
                raise CompilationError("source interval original selection changed")
            if member.association is not None:
                call = member.association
                call.validate(builder.analysis, call._scope, call._original)
        for anchor in self.numerical_anchors:
            if anchor.prepared is not None:
                anchor.prepared.validate(builder.analysis, anchor.procedure, anchor.originals)
        if plan is not None:
            if plan._owner is not owner or owner.execution_plan is not plan:
                raise CompilationError("source interval belongs to another checked plan")
            self.demand.validate_plan(plan, validate_plan=False)
        return self

    def native_source(self):
        self.validate()
        return self._owner.original(self.originals)

    def has_numerical_child(self, original_call):
        self.validate()
        children = {member.ordinal for member in self.members
                    if member.association is not None and member.association._original is original_call}
        for member in self.members:
            if member.parent in children:
                children.add(member.ordinal)
        procedures = {member.procedure for member in self.members if member.ordinal in children}
        return any(anchor.procedure in procedures and anchor.prepared is not None
                   for anchor in self.numerical_anchors)

    def member(self, owner):
        self.validate()
        matches = [member for member in self.members if member.procedure == owner.routine.qualified
                   and all(owner.canonical(local) is target for local, target in member.mappings)]
        if not matches:
            raise CompilationError("native interval member lacks its exact original call association")
        # Existing companions admit one canonical mapping. Repeated occurrences
        # with that mapping share the implementation, not their source edges.
        return matches[0]

    def native_context(self, owner):
        member = self.member(owner)
        canonical = tuple(binding for binding in self.demand.bindings
                          if binding.dtype in {"real", "integer", "logical"})
        local = []
        for binding in canonical:
            mapped = tuple(source for source, target in member.mappings if target is binding)
            local.extend(mapped or (binding,))
        registry = getattr(self._owner.builder, "_interval_native_contexts", None)
        if registry is None:
            registry = self._owner.builder._interval_native_contexts = {}
        key = self.identity, member.ordinal
        if key not in registry:
            registry[key] = NativeIntervalContext(self.demand, tuple({id(value): value for value in local}.values()),
                member.mappings, member.chain, self, member, _ISSUER)
        return registry[key]

    def native_operation(self, owner, originals):
        from compiler.scopes.segments import native_demand_fragment

        context = self.native_context(owner)
        return native_demand_fragment(owner.builder, originals, context.demand, context=context)

    def finalize(self, owner):
        from compiler.scopes.execution_plan import build_execution_plan

        if owner is not self._owner:
            raise CompilationError("source interval cannot finalize a foreign owner")
        self.validate()
        if owner.execution_plan is not None:
            return owner.execution_plan.validate()
        return build_execution_plan(owner)

    def cost_entry(self):
        """Only visible entry values; child-local preparation is never hoisted."""
        plan, owner = self._owner.execution_plan, self._owner
        if plan is None:
            raise CompilationError("interval cost entry requires its finalized checked plan")
        self.validate(plan)
        arrays, integers = [], []
        initialized = tuple(getattr(owner, "closed_initializations", ()))
        for binding in self.demand.bindings:
            try:
                visible = owner.builder.visible_binding(owner.routine, binding.root)
            except CompilationError:
                continue
            if visible is not binding:
                continue
            if binding.attributes & {"pointer", "optional", "volatile", "asynchronous"}:
                continue
            if binding.rank:
                # A later local allocation or child descriptor is not an entry
                # key. Ordinary original dummies and module objects can be
                # queried only under the existing allocation/presence guards.
                if (binding.declaring_scope is owner.routine.scope
                        and binding.name not in owner.routine.arguments):
                    continue
                arrays.append(binding)
            elif binding.dtype == "integer" and (
                    "parameter" in binding.attributes or any(binding is item for item in initialized)
                    or binding.declaring_scope is owner.routine.scope
                    and binding.name in owner.routine.arguments and binding.intent == "in"):
                integers.append(binding)
        identity = _digest({"interval": self.identity,
                            "arrays": [value.public() for value in arrays],
                            "integers": [value.public() for value in integers]})
        return {"arrays": tuple(arrays), "integers": tuple(integers), "identity": identity}

    def completion_manifest(self):
        plan = self._owner.execution_plan
        if plan is None:
            raise CompilationError("interval completion requires its finalized checked plan")
        self.validate(plan)
        return {"schema_version": 1, "interval_identity": self.identity,
                "checked_execution_identity": plan.execution_identity,
                "normal_completion": "only after successful final publication and cleanup",
                "fallback": "sticky owning-decline count; planned native operations are not fallback",
                "operations": [{"procedure": member["procedure"],
                                "units": [{"ordinal": unit["ordinal"],
                                           "selected_node_ids": unit["selected_node_ids"]}
                                          for unit in member["units"]]}
                               for member in plan.public()["procedures"]]}


def prepare_interval_draft(owner, originals):
    """Collect original source and nested numerical anchors before admission."""
    if owner is not owner.owner or owner.units or owner.execution_plan is not None:
        raise CompilationError("source interval must precede owning admission")
    originals = tuple(originals)
    builder, analysis = owner.builder, owner.builder.analysis
    from compiler.scopes.resource_demand import _Inventory, _issue

    inventory = _Inventory()
    provider = NumericalPreparationProvider(analysis, options=builder.options, config=builder.config,
        environment_root=builder._numerical_environment_root,
        immutable_inputs=getattr(builder, "immutable_inputs", {}))
    members, anchors, templates, active, reasons = [], [], {}, [], []
    work = {"source_nodes": 0, "call_instances": 0}

    def references(procedure, nodes, canonical):
        routine = analysis.routines[procedure]
        for original in nodes:
            for node in walk(original):
                work["source_nodes"] += 1
                if work["source_nodes"] > NATIVE_GROUP_SCAN_LIMIT:
                    raise CompilationError("source interval inventory scan budget exhausted")
                if _kind(node) != "Name":
                    continue
                scope = analysis.source_scope_for(node, routine.scope)
                binding = analysis._binding(scope, node)
                if binding is None:
                    continue
                role = ("managed" if binding.rank else "scalar_or_guard") if binding.dtype in {
                    "real", "integer", "logical"} else "source_storage"
                inventory.add(procedure, binding, canonical(binding), role)

    def template(procedure, nodes):
        key = procedure, tuple(map(id, nodes))
        if key in templates:
            return templates[key]
        calls, local_anchors = [], []
        peeled = {}

        def sequence(sequence_nodes):
            for item in grouped_original_sequence(sequence_nodes, peeled):
                if isinstance(item, NativeGroupCandidate):
                    selected = item.original_nodes
                    try:
                        prepared = provider.prepare(analysis, procedure, selected)
                    except CompilationError as error:
                        local_anchors.append(NumericalAnchor(procedure, selected, None, str(error)))
                    else:
                        local_anchors.append(NumericalAnchor(procedure, selected, prepared, None))
                        continue  # Helpers belong to this exact numerical proof.
                    for original in selected:
                        calls.extend(node for node in walk(original) if _kind(node) == "Call_Stmt")
                elif _kind(item) in {"If_Construct", "Associate_Construct"}:
                    sequence(_children(item))
                elif _kind(item) == "Call_Stmt":
                    calls.append(item)
                elif _kind(item) == "If_Stmt":
                    if _kind(item.items[1]) == "Call_Stmt":
                        calls.append(item.items[1])
                elif _kind(item) == "Block_Nonlabel_Do_Construct":
                    # The ordinary lowering may already have prepared this
                    # exact serial loop. Retain its token when available.
                    retained = next((value for value in getattr(analysis, "_numerical_preparations", {}).values()
                                     if value.procedure == procedure and value._selection == (item,)), None)
                    if retained is None:
                        try:
                            retained = provider.prepare(analysis, procedure, (item,))
                        except CompilationError as error:
                            local_anchors.append(NumericalAnchor(procedure, (item,), None, str(error)))
                    if retained is not None:
                        local_anchors.append(NumericalAnchor(procedure, (item,), retained, None))
                    else:
                        calls.extend(node for node in walk(item) if _kind(node) == "Call_Stmt")
        sequence(nodes)
        templates[key] = tuple({id(node): node for node in calls}.values()), tuple(local_anchors)
        anchors.extend(local_anchors)
        return templates[key]

    def visit(procedure, nodes, mappings=(), parent=None, association=None, chain=()):
        if procedure in active or len(active) >= analysis.depth_limit:
            raise CompilationError("source interval recursive or over-depth call graph")
        if len({member.procedure for member in members} | {procedure}) > analysis.procedure_limit:
            raise CompilationError("source interval procedure budget exhausted")
        graph, selected = analysis._selected_source(procedure, nodes)
        member = IntervalMember(len(members), procedure, nodes, selected, graph, parent, association, mappings, chain)
        members.append(member)
        inventory.selections.append((analysis, procedure, graph, nodes, selected))
        inventory.controls(analysis, procedure, graph, selected, member.canonical)
        references(procedure, nodes, member.canonical)
        calls, _anchors = template(procedure, nodes)
        active.append(procedure)
        try:
            for node in calls:
                work["call_instances"] += 1
                if work["call_instances"] > analysis.operation_limit:
                    raise CompilationError("source interval call-instance budget exhausted")
                routine = analysis.routines[procedure]
                scope = analysis.source_scope_for(node, routine.scope)
                try:
                    call = resolve_original_call(analysis, scope, node)
                except CompilationError as error:
                    reasons.append({"procedure": procedure, "source": str(node), "reason": str(error)})
                    # A source routine may have an ordinary callback interface
                    # which cannot issue a numerical/storage association. Its
                    # declarations still belong in the conservative inventory.
                    # Retain no canonical mapping or native-call authority.
                    targets = analysis._candidates(scope, str(node.items[0]).lower())
                    for target in targets:
                        callee = analysis.routines.get(target)
                        if callee is not None:
                            visit(target, tuple(_children(callee.execution)), parent=member.ordinal)
                    continue  # Original conditional close still requires its own proof.
                mapping = []
                for argument in call.associations:
                    if (argument.formal_binding.rank and argument.binding is not None
                            and _kind(argument.actual) == "Name"
                            and argument.formal_binding.rank == argument.binding.rank):
                        mapping.append((argument.formal_binding, member.canonical(argument.binding)))
                visit(call.procedure, tuple(_children(call.callee.execution)), tuple(mapping), member.ordinal,
                      call, (*chain, call))
        finally:
            active.pop()

    visit(owner.routine.qualified, originals)
    for anchor in anchors:
        if anchor.prepared is None:
            continue
        for member in members:
            if member.procedure != anchor.procedure:
                continue
            for binding in anchor.prepared.extraction.bindings:
                inventory.add(member.procedure, binding, member.canonical(binding),
                              "managed" if binding.rank else "planning")
            for precondition in anchor.prepared.preconditions:
                for dependency in precondition.dependencies:
                    inventory.add(member.procedure, dependency.binding, member.canonical(dependency.binding),
                                  "descriptor_or_control")
    demand = _issue(builder, inventory)
    first, last = statement_span(originals[0])[0], statement_span(originals[-1])[1]
    text = "".join(analysis.source_text(owner.routine.scope.path).splitlines(keepends=True)[first-1:last])
    context = {"schema_version": 1, "procedure": owner.routine.qualified,
               "selected_node_ids": list(members[0].selected), "first_line": first, "last_line": last,
               "original_source_sha256": _digest(text),
               "activation": "original source position and original control",
               "fallback": "publish and continue only the unexecuted suffix"}
    record = {"schema_version": 1, "selection_context": context,
              "members": [member.public() for member in members],
              "numerical_anchors": [anchor.public() for anchor in anchors],
              "unresolved_source_calls": reasons, "work": work,
              "sources": dict(analysis.sources), "configuration_identity": _configuration(builder),
              "capture_facts_identity": _digest(builder.facts),
              "execution_authorized": False, "placement_authorized": False}
    # Preparation tokens are exact in-process authorities. The interval's
    # source identity does not depend on their per-process Python identities.
    identity = _digest({key: value for key, value in record.items() if key != "numerical_anchors"})
    record["identity"] = identity
    result = SourceIntervalDraft(identity, originals, tuple(members), tuple(anchors), demand, owner,
        _configuration(builder), _digest(builder.facts), _json(context), _json(record), _ISSUER)
    registry = getattr(builder, "_interval_drafts", None)
    if registry is None:
        registry = builder._interval_drafts = {}
    if identity in registry:
        raise CompilationError("source interval already has an issuing owner")
    registry[identity] = result
    owner.interval_draft = result
    return result.validate()
