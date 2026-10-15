"""Sealed resource inventories for one original interval, never effect proofs.

A reached-window inventory is prospective: every eventual checked-plan use
must be covered before emission. An already scanned prefix is never an input.
Neither factory authorizes native execution, aliases, or automatic placement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from fparser.two.utils import walk

from compiler.frontend.component_bindings import references
from compiler.ir import CompilationError
from compiler.scopes.execution_plan import CheckedExecutionPlan, _configuration, _digest, _json

_ISSUER = object()
_COST_REASON = "native interval costs require compatible source-bound observations"


@dataclass(frozen=True, slots=True)
class _Resource:
    procedure: str
    binding: object
    canonical: object
    roles: tuple[str, ...]


class _Inventory:
    def __init__(self):
        self.entries = {}
        self.selections = []
        self.parents = {}
        self.metadata_projections = {}

    def allow_native_metadata(self, analysis, bindings):
        """Reuse admitted opaque objects only; never enable new field captures."""
        objects = {id(binding): binding for binding in bindings if binding.dtype.startswith("type(")}
        if not objects:
            return
        previous = self.metadata_projections.get(id(analysis))
        if previous is not None:
            previous[1].update(objects)
            return
        projection = analysis.private_projection()
        projection._native_metadata = True
        self.metadata_projections[id(analysis)] = projection, objects

    def references(self, analysis, scope, node):
        projection, objects = self.metadata_projections.get(id(analysis), (analysis, {}))
        for binding, reference in references(projection, scope, node):
            metadata = getattr(binding, "native_metadata_object", None)
            if metadata is not None:
                if objects.get(id(metadata)) is not metadata:
                    raise CompilationError("resource demand references unadmitted opaque native metadata")
                # The temporary scalar field is not a declared allocation and
                # must not become a second, synthetic resource authority.
                binding = metadata
            yield binding, reference

    def add(self, procedure, binding, canonical=None, role="dependency"):
        canonical = binding if canonical is None else canonical
        key = procedure, id(binding), id(canonical)
        previous = self.entries.get(key)
        roles = set(previous.roles) if previous is not None else set()
        roles.add(role)
        self.entries[key] = _Resource(procedure, binding, canonical, tuple(sorted(roles)))

    def select(self, analysis, procedure, originals, canonical):
        graph, selected = analysis._selected_source(procedure, originals)
        self.selections.append((analysis, procedure, graph, tuple(originals), selected))
        routine = analysis.routines[procedure]
        self.controls(analysis, procedure, graph, selected, canonical)
        # Conservatively retain scalar operands from the entire selection,
        # including later units. Array operands of conditions, loop headers,
        # and descriptor inquiries are dependencies even if never GPU payload.
        for node in originals:
            for binding, _ in self.references(analysis, routine.scope, node):
                if not binding.rank:
                    self.add(procedure, binding, canonical(binding), "scalar_or_guard")
            for expression in walk(node):
                kind = type(expression).__name__
                if kind in {"If_Then_Stmt", "Else_If_Stmt", "If_Stmt"}:
                    expression = expression.items[0]
                elif kind == "Intrinsic_Function_Reference":
                    if str(expression.items[0]).lower() not in {"lbound", "ubound", "size", "shape", "allocated"}:
                        continue
                elif kind != "Nonlabel_Do_Stmt":
                    continue
                for binding, _ in self.references(analysis, routine.scope, expression):
                    self.add(procedure, binding, canonical(binding), "descriptor_or_control")

    def controls(self, analysis, procedure, graph, selected, canonical):
        routine = analysis.routines[procedure]
        if id(graph) not in self.parents:
            parents = {}
            for identity, item in graph.nodes.items():
                children = (*item.children, *(child for pair in item.alternatives for child in pair if child is not None))
                for child in children:
                    parents.setdefault(child, set()).add(identity)
            self.parents[id(graph)] = parents
        parents = self.parents[id(graph)]
        ancestors, pending = set(selected), list(selected)
        while pending:
            for identity in parents.get(pending.pop(), ()):
                if identity not in ancestors:
                    ancestors.add(identity)
                    pending.append(identity)
        for identity in sorted(ancestors):
            item = graph.nodes[identity]
            controls = [condition for condition, _body in item.alternatives if condition is not None]
            controls.extend(child for child in item.children if graph.nodes[child].details.get("evaluation")
                            in {"loop_header", "associate_selector"})
            for control in controls:
                for expression in graph.source_nodes(control):
                    for binding, _ in self.references(analysis, routine.scope, expression):
                        self.add(procedure, binding, canonical(binding), "enclosing_control")
    def sections(self, procedure, sections, canonical):
        for resource in getattr(sections, "resources", ()):
            for dependency in resource.dependencies:
                self.add(procedure, dependency.binding, canonical(dependency.binding), "section_" + dependency.kind)

    def finish(self):
        return tuple(sorted(self.entries.values(), key=lambda item: (
            item.procedure, item.binding.root, item.canonical.root, item.roles)))


def _plan_inventory(plan):
    inventory = _Inventory()
    for member in plan.members:
        inventory.allow_native_metadata(member.builder.analysis,
            (binding for unit in plan.units(member) for operation in unit.scope.native
             for binding in operation.host_metadata.values()))
    for member in plan.members:
        procedure, builder = member.routine.qualified, member.builder
        # A scalar-only close guard has no admitted Native/Unit. Read its
        # original graph condition, without demanding the excluded body.
        spans = {(item["first_line"], item["last_line"]) for item in member.boundaries}
        boundary_ids = tuple(identity for identity, item in member.graph.nodes.items() if item.span in spans)
        inventory.controls(builder.analysis, procedure, member.graph, boundary_ids, member.canonical)
        for binding, _number in member.resources.values():
            inventory.add(procedure, binding, member.canonical(binding), "registered")
        for unit in plan.units(member):
            originals = builder.inline.original_selection(unit.nodes)
            inventory.select(builder.analysis, procedure, originals, member.canonical)
            for binding in unit.arrays.values():
                inventory.add(procedure, binding, member.canonical(binding), "managed")
            for binding in unit.scalars.values():
                inventory.add(procedure, binding, member.canonical(binding), "planning")
            for operation in unit.scope.native:
                for binding in operation.bindings.values():
                    inventory.add(procedure, binding, member.canonical(binding), "native_dependency")
                inventory.sections(procedure, operation.sections, member.canonical)
                indirect = getattr(operation, "indirect_sections", None)
                for binding in (*getattr(indirect, "metadata", ()), *getattr(indirect, "scalars", ())):
                    inventory.add(procedure, binding, member.canonical(binding), "inspector")
                for state in operation.native_states.values():
                    inventory.add(procedure, state.binding, member.canonical(state.binding), "native_environment")
            for call in unit.scope.calls:
                if call.region is not None:
                    for binding in call.region.bindings:
                        inventory.add(procedure, binding, member.canonical(binding), "numerical_or_guard")
                resolved = getattr(call, "resolved", None)
                for mapping in getattr(resolved, "mappings", ()):
                    section = getattr(mapping, "section", None)
                    for dependency in getattr(section, "dependencies", ()):
                        inventory.add(procedure, dependency.binding, member.canonical(dependency.binding), "call_section")
    return inventory


def _reached_inventory(builder, proof):
    inventory = _Inventory()
    analysis = builder.analysis

    def visit(procedure, originals, record, children, mappings):
        routine = analysis.routines[procedure]

        def canonical(binding):
            return mappings.get(binding.root, binding)
        inventory.select(analysis, procedure, originals, canonical)
        for state in record["states"]:
            binding = builder.visible_binding(routine, state["resource"])
            inventory.add(procedure, binding, canonical(binding), "managed")
            inventory.add(procedure, binding, canonical(binding), "registered")
        for child in children:
            child.coordinates.validate(analysis, procedure, child.original_call)
            child_mappings = {item.formal.root: canonical(item.actual) for item in child.coordinates._arrays}
            visit(child.procedure, child.selection, child.public()["plan"], child.children, child_mappings)

    visit(proof.procedure, proof._selection, proof.public(), proof._children, {})
    for _root, sections in proof._activation_resources:
        for dependency in sections.dependencies:
            inventory.add(proof.procedure, dependency.binding, role="section_" + dependency.kind)
    return inventory


def _covered_nodes(graph, selected):
    covered, pending = set(), list(selected)
    while pending:
        identity = pending.pop()
        if identity in covered:
            continue
        covered.add(identity)
        node = graph.nodes[identity]
        pending.extend(node.children)
        for condition, body in node.alternatives:
            pending.extend((body,) if condition is None else (condition, body))
    return covered


@dataclass(frozen=True, slots=True)
class IntervalResourceDemand:
    identity: str
    _builder: object
    _plan: object
    _proof: object
    _entries: tuple
    _selections: tuple
    _context: tuple
    _record: str
    _issuer: object

    @property
    def bindings(self):
        """Exact original canonical bindings, including non-payload dependencies."""
        return tuple({id(item.canonical): item.canonical for item in self._entries}.values())

    @property
    def managed_bindings(self):
        return tuple({id(item.canonical): item.canonical for item in self._entries
                      if "managed" in item.roles}.values())

    def public(self):
        return json.loads(self._record)

    def validate(self):
        if (self._issuer is not _ISSUER
                or getattr(self._builder, "_resource_demands", {}).get(self.identity) is not self):
            raise CompilationError("resource demand requires its exact issued inventory")
        analysis, routine, configuration, facts = self._context
        if (self._builder.analysis is not analysis or self._builder.entry is not routine
                or _configuration(self._builder) != configuration or _digest(self._builder.facts) != facts):
            raise CompilationError("resource demand source, configuration or facts changed")
        analysis.verify_publication()
        if self._plan is not None:
            # CheckedExecutionPlan.validate does not consult resource demands;
            # validate all child contexts and numerical authorities as well.
            self._plan.validate()
            if self._plan._owner.builder is not self._builder:
                raise CompilationError("resource demand belongs to a different checked plan")
        if self._proof is not None:
            self._proof.validate(analysis, routine.qualified, self._proof._selection)
        for source, procedure, graph, originals, selected in self._selections:
            current_graph, current_selected = source._selected_source(procedure, originals)
            if current_graph is not graph or current_selected != selected:
                raise CompilationError("resource demand original selection changed")
        return self

    def validate_plan(self, plan):
        """Require every final member/use; prospective inventories cannot shrink."""
        self.validate()
        if type(plan) is not CheckedExecutionPlan:
            raise CompilationError("resource demand requires an exact checked execution plan")
        plan.validate()
        if self._plan is not None and plan is not self._plan:
            raise CompilationError("resource demand belongs to a different checked plan")
        if (plan._owner.builder.analysis is not self._context[0]
                or plan._owner.routine is not self._context[1]
                or _configuration(plan._owner.builder) != self._context[2]
                or _digest(plan._owner.builder.facts) != self._context[3]):
            raise CompilationError("resource demand and checked plan source contexts differ")
        required = _plan_inventory(plan)
        covered = {id(binding) for binding in self.bindings}
        managed = {id(binding) for binding in self.managed_bindings}
        missing = sorted({item.canonical.root for item in required.finish()
                          if id(item.canonical) not in covered
                          or "managed" in item.roles and id(item.canonical) not in managed})
        if missing:
            raise CompilationError("resource demand misses final interval resources: " + ", ".join(missing))
        selections = {}
        for analysis, procedure, graph, _originals, selected in self._selections:
            selections.setdefault((id(analysis), procedure, id(graph)), set()).update(_covered_nodes(graph, selected))
        for analysis, procedure, graph, _originals, selected in required.selections:
            if not set(selected) <= selections.get((id(analysis), procedure, id(graph)), set()):
                raise CompilationError("resource demand misses final original selection: " + procedure)
        return self


def _issue(builder, inventory, *, plan=None, proof=None):
    entries = inventory.finish()
    record = {"schema_version": 1, "role": "selected interval resource demand",
              "checked_execution_identity": plan.execution_identity if plan is not None else None,
              "reached_window_identity": proof.identity if proof is not None else None,
              "prospective": plan is None, "requires_final_plan_validation": plan is None,
              "resources": [{"procedure": item.procedure, "resource": item.binding.root,
                             "canonical_resource": item.canonical.root, "roles": list(item.roles),
                             "descriptor": item.canonical.public()} for item in entries],
              "selections": [{"procedure": procedure, "structured_identity": graph.identity,
                              "selected_node_ids": list(selected)}
                             for _analysis, procedure, graph, _originals, selected in inventory.selections],
              "sources": dict(builder.analysis.sources), "configuration_identity": _configuration(builder),
              "capture_facts_identity": _digest(builder.facts), "complete_effect_summary": False,
              "execution_authorized": False, "placement_authorized": False,
              "cost_coverage": "not_established_by_inventory",
              "existing_cost_requirements": plan.public()["cost_requirements"] if plan is not None else [],
              "prospective_cost_requirements": ([{"kind": "source_native_observations", "reason": _COST_REASON}]
                                                if proof is not None else [])}
    identity = _digest(record)
    registry = getattr(builder, "_resource_demands", None)
    if registry is None:
        registry = builder._resource_demands = {}
    previous = registry.get(identity)
    if previous is not None:
        if previous._plan is not plan or previous._proof is not proof:
            raise CompilationError("resource demand identity belongs to another source authority")
        return previous.validate()
    record["identity"] = identity
    result = IntervalResourceDemand(identity, builder, plan, proof, entries, tuple(inventory.selections),
                                    (builder.analysis, builder.entry, _configuration(builder), _digest(builder.facts)),
                                    _json(record), _ISSUER)
    registry[identity] = result
    return result.validate()


def resource_demand_from_plan(plan):
    """Seal all admitted members; this cannot authorize pre-scan admission."""
    if type(plan) is not CheckedExecutionPlan:
        raise CompilationError("resource demand requires an exact checked execution plan")
    plan.validate()
    return _issue(plan._owner.builder, _plan_inventory(plan), plan=plan)


def resource_demand_from_reached_window(builder, proof):
    """Issue only from registered full-window states, never a scanned prefix."""
    from compiler.frontend.reached_windows import ReachedWindowProof

    if type(proof) is not ReachedWindowProof:
        raise CompilationError("resource demand requires a registered reached-window proof")
    proof.validate(builder.analysis, builder.entry.qualified, proof._selection)
    if proof.public()["entry_facts_identity"] != _digest(builder.facts):
        raise CompilationError("resource demand reached-window capture facts differ")
    return _issue(builder, _reached_inventory(builder, proof), proof=proof)
