"""Cost applicability inputs from one checked interval, without placement authority.

This is the shared input to a future measured attachment, not a second planner.
It inventories positive key candidates and every currently missing proof from
already admitted resources/operations. No source expression is evaluated and no
native call graph is rediscovered. Existing exact-window admission and its
issuer-authenticated decision remain unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from compiler.ir import CompilationError
from compiler.offload.exact_window import digest
from compiler.scopes.execution_plan import CheckedExecutionPlan


@dataclass(frozen=True, slots=True)
class IntervalApplicability:
    """Immutable diagnostic projection; public data cannot authorize a cost."""

    _plan: object
    _demand: object
    _placement: object
    _native: tuple
    _record: str

    def validate(self, plan):
        if type(plan) is not CheckedExecutionPlan or plan is not self._plan:
            raise CompilationError("interval applicability requires its exact checked plan")
        if self._demand._plan is not plan:
            raise CompilationError("interval applicability demand belongs to another checked plan")
        # Demand validation already validates this exact plan and all contexts.
        self._demand.validate()
        if plan.exact_window_placement() is not self._placement:
            raise CompilationError("interval applicability placement changed; inspect the checked plan again")
        for proof, graph in self._native:
            if proof.native_graph is not graph:
                raise CompilationError("interval applicability native graph changed after inspection")
            proof.validate(proof._analysis, proof.procedure, proof._selection, proof.demanded)
        return self

    def public(self):
        self.validate(self._plan)
        return json.loads(self._record)


def _key_inputs(demand):
    """Keep candidate values separate from permission to evaluate or reuse them."""
    candidates, dependencies = [], []
    grouped = {}
    for item in demand._entries:
        row = grouped.setdefault(id(item.canonical), (item.canonical, set(), set()))
        row[1].update(item.roles)
        row[2].add(item.procedure)
    for binding, roles, procedures in sorted(grouped.values(), key=lambda item: item[0].root):
        unstable = sorted(binding.attributes & {"optional", "pointer", "allocatable", "volatile", "asynchronous"})
        base = {"resource": binding.root, "descriptor": binding.public(),
                "roles": sorted(roles), "procedures": sorted(procedures)}
        if binding.rank:
            properties = [{"property": prop, "axis": axis}
                          for axis in range(1, binding.rank + 1) for prop in ("size", "lbound")]
            requirements = ["Use the existing reached descriptor; never re-evaluate specification expressions.",
                            "Bind physical strides/contiguity and allocation lifetime across the optimized interval."]
            if roles & {"descriptor_or_control", "enclosing_control", "inspector", "call_section"}:
                requirements.append("Classify descriptor inquiries versus payload-dependent controls; equal shapes or section boxes do not prove equal native work.")
            if unstable:
                requirements.append("Establish safe presence/association before descriptor inquiries: " + ", ".join(unstable))
            candidates.append({**base, "kind": "array_descriptor", "properties": properties,
                               "basis": "exact canonical resource declaration and checked interval demand",
                               "safe_evaluation_authorized": False, "missing_proofs": requirements})
        elif "parameter" in binding.attributes:
            candidates.append({**base, "kind": "source_constant", "properties": [],
                               "basis": "constant declaration bound by authenticated source",
                               "safe_evaluation_authorized": False, "missing_proofs": []})
        elif binding.dtype in {"integer", "logical"} and binding.intent == "in" and not unstable:
            candidates.append({**base, "kind": binding.dtype + "_input",
                               "properties": [{"property": binding.dtype + "_value"}],
                               "basis": "ordinary original " + binding.dtype + " INTENT(IN) declaration",
                               "safe_evaluation_authorized": False,
                               "missing_proofs": ["Prove no aliased or indirect write changes this value throughout the interval.",
                                                  "Read at its original safe evaluation point; preserve caller argument evaluation."]})
        else:
            dependencies.append({**base, "status": "not_a_proven_entry_key",
                                 "reason": "Demand membership does not prove a safely readable stable workload value.",
                                 "missing_proofs": ["Classify whether this value controls work or is an internal iterator/payload.",
                                                    "For workload controls, prove original evaluation point and interval stability."]})
    return candidates, dependencies


def analyze_interval_applicability(plan):
    """Collect independent pricing requirements without changing any decision.

    The checked plan already proves execution. Its source-bound demand is a
    conservative input inventory, not a workload key or a numerical cost model.
    A registered native graph supplies topology once; it does not certify that
    its loop bounds/branches can be inspected before original execution.
    """
    if type(plan) is not CheckedExecutionPlan:
        raise CompilationError("interval applicability requires a checked execution plan")
    # The cached demand authenticates this exact plan; do not rescan its inventory.
    demand = plan.resource_demand()
    public = plan.public()
    placement = plan.exact_window_placement()
    existing = None
    if placement is not None:
        issued = placement.public()
        existing = {key: issued[key] for key in (
            "available", "selection", "reason", "contract_sha256", "evidence_sha256", "capture") if key in issued}
    candidates, dependencies = _key_inputs(demand)
    native, numerical, controls, proofs, requirements = [], [], [], [], []

    def require(kind, reason, **context):
        row = {"kind": kind, "status": "not_established", "reason": reason, **context}
        if row not in requirements:
            requirements.append(row)

    for member_index, member in enumerate(plan.members):
        procedure = member.routine.qualified
        for ordinal, unit in enumerate(plan.units(member)):
            location = {"procedure": procedure, "unit": ordinal}
            operations = plan.operations(member, ordinal)
            selected = public["procedures"][member_index]["units"][ordinal]["selected_node_ids"]
            seen, pending, source_controls = set(), list(selected), []
            while pending:
                identity = pending.pop()
                if identity in seen:
                    continue
                seen.add(identity)
                node = member.graph.nodes[identity]
                pending.extend(node.children)
                pending.extend(child for pair in node.alternatives for child in pair if child is not None)
                if node.kind in {"loop", "branch", "associate"} or node.details.get("evaluation"):
                    source_controls.append(node.public())
            # This is the sealed operation tree, not another traversal of ASTs.
            controls.append({**location, "tree": operations.get("structured_tree"),
                             "admitted_operations": operations if "structured_tree" not in operations else None,
                             "source_control_nodes": source_controls,
                             "selected_node_ids": selected,
                             "entry_key_authorized": False})
            for index, operation in enumerate(unit.scope.native):
                effects, unmapped = {}, []
                for root, actions in operation.effects.items():
                    if root not in operation.bindings:
                        unmapped.append(root)
                    else:
                        canonical = member.canonical(operation.bindings[root]).root
                        effects.setdefault(canonical, set()).update(actions)
                if unmapped:
                    require("native_effect_mapping", "Native workload references need exact canonical declaration mappings.",
                            **location, operation=index, resources=sorted(unmapped))
                row = {**location, "operation": index, "kind": operation.kind,
                       "canonical_effects": {root: sorted(actions) for root, actions in sorted(effects.items())},
                       "unmapped_effects": sorted(unmapped),
                       "completion": operation.summary["native_completion"],
                       "sections": operation.sections.public() if operation.sections is not None else None,
                       "runtime_workload_key": None}
                if operation.demand_effects is not None:
                    proof = operation.demand_effects.reachability
                    proof.validate(proof._analysis, proof.procedure, proof._selection, proof.demanded)
                    from compiler.frontend.resource_reachability import NativeEffectGraph

                    graph = proof.native_graph
                    if type(graph) is not NativeEffectGraph:
                        raise CompilationError("interval applicability requires the registered native effect graph")
                    record = graph.public()
                    row["native_graph"] = record
                    row["native_graph_identity"] = digest(record)
                    proofs.append((proof, graph))
                    require("native_graph_workload", "Bind this original native graph to complete-process source/workload observations; internal branch values and iteration counts need not be predicted at entry.", **location, operation=index)
                else:
                    row["native_graph"] = None
                    require("native_workload", "Native operation has execution/effect authority but no source-bound workload observation contract.", **location, operation=index)
                if getattr(operation, "indirect_sections", None) is not None:
                    require("inspector_workload", "Exact section boxes do not identify inspector traversal work; include its work controls and preparation cost.", **location, operation=index)
                native.append(row)
            for index, call in enumerate(unit.scope.calls):
                region = call.region
                numerical.append({**location, "call": index, "procedure_called": call.procedure,
                                  "source_region": region.public() if region is not None else None,
                                  "tiled": bool(region is not None and region.tile_domains),
                                  "runtime_guards": list(region.runtime_guards) if region is not None else [],
                                  "native_participation": "preserve original checked completion and team context"})
    for item in public["cost_requirements"]:
        require(item["kind"], item["reason"], **{key: value for key, value in item.items() if key not in {"kind", "reason"}})
    require("runtime_workload_key", "Bind only the finalized safe root-entry geometry and original workload identity; never hoist internal native controls or child initializers.")
    require("same_measured_alternative", "Preserve the checked source/configuration/worker identity across measurement and production; policy/profile fields have not been removed or declared equivalent.")
    require("application_build_attestation", "Bind executed compiler/options/source/header/object receipts for every original and generated object to a compiler-issued application build contract.")
    require("actual_execution_outcome", "Record whether the measured resident invocation executed the admitted alternative or guarded native fallback, with the independently observed native identity.")
    require("complete_interval_observations", "Measure paired complete original invocations including guards, inspectors, original teams, entry/interruption/exit transfers, synchronization, planning and cleanup; do not add these costs twice.")
    if placement is not None:
        from compiler.offload.selection_cost import PROTOCOL

        attached = placement.public()
        if attached["contract"].get("protocol") == PROTOCOL and attached.get("available") is True:
            for requirement in requirements:
                requirement["status"] = "covered_by_checked_interval_attachment"
                requirement["attachment"] = attached["contract_sha256"]
    record = {
        "schema_version": 1, "kind": "checked-interval-cost-applicability",
        "plan_identity": plan.identity, "execution_identity": plan.execution_identity,
        **{key: public[key] for key in ("entry", "sources", "source_inputs_identity", "configuration_identity", "capture_facts_identity")},
        "canonical_resources": public["resources"], "key_candidates": candidates,
        "unclassified_dependencies": dependencies, "original_control_trees": controls,
        "native_operations": native, "numerical_operations": numerical,
        "requirements": requirements,
        "existing_exact_window": existing,
        "placement_authorized": False, "execution_authorized": False,
        "existing_placement_unchanged": True,
        "identity_policy": "Exact existing policy/configuration identity retained; no measured/production equivalence inferred.",
        "coverage": "Application measurement attachment requirements, not a replacement for existing exact-window admission or its issued decision.",
    }
    record["identity"] = digest(record)
    result = IntervalApplicability(plan, demand, placement, tuple(proofs),
                                   json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return result
