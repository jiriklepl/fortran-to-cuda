"""One-shot execution of an explicitly proved original source window.

This emitter does not discover windows or authorize storage.  It consumes a
registered source-position proof and reuses reached unit emission, including
native coherence hooks and failure handling.  The original prefix and excluded
tail stay in their original lexical owner.
"""

from __future__ import annotations

from compiler.frontend.reached_windows import ReachedWindowProof, plan_reached_window
from compiler.frontend.source_effects import _children, _part
from compiler.ir import CompilationError
from compiler.scopes.lexical import LexicalOwner
from compiler.scopes.provenance import source_id
from compiler.scopes.segments import fortran_lines, statement_span
from compiler.scopes.source import _name, _span


class ReachedWindowOwner(LexicalOwner):
    """Emit one dormant -> active -> spent owner, never a reopened owner."""

    def __init__(self, builder, proof, original_selection):
        if not isinstance(proof, ReachedWindowProof):
            raise CompilationError("reached window emission requires a registered source-position proof")
        if not isinstance(original_selection, tuple) or not original_selection:
            raise CompilationError("reached window emission requires exact original statements")
        proof.validate(builder.analysis, builder.entry.qualified, original_selection)
        self.window_proof = proof
        self.control_source_proof_identity = proof.identity
        self.window_originals = original_selection
        proxy = builder.bind_reached_window(proof, original_selection)
        super().__init__(proxy)
        self.spent = _name("fort_window_spent_", proof.identity)
        self.trace_owner = source_id(proxy, "owner", lifetime="one-shot reached window",
                                     reached_window_proof=proof.identity)

    def allow_child_continuation(self):
        return not self.window_proof.public().get("planned_closures", {}).get("enabled")

    def planned_source(self, nodes):
        planned = self.window_proof.public().get("planned_closures", {})
        if not planned.get("enabled") or all(type(node).__name__ == "Comment" for node in nodes):
            return False
        originals = self.builder.inline.original_selection(tuple(nodes))
        graph, selected = self.builder.analysis._selected_source(self.routine.qualified, originals)
        cuts = {point["node_id"]: point for point in planned["close_points"]}
        if len(selected) == 1 and selected[0] in cuts:
            self.boundary(tuple(nodes), cuts[selected[0]]["reason"], force_unconditional=True)
            return True
        active = set(planned["active_node_ids"])
        pending, descendants = list(selected), set()
        while pending:
            identity = pending.pop()
            if identity in descendants:
                continue
            descendants.add(identity)
            node = graph.nodes[identity]
            pending.extend(node.children)
            for condition, body in node.alternatives:
                pending.extend((body,) if condition is None else (condition, body))
        # All paths arriving here have closed. Leave the original native source
        # untouched, rather than even constructing its protected-descriptor query.
        return not bool(descendants & active)

    def boundary(self, nodes, reason, *, force_unconditional=False):
        originals = self.builder.inline.original_selection(tuple(nodes))
        _graph, identities = self.builder.analysis._selected_source(self.routine.qualified, originals)
        super().boundary(nodes, reason, force_unconditional=force_unconditional)
        self.boundaries[-1]["original_node_ids"] = list(identities)

    def validate_plan(self, nodes):
        planned = self.window_proof.public().get("planned_closures", {})
        if not planned.get("enabled"):
            return
        # Child companions cannot borrow a parent's conditional descriptor
        # plan. A separately mapped plan is still an independent requirement.
        if any(member.boundaries or member.units for member in self.members.values() if member is not self):
            raise CompilationError("planned reached child continuation requires separate mapped closure authority; "
                                   "close at its original caller instead")
        graph = self.builder.analysis.structure(self.routine.qualified)
        known = {point["node_id"] for point in planned["close_points"]}
        extra = []
        for boundary in self.boundaries:
            identities = tuple(boundary["original_node_ids"])
            if len(identities) != 1:
                raise CompilationError("planned reached closure cannot cut a partial original joined construct")
            if identities[0] not in known:
                extra.append((graph.source_nodes(identities[0]), boundary["reason"]))
        if extra:
            forced = (*self.window_proof._planned_close_before, *extra)
            proof = plan_reached_window(self.builder.analysis, self.routine.qualified, self.window_originals,
                                        self.window_proof._roots, self.builder.facts, close_before=forced)
            self.window_proof = proof
            self.builder._reached_window_binding = proof, self.window_originals
            planned = proof.public()["planned_closures"]
        cuts = {point["node_id"] for point in planned["close_points"]}
        actual = {identity for boundary in self.boundaries for identity in boundary["original_node_ids"]}
        if actual != cuts:
            raise CompilationError("planned reached close points differ from the exact staged original emission")
        retained = []
        active = set(planned["active_node_ids"])
        for unit in self.units:
            originals = self.builder.inline.original_selection(unit.nodes)
            _graph, identities = self.builder.analysis._selected_source(self.routine.qualified, originals)
            if set(identities) <= active:
                retained.append(unit)
            elif set(identities) & active:
                raise CompilationError("planned reached numerical unit crosses an original close point")
        self.units = retained

    def _validate_selection(self, nodes):
        self.builder.validate_reached_window()
        self.window_proof.validate(self.builder.analysis, self.routine.qualified, self.window_originals)
        originals = self.builder.inline.original_selection(tuple(nodes))
        if len(originals) != len(self.window_originals) or any(
                left is not right for left, right in zip(originals, self.window_originals, strict=True)):
            raise CompilationError("reached window emission selection differs from its registered source proof")

    def emit_control(self, nodes):
        """Declare at entry, spend at the reached window, close before its tail."""
        self._validate_selection(nodes)
        builder = self.builder
        original_body = tuple(_children(self.routine.execution))
        if not original_body:
            raise CompilationError("reached window original lexical body is unavailable")
        first_execution = statement_span(original_body[0])[0]
        first, last = statement_span(self.window_originals[0])[0], statement_span(self.window_originals[-1])[1]
        start = _span(_part(self.routine.scope.node, "Subroutine_Stmt"))[1] + 1
        builder.add_edit(self.routine.scope.path, start, start - 1,
                         f"use iso_c_binding, only: {self.kind} => c_int64_t\n")
        declarations = [f"integer({self.kind}) :: {self.context}", f"logical :: {self.enabled}, {self.spent}",
                        *[f"integer({self.kind}) :: fort_buffer_{number}"
                          for _binding, number in self.resources.values()]]
        initialize = [f"{self.context} = 0", f"{self.enabled} = .false.", f"{self.spent} = .false.",
                      *[f"fort_buffer_{number} = 0" for _binding, number in self.resources.values()]]
        builder.add_edit(self.routine.scope.path, first_execution, first_execution - 1,
                         "\n".join(fortran_lines([*declarations, *initialize])) + "\n", prepend=True)
        activation = [f"if (.not. {self.spent}) then", f"{self.spent} = .true.",
                      "block", "use fort_scoped_memory, only: fort_scope_serial_caller",
                      f"{self.enabled} = fort_scope_serial_caller() /= 0", "end block", "endif"]
        builder.add_edit(self.routine.scope.path, first, first - 1,
                         "\n".join(fortran_lines(activation)) + "\n")
        builder.add_edit(self.routine.scope.path, last + 1, last,
                         "\n".join(fortran_lines(self.close(disable=True,
                             reason="proved reached window end before excluded tail or enclosing backedge",
                             span=(last, last)))) + "\n")

    def run(self, nodes=None):
        """Emit only selected originals; the caller may supply approved facades."""
        if nodes is None:
            nodes = self.builder.inline.prepare(self.window_originals)
            self.builder.generated.update(self.builder.inline.generated)
            self.builder.numerical_ir.update(self.builder.inline.ir)
            self.builder.numerical_reasons.update({name: None for name in self.builder.inline.regions})
        nodes = tuple(nodes)
        self._validate_selection(nodes)
        before = len(self.builder.scopes)
        emitted = super().run(nodes)
        if emitted:
            public = self.window_proof.public()
            for scope in self.builder.scopes[before:]:
                scope["reached_window"] = {
                    "schema_version": 1, "proof_identity": self.window_proof.identity,
                    "source_proof_version": public["schema_version"],
                    "control_source_proof_identity": self.control_source_proof_identity,
                    "validated_emission_proof_identity": self.window_proof.identity,
                    "selected_node_ids": public["selected_node_ids"],
                    "prefix_node_ids": public["prefix_node_ids"],
                    "first_traversal_controls": public["first_traversal_controls"],
                    "entry_facts_identity": public["entry_facts_identity"],
                    "lifecycle": "one invocation; dormant -> active -> spent; no reopening",
                    "activation": "first reached selected source point; spend before every guard",
                    "mandatory_close": public["mandatory_close"],
                    "managed_resources": [state["resource"] for state in public["states"]],
                    "definition_proof_identities": public["native_definition_proof_identities"],
                    "planned_closures": public["planned_closures"],
                    "defined_coverage": [{"resource": state["resource"],
                        "on_activation": state["initialized"], "on_return": state["coverage_on_return"],
                        "return_sections": state["defined_sections_on_return"],
                        "authority": "conditional original reached transfer; return sections never seed registration"}
                        for state in public["states"]],
                    "prefix_execution": "unchanged original native source before any active context",
                    "allocation_and_alias_authority": "independent admitted facts and reached runtime guards",
                    "automatic_discovery": False,
                }
                scope["ownership"] = {
                    "lifetime": "one-shot original reached window", "planning_mode": "continuation",
                    "retained_resources": list(self.resources),
                    "end_reason": "first reached boundary or end of selected window before excluded tail/backedge",
                }
        return emitted
