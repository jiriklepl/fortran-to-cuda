"""One-shot execution of an explicitly proved original source window.

This emitter does not discover windows or authorize storage.  It consumes a
registered source-position proof and reuses reached unit emission, including
native coherence hooks and failure handling.  The original prefix and excluded
tail stay in their original lexical owner.
"""

from __future__ import annotations

from compiler.frontend.reached_windows import ReachedWindowProof, _single_if_action_source, plan_reached_window
from compiler.frontend.source_effects import _children, _part
from compiler.ir import CompilationError
from compiler.scopes.lexical import LexicalOwner
from compiler.scopes.provenance import source_id
from compiler.scopes.segments import fortran_lines, statement_span
from compiler.scopes.source import _name, _span


def _close_selection(point):
    selected = point.get("selected_node_ids")
    if (not isinstance(selected, list) or not 1 <= len(selected) <= 256
            or any(not isinstance(identity, str) for identity in selected)
            or len(set(selected)) != len(selected) or selected[0] != point.get("node_id")):
        raise CompilationError("reached closure lacks its exact ordered original selection")
    return tuple(selected)


class MappedReachedControl:
    """One source plan for both a lexical owner and its original-body child."""

    def reached_record(self):
        return self.builder.reached_plan_record()

    def allow_child_continuation(self):
        record = self.reached_record()
        return record is None or not record.get("planned_closures", {}).get("enabled")

    def mapped_child(self, node):
        record = self.reached_record()
        if record is None or hasattr(node, "fort_inline_region"):
            return False
        originals = self.builder.inline.original_selection((node,))
        _graph, selected = self.builder.analysis._selected_source(self.routine.qualified, originals)
        mapped = record["planned_closures"]["mapped_child_plans"]
        if len(selected) != 1 or selected[0] not in {item["caller_node_id"] for item in mapped}:
            return False
        path, child = self.builder.bind_reached_child(node)
        from compiler.scopes.module_owner import borrow_module
        return borrow_module(self, node, reached=(path, child))

    def planned_source(self, nodes):
        record = self.reached_record()
        planned = record.get("planned_closures", {}) if record is not None else {}
        if not planned.get("enabled") or all(type(node).__name__ == "Comment" for node in nodes):
            return False
        originals = self.builder.inline.original_selection(tuple(nodes))
        graph, selected = self.builder.analysis._selected_source(self.routine.qualified, originals)
        cuts = {point["node_id"]: point for point in planned["close_points"]}
        if selected and selected[0] in cuts:
            if tuple(selected) != _close_selection(cuts[selected[0]]):
                raise CompilationError("reached close cuts a partial original joined selection")
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
        if self.reached_record() is None:
            return super().boundary(nodes, reason, force_unconditional=force_unconditional)
        originals = self.builder.inline.original_selection(tuple(nodes))
        _graph, identities = self.builder.analysis._selected_source(self.routine.qualified, originals)
        super().boundary(nodes, reason, force_unconditional=force_unconditional)
        self.boundaries[-1]["original_node_ids"] = list(identities)

    def validate_mapped_plan(self):
        record = self.reached_record()
        if record is None:
            return
        planned = record["planned_closures"]
        cuts = tuple(_close_selection(point) for point in planned["close_points"])
        actual = tuple(tuple(boundary["original_node_ids"]) for boundary in self.boundaries)
        if actual != cuts:
            raise CompilationError("mapped child staged closes differ from its exact registered reached plan")
        active = set(planned["active_node_ids"])
        for unit in self.units:
            originals = self.builder.inline.original_selection(unit.nodes)
            _graph, identities = self.builder.analysis._selected_source(self.routine.qualified, originals)
            if not set(identities) <= active:
                raise CompilationError("mapped child numerical unit crosses an original reached close")


class ReachedWindowOwner(MappedReachedControl, LexicalOwner):
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
        self.prefix_eligible = _name("fort_prefix_eligible_", proof.identity)
        self.trace_owner = source_id(proxy, "owner", lifetime="one-shot reached window",
                                     reached_window_proof=proof.identity)

    def validate_plan(self, nodes):
        planned = self.window_proof.public().get("planned_closures", {})
        if not planned.get("enabled"):
            return
        for member in self.members.values():
            if member is not self:
                if getattr(member.builder, "_mapped_reached_child", None) is None:
                    raise CompilationError("planned reached child lacks exact mapped authority")
                member.validate_mapped_plan()
        graph = self.builder.analysis.structure(self.routine.qualified)
        known = {_close_selection(point) for point in planned["close_points"]}
        extra = []
        for boundary in self.boundaries:
            identities = tuple(boundary["original_node_ids"])
            if identities not in known:
                originals = tuple(original for identity in identities for original in graph.source_nodes(identity))
                extra.append((originals, boundary["reason"]))
        if extra:
            forced = (*self.window_proof._planned_close_before, *extra)
            proof = plan_reached_window(self.builder.analysis, self.routine.qualified, self.window_originals,
                                        self.window_proof._roots, self.builder.facts, close_before=forced,
                                        numerical_provider=self.window_proof._numerical_provider)
            self.window_proof = proof
            self.builder._reached_window_binding = proof, self.window_originals
            planned = proof.public()["planned_closures"]
            for member in self.members.values():
                if member is self:
                    continue
                old = member.builder._mapped_reached_child
                child = proof.child_plan(member.builder._mapped_reached_path, old.original_call)
                if child.public() != old.public():
                    raise CompilationError("generation close changes an already staged mapped child; owner remains native")
                member.builder._reached_window_binding = proof, self.window_originals
                member.builder._mapped_reached_child = child
        cuts = tuple(_close_selection(point) for point in planned["close_points"])
        actual = tuple(tuple(boundary["original_node_ids"]) for boundary in self.boundaries)
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
        declarations = [f"integer({self.kind}) :: {self.context}",
                        f"logical :: {self.enabled}, {self.spent}, {self.prefix_eligible}",
                        *[f"integer({self.kind}) :: fort_buffer_{number}"
                          for _binding, number in self.resources.values()]]
        initialize = [f"{self.context} = 0", f"{self.enabled} = .false.", f"{self.spent} = .false.",
                      f"{self.prefix_eligible} = .true.",
                      *[f"fort_buffer_{number} = 0" for _binding, number in self.resources.values()]]
        builder.add_edit(self.routine.scope.path, first_execution, first_execution - 1,
                         "\n".join(fortran_lines([*declarations, *initialize])) + "\n", prepend=True)
        prefix = self.window_proof.public()["prefix_eligibility"]
        graph = builder.analysis.structure(self.routine.qualified)
        for point in prefix["disable_points"]:
            originals = tuple(original for identity in _close_selection(point) for original in graph.source_nodes(identity))
            if not originals or [statement_span(originals[0])[0], statement_span(originals[-1])[1]] != point["span"]:
                raise CompilationError("pre-owner eligibility edit lacks its exact original source span")
            conditional = point.get("conditional_action")
            if conditional is not None:
                if len(originals) != 1 or _close_selection(point) != (point["node_id"],):
                    raise CompilationError("conditional prefix edit requires its exact original single-line IF")
                expected, unsafe = _single_if_action_source(builder.analysis, self.routine, graph, point["node_id"])
                if unsafe is not None or conditional != expected:
                    raise CompilationError("conditional prefix edit differs from its registered original source")
                condition, action = originals[0].items
                replacement = [f"if ({condition}) then", f"{self.prefix_eligible} = .false.", str(action), "endif"]
                builder.add_edit(self.routine.scope.path, point["span"][0], point["span"][1],
                                 "\n".join(fortran_lines(replacement)) + "\n")
                continue
            line = point["span"][0]
            builder.add_edit(self.routine.scope.path, line, line - 1,
                             f"{self.prefix_eligible} = .false.\n")
        activation = [f"if (.not. {self.spent}) then", f"{self.spent} = .true.",
                      f"if ({self.prefix_eligible}) then",
                      "block", "use fort_scoped_memory, only: fort_scope_serial_caller",
                      f"{self.enabled} = fort_scope_serial_caller() /= 0", "end block", "endif", "endif"]
        builder.add_edit(self.routine.scope.path, first, first - 1,
                         "\n".join(fortran_lines(activation)) + "\n")
        builder.add_edit(self.routine.scope.path, last + 1, last,
                         "\n".join(fortran_lines(self.close(disable=True,
                             reason="proved reached window end before excluded tail or enclosing backedge",
                             span=(last, last)))) + "\n")

    def run(self, nodes=None):
        """Emit only selected originals; the caller may supply approved facades."""
        self.builder.validate_reached_window()
        self.window_proof.validate(self.builder.analysis, self.routine.qualified, self.window_originals)
        if not self.window_proof.public()["prefix_eligibility"]["activation_path_possible"]:
            # A valid all-native decision needs no synthetic source preparation,
            # protected descriptor association or dormant control edits.
            return False
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
                    "prefix_eligibility": public["prefix_eligibility"],
                    "first_traversal_controls": public["first_traversal_controls"],
                    "entry_facts_identity": public["entry_facts_identity"],
                    "lifecycle": "one invocation; dormant -> active -> spent; no reopening",
                    "activation": "first reached selected source point; spend before eligibility and every guard",
                    "mandatory_close": public["mandatory_close"],
                    "managed_resources": [state["resource"] for state in public["states"]],
                    "definition_proof_identities": public["native_definition_proof_identities"],
                    "planned_closures": public["planned_closures"],
                    "defined_coverage": [{"resource": state["resource"],
                        "on_activation": state["initialized"], "on_return": state["coverage_on_return"],
                        "source_sections_on_activation": state.get("source_sections_on_activation"),
                        "return_sections": state["defined_sections_on_return"],
                        "authority": "conditional original reached transfer; return sections never seed registration"}
                        for state in public["states"]],
                    "prefix_execution": "original native evaluation before any active context; registered single-line IF actions retain condition and action once",
                    "allocation_and_alias_authority": "independent admitted facts and reached runtime guards",
                    "automatic_discovery": False,
                }
                scope["ownership"] = {
                    "lifetime": "one-shot original reached window", "planning_mode": "continuation",
                    "retained_resources": list(self.resources),
                    "end_reason": "first reached boundary or end of selected window before excluded tail/backedge",
                }
        return emitted
