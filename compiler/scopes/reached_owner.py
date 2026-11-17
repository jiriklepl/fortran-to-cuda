"""One-shot execution of an explicitly proved original source window.

This emitter does not discover windows or authorize storage.  It consumes a
registered source-position proof and reuses reached unit emission, including
native coherence hooks and failure handling.  The original prefix and excluded
tail stay in their original lexical owner.
"""

from __future__ import annotations

from compiler.frontend.reached_windows import _hash, _single_if_action_source
from compiler.ir import CompilationError
from compiler.scopes.segments import fortran_lines, statement_span


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
            point = cuts[selected[0]]
            if tuple(selected) != _close_selection(point):
                raise CompilationError("reached close cuts a partial original joined selection")
            if point.get("conditional_action") is not None:
                self.conditional_boundary(tuple(nodes), originals, graph, point)
            else:
                self.boundary(tuple(nodes), point["reason"], force_unconditional=True)
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

    def conditional_boundary(self, nodes, originals, graph, point):
        """Close inside one authenticated original IF; never infer permission."""
        if (len(nodes) != 1 or len(originals) != 1 or type(originals[0]).__name__ != "If_Stmt"
                or _close_selection(point) != (point["node_id"],)):
            raise CompilationError("conditional reached close requires its exact original single-line IF")
        conditional, unsafe = _single_if_action_source(self.builder.analysis, self.routine, graph, point["node_id"])
        span = statement_span(originals[0])
        if (unsafe is not None or conditional != point["conditional_action"]
                or point["original_source_kind"] != "If_Stmt"
                or point["original_source_sha256"] != _hash([str(originals[0])])
                or point["span"] != list(span) or point["guard"] != list(graph.nodes[point["node_id"]].guard)):
            raise CompilationError("conditional reached close differs from its registered original source")
        condition, action = originals[0].items
        reason = point["reason"]
        self.boundaries.append({"first_line": span[0], "last_line": span[1], "reason": reason,
                                "execution": point["execution"], "conditional_close": str(condition),
                                "conditional_action": conditional, "original_node_ids": [point["node_id"]],
                                "reopen": False})
        replacement = [f"if ({condition}) then", *self.close(reason=reason, span=span), str(action), "endif"]
        self.patches.append((*span, "\n".join(fortran_lines(replacement)) + "\n"))

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
