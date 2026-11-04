"""Reached selected regions reuse the existing checked child-route proof.

The root selection is only a source/edit boundary. Each admitted operation and
child still needs its own effect/storage proof; every reached retirement needs
complete synchronous noescape lifecycle evidence. No whole-child descriptor
claim is inferred from a preserving prefix.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256

from fparser.two.utils import walk

from compiler.frontend.source_effects import _children, _kind
from compiler.ir import CompilationError
from compiler.scopes.closed_selection import ClosedIntervalOwner, OriginalSelectionOwner, _require_original_empty_body
from compiler.scopes.graph import GraphOwner
from compiler.scopes.segments import statement_span
from compiler.scopes.source import _name

_ISSUER = object()
_EXITS = frozenset({"Return_Stmt", "Exit_Stmt", "Cycle_Stmt", "Goto_Stmt", "Computed_Goto_Stmt",
                    "Arithmetic_If_Stmt", "Entry_Stmt"})


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _selection_record(owner, originals, cut):
    analysis, graph = owner.builder.analysis, owner.graph
    current, selected = analysis._selected_source(owner.routine.qualified, originals)
    if current is not graph or not selected:
        raise CompilationError("hierarchical region requires its original source selection")
    parents = [node for node in graph.nodes.values() if node.kind == "sequence"
               and selected[0] in node.children
               and tuple(node.children[node.children.index(selected[0]):
                                       node.children.index(selected[0])+len(selected)]) == selected]
    if len(parents) != 1:
        raise CompilationError("hierarchical region requires one contiguous original sequence")
    parent = parents[0]
    end = parent.children.index(selected[0]) + len(selected)
    if end >= len(parent.children) or parent.children[end] != cut:
        raise CompilationError("hierarchical region changed its exact original final retirement frontier")
    if any(_kind(node) in _EXITS for original in originals for node in walk(original)):
        raise CompilationError("hierarchical root region cannot bypass its mandatory cleanup")
    ClosedIntervalOwner._validate_edit_boundaries(owner.builder, originals)
    first, last = statement_span(originals[0])[0], statement_span(originals[-1])[1]
    text = "".join(analysis.source_text(owner.routine.scope.path).splitlines(keepends=True)[first-1:last])
    return {"version": 1, "kind": "original_hierarchical_root_selection", "procedure": owner.routine.qualified,
            "source_graph": graph.identity, "parent_sequence": parent.id,
            "selected_node_ids": list(selected), "cut_before": cut, "source_span": [first, last],
            "original_source_sha256": sha256(text.encode()).hexdigest(),
            "activation": "fresh lazy domain at each reached original selection; no prefix definitions",
            "cleanup": "publish and wait; retire the domain immediately after the selected original span",
            "outside_selection": "unchanged original source; no context or handles retained across the selection end",
            "payload_access_authorized": False, "descriptor_stability_authorized": False,
            "entry_definition_authority": False, "execution_authorized": False,
            "cross_exit_retention_authorized": False, "alias_authority": False}


@dataclass(frozen=True, slots=True)
class SelectedRoot:
    identity: str
    selected: tuple
    cut_before: str
    _owner: object = field(repr=False, compare=False)
    _originals: tuple = field(repr=False, compare=False)
    _record: str = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)

    def public(self):
        return {"identity": self.identity, **json.loads(self._record)}

    def validate(self, owner):
        draft = getattr(owner.builder, "_conditional_return_draft", None)
        if (self._issuer is not _ISSUER or self._owner is not owner or owner.parent is not None
                or getattr(owner, "hierarchical_selection", None) is not self
                or draft is None or draft.selected_root is not self or draft._selected_root_receipt is not self
                or len(owner.closed_selection) != len(self._originals)
                or any(a is not b for a, b in zip(owner.closed_selection, self._originals, strict=True))):
            raise CompilationError("hierarchical selection lost its exact original owning region")
        record = _selection_record(owner, self._originals, self.cut_before)
        if (tuple(record["selected_node_ids"]) != self.selected or _encoded(record) != self._record
                or sha256(self._record.encode()).hexdigest() != self.identity):
            raise CompilationError("hierarchical selection source, control or cleanup changed")
        return self


class HierarchicalRegionOwner(GraphOwner, OriginalSelectionOwner):
    """One original root span, with complete checked routes in its children."""

    def __init__(self, builder, originals, cut):
        super().__init__(builder, originals)
        record = _encoded(_selection_record(self, self.closed_selection, cut))
        selected = tuple(json.loads(record)["selected_node_ids"])
        self.hierarchical_selection = SelectedRoot(sha256(record.encode()).hexdigest(), selected, cut,
                                                   self, self.closed_selection, record, _ISSUER)
        self.retention_allowed = _name("fort_retention_allowed_", self.routine.qualified)
        self.return_calls, self.return_conditions = [], []
        self.selection_context = {**self.selection_context,
            "activation": "fresh lazy domain at each reached original selection; no prefix definitions",
            "lifetime": "checked child regions and synchronous retirement cuts; mandatory selected-end publication",
            "fallback": "only the current original operation or remaining original suffix; no prefix replay"}
        from compiler.scopes.conditional_return import prepare_selected
        prepare_selected(self, self.hierarchical_selection)

    def validate_plan(self, nodes):
        self.hierarchical_selection.validate(self)
        for member in self.members.values():
            originals = self.closed_selection if member is self else tuple(_children(member.routine.execution))
            if member is not self and not originals:
                _require_original_empty_body(member.builder.analysis, member.routine)
                # The authenticated original has no executable source interval.
                # Its entry, CALL, return and graph checks still run below.
                continue
            ClosedIntervalOwner._validate_edit_boundaries(self.builder, originals, routine=member.routine)
        return GraphOwner.validate_plan(self, nodes)

    def emit_control(self, nodes):
        self.hierarchical_selection.validate(self)
        OriginalSelectionOwner.emit_control(self, nodes)
        last = self.selection_context["last_line"]
        endings = [edit for edit in self.builder.edits[self.routine.scope.path]
                   if edit[:2] == (last+1, last)]
        if len(endings) != 1:
            raise CompilationError("hierarchical region requires one exact emitted selected-end retirement")
        self.emission.selected_end_retirement = {
            "version": 1, "kind": "emitted_selected_end_domain_retirement", "source": str(self.routine.scope.path),
            "insertion_span": [last+1, last], "context": self.context,
            "replacement_sha256": sha256(endings[0][2].encode()).hexdigest()}

    def _emit_position_epoch(self, nodes, checkpoint, *, native_cleanup=True):
        accepted = GraphOwner._emit_position_epoch(self, nodes, checkpoint, native_cleanup=native_cleanup)
        if accepted and not getattr(self.emission, "owned_hierarchical_record_finished", False):
            self._finish_hierarchical_record()
        return accepted

    def _finish_hierarchical_record(self):
        if self.builder.scopes[-1].get("owner_variant") is not None:
            self.builder.scopes[-1]["selected_end_retirement"] = dict(self.emission.selected_end_retirement)

    def _finish_prepared_graph_record(self):
        """Retain selected-end publication before the owned image is sealed."""
        GraphOwner._finish_prepared_graph_record(self)
        self._finish_hierarchical_record()
        self.emission.owned_hierarchical_record_finished = True


def _has_nested_calls(owner, selected):
    """Search hint only: leaf-only pairs already use the direct epoch route."""
    from compiler.frontend.call_bindings import resolve_original_call
    from compiler.scopes.graph_epochs import _call_nodes

    try:
        for identity in _call_nodes(owner, selected):
            original, = owner.graph.source_nodes(identity)
            scope = owner.builder.analysis.source_scope_for(original, owner.routine.scope)
            call = resolve_original_call(owner.builder.analysis, scope, original)
            routine = owner.builder.analysis._require_original(call.procedure)
            for node in walk(routine.execution):
                if _kind(node) != "Call_Stmt":
                    continue
                try:
                    nested_scope = owner.builder.analysis.source_scope_for(node, routine.scope)
                    nested = resolve_original_call(owner.builder.analysis, nested_scope, node)
                    owner.builder.analysis._require_original(nested.procedure)
                except CompilationError:
                    continue  # Intrinsics such as MOVE_ALLOC are not checked child regions.
                return True
    except CompilationError:
        pass  # The original complete candidate path remains eligible.
    return False


def try_hierarchical_regions(builder):
    """Use the existing complete candidate iterator; ranking grants no proof."""
    if builder._native_ancestor_selection is not None:
        return False
    roots = []
    for root, fact in builder.facts["captures"].items():
        if fact.get("lifetime") != "selected_window":
            continue
        binding = builder.visible_binding(builder.entry, root)
        if binding.rank:
            builder.capture_fact(binding)
            roots.append(root)
    if not roots:
        return False
    from compiler.scopes.generation_transaction import GenerationTransaction
    from compiler.scopes.graph_epoch_candidates import candidates

    scout = GraphOwner(builder)
    attempts = []
    for selected, cut, priority in candidates(scout, tuple(roots)):
        if not _has_nested_calls(scout, selected):
            continue
        originals = tuple(node for identity in selected for node in scout.graph.source_nodes(identity))
        attempt = {**priority, "selected_node_ids": list(selected), "cut_before": cut, "outcome": "preparing"}
        attempts.append(attempt)
        try:
            with GenerationTransaction(builder) as transaction:
                candidate = transaction.builder
                owner = HierarchicalRegionOwner(candidate, originals, cut)
                nodes = candidate.inline.prepare(owner.closed_selection)
                candidate.generated.update(candidate.inline.generated)
                candidate.numerical_ir.update(candidate.inline.ir)
                candidate.numerical_reasons.update({name: None for name in candidate.inline.regions})
                accepted = owner.run(nodes)
                if accepted:
                    attempt["outcome"] = "admitted checked hierarchical region"
                    candidate.graph_owner_attempt = {
                        "procedure": candidate.entry.qualified, "admitted": True,
                        "hierarchical_region_search": {"attempts": attempts,
                            "candidate_policy": scout.position_epoch_search},
                        "boundaries": [dict(item) for member in owner.members.values() for item in member.boundaries]}
                    transaction.adopt(candidate)
                    return True
                attempt["outcome"] = "no admitted numerical tasks"
        except CompilationError as error:
            attempt.update(outcome="declined", reason=str(error))
    builder.hierarchical_region_attempt = {"attempts": attempts, "candidate_policy": scout.position_epoch_search}
    return False
