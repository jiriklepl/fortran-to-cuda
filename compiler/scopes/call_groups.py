"""Owner-local original CALL groups through the existing checked graph engine."""
from compiler.frontend.source_effects import _children
from compiler.frontend.source_span_exit import prove_original_sequence_exit
from compiler.ir import CompilationError
from compiler.scopes.closed_selection import ClosedIntervalOwner, OriginalSelectionOwner
from compiler.scopes.generation_transaction import GenerationTransaction
from compiler.scopes.graph import GraphOwner
from compiler.scopes.graph_epochs import finalize_call_group_epoch, prepare_call_group_epoch


class OwnerLocalCallGroup(GraphOwner, OriginalSelectionOwner):
    def __init__(self, builder, originals):
        ClosedIntervalOwner._validate_edit_boundaries(builder, originals)
        super().__init__(builder, originals)
        prepare_call_group_epoch(self)
        self.selection_context = {**self.selection_context,
            "activation": "fresh domain inside the reached original CALL group; no prefix definitions",
            "lifetime": "publish and close immediately after last original CALL before branch exit",
            "fallback": "only current child or remaining source suffix once; never replay committed prefix"}

    def scan(self, nodes, depth=0):
        if len(nodes) != len(self.closed_selection) or any(
                a is not b for a, b in zip(nodes, self.closed_selection, strict=True)):
            raise CompilationError("owner-local call group selection changed")
        return super().scan(nodes, depth)

    def validate_plan(self, nodes):
        self.local_exit.validate(self.builder.analysis, self.routine.qualified, self.closed_selection)
        for member in self.members.values():
            originals = self.closed_selection if member is self else tuple(_children(member.routine.execution))
            ClosedIntervalOwner._validate_edit_boundaries(self.builder, originals, routine=member.routine)
        if self.boundaries or len(self.members) < 2:
            raise CompilationError("owner-local call group requires checked original child continuations")
        finalize_call_group_epoch(self)
        super().validate_plan(nodes)

    def emit_control(self, nodes):
        self.local_exit.validate(self.builder.analysis, self.routine.qualified, self.closed_selection)
        return super().emit_control(nodes)


def _candidates(builder):
    selection = builder._native_ancestor_selection.validate(builder.analysis)
    selected_roots = {item["procedure"] for item in selection.public()["device_roots"]}
    graph = builder.analysis.structure(builder.entry.qualified)
    for sequence in graph.nodes.values():
        if sequence.kind != "sequence":
            continue
        calls = []
        for identity in reversed(sequence.children):
            if graph.nodes[identity].kind != "call":
                break
            original, = graph.source_nodes(identity)
            call = builder.execution_source_call(builder.entry, original)
            if call.procedure not in selected_roots:
                break
            calls.append(original)
        if len(calls) < 2:
            continue
        originals = tuple(reversed(calls))
        try:
            prove_original_sequence_exit(builder.analysis, builder.entry.qualified, originals)
        except CompilationError:
            continue
        yield originals


def run_owner_local_call_groups(builder):
    """The request chooses discovery, never a worker, effects or entry facts."""
    for originals in tuple(_candidates(builder)):
        try:
            with GenerationTransaction(builder) as transaction:
                owner = OwnerLocalCallGroup(transaction.builder, originals)
                if owner.run(originals):
                    transaction.builder.scopes[-1]["graph_root_kind"] = "owner-local original CALL group"
                    transaction.builder.scopes[-1]["owner_local_call_group"] = owner.position_epoch.public()
                    transaction.adopt(transaction.builder)
        except CompilationError as error:
            builder.boundaries.append({"procedure": builder.entry.qualified,
                "reason": "owner-local CALL group unavailable: " + str(error),
                "execution": "unchanged original CALL group; no partial publication"})
    return builder.finish(materialize_native_effects=False)
