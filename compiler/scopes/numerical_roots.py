"""Admit original complete numerical tasks when a caller graph has no work.

An unsupported native prefix need not prevent compiling an independent original
parallel team. These are original task nodes with checked effects, rather than
enumerated multi-statement source windows. Original branch/prefix evaluation is
retained and no invocation-entry state is assumed at a later source point.
"""

from compiler.ir import CompilationError
from compiler.scopes.closed_selection import ClosedNumericalOwner, candidates
from compiler.scopes.generation_transaction import GenerationTransaction
from compiler.scopes.graph import GraphOwner


class NumericalTaskOwner(GraphOwner, ClosedNumericalOwner):
    def validate_plan(self, nodes):
        ClosedNumericalOwner.validate_plan(self, nodes)
        GraphOwner.validate_plan(self, nodes)


def admit_numerical_roots(builder):
    """Use the ordinary numerical proof for each complete original team."""
    admitted = False
    for originals, prefix, controls in tuple(candidates(builder)):
        try:
            with GenerationTransaction(builder) as transaction:
                candidate = transaction.builder
                owner = NumericalTaskOwner(candidate, originals, prefix, controls)
                if owner.run(originals):
                    candidate.scopes[-1]["graph_root_kind"] = "original complete numerical task"
                    transaction.adopt(candidate)
                    admitted = True
        except CompilationError as error:
            builder.boundaries.append({
                "procedure": builder.entry.qualified,
                "reason": "original numerical task unavailable: " + str(error),
                "kind": "candidate_rejection",
                "execution": "unchanged original task",
            })
    return admitted
