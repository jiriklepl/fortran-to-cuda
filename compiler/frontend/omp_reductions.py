"""Original OpenMP clauses can supply a bounded reduction-order contract.

This module proves source semantics only. It neither relaxes ordinary native
completion tokens nor supplies a generated executor or placement estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field


from compiler.frontend.structured_effects import _thaw
from compiler.ir import CompilationError, ScalarType

OPENMP_REDUCTION_PROOF_VERSION = 1
_UNSAFE = frozenset({"pointer", "allocatable", "optional", "target", "volatile", "asynchronous"})
_TYPES = {("integer", 4, 0): ScalarType.INTEGER, ("real", 4, 0): ScalarType.REAL32,
          ("real", 8, 0): ScalarType.REAL}
_SIGNATURES = {value: key for key, value in _TYPES.items()}


@dataclass(frozen=True)
class OpenMPReductionProof:
    procedure: str
    structured_identity: str
    identity: str
    selected_node_ids: tuple[str, ...]
    available: bool
    reason: str | None
    _record: object = field(repr=False, compare=False)
    _originals: tuple[object, ...] = field(repr=False, compare=False)

    def public(self):
        return {"schema_version": OPENMP_REDUCTION_PROOF_VERSION, "procedure": self.procedure,
                "structured_identity": self.structured_identity, "proof_identity": self.identity,
                "selected_original_nodes": list(self.selected_node_ids),
                "source_analysis_available": self.available, "execution_supported": False,
                "execution_reason": "source contract only; no generated OpenMP reduction executor or calibration",
                "reason": self.reason, **_thaw(self._record)}

    def validate(self, analysis):
        graph, identities = analysis._selected_source(self.procedure, self._originals)
        if (getattr(analysis, "_omp_reduction_proofs", {}).get(self.identity) is not self
                or graph.identity != self.structured_identity or identities != self.selected_node_ids):
            raise CompilationError("OpenMP reduction proof lacks original complete-group authority")
        return self
