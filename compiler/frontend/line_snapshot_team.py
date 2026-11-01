"""A complete original combined OpenMP team containing one private line unit.

This adapter retains the line issuer's exact joined-completion authority. It
supplies source-task inventory, without inventing a tile-width/MIN certificate,
scalar initialization, array initialization, alias, pricing or mapping authority.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256

from compiler.frontend.line_snapshot import OriginalLineSnapshotProof, prove_line_snapshot
from compiler.ir import CompilationError

_ISSUER = object()


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _record(proof):
    proof.validate()
    value = proof.public()
    return {
        "version": 1,
        "kind": "original_line_snapshot_team",
        "procedure": proof._routine.qualified,
        "source_graph": value["source_graph"],
        "source_inputs": value["source_inputs"],
        "selected_node_ids": value["selected_node_ids"],
        "unit_proof_identities": [proof.identity],
        "transverse_domain": value["transverse_domain"],
        "complete_combined_team": True,
        "completion": "exact underlying original line proof and joined-completion token",
        "source_order": "complete original line snapshot before original serial consumer",
        "fields": [{"descriptor": binding.public(), "written": written} for binding, written in proof.fields],
        "scalars": [binding.public() for binding in proof.scalars],
        "scratch": [proof.scratch_binding.public()],
        "definitions_authorized": False,
        "placement_authorized": False,
        "pricing_authorized": False,
        "cooperative_mapping_authorized": False,
    }


@dataclass(frozen=True, slots=True)
class OriginalLineSnapshotTeamProof:
    identity: str
    units: tuple = field(repr=False, compare=False)
    fields: tuple = field(repr=False, compare=False)
    scalars: tuple = field(repr=False, compare=False)
    scratch_bindings: tuple = field(repr=False, compare=False)
    _analysis: object = field(repr=False, compare=False)
    _routine: object = field(repr=False, compare=False)
    _nodes: tuple = field(repr=False, compare=False)
    _record: str = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)

    def public(self):
        return {"identity": self.identity, **json.loads(self._record)}

    def validate(self, analysis=None, routine=None, nodes=None):
        analysis = self._analysis if analysis is None else analysis
        routine = self._routine if routine is None else routine
        nodes = self._nodes if nodes is None else tuple(nodes)
        if (
            self._issuer is not _ISSUER
            or analysis is not self._analysis
            or routine is not self._routine
            or len(nodes) != len(self._nodes)
            or any(a is not b for a, b in zip(nodes, self._nodes, strict=True))
            or getattr(analysis, "_line_snapshot_team_proofs", {}).get(self.identity) is not self
            or self.identity != sha256(self._record.encode()).hexdigest()
            or len(self.units) != 1
            or type(self.units[0]) is not OriginalLineSnapshotProof
        ):
            raise CompilationError("line team requires its registered exact original complete-team proof")
        proof = self.units[0]
        if (
            proof._analysis is not analysis
            or proof._routine is not routine
            or len(proof._nodes) != len(nodes)
            or any(a is not b for a, b in zip(proof._nodes, nodes, strict=True))
        ):
            raise CompilationError("line team lost its exact original source unit")
        proof.validate()
        if (
            _encoded(_record(proof)) != self._record
            or len(self.fields) != len(proof.fields)
            or any(a is not b or aw != bw for (a, aw), (b, bw) in zip(self.fields, proof.fields, strict=True))
            or len(self.scalars) != len(proof.scalars)
            or any(a is not b for a, b in zip(self.scalars, proof.scalars, strict=True))
            or len(self.scratch_bindings) != 1
            or self.scratch_bindings[0] is not proof.scratch_binding
        ):
            raise CompilationError("line team changed original inventory or completion authority")
        return self


def prove_line_snapshot_team(analysis, routine, original_nodes):
    nodes = tuple(original_nodes)
    proof = prove_line_snapshot(analysis, routine, nodes)
    record = _encoded(_record(proof))
    identity = sha256(record.encode()).hexdigest()
    registry = getattr(analysis, "_line_snapshot_team_proofs", None)
    if registry is None:
        registry = analysis._line_snapshot_team_proofs = {}
    if identity not in registry:
        registry[identity] = OriginalLineSnapshotTeamProof(
            identity,
            (proof,),
            tuple(proof.fields),
            tuple(proof.scalars),
            (proof.scratch_binding,),
            analysis,
            routine,
            nodes,
            record,
            _ISSUER,
        )
    return registry[identity].validate(analysis, routine, nodes)
