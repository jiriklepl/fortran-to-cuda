"""Checked original source work using the existing typed device transaction.

The tile issuer proves numerical work and complete original team coverage.
This attachment retains that exact authority for graph storage/emission; its
public record grants no initialization, motion, placement or alias authority.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256

from compiler.ir import CompilationError

_ISSUER = object()


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _domain_reads(analysis, routine, team):
    from compiler.frontend.component_bindings import references
    from compiler.frontend.line_snapshot_team import OriginalLineSnapshotTeamProof
    from compiler.frontend.tile_private import _nest
    domain = team.public()["transverse_domain" if type(team) is OriginalLineSnapshotTeamProof else "domain"]
    outer, _body = _nest(team.units[0]._node, len(domain))
    return {binding.root for _node, _name, bounds in outer for binding, _reference in
            references(analysis, routine.scope, bounds) if not binding.rank and binding.dtype == "integer"}


def _prefix_demands(analysis, routine, team):
    from compiler.frontend.line_snapshot_team import OriginalLineSnapshotTeamProof

    if type(team) is not OriginalLineSnapshotTeamProof:
        return ()
    domain_reads = _domain_reads(analysis, routine, team)
    return tuple(binding for binding in team.scalars if binding.root not in domain_reads
                 and "parameter" not in binding.attributes and binding.declaring_scope is routine.scope
                 and binding.name not in routine.arguments)


def _scalar_captures(analysis, routine, team, builder, nodes, prefix):
    from compiler.frontend.reached_scalar_definitions import ReachedScalarDefinitions
    from compiler.scopes.immutable_inputs import _declared

    domain_reads = _domain_reads(analysis, routine, team)
    demands = _prefix_demands(analysis, routine, team)
    if demands:
        if type(prefix) is not ReachedScalarDefinitions:
            raise CompilationError("source task local scalar requires original reached prefix authority")
        prefix.validate(builder, nodes, demands)
    elif prefix is not None:
        raise CompilationError("source task has unrelated scalar prefix authority")
    result = []
    for binding in team.scalars:
        if binding.root in domain_reads:
            authority = {"kind": "original positive-domain integer read", "finite_or_constant_authorized": False}
        elif "parameter" in binding.attributes:
            _declared(analysis, binding)
            authority = {"kind": "original PARAMETER declaration", "finite_or_constant_authorized": False}
        elif any(binding is demanded for demanded in demands):
            authority = {"kind": "original reached scalar prefix definition", "proof": prefix.public(),
                         "finite_or_constant_authorized": False}
        else:
            if builder is None or builder.analysis is not analysis or builder.entry is not routine:
                raise CompilationError("source task scalar requires independent whole-defined capture authority: " + binding.root)
            fact = builder.capture(binding)
            if fact["initialized"] != "whole":
                raise CompilationError("source task scalar capture must already be whole-defined: " + binding.root)
            authority = {"kind": "explicit checked whole-defined scalar entry fact", "fact": {
                key: fact[key] for key in ("storage", "initialized", "allocation_changes", "escapes", "lifetime") if key in fact},
                "finite_or_constant_authorized": False}
        result.append({"resource": binding.root, "descriptor": binding.public(), "authority": authority})
    return result


def _derive(analysis, routine, nodes, team, builder, prefix=None):
    from compiler.frontend.line_snapshot_team import OriginalLineSnapshotTeamProof

    team.validate(analysis, routine, nodes)
    graph, selected = analysis._selected_source(routine.qualified, nodes)
    if not selected:
        raise CompilationError("source device task requires a complete original selection")
    fields = tuple(team.fields)
    if not fields or len({binding.root for binding, _written in fields}) != len(fields):
        raise CompilationError("source device task requires independent canonical field bindings")
    return {"version": 1, "kind": "original_line_snapshot_team" if type(team) is OriginalLineSnapshotTeamProof else "original_tile_private_team", "procedure": routine.qualified,
            "source_graph": graph.identity, "source_inputs": analysis.inputs.identity(),
            "selected_node_ids": list(selected), "team": team.public(),
            "resources": [{"resource": binding.root, "descriptor": binding.public(),
                           "access": "readwrite" if written else "read",
                           "regions": "complete physical root; read before partial writes",
                           "initialization_authorized": False} for binding, written in fields],
            "scalars": [binding.public() for binding in team.scalars],
            "scalar_captures": _scalar_captures(analysis, routine, team, builder, nodes, prefix),
            "scratch": [binding.public() for binding in team.scratch_bindings],
            "transaction": {"runtime": "fort_scope_source_execute_v1", "stream": "supplied",
                            "scratch": "prework lease outside non-reentrant transaction",
                            "submission": "started before first attempt; exact accepted source-kernel count",
                            "failure": "prework original whole span once; attempted failure poisons; no replay"},
            "definitions_authorized": False, "placement_authorized": False, "motion_authorized": False,
            "pricing": "unpriced; explicit sections only", "position": "exact original complete OpenMP team"}


@dataclass(frozen=True, slots=True)
class SourceDeviceTaskProof:
    identity: str
    team: object = field(repr=False, compare=False)
    fields: tuple = field(repr=False, compare=False)
    readable_scalars: tuple = field(repr=False, compare=False)
    scalar_capture_facts: tuple = field(repr=False, compare=False)
    _scalar_prefix: object = field(repr=False, compare=False)
    _builder: object = field(repr=False, compare=False)
    _analysis: object = field(repr=False, compare=False)
    _routine: object = field(repr=False, compare=False)
    _nodes: tuple = field(repr=False, compare=False)
    _record: str = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)

    def validate(self, analysis=None, routine=None, nodes=None):
        analysis = self._analysis if analysis is None else analysis
        routine = self._routine if routine is None else routine
        nodes = self._nodes if nodes is None else tuple(nodes)
        if (self._issuer is not _ISSUER or analysis is not self._analysis or routine is not self._routine
                or len(nodes) != len(self._nodes) or any(a is not b for a, b in zip(nodes, self._nodes, strict=True))
                or getattr(analysis, "_source_device_task_proofs", {}).get(self.identity) is not self):
            raise CompilationError("source device task requires its registered original proof")
        record = _derive(analysis, routine, nodes, self.team, self._builder, self._scalar_prefix)
        if _encoded(record) != self._record:
            raise CompilationError("source device task lost original source, numerical work or field identity")
        if (len(self.fields) != len(self.team.fields) or any(a is not b or written != old_written
                for (a, written), (b, old_written) in zip(self.fields, self.team.fields, strict=True))):
            raise CompilationError("source device task changed its exact original field objects")
        if (len(self.readable_scalars) != len(self.team.scalars)
                or any(a is not b for a, b in zip(self.readable_scalars, self.team.scalars, strict=True))
                or self.scalar_capture_facts != tuple(_encoded(item) for item in record["scalar_captures"])):
            raise CompilationError("source device task scalar readability attachment changed")
        return self

    def public(self):
        return {"identity": self.identity, **json.loads(self._record)}


def source_device_task(analysis, routine, original_nodes, *, builder=None):
    from compiler.frontend.line_snapshot_team import prove_line_snapshot_team
    from compiler.frontend.tile_private import prove_tile_private_team
    from compiler.scopes.segments import directive

    nodes = tuple(original_nodes)
    for proof in getattr(analysis, "_source_device_task_proofs", {}).values():
        if (proof._builder is builder and len(nodes) == len(proof._nodes)
                and all(a is b for a, b in zip(nodes, proof._nodes, strict=True))):
            return proof.validate(analysis, routine, nodes)
    issuer = prove_line_snapshot_team if any((directive(node) or "").startswith("parallel do") for node in nodes) else prove_tile_private_team
    team = issuer(analysis, routine, nodes)
    prefix = None
    demands = _prefix_demands(analysis, routine, team)
    if demands:
        from compiler.frontend.reached_scalar_definitions import prove_reached_scalar_definitions

        if builder is None or builder.analysis is not analysis or builder.entry is not routine:
            raise CompilationError("source task scalar requires independent whole-defined capture authority")
        prefix = prove_reached_scalar_definitions(builder, nodes, demands)
    public = _derive(analysis, routine, nodes, team, builder, prefix)
    record = _encoded(public)
    registry = getattr(analysis, "_source_device_task_proofs", None)
    if registry is None:
        registry = analysis._source_device_task_proofs = {}
    identity = sha256(record.encode()).hexdigest()
    if identity in registry:
        return registry[identity].validate(analysis, routine, nodes)
    proof = SourceDeviceTaskProof(identity, team, tuple(team.fields), tuple(team.scalars),
        tuple(_encoded(item) for item in public["scalar_captures"]), prefix, builder, analysis, routine, nodes, record, _ISSUER)
    registry[proof.identity] = proof
    return proof.validate()
