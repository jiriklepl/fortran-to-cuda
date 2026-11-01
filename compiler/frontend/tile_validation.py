"""Ephemeral reuse of fully checked teams over an immutable source session.

This stores no lowered IR and accepts no public proof record. Mutable analyses
always rederive. Capture/readability facts and emitter IR checks remain outside
this geometry receipt and must still run at each of their existing consumers.
"""
import json
from dataclasses import dataclass, fields

from compiler.frontend.component_bindings import references
from compiler.frontend.sealed_syntax import sealed

_ISSUER = object()


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _context(proof):
    analysis, routine = proof._analysis, proof._routine
    session = getattr(analysis, "_source_session", None)
    if session is None or not analysis._source_syntax.immutable:
        return None
    session.verify(analysis)
    if (session.roots["routines"].get(routine.qualified) is not routine
            or analysis.routines.get(routine.qualified) is not routine):
        return None
    analysis.verify_source()
    inventory = session.execution_nodes(analysis, routine)
    nodes = (*proof._nodes, *(unit._node for unit in proof.units),
             *(phase for unit in proof.units for phase in unit.phases))
    if any(inventory.get(id(node)) is not node or not sealed(node) for node in nodes):
        return None
    graph, selected = analysis._selected_source(routine.qualified, proof._nodes)
    return session, _encoded(analysis._summary_authority()), graph.identity, tuple(selected)


def _state(token):
    return tuple((part.name, getattr(token, part.name)) for part in fields(token))


@dataclass(frozen=True, slots=True)
class _Receipt:
    proof: object
    analysis: object
    context: tuple
    states: tuple
    bindings: tuple
    issuer: object

    def matches(self, proof):
        if self.issuer is not _ISSUER or self.proof is not proof or self.analysis is not proof._analysis:
            return False
        for token, state in self.states:
            if any(getattr(token, name) is not expected for name, expected in state):
                return False
        analysis = self.analysis
        for unit in proof.units:
            if (unit._analysis is not analysis or unit._routine is not proof._routine
                    or getattr(analysis, "_tile_private_proofs", {}).get(unit.identity) is not unit):
                return False
        for binding, declaring_scope, descriptor in self.bindings:
            if (not self.context[0].contains(binding) or binding.declaring_scope is not declaring_scope
                    or declaring_scope.bindings.get(binding.name) is not binding
                    or _encoded(binding.public()) != descriptor):
                return False
        current = _context(proof)
        return (current is not None and current[0] is self.context[0]
                and current[1:] == self.context[1:])


def _registry(analysis, *, create=False):
    if getattr(analysis, "_tile_team_validation_owner", None) is not analysis:
        if not create:
            return {}
        analysis._tile_team_validation_owner = analysis
        analysis._tile_team_validation_receipts = {}
    return analysis._tile_team_validation_receipts


def reuse_validated_team(proof):
    """A hit supplements the caller's exact issuer and token-registry checks."""
    receipt = _registry(proof._analysis).get(id(proof))
    return type(receipt) is _Receipt and receipt.matches(proof)


def reuse_validated_unit(unit):
    """Reuse only the existing whole-team receipt containing this exact unit.

    Unit validation already checked its issuer and exact source/token registry.
    A unit has no separate geometry cache or numerical IR authority here.
    """
    from compiler.frontend.tile_private import TilePrivateTeamProof

    analysis = unit._analysis
    for receipt in _registry(analysis).values():
        if type(receipt) is not _Receipt:
            continue
        team = receipt.proof
        if (type(team) is TilePrivateTeamProof and type(team.units) is tuple
                and any(original is unit for original in team.units)
                and receipt.matches(team)
                and getattr(analysis, "_tile_private_team_proofs", {}).get(team.identity) is team):
            return True
    return False


def remember_validated_team(proof):
    """Called only after complete typed derivation and all identity comparisons."""
    context = _context(proof)
    if context is None:
        return
    analysis, routine = proof._analysis, proof._routine
    bindings = {id(binding): binding for binding, _written in proof.fields}
    for unit in proof.units:
        for binding, _reference in references(analysis, routine.scope, unit._node):
            bindings[id(binding)] = binding
    for binding in (*proof.scalars, *proof.scratch_bindings):
        bindings[id(binding)] = binding
    if any(not context[0].contains(binding) or binding.declaring_scope is None
           or binding.declaring_scope.bindings.get(binding.name) is not binding
           for binding in bindings.values()):
        return
    receipt = _Receipt(proof, analysis, context,
        tuple((token, _state(token)) for token in (proof, *proof.units)),
        tuple((binding, binding.declaring_scope, _encoded(binding.public())) for binding in bindings.values()), _ISSUER)
    registry = _registry(analysis, create=True)
    if len(registry) >= max(1, analysis.procedure_limit):
        registry.pop(next(iter(registry)))
    registry[id(proof)] = receipt
