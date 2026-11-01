"""Compose original team storage and mutation issuers for one deferred group.

This service does not establish native execution, numerical completion, payload
access, definitions, or aliases. The unchanged complete original team and full
storage closure remain independently authenticated by their existing issuers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256

from compiler.frontend.descriptor_teams import DescriptorTeamProof, prove_descriptor_team
from compiler.frontend.native_group_structure import NativeGroupStructure
from compiler.frontend.source_functions import _hash
from compiler.frontend.storage_mutation_effects import (
    StorageMutationEffects,
    _preserves_record,
    prove_storage_mutation_effects,
)
from compiler.frontend.structured_effects import _thaw
from compiler.ir import CompilationError

_ISSUER = object()
DESCRIPTOR_WINDOW_STORAGE_GROUP_VERSION = 13


def _group(analysis, routine, graph, identity):
    if (analysis._require_original(routine.qualified) is not routine
            or graph is not analysis.structure(routine.qualified)):
        raise CompilationError("descriptor storage group requires its exact original analysis and graph")
    group = graph.native_groups.get(identity)
    node = graph.nodes.get(identity)
    if (type(group) is not NativeGroupStructure or node is None or node.kind != "boundary"
            or group.node_id != identity
            or _thaw(node.details.get("deferred_native_group")) != group.public()
            or not group.matches(graph.source_nodes(identity))):
        raise CompilationError("descriptor storage group requires the complete original deferred group")
    selected_graph, selected = analysis._selected_source(routine.qualified, group.original_nodes)
    if selected_graph is not graph or selected != (identity,):
        raise CompilationError("descriptor storage group lost exact original source membership")
    return group


@dataclass(frozen=True, slots=True)
class DescriptorStorageGroup:
    """Ephemeral composition; the retained existing tokens grant storage facts."""
    _analysis: object = field(repr=False)
    _routine: object = field(repr=False)
    _graph: object = field(repr=False)
    _identity: str
    _group: object = field(repr=False)
    _team: object = field(repr=False)
    _storage: object = field(repr=False)
    _issuer: object = field(repr=False)

    def validate(self):
        attachments = tuple((name, getattr(self, name)) for name in
            ("_analysis", "_routine", "_graph", "_group", "_team", "_storage", "_issuer"))
        identity = self._identity
        if self._issuer is not _ISSUER or type(self) is not DescriptorStorageGroup:
            raise CompilationError("descriptor storage group requires its compiler composition")
        group = _group(self._analysis, self._routine, self._graph, self._identity)
        if (group is not self._group or type(self._team) is not DescriptorTeamProof
                or type(self._storage) is not StorageMutationEffects
                or self._team._analysis is not self._analysis
                or self._storage._analysis is not self._analysis):
            raise CompilationError("descriptor storage group changed its independent issuers")
        team_attachments = tuple((name, getattr(self._team, name)) for name in
            ("identity", "procedure", "structured_identity", "selected_node_ids", "_analysis", "_originals", "_record", "_issuer"))
        storage_attachments = tuple((name, getattr(self._storage, name)) for name in
            ("identity", "_analysis", "_procedure", "_record", "_issuer", "_expression", "_selection", "_joined_storage"))
        self._team.validate(self._analysis, self._routine.qualified, (self._identity,))
        self._storage.validate(self._analysis, self._routine.qualified)
        if (self._storage._selection != (self._identity,)
                or not _preserves_record(self._storage._record)):
            raise CompilationError("descriptor storage group has allocation mutation, escape or unknown closure")
        if _group(self._analysis, self._routine, self._graph, self._identity) is not group:
            raise CompilationError("descriptor storage group attachment changed before publication")
        self._analysis.verify_publication()
        if (self._identity != identity or any(getattr(self, name) is not value for name, value in attachments)
                or any(getattr(self._team, name) is not value for name, value in team_attachments)
                or any(getattr(self._storage, name) is not value for name, value in storage_attachments)
                or getattr(self._analysis, "_descriptor_team_proofs", {}).get(self._team.identity) is not self._team
                or getattr(self._analysis, "_storage_mutation_effects", {}).get(self._storage.identity) is not self._storage
                or sha256(self._team._record.encode()).hexdigest() != self._team.identity
                or _hash({key: value for key, value in self._storage.public().items() if key != "identity"}) != self._storage.identity):
            raise CompilationError("descriptor storage group issuer attachment changed at final publication")
        return self

    def public(self):
        return {"version": 1, "procedure": self._routine.qualified,
            "source_graph": self._graph.identity, "original_node_id": self._identity,
            "group_identity": self._group.identity, "team_storage": self._team.public(),
            "storage_mutation": self._storage.public(), "complete_original_group_required": True,
            "descriptor_evidence_only": True, "entry_definition_authority": False,
            "payload_access_authorized": False, "alias_authority": False,
            "execution_authorized": False, "numerical_completion_authorized": False}


def original_storage_group(analysis, routine, graph, identity):
    group = _group(analysis, routine, graph, identity)
    team = prove_descriptor_team(analysis, routine.qualified, group.original_nodes)
    storage = prove_storage_mutation_effects(analysis, routine.qualified, selection=(identity,), joined_storage=team)
    return DescriptorStorageGroup(analysis, routine, graph, identity, group, team, storage, _ISSUER).validate()
