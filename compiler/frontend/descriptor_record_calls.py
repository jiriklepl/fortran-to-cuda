"""Exact owned record CALL composed through existing original team and SAME.

The region token describes only bounded pointer association. The existing team
and mutation issuers independently authenticate synchronization and scan every
original executable statement. No payload/coherence or execution API is added.
"""
from dataclasses import dataclass
from types import MappingProxyType

from compiler.frontend.descriptor_associations import _attached, _attachments
from compiler.frontend.descriptor_teams import DescriptorTeamProof, prove_descriptor_team
from compiler.frontend.storage_mutation_effects import (
    StorageMutationEffects,
    _preserves_record,
    prove_storage_mutation_effects,
)
from compiler.frontend.storage_record_regions import (
    OwnedRecordRegion,
    RecordRegionCheckpoint,
    prove_owned_record_region,
)
from compiler.ir import CompilationError

DESCRIPTOR_WINDOW_RECORD_REGION_VERSION = 27
_ISSUER = object()


@dataclass(frozen=True, slots=True)
class DescriptorRecordCall:
    _region: object
    _team: object
    _storage: object
    _issuer: object

    def public(self):
        return {"version": 1, "owned_record_region": self._region.public(),
                "team_storage": self._team.public(), "storage_mutation": self._storage.public(),
                "complete_original_body_and_hidden_effects_scanned": True,
                "complete_original_callee_specification_and_lifecycle_scanned": True,
                "descriptor_evidence_only": True, "payload_access_authorized": False,
                "execution_authorized": False, "entry_definition_authority": False,
                "numerical_completion_authorized": False, "alias_authority": False}

    def validate(self):
        if type(self) is not DescriptorRecordCall or self._issuer is not _ISSUER:
            raise CompilationError("descriptor record call requires exact original composition")
        if (type(self._region) is not OwnedRecordRegion or type(self._team) is not DescriptorTeamProof
                or type(self._storage) is not StorageMutationEffects):
            raise CompilationError("descriptor record call requires exact independent original issuers")
        region, team, storage = self._region, self._team, self._storage
        analysis = region._analysis
        if getattr(analysis, "_source_session", None) is not None:
            region.validate(analysis)
            team.validate(analysis, region._roles[1].qualified, team.selected_node_ids)
            storage.validate(analysis)
            if (team._record_region is not region or storage._joined_storage is not team
                    or storage._procedure != region._roles[1].qualified
                    or storage._selection != tuple(region.public()["complete_original_body"]["selected_node_ids"])
                    or not _preserves_record(storage._record)):
                raise CompilationError("descriptor record call has unknown mutation, escape or incomplete original closure")
            return self
        states = tuple((value, _attachments(value)) for value in (self, self._region, self._team, self._storage))
        region, team, storage = self._region, self._team, self._storage
        analysis = region._analysis
        storage._require_registration(analysis, storage._procedure, storage._expression,
                                      storage._selection, storage._joined_storage)
        if type(storage._record) is not MappingProxyType:
            raise CompilationError("descriptor record call requires its immutable registered storage record")
        closure = tuple(analysis.numerical_helpers[member["procedure"]] for member in storage._record["closure"])
        checkpoint = RecordRegionCheckpoint(analysis, region, closure, calls=storage._calls,
                                            terminating_calls=storage._terminating_calls)
        team_registry, storage_registry = analysis._descriptor_team_proofs, analysis._storage_mutation_effects
        for registry in (team_registry, storage_registry):
            if type(registry) is not dict or any(type(key) is not str for key in registry):
                raise CompilationError("descriptor record call requires original compiler issuer registries")
        registrations = tuple((registry, tuple(registry.items())) for registry in (team_registry, storage_registry))
        region.validate(analysis)
        team.validate(analysis, region._roles[1].qualified, team.selected_node_ids)
        storage.validate(analysis)
        if (team._record_region is not region or storage._joined_storage is not team
                or storage._procedure != region._roles[1].qualified
                or storage._selection != tuple(region.public()["complete_original_body"]["selected_node_ids"])
                or not _preserves_record(storage._record)):
            raise CompilationError("descriptor record call has unknown mutation, escape or incomplete original closure")
        analysis.verify_publication()
        analysis.inputs.verify_live()
        checkpoint.require()
        if (any(type(key) is not str for registry, _entries in registrations for key in registry)
                or any(len(registry) != len(entries) or any(registry.get(key) is not token for key, token in entries)
                       for registry, entries in registrations)
                or analysis._descriptor_team_proofs is not team_registry or team_registry.get(team.identity) is not team
                or analysis._storage_mutation_effects is not storage_registry or storage_registry.get(storage.identity) is not storage
                or any(not _attached(value, state) for value, state in states)):
            raise CompilationError("descriptor record call consuming attachment changed")
        return self


def original_record_call(analysis, caller, call, numeric):
    region = prove_owned_record_region(analysis, caller.qualified, call, numeric)
    callee = region._roles[1]
    team = prove_descriptor_team(analysis, callee.qualified, region._roles[6], record_region=region)
    _graph, selected = analysis._selected_source(callee.qualified, region._roles[8])
    storage = prove_storage_mutation_effects(analysis, callee.qualified,
        selection=selected, joined_storage=team)
    return DescriptorRecordCall(region, team, storage, _ISSUER).validate()
