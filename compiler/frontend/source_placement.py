"""Issued binding between one checked source task and its deferred placement.

The public policy is diagnostic. Only this registered object, the numerical
source proof and the sealed graph together authorize placement dispatch.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256

from compiler.ir import CompilationError

_ISSUER = object()


def _encoded(record):
    return json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _record(member, proof, pricing, task_id):
    proof.validate(member.builder.analysis, member.routine, proof._nodes)
    pricing.validate()
    if not pricing.native_live or pricing.team is not proof.team:
        raise CompilationError("source placement requires its exact native-live source pricing")
    return {"version": 1, "kind": "checked_source_runtime_placement", "policy": "live_source_quote_v2",
            "task_id": task_id, "owner": member.owner.routine.qualified, "procedure": member.routine.qualified,
            "source_proof_identity": proof.identity, "pricing_identity": pricing.identity,
            "canonical_resources": {binding.root: member.canonical(binding).root for binding, _ in proof.fields},
            "registration_slots": {binding.root: member.owner.resources[member.canonical(binding).root][1]
                                   for binding, _ in proof.fields},
            "implementations": ["cpu", "gpu"], "unavailable_target": "cpu",
            "native_endpoint": "typed_host_live", "gpu_endpoint": "source_device_live",
            "execution_authorized": False}


@dataclass(frozen=True, slots=True)
class SourcePlacement:
    identity: str
    proof: object = field(repr=False, compare=False)
    pricing: object = field(repr=False, compare=False)
    _member: object = field(repr=False, compare=False)
    _task_id: str = field(repr=False, compare=False)
    _record: str = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)

    def validate(self, member=None):
        member = self._member if member is None else member
        if (self._issuer is not _ISSUER or member is not self._member
                or getattr(member.builder.analysis, "_source_placements", {}).get(self.identity) is not self
                or sha256(self._record.encode()).hexdigest() != self.identity
                or _encoded(_record(member, self.proof, self.pricing, self._task_id)) != self._record):
            raise CompilationError("source placement requires its exact registered graph/source binding")
        return self

    def public(self):
        self.validate()
        return {"identity": self.identity, **json.loads(self._record)}

    def deferred(self):
        from compiler.ir.task_graph import DeferredPlacement

        self.validate()
        return DeferredPlacement("live_source_quote_v2", self.identity)


def source_placement(member, proof, pricing, task_id):
    """No source, cost, definition or execution authority is reconstructed."""
    from compiler.frontend.source_device_tasks import SourceDeviceTaskProof
    from compiler.offload.source_task_pricing import SourceTaskPricing

    if type(proof) is not SourceDeviceTaskProof or type(pricing) is not SourceTaskPricing:
        raise CompilationError("source placement requires issued source and pricing objects")
    expected = f"{member.routine.qualified}:source_device:{len(member.source_device_tasks)}"
    if task_id != expected:
        raise CompilationError("source placement must bind the next exact original graph task")
    encoded = _encoded(_record(member, proof, pricing, task_id))
    identity = sha256(encoded.encode()).hexdigest()
    analysis = member.builder.analysis
    if not hasattr(analysis, "_source_placements"):
        analysis._source_placements = {}
    registry = analysis._source_placements
    if identity not in registry:
        registry[identity] = SourcePlacement(identity, proof, pricing, member, task_id, encoded, _ISSUER)
    return registry[identity].validate(member)
