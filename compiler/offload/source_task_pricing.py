"""Source-bound placement estimates, separate from numerical/definition proof."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from hashlib import sha256

from compiler.emission.cuda.batch import base_cost_fields
from compiler.frontend.tile_private import TilePrivateTeamProof
from compiler.ir import CompilationError
from compiler.offload.profile import compiler_identity, scoped_costs, validate_profile
from compiler.offload.source_protocol import source_native_costs, source_task_costs, validate_source_costs
from compiler.offload.tile_work import tile_team_work
from compiler.scopes.segments import directive

_ISSUER = object()


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True, slots=True)
class SourceTaskPricing:
    identity: str
    team: object = field(repr=False, compare=False)
    work: object = field(repr=False, compare=False)
    _profile: str = field(repr=False, compare=False)
    _record: str = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)

    def validate(self):
        if (self._issuer is not _ISSUER
                or getattr(self.team._analysis, "_source_task_pricings", {}).get(self.identity) is not self
                or self.work._team is not self.team or sha256(self._record.encode()).hexdigest() != self.identity):
            raise CompilationError("source pricing requires its original registered cost attachment")
        self.team.validate()
        self.work.validate()
        record = json.loads(self._record)
        if (record["team_identity"] != self.team.identity or record["work"]["identity"] != self.work.identity
                or record["profile_identity"] != sha256(self._profile.encode()).hexdigest()):
            raise CompilationError("source pricing changed its source/work/profile identity")
        return self

    def public(self):
        self.validate()
        return {"identity": self.identity, **json.loads(self._record)}

    @property
    def native_live(self):
        return json.loads(self._record)["version"] in (2, 3, 4)

    def profile(self):
        self.validate()
        return json.loads(self._profile)


def source_task_pricing(team, profile, runtime_id, *, mapping="cooperative", host_threads=4, native_live=False):
    """No measurements, descriptors, payload, CUDA or execution are touched."""
    if type(team) is not TilePrivateTeamProof:
        raise CompilationError("source pricing requires the exact complete-team source proof")
    team.validate()
    work = tile_team_work(team, mapping=mapping)
    kinds = {binding.kind for binding, _ in team.fields if binding.dtype == "real"}
    if len(kinds) != 1:
        raise ValueError("source pricing requires one calibrated REAL precision")
    precision = 8 * next(iter(kinds))
    validate_profile(profile, precision_bits=precision, cpu_threads=host_threads, scoped_runtime_id=runtime_id)
    compiler = compiler_identity(profile)
    protocol = (validate_source_costs(profile, runtime_id) if native_live else source_task_costs(profile, runtime_id))
    directives = [directive(node) for node in team._nodes]
    loops = [text for text in directives if text is not None and re.match(r"do(?:\s|$)", text)]
    collapses = []
    for text in loops:
        if not re.search(r"\bschedule\s*\(\s*runtime\s*\)", text):
            raise ValueError("source pricing requires its measured original runtime schedule")
        collapse = re.search(r"\bcollapse\s*\(\s*([123])\s*\)", text)
        collapses.append(int(collapse.group(1)) if collapse else 1)
    if not collapses or len(set(collapses)) != 1 or len(loops) != len(work.units):
        raise ValueError("source pricing requires one exact original worksharing protocol")
    rank, collapse = len(work.units[0].axes), collapses[0]
    rows = [row for row in protocol["original_teams"]["costs"]
            if (row["rank"], row["collapse"], row["units"]) == (rank, collapse, len(loops))]
    if len(rows) != 1:
        raise ValueError("source pricing has no matching native team rank/collapse/unit observation")
    rates = profile["rates"]
    record = {
        "version": 1, "kind": "static_roofline_and_measured_source_protocol",
        "team_identity": team.identity, "work": work.public(),
        "profile_identity": sha256(_encoded(profile).encode()).hexdigest(), "runtime_id": runtime_id,
        "profile_artifact": "source-profile-" + sha256(_encoded(profile).encode()).hexdigest() + ".json",
        "protocol_id": protocol["protocol_id"], "precision_bits": precision, "cpu_threads": host_threads,
        "host_identity": compiler, "hardware_identity": profile["hardware"],
        "source_costs": {key: protocol[key] for key in ("max_bindings", "max_launches", "max_scratch_bytes", "costs")},
        "base_costs": {"max_allocation_bytes": profile["scoped"]["max_allocation_bytes"],
                       **base_cost_fields(profile, scoped_costs(profile, runtime_id))},
        "native_protocol": {**rows[0], "c_schedule": protocol["original_teams"]["schedule"]["c_identity"]},
        "rates": {key: rates[key] for key in ("cpu_flops_per_second", "cpu_memory_bytes_per_second",
                                                "gpu_flops_per_second", "gpu_memory_bytes_per_second")},
        "endpoints": {"native": "complete context/domain close then original team",
            "gpu": "source transaction; domain remains live; complete close is only a ranking projection",
            "prefix_fence": "existing pre-scratch wait before quote; common incurred term excluded"},
        "exclusions": ["common quote computation", "PRIVATE allocation", "source-body/cache/barrier inefficiencies"],
        "model": "static arithmetic and source-array-traffic roofline estimates, not source-family timing attestation",
        "submit_model": "bounded maximum-median constant; not fitted launch scaling",
        "whole_owner_estimate_available": False, "full_wall_prediction": False,
        "definitions_authorized": False, "execution_authorized": False, "motion_authorized": False,
    }
    if native_live:
        record["version"] = 2
        record["native_transaction_costs"] = source_native_costs(profile, runtime_id)
        record["endpoints"] = {
            "native": "typed host transaction then original team; domain and readonly replicas remain live",
            "gpu": "source transaction; domain remains live; no terminal retirement projection",
            "prefix_fence": "GPU-only pre-scratch wait after choice; native acquisition waits exact conflicts",
        }
        if not protocol["available"]:
            record.update(version=3, gpu_protocol_available=False)
        if "source_submit_model" in profile:
            from compiler.offload.source_submit_model import validate_model

            model = validate_model(profile["source_submit_model"], profile, runtime_id)
            if model["available"]:
                record.update(version=4, gpu_protocol_available=True,
                    submit_model="nonnegative binding/launch model; fixed engineering reserve; no statistical guarantee",
                    source_submit_model={key: model[key] for key in ("version", "model_identity", "method", "runtime_id",
                        "max_bindings", "max_launches", "kernel_argument_words", "kernel_argument_bytes",
                        "block_threads", "calibration_grid_blocks",
                        "coverage_factor", "coverage_policy", "coefficients", "available", "fit_bindings",
                        "fit_launches", "observed_bindings", "observed_launches")})
    encoded = _encoded(record)
    identity = sha256(encoded.encode()).hexdigest()
    if not hasattr(team._analysis, "_source_task_pricings"):
        team._analysis._source_task_pricings = {}
    registry = team._analysis._source_task_pricings
    if identity not in registry:
        registry[identity] = SourceTaskPricing(identity, team, work, _encoded(profile), encoded, _ISSUER)
    return registry[identity].validate()
