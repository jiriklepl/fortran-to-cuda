"""Closed cost-attachment integrity checks, without pricing or source authority."""
from __future__ import annotations

import math
import re

from .checked_execution_plan import _digest, _require, _sha256, _text

_PRICING = {"identity", "version", "kind", "team_identity", "work", "profile_identity", "profile_artifact", "runtime_id",
            "protocol_id", "precision_bits", "cpu_threads", "host_identity", "hardware_identity", "source_costs",
            "base_costs", "native_protocol", "rates", "endpoints", "exclusions", "model", "submit_model",
            "whole_owner_estimate_available", "full_wall_prediction", "definitions_authorized",
            "execution_authorized", "motion_authorized"}
_COUNTS = {"real_work_proxy", "external_read_bytes", "external_write_bytes", "scratch_read_bytes",
           "scratch_write_bytes"}
_SOURCE_COSTS = {"source_submit_seconds", "scratch_cold_acquire_seconds", "scratch_reuse_acquire_seconds",
                 "scratch_idle_grow_seconds", "scratch_release_seconds"}
_SCOPED_COSTS = {"cold_driver_startup_seconds", "create_seconds", "register_seconds", "host_access_seconds",
                 "gpu_setup_seconds", "device_access_seconds", "launch_enqueue_seconds", "wait_seconds",
                 "allocation_seconds", "release_seconds", "planning_operation_seconds"}
_BASE_RATES = {"cpu_flops", "cpu_bandwidth", "gpu_flops", "gpu_bandwidth", "h2d_bandwidth", "d2h_bandwidth"}
_RATES = {"cpu_flops_per_second", "cpu_memory_bytes_per_second", "gpu_flops_per_second",
          "gpu_memory_bytes_per_second"}


def _keys(value, keys):
    _require(isinstance(value, dict) and set(value) == keys, "source pricing requires its exact schema")


def _uint(value, *, maximum=(1 << 64) - 1, minimum=0):
    return type(value) is int and minimum <= value <= maximum


def _number(value, *, positive=False):
    try:
        return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)
    except OverflowError:
        return False


def _schedule(value):
    _keys(value, {"raw_kind", "base_kind", "monotonic_modifier", "chunk"})
    raw = value["raw_kind"]
    _require(type(raw) is int and -(1 << 31) <= raw < 1 << 32
             and _uint(value["chunk"], maximum=(1 << 31) - 1)
             and type(value["base_kind"]) is int and value["base_kind"] in {1, 2, 3, 4}
             and value["base_kind"] == (raw & 0x7fffffff)
             and type(value["monotonic_modifier"]) is bool
             and value["monotonic_modifier"] == bool(raw & 0x80000000),
             "source pricing original schedule identity differs")


def validate_source_pricing(value, source, manifest, *, native_live=False):
    """Validate issued JSON linkage; never infer a fit, speedup or legality."""
    native_only = native_live and isinstance(value, dict) and type(value.get("version")) is int and value["version"] == 3
    _keys(value, _PRICING | ({"native_transaction_costs"} if native_live else set())
          | ({"gpu_protocol_available"} if native_only else set()))
    if native_only:
        _require(value["gpu_protocol_available"] is False,
                 "source pricing native-only attachment cannot qualify the GPU protocol")
    team = source["team"]
    runtime = manifest.get("runtime")
    _require(isinstance(runtime, dict), "source pricing requires the manifest runtime identity")
    _require(type(value["version"]) is int and value["version"] in ({2, 3} if native_live else {1})
             and value["kind"] == "static_roofline_and_measured_source_protocol"
             and _sha256(value["identity"])
             and value["identity"] == _digest({k: v for k, v in value.items() if k != "identity"})
             and value["team_identity"] == team["identity"] and _sha256(value["profile_identity"])
             and value["profile_artifact"] == "source-profile-" + value["profile_identity"] + ".json"
             and manifest.get("artifacts_sha256", {}).get(value["profile_artifact"]) == value["profile_identity"]
             and _sha256(value["runtime_id"])
             and value["runtime_id"] == runtime.get("runtime_id")
             and type(value["protocol_id"]) is int and value["protocol_id"] == 0x4654534f55520001
             and type(value["precision_bits"]) is int and value["precision_bits"] in {32, 64}
             and _uint(value["cpu_threads"], maximum=(1 << 31) - 1, minimum=1),
             "source pricing source/profile/runtime/protocol identity differs")
    endpoints = ({"native": "typed host transaction then original team; domain and readonly replicas remain live",
                  "gpu": "source transaction; domain remains live; no terminal retirement projection",
                  "prefix_fence": "GPU-only pre-scratch wait after choice; native acquisition waits exact conflicts"}
                 if native_live else {
                     "native": "complete context/domain close then original team",
                     "gpu": "source transaction; domain remains live; complete close is only a ranking projection",
                     "prefix_fence": "existing pre-scratch wait before quote; common incurred term excluded"})
    _require(all(value[key] is False for key in ("whole_owner_estimate_available", "full_wall_prediction",
                 "definitions_authorized", "execution_authorized", "motion_authorized"))
             and value["model"] == "static arithmetic and source-array-traffic roofline estimates, not source-family timing attestation"
             and value["submit_model"] == "bounded maximum-median constant; not fitted launch scaling"
             and value["exclusions"] == ["common quote computation", "PRIVATE allocation", "source-body/cache/barrier inefficiencies"]
             and value["endpoints"] == endpoints,
             "source pricing cannot authorize execution, definitions, motion or whole-wall estimates")
    if native_live:
        native = value["native_transaction_costs"]
        _keys(native, {"version", "protocol_id", "max_bindings", "prepare_start_complete_seconds"})
        _require(type(native["version"]) is int and native["version"] == 1
                 and type(native["protocol_id"]) is int and native["protocol_id"] == 0x465453484f530001
                 and _uint(native["max_bindings"], minimum=1, maximum=8)
                 and _number(native["prepare_start_complete_seconds"], positive=True),
                 "source pricing native transaction costs differ")
    work = value["work"]
    _keys(work, {"identity", "version", "team_identity", "mapping", "units", "model",
                 "original_fortran_family_attestation", "speedup_guaranteed", "placement_authorized",
                 "definitions_authorized", "unpriced"})
    _require(type(work["version"]) is int and work["version"] == 1
             and _sha256(work["identity"]) and work["identity"] == _digest({k: v for k, v in work.items() if k != "identity"})
             and work["team_identity"] == team["identity"] and isinstance(work["mapping"], str)
             and work["mapping"] in {"cooperative", "thread"}
             and all(work[key] is False for key in ("original_fortran_family_attestation", "speedup_guaranteed",
                                                   "placement_authorized", "definitions_authorized"))
             and work["model"] == "static real-arithmetic and source-array-traffic proxy; measured rates required separately"
             and work["unpriced"] == ["source transaction and scratch lifecycle", "native team/worksharing protocol",
                                       "mapping/control/barrier efficiency"],
             "source pricing work attachment differs")
    units = work["units"]
    _require(isinstance(units, list) and len(units) == len(team["unit_proof_identities"]),
             "source pricing work-unit inventory differs")
    for unit, identity in zip(units, team["unit_proof_identities"], strict=True):
        _keys(unit, {"source_identity", "axes", "producer", "consumer"})
        _require(unit["source_identity"] == identity and isinstance(unit["axes"], list)
                 and len(unit["axes"]) == len(team["domain"]), "source pricing work-unit source differs")
        for axis, domain in zip(unit["axes"], team["domain"], strict=True):
            _keys(axis, {"lower", "upper", "width", "producer_lower_offset", "producer_clip_offset", "producer_after_clip_offset"})
            _require(all(axis[key] == domain[key] for key in ("lower", "upper", "width"))
                     and all(type(axis[key]) is int and -(1 << 31) <= axis[key] < 1 << 31
                             for key in ("producer_lower_offset", "producer_clip_offset", "producer_after_clip_offset")),
                     "source pricing original phase geometry differs")
        for phase in (unit["producer"], unit["consumer"]):
            _keys(phase, _COUNTS | {"conditional_upper_bound"})
            _require(all(_uint(phase[key]) for key in _COUNTS) and type(phase["conditional_upper_bound"]) is bool,
                     "source pricing work counts differ")
    hardware, host = value["hardware_identity"], value["host_identity"]
    _keys(hardware, {"gpu_uuid", "gpu_name", "compute_capability", "cpu_name", "machine", "device_ordinal", "async_engine_count"})
    _keys(host, {"cpu_name", "host_compiler", "cuda_compiler", "cuda_runtime", "cuda_driver"})
    _require(all(_text(hardware[key]) for key in ("gpu_uuid", "gpu_name", "compute_capability", "cpu_name", "machine"))
             and _uint(hardware["device_ordinal"], maximum=(1 << 31) - 1)
             and _uint(hardware["async_engine_count"], maximum=(1 << 31) - 1)
             and host["cpu_name"] == hardware["cpu_name"]
             and all(isinstance(host[key], str) and re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", host[key])
                     for key in ("host_compiler", "cuda_compiler"))
             and all(_uint(host[key], minimum=1, maximum=(1 << 31) - 1) for key in ("cuda_runtime", "cuda_driver")),
             "source pricing hardware/compiler identity differs")
    costs = value["source_costs"]
    _keys(costs, {"max_bindings", "max_launches", "max_scratch_bytes", "costs"})
    _require(_uint(costs["max_bindings"], minimum=1) and _uint(costs["max_launches"], minimum=1)
             and _uint(costs["max_scratch_bytes"], minimum=1), "source pricing protocol bounds differ")
    _keys(costs["costs"], _SOURCE_COSTS)
    _require(all(_number(number, positive=True) for number in costs["costs"].values()), "source pricing protocol costs differ")
    base = value["base_costs"]
    _keys(base, _SCOPED_COSTS | _BASE_RATES | {"max_allocation_bytes", "h2d_latency", "d2h_latency",
                                             "h2d_fit_relative_error", "d2h_fit_relative_error"})
    _require(_uint(base["max_allocation_bytes"], minimum=1)
             and all(_number(base[key], positive=True) for key in _BASE_RATES)
             and all(_number(base[key]) for key in _SCOPED_COSTS | {"h2d_latency", "d2h_latency"})
             and all(_number(base[key]) and base[key] < 1 for key in ("h2d_fit_relative_error", "d2h_fit_relative_error")),
             "source pricing base costs differ")
    rates = value["rates"]
    _keys(rates, _RATES)
    _require(all(_number(number, positive=True) for number in rates.values())
             and all(rates[rate] == base[base_key] for rate, base_key in zip(
                 ("cpu_flops_per_second", "cpu_memory_bytes_per_second", "gpu_flops_per_second", "gpu_memory_bytes_per_second"),
                 ("cpu_flops", "cpu_bandwidth", "gpu_flops", "gpu_bandwidth"), strict=True)),
             "source pricing roofline rates differ")
    native = value["native_protocol"]
    _keys(native, {"rank", "collapse", "units", "seconds", "schedule", "c_schedule"})
    _require(type(native["rank"]) is int and native["rank"] == len(team["domain"])
             and type(native["collapse"]) is int and 1 <= native["collapse"] <= native["rank"]
             and type(native["units"]) is int and native["units"] == len(units)
             and _number(native["seconds"], positive=True), "source pricing original native protocol differs")
    _schedule(native["schedule"])
    _schedule(native["c_schedule"])
    _require(all(native["schedule"][key] == native["c_schedule"][key] for key in ("base_kind", "chunk"))
             and (not native["schedule"]["monotonic_modifier"] or native["c_schedule"]["monotonic_modifier"]),
             "source pricing native/C schedule linkage differs")
    kinds = {field["binding"]["kind"] for field in team["fields"] if field["binding"]["type"] == "real"}
    _require(kinds == {value["precision_bits"] // 8}, "source pricing original REAL precision differs")
