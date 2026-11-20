"""Measured source-submit/scratch costs; no placement or numerical authority."""

from __future__ import annotations

import math

PROTOCOL_ID = 0x4654534F55520001
NATIVE_PROTOCOL_ID = 0x465453484F530001
SOURCE_COST_NAMES = ("source_submit_seconds", "scratch_cold_acquire_seconds", "scratch_reuse_acquire_seconds",
                     "scratch_idle_grow_seconds", "scratch_release_seconds")
BINDINGS = (1, 2, 4, 6, 8)
LAUNCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
FIT_BINDINGS, FIT_LAUNCHES = (1, 4, 8), (1, 4, 16, 64, 256)


def _positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"source protocol {name} must be finite and positive")
    return float(value)








def _schedule(record):
    """Retain raw API identity; compare only the explicitly known modifier bit.

    GNU Fortran's omp_get_schedule binding can expose the base kind while the
    C entry exposes omp_sched_monotonic. Dispatch must still match the full C
    identity; normalizing this observation does not erase that modifier guard.
    """
    kind, chunk = record.get("schedule_kind"), record.get("schedule_chunk")
    if (type(kind) is not int or not -(1 << 31) <= kind < 1 << 32
            or type(chunk) is not int or not 0 <= chunk < 1 << 31):
        raise ValueError("source protocol requires an exact OpenMP schedule identity")
    bits = kind & 0xFFFFFFFF
    base = bits & 0x7FFFFFFF
    if base not in (1, 2, 3, 4):
        raise ValueError("source protocol has an unknown OpenMP schedule kind or modifier")
    return {"raw_kind": kind, "base_kind": base, "monotonic_modifier": bool(bits & 0x80000000), "chunk": chunk}




def validate_source_costs(profile, runtime_id):
    """Read bounded optional evidence without discovering hardware or CUDA."""
    section = profile.get("source_tasks")
    if (not isinstance(runtime_id, str) or len(runtime_id) != 64
            or any(character not in "0123456789abcdef" for character in runtime_id)
            or not isinstance(section, dict) or type(section.get("version")) is not int or section["version"] != 1
            or section.get("protocol_id") != PROTOCOL_ID or type(section.get("available")) is not bool
            or section.get("runtime_id") != runtime_id or section.get("precision_bits") != profile["precision_bits"]
            or section.get("application_profiled") is not False):
        raise ValueError("source protocol identity or runtime mismatch")
    for name in ("max_bindings", "max_launches", "max_scratch_bytes"):
        if type(section.get(name)) is not int or section[name] <= 0:
            raise ValueError("source protocol requires positive integer bounds")
    if (section["max_bindings"] > max(BINDINGS) or section["max_launches"] > max(LAUNCHES)
            or section["max_scratch_bytes"] > 1024**3):
        raise ValueError("source protocol bounds exceed the measured protocol")
    identity = section.get("identity", {})
    expected = {"version": 1, "gpu_uuid": profile["hardware"]["gpu_uuid"],
                "device": profile["hardware"]["device_ordinal"], "cpu_threads": profile["cpu_threads"],
                "precision": profile["precision_bits"],
                **{key: profile["toolchain"][key] for key in ("cuda_runtime_version", "driver_version")}}
    if not isinstance(identity, dict) or any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError("source protocol hardware identity differs from the hardware profile")
    _schedule(identity)
    if set(section.get("costs", {})) != set(SOURCE_COST_NAMES):
        raise ValueError("source protocol costs are incomplete")
    for name in SOURCE_COST_NAMES:
        _positive(section["costs"][name], name)
    if "native_transactions" in section:
        native = section["native_transactions"]
        if (not isinstance(native, dict) or type(native.get("version")) is not int or native["version"] != 1
                or native.get("protocol_id") != NATIVE_PROTOCOL_ID or type(native.get("available")) is not bool
                or type(native.get("max_bindings")) is not int or not 0 < native["max_bindings"] <= max(BINDINGS)):
            raise ValueError("native source protocol identity or bounds are invalid")
        _positive(native.get("prepare_start_complete_seconds"), "native prepare/start/complete")
    return section






