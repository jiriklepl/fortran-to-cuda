"""Validated, application-independent hardware costs for offload decisions.

All durations are seconds, transfer rates are payload bytes/second, memory
throughput counts bytes read plus written, and compute throughput counts
floating operations (one multiply-add is two). Profiles are keyed by schema,
hardware, toolchains, precision and CPU thread budget, never compiler sources.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
TRANSFER_KINDS = ("h2d_pageable", "d2h_pageable", "h2d_pinned", "d2h_pinned")
THROUGHPUT_RATES = (
    "pack_bytes_per_second",
    "unpack_bytes_per_second",
    "cpu_flops_per_second",
    "cpu_memory_bytes_per_second",
    "gpu_flops_per_second",
    "gpu_memory_bytes_per_second",
)
WORKER_RATES = ("cpu_worker_flops_per_second", "cpu_worker_memory_bytes_per_second")
LATENCY_RATES = ("launch_latency_seconds", "pump_latency_seconds")
HARDWARE_FIELDS = ("cpu_name", "gpu_uuid", "gpu_name", "compute_capability")
TOOLCHAIN_FIELDS = ("nvcc_version", "host_cxx_version", "cuda_runtime_version", "driver_version")


class ProfileError(ValueError):
    """An absent, invalid, or mismatched hardware profile cannot guide offload."""


def compiler_identity(profile: dict) -> dict[str, str | int]:
    """Extract identities that the emitted CUDA translation unit can verify.

    Compiler banners remain in the profile for provenance. Only explicit NVCC
    and GCC version triples can be matched to compiler predefined macros;
    unknown banners do not make an automatic policy eligible for execution.
    """
    toolchain = profile["toolchain"]
    cuda = re.search(r"\bV(\d+\.\d+\.\d+)\b", toolchain["nvcc_version"])
    host_banner = toolchain["host_cxx_version"].splitlines()[0]
    host = re.search(r"\b(\d+\.\d+\.\d+)\s*$", host_banner)
    if not cuda or not host or not re.search(r"(?:g\+\+|gcc|GCC)", host_banner):
        raise ProfileError("hardware profile compiler identity cannot be verified; require NVCC and GCC version triples")
    return {
        "cpu_name": profile["hardware"]["cpu_name"],
        "host_compiler": host.group(1),
        "cuda_compiler": cuda.group(1),
        "cuda_runtime": toolchain["cuda_runtime_version"],
        "cuda_driver": toolchain["driver_version"],
    }


def _number(value: Any, field: str, *, zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ProfileError(f"{field} must be a finite number")
    if value < 0 or (value == 0 and not zero):
        raise ProfileError(f"{field} must be {'nonnegative' if zero else 'positive'}")
    return float(value)


def validate_profile(
    profile: Any,
    *,
    precision_bits: int | None = None,
    cpu_threads: int | None = None,
    hardware: dict | None = None,
    toolchain: dict | None = None,
) -> dict:
    """Validate costs and optionally require the caller's known identity.

    No hardware discovery or CUDA initialization occurs here. Callers can pass
    whichever identity fields they have independently checked; every provided
    field must match. A GPU/runtime wrapper must still verify device and team
    identity when dispatching because these can differ from the build host.
    """
    if not isinstance(profile, dict) or type(profile.get("schema_version")) is not int:
        raise ProfileError("profile must be an object with integer schema_version")
    if profile["schema_version"] != SCHEMA_VERSION:
        raise ProfileError(f"unsupported hardware profile schema: {profile['schema_version']}")
    if type(profile.get("precision_bits")) is not int or profile["precision_bits"] not in (32, 64):
        raise ProfileError("precision_bits must be 32 or 64")
    if type(profile.get("cpu_threads")) is not int or profile["cpu_threads"] < 1:
        raise ProfileError("cpu_threads must be a positive integer")
    for field, requested in (("precision_bits", precision_bits), ("cpu_threads", cpu_threads)):
        if requested is not None and profile[field] != requested:
            raise ProfileError(f"hardware profile {field} mismatch: {profile[field]} != {requested}")
    for section, required, expected in (
        ("hardware", HARDWARE_FIELDS, hardware),
        ("toolchain", TOOLCHAIN_FIELDS, toolchain),
    ):
        values = profile.get(section)
        if not isinstance(values, dict):
            raise ProfileError(f"missing {section} identity")
        for field in required:
            value = values.get(field)
            if field in ("cuda_runtime_version", "driver_version"):
                if type(value) is not int or value <= 0:
                    raise ProfileError(f"{section}.{field} must be a positive version integer")
            elif not isinstance(value, str) or not value.strip():
                raise ProfileError(f"{section}.{field} must be a nonempty string")
        for field, requested in (expected or {}).items():
            actual = values.get(field)
            if field == "gpu_uuid" and isinstance(actual, str) and isinstance(requested, str):
                actual, requested = actual.lower(), requested.lower()
            if actual != requested:
                raise ProfileError(f"hardware profile {section}.{field} mismatch")
    rates = profile.get("rates")
    if not isinstance(rates, dict):
        raise ProfileError("missing rates")
    for name in THROUGHPUT_RATES:
        _number(rates.get(name), f"rates.{name}")
    for name in LATENCY_RATES:
        _number(rates.get(name), f"rates.{name}")
    for name in WORKER_RATES:
        value = _number(rates.get(name), f"rates.{name}", zero=profile["cpu_threads"] == 1)
        if profile["cpu_threads"] == 1 and value != 0:
            raise ProfileError(f"rates.{name} must be zero when the pump leaves no CPU workers")
    for name in TRANSFER_KINDS:
        transfer = rates.get(name)
        if not isinstance(transfer, dict):
            raise ProfileError(f"missing rates.{name}")
        _number(transfer.get("latency_seconds"), f"rates.{name}.latency_seconds", zero=True)
        _number(transfer.get("bandwidth_bytes_per_second"), f"rates.{name}.bandwidth_bytes_per_second")
    return profile


def load_profile(path: str | Path, **expected: Any) -> dict:
    """Load a profile, raising ProfileError rather than supplying guessed costs."""
    try:
        profile = json.loads(Path(path).read_text())
    except (OSError, ValueError, TypeError) as error:
        raise ProfileError(f"cannot load hardware profile {path}: {error}") from error
    return validate_profile(profile, **expected)
