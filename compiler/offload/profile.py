"""Validated, application-independent hardware costs for offload decisions.

All durations are seconds, transfer rates are payload bytes/second, memory
throughput counts bytes read plus written, and compute throughput counts
floating operations (one multiply-add is two). Base costs are keyed by schema,
hardware, toolchains, precision and CPU thread budget. Optional scoped costs
also identify the common runtime whose management operations were measured.
"""
from __future__ import annotations

import json
import math
import re
from hashlib import sha256
from importlib.resources import files
from pathlib import Path
from typing import Any

from compiler.numerical_contract import require_numerical_build_contract

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
SCOPED_COST_NAMES = (
    "create_seconds", "register_seconds", "host_access_seconds", "device_access_seconds",
    "gpu_setup_seconds", "cold_driver_startup_seconds", "allocation_seconds", "release_seconds",
    "wait_seconds", "launch_enqueue_seconds", "planning_operation_seconds",
)
SCOPED_TRANSFER_COST_NAMES = (
    "staging_cold_seconds", "staging_reuse_seconds", "event_record_seconds",
    "event_wait_seconds", "ready_event_seconds", "preparation_operation_seconds",
    "pack_bytes_per_second", "unpack_bytes_per_second", "pack_row_seconds", "unpack_row_seconds",
)
SCOPED_BATCH_PAYLOADS = (256 * 1024, 1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024)
SCOPED_TEAM_PROTOCOL_ID = 0x4654434F4C4C0001
SCOPED_TEAM_RATE_NAMES = ("cpu_flops", "cpu_bandwidth", "native_cpu_flops", "native_cpu_bandwidth")
SCOPED_TEAM_COST_NAMES = (
    "owner_seconds", "descriptor_seconds", "entry_seconds", "cpu_worker_seconds",
    "gpu_worker_seconds", "native_call_seconds", "native_worker_seconds",
)


def collective_protocol_identity() -> dict:
    """Conservatively identify source that emits the measured team protocol."""
    sources = {}
    for package, names in (
        ("compiler", ("numerical_contract.py",)),
        ("compiler.scopes", ("collective.py", "collective_roles.py", "source.py")),
        ("compiler.emission.cuda", ("scoped.py", "offload.py")),
        ("compiler.runtime", ("scoped_team_observer.hpp", "scoped_team_observer.f90")),
    ):
        for name in names:
            sources[package + "/" + name] = sha256(files(package).joinpath(name).read_bytes()).hexdigest()
    identity = sha256("\n".join(f"{name}:{digest}" for name,digest in sorted(sources.items())).encode()).hexdigest()
    return {"identity": identity, "sources": sources}


class ProfileError(ValueError):
    """An absent, invalid, or mismatched hardware profile cannot guide offload."""


def require_profile_numerical_contract(profile: dict) -> None:
    """Legacy profiles remain readable, but cannot price changed arithmetic."""
    try:
        require_numerical_build_contract(profile.get("numerical_contract") if isinstance(profile, dict) else None)
    except ValueError as error:
        raise ProfileError("hardware profile " + str(error)) from error


def compiler_identity(profile: dict) -> dict[str, str | int]:
    """Extract identities that the emitted CUDA translation unit can verify.

    Compiler banners remain in the profile for provenance. Only explicit NVCC
    and GCC version triples can be matched to compiler predefined macros;
    unknown banners do not make an automatic policy eligible for execution.
    """
    require_profile_numerical_contract(profile)
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
    scoped_runtime_id: str | None = None,
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
    if "numerical_contract" in profile:
        require_profile_numerical_contract(profile)
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
    if "numerical" in profile:
        from .numerical_calibration import NumericalCalibrationError, validate_numerical_profile
        try:
            validate_numerical_profile(profile["numerical"], profile)
        except NumericalCalibrationError as error:
            raise ProfileError(str(error)) from error
    if "cpu_execution_protocol" in profile:
        from .cpu_protocol_calibration import validate_cpu_protocol
        from .numerical_calibration import NumericalCalibrationError
        try:
            validate_cpu_protocol(profile)
        except NumericalCalibrationError as error:
            raise ProfileError(str(error)) from error
    if "cpu_dependency" in profile:
        from .cpu_dependency_calibration import validate_dependency_profile
        from .numerical_calibration import NumericalCalibrationError
        try:
            validate_dependency_profile(profile)
        except NumericalCalibrationError as error:
            raise ProfileError(str(error)) from error
    if "scoped" in profile:
        scoped = profile["scoped"]
        if not isinstance(scoped, dict) or type(scoped.get("schema_version")) is not int or scoped["schema_version"] != 1:
            raise ProfileError("scoped costs require schema_version 1")
        runtime_id = scoped.get("runtime_id")
        if not isinstance(runtime_id, str) or not re.fullmatch(r"[0-9a-f]{64}", runtime_id):
            raise ProfileError("scoped.runtime_id must be a lowercase SHA-256 identity")
        bound = scoped.get("max_allocation_bytes")
        if type(bound) is not int or not 0 < bound < 2**64:
            raise ProfileError("scoped.max_allocation_bytes must be a positive 64-bit byte count")
        costs = scoped.get("costs")
        if not isinstance(costs, dict):
            raise ProfileError("missing scoped costs")
        for name in SCOPED_COST_NAMES:
            _number(costs.get(name), f"scoped.costs.{name}")
        if "collective" in scoped:
            collective = scoped["collective"]
            if (not isinstance(collective, dict) or type(collective.get("schema_version")) is not int
                    or collective["schema_version"] != 1):
                raise ProfileError("scoped.collective requires schema_version 1")
            if (type(collective.get("protocol_id")) is not int
                    or collective["protocol_id"] != SCOPED_TEAM_PROTOCOL_ID):
                raise ProfileError("scoped.collective protocol_id mismatch")
            if (type(collective.get("cpu_threads")) is not int
                    or collective["cpu_threads"] != profile["cpu_threads"]):
                raise ProfileError("scoped.collective thread budget mismatch")
            if type(collective.get("expected_omp_level")) is not int or collective["expected_omp_level"] != 1:
                raise ProfileError("scoped.collective requires a level-one persistent team")
            team_costs = collective.get("costs")
            if not isinstance(team_costs, dict):
                raise ProfileError("missing scoped.collective costs")
            for name in SCOPED_TEAM_RATE_NAMES:
                _number(team_costs.get(name), f"scoped.collective.costs.{name}")
            for name in SCOPED_TEAM_COST_NAMES:
                _number(team_costs.get(name), f"scoped.collective.costs.{name}", zero=True)
            fortran = collective.get("fortran")
            if not isinstance(fortran, dict) or any(not isinstance(fortran.get(name), str) or not fortran[name]
                    for name in ("compiler_version", "compiler_options", "semantic_options")):
                raise ProfileError("scoped.collective requires the original Fortran compiler and semantic options")
            # The normalizer is shared with the offline producer; location
            # flags are the only explicitly excluded compiler options.
            from .collective_calibration import normalize_fortran_options
            try:
                semantic = normalize_fortran_options(fortran["compiler_options"])
            except ValueError as error:
                raise ProfileError("invalid collective Fortran compiler options") from error
            if semantic != fortran["semantic_options"]:
                raise ProfileError("collective Fortran semantic options disagree with raw provenance")
            protocol = collective.get("protocol_sources")
            if (not isinstance(protocol,dict) or not isinstance(protocol.get("identity"),str)
                    or not re.fullmatch(r"[0-9a-f]{64}",protocol["identity"])
                    or not isinstance(protocol.get("sources"),dict) or not protocol["sources"]
                    or any(not isinstance(name,str) or not isinstance(digest,str)
                           or not re.fullmatch(r"[0-9a-f]{64}",digest) for name,digest in protocol["sources"].items())):
                raise ProfileError("scoped.collective requires protocol source identities")
        if "transfers" in scoped:
            transfers = scoped["transfers"]
            if (not isinstance(transfers, dict) or type(transfers.get("schema_version")) is not int
                    or transfers["schema_version"] != 1):
                raise ProfileError("scoped.transfers requires schema_version 1")
            if (transfers.get("batch_payload_bytes") != list(SCOPED_BATCH_PAYLOADS)
                    or any(type(value) is not int for value in transfers["batch_payload_bytes"])):
                raise ProfileError("scoped.transfers requires the supported finite batch payloads")
            if (type(transfers.get("max_slot_bytes")) is not int
                    or transfers["max_slot_bytes"] != SCOPED_BATCH_PAYLOADS[-1]):
                raise ProfileError("scoped.transfers.max_slot_bytes must match the measured slot bound")
            transfer_costs = transfers.get("costs")
            if not isinstance(transfer_costs, dict):
                raise ProfileError("missing scoped.transfers.costs")
            for name in SCOPED_TRANSFER_COST_NAMES:
                value = transfer_costs.get(name)
                if name.startswith("staging_"):
                    if not isinstance(value, list) or len(value) != len(SCOPED_BATCH_PAYLOADS):
                        raise ProfileError(f"scoped.transfers.costs.{name} requires four measured slot costs")
                    for index, item in enumerate(value):
                        _number(item, f"scoped.transfers.costs.{name}[{index}]")
                else:
                    _number(value, f"scoped.transfers.costs.{name}", zero=name.endswith("row_seconds"))
    if scoped_runtime_id is not None:
        scoped_costs(profile, scoped_runtime_id)
    return profile


def scoped_costs(profile: dict, runtime_id: str) -> dict:
    """Require costs measured for this common runtime, without guessed defaults.

    Ordinary profiles remain valid without this optional extension. Callers
    requesting scoped automatic execution must explicitly check the runtime
    identity after validating the base hardware profile.
    """
    scoped = profile.get("scoped")
    if not isinstance(scoped, dict):
        raise ProfileError("hardware profile has no scoped runtime calibration")
    if scoped.get("runtime_id") != runtime_id:
        raise ProfileError("hardware profile scoped runtime_id mismatch")
    return scoped["costs"]


def scoped_transfer_costs(profile: dict, runtime_id: str) -> dict:
    """Require separately measured staging costs for the same common runtime.

    The optional extension does not make an older direct-transfer profile
    invalid. No defaults or live measurements fill missing staging costs.
    """
    scoped_costs(profile, runtime_id)
    transfers = profile["scoped"].get("transfers")
    if not isinstance(transfers, dict):
        raise ProfileError("hardware profile has no scoped transfer calibration")
    return transfers["costs"]


def scoped_collective_costs(profile: dict, runtime_id: str, *, cpu_threads: int | None = None) -> dict:
    """Require actual persistent-team costs for the current coordination ABI.

    Serial/fork-join rates never establish an existing team's cost. Callers
    must additionally verify the actual OpenMP level, width and hardware at
    dispatch; this accessor neither enters a team nor initializes CUDA.
    """
    validate_profile(profile, cpu_threads=cpu_threads, scoped_runtime_id=runtime_id)
    collective = profile["scoped"].get("collective")
    if not isinstance(collective, dict):
        raise ProfileError("collective synchronization calibration is unavailable")
    if collective.get("protocol_sources") != collective_protocol_identity():
        raise ProfileError("collective synchronization calibration source identity mismatch")
    return collective["costs"]


def load_profile(path: str | Path, **expected: Any) -> dict:
    """Load a profile, raising ProfileError rather than supplying guessed costs."""
    try:
        profile = json.loads(Path(path).read_text())
    except (OSError, ValueError, TypeError) as error:
        raise ProfileError(f"cannot load hardware profile {path}: {error}") from error
    return validate_profile(profile, **expected)
