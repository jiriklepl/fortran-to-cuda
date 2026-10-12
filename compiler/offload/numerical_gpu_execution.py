"""Read-only GPU execution observations; these never authorize AUTO placement.

Saved observations authenticate a frozen registry and their original backend.
Checking current renderer identities is separate from reading that evidence.
No frontend lowering, CUDA compilation or device discovery occurs here.
"""

from __future__ import annotations

import re
import statistics
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

from compiler.numerical_contract import cuda_compile_options, numerical_build_contract, require_numerical_build_contract
from compiler.offload.numerical_calibration import NumericalCalibrationError
from compiler.offload.numerical_execution import _hash, _number, _sha, _validate_registry

SCHEMA_VERSION = 1
PROTOCOL_ID = "original-source-production-cuda-evidence-v1"
BACKEND_ID = "production-cuda-kernels-v1"
SECTION = "numerical_execution_gpu"
IDENTITY_KIND = "numerical_gpu_execution_identity_v1"
PREFLIGHT_KIND = "numerical_gpu_execution_preflight_v1"
BATCH_KIND = "numerical_gpu_execution_batch_v1"
SAMPLES = 7
CELLS = 97
RAW_SAMPLES = CELLS * SAMPLES
TINY_SIZES = (0, 1, 8, 17)
MIN_BATCH_SECONDS = 0.2
MAX_RECORDS = 1 + 17 * len(TINY_SIZES) + CELLS + RAW_SAMPLES
PREFLIGHT_RECORDS = 17 * len(TINY_SIZES) + CELLS
ENVIRONMENT_KEYS = ("CUDA_VISIBLE_DEVICES", "CUDA_LAUNCH_BLOCKING", "FORT_PHASE_TIMING", "OMP_NUM_THREADS",
                    "OMP_DYNAMIC", "OMP_PROC_BIND", "OMP_WAIT_POLICY", "GOMP_SPINCOUNT")


class NumericalGpuExecutionError(NumericalCalibrationError):
    """Missing, conflicting or incompatible GPU execution evidence."""


def gpu_renderer_sources():
    """Hash the renderer closure without constructing a compiler object."""
    root = Path(__file__).parents[1]
    paths = [root / "numerical_contract.py", root / "runtime/numeric.hpp"]
    for folder in ("emission/c", "emission/common", "emission/cuda"):
        paths.extend(p for p in (root / folder).rglob("*") if p.suffix in {".py", ".cuh"})
    paths.extend(Path(__file__).parent / name for name in (
        "numerical_gpu_execution.py", "numerical_gpu_execution_calibration.py", "numerical_gpu_execution_driver.cu"))
    return {str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}


def gpu_generator_identity(source_generator_id, registry_id, renderer_sources):
    return _hash({"source_generator_id": source_generator_id, "registry_id": registry_id,
                  "renderer_sources": renderer_sources})


def current_gpu_identities(precision):
    from compiler.offload.numerical_execution_workloads import execution_generator_identity, registry_identity

    source, registry = execution_generator_identity(), registry_identity(precision)
    renderer = gpu_renderer_sources()
    return source, registry, gpu_generator_identity(source, registry, renderer), renderer


def _protocol():
    return {"samples": SAMPLES, "minimum_batch_wall_seconds": MIN_BATCH_SECONDS,
            "cells": CELLS, "raw_samples": RAW_SAMPLES, "tiny_sizes": list(TINY_SIZES),
            "global_order": "round -> original registry order -> original size order",
            "resident_event_seconds": "CUDA events around resident launches; may include queue starvation",
            "enqueue_seconds": "host wall time of production launch, error check and launch counter only",
            "wall_seconds": "complete batch including event operations and final synchronization",
            "event_wall_clock_tolerance": "max(1ms,1% of wall); separate clocks, never a startup fit",
            "fixed_launch_price": "unavailable; zero items launch nothing; no fitted startup or subtraction",
            "separate_costs": "transfers, allocation, runtime notification, coordination and CPU execution unpriced"}


def compatibility_identity(profile):
    """Optional cost evidence can evolve without changing this measured backend."""
    return _hash({key: profile.get(key) for key in (
        "hardware", "toolchain", "precision_bits", "cpu_threads", "numerical_contract")})


def gpu_build_commands(paths, architecture):
    """One exact argv grammar; contradictory/hidden option variants have no slot."""
    if (not isinstance(paths, dict) or set(paths) != {"nvcc", "host", "runtime", "artifacts"}
            or any(not isinstance(p, str) or not 0 < len(p) <= 8192 or not Path(p).is_absolute()
                   or any(ord(c) < 32 for c in p) for p in paths.values())
            or not isinstance(architecture, str)
            or re.fullmatch(r"native|all|all-major|(?:sm|compute)_[0-9]{2,3}[af]?", architecture) is None):
        raise NumericalGpuExecutionError("GPU build path/architecture grammar invalid")
    target = Path(paths["artifacts"])
    flags = [paths["nvcc"], "-O3", "-std=c++17", *cuda_compile_options(), "-arch=" + architecture,
             "-ccbin", paths["host"], "-I", str(target), "-I", paths["runtime"]]
    return [[*flags, "-c", str(target / (name + ".cu")), "-o", str(target / (name + ".o"))]
            for name in ("workers", "driver")] + [
                [*flags, str(target / "workers.o"), str(target / "driver.o"), "-o", str(target / "numerical-gpu-execution")]]


def _digest_map(value, name, *, limit=512):
    if (not isinstance(value, dict) or not 1 <= len(value) <= limit
            or any(not isinstance(k, str) or not 0 < len(k) <= 8192 or not _sha(v) for k, v in value.items())):
        raise NumericalGpuExecutionError("GPU " + name + " digest inventory invalid")


def _identity(profile, identity, registry):
    required = {"kind": IDENTITY_KIND, "protocol_id": PROTOCOL_ID, "backend_id": BACKEND_ID,
                "precision_bits": profile["precision_bits"], "registry_id": _hash(registry),
                "numerical_contract": numerical_build_contract()}
    if any(identity.get(k) != v or type(identity.get(k)) is not type(v) for k, v in required.items()):
        raise NumericalGpuExecutionError("GPU source/protocol/precision/contract identity mismatch")
    for key in ("gpu_name", "gpu_uuid", "compute_capability"):
        if identity.get(key) != profile["hardware"].get(key):
            raise NumericalGpuExecutionError("GPU hardware identity mismatch: " + key)
    if identity.get("cpu_name") != profile["hardware"].get("cpu_name"):
        raise NumericalGpuExecutionError("GPU enqueue CPU identity mismatch")
    affinity = identity.get("sampling_cpu_affinity")
    if (not isinstance(affinity, list) or not 1 <= len(affinity) <= 4096
            or any(type(cpu) is not int or not 0 <= cpu < 1 << 20 for cpu in affinity)
            or affinity != sorted(set(affinity))):
        raise NumericalGpuExecutionError("GPU actual enqueue CPU placement missing")
    for key in ("cuda_runtime_version", "driver_version"):
        if type(identity.get(key)) is not int or identity[key] != profile["toolchain"].get(key):
            raise NumericalGpuExecutionError("GPU runtime identity mismatch: " + key)
    for key in ("nvcc_version", "host_cxx_version"):
        if identity.get(key) != profile["toolchain"].get(key):
            raise NumericalGpuExecutionError("GPU compiler identity mismatch: " + key)
    if type(identity.get("device_ordinal")) is not int or not 0 <= identity["device_ordinal"] < 1 << 16:
        raise NumericalGpuExecutionError("GPU selected ordinal invalid")
    _digest_map(identity.get("renderer_sources"), "renderer")
    _digest_map(identity.get("emitted_sources"), "emitted source")
    if set(identity["emitted_sources"]) != {"workers.cu", "gpu_recipes.hpp", "driver.cu"}:
        raise NumericalGpuExecutionError("GPU complete worker/registry/driver sources required")
    _digest_map(identity.get("objects"), "object")
    if set(identity["objects"]) != {"workers.o", "driver.o"}:
        raise NumericalGpuExecutionError("GPU complete worker/driver objects required")
    for key in ("source_generator_id", "generator_id", "binary_sha256"):
        if not _sha(identity.get(key)):
            raise NumericalGpuExecutionError("GPU " + key + " invalid")
    if identity["generator_id"] != gpu_generator_identity(
            identity["source_generator_id"], identity["registry_id"], identity["renderer_sources"]):
        raise NumericalGpuExecutionError("GPU generator identity differs from frozen source inventory")
    if identity.get("build_commands") != gpu_build_commands(identity.get("build_paths"), identity.get("architecture")):
        raise NumericalGpuExecutionError("GPU exact strict numerical build commands required")


def _row(row, recipe, size, precision):
    required = {"recipe": recipe["name"], "recipe_id": recipe["recipe_id"], "items": size,
                "backend": "gpu", "role": "fit" if size in recipe["fit_sizes"] else "holdout",
                "traffic_bytes": size * 3 * (precision // 8), "working_set_bytes": size * 3 * (precision // 8)}
    if not isinstance(row, dict) or any(row.get(k) != v or type(row.get(k)) is not type(v) for k, v in required.items()):
        raise NumericalGpuExecutionError("GPU recipe/size/role/physical coordinate mismatch")
    if type(row.get("agreement_passed")) is not bool:
        raise NumericalGpuExecutionError("GPU numerical agreement result missing")
    extra = {"kind", "agreement_passed", "control", "launches"} if row.get("kind") == PREFLIGHT_KIND else {
        "kind", "agreement_passed", "batch", "global_order", "repetitions", "launches",
        "resident_event_seconds", "enqueue_seconds", "wall_seconds"}
    if set(row) != set(required) | extra:
        raise NumericalGpuExecutionError("GPU raw schema differs from exact protocol")


def validate_gpu_execution_preflight(profile, identity, rows, registry):
    """Reject any incomplete/failed correctness inventory before timing starts."""
    try:
        require_numerical_build_contract(profile.get("numerical_contract"))
    except ValueError as error:
        raise NumericalGpuExecutionError(str(error)) from error
    _validate_registry(registry, profile["precision_bits"])
    if not isinstance(identity, dict):
        raise NumericalGpuExecutionError("GPU identity must be first")
    _identity(profile, identity, registry)
    if not isinstance(rows, list) or len(rows) != PREFLIGHT_RECORDS:
        raise NumericalGpuExecutionError("GPU complete preflight inventory required")
    offset = 0
    for recipe in registry:
        for control, sizes in (("tiny", TINY_SIZES), ("cell", recipe["sizes"])):
            for size in sizes:
                row = rows[offset]
                offset += 1
                _row(row, recipe, size, profile["precision_bits"])
                if (row.get("kind") != PREFLIGHT_KIND or row.get("control") != control
                        or type(row.get("launches")) is not int or row["launches"] != int(size > 0)
                        or row["agreement_passed"] is not True):
                    raise NumericalGpuExecutionError("GPU complete agreement before sampling required")


def _build(profile, records, registry, calibration):
    _hash(records)
    _hash(calibration)
    if not isinstance(records, list) or len(records) != MAX_RECORDS:
        raise NumericalGpuExecutionError("GPU complete identity/preflight/679-batch inventory required")
    identity = records[0]
    validate_gpu_execution_preflight(profile, identity, records[1:1 + PREFLIGHT_RECORDS], registry)
    if (not isinstance(calibration, dict) or calibration.get("application_profiled") is not False
            or calibration.get("automatic_placement_authorized") is not False
            or not _sha(calibration.get("base_profile_identity"))
            or calibration.get("compatibility_identity") != compatibility_identity(profile)):
        raise NumericalGpuExecutionError("GPU immutable base profile/no-application evidence required")
    affinity = identity["sampling_cpu_affinity"]
    target = Path(identity["build_paths"]["artifacts"])
    argv = ["taskset", "-c", ",".join(map(str, affinity)), str(target / "numerical-gpu-execution"),
            str(identity["device_ordinal"])]
    raw = calibration.get("raw_samples")
    if (calibration.get("sampling_cpu_affinity") != affinity
            or calibration.get("artifacts") != str(target)
            or calibration.get("run_commands") != [[*argv, mode] for mode in ("--identity", "--preflight", "--measure")]
            or calibration.get("cpu_evidence_revalidated") is not False
            or not isinstance(raw, dict) or set(raw) != {"path", "sha256", "bytes"}
            or raw["path"] != str(target / "gpu-raw-samples.jsonl") or not _sha(raw["sha256"])
            or type(raw["bytes"]) is not int or not 0 < raw["bytes"] <= 8 * 1024 * 1024):
        raise NumericalGpuExecutionError("GPU raw/run/fixed-placement provenance incomplete")
    environment = calibration.get("environment")
    if (not isinstance(environment, dict) or set(environment) != set(ENVIRONMENT_KEYS)
            or any(value is not None and (not isinstance(value, str) or len(value) > 8192)
                   for value in environment.values())
            or environment["FORT_PHASE_TIMING"] != "0" or environment["CUDA_LAUNCH_BLOCKING"] is not None):
        raise NumericalGpuExecutionError("GPU explicit unprofiled sampling environment required")
    offset = 1 + PREFLIGHT_RECORDS
    indexed = {(r["name"], size): [] for r in registry for size in r["sizes"]}
    order = 0
    for batch in range(SAMPLES):
        for recipe in registry:
            for size in recipe["sizes"]:
                row = records[offset]
                offset += 1
                _row(row, recipe, size, profile["precision_bits"])
                if (row.get("kind") != BATCH_KIND or type(row.get("batch")) is not int or row["batch"] != batch
                        or type(row.get("global_order")) is not int or row["global_order"] != order
                        or type(row.get("repetitions")) is not int or not 0 < row["repetitions"] < 1 << 63
                        or type(row.get("launches")) is not int or row["launches"] != row["repetitions"]):
                    raise NumericalGpuExecutionError("GPU global batch order/repetitions/launches invalid")
                order += 1
                for key in ("resident_event_seconds", "enqueue_seconds", "wall_seconds"):
                    _number(row.get(key), "GPU " + key)
                if (row["wall_seconds"] < MIN_BATCH_SECONDS or row["enqueue_seconds"] > row["wall_seconds"]
                        or row["resident_event_seconds"] > row["wall_seconds"] + max(.001, .01 * row["wall_seconds"])):
                    raise NumericalGpuExecutionError("GPU batch duration/event/enqueue accounting invalid")
                indexed[recipe["name"], size].append(row)
    observations = []
    for recipe in registry:
        for size in recipe["sizes"]:
            rows = indexed[recipe["name"], size]
            observations.append({"recipe": recipe["name"], "recipe_id": recipe["recipe_id"], "items": size,
                "status": "observed" if all(r["agreement_passed"] for r in rows) else "rejected",
                "reason_codes": [] if all(r["agreement_passed"] for r in rows) else ["numerical_agreement_failed"],
                **{"median_" + metric + "_per_launch": statistics.median(r[metric] / r["launches"] for r in rows)
                   for metric in ("resident_event_seconds", "enqueue_seconds", "wall_seconds")}})
    return {"schema_version": SCHEMA_VERSION, "phase": "gpu_evidence_only", "protocol_id": PROTOCOL_ID,
            "backend_id": BACKEND_ID, "identity": deepcopy(identity), "registry": deepcopy(list(registry)),
            "protocol": _protocol(), "measurements": deepcopy(records), "observations": observations,
            "preflight_cases": PREFLIGHT_RECORDS, "gpu_measured": True, "cost_model_available": False,
            "fixed_launch_price_available": False, "automatic_placement_authorized": False,
            "calibration": deepcopy(calibration)}


def profile_from_gpu_execution_measurements(profile, records, calibration):
    from compiler.offload.numerical_execution_workloads import execution_registry

    if SECTION in profile:
        raise NumericalGpuExecutionError("preserve existing GPU evidence; use an unmodified base profile")
    try:
        result = deepcopy(profile)
        result[SECTION] = _build(profile, records, list(execution_registry(profile["precision_bits"])), calibration)
        return result
    except (KeyError, TypeError, OverflowError, ValueError) as error:
        raise NumericalGpuExecutionError(str(error)) from error


def validate_gpu_execution_profile(profile):
    """Reconstruct historical evidence without lowering sources or using CUDA."""
    try:
        section = profile.get(SECTION)
        if not isinstance(section, dict) or type(section.get("schema_version")) is not int:
            raise NumericalGpuExecutionError("GPU execution section missing")
        rebuilt = _build(profile, section.get("measurements"), section.get("registry"), section.get("calibration"))
        if section != rebuilt:
            raise NumericalGpuExecutionError("saved GPU evidence differs from strict raw reconstruction")
    except (KeyError, TypeError, OverflowError, ValueError) as error:
        raise NumericalGpuExecutionError(str(error)) from error


def require_current_gpu_execution(profile):
    """Authenticate current observations, still granting no compute/placement model."""
    validate_gpu_execution_profile(profile)
    section = profile[SECTION]
    source, registry, generator, renderer = current_gpu_identities(profile["precision_bits"])
    identity = section["identity"]
    if (identity["source_generator_id"], identity["registry_id"], identity["generator_id"],
            identity["renderer_sources"]) != (source, registry, generator, renderer):
        raise NumericalGpuExecutionError("GPU current source/renderer identity mismatch")
    return deepcopy(section)
