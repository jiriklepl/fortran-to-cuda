"""Optional, exact-family numerical calibration with independent holdouts.

These workloads measure costs which an FMA-throughput profile cannot supply.
They deliberately do not classify arbitrary expressions as a calibrated mix:
an emitter must provide the matching recipe, backend and generator identities.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import statistics

SCHEMA_VERSION = 1
BACKEND_ID = "standalone-cuda-openmp-cxx17-v1"
INTRINSIC_BACKEND_ID = "cxx17-scalar-real-math-v1"
PRIMITIVES = ("sqrt", "acos", "cos")
PRIMITIVE_STEPS = 16
MIX_FAMILIES = {"angle_mix_v1": {"sqrt": 8, "acos": 1, "cos": 3},
                "angle_mix_skew_v1": {"sqrt": 2, "acos": 3, "cos": 1}}
FAMILIES = ("transcendental_chain_v1", "private_matrix_v1",
            *("primitive_" + name + "_v1" for name in PRIMITIVES), *MIX_FAMILIES)
FIT_SIZES = (16384, 65536, 262144)
HOLDOUT_SIZES = (32768, 131072)
MAX_HOLDOUT_RELATIVE_ERROR = 0.25
HARDWARE_FIELDS = ("cpu_name", "gpu_uuid", "gpu_name", "compute_capability")
TOOLCHAIN_FIELDS = ("nvcc_version", "host_cxx_version", "cuda_runtime_version", "driver_version")


class NumericalCalibrationError(ValueError):
    """Missing, incompatible or unvalidated numerical measurements."""


def apply_numerical_costs(analysis, profile):
    """Resolve static counts only for the independently validated scalar backend."""
    units = []
    for unit in analysis.units:
        if (unit.intrinsic_work_per_iteration and unit.arithmetic_work_per_iteration is not None
                and not unit.work_is_upper_bound and profile is not None):
            try:
                costs = numerical_intrinsic_costs(profile, unit.intrinsic_work_per_iteration,
                    backend_id=INTRINSIC_BACKEND_ID, generator_id=generator_identity())
                unit = replace(unit, work_per_iteration=unit.arithmetic_work_per_iteration,
                    cpu_numerical_seconds_per_iteration=costs["cpu_seconds_per_iteration"],
                    gpu_numerical_seconds_per_iteration=costs["gpu_seconds_per_iteration"],
                    work_estimate_reason=None)
            except NumericalCalibrationError as error:
                unit = replace(unit, work_estimate_reason=str(error))
        units.append(unit)
    return replace(analysis, units=tuple(units))


def generator_identity() -> str:
    """Identify both the fixture implementation and its fitting protocol."""
    root = Path(__file__).parent
    return sha256(b"\n".join((root / name).read_bytes() for name in
                             ("numerical_calibration.py", "numerical_calibration.cu"))).hexdigest()


def recipe_identity(family: str, *, generator_id: str | None = None) -> str:
    if family not in FAMILIES:
        raise NumericalCalibrationError("unknown numerical workload family")
    return sha256((BACKEND_ID + ":" + (generator_id or generator_identity()) + ":" + family).encode()).hexdigest()


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise NumericalCalibrationError(name + " must be finite and positive")
    return float(value)


def _median(record):
    durations = record.get("seconds")
    if not isinstance(durations, list) or len(durations) != 5:
        raise NumericalCalibrationError("numerical observations require five duration samples")
    return statistics.median(_positive(value, "numerical duration") for value in durations)


def parse_measurements(text: str) -> list[dict]:
    records = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError as error:
            raise NumericalCalibrationError("numerical benchmark emitted non-JSON output") from error
        if not isinstance(record, dict) or record.get("kind") not in {"numerical_identity", "numerical_cost"}:
            raise NumericalCalibrationError("unknown numerical measurement kind")
        if record["kind"] == "numerical_cost":
            _median(record)
        records.append(record)
    if not records:
        raise NumericalCalibrationError("numerical benchmark emitted no observations")
    return records


def _fit(records):
    """Fit only training points; neither holdout can affect coefficients."""
    points = [(record["items"], _median(record)) for record in records]
    mean_x = statistics.mean(size for size, _ in points)
    mean_y = statistics.mean(value for _, value in points)
    slope = math.fsum((size - mean_x) * (value - mean_y) for size, value in points) / math.fsum(
        (size - mean_x) ** 2 for size, _ in points)
    intercept = mean_y - slope * mean_x
    if intercept < 0:
        intercept = 0.0
        slope = math.fsum(size * value for size, value in points) / math.fsum(size * size for size, _ in points)
    _positive(slope, "numerical seconds per item")
    return {"fixed_seconds": intercept, "seconds_per_item": slope}


def profile_from_measurements(profile: dict, records: list[dict], *, calibration: dict) -> dict:
    """Attach validated optional costs without changing any existing rates."""
    identities = [record for record in records if record.get("kind") == "numerical_identity"]
    if len(identities) != 1:
        raise NumericalCalibrationError("exactly one numerical identity is required")
    identity = identities[0]
    for name in ("gpu_uuid", "gpu_name", "compute_capability"):
        if identity.get(name) != profile["hardware"][name]:
            raise NumericalCalibrationError("numerical hardware " + name + " mismatch")
    for name in ("cuda_runtime_version", "driver_version"):
        if identity.get(name) != profile["toolchain"][name]:
            raise NumericalCalibrationError("numerical toolchain " + name + " mismatch")
    for name in ("precision_bits", "cpu_threads"):
        if type(identity.get(name)) is not int or identity[name] != profile[name]:
            raise NumericalCalibrationError("numerical " + name + " mismatch")
    if identity.get("backend_id") != BACKEND_ID:
        raise NumericalCalibrationError("numerical backend mismatch")
    indexed = {}
    for record in records:
        if record.get("kind") == "numerical_identity":
            continue
        family, device, items = (record.get(name) for name in ("family", "device", "items"))
        if (record.get("kind") != "numerical_cost" or family not in FAMILIES or device not in {"cpu", "gpu"}
                or type(items) is not int or items not in FIT_SIZES + HOLDOUT_SIZES):
            raise NumericalCalibrationError("unknown numerical workload observation")
        expected_role = "fit" if items in FIT_SIZES else "holdout"
        if record.get("role") != expected_role:
            raise NumericalCalibrationError("numerical fit/holdout role mismatch")
        key = (family, device, items)
        if key in indexed:
            raise NumericalCalibrationError("duplicate numerical workload observation")
        _median(record)
        if record.get("agreement_passed") is not True:
            raise NumericalCalibrationError("numerical workload CPU/GPU agreement failed")
        indexed[key] = record
    expected = {(family, device, items) for family in FAMILIES for device in ("cpu", "gpu")
                for items in FIT_SIZES + HOLDOUT_SIZES}
    if set(indexed) != expected:
        raise NumericalCalibrationError("missing numerical workload observations")
    generator_id = generator_identity()
    families = {}
    for family in FAMILIES:
        fitted = {}
        for device in ("cpu", "gpu"):
            model = _fit([indexed[family, device, size] for size in FIT_SIZES])
            validations = []
            for size in HOLDOUT_SIZES:
                actual = _median(indexed[family, device, size])
                predicted = model["fixed_seconds"] + size * model["seconds_per_item"]
                relative_error = abs(predicted - actual) / actual
                if relative_error > MAX_HOLDOUT_RELATIVE_ERROR:
                    raise NumericalCalibrationError(f"{family} {device} independent holdout exceeds 25% error")
                validations.append({"items": size, "predicted_seconds": predicted,
                                    "measured_seconds": actual, "relative_error": relative_error})
            fitted[device] = {**model, "holdouts": validations}
        families[family] = {"recipe_id": recipe_identity(family, generator_id=generator_id),
                            "fit_sizes": list(FIT_SIZES), "holdout_sizes": list(HOLDOUT_SIZES),
                            "item_range": [min(FIT_SIZES), max(FIT_SIZES)], **fitted}
    result = deepcopy(profile)
    result["numerical"] = {"schema_version": SCHEMA_VERSION, "backend_id": BACKEND_ID,
        "generator_id": generator_id, "precision_bits": profile["precision_bits"], "cpu_threads": profile["cpu_threads"],
        "hardware": {name: profile["hardware"][name] for name in HARDWARE_FIELDS},
        "toolchain": {name: profile["toolchain"][name] for name in TOOLCHAIN_FIELDS}, "families": families,
        "holdout_max_relative_error": MAX_HOLDOUT_RELATIVE_ERROR,
        "applicability": "exact recipe, backend and generator identity only; arbitrary intrinsic mixes remain unestimated",
        "measurements": deepcopy(records), "calibration": {**calibration, "application_profiled": False}}
    result["numerical"]["intrinsic_model"] = _intrinsic_model(families, generator_id)
    validate_numerical_profile(result["numerical"], result)
    return result


def _intrinsic_model(families: dict, generator_id: str) -> dict:
    """Validate additive primitive costs against two unfit expression mixes.

    Each measured primitive includes its surrounding bounded arithmetic; this
    deliberately charges extra arithmetic rather than subtracting an unrelated
    FMA benchmark. Mixed holdouts never adjust any primitive coefficient.
    """
    rates = {device: {name: families["primitive_" + name + "_v1"][device]["seconds_per_item"] / PRIMITIVE_STEPS
                      for name in PRIMITIVES} for device in ("cpu", "gpu")}
    validations, reasons = [], []
    for family, counts in MIX_FAMILIES.items():
        for device in ("cpu", "gpu"):
            variable = PRIMITIVE_STEPS * math.fsum(counts[name] * rates[device][name] for name in PRIMITIVES)
            # All operations share one launch/team. The largest primitive
            # fixed cost is a conservative estimate of that single startup.
            fixed = max(families["primitive_" + name + "_v1"][device]["fixed_seconds"] for name in PRIMITIVES)
            for observed in families[family][device]["holdouts"]:
                predicted = fixed + observed["items"] * variable
                actual = observed["measured_seconds"]
                error = abs(predicted - actual) / actual
                validations.append({"family": family, "device": device, "items": observed["items"],
                                    "predicted_seconds": predicted, "measured_seconds": actual,
                                    "relative_error": error})
                if error > MAX_HOLDOUT_RELATIVE_ERROR:
                    reasons.append(f"{family} {device} mixed holdout at {observed['items']} exceeds 25% error")
    return {"backend_id": INTRINSIC_BACKEND_ID, "generator_id": generator_id,
            "available": not reasons, "reasons": reasons, "seconds_per_operation": rates,
            "validated_intrinsics": list(PRIMITIVES), "mixed_holdouts": validations,
            "arithmetic_policy": "primitive coefficients include bounded fixture arithmetic; retain ordinary work separately"}


def validate_numerical_profile(section: dict, profile: dict) -> None:
    """Validate saved optional evidence without requiring a current generator."""
    if not isinstance(section, dict) or type(section.get("schema_version")) is not int or section["schema_version"] != SCHEMA_VERSION:
        raise NumericalCalibrationError("numerical costs require schema_version 1")
    if section.get("backend_id") != BACKEND_ID:
        raise NumericalCalibrationError("numerical backend mismatch")
    generator = section.get("generator_id")
    if not isinstance(generator, str) or len(generator) != 64 or any(c not in "0123456789abcdef" for c in generator):
        raise NumericalCalibrationError("numerical generator_id must be a lowercase SHA-256")
    for name in ("precision_bits", "cpu_threads", "hardware", "toolchain"):
        expected = ({field: profile[name][field] for field in (HARDWARE_FIELDS if name == "hardware" else TOOLCHAIN_FIELDS)}
                    if name in {"hardware", "toolchain"} else profile[name])
        if section.get(name) != expected:
            raise NumericalCalibrationError("numerical " + name + " mismatch")
    if section.get("holdout_max_relative_error") != MAX_HOLDOUT_RELATIVE_ERROR:
        raise NumericalCalibrationError("numerical holdout threshold mismatch")
    families = section.get("families")
    if not isinstance(families, dict) or set(families) != set(FAMILIES):
        raise NumericalCalibrationError("missing numerical workload families")
    for family, costs in families.items():
        if not isinstance(costs, dict) or costs.get("recipe_id") != recipe_identity(family, generator_id=generator):
            raise NumericalCalibrationError("numerical recipe identity mismatch")
        if (costs.get("fit_sizes") != list(FIT_SIZES) or costs.get("holdout_sizes") != list(HOLDOUT_SIZES)
                or costs.get("item_range") != [min(FIT_SIZES), max(FIT_SIZES)]):
            raise NumericalCalibrationError("numerical workload range mismatch")
        for device in ("cpu", "gpu"):
            model = costs.get(device)
            if not isinstance(model, dict):
                raise NumericalCalibrationError("missing numerical device costs")
            _positive(model.get("seconds_per_item"), "numerical seconds per item")
            fixed = model.get("fixed_seconds")
            if (isinstance(fixed, bool) or not isinstance(fixed, (int, float)) or not math.isfinite(fixed) or fixed < 0):
                raise NumericalCalibrationError("numerical fixed seconds must be finite and nonnegative")
            holdouts = model.get("holdouts")
            if not isinstance(holdouts, list) or [row.get("items") for row in holdouts if isinstance(row, dict)] != list(HOLDOUT_SIZES):
                raise NumericalCalibrationError("missing numerical independent holdouts")
            for row in holdouts:
                actual = _positive(row.get("measured_seconds"), "holdout duration")
                predicted = _positive(row.get("predicted_seconds"), "holdout prediction")
                expected = fixed + row["items"] * model["seconds_per_item"]
                error = abs(predicted - actual) / actual
                reported_error = row.get("relative_error")
                if (isinstance(reported_error, bool) or not isinstance(reported_error, (int, float))
                        or not math.isfinite(reported_error) or reported_error < 0):
                    raise NumericalCalibrationError("numerical holdout error must be finite and nonnegative")
                if not math.isclose(predicted, expected, rel_tol=1e-12) or not math.isclose(reported_error, error, rel_tol=1e-12, abs_tol=1e-15) or error > MAX_HOLDOUT_RELATIVE_ERROR:
                    raise NumericalCalibrationError("numerical independent holdout is inconsistent or failed")
    if section.get("intrinsic_model") != _intrinsic_model(families, generator):
        raise NumericalCalibrationError("numerical intrinsic model differs from its independent mixed holdouts")


def numerical_costs(profile: dict, family: str, *, backend_id: str, generator_id: str, recipe_id: str) -> dict:
    """Expose a cost only to the exact implementation whose work was measured."""
    section = profile.get("numerical")
    if section is None:
        raise NumericalCalibrationError("no numerical calibration")
    validate_numerical_profile(section, profile)
    if (backend_id != section["backend_id"] or generator_id != section["generator_id"]
            or family not in section["families"] or recipe_id != section["families"][family]["recipe_id"]):
        raise NumericalCalibrationError("numerical implementation identity mismatch; estimate unavailable")
    return deepcopy(section["families"][family])


def numerical_intrinsic_costs(profile: dict, counts, *, backend_id: str, generator_id: str) -> dict:
    """Return additional seconds per item for an independently validated class.

    Unknown intrinsics and retained-loop/conditional work must remain unknown;
    this API accepts only statically proved counts of the named scalar math
    primitives. A C++/CUDA scalar-math backend must explicitly identify itself.
    """
    section = profile.get("numerical")
    if section is None:
        raise NumericalCalibrationError("no numerical calibration; intrinsic estimate unavailable")
    validate_numerical_profile(section, profile)
    model = section["intrinsic_model"]
    if backend_id != model["backend_id"] or generator_id != model["generator_id"]:
        raise NumericalCalibrationError("numerical intrinsic implementation identity mismatch")
    if model["available"] is not True:
        raise NumericalCalibrationError("numerical mixed holdout validation failed; intrinsic estimate unavailable")
    pairs = list(counts.items()) if isinstance(counts, dict) else list(counts)
    if (not pairs or len({name for name, _ in pairs}) != len(pairs)
            or any(name not in PRIMITIVES or type(count) is not int or count <= 0 for name, count in pairs)):
        raise NumericalCalibrationError("unsupported or unknown intrinsic counts; estimate unavailable")
    return {device + "_seconds_per_iteration": math.fsum(count * model["seconds_per_operation"][device][name]
                                                        for name, count in pairs) for device in ("cpu", "gpu")}


def calibrate_numerical(profile, args, directory: Path, nvcc: str, host: str, *, run) -> dict:
    target = directory / "numerical"
    target.mkdir(exist_ok=True)
    source = Path(__file__).with_suffix(".cu")
    binary = target / "numerical-calibration"
    command = [nvcc, "-O3", "-std=c++17", "-arch=" + args.arch, "-ccbin", host,
               "-Xcompiler=-fopenmp", "-DCALIBRATION_PRECISION=" + str(args.precision), str(source), "-o", str(binary)]
    run(command, target, target / "build.log", timeout=180)
    environment = {name: value for name, value in os.environ.items()
                   if name not in {"FORT_RUNTIME_TRACE", "FORT_PHASE_TIMING", "CUDA_LAUNCH_BLOCKING"}
                   and not name.startswith("FORT_SCOPE_TEST_")}
    invocation = [str(binary), str(args.threads), str(args.device)]
    text = run(invocation, target, target / "measurements.jsonl", timeout=180, env=environment)
    return profile_from_measurements(profile, parse_measurements(text), calibration={
        "build_command": command, "run_command": invocation, "artifacts": str(target),
        "benchmark_source_sha256": sha256(source.read_bytes()).hexdigest(),
        "cpu_method": "wall time, fixed-budget OpenMP parallel for, five warmed measurements",
        "gpu_method": "CUDA events, resident input/output, five warmed measurements; transfer costs separate",
        "fit_method": "nonnegative linear fit on three fixed sizes; two independent sizes validate within 25%",
        "semantic_flags": "-O3 without fast-math; identical bounded expressions on CPU and GPU"})
