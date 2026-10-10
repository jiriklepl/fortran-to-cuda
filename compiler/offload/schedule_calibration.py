"""Optional validation of runtime-static native work against frozen compute costs.

This evidence never fits coefficients or changes the ordinary numerical v2
profile. Every requested numerical family must independently validate using
the original Fortran backend, placement and static schedule with chunk zero.
"""

import math
import statistics
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

from compiler.offload.numerical_calibration import (
    COMPUTE_CLASSES,
    COMPUTE_FAMILIES,
    COMPUTE_RECIPES,
    COMPUTE_SIZES,
    MAX_HOLDOUT_RELATIVE_ERROR,
    MEMORY_SIZES,
    NumericalCalibrationError,
    _compute_components,
    _compute_fixed,
    _compute_samples,
    memory_compute_seconds,
    native_fixture_source,
    validate_numerical_profile,
)

SCHEDULE_PROTOCOL_ID = "runtime-static-chunk0-v1"
RUNTIME_SCHEDULE = {"kind": "static", "chunk": 0}


def runtime_fixture_source(precision):
    """Render the same generic recipes with a distinct runtime-scheduled ABI."""
    return (native_fixture_source(precision)
            .replace("module numerical_native_v2", "module numerical_native_runtime_v2")
            .replace("fort_numerical_native_v2", "fort_numerical_native_runtime_v1")
            .replace("fort_numerical_fortran_identity_v2", "fort_numerical_runtime_fortran_identity_v1")
            .replace("schedule(static)", "schedule(runtime)"))


def runtime_fixture_identity(precision):
    return sha256(runtime_fixture_source(precision).encode()).hexdigest()


def schedule_protocol_identity():
    try:
        return sha256(Path(__file__).with_name("schedule_calibrate.cpp").read_bytes()).hexdigest()
    except OSError as error:
        raise NumericalCalibrationError("runtime schedule validation protocol source unavailable") from error


def _execution_identity(profile, identity):
    from compiler.offload.collective_calibration import normalize_fortran_options

    numerical = profile["numerical"]["identity"]
    if not isinstance(identity, dict):
        raise NumericalCalibrationError("runtime schedule execution identity unavailable")
    for name in ("cpu_threads", "precision_bits"):
        if type(identity.get(name)) is not int or identity[name] != profile[name]:
            raise NumericalCalibrationError("runtime schedule " + name + " mismatch")
    affinity = identity.get("cpu_affinity")
    if (not isinstance(affinity, list) or any(type(cpu) is not int or cpu < 0 for cpu in affinity)
            or affinity != sorted(set(affinity)) or affinity != numerical["cpu_affinity"]):
        raise NumericalCalibrationError("runtime schedule CPU placement mismatch")
    if identity.get("omp_dynamic") is not False or identity.get("omp_proc_bind") != "false":
        raise NumericalCalibrationError("runtime schedule OpenMP placement mismatch")
    schedule = identity.get("runtime_schedule")
    if (not isinstance(schedule, dict) or schedule != RUNTIME_SCHEDULE
            or type(schedule.get("chunk")) is not int):
        raise NumericalCalibrationError("runtime schedule requires static worksharing with chunk zero")
    fortran = identity.get("fortran")
    if not isinstance(fortran, dict):
        raise NumericalCalibrationError("runtime schedule requires both native Fortran object identities")
    for mode in ("static", "runtime"):
        actual = fortran.get(mode)
        if not isinstance(actual, dict):
            raise NumericalCalibrationError("runtime schedule native Fortran identity unavailable")
        for name in ("compiler_version", "semantic_options"):
            if actual.get(name) != numerical["fortran"][name]:
                raise NumericalCalibrationError("runtime schedule native Fortran " + name + " mismatch")
        options = actual.get("compiler_options")
        if not isinstance(options, str) or normalize_fortran_options(options) != actual["semantic_options"]:
            raise NumericalCalibrationError("runtime schedule native Fortran option normalization mismatch")


def schedule_prediction_seconds(profile, family, items):
    """Use the emitted total-compute formula, including one CPU fixed term."""
    recipe = COMPUTE_RECIPES[family]
    families = profile["numerical"]["families"]
    components = _compute_components(families, "native_fork_join", profile["precision_bits"],
                                     tuple(recipe["intrinsics"]))
    if components is None:
        raise NumericalCalibrationError("required original native numerical families rejected")
    slope = recipe["arithmetic"] * components["arithmetic_seconds_per_operation"] + math.fsum(
        count * components["intrinsic_seconds_per_operation"][name]
        for name, count in recipe["intrinsics"].items())
    traffic = items * recipe.get("memory_arrays", 2) * (profile["precision_bits"] // 8)
    memory = memory_compute_seconds(components["memory_cost_model"], traffic, traffic)
    seconds = _compute_fixed(families, "native_fork_join") + max(items * slope, memory)
    if not math.isfinite(seconds) or seconds <= 0:
        raise NumericalCalibrationError("runtime schedule prediction is unavailable")
    return seconds


def _structural_range(profile, family, sizes):
    """Intersect fixed sample coordinates with calibrated physical applicability.

    This does not examine timing errors. Rejected in-range observations cannot
    shrink the interval or disappear from validation.
    """
    memory = profile["numerical"]["families"]["memory_v2"]["native_fork_join"].get("model", {})
    if memory.get("kind") != "piecewise_bandwidth_v1":
        return list(sizes)
    lower, upper = memory["working_set_range"]
    bytes_per_item = COMPUTE_RECIPES[family].get("memory_arrays", 2) * (profile["precision_bits"] // 8)
    return [items for items in sizes if lower <= items * bytes_per_item <= upper]


def _validation(profile, records):
    if not isinstance(records, list) or len(records) > len(COMPUTE_FAMILIES) * len(MEMORY_SIZES):
        raise NumericalCalibrationError("runtime schedule observations exceed bounded protocol")
    indexed = {}
    for row in records:
        if not isinstance(row, dict) or row.get("kind") != "schedule_cost_v1":
            raise NumericalCalibrationError("unknown runtime schedule observation")
        family, items = row.get("family"), row.get("items")
        sizes = MEMORY_SIZES if family == "memory_v2" else COMPUTE_SIZES
        if family not in COMPUTE_FAMILIES or type(items) is not int or items not in sizes:
            raise NumericalCalibrationError("unknown runtime schedule family or size")
        key = family, items
        if key in indexed:
            raise NumericalCalibrationError("duplicate runtime schedule observation")
        _compute_samples(row)
        _compute_samples({"samples": row.get("static_samples")})
        indexed[key] = row
    result = {}
    for family in COMPUTE_FAMILIES:
        sizes = MEMORY_SIZES if family == "memory_v2" else COMPUTE_SIZES
        admitted = _structural_range(profile, family, sizes)
        validation = {"status": "rejected", "reason_codes": [], "holdouts": [],
                      "item_range": [admitted[0], admitted[-1]] if len(admitted) >= 2 else None}
        if len(admitted) < 2:
            validation["reason_codes"].append("insufficient_structural_range")
        for items in sizes:
            row = indexed.get((family, items))
            if row is None:
                validation["reason_codes"].append("missing_observation")
                continue
            if row.get("agreement_passed") is not True:
                validation["reason_codes"].append("numerical_agreement_failed")
            runtime = statistics.median(_compute_samples(row))
            static = statistics.median(_compute_samples({"samples": row["static_samples"]}))
            observation = {"items": items, "runtime_seconds": runtime, "static_seconds": static}
            if items not in admitted:
                observation.update(predicted_seconds=None, relative_error=None, reason="out_of_memory_range")
                validation["holdouts"].append(observation)
                continue
            try:
                predicted = schedule_prediction_seconds(profile, family, items)
            except NumericalCalibrationError as error:
                observation.update(predicted_seconds=None, relative_error=None, reason=str(error))
                validation["reason_codes"].append("prediction_unavailable")
            else:
                error = abs(predicted - runtime) / runtime
                observation.update(predicted_seconds=predicted, relative_error=error)
                if error > MAX_HOLDOUT_RELATIVE_ERROR:
                    validation["reason_codes"].append("runtime_prediction_error")
            validation["holdouts"].append(observation)
        validation["reason_codes"] = sorted(set(validation["reason_codes"]))
        validation["status"] = "rejected" if validation["reason_codes"] else "accepted"
        result[family] = validation
    return result


def profile_from_schedule_measurements(profile, records, execution_identity, *, calibration=None):
    """Keep rejected observations; produce a separate optional profile section."""
    section = profile.get("numerical")
    if not isinstance(section, dict) or section.get("schema_version") != 2:
        raise NumericalCalibrationError("runtime schedule validation requires native numerical v2")
    validate_numerical_profile(section, profile)
    _execution_identity(profile, execution_identity)
    validation = _validation(profile, records)
    result = deepcopy(profile)
    result["source_schedule_validation"] = {
        "schema_version": 1, "protocol_id": SCHEDULE_PROTOCOL_ID,
        "numerical_identity": deepcopy(section["identity"]), "numerical_generator_id": section["generator_id"],
        "source_sha256": runtime_fixture_identity(profile["precision_bits"]),
        "protocol_source_sha256": schedule_protocol_identity(),
        "execution_identity": deepcopy(execution_identity), "measurements": deepcopy(records),
        "families": validation, "maximum_relative_error": MAX_HOLDOUT_RELATIVE_ERROR,
        "calibration": {**(calibration or {}), "application_profiled": False, "coefficients_fitted": False}}
    return result


def runtime_schedule_requirement(profile, model, primitives):
    """Authenticate requested families from raw observations, without fitting."""
    section = profile.get("source_schedule_validation")
    numerical = profile.get("numerical")
    if not isinstance(section, dict):
        raise NumericalCalibrationError("runtime schedule validation is missing")
    if (type(section.get("schema_version")) is not int or section["schema_version"] != 1
            or section.get("protocol_id") != SCHEDULE_PROTOCOL_ID):
        raise NumericalCalibrationError("runtime schedule validation protocol mismatch")
    if (section.get("numerical_identity") != numerical["identity"]
            or section.get("numerical_generator_id") != numerical["generator_id"]
            or section.get("source_sha256") != runtime_fixture_identity(profile["precision_bits"])
            or section.get("protocol_source_sha256") != schedule_protocol_identity()
            or section.get("maximum_relative_error") != MAX_HOLDOUT_RELATIVE_ERROR):
        raise NumericalCalibrationError("runtime schedule validation source or numerical identity mismatch")
    _execution_identity(profile, section.get("execution_identity"))
    reconstructed = _validation(profile, section.get("measurements"))
    if section.get("families") != reconstructed:
        raise NumericalCalibrationError("runtime schedule validation differs from raw observations")
    validation_class = ("ordinary_expression_v2" if model["workload_class"] == "scalar_expression_v2"
                        and not any(name != "divide" for name in primitives) else model["workload_class"])
    required = ("arithmetic_v2", "memory_v2", *("primitive_" + name + "_v2" for name in primitives),
                *COMPUTE_CLASSES[validation_class])
    lower, upper = model["item_range"]
    for family in required:
        validation = reconstructed[family]
        if validation["status"] != "accepted":
            raise NumericalCalibrationError("runtime schedule " + family + " rejected: " +
                                            ",".join(validation["reason_codes"]))
        lower, upper = max(lower, validation["item_range"][0]), min(upper, validation["item_range"][1])
    if lower > upper:
        raise NumericalCalibrationError("runtime schedule validated item ranges do not intersect")
    return {"runtime_schedule": dict(RUNTIME_SCHEDULE), "item_range": [lower, upper],
            "source_schedule_validation": {"protocol_id": SCHEDULE_PROTOCOL_ID,
                "source_sha256": section["source_sha256"], "protocol_source_sha256": section["protocol_source_sha256"],
                "required_families": list(required), "range_intersection": "structural physical applicability only"}}
