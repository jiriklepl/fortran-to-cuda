"""Predetermined nonnegative source-host model; no execution or source authority."""

from __future__ import annotations

import json
import math
import statistics
from hashlib import sha256

BINDINGS = (1, 2, 3, 4, 6, 7, 8)
LAUNCHES = (1, 2, 4, 8, 16, 18, 32, 54, 64, 128, 256)
FIT_BINDINGS = (1, 4, 8)
FIT_LAUNCHES = (1, 4, 16, 64, 256)
ARGUMENT_WORDS = 128
COVERAGE_FACTOR = 4.0
METHOD = "nonnegative relative least squares: intercept + bindings + launches"
POLICY = "fixed engineering reserve; no statistical confidence or wall-time guarantee"


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _positive(value):
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and value > 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("source submit model requires finite positive observations")
    return float(value)






def _predict(model, bindings, launches):
    if (type(bindings) is not int or type(launches) is not int
            or not 1 <= bindings <= model["max_bindings"] or not 1 <= launches <= model["max_launches"]):
        raise ValueError("source submit model geometry exceeds qualified bounds")
    c = model["coefficients"]
    return _positive(model["coverage_factor"] *
                     (c["intercept_seconds"] + bindings*c["binding_seconds"] + launches*c["launch_seconds"]))


def _observations(records):
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise ValueError("source submit model requires typed observation rows")
    observations = {}
    for row in records:
        if row.get("name") != "source_submit_host_enqueue_included":
            continue
        if (any(type(row.get(key)) is not int for key in ("bytes", "bindings", "launches", "rounds"))
                or row["bytes"] != 0 or row["rounds"] != 32
                or row["bindings"] not in BINDINGS or row["launches"] not in LAUNCHES):
            raise ValueError("source submit model has an out-of-grid row")
        key = row["bindings"], row["launches"]
        if key in observations or not isinstance(row.get("seconds"), list) or len(row["seconds"]) != 5:
            raise ValueError("source submit model has a duplicate or incomplete row")
        observations[key] = statistics.median(_positive(value) for value in row["seconds"])
    if set(observations) != {(b, n) for b in BINDINGS for n in LAUNCHES}:
        raise ValueError("source submit model fixed observation grid is incomplete")
    return observations


def _coverage(model, observations):
    fit_failed, held_failed, diagnostics = [], [], []
    for (bindings, launches), observed in observations.items():
        qualified = _predict(model, bindings, launches)
        fitted = qualified/model["coverage_factor"]
        fit_point = bindings in FIT_BINDINGS and launches in FIT_LAUNCHES
        if observed > qualified:
            (fit_failed if fit_point else held_failed).append([bindings, launches])
        diagnostics.append({"bindings": bindings, "launches": launches, "fit_point": fit_point,
            "observed_median_seconds": observed, "fitted_seconds": fitted,
            "reserved_seconds": qualified, "relative_residual": (fitted-observed)/observed,
            "covered": observed <= qualified})
    return fit_failed, held_failed, diagnostics




def validate_model(model, profile, runtime_id):
    if (not isinstance(model, dict) or model.get("model_identity") != _digest({k: v for k, v in model.items() if k != "model_identity"})
            or type(model.get("version")) is not int or model["version"] != 1 or model.get("method") != METHOD
            or model.get("runtime_id") != runtime_id or model.get("hardware_identity") != profile["hardware"]
            or model.get("precision_bits") != profile["precision_bits"] or model.get("cpu_threads") != profile["cpu_threads"]
            or type(model.get("available")) is not bool or model.get("application_profiled") is not False
            or model.get("coverage_factor") != COVERAGE_FACTOR or model.get("coverage_policy") != POLICY
            or model.get("fit_bindings") != list(FIT_BINDINGS) or model.get("fit_launches") != list(FIT_LAUNCHES)
            or model.get("observed_bindings") != list(BINDINGS) or model.get("observed_launches") != list(LAUNCHES)
            or model.get("kernel_argument_words") != ARGUMENT_WORDS or model.get("kernel_argument_bytes") != 8*ARGUMENT_WORDS
            or model.get("block_threads") != 256 or model.get("calibration_grid_blocks") != 1
            or model.get("max_bindings") != max(BINDINGS) or model.get("max_launches") != max(LAUNCHES)):
        raise ValueError("source submit model source/protocol/runtime identity differs")
    expected = {"version": 1, "gpu_uuid": profile["hardware"]["gpu_uuid"],
        "device": profile["hardware"]["device_ordinal"], "cpu_threads": profile["cpu_threads"],
        "precision": profile["precision_bits"], **{k: profile["toolchain"][k]
            for k in ("cuda_runtime_version", "driver_version")},
        "kernel_argument_words": ARGUMENT_WORDS, "kernel_argument_bytes": 8*ARGUMENT_WORDS}
    identity = model.get("identity")
    if not isinstance(identity, dict) or any(identity.get(k) != v or type(identity.get(k)) is not type(v)
                                               for k, v in expected.items()):
        raise ValueError("source submit model measurement identity differs")
    coefficients = model.get("coefficients")
    if (not isinstance(coefficients, dict) or set(coefficients) != {"intercept_seconds", "binding_seconds", "launch_seconds"}
            or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in coefficients.values())):
        raise ValueError("source submit model requires nonnegative finite coefficients")
    _predict(model, 1, 1)
    observed = _observations(model.get("measurements"))
    fit_failed, held_failed, diagnostics = _coverage(model, observed)
    if (model.get("failed_fit_points") != fit_failed or model.get("failed_holdouts") != held_failed
            or model.get("coverage_diagnostics") != diagnostics
            or model["available"] != (not (fit_failed or held_failed))):
        raise ValueError("source submit model observational coverage differs")
    return model
