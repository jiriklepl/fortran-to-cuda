"""Complete physical-batch predictions; empirical uncertainty is diagnostic."""
from __future__ import annotations

import math

SCHEMA_VERSION = 2
PAYLOADS = (256 * 1024, 1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024)
OPERATION_LIMIT = 256
PATHS = ("direct", "compact_cold", "compact_reuse")
COEFFICIENTS = ("batch_seconds", "submission_seconds", "byte_seconds", "row_seconds", "descriptor_seconds")
ERROR_FIELDS = ("median_relative_error", "p90_relative_error", "maximum_relative_error")
RATE_COVERAGE_FIELDS = (
    "minimum_payload_bytes", "maximum_payload_bytes", "minimum_rows", "maximum_rows",
    "minimum_row_bytes", "maximum_row_bytes", "minimum_pitch_bytes", "maximum_pitch_bytes",
    "maximum_span_bytes", "minimum_operations", "maximum_operations",
    "minimum_descriptor_work", "maximum_descriptor_work", "minimum_kernel_chunks", "maximum_kernel_chunks",
    "word_bytes_mask", "layout_combinations_mask",
)
PREDICTION_KIND = "complete_physical_batch"
FIT_METHOD = "nonnegative_relative_least_squares_of_geometry_medians_v2"
RESIDUAL_CONVENTION = "absolute_relative_to_observation"
MINIMUM_RELATIVE_SAVING = 0.5


class PhysicalCopyCostError(ValueError):
    """A complete-path calibration is malformed or unsupported."""


def _number(value, label, *, zero=False):
    try:
        finite = math.isfinite(value)
    except (TypeError, OverflowError):
        finite = False
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not finite or value < 0 or not zero and value == 0):
        raise PhysicalCopyCostError(label + " must be finite and " + ("nonnegative" if zero else "positive"))
    return float(value)


def validate_physical_copy_costs(profile):
    record = profile.get("scoped", {}).get("physical_copies")
    prefix = "scoped.physical_copies"
    if not isinstance(record, dict):
        raise PhysicalCopyCostError("hardware profile has no physical copy calibration")
    if type(record.get("schema_version")) is not int or record["schema_version"] != SCHEMA_VERSION:
        raise PhysicalCopyCostError(prefix + " requires complete-path schema_version 2; version 1 was removed")
    for name, expected in (("prediction_kind", PREDICTION_KIND), ("fit_method", FIT_METHOD),
                           ("residual_convention", RESIDUAL_CONVENTION)):
        if record.get(name) != expected:
            raise PhysicalCopyCostError(prefix + " " + name + " mismatch")
    if _number(record.get("minimum_relative_saving"), prefix + ".minimum_relative_saving") != MINIMUM_RELATIVE_SAVING:
        raise PhysicalCopyCostError(prefix + " requires the version2 saving fraction 0.5")
    costs = record.get("costs")
    if not isinstance(costs, dict) or set(costs) != set(PATHS):
        raise PhysicalCopyCostError(prefix + " requires direct, cold and reused complete-path costs")
    for path in PATHS:
        rates = costs[path]
        if not isinstance(rates, list) or len(rates) != 2:
            raise PhysicalCopyCostError(prefix + ".costs." + path + " requires H2D/D2H rates")
        for rate in rates:
            if not isinstance(rate, dict) or set(rate) != {*COEFFICIENTS, *ERROR_FIELDS, "coverage"}:
                raise PhysicalCopyCostError(prefix + ".costs." + path + " requires complete rate records")
            for field in (*COEFFICIENTS, *ERROR_FIELDS):
                _number(rate[field], prefix + ".costs." + path + "." + field, zero=True)
            if not any(rate[field] > 0 for field in COEFFICIENTS):
                raise PhysicalCopyCostError(prefix + " complete-path prediction must be positive")
            if not rate["median_relative_error"] <= rate["p90_relative_error"] <= rate["maximum_relative_error"]:
                raise PhysicalCopyCostError(prefix + " empirical error percentiles are inconsistent")
            bounds = rate["coverage"]
            if not isinstance(bounds, dict) or set(bounds) != set(RATE_COVERAGE_FIELDS):
                raise PhysicalCopyCostError(prefix + " requires complete empirical geometry support")
            if any(type(value) is not int or not 0 < value < 2**64 for value in bounds.values()):
                raise PhysicalCopyCostError(prefix + " support fields must be positive bounded integers")
            for suffix in ("payload_bytes", "rows", "row_bytes", "pitch_bytes", "operations", "descriptor_work", "kernel_chunks"):
                if bounds["minimum_" + suffix] > bounds["maximum_" + suffix]:
                    raise PhysicalCopyCostError(prefix + " has inverted empirical support")
            if (bounds["maximum_payload_bytes"] > PAYLOADS[-1]
                    or bounds["maximum_row_bytes"] > bounds["maximum_payload_bytes"]
                    or bounds["maximum_span_bytes"] < bounds["maximum_payload_bytes"]
                    or bounds["maximum_operations"] > OPERATION_LIMIT
                    or bounds["maximum_kernel_chunks"] > (OPERATION_LIMIT + 47) // 48
                    or path != "direct" and bounds["minimum_operations"] < 2
                    or not 1 <= bounds["word_bytes_mask"] <= 7
                    or not 0 < bounds["layout_combinations_mask"] <= 254
                    or bounds["layout_combinations_mask"] & 1):
                raise PhysicalCopyCostError(prefix + " has inconsistent empirical support")
            submissions = bounds["maximum_operations"] if path == "direct" else bounds["maximum_kernel_chunks"]
            features = (1, submissions, bounds["maximum_payload_bytes"], bounds["maximum_rows"],
                        bounds["maximum_descriptor_work"])
            try:
                upper = math.fsum(rate[name] * value for name, value in zip(COEFFICIENTS, features, strict=True))
            except (OverflowError, ValueError):
                upper = float("inf")
            if not math.isfinite(upper) or upper <= 0:
                raise PhysicalCopyCostError(prefix + " has a nonfinite or nonpositive supported prediction")
    return record
