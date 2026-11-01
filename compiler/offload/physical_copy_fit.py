"""Fit complete measured paths without subtracting independent cost maxima."""
from __future__ import annotations

import itertools
import math
import statistics

from .physical_copy_costs import (
    COEFFICIENTS,
    FIT_METHOD,
    MINIMUM_RELATIVE_SAVING,
    OPERATION_LIMIT,
    PATHS,
    PAYLOADS,
    PREDICTION_KIND,
    RESIDUAL_CONVENTION,
    PhysicalCopyCostError,
    validate_physical_copy_costs,
)


class PhysicalCopyFitUnavailable(PhysicalCopyCostError):
    """Valid observations do not cover a complete supported model."""


def _integer(value, label, *, zero=False):
    if type(value) is not int or not (0 if zero else 1) <= value < 2**64:
        raise PhysicalCopyCostError(label + " must be a bounded integer")
    return value


def _samples(record):
    values = record.get("seconds")
    if not isinstance(values, list) or len(values) != 7:
        raise PhysicalCopyCostError("complete-copy measurement requires seven observations")
    for value in values:
        try:
            finite = math.isfinite(value)
        except (TypeError, OverflowError):
            finite = False
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not finite or value <= 0:
            raise PhysicalCopyCostError("complete-copy observations must be positive finite numbers")
    return values


def _geometry(record):
    operations = record.get("operations")
    if not isinstance(operations, list) or not 1 <= len(operations) <= OPERATION_LIMIT:
        raise PhysicalCopyCostError("complete-copy measurement requires bounded original operations")
    physical = _integer(record.get("physical_bytes"), "physical_bytes")
    payload = rows = work = 0
    word, layout = 8, 0
    starts, ends, widths, pitches = [], [], [], []
    previous = []
    for operation in operations:
        if not isinstance(operation, list) or len(operation) != 6:
            raise PhysicalCopyCostError("physical copy operation requires six integer coordinates")
        offset, width, height, depth, pitch, physical_height = [
            _integer(v, "physical copy coordinate", zero=i == 0) for i, v in enumerate(operation)]
        end = offset + (depth - 1) * pitch * physical_height + (height - 1) * pitch + width
        if (width > pitch or depth > 1 and height > physical_height or end > physical
                or pitch * physical_height >= 2**64):
            raise PhysicalCopyCostError("physical copy operation exceeds the original allocation")
        if payload + width * height * depth > PAYLOADS[-1]:
            raise PhysicalCopyCostError("complete-copy payload exceeds the finite batch budget")
        # Independent interval check and exact shared-planner work accounting.
        # This validates measurement geometry; it grants no execution authority.
        for old, old_end in previous:
            work += 1
            if old_end <= offset or end <= old[0]:
                continue
            a = b = 0
            while a < old[2] * old[3] and b < height * depth:
                work += 1
                if work > 1_000_000:
                    raise PhysicalCopyFitUnavailable("offline copy geometry validation budget exhausted")
                x = old[0] + (a // old[2]) * old[4] * old[5] + (a % old[2]) * old[4]
                y = offset + (b // height) * pitch * physical_height + (b % height) * pitch
                if x < y + width and y < x + old[1]:
                    raise PhysicalCopyCostError("overlapping complete-copy destinations")
                if x + old[1] <= y + width:
                    a += 1
                else:
                    b += 1
        work += 1
        previous.append((operation, end))
        while word > 1 and any(v % word for v in (offset, width, pitch, payload)):
            word //= 2
        if word == 2:
            word = 1
        payload += width * height * depth
        rows += height * depth
        starts.append(offset)
        ends.append(end)
        widths.append(width)
        pitches.append(pitch)
        layout |= 4 if depth > 1 else 2 if height > 1 and pitch != width else 1
    chunks = (len(operations) + 47) // 48
    expected = {"payload_bytes": payload, "rows": rows, "word_bytes": word,
                "layout_mask": layout, "kernel_chunks": chunks, "descriptor_work": work}
    if any(type(record.get(name)) is not int or record[name] != value for name, value in expected.items()):
        raise PhysicalCopyCostError("complete-copy geometry metadata differs from its original operations")
    if payload > PAYLOADS[-1]:
        raise PhysicalCopyCostError("complete-copy payload exceeds the finite batch budget")
    return {"bytes": payload, "rows": rows, "word": word, "layout": layout, "chunks": chunks,
            "operations": len(operations), "work": _integer(record.get("descriptor_work"), "descriptor_work"),
            "minimum_row_bytes": min(widths), "maximum_row_bytes": max(widths),
            "minimum_pitch_bytes": min(pitches), "maximum_pitch_bytes": max(pitches),
            "span": max(ends) - min(starts)}


def _support(geometries):
    result = {}
    for field, value in (("payload_bytes", "bytes"), ("rows", "rows"), ("operations", "operations"),
                         ("descriptor_work", "work"), ("kernel_chunks", "chunks")):
        result["minimum_" + field] = min(g[value] for g in geometries)
        result["maximum_" + field] = max(g[value] for g in geometries)
    result.update(minimum_row_bytes=min(g["minimum_row_bytes"] for g in geometries),
                  maximum_row_bytes=max(g["maximum_row_bytes"] for g in geometries),
                  minimum_pitch_bytes=min(g["minimum_pitch_bytes"] for g in geometries),
                  maximum_pitch_bytes=max(g["maximum_pitch_bytes"] for g in geometries),
                  maximum_span_bytes=max(g["span"] for g in geometries))
    word_mask = layout_mask = 0
    for g in geometries:
        word_mask |= {1: 1, 4: 2, 8: 4}[g["word"]]
        layout_mask |= 1 << g["layout"]
    result.update(word_bytes_mask=word_mask, layout_combinations_mask=layout_mask)
    return result


def _solve(matrix, rhs):
    size = len(rhs)
    augmented = [list(row) + [value] for row, value in zip(matrix, rhs, strict=True)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) <= 1e-12:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [v / divisor for v in augmented[column]]
        for row in range(size):
            if row != column:
                factor = augmented[row][column]
                augmented[row] = [x - factor * y for x, y in zip(augmented[row], augmented[column], strict=True)]
    return [row[-1] for row in augmented]


def _fit(points):
    """Five-feature active-subset NNLS, relative to each geometry median."""
    if len(points) < 5 or len({tuple(features) for features, _ in points}) < 5:
        raise PhysicalCopyFitUnavailable("complete-copy fit needs five distinct observed geometries")
    if any(not math.isfinite(duration) or duration <= 0 for _, duration in points):
        raise PhysicalCopyCostError("complete-copy fit received nonpositive observations")
    scales = [max(features[k] for features, _ in points) for k in range(5)]
    if any(not value for value in scales):
        raise PhysicalCopyFitUnavailable("complete-copy fit has an unobserved feature")
    time_scale = min(duration for _, duration in points)
    matrix = [[features[k] / scales[k] * (time_scale / duration) for k in range(5)]
              for features, duration in points]
    candidates = []
    for count in range(1, 6):
        for active in itertools.combinations(range(5), count):
            gram = [[math.fsum(row[i] * row[j] for row in matrix) for j in active] for i in active]
            rhs = [math.fsum(row[i] for row in matrix) for i in active]
            solution = _solve(gram, rhs)
            if solution is None or any(not math.isfinite(value) or value < 0 for value in solution):
                continue
            theta = [0.0] * 5
            for k, value in zip(active, solution, strict=True):
                theta[k] = value
            error = math.fsum((math.fsum(x * q for x, q in zip(row, theta, strict=True)) - 1) ** 2 for row in matrix)
            coefficients = [theta[k] * time_scale / scales[k] for k in range(5)]
            if all(math.isfinite(value) and value >= 0 for value in coefficients) and any(coefficients):
                candidates.append((error, count, coefficients))
    if not candidates:
        raise PhysicalCopyFitUnavailable("complete-copy observations do not establish finite nonnegative costs")
    return dict(zip(COEFFICIENTS, min(candidates, key=lambda candidate: (candidate[0], candidate[1]))[2], strict=True))


def _prediction(rate, features):
    return math.fsum(rate[name] * value for name, value in zip(COEFFICIENTS, features, strict=True))


def _percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def physical_copies_from_measurements(records, profile):
    groups = {(path, direction): [] for path in PATHS for direction in ("h2d", "d2h")}
    seen = set()
    for record in records:
        if (not isinstance(record, dict) or record.get("kind") != "scoped_copy"
                or type(record.get("version")) is not int or record["version"] != 2):
            raise PhysicalCopyCostError("complete-copy observations require scoped_copy version 2")
        samples = _samples(record)
        geometry = _geometry(record)
        name, state, direction = record.get("name"), record.get("pool_state"), record.get("direction")
        if direction not in {"h2d", "d2h"} or record.get("pending") is not False:
            raise PhysicalCopyCostError("complete-copy observations require exact direction and isolated pending=false")
        if name == "direct" and state == "none":
            path = "direct"
            if type(record.get("capacity_bytes")) is not int or record["capacity_bytes"] != 0:
                raise PhysicalCopyCostError("complete direct observation must not own a compact slot")
        elif name == "compact" and state in {"cold", "reuse"}:
            path = "compact_" + state
            capacity = record.get("capacity_bytes")
            if (geometry["operations"] < 2 or type(capacity) is not int or capacity not in PAYLOADS
                    or capacity < geometry["bytes"]):
                raise PhysicalCopyCostError("complete compact observation requires an admitted slot and multiple operations")
        else:
            raise PhysicalCopyCostError("unknown complete-copy backend/pool state")
        key = (path, direction, tuple(tuple(op) for op in record["operations"]),
               record["physical_bytes"], record["capacity_bytes"])
        if key in seen:
            raise PhysicalCopyCostError("duplicate complete-copy geometry/backend observation")
        seen.add(key)
        submissions = geometry["operations"] if path == "direct" else geometry["chunks"]
        features = (1, submissions, geometry["bytes"], geometry["rows"], geometry["work"])
        groups[path, direction].append((geometry, features, samples))
    costs = {}
    for path in PATHS:
        rates = []
        for direction in ("h2d", "d2h"):
            observations = groups[path, direction]
            rate = _fit([(features, statistics.median(samples)) for _g, features, samples in observations])
            errors = []
            for _g, features, samples in observations:
                predicted = _prediction(rate, features)
                if not math.isfinite(predicted) or predicted <= 0:
                    raise PhysicalCopyFitUnavailable("complete-copy supported prediction is nonpositive or nonfinite")
                errors.extend(abs(predicted - observed) / observed for observed in samples)
            rate.update(median_relative_error=statistics.median(errors), p90_relative_error=_percentile(errors, 0.9),
                        maximum_relative_error=max(errors), coverage=_support([g for g, _x, _y in observations]))
            rates.append(rate)
        costs[path] = rates
    result = {"schema_version": 2, "prediction_kind": PREDICTION_KIND, "fit_method": FIT_METHOD,
              "residual_convention": RESIDUAL_CONVENTION, "minimum_relative_saving": MINIMUM_RELATIVE_SAVING,
              "costs": costs, "measurements": records}
    return validate_physical_copy_costs({**profile, "scoped": {**profile["scoped"], "physical_copies": result}})
