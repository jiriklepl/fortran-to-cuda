"""Launch-inclusive GPU costs from production workers and independent holds.

The parent evidence stays immutable and historically readable. A current query
also authenticates its renderer and this derivation. Resident event intervals
include queue starvation: enqueue is recorded, never added to that interval.
"""

from __future__ import annotations

import math
import statistics
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

from compiler.offload import numerical_gpu_execution as gpu
from compiler.offload.numerical_execution import (
    ACCESS_CLASS,
    CLASSES,
    MAX_RELATIVE_ERROR,
    PRIMITIVES,
    SIZES,
    _coefficients,
    _finish,
    _hash,
    _number,
    _predict,
    _result,
    _sha,
    memory_seconds,
)

SECTION = "numerical_execution_gpu_costs"
SCHEMA_VERSION = 1
PROTOCOL_ID = "production-cuda-nonzero-launch-inclusive-v1"
PREFLIGHT_KIND = "numerical_gpu_launch_preflight_v1"
BATCH_KIND = "numerical_gpu_launch_batch_v1"
CONTROL_SIZES = (0, 1, 8, 17)
MEASURED_SIZES = (1, 8, 17)
RAW_SAMPLES = gpu.SAMPLES * len(MEASURED_SIZES)
MAX_RECORDS = 1 + len(CONTROL_SIZES) + RAW_SAMPLES
FIXED_COVERAGE = "one resident production launch interval including enqueue starvation; no additive enqueue"
COST_SOURCE_NAMES = ("numerical_gpu_execution_costs.py", "numerical_gpu_launch_calibration.py",
                     "numerical_gpu_launch_driver.cu", "numerical_gpu_execution.py", "numerical_execution.py")


def cost_sources():
    """Identify the derivation and the exact reused fitting implementation."""
    folder = Path(__file__).parent
    return {name: sha256((folder / name).read_bytes()).hexdigest() for name in COST_SOURCE_NAMES}


def launch_build_commands(paths, architecture):
    """Reuse one authenticated worker object; compile only the control driver."""
    flags = gpu.gpu_build_commands(paths, architecture)[0]
    flags = flags[:flags.index("-c")]
    target = Path(paths["artifacts"])
    return [[*flags, "-c", str(target / "launch_driver.cu"), "-o", str(target / "launch_driver.o")],
            [*flags, str(target / "workers.o"), str(target / "launch_driver.o"),
             "-o", str(target / "numerical-gpu-launch")]]


def _identity(profile, identity):
    parent = profile[gpu.SECTION]
    if not isinstance(identity, dict):
        raise gpu.NumericalGpuExecutionError("GPU launch identity missing")
    # These fields are emitted by the unchanged original driver, on the actual
    # selected device. Extra compilation provenance is attached by the producer.
    keys = ("kind", "protocol_id", "backend_id", "generator_id", "source_generator_id", "registry_id",
            "precision_bits", "device_ordinal", "gpu_name", "gpu_uuid", "compute_capability",
            "cuda_runtime_version", "driver_version", "sampling_cpu_affinity")
    if any(identity.get(k) != parent["identity"][k]
           or type(identity.get(k)) is not type(parent["identity"][k]) for k in keys):
        raise gpu.NumericalGpuExecutionError("GPU launch actual identity differs from parent")
    if (identity.get("launch_protocol_id") != PROTOCOL_ID
            or identity.get("parent_evidence_sha256") != _hash(parent)):
        raise gpu.NumericalGpuExecutionError("GPU launch parent/protocol identity differs")
    gpu._digest_map(identity.get("cost_sources"), "cost implementation")
    if set(identity["cost_sources"]) != set(COST_SOURCE_NAMES):
        raise gpu.NumericalGpuExecutionError("GPU complete cost implementation inventory required")
    sources, objects = identity.get("emitted_sources"), identity.get("objects")
    gpu._digest_map(sources, "launch sources")
    gpu._digest_map(objects, "launch objects")
    if (set(sources) != {"launch_driver.cu", "parent_driver.cu", "gpu_recipes.hpp"}
            or sources["launch_driver.cu"] != identity["cost_sources"]["numerical_gpu_launch_driver.cu"]
            or sources["parent_driver.cu"] != parent["identity"]["emitted_sources"]["driver.cu"]
            or sources["gpu_recipes.hpp"] != parent["identity"]["emitted_sources"]["gpu_recipes.hpp"]
            or set(objects) != {"workers.o", "launch_driver.o"}
            or objects["workers.o"] != parent["identity"]["objects"]["workers.o"]
            or not _sha(identity.get("binary_sha256"))):
        raise gpu.NumericalGpuExecutionError("GPU launch must retain the exact parent worker/header/driver")
    if (identity.get("build_commands") != launch_build_commands(identity.get("build_paths"),
                                                              identity.get("architecture"))
            or any(identity.get(k) != parent["identity"][k] for k in ("nvcc_version", "host_cxx_version"))
            or identity["architecture"] != parent["identity"]["architecture"]
            or identity["build_paths"]["runtime"] != parent["identity"]["build_paths"]["runtime"]):
        raise gpu.NumericalGpuExecutionError("GPU launch strict build/tool identity differs")


def _reference(value, *, path=None, digest=None):
    if (not isinstance(value, dict) or set(value) != {"path", "sha256", "bytes"}
            or not isinstance(value["path"], str) or not 0 < len(value["path"]) <= 8192
            or not Path(value["path"]).is_absolute() or any(ord(c) < 32 for c in value["path"])
            or not _sha(value["sha256"]) or type(value["bytes"]) is not int
            or not 0 < value["bytes"] <= 256 * 2**20
            or path is not None and value["path"] != str(path)
            or digest is not None and value["sha256"] != digest):
        raise gpu.NumericalGpuExecutionError("GPU launch artifact reference differs")


def _holdout(result, row, predicted):
    # GPU events and CPU wall samples have different schemas. Only the fitting
    # algebra is shared; never manufacture CPU elapsed/wall observations.
    measured = statistics.median(row["seconds"])
    predicted = _number(predicted, "GPU predicted time")
    error = abs(predicted - measured) / measured
    result["holdouts"].append({"recipe": row["recipe"], "items": row["items"],
        "measured_seconds": measured, "predicted_seconds": predicted, "relative_error": error})
    if row["agreement_passed"] is not True:
        result["reason_codes"].append("numerical_agreement_failed")
    if error > MAX_RELATIVE_ERROR:
        result["reason_codes"].append("holdout_error")


def _family(recipe, costs, startup):
    result = _result()
    if startup["status"] != "accepted":
        result["reason_codes"] = ["startup_rejected"]
        return result
    rows = [costs[recipe["name"], n] for n in recipe["sizes"]]
    if any(row["agreement_passed"] is not True for row in rows):
        result["reason_codes"] = ["numerical_agreement_failed"]
        return result
    fixed = startup["fixed_seconds"]
    training = [row for row in rows if row["items"] in recipe["fit_sizes"]]
    variable = [statistics.median(row["seconds"]) - fixed for row in training]
    if any(not math.isfinite(v) or v <= 0 for v in variable):
        result["reason_codes"] = ["measured_startup_exceeds_training_time"]
        return result
    if recipe["role"] == "memory":
        knots = [{"working_set_bytes": row["working_set_bytes"], "seconds_per_traffic_byte": v / row["traffic_bytes"]}
                 for row, v in zip(training, variable, strict=True)]
        result["model"] = {"kind": "piecewise_bandwidth_v1", "knots": knots,
            "working_set_range": [knots[0]["working_set_bytes"], knots[-1]["working_set_bytes"]],
            "fixed_cost_coverage": "independently measured nonzero launch interval removed"}

        def predict(row):
            return fixed + memory_seconds(result["model"], row["traffic_bytes"], row["working_set_bytes"])
    else:
        slope = math.fsum(row["items"] * v for row, v in zip(training, variable, strict=True)) /\
            sum(row["items"] ** 2 for row in training)
        result["model"] = {"seconds_per_item": _number(slope, "GPU variable fitted slope"),
            "fixed_seconds": fixed, "fit": "nonnegative slope with independently measured nonzero fixed term"}

        def predict(row):
            return fixed + row["items"] * slope
    for row in rows:
        if row["items"] not in recipe["fit_sizes"]:
            _holdout(result, row, predict(row))
    return _finish(result)


def _class(recipes, costs, families, fits, startup, precision):
    result = _result()
    if not recipes:
        result["reason_codes"] = ["independent_holdouts_missing"]
        return result
    primitives = tuple(p for p in PRIMITIVES if any(p in r["intrinsics"] for r in recipes))
    try:
        coefficients = _coefficients(families, fits, "gpu", primitives)
    except ValueError:
        result["reason_codes"] = ["required_family_rejected"]
        return result
    for recipe in recipes:
        for size in recipe["sizes"]:
            _holdout(result, costs[recipe["name"], size], _predict(recipe, size, coefficients,
                     startup["fixed_seconds"], precision))
    return _finish(result)


def _coordinate(row, recipe, size, precision, kind):
    required = {"kind": kind, "recipe": "memory", "recipe_id": recipe["recipe_id"], "backend": "gpu",
                "items": size, "role": "holdout", "traffic_bytes": size * 3 * (precision // 8),
                "working_set_bytes": size * 3 * (precision // 8), "agreement_passed": True}
    if (not isinstance(row, dict) or any(row.get(k) != v or type(row.get(k)) is not type(v)
                                       for k, v in required.items())):
        raise gpu.NumericalGpuExecutionError("GPU launch coordinate/agreement differs")
    extras = {"launches"} if kind == PREFLIGHT_KIND else {
        "batch", "global_order", "repetitions", "launches", "resident_event_seconds", "enqueue_seconds", "wall_seconds"}
    if set(row) != set(required) | extras:
        raise gpu.NumericalGpuExecutionError("GPU launch exact raw schema differs")


def validate_launch_preflight(profile, identity, rows):
    gpu.validate_gpu_execution_profile(profile)
    _identity(profile, identity)
    recipe = next(r for r in profile[gpu.SECTION]["registry"] if r["name"] == "memory")
    if not isinstance(rows, list) or len(rows) != len(CONTROL_SIZES):
        raise gpu.NumericalGpuExecutionError("GPU launch complete preflight required")
    for row, size in zip(rows, CONTROL_SIZES, strict=True):
        _coordinate(row, recipe, size, profile["precision_bits"], PREFLIGHT_KIND)
        if type(row.get("launches")) is not int or row["launches"] != int(size > 0):
            raise gpu.NumericalGpuExecutionError("GPU nonzero launch/zero skip proof differs")


def _build(profile, records, calibration):
    _hash(records)
    _hash(calibration)
    if not isinstance(records, list) or len(records) != MAX_RECORDS:
        raise gpu.NumericalGpuExecutionError("GPU launch requires four preflights and21 ordered raw batches")
    identity = records[0]
    validate_launch_preflight(profile, identity, records[1:5])
    parent = profile[gpu.SECTION]
    target = Path(identity["build_paths"]["artifacts"])
    argv = ["taskset", "-c", ",".join(map(str, identity["sampling_cpu_affinity"])),
            str(target / "numerical-gpu-launch"), str(identity["device_ordinal"])]
    raw = calibration.get("raw_samples") if isinstance(calibration, dict) else None
    environment = calibration.get("environment") if isinstance(calibration, dict) else None
    if (not isinstance(calibration, dict) or calibration.get("application_profiled") is not False
            or calibration.get("parent_evidence_sha256") != _hash(parent)
            or calibration.get("run_commands") != [[*argv, mode] for mode in ("--identity", "--preflight", "--measure")]
            or not isinstance(raw, dict) or set(raw) != {"path", "sha256", "bytes"}
            or raw["path"] != str(target / "launch-raw-samples.jsonl") or not _sha(raw["sha256"])
            or type(raw["bytes"]) is not int or not 0 < raw["bytes"] <= 2**20
            or not isinstance(environment, dict) or set(environment) != set(gpu.ENVIRONMENT_KEYS)
            or any(v is not None and (not isinstance(v, str) or len(v) > 8192) for v in environment.values())
            or environment["FORT_PHASE_TIMING"] != "0" or environment["CUDA_LAUNCH_BLOCKING"] is not None):
        raise gpu.NumericalGpuExecutionError("GPU launch complete raw/environment provenance required")
    _reference(calibration.get("input_profile"))
    tools, artifacts = calibration.get("tools"), calibration.get("parent_artifacts")
    if not isinstance(tools, dict) or set(tools) != {"nvcc", "host"} or not isinstance(artifacts, dict):
        raise gpu.NumericalGpuExecutionError("GPU launch complete tools/parent artifacts required")
    for name, ref in tools.items():
        _reference(ref, path=identity["build_paths"][name])
    parent_hashes = {**parent["identity"]["emitted_sources"], **parent["identity"]["objects"],
                     "numerical-gpu-execution": parent["identity"]["binary_sha256"]}
    if set(artifacts) != set(parent_hashes):
        raise gpu.NumericalGpuExecutionError("GPU launch complete parent artifact inventory required")
    for name, digest in parent_hashes.items():
        _reference(artifacts[name], path=Path(parent["calibration"]["artifacts"]) / name, digest=digest)
    recipe = next(r for r in parent["registry"] if r["name"] == "memory")
    indexed = {size: [] for size in MEASURED_SIZES}
    for order, row in enumerate(records[5:]):
        batch, offset = divmod(order, len(MEASURED_SIZES))
        size = MEASURED_SIZES[offset]
        _coordinate(row, recipe, size, profile["precision_bits"], BATCH_KIND)
        if (type(row.get("batch")) is not int or row["batch"] != batch
                or type(row.get("global_order")) is not int or row["global_order"] != order
                or type(row.get("repetitions")) is not int or not 0 < row["repetitions"] < 1 << 63
                or type(row.get("launches")) is not int or row["launches"] != row["repetitions"]):
            raise gpu.NumericalGpuExecutionError("GPU launch batch order/repetition proof differs")
        for key in ("resident_event_seconds", "enqueue_seconds", "wall_seconds"):
            _number(row.get(key), "GPU launch " + key)
        if (row["wall_seconds"] < gpu.MIN_BATCH_SECONDS or row["enqueue_seconds"] > row["wall_seconds"]
                or row["resident_event_seconds"] > row["wall_seconds"] + max(.001, .01 * row["wall_seconds"])):
            raise gpu.NumericalGpuExecutionError("GPU launch duration accounting differs")
        indexed[size].append(row)
    startup = _result()
    startup.update(fixed_seconds=statistics.median(r["resident_event_seconds"] / r["launches"] for r in indexed[1]),
                   coverage=FIXED_COVERAGE, fit_items=[1], holdout_items=[8, 17],
                   tiny_work="same production memory worker; two loads, one store and two arithmetic operations/item")
    for size in (8, 17):
        _holdout(startup, {"recipe": "memory_nonzero_launch", "items": size, "agreement_passed": True,
                          "seconds": [r["resident_event_seconds"] / r["launches"] for r in indexed[size]]},
                 startup["fixed_seconds"])
    _finish(startup)
    # Reuse the frozen scalar arithmetic fitting algebra, never CPU timings.
    costs = {}
    for recipe in parent["registry"]:
        for size in recipe["sizes"]:
            rows = [r for r in parent["measurements"][1 + gpu.PREFLIGHT_RECORDS:]
                    if r["recipe"] == recipe["name"] and r["items"] == size]
            costs[recipe["name"], size] = {
                **{k: rows[0][k] for k in ("recipe", "items", "traffic_bytes", "working_set_bytes")},
                "agreement_passed": all(r["agreement_passed"] for r in rows),
                "seconds": [r["resident_event_seconds"] / r["launches"] for r in rows]}
    fitting = {r["family"]: r for r in parent["registry"] if r["role"] in {"coefficient", "memory"}}
    families = {name: {"gpu": _family(recipe, costs, startup)} for name, recipe in fitting.items()}
    classes = {name: _class([r for r in parent["registry"] if r["role"] == "holdout"
                            and r["workload_class"] == name], costs, families, fitting, startup,
                           profile["precision_bits"]) for name in CLASSES}
    domains = {name: _class([r for r in parent["registry"] if r["holdout_kind"] == "domain"
                            and r["family"] == name], costs, families, fitting, startup,
                           profile["precision_bits"]) for name in ("sqrt", "acos", "cos")}
    return {"schema_version": SCHEMA_VERSION, "protocol_id": PROTOCOL_ID,
            "identity": deepcopy(identity), "measurements": deepcopy(records), "calibration": deepcopy(calibration),
            "startup": startup, "families": families, "workload_validation": classes, "domain_validation": domains,
            "holdout_error_ceiling": MAX_RELATIVE_ERROR, "metric": "resident_event_seconds_per_launch",
            "fixed_cost_coverage": FIXED_COVERAGE, "launch_enqueue_included": True,
            "cost_model_available": any(r["status"] == "accepted" for r in classes.values()),
            "unpriced": ["cold driver/setup", "allocation", "transfers", "runtime notification", "team coordination", "wait API"],
            "scope": "GPU execution costs; other AUTO roles and source applicability are independently required"}


def profile_from_gpu_launch_measurements(profile, records, calibration):
    if SECTION in profile:
        raise gpu.NumericalGpuExecutionError("preserve existing GPU costs; use an unmodified parent")
    try:
        result = deepcopy(profile)
        result[SECTION] = _build(profile, records, calibration)
        return result
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise gpu.NumericalGpuExecutionError(str(error)) from error


def validate_gpu_execution_costs(profile):
    """Read historical costs without lowering, discovery or current-identity rebinding."""
    try:
        section = profile.get(SECTION)
        if (not isinstance(section, dict) or type(section.get("schema_version")) is not int
                or section["schema_version"] != SCHEMA_VERSION
                or section.get("launch_enqueue_included") is not True):
            raise gpu.NumericalGpuExecutionError("GPU execution costs missing")
        if section != _build(profile, section.get("measurements"), section.get("calibration")):
            raise gpu.NumericalGpuExecutionError("GPU costs differ from raw reconstruction")
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise gpu.NumericalGpuExecutionError(str(error)) from error


def gpu_execution_costs(profile, counts, *, workload_class, workload_features, access_class):
    """Return accepted current GPU costs, including exactly one launch interval.

    Consumers must omit their separate enqueue charge for this contract. This
    grants no CPU, existing-team, descriptor, definition or primitive-domain
    authority; those source applicability obligations remain independent.
    """
    validate_gpu_execution_costs(profile)
    parent = gpu.require_current_gpu_execution(profile)
    section = profile[SECTION]
    if section["identity"]["cost_sources"] != cost_sources() or access_class != ACCESS_CLASS:
        raise gpu.NumericalGpuExecutionError("GPU cost implementation/access identity differs")
    if not isinstance(counts, (dict, list, tuple)):
        raise gpu.NumericalGpuExecutionError("GPU primitive counts unavailable")
    pairs = list(counts.items()) if isinstance(counts, dict) else list(counts)
    if (any(not isinstance(p, (tuple, list)) or len(p) != 2 for p in pairs)
            or len(pairs) > len(PRIMITIVES)
            or any(not isinstance(name, str) or name not in PRIMITIVES or type(count) is not int
                   or not 0 < count <= 100000 for name, count in pairs)
            or len({p[0] for p in pairs}) != len(pairs)):
        raise gpu.NumericalGpuExecutionError("GPU primitive counts unavailable")
    selected = "ordinary_expression_v2" if workload_class == "scalar_expression_v2" and\
        not any(name != "divide" for name, _ in pairs) else workload_class
    if selected not in CLASSES or section["workload_validation"][selected]["status"] != "accepted":
        raise gpu.NumericalGpuExecutionError("independent GPU workload class rejected")
    features = workload_features.to_dict() if hasattr(workload_features, "to_dict") else workload_features
    if (not isinstance(features, dict) or type(features.get("schema_version")) is not int
            or features["schema_version"] != 2 or features.get("classification_complete") is not True):
        raise gpu.NumericalGpuExecutionError("GPU private-array classification unavailable")
    if workload_class == "fixed_private_array_v2":
        for key in ("private_array_groups", "private_array_elements", "max_private_array_elements", "max_private_array_rank"):
            limit = max(r["private_features"][key] for r in parent["registry"] if r["workload_class"] == workload_class)
            if type(features.get(key)) is not int or not 0 < features[key] <= limit:
                raise gpu.NumericalGpuExecutionError("GPU private workload outside validated applicability")
    elif type(features.get("private_array_elements")) is not int or features["private_array_elements"] != 0:
        raise gpu.NumericalGpuExecutionError("GPU scalar workload contains private arrays")
    for name, _ in pairs:
        if name in section["domain_validation"] and section["domain_validation"][name]["status"] != "accepted":
            raise gpu.NumericalGpuExecutionError("GPU primitive domain holdout rejected")
    fitting = {r["family"]: r for r in parent["registry"] if r["role"] in {"coefficient", "memory"}}
    try:
        components = _coefficients(section["families"], fitting, "gpu", tuple(name for name, _ in pairs))
    except ValueError as error:
        raise gpu.NumericalGpuExecutionError(str(error)) from error
    return {**components, "backend_identity": "gpu", "fixed_seconds": section["startup"]["fixed_seconds"],
            "fixed_cost_coverage": FIXED_COVERAGE, "launch_enqueue_included": True,
            "intrinsic_seconds_per_item": sum(count * components["intrinsic_seconds_per_operation"][name] for name, count in pairs),
            "item_range": [min(SIZES), max(SIZES)], "access_class": ACCESS_CLASS,
            "cpu_affinity": deepcopy(parent["identity"]["sampling_cpu_affinity"]),
            "gpu_generator_id": parent["identity"]["generator_id"], "cost_protocol_id": PROTOCOL_ID}
