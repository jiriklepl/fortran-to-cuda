"""Independent original-Fortran and production CPU execution evidence.

Schema one deliberately prices CPU roles only. Saved observations remain
readable when producer sources change; a query additionally requires the
current source registry. No legacy numerical or GPU coefficients are blended.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from copy import deepcopy

from compiler.numerical_contract import HOST_OPTIONS
from compiler.offload.collective_calibration import normalize_fortran_options
from compiler.offload.numerical_calibration import NumericalCalibrationError, require_numerical_profile_contract

SCHEMA_VERSION = 1
PROTOCOL_ID = "original-fortran-production-cyclic-v1"
BACKEND_ID = "native-fortran-production-cyclic-cxx17-v1"
PROOF_BACKEND = "gnu-GOMP_parallel-wrap-v1"
CPU_BACKENDS = ("native_serial", "native_fork_join", "generated_cpu")
STARTUP_BACKENDS = CPU_BACKENDS[1:]
STARTUP_SIZES = (0, 1, 8)
FIT_SIZES = (65536, 262144, 1048576)
HOLDOUT_SIZES = (131072, 524288)
SIZES = tuple(sorted(FIT_SIZES + HOLDOUT_SIZES))
MEMORY_FIT_SIZES = tuple(65536 * 3**k // 2**k for k in range(8))
MEMORY_HOLDOUT_SIZES = tuple(sorted({131072, 524288, *(round(math.sqrt(a * b))
    for a, b in zip(MEMORY_FIT_SIZES[:-1], MEMORY_FIT_SIZES[1:], strict=True))}))
MEMORY_SIZES = tuple(sorted(MEMORY_FIT_SIZES + MEMORY_HOLDOUT_SIZES))
SAMPLES = 7
MIN_BATCH_SECONDS = 0.2
MAX_RELATIVE_ERROR = 0.25
PRIMITIVES = ("sqrt", "acos", "cos", "divide")
ACCESS_CLASS = "pointwise_three_array_v1"
CLASSES = ("ordinary_expression_v2", "scalar_expression_v2", "fixed_private_array_v2")
HOST_FLAGS = ("-O3", "-std=c++17", "-fopenmp", *HOST_OPTIONS)
FIXED_COVERAGE = "one measured execution startup outside max(compute,variable_memory)"
KINDS = frozenset({"numerical_execution_identity_v1", "numerical_execution_startup_v1",
    "numerical_execution_team_proof_v1", "numerical_execution_cost_v1"})
MAX_RECIPES = 17
MAX_RECORDS = 304
# Frozen schema-one source accounting. Source hashes may describe an archived
# producer, but a different recipe/count/domain contract requires a new schema.
RECIPE_NAMES = ("arithmetic", "memory", "sqrt", "acos", "cos", "scalar_mix", "scalar_mix_skew",
    "private_mix", "private_mix_skew", "divide", "ordinary_mix", "helper_scalar_mix",
    "helper_private_mix", "dependent_arithmetic", "domain_sqrt", "domain_acos", "domain_cos")
RECIPE_COUNTS = {
    "arithmetic": (520, {}), "memory": (2, {}), "sqrt": (82, {"sqrt": 16}),
    "acos": (50, {"acos": 16}), "cos": (34, {"cos": 16}),
    "scalar_mix": (850, {"sqrt": 96, "acos": 32, "cos": 80, "divide": 16}),
    "scalar_mix_skew": (498, {"sqrt": 32, "acos": 64, "cos": 16, "divide": 16}),
    "private_mix": (1031, {"sqrt": 96, "acos": 32, "cos": 80, "divide": 16}),
    "private_mix_skew": (529, {"sqrt": 32, "acos": 64, "cos": 16, "divide": 16}),
    "divide": (328, {"divide": 64}), "ordinary_mix": (578, {"divide": 64}),
    "dependent_arithmetic": (130, {}), "domain_sqrt": (2, {"sqrt": 1}),
    "domain_acos": (2, {"acos": 1}), "domain_cos": (2, {"cos": 1}),
}
DOMAINS = {"sqrt": "finite_nonnegative_normal_or_zero", "acos": "finite_unit_interval",
    "cos": "finite_pi_interval", "divide": "finite_positive_denominator_lattice_v1"}
RECIPE_KEYS = frozenset({"name", "family", "role", "holdout_kind", "arithmetic", "intrinsics",
    "workload_class", "private_features", "domain_ids", "precision_bits", "access_class", "sizes",
    "fit_sizes", "source_sha256", "helper_form", "source_normalization", "recipe_id"})
RECIPE_KEYS = RECIPE_KEYS | {"input_lattices"}


class NumericalExecutionError(NumericalCalibrationError):
    """Incomplete, incompatible or rejected independent execution evidence."""


def _hash(value):
    try:
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                        allow_nan=False).encode()).hexdigest()
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise NumericalExecutionError("execution metadata is not finite canonical JSON") from error


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _number(value, name, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NumericalExecutionError(name + " must be a finite number")
    try:
        result = float(value)
    except OverflowError as error:
        raise NumericalExecutionError(name + " is unrepresentable") from error
    if not math.isfinite(result) or result < 0 or not zero and result == 0:
        raise NumericalExecutionError(name + " must be finite and " + ("nonnegative" if zero else "positive"))
    return result


def _registry(precision):
    # Metadata-only provider: no frontend lowering, compilation or observations.
    from compiler.offload.numerical_execution_workloads import execution_registry
    return tuple(deepcopy(execution_registry(precision)))


def _current_identities(precision):
    from compiler.offload.numerical_execution_workloads import execution_generator_identity, registry_identity
    return execution_generator_identity(), registry_identity(precision)


def _expected_lattices(name, precision):
    epsilon = 2.0 ** (-24 if precision == 32 else -53)
    lattices = {
        "sqrt": ((0.0, 2.0**-60, 2.0**-30, .25, 1.0, 2.0, 2.0**30, 2.0**60),
                 (0.0, 2.0**-60, 2.0**-45, 2.0**-15, .5, 3.0, 9.0, 2.0**15, 2.0**45, 2.0**60)),
        "acos": ((-1.0, -1.0 + epsilon, -.75, -.25, 0.0, .25, .75, 1.0 - epsilon, 1.0),
                 (-1.0, -1.0 + 4 * epsilon, -.875, -.5, -.125, 0.0, .125, .5, .875, 1.0 - 4 * epsilon, 1.0)),
        "cos": ((-math.pi, -math.pi / 2, -1.0, 0.0, 1.0, math.pi / 2, math.pi),
                (-math.pi, -math.pi / 2 - epsilon, -math.pi / 2 + epsilon, -2.0, -.5, 0.0, .5,
                 2.0, math.pi / 2 - epsilon, math.pi / 2 + epsilon, math.pi)),
    }
    values = lattices.get(name.removeprefix("domain_") if name.startswith("domain_") else None,
        ((-.875, -.5, -.125, 0.0, .125, .5, .875), (-1.0, -.625, -.25, -.0625, 0.0, .0625, .25, .625, 1.0)))
    return {split: {"a": list(values[index]), "b": list((2.0, 2.5, 3.0, 3.5, 4.0)
        if index == 0 else (2.125, 2.625, 3.125, 3.625, 3.875))}
        for index, split in enumerate(("training", "holdout"))}


def _validate_registry(registry, precision):
    if not isinstance(registry, (list, tuple)) or len(registry) != MAX_RECIPES:
        raise NumericalExecutionError("execution registry requires the fixed seventeen recipes")
    names, fits, memory, holds = set(), set(), 0, {name: 0 for name in CLASSES}
    for expected_name, recipe in zip(RECIPE_NAMES, registry, strict=True):
        if (not isinstance(recipe, dict) or not isinstance(recipe.get("name"), str)
                or not recipe["name"] or len(recipe["name"]) > 128 or recipe["name"] in names
                or set(recipe) != RECIPE_KEYS or recipe["name"] != expected_name
                or not _sha(recipe.get("recipe_id")) or not _sha(recipe.get("source_sha256"))
                or recipe.get("access_class") != ACCESS_CLASS
                or type(recipe.get("arithmetic")) is not int or not 0 <= recipe["arithmetic"] <= 100000
                or not isinstance(recipe.get("intrinsics"), dict)):
            raise NumericalExecutionError("invalid bounded execution recipe")
        if _hash({key: value for key, value in recipe.items() if key != "recipe_id"}) != recipe["recipe_id"]:
            raise NumericalExecutionError("execution recipe identity differs from frozen metadata")
        names.add(recipe["name"])
        if any(name not in PRIMITIVES or type(count) is not int or not 0 < count <= 100000
               for name, count in recipe["intrinsics"].items()):
            raise NumericalExecutionError("invalid execution primitive counts")
        if recipe.get("role") == "memory":
            memory += 1
            expected_sizes, expected_fits = MEMORY_SIZES, MEMORY_FIT_SIZES
            if recipe.get("family") != "memory" or recipe["intrinsics"]:
                raise NumericalExecutionError("invalid execution memory recipe")
        else:
            expected_sizes, expected_fits = SIZES, FIT_SIZES if recipe.get("role") == "coefficient" else ()
            if recipe.get("role") == "coefficient":
                family = recipe.get("family")
                if family not in ("arithmetic", *PRIMITIVES) or family in fits:
                    raise NumericalExecutionError("invalid execution coefficient family")
                fits.add(family)
                if (family == "arithmetic" and (recipe["intrinsics"] or recipe["arithmetic"] == 0)
                        or family in PRIMITIVES and set(recipe["intrinsics"]) != {family}):
                    raise NumericalExecutionError("coefficient recipe counts mismatch")
            elif recipe.get("role") == "holdout" and recipe.get("workload_class") in CLASSES:
                holds[recipe["workload_class"]] += 1
            else:
                raise NumericalExecutionError("invalid independent execution holdout")
        if recipe.get("sizes") != list(expected_sizes) or recipe.get("fit_sizes") != list(expected_fits):
            raise NumericalExecutionError("execution recipe size membership changed")
        if not isinstance(recipe.get("domain_ids"), dict) or not isinstance(recipe.get("private_features"), dict):
            raise NumericalExecutionError("execution source applicability missing")
        name = recipe["name"]
        base = name.removeprefix("helper_")
        private = base.startswith("private_")
        width = 2 if base == "private_mix_skew" else 4
        expected_private = {"private_array_groups": 4 if private else 0,
            "private_array_elements": 4 * width * width if private else 0,
            "max_private_array_elements": width * width if private else 0,
            "max_private_array_rank": 2 if private else 0}
        expected_role = "memory" if base == "memory" else "coefficient" if name in ("arithmetic", *PRIMITIVES) else "holdout"
        expected_kind = "separate_helper" if name.startswith("helper_") else "domain" if name.startswith("domain_") else \
            "dependency" if name == "dependent_arithmetic" else "expression" if expected_role == "holdout" else None
        expected_class = "fixed_private_array_v2" if private else "ordinary_expression_v2" if base in \
            {"ordinary_mix", "dependent_arithmetic"} else "scalar_expression_v2"
        if (recipe["family"] != base.removeprefix("domain_") or recipe["role"] != expected_role
                or recipe["holdout_kind"] != expected_kind or recipe["workload_class"] != expected_class
                or (recipe["arithmetic"], recipe["intrinsics"]) != RECIPE_COUNTS[base]
                or recipe["domain_ids"] != {key: DOMAINS[key] for key in recipe["intrinsics"]}
                or recipe["input_lattices"] != _expected_lattices(name, precision)
                or recipe["private_features"] != expected_private
                or any(type(value) is not int for value in recipe["private_features"].values())
                or type(recipe["precision_bits"]) is not int or recipe["precision_bits"] != precision
                or recipe["helper_form"] != ("separate" if name.startswith("helper_") else "same_translation_unit")
                or recipe["source_normalization"] != "predeclared bounded straight-line work item; no application source rewriting"):
            raise NumericalExecutionError("execution schema-one recipe/count/domain contract changed")
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               for split in recipe["input_lattices"].values() for values in split.values() for value in values):
            raise NumericalExecutionError("execution input lattice types changed")
    if fits != {"arithmetic", *PRIMITIVES} or memory != 1 or any(not value for value in holds.values()):
        raise NumericalExecutionError("execution registry family/class coverage incomplete")


def _samples(row):
    samples = row.get("samples")
    if not isinstance(samples, list) or len(samples) != SAMPLES:
        raise NumericalExecutionError("execution observations require seven batches")
    values = []
    for batch, sample in enumerate(samples):
        if (not isinstance(sample, dict) or type(sample.get("batch")) is not int or sample["batch"] != batch
                or type(sample.get("repetitions")) is not int or not 0 < sample["repetitions"] < 1 << 63
                or type(sample.get("global_order")) is not int or sample["global_order"] < 0):
            raise NumericalExecutionError("execution batch order/repetition metadata invalid")
        elapsed = _number(sample.get("elapsed_seconds"), "execution elapsed duration")
        wall = _number(sample.get("wall_seconds"), "execution wall duration")
        if wall < MIN_BATCH_SECONDS or elapsed != wall:
            raise NumericalExecutionError("CPU execution batches require matching >=200ms wall durations")
        values.append(_number(elapsed / sample["repetitions"], "execution per-call duration"))
    return values


def parse_execution_measurements(text):
    records = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as error:
            raise NumericalExecutionError("execution benchmark emitted non-JSON output") from error
        if not isinstance(row, dict) or not isinstance(row.get("kind"), str) or row["kind"] not in KINDS:
            raise NumericalExecutionError("unknown execution observation kind")
        if row["kind"] in {"numerical_execution_startup_v1", "numerical_execution_cost_v1"}:
            _samples(row)
        records.append(row)
        if len(records) > MAX_RECORDS:
            raise NumericalExecutionError("execution observation budget exhausted")
    if not records:
        raise NumericalExecutionError("execution benchmark emitted no observations")
    return records


def _identity(profile, identity, generator, registry_id):
    _hash(identity)
    if (identity.get("protocol_id") != PROTOCOL_ID or identity.get("backend_id") != BACKEND_ID
            or identity.get("generator_id") != generator or identity.get("registry_id") != registry_id):
        raise NumericalExecutionError("execution source/protocol identity mismatch")
    for name in ("cpu_threads", "precision_bits"):
        if type(identity.get(name)) is not int or identity[name] != profile.get(name):
            raise NumericalExecutionError("execution " + name + " mismatch")
    threads, affinity = profile["cpu_threads"], identity.get("cpu_affinity")
    if (not isinstance(affinity, list) or len(affinity) != threads
            or any(type(cpu) is not int or not 0 <= cpu < 1 << 20 for cpu in affinity)
            or affinity != sorted(set(affinity)) or identity.get("omp_dynamic") is not False
            or identity.get("omp_proc_bind") != "false"
            or type(identity.get("actual_team_threads")) is not int or identity["actual_team_threads"] != threads
            or type(identity.get("thread_limit")) is not int or not threads <= identity["thread_limit"] < 1 << 31):
        raise NumericalExecutionError("execution actual team/placement identity mismatch")
    for name in ("omp_wait_policy", "gomp_spincount"):
        value = identity.get(name)
        if name not in identity or value is not None and (not isinstance(value, str) or len(value) > 128
                or any(ord(char) < 32 or ord(char) > 126 for char in value)):
            raise NumericalExecutionError("execution bounded wait/spin identity missing")
    native, host = identity.get("fortran"), identity.get("host")
    hardware = profile.get("hardware")
    if (not isinstance(hardware, dict) or not isinstance(identity.get("cpu_name"), str)
            or not 0 < len(identity["cpu_name"]) <= 8192 or identity["cpu_name"] != hardware.get("cpu_name")):
        raise NumericalExecutionError("execution actual CPU identity mismatch")
    if (not isinstance(native, dict) or not isinstance(native.get("compiler_version"), str)
            or not 0 < len(native["compiler_version"]) <= 8192 or not isinstance(native.get("compiler_options"), str)
            or not 0 < len(native["compiler_options"]) <= 65536):
        raise NumericalExecutionError("execution actual Fortran semantic identity missing")
    try:
        semantic_options = normalize_fortran_options(native["compiler_options"])
    except ValueError as error:
        raise NumericalExecutionError("execution actual Fortran semantic identity invalid") from error
    if native.get("semantic_options") != semantic_options:
        raise NumericalExecutionError("execution actual Fortran semantic identity missing")
    toolchain = profile.get("toolchain")
    if (not isinstance(host, dict) or not isinstance(toolchain, dict)
            or not isinstance(host.get("compiler_version"), str) or not 0 < len(host["compiler_version"]) <= 8192
            or host["compiler_version"] != toolchain.get("host_cxx_version")
            or host.get("semantic_options") != list(HOST_FLAGS)):
        raise NumericalExecutionError("execution generated CPU backend identity mismatch")
    objects = identity.get("timed_objects")
    if (identity.get("proof_backend") != PROOF_BACKEND or not isinstance(objects, dict)
            or not 1 <= len(objects) <= 128 or any(not isinstance(key, str) or not key or len(key) > 256
                                                  or not _sha(value) for key, value in objects.items())):
        raise NumericalExecutionError("execution same-object team proof identity missing")


def _expected_orders(registry):
    orders, ordinal = {}, 0
    for batch in range(SAMPLES):
        for size in STARTUP_SIZES:
            for index in range(len(STARTUP_BACKENDS)):
                backend = STARTUP_BACKENDS[(batch + index) % len(STARTUP_BACKENDS)]
                orders[("startup", backend, size, batch)] = ordinal
                ordinal += 1
        for recipe in registry:
            for size in recipe["sizes"]:
                for index in range(len(CPU_BACKENDS)):
                    backend = CPU_BACKENDS[(batch + index) % len(CPU_BACKENDS)]
                    orders[(recipe["name"], backend, size, batch)] = ordinal
                    ordinal += 1
    return orders


def _control_identity(row, registry, precision):
    control = next(recipe for recipe in registry if recipe["name"] == "memory")
    size = row.get("items")
    if type(size) is not int or size not in STARTUP_SIZES:
        raise NumericalExecutionError("execution startup/team proof coordinate invalid")
    traffic = 3 * (precision // 8) * size
    if (row.get("recipe") != "memory" or row.get("recipe_id") != control["recipe_id"]
            or row.get("family") != "memory" or row.get("role") != "holdout"
            or any(type(row.get(field)) is not int or row[field] != traffic
                   for field in ("traffic_bytes", "working_set_bytes"))):
        raise NumericalExecutionError("execution startup/team proof control identity mismatch")


def _proof_passed(proof, threads):
    return (proof.get("agreement_passed") is True and type(proof.get("parallel_entries")) is int
            and proof["parallel_entries"] == 1 and type(proof.get("wrong_team_visits")) is int
            and proof["wrong_team_visits"] == 0 and proof.get("thread_visits") == [1] * threads
            and all(type(value) is int for value in proof["thread_visits"]))


def _index(records, registry, identity):
    if not isinstance(records, list) or len(records) != MAX_RECORDS:
        raise NumericalExecutionError("execution observations incomplete or over budget")
    _hash(records)
    recipes, costs, startup, proofs = {r["name"]: r for r in registry}, {}, {}, {}
    expected_orders = _expected_orders(registry)
    for row in records:
        if not isinstance(row, dict) or not isinstance(row.get("kind"), str) or row["kind"] not in KINDS:
            raise NumericalExecutionError("unknown execution observation kind")
        kind = row["kind"]
        if kind == "numerical_execution_identity_v1":
            continue
        backend, size = row.get("backend"), row.get("items")
        if backend not in CPU_BACKENDS or type(size) is not int:
            raise NumericalExecutionError("execution backend/item coordinate invalid")
        if type(row.get("agreement_passed")) is not bool:
            raise NumericalExecutionError("execution numerical agreement flag missing")
        if kind == "numerical_execution_cost_v1":
            name = row.get("recipe")
            recipe = recipes.get(name) if isinstance(name, str) else None
            if (recipe is None or size not in recipe["sizes"] or row.get("recipe_id") != recipe["recipe_id"]
                    or row.get("family") != recipe["family"]
                    or row.get("role") != ("fit" if size in recipe["fit_sizes"] else "holdout")):
                raise NumericalExecutionError("execution recipe/fit/holdout identity mismatch")
            key = (name, backend, size)
            if key in costs:
                raise NumericalExecutionError("duplicate execution coordinate")
            costs[key] = row
            # All original fixtures explicitly load both distinct inputs once.
            width = 3 * (identity["precision_bits"] // 8) * size
            if any(type(row.get(field)) is not int or row[field] != width
                   for field in ("traffic_bytes", "working_set_bytes")):
                raise NumericalExecutionError("execution physical three-array footprint mismatch")
            order_key = name
        elif kind == "numerical_execution_startup_v1":
            if backend not in STARTUP_BACKENDS or size not in STARTUP_SIZES or (backend, size) in startup:
                raise NumericalExecutionError("duplicate/invalid execution startup coordinate")
            startup[backend, size] = row
            order_key = "startup"
        else:
            if (backend not in STARTUP_BACKENDS or size not in STARTUP_SIZES or (backend, size) in proofs
                    or row.get("timed_objects") != identity["timed_objects"]
                    or row.get("proof_backend") != identity["proof_backend"]):
                raise NumericalExecutionError("execution team proof coordinate/object mismatch")
            proofs[backend, size] = row
        if kind in {"numerical_execution_startup_v1", "numerical_execution_team_proof_v1"}:
            _control_identity(row, registry, identity["precision_bits"])
            if kind == "numerical_execution_team_proof_v1":
                continue
        _samples(row)
        for batch, sample in enumerate(row["samples"]):
            if sample["global_order"] != expected_orders[order_key, backend, size, batch]:
                raise NumericalExecutionError("execution seven-global-round order mismatch")
    expected = {(r["name"], backend, n) for r in registry for n in r["sizes"] for backend in CPU_BACKENDS}
    coordinates = {(backend, n) for backend in STARTUP_BACKENDS for n in STARTUP_SIZES}
    if set(costs) != expected or set(startup) != coordinates or set(proofs) != coordinates:
        raise NumericalExecutionError("execution observations incomplete")
    return costs, startup, proofs


def _result():
    return {"status": "rejected", "reason_codes": [], "holdouts": []}


def _finish(result):
    result["reason_codes"] = sorted(set(result["reason_codes"]))
    result["status"] = "rejected" if result["reason_codes"] else "accepted"
    return result


def _holdout(result, row, predicted):
    measured = statistics.median(_samples(row))
    predicted = _number(predicted, "execution predicted time")
    error = abs(predicted - measured) / measured
    result["holdouts"].append({"recipe": row.get("recipe"), "items": row["items"],
        "measured_seconds": measured, "predicted_seconds": predicted, "relative_error": error})
    if row["agreement_passed"] is not True:
        result["reason_codes"].append("numerical_agreement_failed")
    if error > MAX_RELATIVE_ERROR:
        result["reason_codes"].append("holdout_error")


def _startup_cost(backend, rows, proofs, threads):
    result = _result()
    result["coverage"] = FIXED_COVERAGE
    if backend == "native_serial":
        result.update(fixed_seconds=0.0, coverage="per-region serial counterfactual; no ABI call startup")
        return _finish(result)
    result["fixed_seconds"] = statistics.median(_samples(rows[backend, 0]))
    for size in STARTUP_SIZES:
        row, proof = rows[backend, size], proofs[backend, size]
        if row["agreement_passed"] is not True or not _proof_passed(proof, threads):
            result["reason_codes"].append("team_protocol_or_sentinel_proof_failed")
        if size:
            _holdout(result, row, result["fixed_seconds"])
    return _finish(result)


def memory_seconds(model, traffic_bytes, working_set_bytes):
    """Exact variable-memory interpolation; no range extrapolation."""
    traffic = _number(traffic_bytes, "execution memory traffic", zero=True)
    if type(working_set_bytes) is not int or not 0 <= working_set_bytes < 1 << 64:
        raise NumericalExecutionError("execution memory working set is unrepresentable")
    if not isinstance(model, dict) or model.get("kind") != "piecewise_bandwidth_v1":
        raise NumericalExecutionError("execution variable memory model unavailable")
    knots = model.get("knots")
    if not isinstance(knots, list) or not 2 <= len(knots) <= 8:
        raise NumericalExecutionError("execution memory knot budget invalid")
    previous = -1
    for knot in knots:
        if (not isinstance(knot, dict) or type(knot.get("working_set_bytes")) is not int
                or not max(previous, 0) < knot["working_set_bytes"] < 1 << 64):
            raise NumericalExecutionError("execution memory knot coordinates invalid")
        previous = knot["working_set_bytes"]
        _number(knot.get("seconds_per_traffic_byte"), "execution memory coefficient")
    if model.get("working_set_range") != [knots[0]["working_set_bytes"], knots[-1]["working_set_bytes"]]:
        raise NumericalExecutionError("execution memory range mismatch")
    if not knots[0]["working_set_bytes"] <= working_set_bytes <= knots[-1]["working_set_bytes"]:
        raise NumericalExecutionError("execution working set outside calibrated range")
    if traffic == 0:
        return 0.0
    coefficient = None
    for index, right in enumerate(knots):
        if working_set_bytes == right["working_set_bytes"]:
            coefficient = right["seconds_per_traffic_byte"]
            break
        if working_set_bytes < right["working_set_bytes"]:
            left = knots[index - 1]
            fraction = (working_set_bytes - left["working_set_bytes"]) / (right["working_set_bytes"] - left["working_set_bytes"])
            coefficient = left["seconds_per_traffic_byte"] + fraction * (right["seconds_per_traffic_byte"] - left["seconds_per_traffic_byte"])
            break
    return _number(traffic * coefficient, "execution memory product")


def _family(recipe, backend, costs, startup):
    result = _result()
    if startup["status"] != "accepted":
        result["reason_codes"] = ["startup_rejected"]
        return result
    rows = [costs[recipe["name"], backend, n] for n in recipe["sizes"]]
    if any(row["agreement_passed"] is not True for row in rows):
        result["reason_codes"] = ["numerical_agreement_failed"]
        return result
    fixed = startup["fixed_seconds"]
    training = [row for row in rows if row["items"] in recipe["fit_sizes"]]
    variable = [statistics.median(_samples(row)) - fixed for row in training]
    if any(not math.isfinite(v) or v <= 0 for v in variable):
        result["reason_codes"] = ["measured_startup_exceeds_training_time"]
        return result
    if recipe["role"] == "memory":
        knots = [{"working_set_bytes": row["working_set_bytes"],
                  "seconds_per_traffic_byte": v / row["traffic_bytes"]}
                 for row, v in zip(training, variable, strict=True)]
        result["model"] = {"kind": "piecewise_bandwidth_v1", "knots": knots,
            "working_set_range": [knots[0]["working_set_bytes"], knots[-1]["working_set_bytes"]],
            "fixed_cost_coverage": "independently measured startup removed"}
        def predict(row):
            return fixed + memory_seconds(result["model"], row["traffic_bytes"], row["working_set_bytes"])
    else:
        try:
            slope = math.fsum(row["items"] * v for row, v in zip(training, variable, strict=True)) /\
                sum(row["items"] ** 2 for row in training)
        except OverflowError as error:
            raise NumericalExecutionError("execution fitted slope overflow") from error
        result["model"] = {"seconds_per_item": _number(slope, "execution variable fitted slope"),
                           "fixed_seconds": fixed, "fit": "nonnegative slope with independently measured fixed term"}
        def predict(row):
            return fixed + row["items"] * slope
    for row in rows:
        if row["items"] not in recipe["fit_sizes"]:
            _holdout(result, row, predict(row))
    return _finish(result)


def _coefficients(families, recipes, backend, primitives):
    required = ("arithmetic", "memory", *primitives)
    if any(families[name][backend]["status"] != "accepted" for name in required):
        raise NumericalExecutionError("required execution family rejected")
    arithmetic = families["arithmetic"][backend]["model"]["seconds_per_item"] / recipes["arithmetic"]["arithmetic"]
    intrinsic = {}
    for primitive in primitives:
        recipe = recipes[primitive]
        intrinsic[primitive] = max(0.0, (families[primitive][backend]["model"]["seconds_per_item"] -
            recipe["arithmetic"] * arithmetic) / recipe["intrinsics"][primitive])
    return {"arithmetic_seconds_per_operation": arithmetic, "intrinsic_seconds_per_operation": intrinsic,
            "memory_cost_model": deepcopy(families["memory"][backend]["model"]), "memory_seconds_per_byte": None}


def _predict(recipe, size, coefficients, fixed, precision):
    try:
        compute = _number(recipe["arithmetic"] * coefficients["arithmetic_seconds_per_operation"] + math.fsum(
            count * coefficients["intrinsic_seconds_per_operation"][name] for name, count in recipe["intrinsics"].items()),
            "execution per-item compute duration", zero=True)
    except OverflowError as error:
        raise NumericalExecutionError("execution per-item compute duration overflow") from error
    traffic = 3 * size * (precision // 8)
    return _number(fixed + max(size * compute, memory_seconds(coefficients["memory_cost_model"], traffic, traffic)),
                   "execution predicted duration")


def _class(recipes, backend, costs, families, fits, startup, precision):
    result = _result()
    if not recipes:
        result["reason_codes"] = ["independent_holdouts_missing"]
        return result
    primitives = tuple(p for p in PRIMITIVES if any(p in r["intrinsics"] for r in recipes))
    try:
        coefficients = _coefficients(families, fits, backend, primitives)
    except NumericalExecutionError:
        result["reason_codes"] = ["required_family_rejected"]
        return result
    for recipe in recipes:
        for size in recipe["sizes"]:
            row = costs[recipe["name"], backend, size]
            _holdout(result, row, _predict(recipe, size, coefficients, startup["fixed_seconds"], precision))
    return _finish(result)


def _source_identity(profile, registry, generator, registry_id):
    require_numerical_profile_contract(profile)
    if (type(profile.get("precision_bits")) is not int or profile["precision_bits"] not in {32, 64}
            or type(profile.get("cpu_threads")) is not int or profile["cpu_threads"] != 4):
        raise NumericalExecutionError("execution schema one requires precision32/64 and four host threads")
    _validate_registry(registry, profile["precision_bits"])
    if not _sha(generator) or not _sha(registry_id) or _hash(list(registry)) != registry_id:
        raise NumericalExecutionError("execution frozen registry identity mismatch")


def validate_execution_preflight(profile, identity, proofs):
    """Reject source/team protocol drift before the first timed batch."""
    registry = _registry(profile.get("precision_bits"))
    generator, registry_id = _current_identities(profile.get("precision_bits"))
    _source_identity(profile, registry, generator, registry_id)
    if not isinstance(identity, dict) or identity.get("kind") != "numerical_execution_identity_v1":
        raise NumericalExecutionError("execution preflight identity missing")
    _identity(profile, identity, generator, registry_id)
    if not isinstance(proofs, list) or len(proofs) != len(STARTUP_BACKENDS) * len(STARTUP_SIZES):
        raise NumericalExecutionError("execution preflight requires six same-object team proofs")
    _hash(proofs)
    expected = {(backend, size) for backend in STARTUP_BACKENDS for size in STARTUP_SIZES}
    seen = set()
    for row in proofs:
        if not isinstance(row, dict) or row.get("kind") != "numerical_execution_team_proof_v1":
            raise NumericalExecutionError("execution preflight team proof missing")
        key = (row.get("backend"), row.get("items"))
        if (not isinstance(row.get("backend"), str) or type(row.get("items")) is not int
                or key not in expected or key in seen
                or row.get("timed_objects") != identity["timed_objects"]
                or row.get("proof_backend") != PROOF_BACKEND or not _proof_passed(row, profile["cpu_threads"])):
            raise NumericalExecutionError("execution preflight same-object team/sentinel proof failed")
        _control_identity(row, registry, profile["precision_bits"])
        seen.add(key)


def _build(profile, records, calibration, registry, generator, registry_id):
    _source_identity(profile, registry, generator, registry_id)
    if (not isinstance(calibration, dict) or len(calibration) > 128
            or calibration.get("application_profiled") is not False):
        raise NumericalExecutionError("execution calibration requires application_profiled=false")
    _hash(calibration)
    if not isinstance(records, list):
        raise NumericalExecutionError("execution records must be a bounded list")
    identities = [row for row in records if isinstance(row, dict) and row.get("kind") == "numerical_execution_identity_v1"]
    if len(identities) != 1:
        raise NumericalExecutionError("execution requires one original backend identity")
    identity = identities[0]
    _identity(profile, identity, generator, registry_id)
    costs, rows, proofs = _index(records, registry, identity)
    startup = {backend: _startup_cost(backend, rows, proofs, profile["cpu_threads"]) for backend in CPU_BACKENDS}
    fitting = {r["family"]: r for r in registry if r["role"] in {"coefficient", "memory"}}
    families = {family: {backend: _family(recipe, backend, costs, startup[backend]) for backend in CPU_BACKENDS}
                for family, recipe in fitting.items()}
    classes = {name: {backend: _class([r for r in registry if r["role"] == "holdout" and r["workload_class"] == name],
        backend, costs, families, fitting, startup[backend], profile["precision_bits"]) for backend in CPU_BACKENDS}
        for name in CLASSES}
    domains = {p: {backend: _class([r for r in registry if r["role"] == "holdout" and
        r.get("holdout_kind") == "domain" and r["family"] == p], backend, costs, families,
        fitting, startup[backend], profile["precision_bits"]) for backend in CPU_BACKENDS} for p in ("sqrt", "acos", "cos")}
    return {"schema_version": SCHEMA_VERSION, "phase": "cpu_only", "protocol_id": PROTOCOL_ID,
        "backend_id": BACKEND_ID, "generator_id": generator, "registry_id": registry_id,
        "registry": deepcopy(list(registry)), "identity": deepcopy(identity),
        "protocol": {"samples": SAMPLES, "minimum_batch_wall_seconds": MIN_BATCH_SECONDS,
                     "global_order": "round -> startup-size/rotating-two-backends -> recipe/size/rotating-three-backends",
                     "startup_sizes": list(STARTUP_SIZES), "holdout_error_ceiling": MAX_RELATIVE_ERROR,
                     "fixed_cost_coverage": FIXED_COVERAGE, "memory_access_class": ACCESS_CLASS},
        "measurements": deepcopy(records), "startup": startup, "families": families,
        "workload_validation": classes, "domain_validation": domains,
        "gpu_evidence_available": False, "calibration": deepcopy(calibration)}


def profile_from_execution_measurements(profile, records, calibration=None):
    registry = _registry(profile.get("precision_bits"))
    generator, registry_id = _current_identities(profile.get("precision_bits"))
    result = deepcopy(profile)
    if calibration is None:
        calibration = {"application_profiled": False}
    elif isinstance(calibration, dict):
        calibration = {"application_profiled": False, **calibration}
    result["numerical_execution"] = _build(profile, records, calibration, registry, generator, registry_id)
    return result


def validate_execution_profile(profile):
    section = profile.get("numerical_execution")
    if (not isinstance(section, dict) or type(section.get("schema_version")) is not int
            or section["schema_version"] != SCHEMA_VERSION or section.get("phase") != "cpu_only"):
        raise NumericalExecutionError("unsupported numerical execution section")
    reconstructed = _build(profile, section.get("measurements"), section.get("calibration"),
                           section.get("registry"), section.get("generator_id"), section.get("registry_id"))
    if section != reconstructed:
        raise NumericalExecutionError("saved execution evidence differs from strict raw reconstruction")


def _current_section(profile):
    validate_execution_profile(profile)
    section = profile["numerical_execution"]
    generator, registry_id = _current_identities(profile["precision_bits"])
    if (section["generator_id"] != generator or section["registry_id"] != registry_id
            or section["registry"] != list(_registry(profile["precision_bits"]))):
        raise NumericalExecutionError("execution current source/generator identity mismatch")
    return section


def execution_cpu_costs(profile, backend, counts, *, workload_class, workload_features, access_class):
    """Independent CPU evidence; this API never authorizes GPU placement."""
    section = _current_section(profile)
    if backend not in CPU_BACKENDS or access_class != ACCESS_CLASS:
        raise NumericalExecutionError("execution backend or source memory access class unavailable")
    try:
        pairs = list(counts.items()) if isinstance(counts, dict) else list(counts)
        if any(not isinstance(pair, (tuple, list)) or len(pair) != 2 for pair in pairs):
            raise ValueError("invalid primitive pair")
    except (TypeError, ValueError) as error:
        raise NumericalExecutionError("execution primitive counts unavailable") from error
    if (len(pairs) > len(PRIMITIVES) or any(not isinstance(name, str) for name, _ in pairs)
            or len({name for name, _ in pairs}) != len(pairs)
            or any(name not in PRIMITIVES or type(count) is not int or not 0 < count <= 100000 for name, count in pairs)):
        raise NumericalExecutionError("execution primitive counts unavailable")
    selected = "ordinary_expression_v2" if workload_class == "scalar_expression_v2" and\
        not any(name != "divide" for name, _ in pairs) else workload_class
    if selected not in CLASSES or section["workload_validation"][selected][backend]["status"] != "accepted":
        raise NumericalExecutionError("independent execution workload class rejected")
    features = workload_features.to_dict() if hasattr(workload_features, "to_dict") else workload_features
    if (not isinstance(features, dict) or type(features.get("schema_version")) is not int
            or features["schema_version"] != 2 or features.get("classification_complete") is not True):
        raise NumericalExecutionError("execution private-array classification unavailable")
    if workload_class == "fixed_private_array_v2":
        recipes = [r for r in section["registry"] if r["workload_class"] == workload_class]
        for key in ("private_array_groups", "private_array_elements", "max_private_array_elements", "max_private_array_rank"):
            limit = max(r["private_features"].get(key, 0) for r in recipes)
            if type(features.get(key)) is not int or not 0 < features[key] <= limit:
                raise NumericalExecutionError("execution private workload outside validated applicability")
    elif type(features.get("private_array_elements")) is not int or features["private_array_elements"] != 0:
        raise NumericalExecutionError("execution scalar workload contains private arrays")
    for primitive, _ in pairs:
        if primitive in section["domain_validation"] and section["domain_validation"][primitive][backend]["status"] != "accepted":
            raise NumericalExecutionError("execution primitive domain holdout rejected")
    fitting = {r["family"]: r for r in section["registry"] if r["role"] in {"coefficient", "memory"}}
    coefficients = _coefficients(section["families"], fitting, backend, tuple(name for name, _ in pairs))
    return {**coefficients, "backend_identity": backend, "fixed_seconds": section["startup"][backend]["fixed_seconds"],
        "fixed_cost_coverage": FIXED_COVERAGE, "item_range": [min(SIZES), max(SIZES)],
        "intrinsic_seconds_per_item": math.fsum(count * coefficients["intrinsic_seconds_per_operation"][name]
                                                for name, count in pairs),
        "cpu_affinity": deepcopy(section["identity"]["cpu_affinity"]), "fortran": deepcopy(section["identity"]["fortran"]),
        "cpu_protocol_environment": {name: section["identity"][name] for name in ("omp_wait_policy", "gomp_spincount")},
        "generator_id": section["generator_id"], "registry_id": section["registry_id"], "access_class": ACCESS_CLASS}


def execution_compute_model(profile, counts, *, workload_class="scalar_expression_v2",
                            workload_features=None, native_participation=None):
    """Return no complete source model until an independent GPU phase exists."""
    _current_section(profile)
    raise NumericalExecutionError("numerical execution CPU-only phase: independent GPU evidence unavailable")
