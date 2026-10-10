"""Independent validation of a CPU source work/span cost hypothesis.

Only the predefined basis recipes fit coefficients. Size, helper, expression,
private-storage and argument-domain holdouts remain independent. This additive
section neither replaces numerical-v2 evidence nor grants GPU legality.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from pathlib import Path

from compiler.numerical_contract import numerical_build_contract

from compiler.offload.calibrate import CalibrationError, _run, _tool
from compiler.offload.collective_calibration import normalize_fortran_options
from compiler.offload.cpu_dependency_model import (
    COEFFICIENT_FAMILIES,
    Coefficient,
    DependencyModelError,
    coefficient_family,
    evaluate_work_span,
    fit_basis_pair,
)
from compiler.offload.cpu_protocol_calibration import (
    CPU_BACKENDS,
    HOST_FLAGS,
    cpu_protocol_costs,
    validate_cpu_protocol,
    worker_renderer_identities,
)
from compiler.offload.numerical_calibration import (
    COMPUTE_FIT_SIZES,
    COMPUTE_SIZES,
    MAX_HOLDOUT_RELATIVE_ERROR,
    NumericalCalibrationError,
    _compute_samples,
    memory_compute_seconds,
    require_numerical_profile_contract,
)
from compiler.offload.schedule_calibrate import measurement_environment

PROTOCOL_ID = "source-work-span-cpu-v1"
ACCESS_CLASS = "pointwise_three_array_v1"
DOMAINS = {"sqrt": "finite_nonnegative_normal_or_zero", "acos": "finite_unit_interval",
           "cos": "finite_pi_interval", "divide_constant": "literal_three"}
PRIVATE_LIMITS = {"private_array_groups": 2, "private_array_elements": 32,
                  "referenced_private_array_elements": 32, "max_private_array_elements": 16,
                  "max_private_array_rank": 2}
BASIS_MEMORY_ENVELOPE = 1.0 / (1.0 - MAX_HOLDOUT_RELATIVE_ERROR)


def _hash_json(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def dependency_generator_identity():
    """Changes to graphs, fixtures, fitting or validation invalidate evidence."""
    directory = Path(__file__).parent
    names = ("compute_dependencies.py", "cpu_dependency_model.py", "cpu_dependency_workloads.py",
             "cpu_dependency_calibration.py", "cpu_dependency_driver.cpp")
    try:
        sources = b"".join(name.encode() + (directory / name).read_bytes() for name in names)
        sources += (directory.parent / "runtime" / "numeric.hpp").read_bytes()
        sources += numerical_build_contract()["identity"].encode()
        sources += _hash_json(worker_renderer_identities()).encode()
        return sha256(sources).hexdigest()
    except OSError as error:
        raise NumericalCalibrationError("CPU dependency calibration sources unavailable") from error


def _recipes(precision):
    from compiler.offload.cpu_dependency_workloads import dependency_recipes

    return {recipe.name: recipe for recipe in dependency_recipes(precision)}


def _identity(profile, identity, protocol):
    if not isinstance(identity, dict):
        raise NumericalCalibrationError("CPU dependency execution identity missing")
    if any(type(identity.get(name)) is not int for name in ("precision_bits", "cpu_threads")):
        raise NumericalCalibrationError("CPU dependency execution counts must be integers")
    affinity = identity.get("cpu_affinity")
    if not isinstance(affinity, list) or any(type(value) is not int for value in affinity):
        raise NumericalCalibrationError("CPU dependency placement must contain integer CPU indices")
    for name in ("actual_team_threads", "thread_limit"):
        if type(identity.get(name)) is not int or identity[name] < profile["cpu_threads"]:
            raise NumericalCalibrationError("CPU dependency actual team/thread limit unavailable")
    for name in ("omp_wait_policy", "gomp_spincount"):
        if name not in identity or (identity[name] is not None and
                (not isinstance(identity[name], str) or len(identity[name]) > 128)):
            raise NumericalCalibrationError("CPU dependency execution environment unavailable")
    expected = {"protocol_id": PROTOCOL_ID, "generator_id": dependency_generator_identity(),
                "precision_bits": profile["precision_bits"], "cpu_threads": profile["cpu_threads"],
                "cpu_affinity": protocol["execution_identity"]["cpu_affinity"],
                "fortran": protocol["execution_identity"]["fortran"],
                "host": protocol["execution_identity"]["host"],
                "omp_dynamic": False, "omp_proc_bind": "false",
                "base_numerical_identity": _hash_json(profile["numerical"]["identity"]),
                "cpu_protocol_identity": _hash_json(protocol)}
    for name in ("actual_team_threads", "thread_limit", "omp_wait_policy", "gomp_spincount"):
        expected[name] = protocol["execution_identity"].get(name)
    if any(identity.get(name) != value for name, value in expected.items()):
        raise NumericalCalibrationError("CPU dependency source/backend/placement identity mismatch")
    objects = identity.get("timed_objects")
    if (not isinstance(objects, dict) or not 1 <= len(objects) <= 160 or
            any(not isinstance(name, str) or not name or len(name) > 256 or
                not isinstance(value, str) or len(value) != 64 or
                any(char not in "0123456789abcdef" for char in value) for name, value in objects.items())):
        raise NumericalCalibrationError("CPU dependency timed object identities missing")


def _indexed(records, recipes):
    if not isinstance(records, list) or len(records) > len(recipes) * len(CPU_BACKENDS) * len(COMPUTE_SIZES):
        raise NumericalCalibrationError("CPU dependency observations exceed the bounded protocol")
    indexed = {}
    for ordinal, row in enumerate(records):
        if (not isinstance(row, dict) or row.get("kind") != "cpu_dependency_cost_v1" or
                not isinstance(row.get("recipe"), str) or not isinstance(row.get("backend"), str) or
                row.get("recipe") not in recipes or row.get("backend") not in CPU_BACKENDS or
                type(row.get("items")) is not int or row["items"] not in COMPUTE_SIZES):
            raise NumericalCalibrationError("unknown CPU dependency observation")
        recipe = recipes[row["recipe"]]
        role = "fit" if recipe.role == "basis" and row["items"] in COMPUTE_FIT_SIZES else "holdout"
        if row.get("role") != role or row.get("recipe_identity") != recipe.identity:
            raise NumericalCalibrationError("CPU dependency observation source/role mismatch")
        _compute_samples(row)
        key = row["recipe"], row["backend"], row["items"]
        if key in indexed:
            raise NumericalCalibrationError("duplicate CPU dependency observation")
        indexed[key] = {**row, "observation_id": ordinal}
    return indexed


def _slope(rows, costs, recipe, precision):
    if any(row is None or row.get("agreement_passed") is not True for row in rows):
        raise DependencyModelError("missing or incorrect basis observation")
    variable = [statistics.median(_compute_samples(row)) - costs["fixed_seconds"] for row in rows]
    if any(not math.isfinite(value) or value <= 0 for value in variable):
        raise DependencyModelError("startup exceeds basis duration")
    for row, value in zip(rows, variable, strict=True):
        traffic = row["items"] * recipe.memory_arrays * (precision // 8)
        memory = memory_compute_seconds(costs["memory_cost_model"], traffic, traffic)
        if value <= BASIS_MEMORY_ENVELOPE * memory:
            raise DependencyModelError("basis does not identify compute beyond the validated memory-error envelope")
    slope = math.fsum(row["items"] * value for row, value in zip(rows, variable, strict=True)) /\
        sum(row["items"] ** 2 for row in rows)
    if not math.isfinite(slope) or slope <= 0:
        raise DependencyModelError("nonpositive basis slope")
    return slope


def _predict(recipe, items, coefficients, costs, precision):
    compute = evaluate_work_span(recipe.graph, coefficients, precision_bits=precision).seconds_per_item
    traffic = items * recipe.memory_arrays * (precision // 8)
    # Every recipe uses distinct input/output arrays and one access per item.
    # A separate access contract is required when attaching this to a region.
    memory = memory_compute_seconds(costs["memory_cost_model"], traffic, traffic)
    return costs["fixed_seconds"] + max(items * compute, memory)


def _validate_rows(recipes, sizes, indexed, backend, coefficients, costs, precision):
    result = {"status": "rejected", "reason_codes": [], "holdouts": []}
    for recipe in recipes:
        for items in sizes:
            row = indexed.get((recipe.name, backend, items))
            if row is None or row.get("agreement_passed") is not True:
                result["reason_codes"].append("missing_or_incorrect_holdout")
                continue
            actual = statistics.median(_compute_samples(row))
            try:
                predicted = _predict(recipe, items, coefficients, costs, precision)
                error = abs(predicted - actual) / actual
            except (DependencyModelError, NumericalCalibrationError, OverflowError) as error:
                result["reason_codes"].append("holdout_model_unavailable")
                result["holdouts"].append({"recipe": recipe.name, "items": items, "reason": str(error),
                                           "observation_id": row["observation_id"]})
                continue
            result["holdouts"].append({"recipe": recipe.name, "items": items, "measured_seconds": actual,
                "predicted_seconds": predicted, "relative_error": error, "observation_id": row["observation_id"]})
            if not math.isfinite(error) or error > MAX_HOLDOUT_RELATIVE_ERROR:
                result["reason_codes"].append("holdout_error")
    if not recipes:
        result["reason_codes"].append("missing_holdout_recipes")
    result["reason_codes"] = sorted(set(result["reason_codes"]))
    result["status"] = "rejected" if result["reason_codes"] else "accepted"
    return result


def _backend(profile, recipes, indexed, backend):
    result = {"status": "rejected", "reason_codes": [], "families": {},
              "workload_validation": {}, "domain_validation": {}}
    try:
        costs = cpu_protocol_costs(profile, backend, access_class=ACCESS_CLASS)
    except NumericalCalibrationError as error:
        result["reason_codes"] = ["startup_or_memory_protocol_unavailable"]
        result["reason"] = str(error)
        return result
    result["execution_costs"] = costs
    known = {}
    for family in COEFFICIENT_FAMILIES:
        basis = sorted((recipe for recipe in recipes.values() if recipe.role == "basis" and
                        recipe.coefficient_family == family), key=lambda recipe: recipe.width)
        evidence = {"status": "rejected", "reason_codes": [], "holdouts": []}
        result["families"][family] = evidence
        if len(basis) != 2 or [recipe.width for recipe in basis] != [1, 4]:
            evidence["reason_codes"] = ["missing_prescribed_basis_pair"]
            continue
        try:
            slopes = [_slope([indexed.get((recipe.name, backend, size)) for size in COMPUTE_FIT_SIZES],
                             costs, recipe, profile["precision_bits"]) for recipe in basis]
            coefficient = fit_basis_pair(basis[0].graph, basis[1].graph, *slopes, family=family,
                                         known=known, precision_bits=profile["precision_bits"])
        except (DependencyModelError, NumericalCalibrationError, OverflowError) as error:
            evidence["reason_codes"] = ["basis_identification_failed"]
            evidence["reason"] = str(error)
            continue
        candidate = {**known, family: coefficient}
        evidence.update(_validate_rows(basis, tuple(size for size in COMPUTE_SIZES if size not in COMPUTE_FIT_SIZES),
            indexed, backend, candidate, costs, profile["precision_bits"]))
        evidence["coefficient"] = coefficient.to_dict()
        evidence["basis_slopes_seconds_per_item"] = slopes
        if evidence["status"] == "accepted":
            known[family] = coefficient
    for workload in ("ordinary_expression_v2", "scalar_expression_v2", "fixed_private_array_v2"):
        selected = [recipe for recipe in recipes.values() if recipe.role == "structural_holdout" and
                    recipe.workload_class == workload]
        result["workload_validation"][workload] = _validate_rows(selected, COMPUTE_SIZES, indexed, backend,
            known, costs, profile["precision_bits"])
    for family in DOMAINS:
        selected = [recipe for recipe in recipes.values() if recipe.role == "domain_holdout" and
                    recipe.coefficient_family == family]
        # Literal-three applicability is exercised by both independent basis
        # sizes and every mixed expression; it is not an argument lattice.
        if family == "divide_constant":
            result["domain_validation"][family] = {"status": result["families"][family]["status"],
                "reason_codes": list(result["families"][family]["reason_codes"]), "domain": DOMAINS[family]}
        else:
            evidence = _validate_rows(selected, COMPUTE_SIZES, indexed, backend, known, costs, profile["precision_bits"])
            result["domain_validation"][family] = {**evidence, "domain": DOMAINS[family]}
    result["status"] = "accepted" if all(record["status"] == "accepted" for record in
        (*result["families"].values(), *result["workload_validation"].values(), *result["domain_validation"].values())) else "rejected"
    if result["status"] != "accepted":
        result["reason_codes"] = ["one_or_more_independent_classes_rejected"]
    return result


def profile_from_dependency_measurements(profile, records, execution_identity, *, calibration=None):
    """Retain all raw observations and independently rejected classes."""
    if calibration is not None and not isinstance(calibration, dict):
        raise NumericalCalibrationError("CPU dependency calibration metadata must be an object")
    protocol = validate_cpu_protocol(profile)
    _identity(profile, execution_identity, protocol)
    recipes = _recipes(profile["precision_bits"])
    indexed = _indexed(records, recipes)
    result = deepcopy(profile)
    result["cpu_dependency"] = {"schema_version": 1, "protocol_id": PROTOCOL_ID,
        "generator_id": dependency_generator_identity(), "execution_identity": deepcopy(execution_identity),
        "recipe_identities": {name: recipe.identity for name, recipe in recipes.items()},
        "protocol": {"fit_sizes": list(COMPUTE_FIT_SIZES), "sizes": list(COMPUTE_SIZES),
                     "batches": 7, "minimum_batch_wall_seconds": 0.2, "interleaved": True,
                     "single_chain_component": "span", "four_chain_component": "work",
                     "basis_minimum_memory_ratio": BASIS_MEMORY_ENVELOPE,
                     "constant_or_invariant_numeric_source_applicability": "unavailable",
                     "dynamic_division_source_applicability": "unavailable_without_source_domain_and_holdouts",
                     "access_class": ACCESS_CLASS, "domains": dict(DOMAINS), "private_limits": dict(PRIVATE_LIMITS),
                     "maximum_holdout_relative_error": MAX_HOLDOUT_RELATIVE_ERROR},
        "measurements": deepcopy(records),
        "backends": {backend: _backend(profile, recipes, indexed, backend) for backend in CPU_BACKENDS},
        "calibration": {**(calibration or {}), "application_profiled": False, "favorable_retry": False}}
    return result


def validate_dependency_profile(profile):
    """Saved coefficients/statuses cannot override raw/source validation."""
    saved = profile.get("cpu_dependency")
    if not isinstance(saved, dict) or type(saved.get("schema_version")) is not int or saved["schema_version"] != 1:
        raise NumericalCalibrationError("CPU dependency evidence missing or unsupported")
    rebuilt = profile_from_dependency_measurements(profile, saved.get("measurements"),
        saved.get("execution_identity"), calibration=saved.get("calibration"))["cpu_dependency"]
    if saved != rebuilt:
        raise NumericalCalibrationError("CPU dependency evidence differs from raw observations or source identity")
    return saved


def cpu_dependency_costs(profile, graph, backend, *, workload_class, workload_features,
                         primitive_domains, access_class):
    """Return cost-only coefficients after every required independent proof."""
    require_numerical_profile_contract(profile)
    if backend not in CPU_BACKENDS or access_class != ACCESS_CLASS:
        raise NumericalCalibrationError("CPU dependency execution/access class unavailable")
    section = validate_dependency_profile(profile)
    evidence = section["backends"][backend]
    try:
        families = {coefficient_family(operation.family) for operation in graph.operations}
    except (DependencyModelError, AttributeError) as error:
        raise NumericalCalibrationError(str(error)) from error
    if any(operation.constant is not False or operation.invariant is not False for operation in graph.operations):
        raise NumericalCalibrationError("constant/invariant native folding or hoisting applicability unavailable")
    if "divide_dynamic" in families:
        raise NumericalCalibrationError("dynamic division source domain applicability unavailable")
    validation_class = ("ordinary_expression_v2" if workload_class == "scalar_expression_v2" and
                        not families - {"ordinary", "divide_dynamic", "divide_constant"} else workload_class)
    if "execution_costs" not in evidence or validation_class not in evidence["workload_validation"]:
        raise NumericalCalibrationError("CPU dependency startup/memory/workload unavailable")
    if evidence["workload_validation"][validation_class]["status"] != "accepted":
        raise NumericalCalibrationError("CPU dependency expression class rejected")
    features = workload_features.to_dict() if hasattr(workload_features, "to_dict") else workload_features
    if (not isinstance(features, dict) or type(features.get("schema_version")) is not int or
            features["schema_version"] != 2 or features.get("classification_complete") is not True):
        raise NumericalCalibrationError("CPU dependency private storage classification unavailable")
    for name, limit in PRIVATE_LIMITS.items():
        if type(features.get(name)) is not int or not 0 <= features[name] <= limit:
            raise NumericalCalibrationError("CPU dependency private storage outside applicability")
    if (workload_class == "scalar_expression_v2" and features["private_array_elements"] != 0 or
            workload_class == "fixed_private_array_v2" and features["private_array_elements"] == 0):
        raise NumericalCalibrationError("CPU dependency storage/workload class mismatch")
    try:
        coefficients = {}
        for family in families:
            record = evidence["families"].get(family, {})
            if record.get("status") != "accepted":
                raise NumericalCalibrationError("CPU dependency primitive rejected: " + family)
            if family in DOMAINS:
                validation = evidence["domain_validation"].get(family, {})
                if (not isinstance(primitive_domains, dict) or primitive_domains.get(family) != DOMAINS[family] or
                        validation.get("status") != "accepted"):
                    raise NumericalCalibrationError("CPU dependency primitive domain unavailable: " + family)
            coefficients[family] = Coefficient(**record["coefficient"])
        for operation in graph.operations:
            if operation.family == "divide_constant":
                _require_literal_three(operation)
        estimate = evaluate_work_span(graph, coefficients, precision_bits=profile["precision_bits"])
    except (DependencyModelError, AttributeError) as error:
        raise NumericalCalibrationError(str(error)) from error
    return {**deepcopy(evidence["execution_costs"]), **estimate.to_dict(),
            "item_range": [min(COMPUTE_SIZES), max(COMPUTE_SIZES)], "dependency_identity": graph.identity,
            "generator_id": section["generator_id"], "access_class": access_class}


def _require_literal_three(operation):
    """A measured literal divisor is not an arbitrary constant-division cost."""
    operands = operation.literal_operands
    if (not isinstance(operands, tuple) or len(operands) != 2 or not isinstance(operands[1], tuple) or
            len(operands[1]) != 2 or operands[1][0] != operation.dtype.value or
            not isinstance(operands[1][1], str) or len(operands[1][1]) > 128):
        raise NumericalCalibrationError("constant division lacks literal-three provenance")
    spelling = operands[1][1].strip().lower()
    if not re.fullmatch(r"\+?(?:\d+(?:\.\d*)?|\.\d+)(?:[ed][+-]?\d+)?(?:_[a-z0-9]+)?", spelling):
        raise NumericalCalibrationError("constant division literal is unsupported")
    try:
        value = Decimal(spelling.split("_", 1)[0].replace("d", "e"))
    except InvalidOperation as error:
        raise NumericalCalibrationError("constant division literal is unsupported") from error
    if value != Decimal(3):
        raise NumericalCalibrationError("constant division is outside calibrated literal-three applicability")


def parse_dependency_measurements(text):
    """Parse bounded producer output without discarding failed observations."""
    identities, rows = [], []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as error:
            raise NumericalCalibrationError("CPU dependency emitted non-JSON output") from error
        if not isinstance(row, dict):
            raise NumericalCalibrationError("CPU dependency emitted non-object output")
        if row.get("kind") == "cpu_dependency_identity_v1":
            identities.append({name: value for name, value in row.items() if name != "kind"})
        elif row.get("kind") == "cpu_dependency_cost_v1":
            if len(rows) >= 26 * len(CPU_BACKENDS) * len(COMPUTE_SIZES):
                raise NumericalCalibrationError("CPU dependency emitted too many observations")
            _compute_samples(row)
            rows.append(row)
        else:
            raise NumericalCalibrationError("unknown CPU dependency output kind")
    if len(identities) != 1:
        raise NumericalCalibrationError("exactly one CPU dependency execution identity is required")
    return rows, identities[0]


def calibrate_dependency(profile, args, *, run=_run):
    """Compile fixed fixtures once, validate the backend, then sample once.

    This CPU-only producer needs a separate guarded execution lane. It never
    reads application timings, changes thread budgets, retries observations or
    overwrites an existing attempt directory.
    """
    from compiler.offload.cpu_dependency_workloads import (
        dependency_recipes,
        native_identity_source,
        write_dependency_registry,
    )

    require_numerical_profile_contract(profile)
    protocol = validate_cpu_protocol(profile)
    available_memory = []
    for backend in CPU_BACKENDS:
        try:
            cpu_protocol_costs(profile, backend, access_class=ACCESS_CLASS)
        except NumericalCalibrationError:
            continue
        available_memory.append(backend)
    if not available_memory:
        raise NumericalCalibrationError("CPU dependency calibration has no accepted fresh execution/memory protocol")
    flags = list(args.fortran_flag or [])
    if (not flags or "-fopenmp" not in flags or any("fast-math" in flag or
            "lto" in flag or flag == "-Ofast" for flag in flags)):
        raise NumericalCalibrationError("supply original Fortran flags including -fopenmp, without LTO or fast-math")
    affinity = protocol["execution_identity"]["cpu_affinity"]
    if not set(affinity) <= os.sched_getaffinity(0):
        raise NumericalCalibrationError("CPU dependency calibrated placement is unavailable")
    fortran, host = _tool(args.fortran, ("gfortran",)), _tool(args.host_cxx, ("g++",))
    target = Path(args.build_dir).resolve()
    target.mkdir(parents=True, exist_ok=False)
    generator = dependency_generator_identity()
    recipes = dependency_recipes(profile["precision_bits"])
    (target / "recipes.json").write_text(json.dumps([recipe.to_dict() for recipe in recipes], indent=2) + "\n")
    write_dependency_registry(target / "cpu_dependency_recipes.hpp", recipes, generator)
    native_identity = target / "native-identity.f90"
    native_identity.write_text(native_identity_source())
    runtime = Path(__file__).resolve().parents[1] / "runtime"
    objects, commands = [], []
    used_names = set()
    for recipe in recipes:
        for name, source in (*recipe.fortran_sources, *recipe.cpp_sources):
            if Path(name).name != name or name in used_names:
                raise NumericalCalibrationError("CPU dependency fixture source names are not unique basenames")
            used_names.add(name)
            path = target / name
            path.write_text(source)
            obj = path.with_name(path.name + ".o")
            compiler, options = (fortran, flags) if path.suffix == ".f90" else (host, list(HOST_FLAGS))
            command = [compiler, *options, "-I", str(runtime), "-c", str(path), "-o", str(obj)]
            run(command, target, target / (name + ".build.log"), timeout=180)
            commands.append(command)
            objects.append(obj)
    identity_object = target / "native-identity.o"
    command = [fortran, *flags, "-c", str(native_identity), "-o", str(identity_object)]
    run(command, target, target / "identity-build.log", timeout=180)
    commands.append(command)
    objects.append(identity_object)
    driver = target / "driver.o"
    command = [host, *HOST_FLAGS, "-I", str(target), "-c",
               str(Path(__file__).with_name("cpu_dependency_driver.cpp")), "-o", str(driver)]
    run(command, target, target / "driver-build.log", timeout=180)
    commands.append(command)
    objects.append(driver)
    binary = target / "cpu-dependency"
    command = [host, "-fopenmp", *map(str, objects), "-lgfortran", "-o", str(binary)]
    run(command, target, target / "link.log", timeout=180)
    commands.append(command)
    (target / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    host_version = run([host, "--version"], target, target / "host-version.txt", timeout=30).strip()
    environment = measurement_environment(os.environ)
    environment["OMP_NUM_THREADS"] = str(profile["cpu_threads"])
    invocation = ["taskset", "-c", ",".join(map(str, affinity)), str(binary), str(profile["cpu_threads"])]
    rows, actual = parse_dependency_measurements(run([*invocation, "--identity"], target,
        target / "identity.jsonl", timeout=30, env=environment))
    if rows:
        raise NumericalCalibrationError("CPU dependency identity probe unexpectedly measured work")
    if actual.get("registry_generator_id") != generator:
        raise NumericalCalibrationError("CPU dependency registry/source identity mismatch")
    try:
        actual["fortran"]["semantic_options"] = normalize_fortran_options(actual["fortran"]["compiler_options"])
    except (KeyError, TypeError, ValueError) as error:
        raise NumericalCalibrationError("CPU dependency native Fortran identity missing") from error
    actual.update(protocol_id=PROTOCOL_ID, generator_id=generator,
        host={"compiler_version": host_version, "semantic_options": list(HOST_FLAGS)},
        base_numerical_identity=_hash_json(profile["numerical"]["identity"]),
        cpu_protocol_identity=_hash_json(protocol),
        timed_objects={path.name: sha256(path.read_bytes()).hexdigest() for path in objects})
    # Check all identities before the first timed batch. Unsupported placement
    # or semantic flags cannot silently create observations for another model.
    _identity(profile, actual, protocol)
    measured_text = run(invocation, target, target / "measurements.jsonl", timeout=3600, env=environment)
    rows, measured_identity = parse_dependency_measurements(measured_text)
    for name in ("cpu_threads", "precision_bits", "cpu_affinity", "omp_dynamic", "omp_proc_bind", "registry_generator_id",
                 "actual_team_threads", "thread_limit", "omp_wait_policy", "gomp_spincount"):
        if measured_identity.get(name) != actual[name]:
            raise NumericalCalibrationError("CPU dependency measurement identity changed")
    if measured_identity.get("fortran") != {name: actual["fortran"][name] for name in
                                           ("compiler_version", "compiler_options")}:
        raise NumericalCalibrationError("CPU dependency measurement native flags changed")
    raw_batches = target / "cpu-dependency-raw-samples.jsonl"
    return profile_from_dependency_measurements(profile, rows, actual,
        calibration={"commands": commands, "build_directory": str(target),
                     "timed_binary_sha256": sha256(binary.read_bytes()).hexdigest(),
                     "raw_batch_artifact": {"path": str(raw_batches), "sha256": sha256(raw_batches.read_bytes()).hexdigest()}})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--fortran")
    parser.add_argument("--host-cxx")
    parser.add_argument("--fortran-flag", action="append")
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        parser.error("output already exists; preserve every calibration attempt")
    try:
        profile = json.loads(Path(args.profile).read_text())
        result = calibrate_dependency(profile, args)
        with output.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    except (OSError, ValueError, CalibrationError) as error:
        parser.exit(1, str(error) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
