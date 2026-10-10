"""Strict reconstruction and independent failures of the CPU-only protocol."""

import json
import math
from copy import deepcopy

import pytest

from compiler.numerical_contract import numerical_build_contract
from compiler.offload import numerical_execution as execution
from compiler.offload.collective_calibration import normalize_fortran_options
from compiler.offload.numerical_execution_workloads import execution_registry

GENERATOR = "e" * 64
ARITHMETIC = 1e-9
MEMORY = 1e-10
INTRINSICS = {"sqrt": 20e-9, "acos": 100e-9, "cos": 60e-9, "divide": 5e-9}
FIXED = {"native_serial": 0.0, "native_fork_join": 1e-6, "generated_cpu": 1.2e-6}


def samples(seconds, orders):
    repeats = math.ceil(0.25 / seconds)
    return [{"batch": batch, "repetitions": repeats,
             "elapsed_seconds": repeats * seconds, "wall_seconds": repeats * seconds,
             "global_order": orders[batch]} for batch in range(execution.SAMPLES)]


def observations(precision=64):
    registry = list(execution_registry(precision))
    registry_id = execution._hash(registry)
    orders = execution._expected_orders(registry)
    objects = {"native.o": "a" * 64, "generated.o": "b" * 64}
    options = "-O3 -fopenmp -ffp-contract=off"
    profile = {"precision_bits": precision, "cpu_threads": 4,
               "numerical_contract": numerical_build_contract(),
               "hardware": {"cpu_name": "synthetic test CPU"},
               "toolchain": {"host_cxx_version": "GNU C++ test backend"},
               "numerical": {"retained_legacy_evidence": True}}
    identity = {"kind": "numerical_execution_identity_v1", "protocol_id": execution.PROTOCOL_ID,
                "backend_id": execution.BACKEND_ID, "generator_id": GENERATOR, "registry_id": registry_id,
                "cpu_threads": 4, "precision_bits": precision, "cpu_affinity": [4, 5, 6, 7],
                "actual_team_threads": 4, "thread_limit": 2147483647, "omp_dynamic": False,
                "cpu_name": profile["hardware"]["cpu_name"],
                "omp_proc_bind": "false", "omp_wait_policy": None, "gomp_spincount": None,
                "fortran": {"compiler_version": "GNU Fortran test backend", "compiler_options": options,
                            "semantic_options": normalize_fortran_options(options)},
                "host": {"compiler_version": profile["toolchain"]["host_cxx_version"],
                         "semantic_options": list(execution.HOST_FLAGS)},
                "proof_backend": execution.PROOF_BACKEND, "timed_objects": objects}
    records = [identity]
    for backend in execution.STARTUP_BACKENDS:
        for size in execution.STARTUP_SIZES:
            control = {"recipe": "memory", "recipe_id": registry[1]["recipe_id"], "family": "memory",
                       "role": "holdout", "traffic_bytes": 3 * size * (precision // 8),
                       "working_set_bytes": 3 * size * (precision // 8)}
            records.append({"kind": "numerical_execution_startup_v1", "backend": backend, "items": size,
                            "agreement_passed": True, **control,
                            "samples": samples(FIXED[backend], [orders["startup", backend, size, batch]
                                                               for batch in range(execution.SAMPLES)])})
            records.append({"kind": "numerical_execution_team_proof_v1", "backend": backend, "items": size,
                            "agreement_passed": True, **control, "proof_backend": identity["proof_backend"],
                            "timed_objects": objects, "parallel_entries": 1, "wrong_team_visits": 0,
                            "thread_visits": [1, 1, 1, 1]})
    for recipe in registry:
        compute = recipe["arithmetic"] * ARITHMETIC + math.fsum(
            count * INTRINSICS[name] for name, count in recipe["intrinsics"].items())
        for size in recipe["sizes"]:
            traffic = 3 * size * (precision // 8)
            for backend in execution.CPU_BACKENDS:
                variable = traffic * MEMORY if recipe["role"] == "memory" else size * compute
                if recipe["role"] == "holdout":
                    variable = max(variable, traffic * MEMORY)
                seconds = FIXED[backend] + variable
                records.append({"kind": "numerical_execution_cost_v1", "recipe": recipe["name"],
                                "family": recipe["family"], "backend": backend, "items": size,
                                "role": "fit" if size in recipe["fit_sizes"] else "holdout",
                                "recipe_id": recipe["recipe_id"], "agreement_passed": True,
                                "traffic_bytes": traffic, "working_set_bytes": traffic,
                                "samples": samples(seconds, [orders[recipe["name"], backend, size, batch]
                                                             for batch in range(execution.SAMPLES)])})
    return profile, records, registry


@pytest.fixture
def evidence(monkeypatch):
    profile, records, registry = observations()
    monkeypatch.setattr(execution, "_registry", lambda precision: deepcopy(registry))
    monkeypatch.setattr(execution, "_current_identities", lambda precision: (GENERATOR, execution._hash(registry)))
    return profile, records, registry


def build(evidence):
    profile, records, _ = evidence
    return execution.profile_from_execution_measurements(profile, records)


def select(records, kind="numerical_execution_cost_v1", **coordinates):
    return next(row for row in records if row["kind"] == kind and
                all(row.get(key) == value for key, value in coordinates.items()))


def scale(row, factor):
    for sample in row["samples"]:
        sample["elapsed_seconds"] *= factor
        sample["wall_seconds"] *= factor


def scalar_features():
    return {"schema_version": 2, "classification_complete": True, "private_array_elements": 0}


def cpu_model(profile, backend="native_fork_join", counts=None, **kwargs):
    return execution.execution_cpu_costs(profile, backend, {} if counts is None else counts,
        workload_class=kwargs.pop("workload_class", "scalar_expression_v2"),
        workload_features=kwargs.pop("workload_features", scalar_features()),
        access_class=kwargs.pop("access_class", execution.ACCESS_CLASS), **kwargs)


@pytest.mark.parametrize("precision", [32, 64])
def test_fixed_source_registry_and_complete_global_protocol(monkeypatch, precision):
    data = observations(precision)
    profile, records, registry = data
    monkeypatch.setattr(execution, "_registry", lambda value: deepcopy(registry))
    monkeypatch.setattr(execution, "_current_identities", lambda value: (GENERATOR, execution._hash(registry)))
    original = deepcopy(profile)
    result = build(data)
    section = result["numerical_execution"]
    execution.validate_execution_profile(result)
    assert profile == original
    assert result["numerical"] == original["numerical"]
    assert len(records) == execution.MAX_RECORDS == 304
    assert len(execution._expected_orders(registry)) == 2079
    assert section["measurements"] == records
    assert section["calibration"] == {"application_profiled": False}
    assert section["phase"] == "cpu_only"
    assert section["gpu_evidence_available"] is False
    for group in ("families", "workload_validation", "domain_validation"):
        assert all(role["status"] == "accepted" for family in section[group].values() for role in family.values())


def test_independent_startup_removed_from_memory_and_added_once(evidence):
    result = build(evidence)
    for backend in execution.CPU_BACKENDS:
        model = cpu_model(result, backend, {"divide": 11})
        assert model["fixed_seconds"] == pytest.approx(FIXED[backend])
        assert model["arithmetic_seconds_per_operation"] == pytest.approx(ARITHMETIC)
        assert model["intrinsic_seconds_per_operation"]["divide"] == pytest.approx(INTRINSICS["divide"])
        assert model["intrinsic_seconds_per_item"] == pytest.approx(11 * INTRINSICS["divide"])
        assert all(knot["seconds_per_traffic_byte"] == pytest.approx(MEMORY)
                   for knot in model["memory_cost_model"]["knots"])
    memory = result["numerical_execution"]["families"]["memory"]["native_fork_join"]
    holdout = memory["holdouts"][0]
    traffic = 3 * holdout["items"] * 8
    assert holdout["predicted_seconds"] == pytest.approx(FIXED["native_fork_join"] + traffic * MEMORY)
    assert "intercept" not in result["numerical_execution"]["families"]["arithmetic"]["native_fork_join"]["model"]


def test_cpu_only_model_never_inherits_legacy_gpu_evidence(evidence):
    result = build(evidence)
    with pytest.raises(execution.NumericalExecutionError, match="CPU-only.*GPU evidence unavailable"):
        execution.execution_compute_model(result, {"sqrt": 5, "divide": 11}, native_participation="fork_join")


def test_preflight_authenticates_same_full_contract_without_any_costs(evidence):
    profile, records, _ = evidence
    proofs = [row for row in records if row["kind"] == "numerical_execution_team_proof_v1"]
    assert execution.validate_execution_preflight(profile, records[0], proofs) is None
    assert "numerical_execution" not in profile


@pytest.mark.parametrize("mutation", ["incomplete", "duplicate", "wrong_team", "objects", "control",
                                      "footprint", "backend_type", "cpu", "host", "fortran"])
def test_preflight_rejects_protocol_drift_before_sampling(evidence, mutation):
    profile, records, _ = evidence
    proofs = [row for row in records if row["kind"] == "numerical_execution_team_proof_v1"]
    if mutation == "incomplete":
        proofs.pop()
    elif mutation == "duplicate":
        proofs[-1] = deepcopy(proofs[0])
    elif mutation == "wrong_team":
        proofs[0]["thread_visits"] = [4, 0, 0, 0]
    elif mutation == "objects":
        proofs[0]["timed_objects"] = {"different.o": "f" * 64}
    elif mutation == "control":
        proofs[0]["recipe_id"] = "f" * 64
    elif mutation == "footprint":
        proofs[0]["traffic_bytes"] += 1
    elif mutation == "backend_type":
        proofs[0]["backend"] = []
    elif mutation == "cpu":
        records[0]["cpu_name"] = "different processor"
    elif mutation == "host":
        records[0]["host"]["semantic_options"].pop()
    else:
        records[0]["fortran"]["compiler_options"] = "-I"
    with pytest.raises(execution.NumericalExecutionError):
        execution.validate_execution_preflight(profile, records[0], proofs)


@pytest.mark.parametrize(("kind", "field", "value"), [
    ("numerical_execution_startup_v1", "recipe", "arithmetic"),
    ("numerical_execution_startup_v1", "recipe_id", "f" * 64),
    ("numerical_execution_startup_v1", "working_set_bytes", 999),
    ("numerical_execution_team_proof_v1", "role", "fit"),
    ("numerical_execution_team_proof_v1", "family", "arithmetic"),
])
def test_every_startup_and_team_record_binds_the_original_memory_control(evidence, kind, field, value):
    select(evidence[1], kind, backend="native_fork_join", items=0)[field] = value
    with pytest.raises(execution.NumericalExecutionError, match="control identity"):
        build(evidence)


@pytest.mark.parametrize(("field", "value"), [("actual_team_threads", 1), ("thread_limit", 1),
    ("cpu_affinity", [0, 1, 2, 2]), ("omp_dynamic", True), ("omp_proc_bind", "true"),
    ("omp_wait_policy", "bad\nvalue"), ("gomp_spincount", "x" * 129)])
def test_original_team_and_environment_identity_required(evidence, field, value):
    evidence[1][0][field] = value
    with pytest.raises(execution.NumericalExecutionError, match="identity"):
        build(evidence)


@pytest.mark.parametrize("field", ["omp_wait_policy", "gomp_spincount"])
def test_effective_wait_and_spin_values_cannot_be_missing(evidence, field):
    del evidence[1][0][field]
    with pytest.raises(execution.NumericalExecutionError, match="wait/spin"):
        build(evidence)


@pytest.mark.parametrize(("proof_field", "value"), [("parallel_entries", 0), ("thread_visits", [1]),
    ("thread_visits", [True] * 4), ("wrong_team_visits", 1), ("agreement_passed", False)])
def test_zero_trip_team_proof_is_independently_required(evidence, proof_field, value):
    row = select(evidence[1], "numerical_execution_team_proof_v1", backend="generated_cpu", items=0)
    row[proof_field] = value
    section = build(evidence)["numerical_execution"]
    assert section["startup"]["generated_cpu"]["status"] == "rejected"
    assert section["families"]["memory"]["generated_cpu"]["reason_codes"] == ["startup_rejected"]
    assert section["families"]["memory"]["native_fork_join"]["status"] == "accepted"


def test_startup_dominance_controls_and_negative_subtraction_reject(evidence):
    scale(select(evidence[1], "numerical_execution_startup_v1", backend="native_fork_join", items=8), 2)
    section = build(evidence)["numerical_execution"]
    assert section["startup"]["native_fork_join"]["reason_codes"] == ["holdout_error"]
    assert section["startup"]["generated_cpu"]["status"] == "accepted"
    profile, rows, registry = observations()
    for size in execution.STARTUP_SIZES:
        scale(select(rows, "numerical_execution_startup_v1", backend="generated_cpu", items=size), 1e9)
    section = build((profile, rows, registry))["numerical_execution"]
    assert section["families"]["memory"]["generated_cpu"]["reason_codes"] == ["measured_startup_exceeds_training_time"]


def test_independent_math_failure_does_not_reject_ordinary_work(evidence):
    scale(select(evidence[1], recipe="sqrt", backend="native_serial", items=131072), 2)
    result = build(evidence)
    section = result["numerical_execution"]
    assert section["families"]["sqrt"]["native_serial"]["status"] == "rejected"
    assert section["workload_validation"]["ordinary_expression_v2"]["native_serial"]["status"] == "accepted"
    assert section["workload_validation"]["scalar_expression_v2"]["native_serial"]["status"] == "rejected"
    assert cpu_model(result, "native_serial", {"divide": 1})
    with pytest.raises(execution.NumericalExecutionError, match="class rejected"):
        cpu_model(result, "native_serial", {"sqrt": 1})


@pytest.mark.parametrize(("factor", "status"), [(1.32, "accepted"), (1.34, "rejected")])
def test_independent_holdout_error_ceiling_remains_twenty_five_percent(evidence, factor, status):
    scale(select(evidence[1], recipe="sqrt", backend="native_serial", items=131072), factor)
    result = build(evidence)["numerical_execution"]["families"]["sqrt"]["native_serial"]
    assert result["status"] == status
    assert (result["holdouts"][0]["relative_error"] <= execution.MAX_RELATIVE_ERROR) == (status == "accepted")


@pytest.mark.parametrize(("recipe", "workload"), [("ordinary_mix", "ordinary_expression_v2"),
    ("dependent_arithmetic", "ordinary_expression_v2"), ("helper_scalar_mix", "scalar_expression_v2"),
    ("helper_private_mix", "fixed_private_array_v2"), ("domain_acos", "scalar_expression_v2")])
def test_every_independent_class_helper_dependency_and_domain_control_is_used(evidence, recipe, workload):
    scale(select(evidence[1], recipe=recipe, backend="generated_cpu", items=524288), 2)
    section = build(evidence)["numerical_execution"]
    assert section["workload_validation"][workload]["generated_cpu"]["status"] == "rejected"
    assert section["workload_validation"][workload]["native_fork_join"]["status"] == "accepted"
    if recipe == "domain_acos":
        assert section["domain_validation"]["acos"]["generated_cpu"]["status"] == "rejected"


def test_raw_numerical_failure_is_retained_even_with_good_cost(evidence):
    select(evidence[1], recipe="private_mix", backend="native_fork_join", items=65536)["agreement_passed"] = False
    result = build(evidence)
    assert "numerical_agreement_failed" in result["numerical_execution"]["workload_validation"]["fixed_private_array_v2"]["native_fork_join"]["reason_codes"]
    assert result["numerical_execution"]["measurements"] == evidence[1]


@pytest.mark.parametrize("mutation", ["order", "duration", "mismatched_wall", "duplicate", "incomplete", "recipe_id", "footprint", "role", "family"])
def test_malformed_or_interrupted_evidence_cannot_authorize(mutation, evidence):
    row = select(evidence[1], recipe="arithmetic", backend="native_serial", items=65536)
    if mutation == "order":
        row["samples"][0]["global_order"] += 1
    elif mutation == "duration":
        row["samples"][0].update(elapsed_seconds=.199, wall_seconds=.199)
    elif mutation == "mismatched_wall":
        row["samples"][0]["wall_seconds"] *= 2
    elif mutation == "duplicate":
        evidence[1][-1] = deepcopy(row)
    elif mutation == "incomplete":
        evidence[1].pop()
    elif mutation == "recipe_id":
        row["recipe_id"] = "f" * 64
    elif mutation == "footprint":
        row["working_set_bytes"] += 8
    elif mutation == "role":
        row["role"] = "holdout"
    elif mutation == "family":
        row["family"] = "sqrt"
    with pytest.raises(execution.NumericalExecutionError):
        build(evidence)


@pytest.mark.parametrize("mutation", ["id", "count", "domain", "helper", "order", "class", "extra", "lattice"])
def test_saved_registry_requires_complete_canonical_fixed_contract(evidence, monkeypatch, mutation):
    registry = evidence[2]
    recipe = registry[0]
    if mutation == "id":
        recipe["source_sha256"] = "0" * 64
    elif mutation == "count":
        recipe["arithmetic"] += 1
    elif mutation == "domain":
        recipe["domain_ids"] = {"acos": "unproved"}
    elif mutation == "helper":
        recipe["helper_form"] = "separate"
    elif mutation == "order":
        registry[0], registry[1] = registry[1], registry[0]
    elif mutation == "class":
        recipe["workload_class"] = "fixed_private_array_v2"
    elif mutation == "lattice":
        recipe["input_lattices"]["holdout"]["a"] = recipe["input_lattices"]["training"]["a"]
    else:
        recipe["unknown"] = True
    if mutation != "id":
        recipe["recipe_id"] = execution._hash({key: value for key, value in recipe.items() if key != "recipe_id"})
    evidence[1][0]["registry_id"] = execution._hash(registry)
    with pytest.raises(execution.NumericalExecutionError, match="recipe|contract"):
        build(evidence)


def test_saved_status_is_reconstructed_and_current_source_identity_is_separate(evidence, monkeypatch):
    result = build(evidence)
    monkeypatch.setattr(execution, "_current_identities", lambda precision: ("f" * 64, execution._hash(evidence[2])))
    execution.validate_execution_profile(result)
    with pytest.raises(execution.NumericalExecutionError, match="current source"):
        cpu_model(result)
    result["numerical_execution"]["families"]["sqrt"]["native_serial"]["status"] = "rejected"
    with pytest.raises(execution.NumericalExecutionError, match="raw reconstruction"):
        execution.validate_execution_profile(result)


@pytest.mark.parametrize("calibration", [{"application_profiled": True}, [], {"value": float("nan")}])
def test_application_fits_or_malformed_calibration_are_rejected(evidence, calibration):
    with pytest.raises(execution.NumericalExecutionError):
        execution.profile_from_execution_measurements(evidence[0], evidence[1], calibration)


@pytest.mark.parametrize("counts", [[("divide", 1), ("divide", 2)], [None], None, {"divide": True},
                                   {"divide": 10**1000}, [([], 1)]])
def test_invalid_counts_have_typed_unavailable_failure(evidence, counts):
    with pytest.raises(execution.NumericalExecutionError, match="counts"):
        execution.execution_cpu_costs(build(evidence), "native_serial", counts,
            workload_class="scalar_expression_v2", workload_features=scalar_features(), access_class=execution.ACCESS_CLASS)


def test_private_capacity_and_memory_class_are_source_requirements(evidence):
    result = build(evidence)
    features = {"schema_version": 2, "classification_complete": True, "private_array_groups": 2,
                "private_array_elements": 18, "max_private_array_elements": 9, "max_private_array_rank": 2}
    assert cpu_model(result, counts={"sqrt": 5}, workload_class="fixed_private_array_v2", workload_features=features)
    features["private_array_elements"] = 65
    with pytest.raises(execution.NumericalExecutionError, match="applicability"):
        cpu_model(result, counts={"sqrt": 5}, workload_class="fixed_private_array_v2", workload_features=features)
    with pytest.raises(execution.NumericalExecutionError, match="access class"):
        cpu_model(result, access_class="stencil")
    features = scalar_features()
    features["schema_version"] = 1
    with pytest.raises(execution.NumericalExecutionError, match="classification"):
        cpu_model(result, workload_features=features)


def test_memory_interpolation_has_strict_physical_ranges_and_products():
    model = {"kind": "piecewise_bandwidth_v1", "working_set_range": [10, 30], "knots": [
        {"working_set_bytes": 10, "seconds_per_traffic_byte": 1.0},
        {"working_set_bytes": 30, "seconds_per_traffic_byte": 3.0}]}
    assert execution.memory_seconds(model, 5, 10) == 5
    assert execution.memory_seconds(model, 5, 20) == 10
    assert execution.memory_seconds(model, 0, 30) == 0
    for traffic, working_set in [(0, 9), (1, 31), (10**1000, 10), (float("inf"), 10), (True, 10)]:
        with pytest.raises(execution.NumericalExecutionError):
            execution.memory_seconds(model, traffic, working_set)
    model["knots"][0]["working_set_bytes"] = 0
    model["working_set_range"][0] = 0
    with pytest.raises(execution.NumericalExecutionError, match="coordinates"):
        execution.memory_seconds(model, 0, 0)
    model["knots"][0]["working_set_bytes"] = 10
    model["working_set_range"][0] = 10
    model["knots"][0]["seconds_per_traffic_byte"] = 1e308
    with pytest.raises(execution.NumericalExecutionError, match="product"):
        execution.memory_seconds(model, 2, 10)
    model["knots"][0]["seconds_per_traffic_byte"] = 5e-324
    with pytest.raises(execution.NumericalExecutionError, match="product"):
        execution.memory_seconds(model, .1, 10)


def test_finite_but_unrepresentable_fit_and_json_have_typed_failures(evidence):
    row = select(evidence[1], recipe="arithmetic", backend="native_serial", items=65536)
    for sample in row["samples"]:
        sample.update(repetitions=1, elapsed_seconds=1e308, wall_seconds=1e308)
    with pytest.raises(execution.NumericalExecutionError):
        build(evidence)
    with pytest.raises(execution.NumericalExecutionError, match="JSON"):
        execution._hash({"bad": float("nan")})


def test_nonfinite_unknown_raw_metadata_is_not_silently_retained(evidence):
    evidence[1][-1]["unrecognized_metadata"] = float("nan")
    with pytest.raises(execution.NumericalExecutionError, match="JSON"):
        build(evidence)


def test_jsonl_parser_retains_valid_rejections_but_rejects_partial_batches(evidence):
    records = evidence[1]
    assert execution.parse_execution_measurements("\n".join(json.dumps(row) for row in records)) == records
    select(records, recipe="cos", backend="native_serial", items=65536)["samples"].pop()
    with pytest.raises(execution.NumericalExecutionError, match="seven batches"):
        execution.parse_execution_measurements("\n".join(json.dumps(row) for row in records))
    for text in ("", "not json", '{"kind":"unknown"}', '{"kind":[]}'):
        with pytest.raises(execution.NumericalExecutionError):
            execution.parse_execution_measurements(text)
