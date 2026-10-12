"""Current startup/memory foundation and independent work/span controls."""

import json
import math
from copy import deepcopy

import pytest

from compiler.offload import numerical_execution as execution
from compiler.offload.analysis import _workload_features
from compiler.offload.cpu_dependency_calibration import (
    ACCESS_CLASS,
    COMPILED_FORMS,
    CURRENT_PROTOCOL_ID,
    DOMAINS,
    _foundation,
    _hash_json,
    _reconcile_raw,
    _validate_preflight,
    calibrate_dependency,
    cpu_dependency_costs,
    dependency_generator_identity,
    profile_from_dependency_measurements,
    validate_dependency_profile,
)
from compiler.offload.cpu_dependency_model import COEFFICIENT_FAMILIES, Coefficient, evaluate_work_span
from compiler.offload.cpu_dependency_workloads import (
    BASIS_FAMILIES,
    current_dependency_recipes,
    dependency_recipes,
    registry_source,
)
from compiler.offload.numerical_calibration import COMPUTE_FIT_SIZES, COMPUTE_SIZES, NumericalCalibrationError
from compiler.offload.numerical_execution_workloads import execution_recipes
from compiler.tests.test_cpu_dependency_producer import FakeDependencyRun
from compiler.tests.test_numerical_execution import GENERATOR, observations, scale


@pytest.fixture
def current_observations(monkeypatch):
    base, records, registry = observations()
    monkeypatch.setattr(execution, "_registry", lambda precision: deepcopy(registry))
    monkeypatch.setattr(execution, "_current_identities", lambda precision: (GENERATOR, execution._hash(registry)))
    # The new foundation must remain useful despite rejected old compute holds.
    for row in records:
        if row.get("kind") == "numerical_execution_cost_v1" and row.get("role") == "holdout" and row["recipe"] != "memory":
            scale(row, 3)
    profile = execution.profile_from_execution_measurements(base, records)
    foundation = _foundation(profile, current=True)
    source_identity = foundation["execution_identity"]
    identity = {key: deepcopy(source_identity[key]) for key in (
        "precision_bits", "cpu_threads", "cpu_affinity", "fortran", "host", "actual_team_threads",
        "thread_limit", "omp_dynamic", "omp_proc_bind", "omp_wait_policy", "gomp_spincount", "cpu_name")}
    identity.update(protocol_id=CURRENT_PROTOCOL_ID, generator_id=dependency_generator_identity(),
        foundation_kind=foundation["kind"], foundation_identity=foundation["identity"],
        base_execution_identity=_hash_json(source_identity), timed_objects={"test.o": "a" * 64})
    recipes = current_dependency_recipes(64)
    backend_costs = {backend: execution.execution_startup_memory_costs(profile, backend, access_class=ACCESS_CLASS)
                     for backend in execution.CPU_BACKENDS}
    rows = []
    for ordinal, recipe in enumerate(recipes):
        for size_index, items in enumerate(COMPUTE_SIZES):
            cell = ordinal * len(COMPUTE_SIZES) + size_index
            for backend_index, backend in enumerate(execution.CPU_BACKENDS):
                costs = backend_costs[backend]
                coefficients = {family: Coefficient(1e-10 if family == "ordinary" else 3e-10,
                                                    2.8e-10 if family == "ordinary" else 8e-10)
                                for family in COEFFICIENT_FAMILIES}
                compute = evaluate_work_span(recipe.graph, coefficients, precision_bits=64).seconds_per_item
                seconds = costs["fixed_seconds"] + max(items * compute,
                    execution.memory_seconds(costs["memory_cost_model"], items * 24, items * 24))
                repetitions = math.ceil(0.25 / seconds)
                samples = [{"batch": batch, "repetitions": repetitions,
                    "elapsed_seconds": repetitions * seconds, "wall_seconds": repetitions * seconds,
                    "global_order": batch * len(recipes) * len(COMPUTE_SIZES) * 3 + cell * 3 + (backend_index - batch) % 3}
                    for batch in range(7)]
                rows.append({"kind": "cpu_dependency_cost_v1", "recipe": recipe.name,
                    "recipe_identity": recipe.identity, "backend": backend, "items": items,
                    "role": "fit" if recipe.role == "basis" and items in COMPUTE_FIT_SIZES else "holdout",
                    "agreement_passed": True, "samples": samples})
    return profile, rows, identity


@pytest.mark.parametrize("precision", [32, 64])
def test_exact_original_workers_and_input_lattices_are_independent_holdouts(precision):
    recipes = current_dependency_recipes(precision)
    assert len(recipes) == 49
    assert recipes[:26] == dependency_recipes(precision)
    assert [(recipe.coefficient_family, recipe.width, recipe.role) for recipe in recipes[26:32]] == [
        (family, 2, "width_holdout") for family in BASIS_FAMILIES]
    assert all(recipe.compiled_form == ("value_function_separate" if recipe.helper_form == "separate" else "direct_loop")
               for recipe in recipes[:32])
    for held, original in zip(recipes[32:], execution_recipes(precision), strict=True):
        assert held.role == "original_holdout"
        assert held.original_recipe_identity == original.metadata["recipe_id"]
        assert held.fortran_sources == original.native_sources
        assert held.cpp_sources == original.cpp_sources
        assert held.lattices("training") == original.lattices("training")
        assert held.lattices("holdout") == original.lattices("holdout")
        period = original.reference_period("holdout")
        assert held.reference(held.inputs(17)) == tuple(period[index % len(period)] for index in range(17))
    private = next(recipe for recipe in recipes if recipe.name == "original_private_mix")
    features = _workload_features(private.region.private_symbols)
    assert (features.private_array_groups, features.private_array_elements) == (4, 64)
    header = registry_source(recipes, "a" * 64)
    assert "execution_helper_private_mix_serial" in header
    assert '"width_holdout"' in header


def test_current_foundation_preserves_rejected_compute_and_requires_current_identity(current_observations, monkeypatch):
    profile, _, _ = current_observations
    before = deepcopy(profile)
    assert profile["numerical_execution"]["workload_validation"]["ordinary_expression_v2"]["native_serial"]["status"] == "rejected"
    costs = execution.execution_startup_memory_costs(profile, "native_serial", access_class=ACCESS_CLASS)
    costs["memory_cost_model"]["knots"][0]["seconds_per_traffic_byte"] = 99
    assert profile == before
    monkeypatch.setattr(execution, "_current_identities", lambda precision: ("f" * 64, "e" * 64))
    with pytest.raises(NumericalCalibrationError, match="current source"):
        execution.execution_startup_memory_costs(profile, "native_serial", access_class=ACCESS_CLASS)


def test_current_reader_requires_explicit_independently_accepted_form(current_observations):
    profile, rows, identity = current_observations
    before = deepcopy(profile)
    result = profile_from_dependency_measurements(profile, rows, identity)
    assert profile == before
    assert result["numerical_execution"] == before["numerical_execution"]
    assert validate_dependency_profile(result) == result["cpu_dependency"]
    assert result["cpu_dependency"]["protocol"]["automatic_source_selection"] is False
    graph = dependency_recipes(64)[0].graph
    features = {"schema_version": 2, "classification_complete": True,
        "private_array_groups": 0, "private_array_elements": 0, "referenced_private_array_elements": 0,
        "max_private_array_elements": 0, "max_private_array_rank": 0}
    kwargs = {"workload_class": "scalar_expression_v2", "workload_features": features,
              "primitive_domains": DOMAINS, "access_class": ACCESS_CLASS}
    with pytest.raises(NumericalCalibrationError, match="proved compiled form"):
        cpu_dependency_costs(result, graph, "native_serial", **kwargs)
    for form in COMPILED_FORMS:
        costs = cpu_dependency_costs(result, graph, "native_serial", compiled_form=form, **kwargs)
        assert costs["compiled_form"] == form
        assert costs["seconds_per_item"] > 0


def test_private_storage_limits_belong_to_the_independently_validated_form(current_observations):
    profile, rows, identity = current_observations
    result = profile_from_dependency_measurements(profile, rows, identity)
    forms = result["cpu_dependency"]["backends"]["native_serial"]["compiled_form_validation"]
    assert forms["direct_loop"]["private_limits"]["private_array_elements"] == 32
    assert forms["value_function_same_unit"]["private_limits"]["private_array_elements"] == 64
    assert forms["value_function_separate"]["private_limits"]["private_array_elements"] == 64
    graph = dependency_recipes(64)[0].graph
    features = {"schema_version": 2, "classification_complete": True,
        "private_array_groups": 4, "private_array_elements": 64, "referenced_private_array_elements": 64,
        "max_private_array_elements": 16, "max_private_array_rank": 2}
    kwargs = {"workload_class": "fixed_private_array_v2", "workload_features": features,
              "primitive_domains": DOMAINS, "access_class": ACCESS_CLASS}
    with pytest.raises(NumericalCalibrationError, match="compiled-form applicability"):
        cpu_dependency_costs(result, graph, "native_serial", compiled_form="direct_loop", **kwargs)
    for form in COMPILED_FORMS[1:]:
        assert cpu_dependency_costs(result, graph, "native_serial", compiled_form=form, **kwargs)["compiled_form"] == form


@pytest.mark.parametrize("recipe", ["dep_ordinary_w2_inline_p64", "dep_ordinary_guarded_w1_separate_p64",
                                    "original_private_mix", "original_domain_cos"])
def test_failed_independent_holdout_cannot_change_basis_or_poison_another_form(current_observations, recipe):
    profile, rows, identity = current_observations
    original = profile_from_dependency_measurements(profile, rows, identity)["cpu_dependency"]
    changed = deepcopy(rows)
    for row in changed:
        if row["recipe"] == recipe and row["backend"] == "native_serial":
            scale(row, 3)
    result = profile_from_dependency_measurements(profile, changed, identity)["cpu_dependency"]["backends"]["native_serial"]
    if "w2" in recipe:
        assert result["families"]["ordinary"]["status"] == "rejected"
    else:
        expected = original["backends"]["native_serial"]["families"]
        assert result["families"] == expected
        assert result["compiled_form_validation"]["direct_loop"] == original["backends"]["native_serial"]["compiled_form_validation"]["direct_loop"]
        key = "fixed_private_array_v2" if "private" in recipe else "ordinary_expression_v2" if "ordinary" in recipe else "scalar_expression_v2"
        form = "value_function_separate" if "separate" in recipe else "value_function_same_unit"
        assert result["compiled_form_validation"][form]["workload_validation"][key]["status"] == "rejected"


@pytest.mark.parametrize("mutation", ["order", "duration", "missing", "fit_role", "source", "foundation"])
def test_current_raw_authority_and_fixed_roles_cannot_be_rebound(current_observations, mutation):
    profile, rows, identity = deepcopy(current_observations)
    if mutation == "order":
        rows[0]["samples"][0]["global_order"] += 1
    elif mutation == "duration":
        rows[0]["samples"][0]["elapsed_seconds"] *= 0.5
    elif mutation == "missing":
        rows.pop()
    elif mutation == "fit_role":
        next(row for row in rows if row["recipe"] == "original_arithmetic")["role"] = "fit"
    elif mutation == "source":
        rows[0]["recipe_identity"] = "f" * 64
    else:
        identity["foundation_identity"] = "f" * 64
    with pytest.raises(NumericalCalibrationError):
        profile_from_dependency_measurements(profile, rows, identity)


def test_reconcile_exact_global_raw_rounds_and_reject_changed_sample(current_observations, tmp_path):
    _, rows, _ = current_observations
    raw = [{**{key: value for key, value in row.items() if key not in {"kind", "samples"}},
            **sample, "kind": "cpu_dependency_batch_v1"} for row in rows for sample in row["samples"]]
    raw.sort(key=lambda row: row["global_order"])
    path = tmp_path / "raw.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in raw) + "\n")
    _reconcile_raw(path, rows, current_dependency_recipes(64))
    raw[0]["repetitions"] += 1
    path.write_text("\n".join(json.dumps(row) for row in raw) + "\n")
    with pytest.raises(NumericalCalibrationError, match="reconciliation"):
        _reconcile_raw(path, rows, current_dependency_recipes(64))


def test_preflight_must_cover_every_exact_control_before_sampling(current_observations):
    _, _, identity = current_observations
    raw_identity = {key: value for key, value in identity.items() if key not in {
        "protocol_id", "generator_id", "foundation_kind", "foundation_identity", "base_execution_identity", "timed_objects", "cpu_name"}}
    raw_identity["registry_generator_id"] = identity["generator_id"]
    raw_identity["fortran"] = {key: identity["fortran"][key] for key in ("compiler_version", "compiler_options")}
    raw_identity.pop("host")
    controls = [{"kind": "cpu_dependency_smoke_v1", "recipe": recipe.name,
        "recipe_identity": recipe.identity, "backend": backend, "items": items, "agreement_passed": True}
        for recipe in current_dependency_recipes(64) for items in (0, 1, 8, 17) for backend in execution.CPU_BACKENDS]
    records = [{"kind": "cpu_dependency_identity_v1", **raw_identity}, *controls]
    identity["registry_generator_id"] = identity["generator_id"]
    _validate_preflight("\n".join(json.dumps(row) for row in records), identity, current_dependency_recipes(64))
    controls[0]["agreement_passed"] = False
    with pytest.raises(NumericalCalibrationError, match="preflight"):
        _validate_preflight("\n".join(json.dumps(row) for row in records), identity, current_dependency_recipes(64))


@pytest.mark.parametrize("fail_preflight", [False, True])
def test_current_producer_proves_preflight_before_one_sampling_attempt(current_observations, monkeypatch, tmp_path,
                                                                      fail_preflight):
    from types import SimpleNamespace

    profile, rows, _ = current_observations
    source = profile["numerical_execution"]["identity"]
    fake_profile = {**profile, "cpu_execution_protocol": {"execution_identity": source}}
    fake = FakeDependencyRun(fake_profile, rows)
    monkeypatch.setattr("compiler.offload.cpu_dependency_calibration.os.sched_getaffinity", lambda _: set(source["cpu_affinity"]))
    monkeypatch.setattr("compiler.offload.cpu_dependency_calibration.cpu_identity", lambda: profile["hardware"]["cpu_name"])
    monkeypatch.setattr("compiler.offload.cpu_dependency_calibration._tool", lambda chosen, defaults: chosen or defaults[0])

    def run(command, cwd, log, *, timeout, env=None):
        if "--smoke" in command:
            records = [fake.identity()]
            records += [{"kind": "cpu_dependency_smoke_v1", "recipe": recipe.name,
                "recipe_identity": recipe.identity, "backend": backend, "items": items,
                "agreement_passed": not fail_preflight}
                for recipe in current_dependency_recipes(64) for items in (0, 1, 8, 17)
                for backend in execution.CPU_BACKENDS]
            text = "\n".join(json.dumps(row) for row in records) + "\n"
            log.write_text(text)
            return text
        text = fake(command, cwd, log, timeout=timeout, env=env)
        if command[0] == "taskset" and "--identity" not in command:
            path = cwd / "cpu-dependency-raw-samples.jsonl"
            raw = [json.loads(line) for line in path.read_text().splitlines()]
            raw.sort(key=lambda row: row["global_order"])
            path.write_text("\n".join(json.dumps(row) for row in raw) + "\n")
        return text

    args = SimpleNamespace(fortran="gfortran", host_cxx="g++", build_dir=tmp_path / "attempt",
                           fortran_flag=["-O3", "-fopenmp", "-ffp-contract=off"])
    if fail_preflight:
        with pytest.raises(NumericalCalibrationError, match="preflight"):
            calibrate_dependency(profile, args, run=run)
        assert fake.measurements == 0
        assert not (args.build_dir / "measurements.jsonl").exists()
    else:
        result = calibrate_dependency(profile, args, run=run)
        assert fake.measurements == 1
        assert result["cpu_dependency"]["protocol_id"] == CURRENT_PROTOCOL_ID
        assert result["numerical_execution"] == profile["numerical_execution"]
        assert len(result["cpu_dependency"]["measurements"]) == 49 * 5 * 3
