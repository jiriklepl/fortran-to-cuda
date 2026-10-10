"""Independent cost holdouts, raw evidence and applicability boundaries."""

from copy import deepcopy
from dataclasses import replace

import pytest

from compiler.offload.cpu_dependency_calibration import (
    ACCESS_CLASS,
    DOMAINS,
    PROTOCOL_ID,
    _hash_json,
    cpu_dependency_costs,
    dependency_generator_identity,
    profile_from_dependency_measurements,
    validate_dependency_profile,
)
from compiler.offload.cpu_dependency_model import COEFFICIENT_FAMILIES, Coefficient, evaluate_work_span
from compiler.offload.cpu_dependency_workloads import dependency_recipes
from compiler.offload.cpu_protocol_calibration import CPU_BACKENDS, cpu_protocol_costs
from compiler.offload.numerical_calibration import COMPUTE_FIT_SIZES, COMPUTE_SIZES, NumericalCalibrationError
from compiler.tests.test_cpu_protocol_calibration import corrected, samples


def observations():
    base = corrected()
    protocol = base["cpu_execution_protocol"]
    identity = {"protocol_id": PROTOCOL_ID, "generator_id": dependency_generator_identity(),
        "precision_bits": base["precision_bits"], "cpu_threads": base["cpu_threads"],
        "cpu_affinity": deepcopy(protocol["execution_identity"]["cpu_affinity"]),
        "fortran": deepcopy(protocol["execution_identity"]["fortran"]),
        "host": deepcopy(protocol["execution_identity"]["host"]), "omp_dynamic": False, "omp_proc_bind": "false",
        "base_numerical_identity": _hash_json(base["numerical"]["identity"]),
        "cpu_protocol_identity": _hash_json(protocol), "timed_objects": {"driver": "a" * 64, "native": "b" * 64}}
    for name in ("actual_team_threads", "thread_limit", "omp_wait_policy", "gomp_spincount"):
        identity[name] = protocol["execution_identity"].get(name)
    rows = []
    for backend in CPU_BACKENDS:
        costs = cpu_protocol_costs(base, backend, access_class=ACCESS_CLASS)
        factor = 1.0 if backend == "native_serial" else 0.35
        coefficients = {family: Coefficient(factor * (1e-10 if family == "ordinary" else 3e-10),
                                             factor * (2.8e-10 if family == "ordinary" else 8e-10))
                        for family in COEFFICIENT_FAMILIES}
        for recipe in dependency_recipes(base["precision_bits"]):
            slope = evaluate_work_span(recipe.graph, coefficients, precision_bits=base["precision_bits"]).seconds_per_item
            for n in COMPUTE_SIZES:
                rows.append({"kind": "cpu_dependency_cost_v1", "recipe": recipe.name,
                    "recipe_identity": recipe.identity, "backend": backend, "items": n,
                    "role": "fit" if recipe.role == "basis" and n in COMPUTE_FIT_SIZES else "holdout",
                    "agreement_passed": True, "samples": samples(costs["fixed_seconds"] + n * slope)})
    return base, rows, identity


def test_generated_worker_renderer_changes_invalidate_dependency_evidence(monkeypatch):
    from compiler.offload import cpu_dependency_calibration as calibration

    original = calibration.dependency_generator_identity()
    identities = calibration.worker_renderer_identities()
    changed = {**identities, "emission/cuda/offload.py": "0" * 64}
    monkeypatch.setattr(calibration, "worker_renderer_identities", lambda: changed)
    assert calibration.dependency_generator_identity() != original


def calibrated():
    base, rows, identity = observations()
    return profile_from_dependency_measurements(base, rows, identity)


def source_features():
    return {"schema_version": 2, "classification_complete": True, "private_array_groups": 0,
        "private_array_elements": 0, "referenced_private_array_elements": 0,
        "max_private_array_elements": 0, "max_private_array_rank": 0}


def ordinary_recipe():
    # A different constant-free source graph queries the independently held
    # ordinary class. Guarded holdouts also contain literal unary negations;
    # arbitrary source folding is intentionally outside initial applicability.
    return next(recipe for recipe in dependency_recipes(64) if recipe.role == "basis" and
                recipe.coefficient_family == "divide_constant" and recipe.width == 1)


def test_independent_reader_preserves_v2_and_reconstructs_raw_evidence():
    base, rows, identity = observations()
    before = deepcopy(base)
    result = profile_from_dependency_measurements(base, rows, identity)
    assert base == before
    assert result["numerical"] == before["numerical"]
    assert result["cpu_dependency"]["measurements"] == rows
    assert validate_dependency_profile(result) == result["cpu_dependency"]
    for backend in CPU_BACKENDS:
        assert result["cpu_dependency"]["backends"][backend]["status"] == "accepted"


def test_ordinary_cost_requires_its_own_accepted_class():
    result = calibrated()
    recipe = ordinary_recipe()
    costs = cpu_dependency_costs(result, recipe.graph, "native_serial", workload_class="scalar_expression_v2",
        workload_features=source_features(), primitive_domains=DOMAINS, access_class=ACCESS_CLASS)
    assert costs["seconds_per_item"] > 0
    assert costs["fixed_seconds"] == 0
    assert costs["item_range"] == [65536, 1048576]


def test_failed_separate_helper_holdout_is_not_trimmed_or_fit():
    base, rows, identity = observations()
    recipe = next(recipe for recipe in dependency_recipes(64) if recipe.role == "structural_holdout" and
                  recipe.workload_class == "ordinary_expression_v2" and recipe.helper_form == "separate")
    for row in rows:
        if row["recipe"] == recipe.name and row["backend"] == "native_serial":
            for sample in row["samples"]:
                sample["elapsed_seconds"] *= 3
                sample["wall_seconds"] *= 3
    result = profile_from_dependency_measurements(base, rows, identity)
    evidence = result["cpu_dependency"]["backends"]["native_serial"]
    assert evidence["families"]["ordinary"]["status"] == "accepted"
    assert evidence["workload_validation"]["ordinary_expression_v2"]["status"] == "rejected"
    with pytest.raises(NumericalCalibrationError, match="expression class rejected"):
        cpu_dependency_costs(result, ordinary_recipe().graph, "native_serial", workload_class="scalar_expression_v2",
            workload_features=source_features(), primitive_domains=DOMAINS, access_class=ACCESS_CLASS)


@pytest.mark.parametrize("mutation", ["source", "backend", "placement", "role", "short_batch", "duplicate"])
def test_incompatible_raw_observations_cannot_authorize_estimates(mutation):
    base, rows, identity = observations()
    if mutation == "source":
        rows[0]["recipe_identity"] = "f" * 64
    elif mutation == "backend":
        identity["fortran"]["semantic_options"] = "-Ofast"
    elif mutation == "placement":
        identity["cpu_affinity"] = [0, 1, 2, 3]
    elif mutation == "role":
        rows[0]["role"] = "holdout"
    elif mutation == "short_batch":
        rows[0]["samples"][0]["wall_seconds"] = 0.1
    else:
        rows[1] = deepcopy(rows[0])
    with pytest.raises(NumericalCalibrationError):
        profile_from_dependency_measurements(base, rows, identity)


def test_saved_acceptance_cannot_override_failed_or_changed_observations():
    result = calibrated()
    result["cpu_dependency"]["backends"]["native_serial"]["families"]["ordinary"]["coefficient"]["work_seconds"] *= 10
    with pytest.raises(NumericalCalibrationError, match="differs"):
        validate_dependency_profile(result)


@pytest.mark.parametrize("access_class", ["stencil", "read_modify_write_v1", "unknown"])
def test_three_array_memory_observations_do_not_price_other_shapes(access_class):
    with pytest.raises(NumericalCalibrationError, match="access class"):
        cpu_dependency_costs(calibrated(), ordinary_recipe().graph, "native_serial", workload_class="scalar_expression_v2",
            workload_features=source_features(), primitive_domains=DOMAINS, access_class=access_class)


def test_private_storage_and_missing_domain_proofs_remain_unavailable():
    result = calibrated()
    recipe = ordinary_recipe()
    bad = {**source_features(), "private_array_elements": 1}
    with pytest.raises(NumericalCalibrationError, match="storage/workload"):
        cpu_dependency_costs(result, recipe.graph, "native_serial", workload_class="scalar_expression_v2",
            workload_features=bad, primitive_domains=DOMAINS, access_class=ACCESS_CLASS)
    with pytest.raises(NumericalCalibrationError, match="domain unavailable"):
        cpu_dependency_costs(result, recipe.graph, "native_serial", workload_class="scalar_expression_v2",
            workload_features=source_features(), primitive_domains={}, access_class=ACCESS_CLASS)


@pytest.mark.parametrize("flag", ["constant", "invariant"])
def test_uniform_or_constant_numerics_cannot_inflate_original_native_cost(flag):
    graph = ordinary_recipe().graph
    graph = replace(graph, operations=(replace(graph.operations[0], **{flag: True}), *graph.operations[1:]))
    with pytest.raises(NumericalCalibrationError, match="folding or hoisting"):
        cpu_dependency_costs(calibrated(), graph, "native_serial", workload_class="scalar_expression_v2",
            workload_features=source_features(), primitive_domains=DOMAINS, access_class=ACCESS_CLASS)


def test_memory_limited_basis_cannot_be_fitted_as_compute():
    base, rows, identity = observations()
    names = {recipe.name for recipe in dependency_recipes(64) if recipe.role == "basis" and
             recipe.coefficient_family == "ordinary"}
    for row in rows:
        if row["backend"] == "native_serial" and row["recipe"] in names:
            row["samples"] = samples(row["items"] * 24e-11)
    result = profile_from_dependency_measurements(base, rows, identity)
    evidence = result["cpu_dependency"]["backends"]["native_serial"]["families"]["ordinary"]
    assert evidence["status"] == "rejected"
    assert "memory-error envelope" in evidence["reason"]


def test_dynamic_division_needs_source_domain_and_independent_domain_holdouts():
    recipe = next(recipe for recipe in dependency_recipes(64) if recipe.role == "basis" and
                  recipe.coefficient_family == "divide_dynamic" and recipe.width == 1)
    with pytest.raises(NumericalCalibrationError, match="dynamic division source domain"):
        cpu_dependency_costs(calibrated(), recipe.graph, "native_serial", workload_class="scalar_expression_v2",
            workload_features=source_features(), primitive_domains=DOMAINS, access_class=ACCESS_CLASS)
