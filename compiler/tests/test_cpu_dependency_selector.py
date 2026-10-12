"""Validated CPU projections retain the public selector's applicability gates."""

from copy import deepcopy
from dataclasses import replace

import pytest

from compiler.offload import cpu_dependency_calibration as calibration
from compiler.offload.cpu_dependency_workloads import dependency_recipes
from compiler.offload.cpu_protocol_calibration import CPU_BACKENDS
from compiler.offload.numerical_calibration import NumericalCalibrationError
from compiler.tests.test_cpu_dependency_calibration import calibrated, ordinary_recipe, source_features
from compiler.tests.test_cpu_dependency_current import current_observations  # noqa: F401
from compiler.tests.test_numerical_execution import scale


def query_options(**changes):
    return {"workload_class": "scalar_expression_v2", "workload_features": source_features(),
            "primitive_domains": calibration.DOMAINS, "access_class": calibration.ACCESS_CLASS, **changes}


def projection(profile):
    section = profile["cpu_dependency"]
    return {"protocol_id": section["protocol_id"], "generator_id": section["generator_id"],
            "backends": deepcopy(section["backends"])}


def test_selector_matches_fresh_public_queries_for_both_protocols(current_observations):  # noqa: F811 - imported pytest fixture
    base, rows, identity = current_observations
    current = calibration.profile_from_dependency_measurements(base, rows, identity)
    graph = dependency_recipes(64)[0].graph
    for profile, forms in ((calibrated(), (None,)), (current, calibration.COMPILED_FORMS)):
        section = projection(profile)
        before = deepcopy(section)
        for backend in CPU_BACKENDS:
            for form in forms:
                options = query_options(compiled_form=form)
                expected = calibration.cpu_dependency_costs(profile, graph, backend, **options)
                actual = calibration._cpu_costs_from_section(
                    {"precision_bits": profile["precision_bits"]}, section, graph, backend, **options)
                assert actual == expected
                actual["memory_cost_model"]["knots"][0]["seconds_per_traffic_byte"] = 99
                assert section == before


def test_selector_and_public_query_reject_the_same_source_applicability(current_observations):  # noqa: F811 - imported pytest fixture
    base, rows, identity = current_observations
    profile = calibration.profile_from_dependency_measurements(base, rows, identity)
    graph = ordinary_recipe().graph
    options = query_options(compiled_form="direct_loop")
    cases = [
        (graph, "gpu", options),
        (graph, "native_serial", {**options, "access_class": "stencil"}),
        (graph, "native_serial", {**options, "compiled_form": None}),
        (graph, "native_serial", {**options, "compiled_form": "inferred"}),
        (graph, "native_serial", {**options, "primitive_domains": {}}),
        (graph, "native_serial", {**options, "workload_features": {}}),
        (graph, "native_serial", {**options, "workload_features": {
            **source_features(), "private_array_elements": 1}}),
        (replace(graph, operations=(replace(graph.operations[0], invariant=True), *graph.operations[1:])),
         "native_serial", options),
    ]
    section = projection(profile)
    for selected_graph, backend, selected_options in cases:
        with pytest.raises(NumericalCalibrationError) as public:
            calibration.cpu_dependency_costs(profile, selected_graph, backend, **selected_options)
        with pytest.raises(NumericalCalibrationError) as private:
            calibration._cpu_costs_from_section({"precision_bits": 64}, section,
                selected_graph, backend, **selected_options)
        assert str(private.value) == str(public.value)


def test_independently_rejected_form_stays_rejected_in_projection(current_observations):  # noqa: F811 - imported pytest fixture
    base, rows, identity = current_observations
    for row in rows:
        if row["recipe"] == "original_private_mix" and row["backend"] == "native_serial":
            scale(row, 3)
    profile = calibration.profile_from_dependency_measurements(base, rows, identity)
    options = query_options(workload_class="fixed_private_array_v2", compiled_form="value_function_same_unit",
        workload_features={"schema_version": 2, "classification_complete": True,
            "private_array_groups": 4, "private_array_elements": 64, "referenced_private_array_elements": 64,
            "max_private_array_elements": 16, "max_private_array_rank": 2})
    form = profile["cpu_dependency"]["backends"]["native_serial"]["compiled_form_validation"][
        "value_function_same_unit"]
    assert form["workload_validation"]["fixed_private_array_v2"]["status"] == "rejected"
    with pytest.raises(NumericalCalibrationError, match="expression class rejected"):
        calibration.cpu_dependency_costs(profile, ordinary_recipe().graph, "native_serial", **options)
    with pytest.raises(NumericalCalibrationError, match="expression class rejected"):
        calibration._cpu_costs_from_section({"precision_bits": 64}, projection(profile),
            ordinary_recipe().graph, "native_serial", **options)


def test_public_query_still_revalidates_every_demand(monkeypatch):
    profile = calibrated()
    original_contract = calibration.require_numerical_profile_contract
    original_validate = calibration.validate_dependency_profile
    checks = []

    def contract(value):
        checks.append("contract")
        return original_contract(value)

    def validate(value):
        checks.append("raw")
        return original_validate(value)

    monkeypatch.setattr(calibration, "require_numerical_profile_contract", contract)
    monkeypatch.setattr(calibration, "validate_dependency_profile", validate)
    calibration.cpu_dependency_costs(profile, ordinary_recipe().graph, "native_serial", **query_options())
    profile["cpu_dependency"]["measurements"][0]["samples"][0]["wall_seconds"] = 0.1
    with pytest.raises(NumericalCalibrationError):
        calibration.cpu_dependency_costs(profile, ordinary_recipe().graph, "native_serial", **query_options())
    assert checks == ["contract", "raw", "contract", "raw"]


def test_public_invalid_access_keeps_its_original_validation_order(monkeypatch):
    profile = calibrated()

    def unexpected_raw_validation(_):
        pytest.fail("invalid access must decline before raw reconstruction")

    monkeypatch.setattr(calibration, "validate_dependency_profile", unexpected_raw_validation)
    with pytest.raises(NumericalCalibrationError, match="execution/access class unavailable"):
        calibration.cpu_dependency_costs(profile, ordinary_recipe().graph, "native_serial",
            **query_options(access_class="stencil"))
