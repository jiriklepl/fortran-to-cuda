"""One fresh GPU raw demand can issue detached private query projections."""

from copy import deepcopy

import pytest

from compiler.offload import numerical_gpu_execution as gpu
from compiler.offload import numerical_gpu_execution_costs as costs
from compiler.offload.numerical_execution import ACCESS_CLASS
from compiler.tests.test_numerical_gpu_execution_costs import evidence, launch_observations


def fixture(monkeypatch, precision=64):
    parent = evidence(precision)
    rows, calibration = launch_observations(parent)
    profile = costs.profile_from_gpu_launch_measurements(parent, rows, calibration)
    identity = profile[gpu.SECTION]["identity"]
    monkeypatch.setattr(gpu, "current_gpu_identities", lambda _: (
        identity["source_generator_id"], identity["registry_id"], identity["generator_id"],
        identity["renderer_sources"]))
    return profile


def arguments(**changes):
    return {"workload_class": "scalar_expression_v2", "access_class": ACCESS_CLASS,
            "workload_features": {"schema_version": 2, "classification_complete": True,
                                  "private_array_elements": 0}, **changes}


@pytest.mark.parametrize("precision", [32, 64])
def test_private_selectors_match_full_public_queries_with_one_raw_demand(monkeypatch, precision):
    profile = fixture(monkeypatch, precision)
    original = deepcopy(profile)
    calls = {"parent": 0, "costs": 0}
    original_parent = gpu._build
    original_costs = costs._build_from_parent

    def parent_build(*args, **kwargs):
        calls["parent"] += 1
        return original_parent(*args, **kwargs)

    def cost_build(*args, **kwargs):
        calls["costs"] += 1
        return original_costs(*args, **kwargs)

    monkeypatch.setattr(gpu, "_build", parent_build)
    monkeypatch.setattr(costs, "_build_from_parent", cost_build)
    parent, section = costs._current_gpu_cost_sections(profile)
    assert calls == {"parent": 1, "costs": 1}
    requests = [({}, arguments()), ({"sqrt": 2}, arguments()),
                ({"divide": 1}, arguments(workload_class="ordinary_expression_v2")),
                ({"cos": 1}, arguments(workload_class="fixed_private_array_v2",
                    workload_features={"schema_version": 2, "classification_complete": True,
                        "private_array_groups": 4, "private_array_elements": 64,
                        "max_private_array_elements": 16, "max_private_array_rank": 2}))]
    results = [costs._gpu_costs_from_sections(parent, section, counts, **kwargs)
               for counts, kwargs in requests]
    assert calls == {"parent": 1, "costs": 1}
    for (counts, kwargs), result in zip(requests, results, strict=True):
        assert result == costs.gpu_execution_costs(profile, counts, **kwargs)
    assert calls == {"parent": 5, "costs": 5}
    assert profile == original

    # Neither the detached sections nor a returned cost payload alias raw
    # profile metadata or later results from the same private projection.
    first = results[0]
    first["memory_cost_model"]["knots"][0]["total_seconds"] *= 2
    first["cpu_affinity"].append(99)
    assert costs._gpu_costs_from_sections(parent, section, {}, **arguments()) == costs.gpu_execution_costs(
        profile, {}, **arguments())
    parent["measurements"][-1]["global_order"] = -1
    section["measurements"][-1]["global_order"] = -1
    assert profile == original


@pytest.mark.parametrize(("counts", "changes"), [
    ({"sqrt": True}, {}),
    ({}, {"access_class": "strided"}),
    ({}, {"workload_class": "unknown"}),
    ({}, {"workload_features": {"schema_version": 2, "classification_complete": True,
                                "private_array_elements": 1}}),
    ({}, {"workload_class": "fixed_private_array_v2", "workload_features": {
        "schema_version": 2, "classification_complete": True, "private_array_groups": 4,
        "private_array_elements": 65, "max_private_array_elements": 16, "max_private_array_rank": 2}}),
])
def test_private_selector_preserves_public_query_rejections(monkeypatch, counts, changes):
    profile = fixture(monkeypatch)
    parent, section = costs._current_gpu_cost_sections(profile)
    kwargs = arguments(**changes)
    with pytest.raises(gpu.NumericalGpuExecutionError) as public:
        costs.gpu_execution_costs(profile, counts, **kwargs)
    with pytest.raises(gpu.NumericalGpuExecutionError) as private:
        costs._gpu_costs_from_sections(parent, section, counts, **kwargs)
    assert str(public.value) == str(private.value)


@pytest.mark.parametrize("mutation", ["parent_raw", "launch_raw", "derived", "renderer", "cost_source"])
def test_each_factory_demand_rejects_later_raw_or_identity_changes(monkeypatch, mutation):
    profile = fixture(monkeypatch)
    costs._current_gpu_cost_sections(profile)
    if mutation == "parent_raw":
        profile[gpu.SECTION]["measurements"][-1]["global_order"] = -1
    elif mutation == "launch_raw":
        profile[costs.SECTION]["measurements"][-1]["launches"] = 1
    elif mutation == "derived":
        profile[costs.SECTION]["startup"]["fixed_seconds"] *= 2
    elif mutation == "renderer":
        monkeypatch.setattr(gpu, "current_gpu_identities", lambda _: ("0" * 64, "0" * 64, "0" * 64, {}))
    else:
        monkeypatch.setattr(costs, "cost_sources", lambda: {"changed": "0" * 64})
    with pytest.raises(gpu.NumericalGpuExecutionError):
        costs._current_gpu_cost_sections(profile)
    with pytest.raises(gpu.NumericalGpuExecutionError):
        costs.gpu_execution_costs(profile, {}, **arguments())


def test_public_preflight_reauthenticates_parent_after_a_private_demand(monkeypatch):
    profile = fixture(monkeypatch)
    costs._current_gpu_cost_sections(profile)
    section = profile[costs.SECTION]
    costs.validate_launch_preflight(profile, section["identity"], section["measurements"][1:5])
    profile[gpu.SECTION]["measurements"][-1]["global_order"] = -1
    with pytest.raises(gpu.NumericalGpuExecutionError):
        costs.validate_launch_preflight(profile, section["identity"], section["measurements"][1:5])
