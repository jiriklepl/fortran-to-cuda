"""Versioned total launch intervals do not fit a negative memory residual."""

import json
from copy import deepcopy

import pytest

from compiler.offload import numerical_gpu_execution as gpu
from compiler.offload import numerical_gpu_execution_costs as costs
from compiler.offload.numerical_execution import _hash
from compiler.tests.test_numerical_gpu_execution_costs import (
    FIXED,
    MEMORY,
    evidence,
    launch_observations,
    query,
)


def memory_parent(precision):
    parent = evidence(precision)
    section = parent[gpu.SECTION]
    raw = deepcopy(section["measurements"])
    for row in raw[1 + gpu.PREFLIGHT_RECORDS:]:
        if row["recipe"] == "memory":
            # An independently measured total interval may be below the tiny
            # control without identifying a negative physical bandwidth.
            row["resident_event_seconds"] = (2e-6 + row["traffic_bytes"] * MEMORY) * row["launches"]
    base = {key: value for key, value in parent.items() if key != gpu.SECTION}
    return gpu.profile_from_gpu_execution_measurements(base, raw, section["calibration"])


def legacy_profile(parent, rows, calibration):
    archived = deepcopy(rows)
    archived[0]["launch_protocol_id"] = costs.LEGACY_PROTOCOL_ID
    result = deepcopy(parent)
    result[costs.SECTION] = costs._build(parent, archived, calibration)
    return result


@pytest.mark.parametrize("precision", [32, 64])
def test_total_curve_preserves_raw_independent_holds_when_memory_is_below_fixed(precision, monkeypatch):
    parent = memory_parent(precision)
    original = deepcopy(parent)
    rows, calibration = launch_observations(parent)
    profile = costs.profile_from_gpu_launch_measurements(parent, rows, calibration)
    section = profile[costs.SECTION]
    memory = section["families"]["memory"]["gpu"]
    assert memory["status"] == "accepted"
    assert len(memory["holdouts"]) == 9
    assert memory["model"]["knots"][0]["total_seconds"] < FIXED
    assert section["startup"]["fixed_seconds"] == FIXED
    assert all(row["status"] == "accepted" for row in section["workload_validation"].values())
    assert all(row["status"] == "accepted" for row in section["domain_validation"].values())
    assert section["measurements"] == rows
    assert profile[gpu.SECTION] == parent[gpu.SECTION]
    assert parent == original
    assert query(profile, monkeypatch)["memory_combination"] == "max_fixed_compute_total_memory_v1"

    old = legacy_profile(parent, rows, calibration)
    assert old[costs.SECTION]["protocol_id"] == costs.LEGACY_PROTOCOL_ID
    assert "memory_combination" not in old[costs.SECTION]
    assert old[costs.SECTION]["families"]["memory"]["gpu"]["reason_codes"] == [
        "measured_startup_exceeds_training_time"]
    assert old[costs.SECTION]["cost_model_available"] is False
    costs.validate_gpu_execution_costs(old)
    with pytest.raises(gpu.NumericalGpuExecutionError, match="current total-memory"):
        query(old, monkeypatch)


def test_legacy_profiles_remain_readable_but_cannot_issue_current_queries(tmp_path, monkeypatch):
    from compiler.offload.profile import ProfileError, load_profile

    parent = evidence()
    rows, calibration = launch_observations(parent)
    old = legacy_profile(parent, rows, calibration)
    assert old[costs.SECTION]["families"]["memory"]["gpu"]["status"] == "accepted"
    assert old[costs.SECTION]["families"]["memory"]["gpu"]["model"]["kind"] == "piecewise_bandwidth_v1"
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(old))
    with monkeypatch.context() as patch:
        patch.setattr(gpu, "current_gpu_identities", lambda _: pytest.fail("historical current identity request"))
        patch.setattr(costs, "cost_sources", lambda: pytest.fail("historical cost-source rebinding"))
        assert load_profile(path) == old
    with pytest.raises(gpu.NumericalGpuExecutionError, match="current total-memory"):
        query(old, monkeypatch)
    with pytest.raises(gpu.NumericalGpuExecutionError, match="current total-memory"):
        costs.profile_from_gpu_launch_measurements(parent, old[costs.SECTION]["measurements"], calibration)
    old[costs.SECTION]["families"]["memory"]["gpu"]["model"]["knots"][0]["seconds_per_traffic_byte"] *= 2
    path.write_text(json.dumps(old))
    with pytest.raises(ProfileError, match="reconstruction"):
        load_profile(path)


@pytest.mark.parametrize("defect", ["memory_holdout", "class_holdout", "missing_batch", "version"])
def test_v2_does_not_hide_independent_rejection_or_raw_protocol_corruption(defect):
    parent = memory_parent(64)
    section = parent[gpu.SECTION]
    raw = deepcopy(section["measurements"])
    if defect in {"memory_holdout", "class_holdout"}:
        recipe, size = ("memory", 80265) if defect == "memory_holdout" else ("ordinary_mix", 65536)
        for row in raw[1 + gpu.PREFLIGHT_RECORDS:]:
            if row["recipe"] == recipe and row["items"] == size:
                row["resident_event_seconds"] *= .5
        base = {key: value for key, value in parent.items() if key != gpu.SECTION}
        parent = gpu.profile_from_gpu_execution_measurements(base, raw, section["calibration"])
    rows, calibration = launch_observations(parent)
    if defect == "missing_batch":
        rows.pop()
    elif defect == "version":
        rows[0]["launch_protocol_id"] = "unknown"
    if defect in {"missing_batch", "version"}:
        with pytest.raises(gpu.NumericalGpuExecutionError):
            costs.profile_from_gpu_launch_measurements(parent, rows, calibration)
        return
    result = costs.profile_from_gpu_launch_measurements(parent, rows, calibration)[costs.SECTION]
    if defect == "memory_holdout":
        assert result["families"]["memory"]["gpu"]["reason_codes"] == ["holdout_error"]
        assert all(row["reason_codes"] == ["required_family_rejected"] for row in result["workload_validation"].values())
    else:
        assert result["workload_validation"]["ordinary_expression_v2"]["reason_codes"] == ["holdout_error"]
    assert result["measurements"] == rows
    assert result["identity"]["parent_evidence_sha256"] == _hash(parent[gpu.SECTION])


def test_total_composition_places_fixed_only_on_the_compute_alternative():
    model = {"kind": costs.TOTAL_MEMORY_KIND, "traffic_equals_working_set": True,
        "working_set_range": [24, 48], "knots": [
            {"working_set_bytes": 24, "total_seconds": 30.0},
            {"working_set_bytes": 48, "total_seconds": 32.0}]}
    coefficients = {"arithmetic_seconds_per_operation": 2.0,
        "intrinsic_seconds_per_operation": {}, "memory_cost_model": model}
    recipe = {"arithmetic": 1, "intrinsics": {}}
    assert costs._predict_total(recipe, 1, coefficients, 10.0, 64) == 30.0
    assert costs._predict_total(recipe, 2, coefficients, 40.0, 64) == 44.0
