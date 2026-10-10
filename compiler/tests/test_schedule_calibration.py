"""Runtime-static applicability is independently validated, never assumed."""

import math
from copy import deepcopy

import pytest

from compiler.offload.numerical_calibration import (
    COMPUTE_FAMILIES,
    COMPUTE_SIZES,
    MEMORY_SIZES,
    NumericalCalibrationError,
    numerical_compute_model,
)
from compiler.offload.schedule_calibration import (
    RUNTIME_SCHEDULE,
    profile_from_schedule_measurements,
    runtime_fixture_source,
    runtime_schedule_requirement,
    schedule_prediction_seconds,
)
from compiler.offload.source_compute import apply_source_compute_costs
from compiler.tests.test_source_compute_costs import profile_v2, source_analysis


def schedule_profile(monkeypatch, *, change=None):
    monkeypatch.setattr("compiler.offload.schedule_calibration.schedule_protocol_identity", lambda: "a" * 64)
    profile = profile_v2()
    identity = profile["numerical"]["identity"]
    execution = {"cpu_threads": identity["cpu_threads"], "precision_bits": identity["precision_bits"],
        "cpu_affinity": deepcopy(identity["cpu_affinity"]), "omp_dynamic": False, "omp_proc_bind": "false",
        "runtime_schedule": dict(RUNTIME_SCHEDULE),
        "fortran": {mode: deepcopy(identity["fortran"]) for mode in ("static", "runtime")}}
    records = []
    for family in COMPUTE_FAMILIES:
        for items in MEMORY_SIZES if family == "memory_v2" else COMPUTE_SIZES:
            try:
                seconds = schedule_prediction_seconds(profile, family, items)
            except NumericalCalibrationError:
                seconds = .01  # Structurally out-of-range measurements remain raw evidence.
            repetitions = max(1, math.ceil(.21 / seconds))
            samples = [{"batch": batch, "repetitions": repetitions,
                "elapsed_seconds": repetitions * seconds, "wall_seconds": repetitions * seconds}
                for batch in range(7)]
            records.append({"kind": "schedule_cost_v1", "family": family, "items": items,
                "samples": samples, "static_samples": deepcopy(samples), "agreement_passed": True})
    if change is not None:
        change(records)
    return profile_from_schedule_measurements(profile, records, execution, calibration={"application_profiled": True})


def request(profile, counts=None):
    counts = {"sqrt": 1, "acos": 1, "cos": 3, "divide": 2} if counts is None else counts
    model = numerical_compute_model(profile, counts, native_participation="fork_join")
    return runtime_schedule_requirement(profile, model, counts)


def test_runtime_fixture_retains_original_recipes_with_separate_abi_and_schedule():
    source = runtime_fixture_source(64)
    assert "module numerical_native_runtime_v2" in source
    assert "subroutine fort_numerical_native_runtime_v1(" in source
    assert "subroutine fort_numerical_runtime_fortran_identity_v1(" in source
    assert "schedule(runtime)" in source
    assert "schedule(static)" not in source


def test_validated_runtime_model_has_structural_range_and_original_costs(monkeypatch):
    profile = schedule_profile(monkeypatch)
    requirement = request(profile)
    assert requirement["runtime_schedule"] == RUNTIME_SCHEDULE
    assert requirement["item_range"] == [131072, 1048576]
    validation = profile["source_schedule_validation"]
    assert not validation["calibration"]["application_profiled"]
    assert not validation["calibration"]["coefficients_fitted"]
    arithmetic = validation["families"]["arithmetic_v2"]
    assert arithmetic["status"] == "accepted"
    assert arithmetic["holdouts"][0]["items"] == 65536
    assert arithmetic["holdouts"][0]["reason"] == "out_of_memory_range"
    assert len(validation["measurements"]) == sum(len(MEMORY_SIZES if family == "memory_v2" else COMPUTE_SIZES)
                                                   for family in COMPUTE_FAMILIES)
    _, _, analysis = source_analysis()
    modeled, = apply_source_compute_costs(analysis, profile, "fork_join_runtime").units
    assert modeled.compute_model["native_participation"] == "fork_join_runtime"
    assert modeled.compute_model["native_fortran"]["backend_identity"] == "native_fork_join"
    assert modeled.compute_model["runtime_schedule"] == RUNTIME_SCHEDULE
    assert modeled.compute_model["item_range"] == [131072, 1048576]


def test_missing_optional_validation_keeps_original_static_model_and_declines_runtime():
    profile = profile_v2()
    _, _, analysis = source_analysis()
    static, = apply_source_compute_costs(analysis, profile, "fork_join").units
    runtime, = apply_source_compute_costs(analysis, profile, "fork_join_runtime").units
    assert static.compute_model is not None
    assert runtime.compute_model is None
    assert "runtime schedule validation is missing" in runtime.work_estimate_reason


@pytest.mark.parametrize("field", ["numerical_generator_id", "source_sha256", "protocol_source_sha256"])
def test_changed_source_or_backend_identity_cannot_reuse_validation(monkeypatch, field):
    profile = schedule_profile(monkeypatch)
    profile["source_schedule_validation"][field] = "b" * 64
    with pytest.raises(NumericalCalibrationError, match="identity mismatch"):
        request(profile)


@pytest.mark.parametrize("change", [
    lambda execution: execution.update(cpu_affinity=[0, 1, 2, 3]),
    lambda execution: execution.update(cpu_threads=True),
    lambda execution: execution.update(omp_dynamic=True),
    lambda execution: execution.update(omp_proc_bind="close"),
    lambda execution: execution.update(runtime_schedule={"kind": "static", "chunk": 1}),
    lambda execution: execution.update(runtime_schedule={"kind": "static", "chunk": False}),
    lambda execution: execution["fortran"]["runtime"].update(semantic_options="-O0\x1f-fopenmp"),
])
def test_runtime_execution_contract_matches_native_fortran_and_placement(monkeypatch, change):
    profile = schedule_profile(monkeypatch)
    change(profile["source_schedule_validation"]["execution_identity"])
    with pytest.raises(NumericalCalibrationError, match="runtime schedule"):
        request(profile)


def test_status_cannot_substitute_for_raw_observations(monkeypatch):
    profile = schedule_profile(monkeypatch)
    profile["source_schedule_validation"]["measurements"][1]["samples"][0]["elapsed_seconds"] *= 10
    # Change all batches so the robust median actually changes.
    for sample in profile["source_schedule_validation"]["measurements"][1]["samples"]:
        sample["elapsed_seconds"] *= 10
    with pytest.raises(NumericalCalibrationError, match="differs from raw"):
        request(profile)


def test_rejected_in_range_measurement_cannot_shrink_accepted_range(monkeypatch):
    def change(records):
        row = next(row for row in records if row["family"] == "primitive_cos_v2" and row["items"] == 131072)
        for sample in row["samples"]:
            sample["elapsed_seconds"] *= 2
            sample["wall_seconds"] *= 2
    profile = schedule_profile(monkeypatch, change=change)
    failed = profile["source_schedule_validation"]["families"]["primitive_cos_v2"]
    assert failed["item_range"] == [131072, 1048576]
    assert failed["status"] == "rejected"
    assert failed["reason_codes"] == ["runtime_prediction_error"]
    with pytest.raises(NumericalCalibrationError, match="primitive_cos_v2 rejected"):
        request(profile)
    # Independent ordinary-only work does not borrow the rejected cosine family.
    assert request(profile, {"divide": 1})["runtime_schedule"] == RUNTIME_SCHEDULE


@pytest.mark.parametrize("field", ["samples", "static_samples"])
def test_both_interleaved_modes_require_seven_bounded_batches(monkeypatch, field):
    profile = schedule_profile(monkeypatch)
    profile["source_schedule_validation"]["measurements"][0][field][0]["wall_seconds"] = .1
    with pytest.raises(NumericalCalibrationError, match="200 ms"):
        request(profile)


def test_missing_or_duplicate_rows_cannot_be_accepted(monkeypatch):
    profile = schedule_profile(monkeypatch)
    records = profile["source_schedule_validation"]["measurements"]
    records.append(deepcopy(records[0]))
    with pytest.raises(NumericalCalibrationError, match="duplicate"):
        request(profile)
