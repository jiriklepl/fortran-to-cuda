"""Numerical costs retain independent evidence and never guess applicability."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from compiler.offload.numerical_calibration import (
    BACKEND_ID, INTRINSIC_BACKEND_ID, FAMILIES, FIT_SIZES, HOLDOUT_SIZES, MIX_FAMILIES, NumericalCalibrationError,
    _fit, calibrate_numerical, generator_identity, numerical_costs,
    numerical_intrinsic_costs, parse_measurements, profile_from_measurements,
)
from compiler.offload.profile import ProfileError, validate_profile
from compiler.tests.test_offload_profile import profile


def observations():
    base = profile()
    identity = {"kind": "numerical_identity", "backend_id": BACKEND_ID,
                **{name: base["hardware"][name] for name in ("gpu_name", "gpu_uuid", "compute_capability")},
                **{name: base["toolchain"][name] for name in ("cuda_runtime_version", "driver_version")},
                "cpu_threads": 4, "precision_bits": 64}
    records = [identity]
    for family in FAMILIES:
        for device, slope in (("cpu", 1e-7), ("gpu", 1e-9)):
            if family in MIX_FAMILIES:
                slope *= sum(MIX_FAMILIES[family].values())
            for size in FIT_SIZES + HOLDOUT_SIZES:
                duration = 2e-6 + slope * size
                records.append({"kind": "numerical_cost", "family": family, "device": device,
                                "items": size, "role": "fit" if size in FIT_SIZES else "holdout",
                                "seconds": [duration, duration, 3 * duration, duration, duration],
                                "agreement_passed": True})
    return records


def calibrated():
    return profile_from_measurements(profile(), observations(), calibration={"application_profiled": False})


def test_optional_numerical_costs_preserve_base_and_raw_evidence():
    original = profile()
    records = observations()
    result = profile_from_measurements(original, records, calibration={"application_profiled": True})
    assert "numerical" not in original
    assert result["rates"] == original["rates"]
    assert result["numerical"]["measurements"] == records
    assert result["numerical"]["generator_id"] == generator_identity()
    assert result["numerical"]["calibration"]["application_profiled"] is False
    assert parse_measurements("\n".join(json.dumps(row) for row in records)) == records
    validate_profile(result)
    for family in FAMILIES:
        costs = result["numerical"]["families"][family]
        multiplier = sum(MIX_FAMILIES[family].values()) if family in MIX_FAMILIES else 1
        assert costs["cpu"]["seconds_per_item"] == pytest.approx(multiplier * 1e-7)
        assert costs["gpu"]["seconds_per_item"] == pytest.approx(multiplier * 1e-9)
        assert costs["cpu"]["fixed_seconds"] == pytest.approx(2e-6)
        assert [row["items"] for row in costs["cpu"]["holdouts"]] == list(HOLDOUT_SIZES)


def test_holdouts_cannot_affect_fitted_coefficients_and_failed_holdout_rejects():
    records = observations()
    fit = [row for row in records if row.get("family") == FAMILIES[0] and row.get("device") == "cpu"
           and row["role"] == "fit"]
    before = _fit(fit)
    for row in records:
        if row.get("role") == "holdout":
            row["seconds"] = [1e10] * 5
    assert _fit(fit) == before
    with pytest.raises(NumericalCalibrationError, match="independent holdout"):
        profile_from_measurements(profile(), records, calibration={})


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "role", "agreement", "team", "precision", "gpu", "backend"])
def test_incomplete_wrong_identity_or_incorrect_observations_never_enable_costs(mutation):
    records = observations()
    if mutation == "missing":
        records.pop()
    elif mutation == "duplicate":
        records.append(deepcopy(records[-1]))
    elif mutation == "role":
        records[-1]["role"] = "fit"
    elif mutation == "agreement":
        records[-1]["agreement_passed"] = False
    else:
        field = {"team": "cpu_threads", "precision": "precision_bits", "gpu": "gpu_uuid", "backend": "backend_id"}[mutation]
        records[0][field] = "wrong"
    with pytest.raises(NumericalCalibrationError):
        profile_from_measurements(profile(), records, calibration={})


@pytest.mark.parametrize("text", ["", "[]", "not json", '{"kind":"rate"}',
                                  '{"kind":"numerical_cost","seconds":[1,1,1]}',
                                  '{"kind":"numerical_cost","seconds":[1,1,1,1,0]}'])
def test_bad_measurement_protocol_fails(text):
    with pytest.raises(NumericalCalibrationError):
        parse_measurements(text)


def test_saved_numerical_profile_revalidates_holdouts_and_identity():
    result = calibrated()
    result["numerical"]["cpu_threads"] = 3
    with pytest.raises(ProfileError, match="cpu_threads mismatch"):
        validate_profile(result)
    result = calibrated()
    result["numerical"]["families"][FAMILIES[0]]["gpu"]["seconds_per_item"] *= 2
    with pytest.raises(ProfileError, match="holdout"):
        validate_profile(result)
    result = calibrated()
    result["numerical"]["families"][FAMILIES[0]]["cpu"]["holdouts"] = []
    with pytest.raises(ProfileError, match="holdouts"):
        validate_profile(result)


@pytest.mark.parametrize("value", [None, "0", True, float("nan"), -1])
def test_malformed_saved_holdout_error_is_a_public_profile_rejection(value):
    result = calibrated()
    result["numerical"]["families"][FAMILIES[0]]["cpu"]["holdouts"][0]["relative_error"] = value
    with pytest.raises(ProfileError, match="finite and nonnegative"):
        validate_profile(result)


@pytest.mark.parametrize("field", ["backend_id", "generator_id", "recipe_id"])
def test_arbitrary_mix_or_different_backend_generator_never_inherits_rates(field):
    result = calibrated()
    numerical = result["numerical"]
    identity = {"backend_id": BACKEND_ID, "generator_id": numerical["generator_id"],
                "recipe_id": numerical["families"][FAMILIES[0]]["recipe_id"]}
    assert numerical_costs(result, FAMILIES[0], **identity) == numerical["families"][FAMILIES[0]]
    identity[field] = "another implementation"
    with pytest.raises(NumericalCalibrationError, match="identity mismatch"):
        numerical_costs(result, FAMILIES[0], **identity)
    with pytest.raises(NumericalCalibrationError, match="no numerical calibration"):
        numerical_costs(profile(), FAMILIES[0], **identity)


def test_primitive_counts_use_independent_mixed_validation_and_never_fma_rates():
    result = calibrated()
    identity = {"backend_id": INTRINSIC_BACKEND_ID, "generator_id": result["numerical"]["generator_id"]}
    costs = numerical_intrinsic_costs(result, {"sqrt": 3, "acos": 1, "cos": 2}, **identity)
    assert costs["cpu_seconds_per_iteration"] == pytest.approx(6 * 1e-7 / 16)
    assert costs["gpu_seconds_per_iteration"] == pytest.approx(6 * 1e-9 / 16)
    result["rates"]["gpu_flops_per_second"] *= 100
    assert numerical_intrinsic_costs(result, {"sqrt": 3, "acos": 1, "cos": 2}, **identity) == costs
    for counts in ({"exp": 1}, {"sqrt": -1}, {"sqrt": True}, {}, [("sqrt", 1), ("sqrt", 2)]):
        with pytest.raises(NumericalCalibrationError, match="counts"):
            numerical_intrinsic_costs(result, counts, **identity)


def test_failed_unfit_mixed_fixture_disables_primitive_model_without_changing_rates():
    records = observations()
    for record in records:
        if record.get("family") in MIX_FAMILIES:
            record["seconds"] = [value * 2 for value in record["seconds"]]
    result = profile_from_measurements(profile(), records, calibration={})
    assert result["numerical"]["intrinsic_model"]["available"] is False
    assert result["numerical"]["intrinsic_model"]["reasons"]
    validate_profile(result)
    with pytest.raises(NumericalCalibrationError, match="mixed holdout"):
        numerical_intrinsic_costs(result, {"sqrt": 1}, backend_id=INTRINSIC_BACKEND_ID,
                                  generator_id=result["numerical"]["generator_id"])


def test_build_and_measure_use_fixed_budget_clean_environment_and_retained_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("FORT_PHASE_TIMING", "1")
    monkeypatch.setenv("FORT_RUNTIME_TRACE", "1")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    monkeypatch.setenv("FORT_SCOPE_TEST_FAIL", "1")
    invocations = []

    def run(command, directory, log, *, timeout, env=None):
        invocations.append({"argv": command, "cwd": directory, "log": log, "env": env})
        if env is not None:
            assert not {"FORT_PHASE_TIMING", "FORT_RUNTIME_TRACE", "CUDA_LAUNCH_BLOCKING", "FORT_SCOPE_TEST_FAIL"} & env.keys()
            return "\n".join(json.dumps(row) for row in observations())
        return ""

    args = SimpleNamespace(arch="sm_86", precision=64, threads=4, device=0)
    result = calibrate_numerical(profile(), args, tmp_path, "/nvcc", "/g++", run=run)
    assert len(invocations) == 2
    assert "-DCALIBRATION_PRECISION=64" in invocations[0]["argv"]
    assert invocations[1]["argv"][-2:] == ["4", "0"]
    assert result["numerical"]["calibration"]["gpu_method"].startswith("CUDA events")
    assert result["numerical"]["calibration"]["benchmark_source_sha256"]
