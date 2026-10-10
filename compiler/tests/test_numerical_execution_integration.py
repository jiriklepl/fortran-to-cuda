"""The repaired optional protocol cannot inherit older placement authority."""

import json
from copy import deepcopy

import pytest

from compiler.offload.numerical_calibration import NumericalCalibrationError
from compiler.offload.source_compute import apply_source_compute_costs
from compiler.tests.test_source_compute_costs import profile_v2, source_analysis


@pytest.mark.parametrize("version", [1, 2, 3])
def test_calibration_version_is_explicit_and_default_unchanged(tmp_path, monkeypatch, version):
    from compiler.offload import calibrate

    calls = []
    monkeypatch.setattr(calibrate, "calibrate", calls.append)
    assert calibrate.main(["--output", str(tmp_path / "default.json")]) == 0
    assert calls[-1].numerical_version == 2
    assert not calls[-1].numerical_costs
    assert calibrate.main(["--output", str(tmp_path / "explicit.json"),
                           "--numerical-costs", "--numerical-version", str(version)]) == 0
    assert calls[-1].numerical_version == version
    assert calls[-1].numerical_costs


@pytest.mark.parametrize("version", [1, 2, 3])
def test_explicit_version_runs_only_its_own_producer(tmp_path, monkeypatch, version):
    from compiler.offload import calibrate, numerical_calibration, numerical_execution_calibration
    from compiler.tests.test_offload_profile import observations

    selected = []
    monkeypatch.setattr(calibrate, "_tool", lambda requested, candidates: candidates[0])

    def run(argv, directory, log, **kwargs):
        if "--version" in argv:
            return "NVCC V13.4.92" if argv[0] == "nvcc" else "GCC 14.4.0"
        if argv[0].endswith("/calibration"):
            return "\n".join(json.dumps(row) for row in observations())
        return ""

    def producer(which):
        def invoke(profile, *args, **kwargs):
            selected.append(which)
            return profile
        return invoke

    monkeypatch.setattr(calibrate, "_run", run)
    monkeypatch.setattr(numerical_calibration, "calibrate_numerical", producer(1))
    monkeypatch.setattr(numerical_calibration, "calibrate_numerical_v2", producer(2))
    monkeypatch.setattr(numerical_execution_calibration, "calibrate_numerical_execution", producer(3))
    assert calibrate.main(["--output", str(tmp_path / "evidence.json"), "--numerical-costs",
                           "--numerical-version", str(version)]) == 0
    assert selected == [version]


def test_new_unavailable_protocol_cannot_borrow_accepted_v2_costs(monkeypatch):
    from compiler.offload import numerical_execution, source_compute

    original = profile_v2()
    evidence = deepcopy(original)
    evidence["numerical_execution"] = {"schema_version": 1, "phase": "cpu_only"}
    seen = []

    def incomplete(profile, counts, **kwargs):
        seen.append((profile, counts, kwargs))
        raise NumericalCalibrationError("numerical execution GPU evidence unavailable")

    monkeypatch.setattr(numerical_execution, "execution_compute_model", incomplete)
    monkeypatch.setattr(source_compute, "numerical_compute_model",
                        lambda *args, **kwargs: pytest.fail("new protocol borrowed older coefficients"))
    _, _, analysis = source_analysis()
    result = apply_source_compute_costs(analysis, evidence, "serial")
    unit, = result.units
    assert unit.compute_model is None
    assert unit.work_per_iteration is None
    assert unit.work_estimate_reason == "numerical execution GPU evidence unavailable"
    assert unit.region is analysis.units[0].region
    assert unit.footprints == analysis.units[0].footprints
    assert len(seen) == 1
    assert original == {key: value for key, value in evidence.items() if key != "numerical_execution"}


def test_optional_protocol_is_validated_before_profile_use(monkeypatch):
    from compiler.offload import numerical_execution
    from compiler.offload.profile import ProfileError, validate_profile

    evidence = profile_v2()
    evidence["numerical_execution"] = {"schema_version": 1, "phase": "cpu_only"}
    calls = []

    def reject(profile):
        calls.append(profile)
        raise NumericalCalibrationError("saved execution evidence disagrees with raw observations")

    monkeypatch.setattr(numerical_execution, "validate_execution_profile", reject)
    with pytest.raises(ProfileError, match="saved execution evidence"):
        validate_profile(evidence)
    assert calls == [evidence]
