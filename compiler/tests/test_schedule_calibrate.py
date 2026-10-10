"""The optional producer preserves frozen costs and original native semantics."""
import json
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from compiler.offload.numerical_calibration import (
    COMPUTE_FAMILIES,
    NumericalCalibrationError,
    compute_generator_identity,
)
from compiler.offload.schedule_calibrate import (
    calibrate_schedule,
    main,
    measurement_environment,
    parse_schedule_measurements,
)
from compiler.offload.schedule_calibration import schedule_protocol_identity
from compiler.tests.test_schedule_calibration import schedule_profile


def native_text(profile):
    section = profile["source_schedule_validation"]
    identity = deepcopy(section["execution_identity"])
    identity["kind"] = "schedule_identity_v1"
    for compiler in identity["fortran"].values():
        compiler.pop("semantic_options")
    return "\n".join(json.dumps(row) for row in [identity, *section["measurements"]])


def test_parser_normalizes_actual_compiler_objects_and_preserves_failed_rows(monkeypatch):
    profile = schedule_profile(monkeypatch)
    text = native_text(profile)
    records, identity = parse_schedule_measurements(text)
    assert identity == profile["source_schedule_validation"]["execution_identity"]
    assert records == profile["source_schedule_validation"]["measurements"]
    records[0]["agreement_passed"] = False
    raw = json.loads(text.splitlines()[0])
    parsed, _ = parse_schedule_measurements("\n".join(json.dumps(row) for row in [raw, *records]))
    assert parsed[0]["agreement_passed"] is False


@pytest.mark.parametrize("text", ["", "not json", "[]", '{"kind":"unknown"}',
                                      '{"kind":"schedule_identity_v1"}'])
def test_parser_rejects_incomplete_or_unrecognized_output(text):
    with pytest.raises(NumericalCalibrationError):
        parse_schedule_measurements(text)


def test_environment_removes_inherited_worker_binding_and_timing_instrumentation():
    env = measurement_environment({"GOMP_CPU_AFFINITY": "0", "OMP_PLACES": "cores",
        "OMP_SCHEDULE": "dynamic,1", "OMP_PROC_BIND": "close", "OMP_DYNAMIC": "TRUE",
        "FORT_PHASE_TIMING": "1", "FORT_RUNTIME_TRACE": "1", "FORT_SCOPE_TEST_FAIL": "1",
        "PATH": "/bin"})
    assert env == {"PATH": "/bin", "OMP_DYNAMIC": "FALSE", "OMP_PROC_BIND": "false", "OMP_SCHEDULE": "static"}


def test_producer_compiles_both_original_objects_and_adds_only_optional_evidence(monkeypatch, tmp_path):
    profile = schedule_profile(monkeypatch)
    original = deepcopy(profile)
    calls = []
    monkeypatch.setattr("compiler.offload.schedule_calibrate._tool", lambda path, _: path)
    def run(command, directory, log, **kwargs):
        calls.append((command, kwargs))
        return native_text(profile) if command[0] == "taskset" else ""
    args = SimpleNamespace(fortran="/tool/gfortran", host_cxx="/tool/g++",
        fortran_flag=["-O3", "-fopenmp"], build_dir=tmp_path / "build")
    result = calibrate_schedule(profile, args, run=run)
    assert profile == original
    assert result["numerical"] == profile["numerical"]
    assert len(calls) == 5
    assert calls[0][0][:3] == ["/tool/gfortran", "-O3", "-fopenmp"]
    assert "-lgfortran" in calls[2][0]
    assert calls[3][0][-1] == "--identity"
    assert calls[-1][0][:3] == ["taskset", "-c", ",".join(map(str, profile["numerical"]["identity"]["cpu_affinity"]))]
    assert calls[-1][1]["env"]["OMP_SCHEDULE"] == "static"
    assert "schedule(static)" in (args.build_dir / "static.f90").read_text()
    assert "schedule(runtime)" in (args.build_dir / "runtime.f90").read_text()
    assert not result["source_schedule_validation"]["calibration"]["coefficients_fitted"]
    assert (args.build_dir / "evidence.json").is_file()


def test_unavailable_base_rejects_before_compiling_or_running(monkeypatch, tmp_path):
    profile = schedule_profile(monkeypatch)
    profile["numerical"]["families"]["memory_v2"]["native_fork_join"]["status"] = "rejected"
    args = SimpleNamespace(fortran=None, host_cxx=None, fortran_flag=["-O3", "-fopenmp"],
                           build_dir=tmp_path / "build")
    with pytest.raises(NumericalCalibrationError):
        calibrate_schedule(profile, args, run=lambda *a, **k: pytest.fail("base is unavailable"))
    assert not args.build_dir.exists()


def test_actual_object_identity_is_checked_before_expensive_sampling(monkeypatch, tmp_path):
    profile = schedule_profile(monkeypatch)
    calls = []
    monkeypatch.setattr("compiler.offload.schedule_calibrate._tool", lambda path, _: path)
    def run(command, directory, log, **kwargs):
        calls.append(command)
        if command[0] != "taskset":
            return ""
        assert command[-1] == "--identity"
        identity = json.loads(native_text(profile).splitlines()[0])
        identity["fortran"]["runtime"]["compiler_options"] = "-O0 -fopenmp"
        return json.dumps(identity)
    args = SimpleNamespace(fortran="/tool/gfortran", host_cxx="/tool/g++",
        fortran_flag=["-O3", "-fopenmp"], build_dir=tmp_path / "build")
    with pytest.raises(NumericalCalibrationError, match="semantic_options mismatch"):
        calibrate_schedule(profile, args, run=run)
    assert len(calls) == 4


def test_cli_preserves_input_and_existing_output(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text("{}")
    with pytest.raises(SystemExit):
        main(["--profile", str(path), "--output", str(path), "--build-dir", str(tmp_path / "build")])
    assert path.read_text() == "{}"


def test_protocol_identity_is_present_without_changing_numerical_generator():
    assert len(compute_generator_identity()) == 64
    assert len(schedule_protocol_identity()) == 64
    source = Path(__file__).parents[1] / "offload/schedule_calibrate.cpp"
    text = source.read_text()
    assert all('"' + family + '"' in text for family in COMPUTE_FAMILIES)
    assert "result.elapsed < 0.2" in text
    assert "batch != 7" in text


def test_cpu_protocol_driver_compiles_without_cuda():
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    source = Path(__file__).parents[1] / "offload/schedule_calibrate.cpp"
    for precision in (32, 64):
        completed = subprocess.run([compiler, "-std=c++17", "-fopenmp", "-fsyntax-only",
            "-DCALIBRATION_PRECISION=" + str(precision), str(source)], capture_output=True, text=True, timeout=30)
        assert completed.returncode == 0, completed.stderr
