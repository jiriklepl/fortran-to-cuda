"""Original joined-team windows retain complete fields and runtime rejection.

This is a numerical/traffic diagnostic, not a calibration or performance result.
"""

import json
import os
import re
import shutil
from hashlib import sha256

import pytest

from compiler.numerical_contract import require_explicit_cuda_environment
from compiler.offload.exact_window import build_contract
from compiler.offload.exact_window_calibration import _run, build, build_native, diagnostic, emit, fixture_source
from compiler.tests.test_scoped_batch_sources_cuda import actual_profile


@pytest.mark.cuda
def test_original_teams_fields_transfers_and_runtime_limit_rejection(tmp_path):
    tools = {
        "nvcc": shutil.which("nvcc"),
        "cxx": shutil.which("g++-14") or shutil.which("g++"),
        "fortran": shutil.which("gfortran-15") or shutil.which("gfortran"),
    }
    if not all(tools.values()):
        pytest.skip("CUDA, C++ and Fortran toolchain required")
    require_explicit_cuda_environment()
    profile = actual_profile(tmp_path, tools["nvcc"], tools["cxx"])
    architecture = "sm_" + profile["hardware"]["compute_capability"].replace(".", "")
    source = tmp_path / "original.f90"
    source.write_text(fixture_source(original_teams=True))
    outputs, report, record = emit(source, {"calibration": True, "build_contract": build_contract(architecture)})
    assert len(record["contract"]["original_participation"]["teams"]) == 3
    environment = {
        **os.environ,
        "OMP_NUM_THREADS": "4",
        "OMP_DYNAMIC": "FALSE",
        "OMP_PROC_BIND": "FALSE",
        "OMP_THREAD_LIMIT": "4",
        "OMP_MAX_ACTIVE_LEVELS": "1",
    }
    for name in ("FORT_RUNTIME_TRACE", "FORT_EXACT_CALIBRATION_MODE"):
        environment.pop(name, None)
    native, _ = build_native(tmp_path / "native", source, environment, **tools)
    candidate, _ = build(tmp_path / "candidate", outputs, report, environment, architecture=architecture, **tools)
    measurements = tmp_path / "diagnostics"
    measurements.mkdir()
    _run([str(native), "16", "9"], cwd=measurements, environment=environment)
    expected = sha256((measurements / "fields.bin").read_bytes()).hexdigest()
    ledger = diagnostic(candidate, measurements, environment, 16, expected, label="resident", resident=True)
    assert ledger["observed"]["launch_count"] == 3
    for setting, value in (("OMP_THREAD_LIMIT", "2"), ("OMP_MAX_ACTIVE_LEVELS", "0")):
        ledger = diagnostic(
            candidate,
            measurements,
            {**environment, setting: value},
            16,
            expected,
            label=setting.lower(),
            resident=False,
        )
        assert ledger["observed"]["launch_count"] == 0


@pytest.mark.cuda
def test_changed_automatic_private_descriptor_rejects_production_window(tmp_path, monkeypatch):
    from compiler.offload import exact_window_calibration as calibration
    from compiler.tests.test_exact_window import observations

    tools = {
        "nvcc": shutil.which("nvcc"),
        "cxx": shutil.which("g++-14") or shutil.which("g++"),
        "fortran": shutil.which("gfortran-15") or shutil.which("gfortran"),
    }
    if not all(tools.values()):
        pytest.skip("CUDA, C++ and Fortran toolchain required")
    require_explicit_cuda_environment()
    profile = actual_profile(tmp_path, tools["nvcc"], tools["cxx"])
    architecture = "sm_" + profile["hardware"]["compute_capability"].replace(".", "")
    source = tmp_path / "original.f90"
    text = fixture_source(original_teams=True).replace(
        "implicit none\ncontains", "implicit none\ninteger::extent=3\ncontains"
    ).replace("real(8)::v", "real(8)::v\nreal(8)::scratch(-2:extent)")
    text = text.replace("audit=sum(a(:,1,1))", """!$omp parallel do private(i,scratch) schedule(static)
do i=1,n
scratch=a(i,1,1)
b(i,1,1)=b(i,1,1)+sum(scratch)
enddo
!$omp end parallel do
audit=sum(a(:,1,1))""")
    source.write_text(text)
    driver = calibration.DRIVER.replace("only:advance", "only:advance,extent").replace(
        "allocate(a(", "if(command_argument_count()>2)then\ncall get_command_argument(3,argument)\n"
        "read(argument,*)extent\nendif\nallocate(a(", 1
    )
    monkeypatch.setattr(calibration, "DRIVER", driver)
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "OMP_PROC_BIND": "FALSE",
                   "OMP_THREAD_LIMIT": "4", "OMP_MAX_ACTIVE_LEVELS": "1"}
    for name in ("FORT_RUNTIME_TRACE", "FORT_EXACT_CALIBRATION_MODE"):
        environment.pop(name, None)
    outputs, report, record = emit(source, {"calibration": True, "build_contract": build_contract(architecture)})
    native, _ = build_native(tmp_path / "native", source, environment, **tools)
    candidate, _ = build(tmp_path / "capture", outputs, report, environment, architecture=architecture, **tools)
    measured = tmp_path / "diagnostic"
    measured.mkdir()
    captured = _run([str(candidate), "16", "9"], cwd=measured,
                    environment={**environment, "FORT_EXACT_CALIBRATION_MODE": "resident"})
    metadata, = [json.loads(line.removeprefix("FORT_EXACT_OBSERVATION "))
                 for line in captured.stderr.splitlines() if line.startswith("FORT_EXACT_OBSERVATION ")]
    assert metadata["key"][-2:] == [6, -2]
    identity = metadata["identity"]
    expected_cpu = {
        "cpu_name": identity["cpu_name"], "fortran_version": identity["fortran_version"],
        "fortran_options": calibration.normalize_fortran_options(identity["fortran_options"]),
        "team": [identity[key] for key in ("threads", "dynamic", "proc_bind", "thread_limit", "max_active_levels")],
        "affinity": identity["affinity"], "environment": identity["environment"],
        "effective_schedule": identity["effective_schedule"],
    }
    # Synthetic profitable observations exercise production guard behavior.
    # They are not measurements, calibration evidence, or performance claims.
    evidence = observations(record["contract"])
    evidence.update(metadata)
    for row in evidence["observations"]:
        row.update(metadata, observed_key=metadata["key"], observed_cpu_identity=expected_cpu)
    outputs, report, automatic = emit(source, {"evidence": evidence})
    assert automatic["selection"] == "resident"
    production, _ = build(tmp_path / "production", outputs, report, environment, architecture=architecture, **tools)
    for extent, launches in ((3, 3), (9, 0), (-3, 0)):
        arguments = ["16", "9", str(extent)]
        _run([str(native), *arguments], cwd=measured, environment=environment)
        expected = (measured / "fields.bin").read_bytes()
        result = _run([str(production), *arguments], cwd=measured,
                      environment={**environment, "FORT_RUNTIME_TRACE": "1"})
        assert (measured / "fields.bin").read_bytes() == expected
        assert len(re.findall(r"^FORT_SCOPED launch ", result.stderr, re.MULTILINE)) == launches
        assert "FORT_EXACT_OBSERVATION" not in result.stderr
        if launches == 0:
            assert not re.search(r"^FORT_SCOPED (upload|download) ", result.stderr, re.MULTILINE)


@pytest.mark.cuda
def test_effective_runtime_schedule_selects_before_ownership_with_unchanged_environment(tmp_path, monkeypatch):
    from compiler.offload import exact_window_calibration as calibration
    from compiler.tests.test_exact_window import observations

    tools = {
        "nvcc": shutil.which("nvcc"),
        "cxx": shutil.which("g++-14") or shutil.which("g++"),
        "fortran": shutil.which("gfortran-15") or shutil.which("gfortran"),
    }
    if not all(tools.values()):
        pytest.skip("CUDA, C++ and Fortran toolchain required")
    require_explicit_cuda_environment()
    profile = actual_profile(tmp_path, tools["nvcc"], tools["cxx"])
    architecture = "sm_" + profile["hardware"]["compute_capability"].replace(".", "")
    source = tmp_path / "original.f90"
    source.write_text(fixture_source(original_teams=True).replace("schedule(static)", "schedule(runtime)"))
    # Change the task's schedule ICV outside the admitted invocation. The
    # environment is identical in capture, matching and all rejected executions.
    driver = calibration.DRIVER.replace("allocate(a(", """call get_command_argument(3,argument)
read(argument,*)i
select case(i)
case(0)
call omp_set_schedule(omp_sched_static,0)
case(1)
call omp_set_schedule(omp_sched_dynamic,1)
case(2)
call omp_set_schedule(omp_sched_static,7)
case(3)
call omp_set_schedule(ibset(omp_sched_static,31),0)
end select
allocate(a(""", 1)
    monkeypatch.setattr(calibration, "DRIVER", driver)
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "OMP_PROC_BIND": "FALSE",
                   "OMP_THREAD_LIMIT": "4", "OMP_MAX_ACTIVE_LEVELS": "1", "OMP_SCHEDULE": "static"}
    for name in ("FORT_RUNTIME_TRACE", "FORT_EXACT_CALIBRATION_MODE"):
        environment.pop(name, None)
    outputs, report, record = emit(source, {"calibration": True, "build_contract": build_contract(architecture)})
    native, _ = build_native(tmp_path / "native", source, environment, **tools)
    candidate, _ = build(tmp_path / "capture", outputs, report, environment, architecture=architecture, **tools)
    measured = tmp_path / "diagnostic"
    measured.mkdir()
    captured = _run([str(candidate), "16", "9", "0"], cwd=measured,
                    environment={**environment, "FORT_EXACT_CALIBRATION_MODE": "resident"})
    metadata, = [json.loads(line.removeprefix("FORT_EXACT_OBSERVATION "))
                 for line in captured.stderr.splitlines() if line.startswith("FORT_EXACT_OBSERVATION ")]
    identity = metadata["identity"]
    assert identity["effective_schedule"] == {"kind": 1, "chunk": 0}
    assert calibration.observed_schedule(captured.stdout) == identity["effective_schedule"]
    expected_cpu = {
        "cpu_name": identity["cpu_name"], "fortran_version": identity["fortran_version"],
        "fortran_options": calibration.normalize_fortran_options(identity["fortran_options"]),
        "team": [identity[key] for key in ("threads", "dynamic", "proc_bind", "thread_limit", "max_active_levels")],
        "affinity": identity["affinity"], "environment": identity["environment"],
        "effective_schedule": identity["effective_schedule"],
    }
    # Synthetic profitable evidence isolates selection; no timing or calibration
    # claim is made. Actual identities and exact float64 fields remain checked.
    evidence = observations(record["contract"])
    evidence.update(metadata)
    for row in evidence["observations"]:
        row.update(metadata, observed_key=metadata["key"], observed_cpu_identity=expected_cpu)
    outputs, report, automatic = emit(source, {"evidence": evidence})
    assert automatic["selection"] == "resident"
    production, _ = build(tmp_path / "production", outputs, report, environment, architecture=architecture, **tools)
    for setting, kind, chunk, launches in ((0, 1, 0, 3), (1, 2, 1, 0), (2, 1, 7, 0), (3, 0x80000001, 0, 0)):
        arguments = ["16", "9", str(setting)]
        native_result = _run([str(native), *arguments], cwd=measured, environment=environment)
        expected = (measured / "fields.bin").read_bytes()
        result = _run([str(production), *arguments], cwd=measured,
                      environment={**environment, "FORT_RUNTIME_TRACE": "1"})
        assert calibration.observed_schedule(native_result.stdout) == {"kind": kind, "chunk": chunk}
        assert calibration.observed_schedule(result.stdout) == {"kind": kind, "chunk": chunk}
        # Accumulation changes out on every invocation; bytewise comparison
        # also rejects replay of a completed invocation or accumulation.
        assert (measured / "fields.bin").read_bytes() == expected
        assert len(re.findall(r"^FORT_SCOPED launch ", result.stderr, re.MULTILINE)) == launches
        assert "FORT_EXACT_OBSERVATION" not in result.stderr
        if launches == 0:
            assert not re.search(r"^FORT_SCOPED ", result.stderr, re.MULTILINE)
