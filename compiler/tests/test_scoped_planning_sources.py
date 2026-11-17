"""Complete-source planning proves controls immutable before querying them."""
from __future__ import annotations

import shutil
import subprocess
from hashlib import sha256
from pathlib import Path

import pytest

from compiler.driver.options import CompilerOptions
from compiler.emission.common.resources import read_scoped_runtime
from compiler.offload.config import OffloadConfig
from compiler.offload.numerical_calibration import profile_from_compute_measurements
from compiler.scopes.source import form_source_scopes
from compiler.tests.test_numerical_calibration_v2 import observations_v2
from compiler.tests.test_numerical_sources import NORMALIZED, ORIGINAL, package
from compiler.tests.test_offload_profile import scoped_profile
from compiler.tests.test_source_scopes import FACT, PROGRAM


def generate(tmp_path, monkeypatch, source=PROGRAM, *, captures=None, numerical_sources=None):
    # Synthetic costs exercise source proof/generation only, never performance.
    monkeypatch.setattr("compiler.emission.cuda.scoped.compiler_identity", lambda profile: {})
    path = tmp_path / "original.f90"
    path.write_text(source)
    profile = scoped_profile()
    profile["scoped"]["runtime_id"] = read_scoped_runtime()[1]["runtime_id"]
    # profile_expression also verifies compiler banner identities.
    profile["toolchain"]["nvcc_version"] = "Cuda compilation tools, release 13.4, V13.4.88"
    profile["toolchain"]["host_cxx_version"] = "g++ (GCC) 14.4.0"
    profile = profile_from_compute_measurements(profile, observations_v2(), calibration={})
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": captures or {"argument::a": FACT, "argument::b": {**FACT, "initialized": "none"},
                                       "argument::out": {**FACT, "initialized": "none"}}}
    outputs, report = form_source_scopes([path], "step", facts=facts,
                                         options=CompilerOptions(fallback="host", gpu_policy="auto", memory_model="scoped"),
                                         config=OffloadConfig("auto", profile), numerical_sources=numerical_sources)
    return outputs, report


def packaged_control(tmp_path, monkeypatch, *, name="query_start", declaration=None, dependency=False,
                     guarded=False, mutable_consumer=False):
    declaration = declaration or f"integer,parameter::{name}=-1"
    original = ORIGINAL.replace("integer::i\n!$omp", "integer::i\n" + declaration + "\n!$omp", 1)
    original = original.replace("do i=-1,n-4", f"do i={name},n-4", 1)
    if dependency:
        original = original.replace("real(8) :: gain=2", "integer,parameter::origin=3\nreal(8) :: gain=2")
        original = original.replace("subroutine producer(a,b,n)\n", "subroutine producer(a,b,n)\nuse settings,only:seed=>origin\n")
    normalized = NORMALIZED.replace("weights,alb,blb,wlb)", f"weights,alb,blb,wlb,{name})")
    normalized = normalized.replace("integer,intent(in)::n,alb,blb,wlb", f"integer,intent(in)::n,alb,blb,wlb,{name}")
    normalized = normalized.replace("do i=-1,n-4", f"do i={name},n-4")
    if guarded:
        span = "call producer(a,b,n)\ncall transform(b)\ncall consumer(b,out,n)"
        original = original.replace(span, "if(n>0) then\n" + span + "\nendif")
        original = original.replace("!$omp parallel do private(i)\ndo i=" + name + ",n-4",
                                    f"if({name}<0) then\n!$omp parallel do private(i)\ndo i={name},n-4", 1)
        original = original.replace("enddo\n!$omp end parallel do", "enddo\n!$omp end parallel do\nendif", 1)
        normalized = normalized.replace("do i=" + name + ",n-4", f"if({name}<0) then\ndo i={name},n-4", 1)
        normalized = normalized.replace("enddo\nend subroutine", "enddo\nendif\nend subroutine", 1)
    if mutable_consumer:
        original = original.replace("integer::i\ndo i=2,n-1", f"integer::i\ninteger::{name}\n{name}=2\ndo i={name},n-1")
        consumer = f"""subroutine consumer(b,out,n,{name})
real(8),intent(in)::b(:)
real(8),intent(inout)::out(:)
integer,intent(in)::n,{name}
integer::i
do i={name},n-1
out(i)=b(i)
enddo
end subroutine
"""
        normalized = normalized.replace("end module", consumer + "end module")
    _source, _numerical, document = package(tmp_path, original=original, normalized=normalized)
    document["entries"][0]["parameters"].append({"name": name, "resource": "original::producer::" + name})
    if mutable_consumer:
        document["entries"].append({**document["entries"][0], "procedure": "original::consumer",
                                   "entry": "extracted::consumer", "parameters": [
                                       {"name": "b", "resource": "argument::b", "physical_origin": [0]},
                                       {"name": "out", "resource": "argument::out", "physical_origin": [0]},
                                       {"name": "n", "resource": "argument::n"},
                                       {"name": name, "resource": "original::consumer::" + name}]})
    captures = {root: FACT for root in ("argument::a", "argument::b", "argument::out", "settings::weights")}
    return generate(tmp_path, monkeypatch, original, captures=captures, numerical_sources=document)


@pytest.mark.parametrize("guarded", [False, True])
def test_normalized_local_parameter_queries_compile_as_fortran(tmp_path, monkeypatch, guarded):
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not fortran:
        pytest.skip("Fortran compiler unavailable")
    outputs, report = packaged_control(tmp_path, monkeypatch, name="lane_lower",
                                       declaration="integer,parameter::lane_lower=seed-4", dependency=True, guarded=guarded)
    assert report["automatic_estimate_available"]
    build = tmp_path / "build"
    for name, value in outputs.items():
        path = build / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    for role in ("common_runtime", "shared_entry", "original_source"):
        for item in report["build_sources"]:
            if item["role"] == role and item["language"] == "fortran":
                path = build / item["path"]
                result = subprocess.run([fortran, "-std=f2018", "-fopenmp", "-c", str(path),
                                         "-o", str(path.with_suffix(".o"))], cwd=build,
                                        capture_output=True, text=True, timeout=60)
                assert result.returncode == 0, result.stdout + result.stderr


def test_generated_planning_owner_and_queries_compile_as_fortran(tmp_path, monkeypatch):
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not fortran:
        pytest.skip("Fortran compiler unavailable")
    outputs, report = generate(tmp_path, monkeypatch)
    for name, value in outputs.items():
        target = tmp_path / "build" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(value)
    build = tmp_path / "build"
    sources = report["build_sources"]
    ordered = [item for role in ("common_runtime", "shared_entry", "original_source")
               for item in sources if item["role"] == role and item["language"] == "fortran"]
    for item in ordered:
        target = build / Path(item["path"]).with_suffix(".o")
        result = subprocess.run([fortran, "-std=f2018", "-fopenmp", "-c", str(build / item["path"]),
                                 "-o", str(target)], cwd=build, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
