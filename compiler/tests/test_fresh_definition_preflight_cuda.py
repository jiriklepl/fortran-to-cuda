"""Fresh coarse rejection preserves original work and later GPU epochs."""

from __future__ import annotations

import json
import os
import shutil
from unittest.mock import patch

import pytest

from compiler.numerical_contract import require_numerical_build_contract
from compiler.scopes import registration
from compiler.tests.test_fresh_definition_preflight import SOURCE_READ
from compiler.tests.test_graph_source_owner import FACT, emit
from compiler.tests.test_graph_source_owner_cuda import OPAQUE_SOURCE
from compiler.tests.test_scoped_batch_sources_cuda import actual_profile
from compiler.tests.test_source_scopes import run

DRIVER = """program verify
use graph_fixture
implicit none
real(8),allocatable::a(:),b(:)
integer,allocatable::x(:)
integer::n,case,repeat,i,unit
integer,parameter::sizes(2)=[1,8]
open(newunit=unit,file='fields.bin',access='stream',form='unformatted',status='replace')
do case=1,size(sizes)
n=sizes(case)
allocate(a(-4:n+3),b(-7:n),x(-11:n-4))
do repeat=1,2
a=[(real(i+7,8)*0.125d0+repeat,i=lbound(a,1),ubound(a,1))]
b=-99.d0
x=[(i,i=lbound(x,1),ubound(x,1))]
call advance(a,b,x,n)
write(unit)n,lbound(a),ubound(a),lbound(b),ubound(b),lbound(x),ubound(x)
write(unit)a,b,x
enddo
deallocate(a,b,x)
enddo
close(unit)
end program
"""


@pytest.fixture(scope="module")
def binaries(tmp_path_factory):
    nvcc = shutil.which("nvcc")
    host = shutil.which("g++-14") or shutil.which("g++")
    fc = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not all((nvcc, host, fc)):
        pytest.skip("CUDA, C++ and Fortran toolchain required")
    directory = tmp_path_factory.mktemp("fresh_definition")
    architecture = actual_profile(directory, nvcc, host)["hardware"]["compute_capability"].replace(".", "")
    # The first coarse read registers table under pristine facts; opaque closes
    # that owning context. The second read sees the same stale slot and must
    # execute its original scalar update once, before later GPU work resumes.
    source = SOURCE_READ.replace("call opaque(a,n)\nm=table(table(n))",
                                 "call opaque(a,n)\nm=m+table(table(n))")
    source = source.replace("subroutine advance(a,b,x,n)", "subroutine helper(a,b,x,n)")
    source = source.replace("end module", """subroutine advance(a,b,x,n)
real(8),intent(inout)::a(:),b(:)
integer,intent(inout)::x(:)
integer,intent(in)::n
call helper(a,b,x,n)
end subroutine
end module""")
    original, outputs, report = emit(directory / "input", source, captures={"graph_fixture::table": FACT})
    with patch.object(registration, "fresh_undefined_read_guard", return_value=()):
        _, baseline_outputs, baseline_report = emit(directory / "input", source,
                                                    captures={"graph_fixture::table": FACT})
    for item in report["build_sources"]:
        if item["role"] != "original_source":
            assert outputs[item["path"]] == baseline_outputs[item["path"]]
    for name, text in outputs.items():
        artifact = directory / name
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(text)
    flags = [fc, "-O2", "-cpp", "-std=f2018", "-fopenmp", "-fcheck=all,array-temps", "-ffp-contract=off"]
    common = []
    originals = []
    for role in ("common_runtime", "shared_entry", "original_source"):
        for item in report["build_sources"]:
            if item["role"] != role:
                continue
            artifact = directory / item["path"]
            target = artifact.with_suffix(".o")
            if item["language"] == "cuda":
                contract = item["numerical_contract"]
                require_numerical_build_contract(contract)
                compiler = [nvcc, "-O2", "-std=c++17", "-ccbin", host, "-arch=sm_"+architecture,
                            "-Xcompiler=-fopenmp", *contract["required_cuda_options"],
                            *("-Xcompiler="+option for option in contract["required_host_options"])]
            else:
                compiler = flags
            run([*compiler, "-I", str(directory), "-c", str(artifact), "-o", str(target)], cwd=directory)
            (originals if role == "original_source" else common).append(str(target))
    driver = directory / "driver.f90"
    driver.write_text(DRIVER)
    opaque = directory / "opaque.f90"
    opaque.write_text(OPAQUE_SOURCE)
    native, candidate, baseline = (directory / name for name in ("native", "candidate", "baseline"))
    link = ["-L/usr/local/cuda/lib64", "-Wl,-rpath,/usr/local/cuda/lib64", "-lcudart", "-lstdc++"]
    run([*flags, str(driver), str(opaque), *common, *originals, *link, "-o", str(candidate)], cwd=directory)
    baseline_objects = []
    for index, item in enumerate(baseline_report["build_sources"]):
        if item["role"] != "original_source":
            continue
        path = directory / f"baseline-{index}.f90"
        path.write_text(baseline_outputs[item["path"]])
        target = path.with_suffix(".o")
        run([*flags, "-I", str(directory), "-c", str(path), "-o", str(target)], cwd=directory)
        baseline_objects.append(str(target))
    run([*flags, str(driver), str(opaque), *common, *baseline_objects, *link, "-o", str(baseline)], cwd=directory)
    run([*flags, str(original), str(driver), str(opaque), "-o", str(native)], cwd=directory)
    return directory, native, baseline, candidate


@pytest.mark.cuda
@pytest.mark.parametrize("disabled", [False, True])
def test_avoiding_doomed_context_keeps_stale_child_slots_and_later_gpu_correct(binaries, disabled):
    directory, native, baseline, candidate = binaries
    env = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "FORT_RUNTIME_TRACE": "1"}
    if disabled:
        env["CUDA_VISIBLE_DEVICES"] = ""
    run([str(native)], cwd=directory, env=env)
    expected = (directory / "fields.bin").read_bytes()
    traces = []
    for executable in (baseline, candidate):
        result = run([str(executable)], cwd=directory, env=env)
        assert (directory / "fields.bin").read_bytes() == expected
        assert "array temporary" not in result.stderr.lower()
        (directory / (executable.name + ("-disabled" if disabled else "") + ".log")).write_text(result.stderr)
        traces.append(result.stderr)
    before, after = traces
    def failures(text):
        return [json.loads(line.split(" evidence ", 1)[1]) for line in text.splitlines()
                if line.startswith("FORT_SCOPED evidence ") and '"event":"definition_validation"' in line
                and '"status":7' in line]
    assert len(failures(after)) < len(failures(before))
    assert after.count("FORT_SCOPED create ") < before.count("FORT_SCOPED create ")
    if disabled:
        assert "FORT_SCOPED launch " not in after
        assert "FORT_SCOPED graph_submit " not in after
        assert "FORT_SCOPED upload " not in after
    else:
        # First and later numerical groups both execute; the guard never
        # turns a failed source preparation into permanent native execution.
        assert after.count("FORT_SCOPED launch ") + after.count("FORT_SCOPED graph_submit ") >= 8
