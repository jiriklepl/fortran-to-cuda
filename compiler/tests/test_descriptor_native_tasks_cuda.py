"""Descriptor-only tasks preserve queued producers, geometry and read mirrors."""

from __future__ import annotations

import json
import os
import re
import shutil
from unittest.mock import patch

import pytest

from compiler.numerical_contract import require_numerical_build_contract
from compiler.scopes import lexical
from compiler.tests.test_descriptor_native_tasks import SOURCE
from compiler.tests.test_graph_source_owner import emit
from compiler.tests.test_scoped_batch_sources_cuda import actual_profile
from compiler.tests.test_source_scopes import run

DRIVER = """program verify
use graph_fixture
implicit none
real(8),allocatable::a(:,:,:),b(:,:,:)
integer,allocatable::x(:)
integer::n,case,repeat,i,j,k,unit
integer,parameter::sizes(3)=[-4,1,5]
open(newunit=unit,file='fields.bin',access='stream',form='unformatted',status='replace')
do case=1,size(sizes)
n=sizes(case)
allocate(a(-3:n,-5:n-2,-7:n-4),b(-9:n-6,2:n+5,-11:n-8),x(1))
do repeat=1,2
do k=lbound(a,3),ubound(a,3)
do j=lbound(a,2),ubound(a,2)
do i=lbound(a,1),ubound(a,1)
a(i,j,k)=real(i+3*j+5*k+32*repeat,8)*0.125d0
enddo
enddo
enddo
b=-99.d0
x=repeat
call advance(a,b,x,n)
write(unit)n,lbound(a),ubound(a),lbound(b),ubound(b)
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
    directory = tmp_path_factory.mktemp("descriptor_native")
    architecture = actual_profile(directory, nvcc, host)["hardware"]["compute_capability"].replace(".", "")
    original, outputs, report = emit(directory / "inputs", SOURCE)
    with patch.object(lexical, "original_descriptor_scalar_operation", return_value=False):
        _, baseline_outputs, baseline_report = emit(directory / "inputs", SOURCE)
    assert len(report["scopes"][0]["planning_segments"]) == 2
    assert len(baseline_report["scopes"][0]["planning_segments"]) == 8
    for item in report["build_sources"]:
        if item["role"] != "original_source":
            assert outputs[item["path"]] == baseline_outputs[item["path"]]
    for name, text in outputs.items():
        artifact = directory / name
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(text)
    flags = [fc, "-O2", "-cpp", "-std=f2018", "-fopenmp", "-fcheck=all,array-temps", "-ffp-contract=off"]
    common, originals = [], []
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
    native, baseline, candidate = (directory / name for name in ("native", "baseline", "candidate"))
    link = ["-L/usr/local/cuda/lib64", "-Wl,-rpath,/usr/local/cuda/lib64", "-lcudart", "-lstdc++"]
    run([*flags, str(driver), *common, *originals, *link, "-o", str(candidate)], cwd=directory)
    baseline_objects = []
    for index, item in enumerate(baseline_report["build_sources"]):
        if item["role"] != "original_source":
            continue
        path = directory / f"baseline-{index}.f90"
        path.write_text(baseline_outputs[item["path"]])
        target = path.with_suffix(".o")
        run([*flags, "-I", str(directory), "-c", str(path), "-o", str(target)], cwd=directory)
        baseline_objects.append(str(target))
    run([*flags, str(driver), *common, *baseline_objects, *link, "-o", str(baseline)], cwd=directory)
    run([*flags, str(original), str(driver), "-o", str(native)], cwd=directory)
    return directory, native, baseline, candidate


def validation_count(text):
    return sum(json.loads(line.split(" evidence ", 1)[1])["event"] == "definition_validation"
               for line in text.splitlines() if line.startswith("FORT_SCOPED evidence "))


def traffic(text):
    return tuple(sum(int(re.search(r"\bbytes=(\d+)", line).group(1))
                     for line in text.splitlines() if line.startswith("FORT_SCOPED " + operation + " "))
                 for operation in ("upload", "download"))


@pytest.mark.cuda
@pytest.mark.parametrize("disabled", [False, True])
def test_descriptor_tasks_preserve_pending_kernels_and_complete_original_fields(binaries, disabled):
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
        path = directory / (executable.name + ("-disabled" if disabled else "") + ".log")
        path.write_text(result.stderr)
        traces.append(result.stderr)
    before, after = traces
    # Six original inquiries execute on every one of the six invocations,
    # including zero-size arrays. Their physical definitions never change.
    assert validation_count(before) - validation_count(after) == 36
    assert traffic(before) == traffic(after)
    assert after.count("FORT_SCOPED create ") <= before.count("FORT_SCOPED create ")
    if disabled:
        assert traffic(after) == (0, 0)
        assert "FORT_SCOPED launch " not in after
        assert "FORT_SCOPED graph_submit " not in after
    else:
        logical = after.count("FORT_SCOPED launch ") + sum(
            int(re.search(r"\bbytes=(\d+)", line).group(1)) for line in after.splitlines()
            if line.startswith("FORT_SCOPED graph_submit "))
        assert logical == 8
