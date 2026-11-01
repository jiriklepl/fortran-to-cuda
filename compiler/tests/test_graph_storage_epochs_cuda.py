"""Real repeated allocation epochs publish before original allocation exchange."""
import json
import os
import re
import shutil
import subprocess
from hashlib import sha256

import pytest

from compiler.tests.test_graph_storage_epochs import emit

DRIVER = """program exercise
use epochs
use iso_fortran_env,only:error_unit
implicit none
real(8),allocatable::left(:),right(:)
integer::case_index,n,i
write(error_unit,'(A)')'EPOCH_ZERO_BEGIN'
call advance(left,right,257,0)
write(error_unit,'(A)')'EPOCH_ZERO_END'
if(allocated(left).or.allocated(right))error stop 'zero trip inspected or allocated storage'
do case_index=1,3
select case(case_index)
case(1)
n=1
case(2)
n=7
case default
n=257
end select
write(error_unit,'(A,1X,I0)')'EPOCH_CASE_BEGIN',n
call advance(left,right,n,2)
write(error_unit,'(A,1X,I0)')'EPOCH_CASE_END',n
if(any(shape(left)/=[n]).or.any(shape(right)/=[n]))error stop 'shape changed'
if(lbound(left,1)/=-3.or.lbound(right,1)/=2)error stop 'original allocation exchange lost bounds'
do i=1,n
if(left(i-4)/=real(i,8))error stop 'full producer output'
if(right(i+1)/=2.0_8*real(i,8))error stop 'full consumer output'
enddo
deallocate(left,right)
enddo
print *, 'REPEATED_ORIGINAL_ALLOCATION_EPOCH_FIELDS_OK'
end program
"""


def checked(command, build, stem, env=None):
    result = subprocess.run(command, cwd=build, env=env, capture_output=True, text=True, timeout=120)
    (build / (stem + ".command.json")).write_text(json.dumps(command) + "\n")
    (build / (stem + ".stdout")).write_text(result.stdout)
    (build / (stem + ".stderr")).write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.fixture(scope="module")
def epoch_binaries(tmp_path_factory):
    directory = tmp_path_factory.mktemp("original_storage_epoch")
    nvcc, host = shutil.which("nvcc"), shutil.which("g++-14") or shutil.which("g++")
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not all((nvcc, host, fortran)):
        pytest.skip("CUDA and Fortran toolchains required")
    original, outputs, manifest = emit(directory)
    build = directory / "build"
    build.mkdir()
    for name, text in outputs.items():
        path = build / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    (build / "scope-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    objects = []
    for role in ("common_runtime", "shared_entry", "original_source"):
        for index, artifact in enumerate(manifest["build_sources"]):
            if artifact["role"] != role:
                continue
            path = build / artifact["path"]
            assert sha256(path.read_bytes()).hexdigest() == manifest["artifacts_sha256"][artifact["path"]]
            target = build / f"object-{index}.o"
            command = ([nvcc, "-std=c++17", "-ccbin", host, "-arch=sm_86", "-Xcompiler=-fopenmp",
                        *artifact["numerical_contract"]["required_cuda_options"], "-I", str(build)]
                       if artifact["language"] == "cuda" else
                       [fortran, "-fopenmp", "-fcheck=all", "-ffp-contract=off", "-J", str(build), "-I", str(build)])
            checked([*command, "-c", str(path), "-o", str(target)], build, f"compile-{index}")
            objects.append(str(target))
    driver = build / "driver.f90"
    driver.write_text(DRIVER)
    candidate, native = build / "candidate", build / "native"
    checked([fortran, "-fopenmp", "-fcheck=all", "-ffp-contract=off", "-I", str(build), str(driver), *objects,
             "-L/usr/local/cuda/lib64", "-Wl,-rpath,/usr/local/cuda/lib64", "-lcudart", "-lstdc++",
             "-o", str(candidate)], build, "link-candidate")
    checked([fortran, "-fopenmp", "-fcheck=all", "-ffp-contract=off", "-J", str(build),
             str(original), str(driver), "-o", str(native)], build, "link-native")
    return build, candidate, native


@pytest.mark.cuda
@pytest.mark.parametrize("hidden", [False, True], ids=["enabled", "hidden"])
def test_repeated_checked_epoch_uses_producer_definitions_and_keeps_unallocated_zero_trip(epoch_binaries, hidden):
    build, candidate, native = epoch_binaries
    env = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "FORT_RUNTIME_TRACE": "1"}
    env.pop("CUDA_VISIBLE_DEVICES", None)
    if hidden:
        env["CUDA_VISIBLE_DEVICES"] = ""
    mode = "hidden" if hidden else "enabled"
    reference = checked([str(native)], build, mode + "-native", env)
    result = checked([str(candidate)], build, mode + "-candidate", env)
    assert result.stdout == reference.stdout
    assert "REPEATED_ORIGINAL_ALLOCATION_EPOCH_FIELDS_OK" in result.stdout
    zero, = re.findall(r"EPOCH_ZERO_BEGIN\n(.*?)EPOCH_ZERO_END", result.stderr, re.S)
    assert "FORT_SCOPED domain_create " not in zero
    assert "FORT_SCOPED register " not in zero
    assert "FORT_SCOPED launch " not in zero
    cases = re.findall(r"EPOCH_CASE_BEGIN (\d+)\n(.*?)EPOCH_CASE_END \1", result.stderr, re.S)
    assert len(cases) == 3
    for size, events in cases:
        lines = events.splitlines()
        kernels = [line for line in lines if line.startswith("FORT_SCOPED launch ")]
        uploads = [line for line in lines if line.startswith("FORT_SCOPED upload ")]
        downloads = [line for line in lines if line.startswith("FORT_SCOPED download ")]
        assert not uploads, events
        if hidden:
            assert not kernels, events
            assert not downloads, events
        else:
            assert len(kernels) == 4, events
            assert len(downloads) == 4, events
            assert sum(int(re.search(r"\bbytes=(\d+)\b", line)[1]) for line in downloads) == 32*int(size)
