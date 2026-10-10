"""A one-shot window publishes before allocation movement and loop backedges."""

from __future__ import annotations

import json
import os
import shutil
from hashlib import sha256

import pytest

from compiler.tests.test_reached_window_owner import emit_window
from compiler.tests.test_scoped_batch_sources_cuda import actual_profile
from compiler.tests.test_source_scopes import run

DRIVER = """program verify
use renamed_stage,only:step,prefix_visits,tail_visits,prefix_total
implicit none
real(8),allocatable::a(:),b(:)
integer,parameter::sizes(3)=[0,8,4096],counts(3)=[0,1,3]
integer::shape,stages,iteration,n,i,unit
open(newunit=unit,file='fields.bin',access='stream',form='unformatted',status='replace')
do shape=1,size(sizes)
 n=sizes(shape)
 do stages=1,size(counts)
  allocate(a(-7:n+2),b(-7:n+2))
  do iteration=1,2
   do i=lbound(a,1),ubound(a,1)
    a(i)=real(i+16,8)*0.125_8+real(iteration,8)
   enddo
   b=-77.0_8
   call step(a,b,n,counts(stages))
   write(unit)prefix_visits,tail_visits,prefix_total,lbound(a),ubound(a),a,b
  enddo
  deallocate(a,b)
 enddo
enddo
close(unit)
end program
"""


@pytest.fixture(scope="module", params=("ordinary", "resource_fallback"))
def window_binaries(tmp_path_factory, request):
    nvcc = shutil.which("nvcc")
    host = shutil.which("g++-14") or shutil.which("g++")
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not all((nvcc, host, fortran)):
        pytest.skip("CUDA, C++ and Fortran toolchain required")
    directory = tmp_path_factory.mktemp("one_shot_window")
    architecture = actual_profile(directory, nvcc, host)["hardware"]["compute_capability"].replace(".", "")
    resource_fallback = request.param == "resource_fallback"
    original, outputs, report, _owner = emit_window(directory, device_budget=0 if resource_fallback else None)
    for name, text in outputs.items():
        artifact = directory / name
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(text)
    objects = []
    flags = [fortran, "-O3", "-cpp", "-std=f2018", "-fopenmp", "-fcheck=all,array-temps"]
    for role in ("common_runtime", "shared_entry", "original_source"):
        for item in report["build_sources"]:
            if item["role"] != role:
                continue
            artifact = directory / item["path"]
            target = artifact.with_suffix(".o")
            compiler = ([nvcc, "-O2", "-std=c++17", "-ccbin", host, "-arch=sm_" + architecture,
                         "-Xcompiler=-fopenmp"] if item["language"] == "cuda" else flags)
            run([*compiler, "-I", str(directory), "-c", str(artifact), "-o", str(target)], cwd=directory)
            objects.append(str(target))
    driver = directory / "driver.f90"
    driver.write_text(DRIVER)
    candidate, native = directory / "candidate", directory / "native"
    cuda_root = os.path.dirname(os.path.dirname(os.path.realpath(nvcc)))
    cuda_lib = os.path.join(cuda_root, "lib64")
    run([*flags, str(driver), *objects, "-L" + cuda_lib, "-Wl,-rpath," + cuda_lib,
         "-lcudart", "-lstdc++", "-o", str(candidate)], cwd=directory)
    run([*flags, str(original), str(driver), "-o", str(native)], cwd=directory)
    run([str(native)], cwd=directory)
    expected = (directory / "fields.bin").read_bytes()
    return directory, candidate, expected, report, resource_fallback


@pytest.mark.cuda
@pytest.mark.parametrize("disabled", [False, True])
def test_one_shot_window_matches_complete_native_arrays_and_all_original_visits(window_binaries, disabled):
    directory, candidate, expected, report, resource_fallback = window_binaries
    env = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "FORT_RUNTIME_TRACE": "1"}
    if disabled:
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    result = run([str(candidate)], cwd=directory, env=env)
    actual = (directory / "fields.bin").read_bytes()
    assert actual == expected
    assert "array temporary" not in result.stderr.lower()
    # Two kernels run only on the first traversal of each nonempty window.
    # Extra original stages still execute once, after mandatory publication and
    # MOVE_ALLOC. Zero-trip enclosing loops never activate the window at all.
    launches = result.stderr.count("FORT_SCOPED launch ")
    assert launches == (0 if disabled or resource_fallback else 16)
    contexts = result.stderr.count("FORT_SCOPED create ")
    # At most one context may be opened at each first reached traversal. A
    # failed pre-execution resource guard cannot retry on later backedges.
    assert contexts <= 12
    receipt = {"schema_version": 1, "disabled": disabled, "fields_bytes": len(actual),
               "native_fields_sha256": sha256(expected).hexdigest(),
               "candidate_fields_sha256": sha256(actual).hexdigest(),
               "launches": launches, "contexts_created": contexts, "resource_fallback": resource_fallback,
               "complete_arrays_and_halos_equal": True,
               "original_prefix_and_tail_counters_equal": True,
               "reached_window": report["scopes"][0]["reached_window"],
               "performance_sample": False}
    (directory / ("window-disabled.json" if disabled else "window-enabled.json")).write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
