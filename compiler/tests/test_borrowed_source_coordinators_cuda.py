"""A calibrated reached child builds independently and preserves native fields."""

from hashlib import sha256
import json
import os
import shutil

import pytest

from compiler.tests.test_borrowed_source_coordinators import source
from compiler.tests.test_scoped_batch_sources_cuda import actual_profile
from compiler.tests.test_source_scopes import FACT, ROOT, generate, run


DRIVER = """program verify
use original,only:step
implicit none
real(8),allocatable::a(:),b(:),out(:)
integer,parameter::sizes(3)=[0,8,4096]
integer::shape,repeat,sign,n,i,unit
open(newunit=unit,file='fields.bin',access='stream',form='unformatted',status='replace')
do shape=1,size(sizes)
 allocate(a(-7:sizes(shape)-8),b(-7:sizes(shape)-8),out(-7:sizes(shape)-8))
 do repeat=1,2
  do sign=-1,1,2
   n=size(a)
   do i=lbound(a,1),ubound(a,1)
    a(i)=real(sign*(i+8),8)*0.25d0+repeat
   enddo
   b=-99.d0
   out=-101.d0
   call step(a,b,out,n)
   write(unit)n,a,b,out
  enddo
 enddo
 deallocate(a,b,out)
enddo
close(unit)
end program
"""


@pytest.fixture(scope="module")
def calibrated_child_binary(tmp_path_factory):
    nvcc, host = shutil.which("nvcc"), shutil.which("g++-14") or shutil.which("g++")
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not all((nvcc, host, fortran)):
        pytest.skip("CUDA/C++/Fortran toolchain unavailable")
    directory = tmp_path_factory.mktemp("calibrated_borrowed_child")
    checkout = directory / "independent-compiler"
    shutil.copytree(ROOT / "compiler", checkout / "compiler", ignore=shutil.ignore_patterns(
        "__pycache__", ".*cache", "CODE_MAP.md", "MEMORY_MODEL_PLAN.md"))
    profile = directory / "profile.json"
    # Favorable synthetic correctness costs, bound to the actual hardware and
    # tool identities, exercise AUTO; these are never performance evidence.
    profile.write_text(json.dumps(actual_profile(directory, nvcc, host)))
    original, output, manifest = generate(directory / "case", source(), mode="auto", profile=profile,
        checkout=checkout, facts={"schema_version": 1, "participation": "serial",
                                 "captures": {"argument::" + name: FACT for name in ("a", "b", "out")}})
    assert manifest["scopes"][0]["estimate_available"]
    assert len(manifest["borrowed_source_coordinators"]) == 1
    objects = []
    flags = [fortran, "-O3", "-std=f2018", "-fopenmp", "-fcheck=all,array-temps"]
    headers = tuple((name, digest) for name, digest in sorted(manifest["artifacts_sha256"].items())
                    if name.endswith((".h", ".hpp", ".cuh")))
    for role in ("common_runtime", "shared_entry", "original_source"):
        for item in manifest["build_sources"]:
            if item["role"] != role:
                continue
            artifact = output / item["path"]
            assert sha256(artifact.read_bytes()).hexdigest() == manifest["artifacts_sha256"][item["path"]]
            target = artifact.with_suffix(".o")
            compiler = ([nvcc, "-O2", "-std=c++17", "-ccbin", host, "-arch=sm_86", "-Xcompiler=-fopenmp"]
                        if item["language"] == "cuda" else flags)
            run([*compiler, "-I", str(output), "-c", str(artifact), "-o", str(target)], cwd=output)
            objects.append(str(target))
    driver = output / "driver.f90"
    driver.write_text(DRIVER)
    target, native = output / "candidate", output / "native"
    run([*flags, str(driver), *objects, "-L/usr/local/cuda/lib64", "-Wl,-rpath,/usr/local/cuda/lib64",
         "-lcudart", "-lstdc++", "-o", str(target)], cwd=output)
    run([*flags, str(original), str(driver), "-o", str(native)], cwd=output)
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE"}
    run([str(native)], cwd=output, env=environment)
    expected = (output / "fields.bin").read_bytes()
    (output / "artifact-evidence.json").write_text(json.dumps({"headers": headers,
        "runtime_id": manifest["runtime"]["runtime_id"], "profile": str(profile)}, indent=2))
    return target, output, expected


@pytest.mark.cuda
@pytest.mark.parametrize("disabled", [False, True])
def test_calibrated_auto_child_preserves_reached_native_fallback(calibrated_child_binary, disabled):
    target, output, expected = calibrated_child_binary
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "FORT_RUNTIME_TRACE": "1"}
    if disabled:
        environment["CUDA_VISIBLE_DEVICES"] = "-1"
    result = run([str(target)], cwd=output, env=environment)
    assert (output / "fields.bin").read_bytes() == expected
    assert "array temporary" not in result.stderr.lower()
    if disabled:
        assert "FORT_SCOPED launch " not in result.stderr
    else:
        assert "FORT_SCOPED launch " in result.stderr
