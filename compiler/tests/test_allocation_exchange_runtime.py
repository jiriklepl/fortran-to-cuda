"""Generated descriptor moves retain pending physical writes and untouched halos."""
import os
import re
import shutil
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_allocation_exchange import SOURCE
from compiler.tests.test_graph_source_owner import FACT
from compiler.tests.test_source_scopes import run

DRIVER = """program exercise
use fields
implicit none
real(8),allocatable::a(:),b(:)
integer::i
allocate(a(-1:9),b(-3:10))
do i=-1,9
 a(i)=real(i+5,8)
enddo
b=-999
call advance(a,b,8)
if(lbound(a,1)/=-3.or.ubound(a,1)/=10)error stop 'first descriptor'
if(lbound(b,1)/=-1.or.ubound(b,1)/=9)error stop 'second descriptor'
do i=-3,10
 if(i>=1.and.i<=8)then
  if(a(i)/=3*real(i+5,8))error stop 'GPU writes follow allocation'
 else
  if(a(i)/=-999)error stop 'first halo'
 endif
enddo
do i=-1,9
 if(b(i)/=real(i+5,8))error stop 'readonly original allocation'
enddo
deallocate(a,b)
print *, 'SWAP_FIELDS_OK'
end program
"""


@pytest.fixture(scope="module")
def swap_binaries(tmp_path_factory):
    nvcc, host = shutil.which("nvcc"), shutil.which("g++-14") or shutil.which("g++")
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not all((nvcc, host, fortran)):
        pytest.skip("CUDA/Fortran toolchain unavailable")
    directory = tmp_path_factory.mktemp("source_swap")
    source = directory / "original.f90"
    source.write_text(SOURCE)
    facts = {"schema_version": 1, "participation": "serial", "sources": {str(source): sha256(source.read_bytes()).hexdigest()},
             "captures": {"argument::a": dict(FACT), "argument::b": dict(FACT)}}
    outputs, manifest = ScopeBuilder([source], "fields::advance", facts=facts, options=CompilerOptions(),
                                    config=OffloadConfig(policy="sections", scope_execution="graph")).run()
    assert not manifest["scopes"][0]["boundaries"]
    build = directory / "build"
    build.mkdir()
    for name, text in outputs.items():
        target = build / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    objects = []
    for role in ("common_runtime", "shared_entry", "original_source"):
        for index, artifact in enumerate(manifest["build_sources"]):
            if artifact["role"] != role:
                continue
            target = build / artifact["path"]
            assert sha256(target.read_bytes()).hexdigest() == manifest["artifacts_sha256"][artifact["path"]]
            obj = build / f"object-{index}.o"
            command = ([nvcc, "-std=c++17", "-ccbin", host, "-arch=sm_86", "-Xcompiler=-fopenmp", "-I", str(build)]
                       if artifact["language"] == "cuda" else
                       [fortran, "-std=f2018", "-fopenmp", "-fcheck=all", "-J", str(build), "-I", str(build)])
            run([*command, "-c", str(target), "-o", str(obj)], cwd=build)
            objects.append(str(obj))
    driver = build / "driver.f90"
    driver.write_text(DRIVER)
    candidate, native = build / "candidate", build / "native"
    run([fortran, "-std=f2018", "-fopenmp", "-fcheck=all", str(driver), *objects,
         "-L/usr/local/cuda/lib64", "-Wl,-rpath,/usr/local/cuda/lib64", "-lcudart", "-lstdc++", "-o", str(candidate)], cwd=build)
    run([fortran, "-std=f2018", "-fopenmp", "-fcheck=all", str(source), str(driver), "-o", str(native)], cwd=build)
    return build, candidate, native


@pytest.mark.native
def test_generated_swap_preserves_original_native_recovery(swap_binaries):
    build, candidate, native = swap_binaries
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "CUDA_VISIBLE_DEVICES": "-1",
                   "FORT_RUNTIME_TRACE": "1"}
    for executable in (native, candidate):
        result = run([str(executable)], cwd=build, env=environment)
        assert "SWAP_FIELDS_OK" in result.stdout
        assert "FORT_SCOPED launch " not in result.stderr


@pytest.mark.cuda
def test_generated_swap_keeps_physical_mirrors_and_queued_readers(swap_binaries):
    build, candidate, _ = swap_binaries
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "FORT_RUNTIME_TRACE": "1"}
    result = run([str(candidate)], cwd=build, env=environment)
    (build / "gpu.stdout").write_text(result.stdout)
    (build / "gpu.stderr").write_text(result.stderr)
    assert "SWAP_FIELDS_OK" in result.stdout
    lines = result.stderr.splitlines()
    assert sum(line.startswith("FORT_SCOPED launch ") for line in lines) == 2, result.stderr
    assert sum(line.startswith("FORT_SCOPED identity_swap ") for line in lines) == 1, result.stderr
    for event in ("upload", "download"):
        assert sum(int(re.search(r"\bbytes=(\d+)\b", line)[1]) for line in lines
                   if line.startswith("FORT_SCOPED " + event + " ")) == 64, result.stderr
