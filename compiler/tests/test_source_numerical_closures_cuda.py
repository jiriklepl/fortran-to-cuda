"""Guarded source closures agree with native fields across runtime tile changes."""

import os
import shutil

import pytest

from compiler.tests.test_source_numerical_closures import SOURCE
from compiler.tests.test_source_scopes import FACT, generate, run


DRIVER = """program check
use renamed_numerics,only:advance
use iso_fortran_env,only:error_unit
use ieee_arithmetic
use ieee_exceptions
implicit none
real(8),allocatable::a(:)
integer::shape,repeat,low,n,tile,i,unit,ordinal
logical::trapping
type(ieee_round_type)::rounding
open(newunit=unit,file='fields.bin',access='stream',form='unformatted',status='replace')
ordinal=0
do shape=0,3
 n=9*shape
 low=-7-shape
 allocate(a(low:low+n+7))
 do repeat=1,4
  tile=3
  if(repeat==2) tile=7
  if(repeat==3) tile=-3
  if(repeat==4) tile=1
  do i=low,low+n+7
   a(i)=real(i-low,8)*0.25d0+1
  enddo
  if(repeat==4) call ieee_set_rounding_mode(ieee_up)
  ordinal=ordinal+1
  write(error_unit,*) 'BEGIN_CALL',ordinal
  call advance(a,n,tile)
  write(error_unit,*) 'END_CALL',ordinal
  call ieee_get_rounding_mode(rounding)
  if(repeat==4.and.rounding/=ieee_up) error stop 'original rounding mode changed'
  call ieee_set_rounding_mode(ieee_nearest)
  call ieee_get_halting_mode(ieee_invalid,trapping)
  if(trapping) error stop 'host trap state changed'
  write(unit) n,tile,a
 enddo
 deallocate(a)
enddo
close(unit)
print *, 'FIELDS_OK'
end program
"""


@pytest.mark.cuda
def test_guarded_private_helper_source_matches_native_complete_fields_and_halos(tmp_path):
    nvcc = shutil.which("nvcc")
    host = shutil.which("g++-14") or shutil.which("g++")
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not nvcc or not host or not fortran:
        pytest.skip("CUDA, C++ and Fortran toolchains required")
    source = SOURCE.replace("do block=-3,n,tile", "i=0\n!$omp parallel do private(g,s,i,block) schedule(runtime)\ndo block=-3,n,tile").replace(
        "enddo\ncontains", "enddo\n!$omp end parallel do\ncontains")
    facts = {"schema_version": 1, "participation": "serial", "captures": {"argument::a": FACT}}
    original, output, manifest = generate(tmp_path / "formed", source, entry="advance", facts=facts)
    assert manifest["scope_count"] == 1, manifest["boundaries"]
    build = tmp_path / "build"
    build.mkdir()
    cflags = [nvcc, "-O2", "-std=c++17", "-ccbin", host, "-arch=sm_86", "-Xcompiler=-fopenmp"]
    fflags = [fortran, "-O3", "-std=f2018", "-fopenmp", "-fcheck=all,array-temps"]
    objects = []
    for role in ("common_runtime", "shared_entry", "original_source"):
        for item in manifest["build_sources"]:
            if item["role"] != role:
                continue
            path = output / item["path"]
            target = build / (str(len(objects)) + ".o")
            flags = cflags if item["language"] == "cuda" else [*fflags, "-J", str(build), "-I", str(build)]
            run([*flags, "-c", str(path), "-o", str(target)], cwd=build)
            objects.append(str(target))
    driver = build / "driver.f90"
    driver.write_text(DRIVER)
    binary = build / "check"
    run([*fflags, "-I", str(build), str(driver), *objects, "-L/usr/local/cuda/lib64",
         "-Wl,-rpath,/usr/local/cuda/lib64", "-lcudart", "-lstdc++", "-o", str(binary)], cwd=build)
    native = tmp_path / "native"
    native.mkdir()
    native_binary = native / "check"
    run([*fflags, str(original), str(driver), "-o", str(native_binary)], cwd=native)
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "OMP_SCHEDULE": "static"}
    reference = run([str(native_binary)], cwd=native, env=environment)
    measured = run([str(binary)], cwd=build, env={**environment, "FORT_RUNTIME_TRACE": "1"})
    assert measured.stdout == reference.stdout
    assert (build / "fields.bin").read_bytes() == (native / "fields.bin").read_bytes()
    assert "array temporary" not in measured.stderr.lower()
    calls = measured.stderr.split("BEGIN_CALL")[1:]
    assert len(calls) == 16
    for ordinal, call in enumerate(calls, 1):
        trace = call.split("END_CALL")[0]
        if ordinal % 4 in {0, 3}:  # Non-nearest rounding or nonpositive tile.
            assert "FORT_SCOPED launch " not in trace
            assert "FORT_SCOPED h2d " not in trace
            assert "FORT_SCOPED d2h " not in trace
        else:
            assert "FORT_SCOPED launch " in trace
