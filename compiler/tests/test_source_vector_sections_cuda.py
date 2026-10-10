"""Imported fixed vectors execute through public source artifacts and guards."""

from __future__ import annotations

import os
import shutil
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_source_scopes import FACT, run
from compiler.tests.test_source_vector_sections import fixture


@pytest.mark.cuda
@pytest.mark.parametrize("precision", [4, 8])
@pytest.mark.parametrize("expression", ["weighted", "sqrt"])
def test_imported_windows_match_native_fields_and_keep_rounding_fallback(tmp_path, precision, expression):
    nvcc = shutil.which("nvcc")
    fc = shutil.which("gfortran-15") or shutil.which("gfortran")
    host = shutil.which("g++-14") or shutil.which("g++")
    if not nvcc or not fc or not host:
        pytest.skip("CUDA, Fortran and supported host C++ compilers required")
    _, _, _, paths = fixture(tmp_path, precision=precision)
    if expression == "sqrt":
        original = paths[1].read_text().replace("integer::i,j,k", f"integer::i,j,k\nreal({precision})::g(4)")
        original = original.replace("out(i,j,k)=sum(coeff*a(i-2:i+1,j,k))*gain",
                                    "g=sqrt(a(i-2:i+1,j,k))\nout(i,j,k)=g(1)")
        paths[1].write_text(original)
    facts = {"schema_version": 1, "participation": "serial",
             "captures": {"argument::a": FACT, "argument::out": FACT},
             "sources": {str(path): sha256(path.read_bytes()).hexdigest() for path in paths}}
    outputs, manifest = ScopeBuilder(paths, "renamed_windows::advance", facts=facts,
        options=CompilerOptions(), config=OffloadConfig("sections")).run()
    assert manifest["scope_count"] == 1, manifest["boundaries"]
    output = tmp_path / "generated"
    output.mkdir()
    for name, contents in outputs.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    build = tmp_path / "build"
    native = tmp_path / "native"
    build.mkdir()
    native.mkdir()
    driver = tmp_path / "driver.f90"
    driver.write_text(f"""program verify
use renamed_windows,only:advance
use ieee_arithmetic
use iso_fortran_env,only:error_unit
implicit none
real({precision}),allocatable::a(:,:,:),out(:,:,:)
integer::shape,n,i,j,k,repeat,unit
type(ieee_round_type)::rounding
open(newunit=unit,file='fields.bin',access='stream',form='unformatted',status='replace')
do shape=0,2
n=3*shape-2
allocate(a(-4:n+2,-2:5,0:5),out(-4:n+2,-2:5,0:5))
do k=0,5
do j=-2,5
do i=-4,n+2
a(i,j,k)=real(i+17*j+31*k,{precision})/32._{precision}
enddo
enddo
enddo
do repeat=1,3
out=-117._{precision}
if(repeat==2) call ieee_set_rounding_mode(ieee_up)
write(error_unit,*) 'BEGIN_CALL',shape,repeat
call advance(a,out,n)
write(error_unit,*) 'END_CALL',shape,repeat
call ieee_get_rounding_mode(rounding)
if(repeat==2.and.rounding/=ieee_up) error stop 'rounding mode changed'
call ieee_set_rounding_mode(ieee_nearest)
write(unit) a,out
enddo
deallocate(a,out)
enddo
close(unit)
print *,'FIELDS_OK'
end program
""")
    fflags = [fc, "-O3", "-fopenmp", "-fcheck=all"]
    cflags = [nvcc, "-O3", "--fmad=false", "-std=c++17", "-ccbin", host,
              "-arch=" + os.environ.get("FORT_TEST_CUDA_ARCH", "native"), "-Xcompiler=-fopenmp"]
    # Unchanged dependency modules remain supplied source inputs, just as in an
    # independent application build; the compiler emits replacement owners.
    run([*fflags, "-J", str(build), "-I", str(build), "-c", str(paths[0]), "-o", "constants.o"], cwd=build)
    objects = [str(build / "constants.o")]
    for role in ("common_runtime", "shared_entry", "original_source"):
        for item in manifest["build_sources"]:
            if item["role"] != role:
                continue
            path = output / item["path"]
            target = build / (str(len(objects)) + ".o")
            flags = cflags if item["language"] == "cuda" else [*fflags, "-J", str(build), "-I", str(build)]
            run([*flags, "-c", str(path), "-o", str(target)], cwd=build)
            objects.append(str(target))
    run([*fflags, "-J", str(build), "-I", str(build), "-c", str(driver), "-o", "driver.o"], cwd=build)
    run([nvcc, "-ccbin", host, "-Xcompiler=-fopenmp", *objects, "driver.o", "-lgfortran", "-o", "verify"], cwd=build)
    run([*fflags, *(str(path) for path in paths), str(driver), "-o", "verify"], cwd=native)
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "OMP_SCHEDULE": "static"}
    reference = run([str(native / "verify")], cwd=native, env=environment)
    checked = run([str(build / "verify")], cwd=build, env={**environment, "FORT_RUNTIME_TRACE": "1"})
    assert checked.stdout == reference.stdout
    assert (build / "fields.bin").read_bytes() == (native / "fields.bin").read_bytes()
    calls = checked.stderr.split("BEGIN_CALL")[1:]
    assert len(calls) == 9
    for ordinal, call in enumerate(calls):
        trace = call.split("END_CALL")[0]
        if ordinal < 3 or ordinal % 3 == 1:
            assert "FORT_SCOPED launch " not in trace
            assert "FORT_SCOPED h2d " not in trace
            assert "FORT_SCOPED d2h " not in trace
        else:
            assert "FORT_SCOPED launch " in trace
