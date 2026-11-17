"""A consumer that applies the published build requirements reproduces unfused arithmetic bit for bit."""

from __future__ import annotations

import os
import shutil
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.numerical_contract import require_numerical_build_contract
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.tests.scoped_batch_sources_cuda_support import actual_profile
from compiler.tests.test_source_scopes import run

FACT = {"storage": "stable", "initialized": "whole", "escapes": False, "allocation_changes": False}


def _source(precision):
    return f"""module cancellation
implicit none
contains
subroutine advance(a,b,c,out,n)
real({precision}),intent(in)::a(:),b(:),c(:)
real({precision}),intent(inout)::out(:)
integer,intent(in)::n
integer::i
do i=2,n-1
out(i)=a(i)*b(i)+c(i)
enddo
end subroutine
end module
"""


def _driver(precision):
    # (1+e)*(1-e)-1 is exactly zero when the product is rounded before the
    # addition and -e**2 when the two operations are contracted into one.
    exponent = 13 if precision == 4 else 27
    return f"""program check
use cancellation
use iso_fortran_env,only:int32,int64
implicit none
real({precision})::a(6),b(6),c(6),out(6)
integer::i
a=2._{precision}
b=3._{precision}
c=4._{precision}
a(2)=1._{precision}+2._{precision}**(-{exponent})
b(2)=1._{precision}-2._{precision}**(-{exponent})
c(2)=-1._{precision}
a(3)=-a(2)
b(3)=b(2)
c(3)=1._{precision}
a(4)=-0._{precision}
b(4)=2._{precision}
c(4)=-0._{precision}
out=-1234._{precision}
call advance(a,b,c,out,6)
do i=1,6
print '(Z{precision * 2}.{precision * 2})',transfer(out(i),0_int{precision * 8})
enddo
end program
"""


@pytest.mark.cuda
@pytest.mark.parametrize("precision", [4, 8])
def test_published_requirements_keep_multiply_and_add_separate_on_the_device(tmp_path, precision):
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    nvcc = shutil.which("nvcc")
    host = shutil.which("g++-14") or shutil.which("g++")
    if not all((fortran, nvcc, host)):
        pytest.skip("CUDA, C++ and Fortran toolchain required")
    architecture = actual_profile(tmp_path, nvcc, host)["hardware"]["compute_capability"].replace(".", "")
    original = tmp_path / "cancellation.f90"
    original.write_text(_source(precision))
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(original): sha256(original.read_bytes()).hexdigest()},
             "captures": {"argument::" + name: dict(FACT) for name in ("a", "b", "c", "out")}}
    outputs, report = ScopeBuilder([original], "cancellation::advance", facts=facts, options=CompilerOptions(),
                                   config=OffloadConfig(policy="sections")).run()
    assert len(report["scopes"]) == 1
    for name, text in outputs.items():
        artifact = tmp_path / name
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(text)

    flags = [fortran, "-O3", "-cpp", "-std=f2018", "-fopenmp", "-ffp-contract=off"]
    objects, contracts = [], 0
    for role in ("common_runtime", "shared_entry", "original_source"):
        for item in report["build_sources"]:
            if item["role"] != role:
                continue
            artifact = tmp_path / item["path"]
            target = artifact.with_suffix(".o")
            if item["language"] == "cuda":
                # Only the published requirements stand between -O3 and a fused multiply-add.
                contract = item["numerical_contract"]
                require_numerical_build_contract(contract)
                contracts += 1
                command = [nvcc, "-O3", "-std=c++17", "-ccbin", host, "-arch=sm_" + architecture,
                           "-Xcompiler=-fopenmp", *contract["required_cuda_options"],
                           *("-Xcompiler=" + option for option in contract["required_host_options"])]
            else:
                command = flags
            run([*command, "-I", str(tmp_path), "-c", str(artifact), "-o", str(target)], cwd=tmp_path)
            objects.append(str(target))
    assert contracts
    driver = tmp_path / "driver.f90"
    driver.write_text(_driver(precision))
    native, candidate = tmp_path / "native", tmp_path / "candidate"
    run([*flags, str(original), str(driver), "-o", str(native)], cwd=tmp_path)
    run([*flags, str(driver), *objects, "-L/usr/local/cuda/lib64", "-Wl,-rpath,/usr/local/cuda/lib64",
         "-lcudart", "-lstdc++", "-o", str(candidate)], cwd=tmp_path)

    env = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "FORT_RUNTIME_TRACE": "1"}
    expected = run([str(native)], cwd=tmp_path, env=env).stdout.split()
    assert len(expected) == 6
    assert int(expected[1], 16) == int(expected[2], 16) == 0
    result = run([str(candidate)], cwd=tmp_path, env=env)
    assert any(line.startswith(("FORT_SCOPED launch ", "FORT_SCOPED graph_submit "))
               for line in result.stderr.splitlines()), result.stderr
    assert result.stdout.split() == expected
