"""An independent consumer applies public requirements to cancellation work."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from compiler.numerical_contract import require_explicit_cuda_environment, require_numerical_build_contract


@pytest.mark.cuda
@pytest.mark.parametrize("precision", [4, 8])
def test_public_cuda_contract_preserves_separate_arithmetic_and_complete_output(tmp_path, precision):
    fc = shutil.which("gfortran-15") or shutil.which("gfortran")
    nvcc = shutil.which("nvcc")
    host = shutil.which("g++-14") or shutil.which("g++")
    if not all((fc, nvcc, host)):
        pytest.skip("native Fortran and CUDA toolchain required")
    require_explicit_cuda_environment()
    source = tmp_path / "source.f90"
    source.write_text(f"""module renamed_cancellation
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
""")
    exponent = 13 if precision == 4 else 27
    (tmp_path / "driver.f90").write_text(f"""program check
use renamed_cancellation
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
""")
    env = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE",
           "PYTHONPATH": str(Path(__file__).resolve().parents[2])}

    commands = []

    def run(argv, *, trace=False):
        effective_env = {**env, **({"FORT_RUNTIME_TRACE": "1"} if trace else {})}
        command_id = len(commands)
        commands.append({"command": argv, "cwd": str(tmp_path),
                         "environment_overrides": {"FORT_RUNTIME_TRACE": "1"} if trace else {}})
        (tmp_path / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        result = subprocess.run(argv, cwd=tmp_path, env=effective_env, text=True, capture_output=True, timeout=90)
        (tmp_path / f"command-{command_id:02}.stdout").write_text(result.stdout)
        (tmp_path / f"command-{command_id:02}.stderr").write_text(result.stderr)
        (tmp_path / f"command-{command_id:02}.result.json").write_text(
            json.dumps({"exit_code": result.returncode}) + "\n")
        assert result.returncode == 0, result.stdout + result.stderr
        if trace:
            assert "FORT_RUNTIME kernel" in result.stderr.splitlines()
        return result.stdout

    # The source/reference both retain the unfused multiply/add tree. The
    # cancellation inputs are exactly representable and distinguish FMA.
    run([fc, "-O3", "-ffp-contract=off", "source.f90", "driver.f90", "-o", "native"])
    expected = run([str(tmp_path / "native")]).split()
    assert int(expected[1], 16) == int(expected[2], 16) == 0
    report = json.loads(run([sys.executable, "-m", "compiler", "-i", str(source), "-k", "advance",
                            "--json", "--output-dir", str(tmp_path)]))
    unit = next(item for item in report["build_sources"] if item["language"] == "cuda")
    contract = unit["numerical_contract"]
    require_numerical_build_contract(contract)
    required = [*contract["required_cuda_options"],
                *("-Xcompiler=" + item for item in contract["required_host_options"])]
    run([nvcc, "-O3", "-std=c++17", "-arch=" + os.environ.get("FORT_TEST_CUDA_ARCH", "native"),
         "-ccbin", host, "-Xcompiler=-fopenmp", *required, "-c", unit["path"], "-o", "implementation.o"])
    run([fc, "-O3", "-fopenmp", "-c", "generated_interface.f90", "driver.f90"])
    run([nvcc, "-ccbin", host, "-Xcompiler=-fopenmp", "implementation.o", "generated_interface.o", "driver.o",
         "-lgfortran", "-o", "generated"])
    assert run([str(tmp_path / "generated")], trace=True).split() == expected
