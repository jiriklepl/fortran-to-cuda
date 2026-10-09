"""Native-toolchain checks for unordered operands and degenerate arithmetic."""
import math
import os
import shutil
import subprocess

import pytest

from compiler.driver.pipeline import prepare_function
from compiler.emission import generate_sources, read_common_header
from compiler.frontend import lower_file
from compiler.tests.test_language import compile_cuda_sources
from compiler.tests.test_native import cuda_device as cuda_device


@pytest.mark.native
@pytest.mark.parametrize("precision", [4, 8])
@pytest.mark.parametrize("backend", ["serial", "cuda"])
def test_ordered_minmax_and_degenerate_clamps(tmp_path, precision, backend, request):
    if backend == "cuda":
        request.getfixturevalue("cuda_device")
    fc = shutil.which("gfortran-15") or shutil.which("gfortran")
    cxx = shutil.which("g++")
    if not fc or not cxx:
        pytest.skip("requires native Fortran and C++ compilers")
    source = tmp_path / "numerical.f90"
    source.write_text(f"""module unordered_case
contains
subroutine entry(x,y,out)
real({precision}),intent(in)::x(:),y(:)
real({precision}),intent(out)::out(:,:)
integer::i
do i=1,size(x)
 out(i,1)=min(x(i),y(i))
 out(i,2)=max(x(i),y(i))
 out(i,3)=min(1._{precision},x(i))
 out(i,4)=max(0._{precision},x(i))
 out(i,5)=max(-1._{precision},min(1._{precision},x(i)))
 out(i,6)=max(0._{precision},x(i)/y(i)**2)
end do
end subroutine
end module
""")
    function, plan = prepare_function(lower_file(source, "entry"))
    generated = generate_sources(function, plan)
    (tmp_path / "implementation.cpp").write_text(generated.cpp)
    (tmp_path / "interface.f90").write_text(generated.fortran)
    (tmp_path / "common_functions.cuh").write_text(read_common_header())
    (tmp_path / "driver.f90").write_text(f"""program verify
use unordered_case
use ieee_arithmetic
implicit none
real({precision})::x(7),y(7),out(7,6),nan,inf
nan=ieee_value(0._{precision},ieee_quiet_nan)
inf=ieee_value(0._{precision},ieee_positive_inf)
x=[nan,-0._{precision},0._{precision},-1._{precision},1._{precision},inf,2._{precision}]
y=[1._{precision},0._{precision},-0._{precision},nan,inf,1._{precision},2._{precision}]
call entry(x,y,out)
print '(ES26.17E3)',out
end program
""")

    def run(command):
        result = subprocess.run(command, cwd=tmp_path, env=dict(os.environ, OMP_NUM_THREADS="4"),
                                capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout

    run([fc, "-O3", str(source), "driver.f90", "-o", "reference"])
    expected = [float(value) for value in run([str(tmp_path / "reference")]).split()]
    if backend == "cuda":
        compile_cuda_sources(tmp_path, function, plan)
        run([fc, "-O3", "-c", "interface.f90", "driver.f90"])
        run(["nvcc", "implementation.o", "interface.o", "driver.o", "-lgfortran", "-o", "generated"])
    else:
        run([cxx, "-std=c++17", "-O3", "-c", "implementation.cpp", "-o", "implementation.o"])
        run([fc, "-O3", "interface.f90", "driver.f90", "implementation.o", "-lstdc++", "-o", "generated"])
    actual = [float(value) for value in run([str(tmp_path / "generated")]).split()]
    assert len(actual) == len(expected)
    for index, (value, reference) in enumerate(zip(actual, expected, strict=True)):
        if math.isnan(reference):
            assert math.isnan(value), (index, value, reference)
        else:
            assert value == reference, (index, value, reference)
            if reference == 0:
                assert math.copysign(1, value) == math.copysign(1, reference), index
