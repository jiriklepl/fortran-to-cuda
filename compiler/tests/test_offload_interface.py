"""The Fortran query bridge may forward a protected bound without reading it."""

import shutil
import subprocess

import pytest

from compiler.driver.options import CompilerOptions
from compiler.driver.pipeline import prepare_function
from compiler.emission import generate_sources, read_common_header
from compiler.frontend import lower_file
from compiler.offload.config import OffloadConfig


@pytest.mark.native
def test_native_query_bridge_does_not_read_guarded_inner_bound(tmp_path):
    fortran = shutil.which("gfortran")
    cxx = shutil.which("g++")
    if not fortran or not cxx:
        pytest.skip("requires GNU Fortran and C++")
    source = tmp_path / "input.f90"
    source.write_text("""module protected_query
contains
subroutine advance(a,n,m,scale)
real(8),intent(inout)::a(:,:)
integer,intent(in)::n,m
real(8),intent(in)::scale
integer::i,j
do j=1,n
 do i=1,m
  a(i,j)=a(i,j)+scale
 end do
end do
end subroutine
end module
""")
    function, plan = prepare_function(lower_file(source, "advance"), options=CompilerOptions(gpu_policy="sections"))
    generated = generate_sources(function, plan, offload_config=OffloadConfig("sections"))
    query = generated.offload["native_fallback_query"]
    (tmp_path / "common_functions.cuh").write_text(read_common_header())
    (tmp_path / "generated.cpp").write_text(generated.cpp)
    (tmp_path / "generated.f90").write_text(generated.fortran)
    (tmp_path / "guard.cpp").write_text("""#include <sys/mman.h>
#include <cstdlib>
extern "C" void *protected_bound() {
    void *page = mmap(nullptr, 4096, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (page == MAP_FAILED) std::abort();
    return page;
}
""")
    (tmp_path / "driver.f90").write_text(f"""program verify
use protected_query
use iso_c_binding
implicit none
interface
 function protected_bound() bind(C) result(address)
  import c_ptr
  type(c_ptr) :: address
 end function
end interface
integer, pointer :: inner
real(8), pointer :: scale
real(8) :: a(1,1)
call c_f_pointer(protected_bound(),inner)
call c_f_pointer(protected_bound(),scale)
a=4.0_8
if ({query}(a,0,inner,scale)) error stop 'CPU query must select native'
if (a(1,1)/=4.0_8) error stop 'query modified output'
print *, 'PROTECTED_QUERY_PASS'
end program
""")
    for argv in (
        [cxx, "-O2", "-std=c++17", "-fopenmp", "-c", "generated.cpp", "guard.cpp"],
        [fortran, "-O2", "-fopenmp", "-fcheck=all", "-ffree-line-length-none", "generated.f90",
         "driver.f90", "generated.o", "guard.o", "-lstdc++", "-o", "verify"],
        [str(tmp_path / "verify")],
    ):
        result = subprocess.run(argv, cwd=tmp_path, text=True, capture_output=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
    assert "PROTECTED_QUERY_PASS" in result.stdout
