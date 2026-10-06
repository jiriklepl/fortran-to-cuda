"""A positive sibling domain must not expose a protected inner bound."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from compiler.driver.options import CompilerOptions
from compiler.driver.pipeline import prepare_function
from compiler.emission import generate_sources, read_common_header
from compiler.frontend import lower_file
from compiler.offload.config import OffloadConfig


@pytest.mark.native
@pytest.mark.cuda
@pytest.mark.parametrize("collective", [False, True])
def test_cuda_query_retains_native_for_mixed_empty_domains_without_reading_inner_bound(tmp_path, collective):
    nvcc = shutil.which(os.environ.get("NVCC", "/usr/local/cuda/bin/nvcc"))
    fc = shutil.which("gfortran-15") or shutil.which("gfortran")
    host = shutil.which(os.environ.get("CUDAHOSTCXX", "g++-14"))
    if not nvcc or not fc or not host:
        pytest.skip("requires CUDA and GNU Fortran/C++")
    source = """module protected_domains
contains
subroutine advance(a,b,m,n,l)
real(8),intent(inout)::a(:,:),b(:)
integer,intent(in)::m,n,l
integer::i,j
do j=1,m
 do i=1,n
  a(i,j)=1.0_8
 end do
end do
do i=1,l
 b(i)=b(i)+1.0_8
end do
end subroutine
end module
"""
    (tmp_path / "input.f90").write_text(source)
    (tmp_path / "reference.f90").write_text(source.replace("module protected_domains", "module native_domains"))
    function, plan = prepare_function(lower_file(tmp_path / "input.f90", "advance"),
                                     options=CompilerOptions(gpu_policy="sections"))
    generated = generate_sources(function, plan, common_header="runtime.hpp",
                                 offload_config=OffloadConfig("sections", collective=collective))
    assert generated.offload["analysis"]["available"]
    assert len(generated.offload["analysis"]["units"]) == 2
    query = generated.offload["native_fallback_query"]
    (tmp_path / "runtime.hpp").write_text(read_common_header())
    (tmp_path / "generated.cu").write_text(generated.cuda)
    (tmp_path / "interface.f90").write_text(generated.fortran)
    (tmp_path / "guard.cpp").write_text("""#include <sys/mman.h>
#include <cstdlib>
extern "C" void *protected_bound() {
    void *page=mmap(nullptr,4096,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if(page==MAP_FAILED) std::abort();
    return page;
}
""")
    begin = "!$omp parallel num_threads(4) private(ready)" if collective else ""
    single = "!$omp single" if collective else ""
    end = "!$omp end single\n!$omp end parallel" if collective else ""
    (tmp_path / "driver.f90").write_text(f"""program verify
use protected_domains,only:{query}
use native_domains,only:native=>advance
use iso_c_binding
implicit none
interface
 function protected_bound() bind(C) result(address)
  import c_ptr
  type(c_ptr)::address
 end function
end interface
integer,pointer::inner
real(8)::a(1,1),b(1)
logical::ready
call c_f_pointer(protected_bound(),inner)
a=4.0_8
b=2.0_8
{begin}
ready={query}(a,b,0,inner,1)
if(ready) error stop 'unsafe mixed domain selected generated entry'
{single}
call native(a,b,0,inner,1)
{end}
if(any(a/=4.0_8).or.any(b/=3.0_8)) error stop 'native fallback differs'
print *, 'MIXED_PROTECTED_QUERY_PASS'
end program
""")
    environment = dict(os.environ, OMP_DYNAMIC="FALSE", FORT_OFFLOAD_TRACE="1", FORT_RUNTIME_TRACE="1")
    commands = (
        [host, "-O2", "-c", "guard.cpp"],
        [nvcc, "-O2", "-std=c++17", "-Xcompiler=-fopenmp", "-arch=sm_86", "-ccbin", host,
         "-c", "generated.cu", "-o", "generated.o"],
        [fc, "-O2", "-fopenmp", "-ffree-line-length-none", "-fcheck=all", "interface.f90", "reference.f90",
         "driver.f90", "generated.o", "guard.o", "-lstdc++", "-L" + str(Path(nvcc).resolve().parents[1] / "lib64"),
         "-lcudart", "-o", "verify"],
        [str(tmp_path / "verify")],
    )
    for command in commands:
        result = subprocess.run(command, cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
    assert "MIXED_PROTECTED_QUERY_PASS" in result.stdout
    assert "mode=native" in result.stderr
    assert "FORT_RUNTIME upload" not in result.stderr
    assert "FORT_RUNTIME kernel" not in result.stderr
