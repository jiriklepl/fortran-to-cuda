"""Compare proved wide addressing with original Fortran on native backends."""

from __future__ import annotations

import pytest

from compiler.driver.options import CompilerOptions
from compiler.driver.pipeline import prepare_function
from compiler.emission import generate_sources, read_common_header
from compiler.frontend import lower_file
from compiler.tests.test_language import compile_cuda_sources, run_reference_and_cpp
from compiler.tests.test_language_cuda import CUDA_RUNTIME
from compiler.tests.test_native import _run, _tool
from compiler.tests.test_native import cuda_device as cuda_device
from compiler.tests.test_sessions import _simulate_cuda

MIXED_SOURCE = """! kernels
module addressing_case
contains
! kernel
subroutine entry(a,out,source,index,lo,hi,stride,inner_stride,divisor)
integer,intent(inout)::a(:,:,:,:,:),out(:)
integer,intent(in)::source(:),index(:)
integer,intent(in)::lo,hi,stride,inner_stride,divisor
integer::i,j,k,p,q
do i=1,size(out,1)
  out(i)=source(index(max(1,min(size(index,1),abs(index(i)))))) &
    +source(max(1,min(size(source,1),abs(i)))) &
    +source(10+(-i)/2)+max(i,3)-min(i,2)+abs(i-4)+(i*3)/2
enddo
do j=lo,hi,stride
  do i=1,size(a,1),2
    do k=1,size(a,3)/divisor
      do p=1,2
        do q=1,2,inner_stride
          a(i,j,k,p,q)=i+10*j+100*k+1000*p+10000*q+out(min(i,size(out,1)))
        enddo
      enddo
    enddo
  enddo
enddo
end subroutine
end module
"""

MIXED_DRIVER = """program main
use addressing_case
implicit none
integer::a(17,5,3,2,2),out(17),source(17),index(17),i
a=-7
out=-9
source=[(11*i,i=1,17)]
index=[(18-i,i=1,17)]
call entry(a,out,source,index,5,1,-2,1,1)
print *,a,out
call entry(a,out,source,index,1,5,2,1,1)
print *,a,out
! Neither the zero inner stride nor division by zero is reached.
call entry(a,out,source,index,5,1,1,0,0)
print *,a,out
end program
"""

BOUNDARY_SOURCE = """! kernels
module addressing_case
contains
! kernel
subroutine entry(a,lo,hi)
integer,intent(inout)::a(:,:)
integer,intent(in)::lo,hi
integer::i
do i=hi,hi-2,-1
  a(hi-i+1,1)=(i-hi)*3+abs(i-hi)+max(i-hi,-1)
enddo
do i=lo,lo+2
  a(i-lo+1,2)=(i-lo)*5+min(i-lo,1)
enddo
do i=2147483647,2147483647,-1
  a(i-2147483646,1)=i
enddo
do i=(-2147483647-1),(-2147483647-1),1
  a(i+2147483647+2,2)=i
enddo
end subroutine
end module
"""

BOUNDARY_DRIVER = """program main
use addressing_case
implicit none
integer::a(3,2),lo,hi
lo=-huge(0)-1
hi=huge(0)
a=-99
! Both loops move inward: their final induction updates also remain representable.
call entry(a,lo,hi)
print *,a
call entry(a,-10,10)
print *,a
end program
"""


def _prepare(tmp_path, source, **options):
    path = tmp_path / "source.f90"
    path.write_text(source)
    return prepare_function(lower_file(path, "entry"), options=CompilerOptions(**options))


def _compare(tmp_path, function, plan, driver, backend, *, generated_driver=None):
    """UBSan checks emitted integer evaluation; Fortran checks every array access."""
    for tool in ("gfortran", "g++"):
        _tool(tool)
    generated = generate_sources(function, plan)
    cpp = generated.cpp
    if backend == "cuda-simulated":
        (tmp_path / "cuda_runtime.h").write_text(CUDA_RUNTIME)
        cpp = _simulate_cuda(generated.cuda)
    (tmp_path / "implementation.cpp").write_text(cpp)
    (tmp_path / "interface.f90").write_text(generated.fortran)
    (tmp_path / "common_functions.cuh").write_text(read_common_header())
    (tmp_path / "driver.f90").write_text(driver)
    _run(["gfortran", "-fcheck=all", function.source, "driver.f90", "-o", "reference"], tmp_path)
    expected = _run(["./reference"], tmp_path).stdout.split()
    if generated_driver is not None:
        (tmp_path / "driver.f90").write_text(generated_driver)
    flags = ["-fsanitize=undefined", "-fno-sanitize-recover=undefined"]
    if backend == "openmp":
        flags.append("-fopenmp")
    _run(["g++", "-std=c++17", "-O1", *flags, "-c", "implementation.cpp", "-o", "implementation.o"], tmp_path)
    _run(
        ["gfortran", *flags, "-fcheck=all", "interface.f90", "driver.f90", "implementation.o", "-lstdc++", "-o", "run"],
        tmp_path,
    )
    actual = _run(["./run"], tmp_path).stdout.split()
    assert [int(value) for value in actual] == [int(value) for value in expected]


PROFILES = [
    pytest.param({"opt_level": 1, "indexing": "source", "schedule": "source"}, id="source-indexing"),
    pytest.param({"opt_level": 0, "indexing": "auto", "schedule": "auto"}, id="explicit-auto-at-level-zero"),
    pytest.param({"indexing": "auto", "tile_sizes": (8, 8, 3, 2, 2)}, id="auto-tiled"),
]


@pytest.mark.native
@pytest.mark.parametrize("backend", ["serial", "openmp", "cuda-simulated"])
@pytest.mark.parametrize("options", PROFILES)
def test_mixed_index_values_indirect_reads_and_signed_rank_five_domains(tmp_path, backend, options):
    function, plan = _prepare(tmp_path, MIXED_SOURCE, **options)
    _compare(tmp_path, function, plan, MIXED_DRIVER, backend)


@pytest.mark.native
@pytest.mark.parametrize("backend", ["serial", "openmp", "cuda-simulated"])
def test_integer_boundaries_use_inward_strides_without_overflow(tmp_path, backend):
    function, plan = _prepare(tmp_path, BOUNDARY_SOURCE, indexing="auto", tile_sizes=(2,))
    for region in plan.regions[-2:]:
        assert region.addressing is not None
        assert region.loops[0].iterator in region.addressing.wide_iterators
        assert any(decision.mode == "wide" for decision in region.addressing.decisions)
    _compare(tmp_path, function, plan, BOUNDARY_DRIVER, backend)


@pytest.mark.native
def test_resident_run_uses_the_same_proved_addressing(tmp_path):
    function, plan = _prepare(tmp_path, MIXED_SOURCE, indexing="auto", tile_sizes=(8, 8, 3, 2, 2))
    resident = MIXED_DRIVER.replace("integer::a(", "type(entry_workspace)::work\ninteger::a(")
    resident = resident.replace(
        "call entry(a,out,source,index,5,1,-2,1,1)",
        "call entry_create(work,a,out,source,index)\n"
        "call entry_run(work,5,1,-2,1,1)\ncall entry_update_host(work,a=a,out=out)",
    )
    for arguments in ("1,5,2,1,1", "5,1,1,0,0"):
        resident = resident.replace(
            f"call entry(a,out,source,index,{arguments})",
            f"call entry_run(work,{arguments})\ncall entry_update_host(work,a=a,out=out)",
        )
    resident = resident.replace("end program", "call entry_destroy(work)\nend program")
    _compare(tmp_path, function, plan, MIXED_DRIVER, "cuda-simulated", generated_driver=resident)


@pytest.mark.native
@pytest.mark.cuda
def test_proved_tiled_rank_five_addressing_compiles_for_cuda(tmp_path):
    function, plan = _prepare(tmp_path, MIXED_SOURCE, indexing="auto", tile_sizes=(8, 8, 3, 2, 2))
    compile_cuda_sources(tmp_path, function, plan)


@pytest.mark.native
@pytest.mark.cuda
@pytest.mark.usefixtures("cuda_device")
def test_proved_addressing_matches_fortran_on_cuda(tmp_path):
    function, plan = _prepare(tmp_path, MIXED_SOURCE, indexing="auto", tile_sizes=(8, 8, 3, 2, 2))
    run_reference_and_cpp(tmp_path, function, plan, MIXED_DRIVER, backend="cuda")
