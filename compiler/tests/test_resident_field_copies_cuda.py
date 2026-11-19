"""A field copy must not land in host memory that a stale registration still covers."""
import shutil
import subprocess
from pathlib import Path

import pytest

RUNTIME = Path(__file__).resolve().parents[1] / "runtime"

CUDA = r'''
#include <cuda_runtime.h>
extern "C" int available() {
    int n=0; return cudaGetDeviceCount(&n)==cudaSuccess && n>0;
}
'''

# The program binds an array, frees it without telling the domain, and then binds
# fields of a list. The allocator hands the freed memory to the field copies, where
# the runtime still has the array registered.
FORTRAN = r'''
program test
  use iso_c_binding
  use fort_scoped_memory
  use fort_resident_domain
  implicit none
  interface
    integer(c_int) function available() bind(C)
      import c_int
    end function
  end interface
  type point
    integer(c_int) :: cell
    real(c_double) :: flux
  end type
  integer, parameter :: n=6000
  type(point), allocatable, target :: points(:)
  real(c_double), allocatable, target :: gone(:)
  type(resident_domain) :: owner
  integer(c_int64_t) :: handle, cells, flux
  integer(c_size_t) :: stride
  integer :: i, round
  if (available()==0) stop 77
  allocate(points(n))
  do i=1,n
    points(i)%cell=3*i
    points(i)%flux=0.5d0*i
  enddo
  stride=int(storage_size(points)/8,c_size_t)
  call owner%begin_epoch(0_c_int,268435456_c_size_t)
  do round=1,2
    ! As many bytes as the first field copy, and small enough to come from the heap.
    allocate(gone(n/2))
    gone=1
    handle=owner%bind(gone,[1],.true.,origin='test::gone')
    deallocate(gone)
    cells=owner%bind_field(c_loc(points(1)%cell),stride,size(points,kind=c_size_t),FORT_SCOPE_INTEGER32, &
                           .false.,origin='test::points%cell')
    flux=owner%bind_field(c_loc(points(1)%flux),stride,size(points,kind=c_size_t),FORT_SCOPE_REAL64, &
                          .true.,origin='test::points%flux')
    if (cells==0 .or. flux==0 .or. cells==flux .or. cells==handle .or. flux==handle) error stop 'field handles'
    ! The same field is the same copy.
    if (owner%bind_field(c_loc(points(1)%cell),stride,size(points,kind=c_size_t),FORT_SCOPE_INTEGER32, &
                         .false.)/=cells) error stop 'second copy of one field'
    call owner%observe([cells,flux])
    ! Publication ends the stale registration and gives the set-aside memory back.
    call owner%publish()
    do i=1,n
      if (points(i)%cell/=3*i .or. points(i)%flux/=0.5d0*i) error stop 'field values changed'
    enddo
  enddo
  call owner%close()
  print '(a)', 'PASS field copies avoid stale registrations'
end program
'''


def test_field_copies_step_around_memory_a_freed_array_left_registered(tmp_path):
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    host = shutil.which("g++-14") or shutil.which("g++")
    fc = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not Path(nvcc).exists() or not all((host, fc)):
        pytest.skip("CUDA/Fortran toolchain unavailable")
    cuda, source = tmp_path / "available.cu", tmp_path / "main.f90"
    cuda.write_text(CUDA)
    source.write_text(FORTRAN)

    def run(command):
        result = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True, timeout=300)
        assert result.returncode == 0, result.stdout + result.stderr

    objects = []
    for path in (cuda, RUNTIME / "scoped_runtime.cu"):
        obj = tmp_path / (path.stem + ".o")
        run([nvcc, "-std=c++17", "-ccbin", host, "-I", str(RUNTIME), "-c", str(path), "-o", str(obj)])
        objects.append(str(obj))
    for path in (RUNTIME / "scoped_memory.f90", RUNTIME / "resident_domain.f90", source):
        obj = tmp_path / (path.stem + ".o")
        run([fc, "-ffree-line-length-none", "-fcheck=all", "-c", str(path), "-o", str(obj)])
        objects.append(str(obj))
    binary = tmp_path / "run"
    run([nvcc, "-ccbin", host, *objects, "-lgfortran", "-o", str(binary)])
    result = subprocess.run([str(binary)], text=True, capture_output=True, timeout=60)
    if result.returncode == 77:
        pytest.skip("CUDA device unavailable")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS field copies avoid stale registrations" in result.stdout
    # Each round reports the first block it had to leave alone, with the array that still holds it.
    reports = [line for line in result.stderr.splitlines() if line.startswith("Resident storage overlap:")]
    assert len(reports) == 2, result.stderr
    assert all("origin=test::points%cell" in line for line in reports)
    assert result.stderr.count("  registered: origin=test::gone") == 2
    assert result.stderr.count("The field copy takes other memory") == 2
