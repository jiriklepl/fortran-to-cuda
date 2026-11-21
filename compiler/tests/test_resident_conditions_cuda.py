"""An entry compiled for some values of its scalars runs on the device for those and declines the others."""
import os
import shutil
import subprocess

import pytest

from compiler.resident.build import build
from compiler.resident.project import ResidentProject, Selection, prepare
from compiler.tests.resident_application_support import fortran_compiler

pytestmark = pytest.mark.cuda

MODULE = """module levels
integer :: n
contains
subroutine blend(u, to, from)
real(8), contiguous, intent(inout) :: u(0:,0:,:)
integer, intent(in) :: to, from
integer :: i, j
do j = 1, n
do i = 1, n
u(i,j,to) = 0.5d0 * u(i,j,to) + 0.125d0 * (u(i-1,j,from) + u(i+1,j,from) + u(i,j-1,from) + u(i,j+1,from))
enddo
enddo
end subroutine
end module
"""

MAIN = """
program march
  use levels
  implicit none
  real(8), allocatable :: u(:,:,:)
  integer :: step, i, j, old, new
  n = 40
  allocate(u(0:n+1, 0:n+1, 2))
  do j = 0, n+1
    do i = 0, n+1
      u(i,j,1) = sin(0.2d0*i) * cos(0.1d0*j)
      u(i,j,2) = 0
    end do
  end do
  old = 1
  new = 2
  !$fort enter
  do step = 1, 6
    call blend(u, new, old)
    old = new
    new = 3 - old
  end do
  ! One level blended with itself: every point reads neighbours that the same nest changes.
  call blend(u, 1, 1)
  call blend(u, 2, 1)
  !$fort leave
  print '(2es24.16)', sum(u(:,:,1)), sum(u(:,:,2))
  !$fort close
end program
"""


def test_the_entry_runs_for_the_values_it_was_compiled_for_and_the_host_for_the_others(tmp_path):
    nvcc, host, fortran = shutil.which("nvcc"), shutil.which("g++-14") or shutil.which("g++"), fortran_compiler()
    if not all((nvcc, host, fortran, shutil.which("cmake"), shutil.which("ninja"))):
        pytest.skip("CUDA, Fortran, CMake and Ninja are required")
    probe = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                           capture_output=True, text=True, check=False)
    if probe.returncode or not probe.stdout.strip():
        pytest.skip("CUDA device unavailable")
    module, source = tmp_path / "levels.f90", tmp_path / "march.f90"
    module.write_text(MODULE)
    source.write_text(MAIN)
    project = ResidentProject((module, source), (Selection("all", ("levels::blend",), minimum_nests=1),))
    prepared = prepare(project, tmp_path / "prepared", fortran=fortran)
    binaries = build(prepared, tmp_path / "built", name="march", programs={"march": source}, fortran=fortran,
                     cxx=host, nvcc=nvcc, architecture=probe.stdout.splitlines()[0].strip().replace(".", ""),
                     jobs=min(8, os.cpu_count() or 2))

    def run(name, **variables):
        environment = {key: value for key, value in os.environ.items() if not key.startswith("FORT_")}
        environment.update(OMP_NUM_THREADS="2", OMP_DYNAMIC="FALSE", **variables)
        result = subprocess.run([str(binaries[name])], cwd=tmp_path, env=environment,
                                capture_output=True, text=True, timeout=300, check=False)
        assert result.returncode == 0, result.stderr[-4000:]
        return result.stdout, result.stderr

    output, trace = run("march", FORT_RUNTIME_TRACE="1")
    assert output == run("march-native")[0]
    launches = sum(line.startswith(("FORT_SCOPED launch ", "FORT_SCOPED graph_kernel ")) for line in trace.splitlines())
    # Seven calls with two different levels run on the device; the one with equal levels runs on the host.
    assert launches == 7
