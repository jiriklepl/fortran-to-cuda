"""Lent storage at run time: the same results as the native build, and no device allocation per call."""
import os
import subprocess

import pytest

from compiler.resident.build import build
from compiler.resident.project import ResidentProject, Selection, prepare
from compiler.tests.resident_application_support import cuda_toolchain
from compiler.tests.test_resident_temporaries import SOURCE

pytestmark = pytest.mark.cuda

MAIN = """
program smoother
  use smooth
  implicit none
  real(8), allocatable :: a(:,:,:)
  integer :: calls, call_number, i, j, k
  character(32) :: argument
  call get_command_argument(1, argument)
  read(argument, *) calls
  n = 24
  allocate(a(n+2, n, n))
  do k = 1, n
    do j = 1, n
      do i = 1, n+2
        a(i,j,k) = sin(0.3d0*i) + 0.01d0*j*k
      end do
    end do
  end do
  !$fort enter
  do call_number = 1, calls
    call twice(a)
  end do
  !$fort leave
  print '(es24.16)', sum(a)
  !$fort close
end program
"""


def test_a_temporary_keeps_its_device_storage_from_call_to_call(tmp_path):
    nvcc, host, fortran, architecture = cuda_toolchain()
    module, source = tmp_path / "smooth.f90", tmp_path / "smoother.f90"
    module.write_text(SOURCE.format(use="", more=""))
    source.write_text(MAIN)
    project = ResidentProject((module, source), (Selection("all", ("smooth::twice",), minimum_nests=2),))
    prepared = prepare(project, tmp_path / "prepared", fortran=fortran)
    assert prepared["temporaries"] == [{"procedure": "smooth::twice", "name": "old"}]
    binaries = build(prepared, tmp_path / "built", name="smoother", programs={"smoother": source}, fortran=fortran,
                     cxx=host, nvcc=nvcc, architecture=architecture,
                     jobs=min(8, os.cpu_count() or 2))

    def run(name, calls, **variables):
        environment = {key: value for key, value in os.environ.items() if not key.startswith("FORT_")}
        environment.update(OMP_NUM_THREADS="2", OMP_DYNAMIC="FALSE", **variables)
        result = subprocess.run([str(binaries[name]), str(calls)], cwd=tmp_path, env=environment,
                                capture_output=True, text=True, timeout=300, check=False)
        assert result.returncode == 0, result.stderr[-4000:]
        return result.stdout, result.stderr

    def events(trace, name):
        return sum(line.startswith(f"FORT_SCOPED {name} ") for line in trace.splitlines())

    few, short = run("smoother", 3, FORT_RUNTIME_TRACE="1")
    many, long = run("smoother", 12, FORT_RUNTIME_TRACE="1")
    assert few == run("smoother-native", 3)[0]
    assert many == run("smoother-native", 12)[0]
    # The device storage is made once, and no call waits for the queue to give it back.
    assert events(short, "allocate") == events(long, "allocate") > 0
    assert events(long, "retire") == 0
    assert events(long, "launch") + events(long, "graph_kernel") == 2 * 12
