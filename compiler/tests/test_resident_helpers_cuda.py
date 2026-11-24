"""A point loop that calls rewritten procedures runs on the device and writes what its native build writes."""
import os
import subprocess

import pytest

from compiler.resident.build import build
from compiler.resident.project import ResidentProject, Selection, prepare
from compiler.tests.resident_application_support import cuda_toolchain
from compiler.tests.test_resident_helpers import WALLS

pytestmark = pytest.mark.cuda

MAIN = """
program law
  use walls
  implicit none
  real(knd), allocatable :: u(:)
  integer :: i, step
  allocate(u(200), points(64), visc(64))
  do i = 1, 200
    u(i) = 1 + 0.5_knd * sin(0.1_knd * i)
  end do
  do i = 1, 64
    points(i)%xi = 3 * i
    points(i)%distx = 0.01_knd * mod(i, 5)
    points(i)%disty = 0.002_knd * i
    points(i)%distz = 0.05_knd + 0.001_knd * i
    points(i)%ustar = 0.05_knd
  end do
  ! The last points see no flow: their friction velocity stays where it is.
  u(180:) = 0
  !$fort enter
  ! Each step starts from the friction velocities of the one before.
  do step = 1, 3
    call viscs(u)
  end do
  !$fort leave
  print '(2es25.17)', sum(visc), sum(points(:)%ustar)
  print '(4es25.17)', visc(1), visc(64), points(1)%ustar, points(64)%ustar
  !$fort close
end program
"""


def test_the_wall_law_runs_on_the_device_with_the_host_results(tmp_path):
    nvcc, host, fortran, architecture = cuda_toolchain()
    module, source = tmp_path / "walls.f90", tmp_path / "law.f90"
    module.write_text(WALLS)
    source.write_text(MAIN)
    project = ResidentProject((module, source), (Selection("all", ("walls::viscs",), minimum_nests=1),))
    prepared = prepare(project, tmp_path / "prepared", fortran=fortran)
    binaries = build(prepared, tmp_path / "built", name="law", programs={"law": source}, fortran=fortran,
                     cxx=host, nvcc=nvcc, architecture=architecture,
                     jobs=min(8, os.cpu_count() or 2))

    def run(name, **variables):
        environment = {key: value for key, value in os.environ.items() if not key.startswith("FORT_")}
        environment.update(OMP_NUM_THREADS="2", OMP_DYNAMIC="FALSE", **variables)
        result = subprocess.run([str(binaries[name])], cwd=tmp_path, env=environment,
                                capture_output=True, text=True, timeout=300, check=False)
        assert result.returncode == 0, result.stderr[-4000:]
        return result.stdout, result.stderr

    output, trace = run("law", FORT_RUNTIME_TRACE="1")
    # NORM2, the iteration to convergence and the copies of the friction velocities give the host's bits.
    assert output == run("law-native")[0]
    assert sum(line.startswith(("FORT_SCOPED launch ", "FORT_SCOPED graph_kernel ")) for line in trace.splitlines()) == 3
