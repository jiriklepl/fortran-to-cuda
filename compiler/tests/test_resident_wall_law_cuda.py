"""The wall law compiled from its own source runs on the device and gives the host's bits."""
import os
import subprocess

import pytest

from compiler.resident.build import build
from compiler.resident.project import ResidentProject, Selection, prepare
from compiler.tests.resident_application_support import cuda_toolchain
from compiler.tests.test_resident_wall_law import LAW

pytestmark = pytest.mark.cuda

MAIN = """
program run_law
  use law
  implicit none
  real(knd), allocatable :: u(:), w(:)
  integer :: i
  allocate(u(-2:120), w(-2:120), points(100), visc(100))
  do i = -2, 120
    u(i) = 1 + 0.3_knd * sin(0.2_knd * i)
    w(i) = 0.1_knd * cos(0.3_knd * i)
  end do
  do i = 1, 100
    points(i)%xi = i
    points(i)%zk = 1
    points(i)%z0 = 0
    points(i)%ustar = 0.05_knd
    points(i)%fixed = .false.
  end do
  u(90:) = 0
  w(90:) = 0
  visc = 0
  !$fort enter
  call fluxes(u, w, 1, 5)
  call fluxes(u, w, 3, 6)
  !$fort leave
  print '(3es25.17)', sum(points(:)%flux), sum(points(:)%ustar), sum(visc)
  print '(3es25.17)', points(1)%flux, points(100)%flux, visc(50)
  !$fort close
end program
"""


def test_the_wall_law_runs_on_the_device_with_the_host_bits(tmp_path):
    nvcc, host, fortran, architecture = cuda_toolchain()
    module, source = tmp_path / "law.f90", tmp_path / "run_law.f90"
    module.write_text(LAW)
    source.write_text(MAIN)
    project = ResidentProject((module, source), (Selection("all", ("law::fluxes",), minimum_nests=1),))
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
    assert output == run("law-native")[0]
    assert sum(line.startswith(("FORT_SCOPED launch ", "FORT_SCOPED graph_kernel ")) for line in trace.splitlines()) == 2
