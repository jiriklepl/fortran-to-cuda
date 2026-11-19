"""An epoch narrows the OpenMP team to FORT_RESIDENT_HOST_THREADS and gives the count back when it ends."""
import os
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

FORTRAN = r'''
program test
  use iso_c_binding
  use omp_lib
  use fort_resident_domain
  implicit none
  interface
    integer(c_int) function available() bind(C)
      import c_int
    end function
  end interface
  type(resident_domain) :: owner
  integer :: round
  if (available()==0) stop 77
  call omp_set_num_threads(4)
  do round=1,2
    call owner%begin_epoch(0_c_int,1048576_c_size_t)
    print '(a,i0)', 'inside ', omp_get_max_threads()
    call owner%end_epoch()
    print '(a,i0)', 'outside ', omp_get_max_threads()
  enddo
  call owner%close()
end program
'''


@pytest.mark.cuda
def test_the_team_is_narrowed_only_while_the_epoch_lasts(tmp_path):
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
        run([nvcc, "-std=c++17", "-ccbin", host, "-Xcompiler=-fopenmp", "-I", str(RUNTIME), "-c", str(path),
             "-o", str(obj)])
        objects.append(str(obj))
    for path in (RUNTIME / "scoped_memory.f90", RUNTIME / "resident_domain.f90", source):
        obj = tmp_path / (path.stem + ".o")
        run([fc, "-fopenmp", "-ffree-line-length-none", "-c", str(path), "-o", str(obj)])
        objects.append(str(obj))
    binary = tmp_path / "run"
    run([nvcc, "-ccbin", host, *objects, "-lgfortran", "-lgomp", "-o", str(binary)])

    def teams(value):
        env = {key: item for key, item in os.environ.items() if key != "FORT_RESIDENT_HOST_THREADS"}
        if value is not None:
            env["FORT_RESIDENT_HOST_THREADS"] = value
        return subprocess.run([str(binary)], env=env, text=True, capture_output=True, timeout=60)

    result = teams("2")
    if result.returncode == 77:
        pytest.skip("CUDA device unavailable")
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.split() == ["inside", "2", "outside", "4"] * 2
    # Without the variable, or with a count that would widen the team, nothing changes.
    assert teams(None).stdout.split() == ["inside", "4", "outside", "4"] * 2
    assert teams("9").stdout.split() == ["inside", "4", "outside", "4"] * 2
    refused = teams("none")
    assert refused.returncode != 0
    assert "FORT_RESIDENT_HOST_THREADS must be a positive integer" in refused.stderr
