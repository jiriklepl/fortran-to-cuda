"""A domain whose caller states no budget takes it from the device or from FORT_RESIDENT_DEVICE_BUDGET."""
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
extern "C" long long free_now() {
    size_t free_bytes=0,total=0;
    return cudaMemGetInfo(&free_bytes,&total)==cudaSuccess ? static_cast<long long>(free_bytes) : -1;
}
'''

FORTRAN = r'''
program test
  use iso_c_binding
  use fort_resident_domain
  implicit none
  interface
    integer(c_int) function available() bind(C)
      import c_int
    end function
    integer(c_long_long) function free_now() bind(C)
      import c_long_long
    end function
  end interface
  type(resident_domain) :: derived, stated
  if (available()==0) stop 77
  call derived%begin_epoch(0_c_int)
  print '(a,i0)', 'derived ', derived%unclaimed()
  print '(a,i0)', 'free ', free_now()
  call derived%close()
  call stated%begin_epoch(0_c_int, 1048576_c_size_t)
  print '(a,i0)', 'stated ', stated%unclaimed()
  call stated%close()
end program
'''


@pytest.mark.cuda
def test_the_budget_follows_the_device_unless_it_is_stated(tmp_path):
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

    def budgets(value):
        env = {key: item for key, item in os.environ.items() if key != "FORT_RESIDENT_DEVICE_BUDGET"}
        if value is not None:
            env["FORT_RESIDENT_DEVICE_BUDGET"] = value
        return subprocess.run([str(binary)], env=env, text=True, capture_output=True, timeout=60)

    result = budgets(None)
    if result.returncode == 77:
        pytest.skip("CUDA device unavailable")
    assert result.returncode == 0, result.stdout + result.stderr
    words = result.stdout.split()
    derived, free = int(words[1]), int(words[3])
    # What is free now, less the reserve of at least 512 MiB; other processes may move it a little.
    assert 0 < derived <= free - (512 << 20) + (64 << 20)
    assert derived >= (free - (512 << 20)) * 7 // 8
    assert words[4:] == ["stated", "1048576"]
    # The variable replaces the derived value, not a budget the caller states.
    for value, expected in (("3145728", 3145728), ("64K", 65536), ("5m", 5 << 20), ("1G", 1 << 30)):
        words = budgets(value).stdout.split()
        assert (words[1], words[4:]) == (str(expected), ["stated", "1048576"])
    for value in ("lots", "-5", "0", "12Q", "1.5G"):
        refused = budgets(value)
        assert refused.returncode != 0
        assert "FORT_RESIDENT_DEVICE_BUDGET must be a positive count of bytes" in refused.stderr
