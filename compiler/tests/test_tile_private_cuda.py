"""Actual original-Fortran equivalence and runtime-leased indexed tile scratch."""

import shutil
import subprocess
from pathlib import Path

import pytest

RUNTIME = Path(__file__).resolve().parents[1] / "runtime"


def checked(command, directory, name, timeout=120):
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    (directory / (name + ".stdout")).write_text(result.stdout)
    (directory / (name + ".stderr")).write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.fixture(scope="module")
def tile_runtime_object(tmp_path_factory):
    nvcc, host = shutil.which("nvcc"), shutil.which("g++-14") or shutil.which("g++")
    fc = shutil.which("gfortran")
    if not nvcc or not host or not fc:
        pytest.skip("CUDA toolkit and gfortran required")
    directory = tmp_path_factory.mktemp("tile-private-runtime")
    target = directory / "runtime.o"
    checked([nvcc, "-std=c++17", "-ccbin", host, "-arch=sm_86", "-I", str(RUNTIME),
             "-c", str(RUNTIME / "scoped_runtime.cu"), "-o", str(target)], directory, "runtime")
    return nvcc, host, fc, target
