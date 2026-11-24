"""The small applications the resident build is tested on, and the tools a test that builds one needs."""
import shutil
import subprocess
from pathlib import Path

import pytest

from compiler.resident.project import ResidentProject, Selection

BOX = Path(__file__).resolve().parent / "apps/box"
UNITS = ("params", "fields", "boundaries", "sources", "stepper", "diagnostics", "main")


def box_project(sources=BOX, *, defines=("DOUBLE",)):
    """The whole description the box needs: its sources, four selections and a precision."""
    return ResidentProject(
        sources=tuple(Path(sources) / (name + ".f90") for name in UNITS),
        selections=(Selection("step", ("stepper::diffuse", "stepper::store"), minimum_nests=2),
                    # The copies of a periodic direction do not meet from two cells on: the compiler finds
                    # that condition and the entry checks it.
                    Selection("halo", ("boundaries::*",), minimum_nests=3),
                    # Each accumulating loop becomes three nests: partial results of a short loop, and
                    # partial results and their fold for a long one.
                    Selection("sources", ("sources::inject", "sources::sense", "sources::range"), minimum_nests=5),
                    Selection("diagnostics", ("diagnostics::summary", "diagnostics::peak"), minimum_nests=6)),
        defines=defines)


def fortran_compiler():
    return shutil.which("gfortran-15") or shutil.which("gfortran")


def cuda_toolchain():
    """The CUDA, C++ and Fortran compilers and the device's architecture; skips the test that lacks one."""
    nvcc, host, fortran = shutil.which("nvcc"), shutil.which("g++-14") or shutil.which("g++"), fortran_compiler()
    if not all((nvcc, host, fortran, shutil.which("cmake"), shutil.which("ninja"), shutil.which("nvidia-smi"))):
        pytest.skip("CUDA, Fortran, CMake and Ninja are required")
    probe = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                           capture_output=True, text=True, check=False)
    if probe.returncode or not probe.stdout.strip():
        pytest.skip("CUDA device unavailable")
    return nvcc, host, fortran, probe.stdout.splitlines()[0].strip().replace(".", "")


RELAX = Path(__file__).resolve().parent / "apps/relax/relax.f90"


def relax_project():
    """A solver written as one main program and one external procedure, both selected."""
    return ResidentProject(sources=(RELAX,), selections=(
        Selection("all", ("$program::relax", "$external::damp"), minimum_nests=12),))
