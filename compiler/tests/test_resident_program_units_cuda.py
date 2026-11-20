"""A solver in one main program and one external procedure runs resident and writes what its native build writes."""
import os
import shutil
import subprocess

import pytest

from compiler.resident.build import build
from compiler.resident.project import prepare
from compiler.tests.resident_application_support import RELAX, fortran_compiler, relax_project

pytestmark = pytest.mark.cuda


def test_the_time_loop_of_a_main_program_runs_its_nests_on_the_device(tmp_path):
    nvcc, host, fortran = shutil.which("nvcc"), shutil.which("g++-14") or shutil.which("g++"), fortran_compiler()
    if not all((nvcc, host, fortran, shutil.which("cmake"), shutil.which("ninja"))):
        pytest.skip("CUDA, Fortran, CMake and Ninja are required")
    probe = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                           capture_output=True, text=True, check=False)
    if probe.returncode or not probe.stdout.strip():
        pytest.skip("CUDA device unavailable")
    prepared = prepare(relax_project(), tmp_path / "prepared", fortran=fortran)
    binaries = build(prepared, tmp_path / "built", name="relax", programs={"relax": RELAX}, fortran=fortran,
                     cxx=host, nvcc=nvcc, architecture=probe.stdout.splitlines()[0].strip().replace(".", ""),
                     jobs=min(8, os.cpu_count() or 2))

    def run(name, **variables):
        directory = tmp_path / name
        directory.mkdir()
        environment = {key: value for key, value in os.environ.items() if not key.startswith("FORT_")}
        environment.update(OMP_NUM_THREADS="4", OMP_DYNAMIC="FALSE", **variables)
        result = subprocess.run([str(binaries[name]), "96", "40"], cwd=directory, env=environment,
                                capture_output=True, text=True, timeout=300, check=False)
        assert result.returncode == 0, result.stderr[-4000:]
        return result.stdout, result.stderr, (directory / "field.bin").read_bytes()

    native_output, _, native_field = run("relax-native")
    output, trace, field = run("relax", FORT_RUNTIME_TRACE="1")
    assert len(field) == 98 * 98 * 8
    assert field == native_field
    # The largest change of a sweep and the largest value come back from the device digit for digit.
    assert output == native_output
    assert output.count("change") == 4
    launches = sum(line.startswith(("FORT_SCOPED launch ", "FORT_SCOPED graph_kernel "))
                   for line in trace.splitlines())
    # Per sweep: the stencil, the partial maxima and the copy; four times the scaling and its maximum.
    assert launches == 3 * 40 + 2 * 4
    # Nothing but partial results comes back inside the loop. The two fields are downloaded once, after
    # the last launch, and only the part the device changed.
    events = [line for line in trace.splitlines()
              if line.startswith(("FORT_SCOPED launch ", "FORT_SCOPED graph_kernel ", "FORT_SCOPED download "))]
    large = [index for index, line in enumerate(events) if line.startswith("FORT_SCOPED download ")
             and int(line.split("bytes=")[1].split()[0]) > 256 * 8]
    assert [int(events[index].split("bytes=")[1].split()[0]) for index in large] == [96 * 96 * 8] * 2
    assert large == [len(events) - 2, len(events) - 1]
