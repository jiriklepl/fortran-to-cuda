"""The box application runs resident on the device and writes what its native build writes."""
import os
import re
import shutil
import subprocess

import pytest

from compiler.resident.build import build
from compiler.resident.project import prepare
from compiler.tests.resident_application_support import BOX, box_project, fortran_compiler

pytestmark = pytest.mark.cuda


@pytest.fixture(scope="module")
def binaries(tmp_path_factory):
    nvcc, host, fortran = shutil.which("nvcc"), shutil.which("g++-14") or shutil.which("g++"), fortran_compiler()
    if not all((nvcc, host, fortran, shutil.which("cmake"), shutil.which("ninja"))):
        pytest.skip("CUDA, Fortran, CMake and Ninja are required")
    probe = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                           capture_output=True, text=True, check=False)
    if probe.returncode or not probe.stdout.strip():
        pytest.skip("CUDA device unavailable")
    architecture = probe.stdout.splitlines()[0].strip().replace(".", "")
    root = tmp_path_factory.mktemp("box")
    prepared = prepare(box_project(), root / "prepared", fortran=fortran)
    return build(prepared, root / "built", name="box", programs={"box": BOX / "main.f90"}, fortran=fortran,
                 cxx=host, nvcc=nvcc, architecture=architecture, jobs=min(8, os.cpu_count() or 2))


def run(binary, directory, *arguments, trace=False):
    directory.mkdir()
    environment = {key: value for key, value in os.environ.items() if not key.startswith("FORT_")}
    environment.update(OMP_NUM_THREADS="4", OMP_DYNAMIC="FALSE")
    if trace:
        environment["FORT_RUNTIME_TRACE"] = "1"
    result = subprocess.run([str(binary), *map(str, arguments)], cwd=directory, env=environment,
                            capture_output=True, text=True, timeout=300, check=False)
    assert result.returncode == 0, result.stderr[-4000:]
    return result.stdout, result.stderr, (directory / "field.bin").read_bytes()


def events(trace):
    return [match.group(1) for match in re.finditer(r"^FORT_SCOPED (\w+)", trace, re.MULTILINE)]


def test_the_resident_run_matches_the_native_run_byte_for_byte(binaries, tmp_path):
    arguments = (48, 40, 32, 30, 10)
    native_output, _, native_field = run(binaries["box-native"], tmp_path / "native", *arguments)
    output, trace, field = run(binaries["box"], tmp_path / "resident", *arguments, trace=True)
    assert field == native_field
    assert len(field) == 52 * 44 * 36 * 8
    # The host diagnostic saw the device's values at each report, digit for digit.
    assert output == native_output
    assert output.count("energy") == 3
    seen = events(trace)
    # Each of the thirty steps queues five device tasks: the three ordered halo copies, the update and the store.
    assert seen.count("queue_enqueue") == 150
    assert seen.count("graph_kernel") + seen.count("launch") == 150
    assert "implementation=" in trace
    assert "generated_cpu" not in trace
    # The native build has no residency code at all.
    assert events(run(binaries["box-native"], tmp_path / "plain", *arguments, trace=True)[1]) == []


def test_host_reads_do_not_send_the_field_back_to_the_device(binaries, tmp_path):
    quiet = events(run(binaries["box"], tmp_path / "quiet", 32, 32, 32, 20, 1000, trace=True)[1])
    chatty = events(run(binaries["box"], tmp_path / "chatty", 32, 32, 32, 20, 2, trace=True)[1])
    # Observing a field downloads it and keeps the device copy valid, so ten reports cost no further upload.
    assert chatty.count("upload") == quiet.count("upload")
    assert chatty.count("download") > quiet.count("download")


def test_below_the_guarded_extent_the_same_binary_runs_its_original_loops(binaries, tmp_path):
    arguments = (1, 6, 6, 5, 1)
    _, trace, field = run(binaries["box"], tmp_path / "resident", *arguments, trace=True)
    native_output, _, native_field = run(binaries["box-native"], tmp_path / "native", *arguments)
    assert field == native_field
    assert "queue_enqueue" not in events(trace)
    assert "launch" not in events(trace)
