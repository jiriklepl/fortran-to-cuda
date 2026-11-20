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


def run(binary, directory, *arguments, trace=False, **variables):
    directory.mkdir()
    environment = {key: value for key, value in os.environ.items() if not key.startswith("FORT_")}
    environment.update(OMP_NUM_THREADS="4", OMP_DYNAMIC="FALSE", **variables)
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
    # The file holds the field with its halos and what each of the five sources saw.
    assert len(field) == (52 * 44 * 36 + 5) * 8
    # The host diagnostic saw the device's values at each report, digit for digit.
    assert output == native_output
    assert output.count("energy") == 3
    assert output.count("peak") == 3
    seen = events(trace)
    # Each of the thirty steps queues seven device tasks: the three ordered halo copies, the update, the
    # store, the injection at the sources and their sensing. Each of the three reports adds three: the
    # partial results of the loops of summary, peak and range, which are short enough to need no fold.
    assert seen.count("queue_enqueue") == 210 + 9
    assert seen.count("graph_kernel") + seen.count("launch") == 210 + 9
    # Of what the reports accumulate, only the folded values come back: one per source for the two
    # scalars of range, and 256 for each of the four scalars of summary and peak.
    sizes = [int(size) for size in re.findall(r"^FORT_SCOPED download .*? bytes=(\d+)", trace, re.MULTILINE)]
    assert sorted(size for size in sizes if size in (1024, 2048)) == [1024] * 6 + [2048] * 6
    assert "implementation=" in trace
    assert "generated_cpu" not in trace
    # The native build has no residency code at all.
    assert events(run(binaries["box-native"], tmp_path / "plain", *arguments, trace=True)[1]) == []


def test_a_long_accumulating_loop_folds_its_partial_results_on_the_device(binaries, tmp_path):
    # More cells than one direct pass takes: 65536 slots store partial results and a second nest folds them.
    arguments = (64, 48, 40, 6, 3)
    native_output, _, native_field = run(binaries["box-native"], tmp_path / "native", *arguments)
    output, trace, field = run(binaries["box"], tmp_path / "resident", *arguments, trace=True)
    assert (field, output) == (native_field, native_output)
    # Per report: two tasks each for summary and peak, one for the five sources of range.
    assert events(trace).count("queue_enqueue") == 6 * 7 + 2 * 5


def test_host_reads_do_not_send_the_field_back_to_the_device(binaries, tmp_path):
    quiet = events(run(binaries["box"], tmp_path / "quiet", 32, 32, 32, 20, 1000, trace=True)[1])
    chatty = events(run(binaries["box"], tmp_path / "chatty", 32, 32, 32, 20, 2, trace=True)[1])
    # Observing a field downloads it and keeps the device copy valid, so ten reports cost no further upload
    # of the field. What the reports upload once is the arrays that hold their partial results.
    more = chatty.count("upload") - quiet.count("upload")
    assert 0 < more <= 12
    assert chatty.count("download") > quiet.count("download")


def test_below_the_guarded_extent_the_same_binary_runs_its_original_loops(binaries, tmp_path):
    arguments = (1, 6, 6, 5, 1)
    _, trace, field = run(binaries["box"], tmp_path / "resident", *arguments, trace=True)
    native_output, _, native_field = run(binaries["box-native"], tmp_path / "native", *arguments)
    assert field == native_field
    assert "queue_enqueue" not in events(trace)
    assert "launch" not in events(trace)


def test_sources_that_share_a_cell_are_injected_on_the_host_with_the_same_result(binaries, tmp_path):
    arguments = (24, 20, 16, 12, 4, -6)
    native_output, _, native_field = run(binaries["box-native"], tmp_path / "native", *arguments)
    output, trace, field = run(binaries["box"], tmp_path / "resident", *arguments, trace=True)
    assert field == native_field
    assert output == native_output
    # Two sources per cell: the uniqueness check fails, so six of the seven tasks per step stay on the device.
    # The three reports queue three tasks each.
    assert events(trace).count("queue_enqueue") == 12 * 6 + 9
    distinct = events(run(binaries["box"], tmp_path / "distinct", 24, 20, 16, 12, 4, 6, trace=True)[1])
    assert distinct.count("queue_enqueue") == 12 * 7 + 9
    # With distinct sources nothing is published during the run, so far fewer transfers happen.
    assert distinct.count("upload") < events(trace).count("upload")


def test_an_empty_source_list_leaves_its_entries_out(binaries, tmp_path):
    arguments = (24, 20, 16, 12, 4, 0)
    native_output, _, native_field = run(binaries["box-native"], tmp_path / "native", *arguments)
    output, trace, field = run(binaries["box"], tmp_path / "resident", *arguments, trace=True)
    assert field == native_field
    assert output == native_output
    assert events(trace).count("queue_enqueue") == 12 * 5 + 6


@pytest.mark.parametrize(("budget", "consequence", "launches"), [
    # Both fields and nothing else: the source lists and a field take turns on the device.
    ("1317988", "the least recently used ones now move back to the host", 219),
    # One field: the two nests that need both fields at once run on the host.
    ("700K", "loop nests whose arrays do not fit together run on the host", 159),
    ("600K", "loop nests whose arrays do not fit together run on the host", 3),
])
def test_a_budget_that_is_too_small_costs_time_and_says_so_once(binaries, tmp_path, budget, consequence, launches):
    arguments = (48, 40, 32, 30, 10)
    output, _, field = run(binaries["box"], tmp_path / "roomy", *arguments)
    tight_output, trace, tight_field = run(binaries["box"], tmp_path / "tight", *arguments, trace=True,
                                           FORT_RESIDENT_DEVICE_BUDGET=budget)
    assert (tight_field, tight_output) == (field, output)
    assert events(trace).count("queue_enqueue") == launches
    assert trace.count("Resident domain: the arrays in use exceed the device budget") == 1
    assert consequence in trace
