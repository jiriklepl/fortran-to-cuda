"""Compile the actual protocols; these smoke observations are never profile costs."""

import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from compiler.offload.tile_protocol_calibration import original_team_benchmark, source_scratch_benchmark


@pytest.fixture(scope="module", params=[1, 3])
def original_probe(request, tmp_path_factory):
    fc = shutil.which("gfortran")
    if fc is None:
        pytest.skip("original Fortran compiler required for real protocol fixture")
    directory = tmp_path_factory.mktemp("original-team-protocol")
    source = directory / "original.f90"
    source.write_text(original_team_benchmark(rank=request.param))
    executable = directory / "observe"
    result = subprocess.run([fc, "-O2", "-fopenmp", "-ffp-contract=off", str(source), "-o", str(executable)],
                            capture_output=True, text=True, timeout=60)
    (directory / "compile.stdout").write_text(result.stdout)
    (directory / "compile.stderr").write_text(result.stderr)
    assert result.returncode == 0, result.stderr
    return directory, executable


def test_original_team_observations_require_exact_width_and_every_original_do(original_probe):
    directory, executable = original_probe
    env = {**os.environ, "OMP_NUM_THREADS": "2", "OMP_THREAD_LIMIT": "2", "OMP_DYNAMIC": "FALSE", "OMP_SCHEDULE": "static,1"}
    result = subprocess.run([str(executable), "2", "3"], capture_output=True, text=True, timeout=30, env=env)
    (directory / "observe.stdout").write_text(result.stdout)
    (directory / "observe.stderr").write_text(result.stderr)
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert [row["units"] for row in rows] == [1, 2, 3, 4, 8]
    for row in rows:
        assert row["kind"] == "original_team_observation"
        assert row["cpu_threads"] == 2
        assert row["collapse"] == row["rank"]
        assert row["rank"] in (1, 3)
        assert row["rounds"] == 3
        assert row["schedule_chunk"] == 1
        assert row["schedule_kind"] & 0x7fffffff == 1
        assert len(row["seconds"]) == 5
        assert all(math.isfinite(value) and value > 0 for value in row["seconds"])
    # One shared original parallel team encloses all source worksharing units;
    # actual runtime symbols rule out a silently serial CPU fixture.
    nm = shutil.which("nm")
    if nm:
        symbols = subprocess.run([nm, "-D", str(executable)], check=True, capture_output=True, text=True, timeout=10).stdout
        assert "GOMP_parallel" in symbols
        assert "GOMP_loop_end" in symbols
        assert "GOMP_loop_maybe_nonmonotonic_runtime_start" in symbols or "GOMP_loop_runtime_start" in symbols


def test_unavailable_original_team_width_cannot_emit_calibration_rows(original_probe):
    directory, executable = original_probe
    result = subprocess.run([str(executable), "2", "1"], capture_output=True, text=True, timeout=30,
        env={**os.environ, "OMP_THREAD_LIMIT": "1", "OMP_DYNAMIC": "FALSE", "OMP_SCHEDULE": "static"})
    (directory / "unavailable.stdout").write_text(result.stdout)
    (directory / "unavailable.stderr").write_text(result.stderr)
    assert result.returncode != 0
    assert not result.stdout.strip()


@pytest.mark.parametrize("units", [(), (1, 1), (0,), (-1,), (True,), (1.0,), (33,)])
def test_native_protocol_rejects_ambiguous_or_unbounded_recipes(units):
    with pytest.raises(ValueError, match="bounded positive unit"):
        original_team_benchmark(units=units)


@pytest.mark.parametrize(("rank", "collapse"), [(0, 0), (4, 1), (2, 3), (True, 1), (3, False)])
def test_native_protocol_rejects_unknown_original_loop_and_collapse_shapes(rank, collapse):
    with pytest.raises(ValueError, match="exact rank/collapse"):
        original_team_benchmark(rank=rank, collapse=collapse)


@pytest.mark.parametrize("precision", [32, 64])
def test_actual_source_and_scratch_abi_probe_compiles_without_executing_cuda(tmp_path, precision):
    nvcc, host = shutil.which("nvcc"), shutil.which("g++-14") or shutil.which("g++")
    if not nvcc or not host:
        pytest.skip("CUDA compiler required for actual source/scratch ABI check")
    source = tmp_path / "source_protocol.cu"
    source.write_text(source_scratch_benchmark(precision=precision))
    result = subprocess.run([nvcc, "-std=c++17", "-ccbin", host, "-arch=sm_86", "-Xcompiler=-fopenmp",
        "-I", str(Path("compiler/runtime").resolve()), "-c", str(source), "-o", str(tmp_path / "protocol.o")],
        capture_output=True, text=True, timeout=120)
    (tmp_path / "compile.stdout").write_text(result.stdout)
    (tmp_path / "compile.stderr").write_text(result.stderr)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "protocol.o").stat().st_size > 0


@pytest.mark.parametrize("precision", [0, 16, True, 32.0])
def test_source_protocol_unknown_representation_cannot_produce_measurement_source(precision):
    with pytest.raises(ValueError, match="REAL32 or REAL64"):
        source_scratch_benchmark(precision=precision)
