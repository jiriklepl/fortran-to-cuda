"""Regressions for shared build configuration and correctness checking."""

import shutil
import subprocess

import pytest

from benchmarks.harness import check
from benchmarks.harness.paths import BENCHMARKS
from benchmarks.harness.run import binary_path


def dry_run(*assignments: str) -> str:
    if not shutil.which("make"):
        pytest.skip("make is unavailable")
    return subprocess.run(
        ["make", "-Bn", "-C", str(BENCHMARKS), "CASE=CDU", *assignments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def test_correctness_and_timing_builds_use_distinct_directories():
    timing = binary_path("CDU", "Fortran", 16, 16, 16, 1, 0)
    correctness = binary_path("CDU", "Fortran", 16, 16, 16, 1, 0, main_src="test_main.f90")
    assert timing != correctness
    parameters = ("VARIANT=Fortran", "NX=16", "NY=16", "NZ=16", "NITER=1", "NWARMUP=0")
    assert f"-o {timing} " in dry_run(*parameters)
    assert f"-o {correctness} " in dry_run(*parameters, "MAIN_SRC=test_main.f90")


def test_pinned_variant_enables_macro_once_without_runner_flags():
    assert dry_run("VARIANT=CUDA-pinned").count("-DUSE_PINNED_MEMORY") == 1
    assert "-DUSE_PINNED_MEMORY" not in dry_run("VARIANT=CUDA")


def test_correctness_check_fails_without_reference(monkeypatch):
    def unavailable(*args, **kwargs):
        raise RuntimeError("compiler unavailable")

    monkeypatch.setattr(check, "build", unavailable)
    assert not check.check_case("CDU", ["Fortran"])


@pytest.mark.parametrize("output", ["nan " * 4096, "1.0"], ids=["nonfinite", "truncated"])
def test_correctness_check_rejects_invalid_reference(monkeypatch, output):
    monkeypatch.setattr(check, "build", lambda *args, **kwargs: "benchmark")
    monkeypatch.setattr(check, "run_once", lambda _: output)
    assert not check.check_case("CDU", ["Fortran"])
