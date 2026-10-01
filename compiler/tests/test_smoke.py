"""Minimal CLI scaffolding; generated code is not compiled or executed."""

import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.mark.parametrize("kernel", ["fill_array", "scale_array"])
def test_generate_trivial_kernel(kernel: str, tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "compiler",
            "--input",
            str(FIXTURES / f"{kernel}.f90"),
            "--kernel",
            kernel,
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    for filename in (
        "generated_code.cu",
        "generated_cpp_impl.cpp",
        "generated_interface.f90",
        "common_functions.cuh",
    ):
        assert (tmp_path / filename).read_text().strip(), filename

    assert "__global__" in (tmp_path / "generated_code.cu").read_text()
