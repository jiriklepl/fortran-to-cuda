"""Offline observations cannot silently become placement or bounded execution."""

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from compiler.driver.options import CompilerOptions
from compiler.offload.config import OffloadConfig
from compiler.frontend.numerical_preparation import _Budget, _configuration


@pytest.mark.parametrize("extra", [[], ["--form-scopes"],
    ["--form-scopes", "--scope-execution", "graph", "--gpu-policy", "auto"],
    ["--form-scopes", "--scope-execution", "graph", "--gpu-policy", "sections", "--gpu-collective"]])
def test_cli_rejects_incompatible_measurement_without_touching_output(tmp_path, extra):
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "sentinel"
    sentinel.write_text("unchanged")
    result = subprocess.run([sys.executable, "-m", "compiler", "--scope-measurement", "native",
        "--input", "missing.f90", "--kernel", "advance", "--output-dir", str(output), *extra],
        cwd=Path(__file__).resolve().parents[2], text=True, capture_output=True, timeout=30)
    assert result.returncode != 0
    assert "scope-measurement" in result.stderr
    assert list(output.iterdir()) == [sentinel]
    assert sentinel.read_text() == "unchanged"
