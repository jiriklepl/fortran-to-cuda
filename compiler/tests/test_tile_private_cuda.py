"""Actual original-Fortran equivalence and runtime-leased indexed tile scratch."""

import subprocess
from pathlib import Path


RUNTIME = Path(__file__).resolve().parents[1] / "runtime"


def checked(command, directory, name, timeout=120):
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    (directory / (name + ".stdout")).write_text(result.stdout)
    (directory / (name + ".stderr")).write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    return result
