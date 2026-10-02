"""Rejected inputs must diagnose their origin before publishing any output."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ("generated_code.cu", "generated_cpp_impl.cpp", "generated_interface.f90", "common_functions.cuh")


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        ("if(n>0) a(1)=1.0_knd", "If_Stmt"),
        ("do i=1,n,0\n a(i)=1.0_knd\nenddo", "zero"),
        ("do i=2,n\n a(i)=a(i-1)\nenddo", "RAW"),
        ("do i=1,n\n a(1)=1.0_knd\nenddo", "WAW"),
        ("s=0.0_knd\ndo i=1,n\n s=s+a(i)\nenddo", "per-iteration definition"),
        ("do i=1,n\n s=a(i)\nenddo\nt=s", "live after"),
        ("do i=1,n\n a(i)=1.0_knd\nenddo\nt=i", "live after"),
        ("do j=1,n\n do i=1,j\n a(i)=1.0_knd\n enddo\nenddo", "WAW"),
    ],
)
def test_rejection_keeps_existing_output_files(tmp_path, body, reason):
    source = tmp_path / "unsafe.f90"
    source.write_text(
        "! kernels\nmodule unsafe_module\n"
        "integer, parameter :: knd=kind(1.0d0)\ncontains\n! kernel\n"
        "subroutine entry(a,n)\nreal(knd), intent(inout) :: a(:)\n"
        "integer, intent(in) :: n\ninteger :: i,j\nreal(knd) :: s,t\n"
        f"{body}\nend subroutine entry\nend module unsafe_module\n"
    )
    output = tmp_path / "outputs"
    output.mkdir()
    before = {filename: f"keep existing {filename}\n" for filename in OUTPUTS}
    for filename, value in before.items():
        (output / filename).write_text(value)
    result = subprocess.run(
        [sys.executable, "-m", "compiler", "-i", str(source), "-k", "entry", "-o", str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode != 0
    assert reason in result.stderr
    assert f"{source}:" in result.stderr
    assert "Traceback" not in result.stderr
    if reason in {"RAW", "WAW"}:
        assert "relation" in result.stderr
        assert "witness" in result.stderr
        assert "accesses at" in result.stderr
    assert {path.name: path.read_text() for path in output.iterdir()} == before
