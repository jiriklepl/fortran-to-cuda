"""Addressing policy is independent of fusion and safe for legacy stage callers."""

import subprocess
import sys
from pathlib import Path

import pytest

from compiler.driver.options import CompilerOptions
from compiler.driver.pipeline import prepare_function
from compiler.frontend import lower_file
from compiler.ir import CompilationError
from compiler.scheduling import schedule_plan
from compiler.tests.test_language import lower
from compiler.transforms import optimize_function

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ("generated_code.cu", "generated_cpp_impl.cpp", "generated_interface.f90", "common_functions.cuh")
POLICIES = [
    (0, None, "source"),
    (1, None, "auto"),
    (0, "source", "source"),
    (0, "auto", "auto"),
    (1, "source", "source"),
    (1, "auto", "auto"),
]


@pytest.mark.parametrize(("opt_level", "indexing", "expected"), POLICIES)
def test_indexing_policy_defaults_and_explicit_overrides(tmp_path, opt_level, indexing, expected):
    options = CompilerOptions(opt_level=opt_level, indexing=indexing)
    assert options.resolved_indexing == expected
    function = lower(tmp_path, "do i=2,n\na(i)=i\nenddo")
    _, plan = prepare_function(function, options=options)
    decisions = plan.regions[0].addressing.decisions
    assert decisions
    assert any(decision.mode == "wide" for decision in decisions) == (expected == "auto")


@pytest.mark.parametrize("indexing", ["wide", "", 1])
def test_invalid_indexing_options_reject(indexing):
    with pytest.raises(CompilationError, match="indexing must be source or auto"):
        CompilerOptions(indexing=indexing)


@pytest.mark.parametrize("existing", [False, True])
def test_invalid_cli_indexing_leaves_outputs_intact(tmp_path, existing):
    function = lower(tmp_path, "do i=2,n\na(i)=i\nenddo")
    output = tmp_path / "generated"
    expected = {filename: f"original {filename}" for filename in OUTPUTS} if existing else {}
    if existing:
        output.mkdir()
        for filename, contents in expected.items():
            (output / filename).write_text(contents)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "compiler",
            "--input",
            function.source,
            "--kernel",
            "entry",
            "--output-dir",
            str(output),
            "--indexing",
            "unchecked",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode != 0
    assert "--indexing" in result.stderr
    assert ({path.name: path.read_text() for path in output.iterdir()} if output.exists() else {}) == expected
