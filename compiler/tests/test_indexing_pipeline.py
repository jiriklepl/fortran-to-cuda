"""Addressing policy is independent of fusion and safe for legacy stage callers."""

import subprocess
import sys
from pathlib import Path

import pytest

from compiler.driver.options import CompilerOptions
from compiler.driver.pipeline import prepare_function
from compiler.emission import generate_sources
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


@pytest.mark.parametrize(("opt_level", "indexing", "expected"), POLICIES)
def test_cli_exposes_indexing_policy_and_proof_reports(tmp_path, opt_level, indexing, expected):
    function = lower(tmp_path, "do i=2,n\na(i)=i\nenddo")
    output = tmp_path / "generated"
    command = [
        sys.executable,
        "-m",
        "compiler",
        "--input",
        function.source,
        "--kernel",
        "entry",
        "--output-dir",
        str(output),
        "--opt-level",
        str(opt_level),
        "--verbose",
    ]
    if indexing is not None:
        command.extend(["--indexing", indexing])
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "indexing" in result.stdout.lower()
    assert expected in result.stdout.lower()
    prepared, plan = prepare_function(function, options=CompilerOptions(opt_level=opt_level, indexing=indexing))
    sources = generate_sources(prepared, plan)
    assert (output / OUTPUTS[0]).read_text() == sources.cuda
    assert (output / OUTPUTS[1]).read_text() == sources.cpp
    assert (output / OUTPUTS[2]).read_text() == sources.fortran
    assert (output / OUTPUTS[3]).exists()


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


@pytest.mark.parametrize("opt_level", [0, 1])
@pytest.mark.parametrize("tiles", [(), (4,)])
def test_source_indexing_preserves_legacy_stage_output(tmp_path, opt_level, tiles):
    function = lower(tmp_path, "do i=2,n\na(i)=i\nenddo")
    options = CompilerOptions(opt_level=opt_level, tile_sizes=tiles, indexing="source")
    legacy_function, legacy_plan = optimize_function(function, options=options)
    legacy_plan = schedule_plan(legacy_function, legacy_plan, options=options)
    assert all(region.addressing is None for region in legacy_plan.regions)
    prepared, plan = prepare_function(function, options=options)
    assert generate_sources(prepared, plan) == generate_sources(legacy_function, legacy_plan)


@pytest.mark.parametrize("kernel", ["CDU", "CDV", "CDW"])
@pytest.mark.parametrize("opt_level", [0, 1])
def test_stencil_indexing_changes_no_interfaces_or_kernel_counts(kernel, opt_level):
    function = lower_file(ROOT / "fortran-stencils" / f"elmm_{kernel.lower()}.f90", kernel)
    baseline_function, baseline_plan = prepare_function(
        function, options=CompilerOptions(opt_level=opt_level, indexing="source")
    )
    prepared, plan = prepare_function(function, options=CompilerOptions(opt_level=opt_level, indexing="auto"))
    assert prepared == baseline_function
    assert len(plan.regions) == len(baseline_plan.regions) == (1 if opt_level else 4)
    decisions = [decision for region in plan.regions for decision in region.addressing.decisions]
    assert {decision.mode for decision in decisions} == {"wide", "source"}
    baseline = generate_sources(baseline_function, baseline_plan)
    sources = generate_sources(prepared, plan)
    assert sources.fortran == baseline.fortran
    assert sources.cuda != baseline.cuda
    assert sources.cpp != baseline.cpp
    assert sources.cuda.count("__global__") == baseline.cuda.count("__global__")
    assert sources.cuda.count("<<<") == baseline.cuda.count("<<<")
