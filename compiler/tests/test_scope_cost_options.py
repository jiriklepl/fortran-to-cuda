"""Public source cost requests retain their distinct execution contracts."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from compiler.driver.cli import _offload_config
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig


@pytest.mark.parametrize(("cost_request", "execution"), [({}, "reached"), ({"coverage": "original_selection"}, "bounded")])
def test_cost_coverage_selects_only_its_execution_mode(cost_request, execution):
    for policy in ("auto", "sections"):
        assert OffloadConfig(policy=policy, scope_execution=execution, exact_window=cost_request).exact_window == cost_request
    other = "bounded" if execution == "reached" else "reached"
    with pytest.raises(ValueError, match="Exact window requires"):
        OffloadConfig(policy="auto", scope_execution=other, exact_window=cost_request)


@pytest.mark.parametrize("cost_request", [[], True, "original_selection", {"coverage": "unknown"}, {"coverage": []}])
def test_cost_request_rejects_unknown_schema(cost_request):
    with pytest.raises(ValueError, match="[Ee]xact window"):
        OffloadConfig(policy="auto", exact_window=cost_request)


def _args(path):
    return SimpleNamespace(scope_cost=str(path), calibration_profile=None, gpu_policy="auto", host_threads=4,
                           gpu_collective=False, scope_execution="bounded")


def test_cli_loads_cost_request_without_profile_or_changing_request(tmp_path):
    path = tmp_path / "cost.json"
    cost_request = {"coverage": "original_selection", "evidence": {"campaigns": []}}
    path.write_text(json.dumps(cost_request))
    config = _offload_config(_args(path))
    assert config.exact_window == cost_request
    assert config.profile is None


@pytest.mark.parametrize("contents", ["not json", "[]", "null", "true"])
def test_cli_rejects_invalid_cost_file(tmp_path, contents):
    path = tmp_path / "cost.json"
    path.write_text(contents)
    with pytest.raises(CompilationError, match="scope-cost"):
        _offload_config(_args(path))


def test_cli_reports_missing_cost_file(tmp_path):
    with pytest.raises(CompilationError, match="scope-cost"):
        _offload_config(_args(tmp_path / "missing.json"))


@pytest.mark.parametrize("extra", [[], ["--form-scopes"],
    ["--form-scopes", "--gpu-policy", "auto", "--gpu-collective"],
    ["--form-scopes", "--scope-execution", "reached", "--gpu-policy", "sections",
     "--scope-measurement", "native"]])
def test_cli_rejects_incompatible_cost_mode_without_touching_output(tmp_path, extra):
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "sentinel"
    sentinel.write_text("unchanged")
    result = subprocess.run([sys.executable, "-m", "compiler", "--scope-cost", "missing.json",
        "--input", "missing.f90", "--kernel", "advance", "--output-dir", str(output), *extra],
        cwd=Path(__file__).resolve().parents[2], text=True, capture_output=True, timeout=30)
    assert result.returncode != 0
    assert "scope-cost" in result.stderr
    assert list(output.iterdir()) == [sentinel]
    assert sentinel.read_text() == "unchanged"
