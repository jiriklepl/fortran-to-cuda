"""Graph source compilation has one execution path and no window-cost knobs."""

import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import pytest

from compiler._legacy.config import LegacyOffloadConfig
from compiler.driver.cli import _parse_args
from compiler.driver.options import CompilerOptions
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig


@pytest.mark.parametrize(("cost_request", "execution"), [({}, "reached"), ({"coverage": "original_selection"}, "bounded")])
def test_cost_coverage_selects_only_its_execution_mode(cost_request, execution):
    for policy in ("auto", "sections"):
        assert LegacyOffloadConfig(policy=policy, scope_execution=execution, exact_window=cost_request).exact_window == cost_request
    other = "bounded" if execution == "reached" else "reached"
    with pytest.raises(ValueError, match="Exact window requires"):
        LegacyOffloadConfig(policy="auto", scope_execution=other, exact_window=cost_request)


@pytest.mark.parametrize("cost_request", [[], True, "original_selection", {"coverage": "unknown"}, {"coverage": []}])
def test_cost_request_rejects_unknown_schema(cost_request):
    with pytest.raises(ValueError, match="[Ee]xact window"):
        LegacyOffloadConfig(policy="auto", exact_window=cost_request)


@pytest.mark.parametrize("extra", [["--scope-cost", "missing.json"], ["--scope-measurement", "native"],
                                   ["--scope-execution", "bounded"], ["--scope-execution", "reached"]])
def test_cli_rejects_removed_flags_without_touching_output(tmp_path, extra):
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "sentinel"
    sentinel.write_text("unchanged")
    result = subprocess.run([sys.executable, "-m", "compiler", "--input", "missing.f90", "--kernel", "advance",
        "--form-scopes", "--scope-facts", "missing.json", "--output-dir", str(output), *extra],
        cwd=Path(__file__).resolve().parents[2], text=True, capture_output=True, timeout=30)
    assert result.returncode != 0
    assert extra[0] in result.stderr
    assert "unrecognized arguments" in result.stderr or "invalid choice" in result.stderr
    assert list(output.iterdir()) == [sentinel]
    assert sentinel.read_text() == "unchanged"


def test_public_configuration_has_no_legacy_execution_or_window_cost_constructors():
    assert OffloadConfig().scope_execution == "graph"
    assert CompilerOptions().scope_execution is None
    for execution in ("bounded", "reached"):
        with pytest.raises(ValueError, match="graph execution"):
            OffloadConfig(scope_execution=execution)
        with pytest.raises(CompilationError, match="graph execution"):
            CompilerOptions(gpu_policy="sections", memory_model="scoped", scope_execution=execution)
    for obsolete in ("exact_window", "scope_measurement"):
        with pytest.raises(TypeError, match=obsolete):
            OffloadConfig(**{obsolete: None})


def test_form_scopes_defaults_to_graph_scoped_memory_and_sections(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["compiler", "--form-scopes", "--scope-facts", "facts.json",
                                     "--input", "source.f90", "--kernel", "advance"])
    args = _parse_args()
    assert (args.scope_execution, args.memory_model, args.gpu_policy) == ("graph", "scoped", "sections")


def test_standalone_defaults_retain_numerical_call_interface(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["compiler", "--input", "source.f90", "--kernel", "advance"])
    args = _parse_args()
    assert (args.scope_execution, args.memory_model, args.gpu_policy) == (None, "call", "always")


def test_default_source_cli_emits_graph_artifacts(tmp_path):
    from compiler.tests.test_graph_source_owner import FACT, SOURCE

    source = tmp_path / "graph.f90"
    source.write_text(SOURCE)
    facts = tmp_path / "facts.json"
    facts.write_text(json.dumps({"schema_version": 1, "participation": "serial",
                                "sources": {str(source): sha256(source.read_bytes()).hexdigest()},
                                "captures": {"argument::" + name: dict(FACT) for name in ("a", "b", "x")}}))
    output = tmp_path / "output"
    result = subprocess.run([sys.executable, "-m", "compiler", "--form-scopes", "--scope-facts", str(facts),
                             "--input", str(source), "--kernel", "graph_fixture::advance", "--json",
                             "--output-dir", str(output)], cwd=Path(__file__).resolve().parents[2],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["supported"]
    assert report["scopes"]["scope_execution"] == "graph"
    assert report["scopes"]["scope_count"] == 1
    assert (output / "scope-manifest.json").is_file()
