"""Graph source compilation has one execution path and no window-cost knobs."""

import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import pytest

from compiler.driver.cli import _parse_args
from compiler.driver.options import CompilerOptions
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig


@pytest.mark.parametrize("extra", [["--scope-cost", "missing.json"], ["--scope-measurement", "native"],
                                   ["--scope-execution", "graph"], ["--memory-model", "scoped"]])
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
    assert CompilerOptions().source_graph is False
    with pytest.raises(CompilationError, match="boolean"):
        CompilerOptions(source_graph="graph")
    for obsolete in ("memory_model", "scope_execution"):
        with pytest.raises(TypeError, match=obsolete):
            CompilerOptions(**{obsolete: None})
    for obsolete in ("exact_window", "scope_measurement", "scope_execution"):
        with pytest.raises(TypeError, match=obsolete):
            OffloadConfig(**{obsolete: None})


@pytest.mark.parametrize("scopes", [["--form-scopes", "--scope-facts", "facts.json"], []])
def test_placement_defaults_to_sections_with_and_without_source_scopes(monkeypatch, scopes):
    monkeypatch.setattr(sys, "argv", ["compiler", *scopes, "--input", "source.f90", "--kernel", "advance"])
    args = _parse_args()
    assert (args.form_scopes, args.gpu_policy) == (bool(scopes), "sections")


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
