"""Offline observations cannot silently become placement or bounded execution."""

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from compiler._legacy.config import LegacyCompilerOptions as CompilerOptions
from compiler._legacy.config import LegacyOffloadConfig as OffloadConfig
from compiler.frontend.numerical_preparation import _Budget, _configuration


@pytest.mark.parametrize("updates", [
    {"scope_measurement": "auto"}, {"scope_measurement": []}, {"scope_measurement": True},
    {"policy": "auto"}, {"policy": "always"}, {"scope_execution": "bounded"},
    {"collective": True}, {"exact_window": {}},
])
def test_measurement_rejects_incompatible_execution_modes(updates):
    options = {"policy": "sections", "scope_execution": "reached", "scope_measurement": "native"}
    options.update(updates)
    with pytest.raises(ValueError, match="Scope measurement|Exact window"):
        OffloadConfig(**options)


def test_observation_variant_does_not_change_numerical_preparation_identity():
    config = OffloadConfig(policy="sections", scope_execution="reached")
    options = CompilerOptions(gpu_policy="sections", memory_model="scoped", scope_execution="reached")
    identities = [_configuration(options, replace(config, scope_measurement=variant), "original", {}, _Budget())
                  for variant in (None, "native", "resident")]
    assert len(set(identities)) == 1
    changed = _configuration(options, replace(config, host_threads=8), "original", {}, _Budget())
    assert changed != identities[0]


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
