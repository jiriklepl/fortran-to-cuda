"""Current graph transport and explicit frozen-control compatibility."""

import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from elmm_pipeline import cli, local_adapter
from elmm_pipeline._legacy_scopes import transform_scopes as historical_scopes
from elmm_pipeline.common import COMPILER_PHASE_TIMEOUT_SECONDS
from test_local_scopes import FACTS, PROGRAM


def test_compiler_phase_cap_is_240_seconds():
    assert COMPILER_PHASE_TIMEOUT_SECONDS == 240


@pytest.mark.parametrize("options", [
    {"scope_execution": "bounded"}, {"scope_execution": "reached"},
    {"scope_cost": None}, {"scope_measurement": None},
    {"scope_cost": "missing.json"}, {"scope_measurement": "native"},
])
def test_public_adapter_rejects_retired_options_before_creating_artifacts(tmp_path, options):
    output = tmp_path / "output"
    with (patch.object(local_adapter, "_invoke", side_effect=AssertionError("no compiler invocation")),
          pytest.raises(local_adapter.LocalAdapterError, match="[Rr]etired|graph execution")):
        local_adapter.transform_scopes(tmp_path / "absent.f90", tmp_path, Path(sys.executable),
                                      output, entry="step", capture_facts=FACTS, gpu_options=options)
    assert not output.exists()


@pytest.mark.parametrize("extra", [["--scope-execution", "bounded"], ["--scope-execution", "reached"],
                                   ["--scope-cost", "absent.json"], ["--scope-measurement", "native"]])
def test_public_cli_rejects_retired_arguments_before_run(monkeypatch, extra):
    monkeypatch.setattr(sys, "argv", ["pipeline", "run", *extra])
    with (patch.object(cli, "run", side_effect=AssertionError("no application build")),
          pytest.raises(SystemExit) as error):
        cli.main()
    assert error.value.code == 2


def test_public_cli_accepts_graph_and_preserves_ordinary_call_default(monkeypatch):
    for extra, expected in [([], None), (["--scope-execution", "graph"], "graph")]:
        monkeypatch.setattr(sys, "argv", ["pipeline", "run", *extra])
        with patch.object(cli, "run") as run:
            cli.main()
        assert run.call_args.args[0].scope_execution == expected


def test_scoped_configuration_derives_graph_without_retired_fields(tmp_path):
    facts = tmp_path / "facts.json"
    facts.write_text(json.dumps(FACTS))
    args = SimpleNamespace(memory_model="scoped", target="gpu", tools=["baseline", "local"],
                           gpu_policy="auto", scope_entry="step", scope_source="application.f90",
                           scope_facts=facts)
    assert cli.scoped_configuration(args)["execution"] == "graph"
    assert not {"measurement", "cost"} & cli.scoped_configuration(args).keys()
    assert cli.scoped_configuration(SimpleNamespace(memory_model="call")) is None


@pytest.mark.parametrize(("historical", "response_execution", "error"), [
    (False, "bounded", "non-graph"), (True, "graph", "Historical compiler source protocol"),
])
def test_source_transport_cannot_accept_the_other_compiler_protocol(tmp_path, historical, response_execution, error):
    source = tmp_path / "application.f90"
    source.write_text(PROGRAM)
    hashes = {str(source): hashlib.sha256(source.read_bytes()).hexdigest()}
    facts = {**FACTS, "sources": hashes}
    manifest = {"schema_version": 1, "scope_execution": response_execution, "scope_count": 0,
                "scopes": [], "source_inputs": hashes, "analysis_sources": None,
                "capture_facts_sha256": hashlib.sha256(json.dumps(facts, sort_keys=True).encode()).hexdigest(),
                "artifacts_sha256": {}}
    response = {"supported": True, "scopes": manifest, "outputs": []}
    adapter = historical_scopes if historical else local_adapter.transform_scopes
    with (patch.object(local_adapter, "_invoke", return_value=(response, [])),
          pytest.raises(local_adapter.LocalAdapterError, match=error)):
        adapter(source, tmp_path, Path(sys.executable), tmp_path / "output", entry="step", capture_facts=FACTS)
    assert source.read_text() == PROGRAM


@pytest.mark.parametrize(("historical", "policy"), [(False, "auto"), (True, "auto"),
                                                  (False, "sections"), (True, "sections")])
def test_current_graph_and_frozen_be58_use_explicit_distinct_transports(tmp_path, historical, policy):
    key = "ELMM_HISTORICAL_COMPILER_ROOT" if historical else "ELMM_TEST_COMPILER_ROOT"
    compiler = os.environ.get(key)
    if not compiler:
        pytest.skip("Set " + key + " for the actual compiler protocol test")
    source = tmp_path / "application.f90"
    source.write_text(PROGRAM)
    adapter = historical_scopes if historical else local_adapter.transform_scopes
    metadata = adapter(source, Path(compiler), Path(sys.executable), tmp_path / "output",
                       entry="step", capture_facts=FACTS, gpu_options={"gpu_policy": policy})
    assert metadata["scope_manifest"]["scope_execution"] == ("bounded" if historical else "graph")
    if policy == "auto":
        assert metadata["scope_manifest"]["scope_count"] == 0
        assert metadata["gpu_sources"] == metadata["extra_sources"] == []
    else:
        assert metadata["scope_manifest"]["scope_count"] > 0
        assert metadata["gpu_sources"] and metadata["extra_sources"]
    assert source.read_text() == PROGRAM
    if historical:
        assert "--scope-execution" not in metadata["command"]
    else:
        assert metadata["command"][metadata["command"].index("--scope-execution") + 1] == "graph"
