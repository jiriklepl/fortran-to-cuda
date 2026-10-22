"""Missing cost evidence retains authenticated source without speculative work."""

from hashlib import sha256

import pytest

from compiler._legacy.config import LegacyCompilerOptions as CompilerOptions
from compiler._legacy.config import LegacyOffloadConfig as OffloadConfig
from compiler.ir import CompilationError
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_collective_source_scopes import generate as collective
from compiler.tests.test_numerical_sources import package
from compiler.tests.test_source_compute_costs import profile_v2
from compiler.tests.test_source_scopes import FACT, PROGRAM


def builder(tmp_path, *, execution="bounded", config=None, normalized=False, entry="step", corrupt=False):
    if normalized:
        path, lowered, document = package(tmp_path)
    else:
        path = lowered = tmp_path / "source.f90"
        path.write_text(PROGRAM)
        document = None
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {root: FACT for root in
                          ("argument::a", "argument::b", "argument::out", "settings::weights")}}
    if corrupt:
        facts["sources"][str(path)] = "0" * 64
    value = ScopeBuilder([path], entry, facts=facts, options=CompilerOptions(),
                         config=config or OffloadConfig("auto", scope_execution=execution),
                         numerical_sources=document)
    return path, lowered, value


def forbid(*args, **kwargs):
    pytest.fail("missing-cost native selection must not discover, lower, or summarize a closure")


def assert_native_only(outputs, report):
    assert set(outputs) == {"scope-manifest.json"}
    assert report["scope_count"] == 0
    assert not report["automatic_estimate_available"]
    for key in ("scopes", "source_edits", "sources", "build_sources", "artifacts_sha256", "numerical_decisions"):
        assert not report[key], key
    assert report["runtime"] is None
    assert report["native_effects"]["closure_materialization"] == "not_requested"
    assert "no calibration profile or exact-window evidence" in report["boundaries"][0]["reason"]


@pytest.mark.parametrize("execution", ["bounded", "reached"])
def test_missing_costs_skip_discovery_lowering_and_transitive_effects(tmp_path, monkeypatch, execution):
    path, _, candidate = builder(tmp_path, execution=execution)
    original = path.read_bytes()
    monkeypatch.setattr(candidate.inline, "prepare", forbid)
    monkeypatch.setattr(candidate, "numerical", forbid)
    monkeypatch.setattr(candidate.analysis, "summarize", forbid)
    monkeypatch.setattr("compiler.scopes.window_discovery.discover_reached_window", forbid)
    outputs, report = candidate.run()
    assert_native_only(outputs, report)
    assert path.read_bytes() == original


@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("execution", ["bounded", "reached"])
def test_missing_costs_do_not_bypass_final_source_authentication(tmp_path, execution, normalized):
    _, lowered, candidate = builder(tmp_path, execution=execution, normalized=normalized)
    lowered.write_text("! changed after source capture\n")
    with pytest.raises(CompilationError, match="changed"):
        candidate.run()
    assert candidate.outputs == {}


@pytest.mark.parametrize(("changes", "message"), [({"corrupt": True}, "source hashes"),
                                             ({"entry": "missing"}, "unavailable or ambiguous")])
def test_missing_costs_do_not_bypass_input_authentication(tmp_path, changes, message):
    with pytest.raises(CompilationError, match=message):
        builder(tmp_path, **changes)


@pytest.mark.parametrize("kind", ["forced", "profile", "exact_request"])
def test_independent_cost_evidence_and_forced_diagnostics_reach_discovery(tmp_path, monkeypatch, kind):
    class ReachedDiscovery(Exception):
        pass

    def reached(*args):
        raise ReachedDiscovery

    config = OffloadConfig("sections" if kind == "forced" else "auto",
                           profile=profile_v2() if kind == "profile" else None,
                           exact_window={} if kind == "exact_request" else None,
                           scope_execution="reached")
    _, _, candidate = builder(tmp_path, config=config)
    monkeypatch.setattr("compiler.scopes.window_discovery.discover_reached_window", reached)
    with pytest.raises(ReachedDiscovery):
        candidate.run()


def test_incompatible_profile_reason_is_preserved(tmp_path):
    config = OffloadConfig("auto", profile={"schema_version": 0})
    assert config.profile is None
    _, _, candidate = builder(tmp_path, config=config)
    outputs, report = candidate.run()
    assert_native_only(outputs, report)
    assert config.profile_reason in report["boundaries"][0]["reason"]


def test_collective_missing_costs_keep_authenticated_participation_without_lowering(tmp_path, monkeypatch):
    monkeypatch.setattr("compiler.scopes.collective.CollectiveScopeBuilder.numerical", forbid)
    _, outputs, report, _ = collective(tmp_path, mode="auto")
    assert_native_only(outputs, report)
    assert report["participation"]["kind"] == "omp_full_team"


def test_collective_missing_costs_still_reject_forged_participation(tmp_path):
    with pytest.raises(CompilationError, match="caller identity"):
        collective(tmp_path, mode="auto", facts_change=lambda facts:
                   facts["participation"]["call_sites"][0].update(caller="wrong::caller"))
