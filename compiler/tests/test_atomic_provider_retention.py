"""Retained provider data cannot replace the original proof/policy checks."""
from copy import copy
from dataclasses import replace

import pytest

from compiler.driver.options import CompilerOptions
from compiler.emission.cuda.atomic_team_provider import emit_atomic_team_provider
from compiler.frontend.atomic_worksharing_team import atomic_worksharing_alternatives, prove_atomic_worksharing_team
from compiler.frontend.numerical_effects import prove_numerical_effects
from compiler.frontend.numerical_preparation import NumericalPreparationProvider
from compiler.frontend.reached_read_requirements import prove_reached_read_requirements
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.atomic_joined_team import AtomicAlternative
from compiler.tests.test_atomic_worksharing_team import ENTRY, fixture


def retained(tmp_path, *, profile=None):
    path, analysis, routine, nodes = fixture(tmp_path)
    analysis.seal_session()
    team = prove_atomic_worksharing_team(analysis, routine, nodes)
    alternative = atomic_worksharing_alternatives(team)[0]
    policy = NumericalPreparationProvider(analysis, options=CompilerOptions(opt_level=0),
        config=OffloadConfig(policy="sections", host_threads=4, profile=profile), environment_root=ENTRY)
    prepared = policy.prepare_atomic_alternative(alternative)
    effects = prove_numerical_effects(analysis, ENTRY, nodes, prepared)
    reads = prove_reached_read_requirements(analysis, ENTRY, nodes, prepared)
    artifact = emit_atomic_team_provider(alternative, prepared, effects, reads)
    return path, analysis, AtomicAlternative(alternative, prepared, effects, reads, artifact, ())


def test_retained_provider_matches_fresh_emission_and_does_not_emit_again(tmp_path, monkeypatch):
    from compiler.scopes import atomic_joined_team as scope

    _, analysis, item = retained(tmp_path)
    fresh = emit_atomic_team_provider(item.source, item.prepared, item.effects, item.reads)
    assert item.artifact.cuda_source == fresh.cuda_source
    assert item.artifact.fortran_source == fresh.fortran_source
    assert item.artifact.public() == fresh.public()

    def unexpected(*_args):
        raise AssertionError("retained artifact should not be generated again")

    monkeypatch.setattr(scope, "emit_atomic_team_provider", unexpected)
    for _ in range(3):
        assert item.provider() is item.artifact
    analysis.verify_publication()


@pytest.mark.parametrize("change", ["cuda", "fortran", "record", "fields", "parameters", "foreign_alternative", "prepared_copy"])
def test_changed_artifact_or_exact_inputs_are_rejected(tmp_path, change):
    _, _, item = retained(tmp_path)
    if change == "cuda":
        item = replace(item, artifact=replace(item.artifact, cuda_source="altered CUDA"))
    elif change == "fortran":
        item = replace(item, artifact=replace(item.artifact, fortran_source="altered interface"))
    elif change == "record":
        item = replace(item, artifact=replace(item.artifact, _record="{}"))
    elif change == "fields":
        item = replace(item, artifact=replace(item.artifact, fields=(copy(item.artifact.fields[0]), *item.artifact.fields[1:])))
    elif change == "parameters":
        item = replace(item, artifact=replace(item.artifact, parameters=(replace(item.artifact.parameters[0], name="other"),
                                                                       *item.artifact.parameters[1:])))
    elif change == "foreign_alternative":
        item = replace(item, source=atomic_worksharing_alternatives(item.source.team)[1])
    else:
        item = replace(item, prepared=copy(item.prepared))
    with pytest.raises(CompilationError):
        item.provider()


@pytest.mark.parametrize("registry", ["_atomic_worksharing_teams", "_joined_completions", "_numerical_preparations",
                                     "_numerical_effect_proofs", "_reached_read_requirements_proofs"])
def test_retained_bytes_do_not_heal_removed_receipts(tmp_path, registry):
    _, analysis, item = retained(tmp_path)
    token = {"_atomic_worksharing_teams": item.source.team, "_joined_completions": item.source.team.parent,
             "_numerical_preparations": item.prepared, "_numerical_effect_proofs": item.effects,
             "_reached_read_requirements_proofs": item.reads}[registry]
    del getattr(analysis, registry)[token.identity]
    with pytest.raises(CompilationError):
        item.provider()


def test_mutable_profile_is_rehashed_before_retained_provider_use(tmp_path):
    from compiler.numerical_contract import numerical_build_contract

    profile = {"numerical_contract": numerical_build_contract(), "raw": [1, 2]}
    _, _, item = retained(tmp_path, profile=profile)
    assert item.provider() is item.artifact
    profile["raw"].append(3)
    with pytest.raises(CompilationError, match="policy authority"):
        item.provider()


def test_stale_source_policy_and_changed_live_source_remain_separate_checks(tmp_path):
    path, analysis, item = retained(tmp_path)
    path.write_text(path.read_text() + "! changed input\n")
    assert item.provider() is item.artifact
    with pytest.raises(CompilationError, match="source changed"):
        analysis.verify_publication()
    analysis.operation_limit += 1
    with pytest.raises(CompilationError):
        item.provider()


def test_source_owner_emits_each_original_provider_once_and_reader_accepts(tmp_path, monkeypatch):
    import json
    from collections import Counter

    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans

    from compiler.scopes import atomic_joined_team as scope
    from compiler.tests.test_atomic_source_call_bindings import atomic_builder

    original, calls = scope.emit_atomic_team_provider, []

    def counted(*args):
        calls.append(tuple(id(item) for item in args))
        return original(*args)

    monkeypatch.setattr(scope, "emit_atomic_team_provider", counted)
    builder = atomic_builder(tmp_path)
    outputs, report = builder.run()
    assert report["scope_count"] == 1, report.get("graph_owner_attempt")
    assert calls
    assert set(Counter(calls).values()) == {1}
    validate_checked_execution_plans(json.loads(outputs["scope-manifest.json"]), capture_facts=builder.facts)
