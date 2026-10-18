"""Discovery and emission retain the same mixed-team numerical candidates."""

from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.frontend.native_group_structure import NativeGroupCandidate, grouped_original_sequence
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.scopes.worksharing_inventory import prepare_worksharing_inventory
from compiler.tests.test_joined_team_source import SOURCE
from compiler.tests.test_source_scopes import FACT


def builder_for(directory, source=SOURCE):
    path = directory / "mixed.f90"
    path.write_text(source)
    return ScopeBuilder([path], "joined_owner::step", options=CompilerOptions(),
        config=OffloadConfig(policy="sections", scope_execution="reached", host_threads=4),
        facts={"schema_version": 1, "participation": "serial",
               "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
               "captures": {"argument::" + name: FACT for name in ("a", "b", "out")}})


def originals(builder):
    group, = [item for item in grouped_original_sequence(builder.entry.execution.content, {})
              if isinstance(item, NativeGroupCandidate)]
    return group.original_nodes


def test_source_inventory_keeps_native_correction_without_emitting_artifacts(tmp_path):
    builder = builder_for(tmp_path)
    _parent, candidates = prepare_worksharing_inventory(builder, builder.entry.qualified, originals(builder))
    assert [candidate.prepared is not None for candidate in candidates] == [True, False, True]
    assert "array section requires bounded constant cardinality" in candidates[1].reason
    assert "sum(a)" in str(candidates[1].unit[1]).lower()
    assert not builder.inline.regions
    assert not builder.outputs
    assert not builder.edits
    for candidate in (candidates[0], candidates[2]):
        prepared = candidate.prepared
        assert prepared.validate(builder.analysis, builder.entry.qualified, (candidate.unit[1],)) is prepared
        assert prepared.completion_proof is candidate.completion


def test_interval_and_emitter_share_mixed_team_preparations(tmp_path):
    builder = builder_for(tmp_path)
    _outputs, report = builder.run()
    scope, = report["scopes"]
    assert not scope["boundaries"]
    assert scope["checked_execution_plan"]["numerical_invocations"] == 2
    draft, = builder._interval_drafts.values()
    supported = [anchor for anchor in draft.numerical_anchors if anchor.prepared is not None]
    assert len(supported) == 2
    emitted = builder.inline.prepared_regions.values()
    assert all(any(anchor.prepared is prepared for prepared in emitted) for anchor in supported)
    assert all(anchor.prepared.completion_proof.public()["proof_role"]
               == "original_team_worksharing_completion" for anchor in supported)
    assert draft.validate(draft._owner.execution_plan) is draft


def test_unrelated_nowait_prevents_partial_team_inventory(tmp_path):
    source = SOURCE.replace("!$omp end do", "!$omp end do nowait", 1)
    builder = builder_for(tmp_path, source)
    with pytest.raises(CompilationError, match="NOWAIT"):
        prepare_worksharing_inventory(builder, builder.entry.qualified, originals(builder))
    assert not builder.inline.regions
    assert not getattr(builder.analysis, "_numerical_preparations", {})


def test_discovery_and_admission_prepare_each_original_unit_once(tmp_path, monkeypatch):
    from compiler.frontend import worksharing_completion
    from compiler.frontend.numerical_preparation import NumericalPreparationProvider

    builder = builder_for(tmp_path)
    prepared, teams = [], []
    original_prepare = NumericalPreparationProvider.prepare_worksharing
    original_team = worksharing_completion._joined_completion_facts

    def prepare(self, analysis, procedure, loop, completion, **kwargs):
        prepared.append(loop)
        return original_prepare(self, analysis, procedure, loop, completion, **kwargs)

    def team(*args, **kwargs):
        teams.append(id(args[0]))
        return original_team(*args, **kwargs)

    monkeypatch.setattr(NumericalPreparationProvider, "prepare_worksharing", prepare)
    monkeypatch.setattr(worksharing_completion, "_joined_completion_facts", team)
    _, report = builder.run()
    assert report["scopes"][0]["checked_execution_plan"]["numerical_invocations"] == 2
    assert len(prepared) == len({id(loop) for loop in prepared}) == 3
    assert teams.count(id(builder.analysis)) == 1
    assert len(teams) == len(set(teams))
    inventories = tuple(builder.analysis._prepared_worksharing_inventories.values())
    assert len(inventories) == 1
    inventory, = inventories
    assert inventory.source.units == tuple(candidate.unit for candidate in inventory.candidates)
    assert [candidate.prepared is not None for candidate in inventory.candidates] == [True, False, True]


def test_reused_inventory_revalidates_prepared_payloads(tmp_path):
    builder = builder_for(tmp_path)
    selection = originals(builder)
    _, candidates = prepare_worksharing_inventory(builder, builder.entry.qualified, selection)
    candidate = candidates[0]
    candidate.prepared.extraction.completion["caller_contract"] = "forged participation"
    with pytest.raises(CompilationError, match="(authority|completion)"):
        prepare_worksharing_inventory(builder, builder.entry.qualified, selection)


def test_policy_change_requires_fresh_ordered_preparation(tmp_path, monkeypatch):
    from dataclasses import replace

    from compiler.frontend.numerical_preparation import NumericalPreparationProvider

    builder = builder_for(tmp_path)
    selection = originals(builder)
    first_parent, first = prepare_worksharing_inventory(builder, builder.entry.qualified, selection)
    calls = []
    original = NumericalPreparationProvider.prepare_worksharing

    def prepare(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(NumericalPreparationProvider, "prepare_worksharing", prepare)
    _, reused = prepare_worksharing_inventory(builder, builder.entry.qualified, selection)
    assert reused is first
    assert not calls
    builder.config = replace(builder.config, host_threads=2)
    parent, changed = prepare_worksharing_inventory(builder, builder.entry.qualified, selection)
    assert parent is first_parent
    assert changed is not first
    assert len(calls) == 3
    assert changed[0].prepared._provider.identity != first[0].prepared._provider.identity


def test_copied_or_stale_prepared_inventory_cannot_be_reused(tmp_path):
    from dataclasses import replace

    builder = builder_for(tmp_path)
    selection = originals(builder)
    prepare_worksharing_inventory(builder, builder.entry.qualified, selection)
    inventory, = builder.analysis._prepared_worksharing_inventories.values()
    with pytest.raises(CompilationError, match="registered source and policy"):
        replace(inventory).validate(builder.analysis, builder.entry.qualified, selection, inventory.provider)
    path = tmp_path / "mixed.f90"
    path.write_text(path.read_text().replace("!$omp end do", "!$omp end do nowait", 1))
    # Compilation uses its immutable original session; publication must still
    # reject a changed live source before any edit or artifact can be issued.
    prepare_worksharing_inventory(builder, builder.entry.qualified, selection)
    with pytest.raises(CompilationError, match="changed"):
        builder.analysis.verify_publication()


def test_private_projection_cannot_borrow_or_mutate_inventory_authority(tmp_path):
    from copy import copy

    from compiler.frontend.worksharing_completion import ordered_worksharing_inventory
    from compiler.scopes.segments import _native_analysis

    builder = builder_for(tmp_path)
    selection = originals(builder)
    prepare_worksharing_inventory(builder, builder.entry.qualified, selection)
    source, = builder.analysis._ordered_worksharing_inventories.values()
    preparation, = builder.analysis._prepared_worksharing_inventories.values()
    saved_source = dict(builder.analysis._ordered_worksharing_inventories)
    saved_prepared = dict(builder.analysis._prepared_worksharing_inventories)
    # Even a plain shallow projection, without the native fork's explicit
    # reset list, must issue its own capabilities and private registries.
    for projected in (builder.analysis.private_projection(), _native_analysis(builder.analysis)):
        with pytest.raises(CompilationError, match="original-team authority"):
            source.validate(projected, builder.entry.qualified, selection)
        proxy = copy(builder)
        proxy.analysis = projected
        fresh = ordered_worksharing_inventory(projected, builder.entry.qualified, selection)
        assert fresh is not source
        if projected._worksharing_completions:
            # A bare shallow projection has not detached/reissued its other
            # numerical capabilities. It must reject those foreign tokens.
            source_completion = next(iter(projected._worksharing_completions.values()))
            with pytest.raises(CompilationError, match="original-team authority"):
                source_completion.validate(projected, builder.entry.qualified, (source_completion.loop,))
        else:
            prepare_worksharing_inventory(proxy, builder.entry.qualified, selection)
            fresh_preparation, = projected._prepared_worksharing_inventories.values()
            assert fresh_preparation is not preparation
            assert projected._prepared_worksharing_inventories is not builder.analysis._prepared_worksharing_inventories
        assert projected._ordered_worksharing_inventories is not builder.analysis._ordered_worksharing_inventories
        assert builder.analysis._ordered_worksharing_inventories == saved_source
        assert builder.analysis._prepared_worksharing_inventories == saved_prepared
