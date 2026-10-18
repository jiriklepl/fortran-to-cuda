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
