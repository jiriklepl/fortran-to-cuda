"""Line teams retain exact original authority on the existing source backend."""

from copy import copy
from dataclasses import replace

import pytest

from compiler.driver.options import CompilerOptions
from compiler.emission.cuda.tile_task_provider import emit_tile_task_provider
from compiler.frontend.line_snapshot_team import prove_line_snapshot_team
from compiler.frontend.source_device_tasks import source_device_task
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_graph_source_owner import FACT
from compiler.tests.test_line_snapshot import fixture, selections, source


def source_builder(tmp_path, text=None):
    path, analysis, routine, _nodes = fixture(tmp_path, source() if text is None else text)
    facts = {
        "schema_version": 1,
        "participation": "serial",
        "sources": dict(analysis.sources),
        "captures": {"argument::" + name: dict(FACT) for name in ("a", "mask", "nx", "ny", "nz")},
    }
    builder = ScopeBuilder(
        [path],
        routine.qualified,
        facts=facts,
        options=CompilerOptions(),
        config=OffloadConfig(policy="sections", scope_execution="graph"),
    )
    return builder, selections(builder.entry)[0]


@pytest.mark.parametrize("axis", [1, 2, 3])
def test_complete_line_team_has_distinct_domain_and_exact_completion(tmp_path, axis):
    _, analysis, routine, nodes = fixture(tmp_path, source(axis))
    team = prove_line_snapshot_team(analysis, routine, nodes)
    assert team.validate() is team
    assert prove_line_snapshot_team(analysis, routine, nodes) is team
    assert team.units[0].validate() is team.units[0]
    assert len(team.units) == 1
    assert "domain" not in team.public()
    assert len(team.public()["transverse_domain"]) == 2
    assert "width" not in str(team.public()["transverse_domain"])
    assert team.public()["cooperative_mapping_authorized"] is False
    for forged in (copy(team), replace(team), replace(team, identity="0" * 64)):
        with pytest.raises(CompilationError, match="registered exact"):
            forged.validate()
    with pytest.raises(CompilationError):
        team.validate(nodes=nodes[:-1])


def test_source_attachment_requires_initialized_consumer_bound(tmp_path):
    _, analysis, routine, nodes = fixture(tmp_path)
    with pytest.raises(CompilationError, match="whole-defined capture authority"):
        source_device_task(analysis, routine, nodes)
    builder, nodes = source_builder(tmp_path)
    issued = source_device_task(builder.analysis, builder.entry, nodes, builder=builder)
    assert issued.validate() is issued
    assert issued.public()["kind"] == "original_line_snapshot_team"
    assert issued.public()["definitions_authorized"] is False
    assert [(b.name, w) for b, w in issued.fields] == [("a", True), ("mask", False)]
    builder.facts["captures"]["argument::nx"]["initialized"] = "none"
    with pytest.raises(CompilationError, match="whole-defined"):
        issued.validate()


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"mapping": "cooperative", "access_envelopes": True},
        {"mapping": "thread"},
        {"mapping": "thread", "access_envelopes": True, "queued_single": True},
        {"mapping": "thread", "access_envelopes": True, "batch_target_bytes": 16 * 1024 * 1024},
    ],
)
def test_line_team_rejects_unqualified_mapping_transport_and_budget_policy(tmp_path, options):
    _, analysis, routine, nodes = fixture(tmp_path)
    team = prove_line_snapshot_team(analysis, routine, nodes)
    with pytest.raises(CompilationError, match="line snapshot first slice"):
        emit_tile_task_provider(team, **options)


def test_line_provider_uses_original_metadata_and_ordinary_transaction(tmp_path):
    builder, nodes = source_builder(tmp_path)
    issued = source_device_task(builder.analysis, builder.entry, nodes, builder=builder)
    provider = emit_tile_task_provider(issued.team, mapping="thread", access_envelopes=True, readable_scalars=issued)
    assert provider.public()["kind"] == "original_line_snapshot_team"
    assert provider.public()["transaction"] == "fort_scope_source_execute_v1"
    assert provider.public()["line_snapshot_prework"]["initialization_authorized"] is False
    assert "fort_scope_scratch_acquire_v1" in provider.cuda_source
    assert "fort_scope_source_execute_v1" in provider.cuda_source
    assert "prepare_tile_access(context,state,bindings,accesses)" in provider.cuda_source
    assert provider.cuda_source.index(
        "prepare_tile_access(context,state,bindings,accesses)"
    ) < provider.cuda_source.index("fort_scope_scratch_acquire_v1")
    assert "kernel_tile_private_0<<<" in provider.cuda_source
    assert "kernel_tile_private_cta" not in provider.cuda_source


def test_generated_source_wrapper_keeps_nested_line_activation_before_descriptors(tmp_path):
    builder, _ = source_builder(tmp_path)
    outputs, report = builder.run()
    assert report["scope_count"] == 1, report
    (scope,) = report["scopes"]
    (task,) = scope["source_device_tasks"]
    assert task["kind"] == "original_line_snapshot_team"
    rewritten = "\n".join(
        value for name, value in outputs.items() if name.endswith(".f90") and "fort_source_task_" in value
    )
    assert rewritten
    first = rewritten.index("if ((1) <= (nz)) then")
    second = rewritten.index("if ((1) <= (ny)) then", first)
    descriptor = rewritten.index("lbound(", second)
    assert first < second < descriptor
    assert "scratch_axis_order" not in str(task)


def test_line_team_registry_forgery_and_discarded_completion_reject(tmp_path):
    _, analysis, routine, nodes = fixture(tmp_path)
    team = prove_line_snapshot_team(analysis, routine, nodes)
    for forged in (replace(team, identity="0" * 64), replace(team, fields=()), replace(team, scalars=())):
        analysis._line_snapshot_team_proofs[forged.identity] = forged
        with pytest.raises(CompilationError):
            forged.validate()
        analysis._line_snapshot_team_proofs[team.identity] = team
    analysis._joined_completions.clear()
    with pytest.raises(CompilationError):
        team.validate()
    assert not analysis._joined_completions


def chain_source(*, repeat_first=False):
    groups = []
    for axis in (1, 2, 3):
        text = source(axis)
        start = text.index("!$omp parallel do")
        stop = text.index("!$omp end parallel do") + len("!$omp end parallel do")
        groups.append(text[start:stop].replace("tmp", "tmp" + str(axis)))
    if repeat_first:
        groups.insert(1, groups[0])
    text = source().replace("advance(a,mask,nx,ny,nz)", "advance(a,b,mask,nx,ny,nz)")
    text = text.replace("integer,intent(in)::mask", "real(8),intent(inout)::b(-2:,-2:,-2:)\ninteger,intent(in)::mask")
    text = text.replace(
        "real(8)::tmp(-2:ubound(a,1)),s,q",
        "real(8)::tmp1(-2:ubound(a,1)),tmp2(-2:ubound(a,2)),tmp3(-2:ubound(a,3)),s,q",
    )
    start = text.index("!$omp parallel do")
    stop = text.index("!$omp end parallel do") + len("!$omp end parallel do")
    simple = """!$omp parallel do private(i,j,k) shared(a,b,nx,ny,nz) schedule(runtime)
do k=1,nz
 do j=1,ny
  do i=1,nx
   WORK
  enddo
 enddo
enddo
!$omp end parallel do"""
    return (
        text[:start]
        + "\n".join(
            [simple.replace("WORK", "a(i,j,k)=a(i,j,k)+1"), *groups, simple.replace("WORK", "b(i,j,k)=2*a(i,j,k)")]
        )
        + text[stop:]
    )


def emit_chain(tmp_path, *, repeat_first=False):
    from compiler.frontend.source_effects import SourceEffects

    path = tmp_path / "source.f90"
    path.write_text(chain_source(repeat_first=repeat_first))
    analysis = SourceEffects([path])
    routine = analysis.routines["lines::advance"]
    facts = {
        "schema_version": 1,
        "participation": "serial",
        "sources": dict(analysis.sources),
        "captures": {"argument::" + name: dict(FACT) for name in ("a", "b", "mask", "nx", "ny", "nz")},
    }
    builder = ScopeBuilder(
        [path],
        routine.qualified,
        facts=facts,
        options=CompilerOptions(),
        config=OffloadConfig(policy="sections", scope_execution="graph"),
    )
    outputs, report = builder.run()
    return path, outputs, report


def test_original_three_axis_chain_emits_one_owner_with_two_numerical_regions(tmp_path):
    _path, _outputs, report = emit_chain(tmp_path)
    assert report["scope_count"] == 1, report
    (scope,) = report["scopes"]
    assert len(scope["source_device_tasks"]) == 3
    assert all(task["kind"] == "original_line_snapshot_team" for task in scope["source_device_tasks"])
    assert len(scope["planning_segments"]) == 2


def test_repeated_axis_keeps_four_original_line_tasks_and_both_numerical_endpoints(tmp_path):
    _path, _outputs, report = emit_chain(tmp_path, repeat_first=True)
    assert report["scope_count"] == 1
    (scope,) = report["scopes"]
    assert len(scope["source_device_tasks"]) == 4
    assert len(scope["planning_segments"]) == 2
    tasks = scope["source_device_tasks"]
    assert len({task["source_proof_identity"] for task in tasks}) == 4
    assert all(task["kind"] == "original_line_snapshot_team" for task in tasks)
    assert all(
        task["execution_binding"]["canonical_resources"]
        == {"argument::a": "argument::a", "argument::mask": "argument::mask"}
        for task in tasks
    )
