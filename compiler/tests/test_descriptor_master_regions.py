"""Original MASTER regions retain bodies and joins, never a completion fact."""
from copy import copy

import pytest

from compiler.frontend.descriptor_storage_groups import original_storage_group
from compiler.frontend.descriptor_teams import prove_descriptor_team
from compiler.frontend.descriptor_windows import prove_descriptor_window
from compiler.frontend.source_effects import SourceEffects
from compiler.frontend.storage_mutation_effects import prove_storage_mutation_effects
from compiler.ir import CompilationError
from compiler.tests.test_descriptor_joined_blocks import source as joined_source
from compiler.tests.test_descriptor_storage_groups import fixture


def source(*, barrier=True, body="", tail=""):
    return joined_source(single="master", ending="end master", body=body,
                         tail=("!$omp barrier\n" if barrier else "") + tail)


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("barrier", [False, True])
def test_original_master_is_descriptor_only_and_never_implies_an_endmaster_join(tmp_path, sealed, barrier):
    _path, analysis, routine, graph, identity = fixture(tmp_path, source(barrier=barrier), seal=sealed)
    group = graph.native_groups[identity]
    assert not group.available
    team = prove_descriptor_team(analysis, routine.qualified, group.original_nodes).public()
    assert team["version"] == 3
    assert team["retained_master_regions"] == 1
    assert team["retained_single_regions"] == 0
    assert team["parent_parallel_join_required"] is True
    assert team["master_barrier_assumed"] is False
    assert team["master_completion_authorized"] is False
    assert team["original_directives"].count("barrier") == int(barrier)
    assert team["original_directives"][-1] == "end parallel"
    proof = original_storage_group(analysis, routine, graph, identity).public()
    assert proof["storage_mutation"]["complete"] is True
    assert len(proof["storage_mutation"]["fixed_scalar_blocks"]) == 1
    for record in (team, proof, proof["storage_mutation"]):
        assert record["entry_definition_authority"] is False
        assert record["payload_access_authorized"] is False
        assert record["execution_authorized"] is False
    assert proof["numerical_completion_authorized"] is False
    with pytest.raises(CompilationError):
        analysis.joined_completion(routine.qualified, group.original_nodes)
    with pytest.raises(CompilationError):
        analysis.numerical_joined_completion(routine.qualified, group.original_nodes)


@pytest.mark.parametrize("change", ["clause", "end_nowait", "missing_end", "nested_master", "nested_single",
                                    "task", "barrier_inside", "masked", "missing_join"])
def test_master_never_admits_unsafe_clauses_tasks_nesting_or_incomplete_joins(tmp_path, change):
    text = source()
    if change == "clause":
        text = text.replace("!$omp master", "!$omp master private(total)")
    elif change == "end_nowait":
        text = text.replace("!$omp end master", "!$omp end master nowait")
    elif change == "missing_end":
        text = text.replace("!$omp end master", "")
    elif change == "nested_master":
        text = text.replace("value=total", "!$omp master\nvalue=total\n!$omp end master")
    elif change == "nested_single":
        text = text.replace("value=total", "!$omp single\nvalue=total\n!$omp end single")
    elif change == "task":
        text = text.replace("value=total", "!$omp task\nvalue=total\n!$omp end task")
    elif change == "barrier_inside":
        text = text.replace("value=total", "!$omp barrier\nvalue=total")
    elif change == "masked":
        text = text.replace("!$omp master", "!$omp masked filter(0)").replace("!$omp end master", "!$omp end masked")
    else:
        text = text.replace("!$omp end parallel", "")
    if change != "missing_join":
        _path, analysis, routine, graph, identity = fixture(tmp_path, text)
        originals = graph.native_groups[identity].original_nodes
    else:
        path = tmp_path / "unsafe.f90"
        path.write_text(text)
        analysis = SourceEffects([path])
        routine = analysis.routines["storage_group::run"]
        originals = tuple(routine.execution.content)
    with pytest.raises(CompilationError):
        prove_descriptor_team(analysis, routine.qualified, originals)


@pytest.mark.parametrize("tail", ["call unknown_export(a)", "deallocate(hidden)"])
def test_master_structure_does_not_hide_unsafe_following_body_effects(tmp_path, tail):
    text = source(tail=tail).replace("real(8)::total", "real(8)::total\nreal(8),allocatable,save::hidden(:)")
    _path, analysis, routine, graph, identity = fixture(tmp_path, text)
    team = prove_descriptor_team(analysis, routine.qualified, graph.native_groups[identity].original_nodes)
    assert team.public()["version"] == 3
    with pytest.raises(CompilationError):
        original_storage_group(analysis, routine, graph, identity)


def test_master_defaults_keep_concurrency_frontiers_and_do_not_authorize_copies(tmp_path):
    _path, analysis, routine, graph, identity = fixture(tmp_path, source())
    default = prove_storage_mutation_effects(analysis, routine.qualified, selection=(identity,)).public()
    assert not default["complete"]
    assert default["frontiers"]
    team = prove_descriptor_team(analysis, routine.qualified, graph.native_groups[identity].original_nodes)
    with pytest.raises(CompilationError):
        copy(team).validate(analysis, routine.qualified, (identity,))


def test_master_consumption_has_conditional_window19_and_explicit_source_salt(tmp_path):
    _path, analysis, routine, graph, identity = fixture(tmp_path, source())
    public = prove_descriptor_window(analysis, routine.qualified, (identity,), ("argument::a",)).public()
    assert public["schema_version"] == 19
    assert analysis._summary_authority()["descriptor_window_master_version"] == 19
    requirement, = public["native_storage_group_requirements"]
    assert requirement["storage_group"]["team_storage"]["version"] == 3
    assert not public["entry_definition_authority"]
    assert not public["execution_authorized"]


def test_old_single_reduction_receipts_keep_version2_and_no_master_fields(tmp_path):
    _path, analysis, routine, graph, identity = fixture(tmp_path, joined_source())
    public = prove_descriptor_team(analysis, routine.qualified, graph.native_groups[identity].original_nodes).public()
    assert public["version"] == 2
    assert "retained_master_regions" not in public
    assert "master_barrier_assumed" not in public
    assert prove_descriptor_window(analysis, routine.qualified, (identity,), ("argument::a",)).public()["schema_version"] == 16


@pytest.mark.parametrize("matrix", [False, True])
def test_ordinary_master_consumption_is19_even_when_matrix18_is_also_used(tmp_path, matrix):
    from compiler.frontend.native_group_structure import NativeGroupCandidate, grouped_original_sequence
    from compiler.tests.test_descriptor_block_storage import source as small_source

    text = small_source().replace("block\nreal(8)::value\nvalue=total\na(1)=value\nend block", "a(1)=total")
    if matrix:
        text = text.replace("real(8)::total", "real(8)::total\nreal(8)::x(3,3),y(3,3),z(3,3)")
        text = text.replace("a(1)=total", "a(1)=total\nz=matmul(x,y)")
    path = tmp_path / "small.f90"
    path.write_text(text)
    analysis = SourceEffects([path])
    analysis.seal_session()
    routine = analysis.routines["blocks::run"]
    graph = analysis.structure(routine.qualified)
    assert not graph.native_groups
    group, = (item for item in grouped_original_sequence(routine.execution.content, {})
              if isinstance(item, NativeGroupCandidate))
    team = prove_descriptor_team(analysis, routine.qualified, group.original_nodes)
    public = prove_descriptor_window(analysis, routine.qualified, team.selected_node_ids, ("argument::a",)).public()
    assert public["schema_version"] == 19
    assert public["native_joined_descriptor_requirements"][0]["storage_structure"]["version"] == 3
    assert "native_storage_group_requirements" not in public
    assert (any(item["intrinsic"] == "matmul" for item in public.get("native_intrinsic_descriptor_requirements", []))) is matrix
