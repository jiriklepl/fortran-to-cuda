"""Graph normalization must retain the exact original joined source authority."""

from copy import copy

import pytest

from compiler.frontend.native_group_structure import NativeGroupCandidate, grouped_original_sequence
from compiler.ir import CompilationError
from compiler.scopes.conditional_return import descriptor_for
from compiler.tests.test_hierarchical_child_regions import fixture, owner_at
from compiler.tests.test_joined_boundary_lifecycle import ordinary_source


def selected(tmp_path):
    source = ordinary_source().replace(
        "!$omp parallel do private(i)\ndo i=1,n",
        "!$omp parallel do private(i)\n! original documentation\n\ndo i=1,n",
    ).replace("enddo\n!$omp end parallel do", "enddo\n! retained before join\n!$omp end parallel do")
    _path, builder = fixture(tmp_path, source)
    owner_at(builder)
    child = copy(builder)
    child.entry = child.analysis.routines["hierarchical::middle"]
    group, = [item for item in grouped_original_sequence(child.entry.execution.content, {})
              if isinstance(item, NativeGroupCandidate)]
    return child, group.original_nodes


def test_graph_selection_recovers_comments_before_exact_joined_proof(tmp_path):
    builder, nodes = selected(tmp_path)
    analysis, procedure = builder.analysis, builder.entry.qualified
    graph, identities = analysis._selected_source(procedure, nodes)
    normalized = tuple(node for identity in identities for node in graph.source_nodes(identity))
    assert len(normalized) < len(nodes)
    proof = descriptor_for(builder, analysis, procedure, nodes)
    assert proof is not None
    assert proof.validate(analysis, procedure, identities) is proof
    team = proof._storage._joined_storage
    assert len(team._originals) == len(nodes)
    assert all(a is b for a, b in zip(team._originals, nodes, strict=True))
    assert not proof.public()["payload_access_authorized"]
    # The graph-ID consumer receives the same complete original authority.
    assert descriptor_for(builder, analysis, procedure, identities) is proof


@pytest.mark.parametrize("part", [slice(1, None), slice(None, -1)])
def test_comments_do_not_lend_a_missing_original_join(tmp_path, part):
    builder, nodes = selected(tmp_path)
    assert descriptor_for(builder, builder.analysis, builder.entry.qualified, nodes[part]) is None


def test_foreign_comment_is_not_an_original_selection(tmp_path):
    builder, nodes = selected(tmp_path)
    foreign_root = tmp_path / "foreign"
    foreign_root.mkdir()
    _other, foreign = selected(foreign_root)
    position = next(i for i, node in enumerate(nodes) if str(node) == "! original documentation")
    changed = (*nodes[:position], foreign[position], *nodes[position + 1:])
    with pytest.raises(CompilationError, match="comment lacks original source authority"):
        descriptor_for(builder, builder.analysis, builder.entry.qualified, changed)


def test_normal_emission_keeps_commented_native_group_inside_domain(tmp_path):
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans

    selected(tmp_path)
    emitted = tmp_path / "emitted"
    emitted.mkdir()
    _path, builder = fixture(emitted, (tmp_path / "source.f90").read_text())
    _outputs, report = builder.run()
    assert report["scope_count"] == 1
    member = builder._conditional_return_draft.owner.members["hierarchical::middle"]
    unit, = [unit for unit in member.units if str(unit.nodes[0]).startswith("!$omp parallel do")]
    operation, = unit.scope.native
    assert operation.summary.get("descriptor_window_identity")
    _, identities = member.builder.analysis._selected_source(member.routine.qualified, unit.nodes)
    assert tuple(operation.summary["selected_node_ids"]) == identities
    validate_checked_execution_plans(report)
