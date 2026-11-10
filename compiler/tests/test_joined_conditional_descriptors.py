"""Exact original joined groups can supply selected descriptor stability."""

from copy import copy

import pytest

from compiler.frontend.native_group_structure import NativeGroupCandidate, grouped_original_sequence
from compiler.frontend.storage_mutation_effects import prove_storage_mutation_effects
from compiler.ir import CompilationError
from compiler.scopes.conditional_return import descriptor_for
from compiler.tests.test_hierarchical_child_regions import fixture, owner_at
from compiler.tests.test_joined_boundary_lifecycle import ordinary_source


def selected_child(tmp_path, source=None):
    _path, builder = fixture(tmp_path, ordinary_source() if source is None else source)
    owner_at(builder)
    child = copy(builder)
    child.entry = builder.analysis.routines["hierarchical::middle"]
    group, = [g for g in grouped_original_sequence(child.entry.execution.content, {})
              if isinstance(g, NativeGroupCandidate)]
    return child, group.original_nodes


def test_complete_join_discharge_keeps_descriptor_and_payload_authority_separate(tmp_path):
    builder, nodes = selected_child(tmp_path)
    analysis, procedure = builder.analysis, builder.entry.qualified
    ordinary = prove_storage_mutation_effects(analysis, procedure, selection=nodes)
    assert not ordinary.preserves_descriptors()
    assert {row["kind"] for row in ordinary.public()["frontiers"]} == {
        "concurrency requires separate joined storage authority"}
    proof = descriptor_for(builder, analysis, procedure, nodes)
    assert proof is not None
    assert proof.validate(analysis, procedure, proof._selected) is proof
    assert proof._storage.preserves_descriptors()
    assert proof._storage._joined_storage is not None
    assert proof._storage._native_graph is None
    assert proof._storage._joined_storage.selected_node_ids == proof._selected
    assert set(proof.module_roots) == {"hierarchical::left", "hierarchical::right"}
    record = proof.public()
    for name in ("entry_exit_lifecycle_authorized", "payload_access_authorized",
                 "entry_definition_authority", "execution_authorized"):
        assert record[name] is False
    assert record["retention_requires_sealed_return_contract"]
    assert not ordinary.preserves_descriptors()  # No mutation of the original weaker projection.


@pytest.mark.parametrize("authority", ["storage", "storage_team"])
def test_changed_join_authority_invalidates_descriptor_consumer(tmp_path, authority):
    builder, nodes = selected_child(tmp_path)
    proof = descriptor_for(builder, builder.analysis, builder.entry.qualified, nodes)
    assert proof is not None
    if authority == "storage":
        token = proof._storage
        registry = builder.analysis._storage_mutation_effects
    else:
        token = proof._storage._joined_storage
        registry = builder.analysis._descriptor_team_proofs
    registry[token.identity] = copy(token)
    with pytest.raises(CompilationError):
        proof.validate(builder.analysis, builder.entry.qualified, proof._selected)


def test_descriptor_stability_does_not_require_payload_graph_budget(tmp_path, monkeypatch):
    from compiler.frontend import resource_reachability

    builder, nodes = selected_child(tmp_path)

    def unavailable(*_args, **_kwargs):
        pytest.fail("descriptor-only proof requested independent payload analysis")

    monkeypatch.setattr(resource_reachability, "prove_native_effect_graph", unavailable)
    proof = descriptor_for(builder, builder.analysis, builder.entry.qualified, nodes)
    assert proof is not None
    assert proof.validate(builder.analysis, builder.entry.qualified, proof._selected) is proof


@pytest.mark.parametrize("replacement", [
    "call unknown_escape(right)",
    "if(i==1)deallocate(right)",
])
def test_join_does_not_hide_unknown_execution_or_shared_descriptor_mutation(tmp_path, replacement):
    source = ordinary_source().replace("right(modulo(i,n)+2)=left(i-4)", replacement)
    builder, nodes = selected_child(tmp_path, source)
    assert descriptor_for(builder, builder.analysis, builder.entry.qualified, nodes) is None


def test_another_selection_cannot_consume_the_complete_join_descriptor(tmp_path):
    builder, nodes = selected_child(tmp_path)
    proof = descriptor_for(builder, builder.analysis, builder.entry.qualified, nodes)
    assert proof is not None
    graph, other = builder.analysis._selected_source(builder.entry.qualified, (builder.entry.execution.content[0],))
    assert graph is builder.analysis.structure(builder.entry.qualified)
    assert other != proof._selected
    with pytest.raises(CompilationError, match="exact registered original selection"):
        proof.validate(builder.analysis, builder.entry.qualified, other)


@pytest.mark.parametrize("selection", [slice(None, -1), slice(1, None)])
def test_incomplete_group_cannot_borrow_its_enclosing_join(tmp_path, selection):
    builder, nodes = selected_child(tmp_path)
    assert descriptor_for(builder, builder.analysis, builder.entry.qualified, nodes[selection]) is None


def test_nowait_cannot_supply_synchronous_descriptor_stability(tmp_path):
    source = ordinary_source().replace("!$omp end parallel do", "!$omp end parallel do nowait")
    _path, builder = fixture(tmp_path, source)
    owner_at(builder)
    child = copy(builder)
    child.entry = builder.analysis.routines["hierarchical::middle"]
    body = child.entry.execution.content
    assert not any(isinstance(g, NativeGroupCandidate) for g in grouped_original_sequence(body, {}))
    start = next(i for i, node in enumerate(body) if str(node).startswith("!$omp parallel do"))
    assert descriptor_for(child, child.analysis, child.entry.qualified, body[start:start + 3]) is None


def test_original_joined_native_consumer_keeps_descriptor_scope_with_directives(tmp_path):
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans

    _path, builder = fixture(tmp_path, ordinary_source())
    _outputs, report = builder.run()
    assert report["scope_count"] == 1
    member = builder._conditional_return_draft.owner.members["hierarchical::middle"]
    unit, = [unit for unit in member.units if str(unit.nodes[0]).startswith("!$omp parallel do")]
    operation, = unit.scope.native
    assert operation.summary.get("descriptor_window_identity")
    graph, selected = member.builder.analysis._selected_source(member.routine.qualified, unit.nodes)
    assert tuple(operation.summary["selected_node_ids"]) == selected
    assert graph.source_nodes(selected[0])[0] is unit.nodes[0]
    validate_checked_execution_plans(report)
