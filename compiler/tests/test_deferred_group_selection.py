"""Complete deferred source groups compose without granting member authority."""

import pytest
from fparser.two import Fortran2003 as F
from fparser.two.utils import walk

from compiler.ir import CompilationError
from compiler.tests.test_native_group_structure import fixture, graph_group, team, worksharing


def example(tmp_path):
    _, analysis = fixture(tmp_path, "a(-2)=b(-2)\n" + team(worksharing() * 6), tail="b(-2)=a(-2)\n")
    graph, group = graph_group(analysis)
    assignments = walk(analysis.routines[graph.procedure].execution, F.Assignment_Stmt)
    return analysis, graph, group, assignments[0], assignments[-1]


@pytest.mark.parametrize(("prefix", "suffix"), [(True, False), (False, True), (True, True)])
def test_complete_group_composes_with_ordinary_original_neighbors(tmp_path, prefix, suffix):
    analysis, graph, group, before, after = example(tmp_path)
    selected = (*((before,) if prefix else ()), *group.original_nodes, *((after,) if suffix else ()))
    actual_graph, identities = analysis._selected_source(graph.procedure, selected)
    assert actual_graph is graph
    assert identities == (*((graph.node_id(before),) if prefix else ()), group.node_id,
                          *((graph.node_id(after),) if suffix else ()))
    assert graph.deferred_native_descendants(identities) == (group.node_id,)
    # Structural inclusion does not authorize an ordinary partial/native proof.
    with pytest.raises(CompilationError, match="whole original native completion"):
        analysis.segment_summary(graph.procedure, selected, capture_locals=True)


@pytest.mark.parametrize("change", ["partial", "reversed", "foreign", "interleaved"])
def test_incomplete_foreign_or_noncontiguous_groups_remain_rejected(tmp_path, change):
    analysis, graph, group, before, after = example(tmp_path)
    nodes = list(group.original_nodes)
    if change == "partial":
        nodes.pop()
    elif change == "reversed":
        nodes.reverse()
    elif change == "foreign":
        original = nodes[0]
        clone = object.__new__(type(original))
        clone.__dict__.update(original.__dict__)
        nodes[0] = clone
    else:
        nodes.insert(1, after)
    with pytest.raises(CompilationError, match="original source authority"):
        analysis._selected_source(graph.procedure, (before, *nodes, after))


def test_collapsed_group_retains_duplicate_and_source_order_checks(tmp_path):
    analysis, graph, group, before, after = example(tmp_path)
    with pytest.raises(CompilationError, match="repeated"):
        analysis._selected_source(graph.procedure, (before, *group.original_nodes, *group.original_nodes, after))
    with pytest.raises(CompilationError, match="source order"):
        analysis._selected_source(graph.procedure, (after, *group.original_nodes))


def test_collapsed_group_keeps_guards_and_ancestor_overlap_checks(tmp_path):
    _, analysis = fixture(tmp_path, "if(flag) then\na(-2)=b(-2)\n" + team(worksharing() * 6)
                          + "endif\n", tail="b(-2)=a(-2)\n")
    graph, group = graph_group(analysis)
    assignments = walk(analysis.routines[graph.procedure].execution, F.Assignment_Stmt)
    with pytest.raises(CompilationError, match="one reached guarded segment"):
        analysis._selected_source(graph.procedure, (*group.original_nodes, assignments[-1]))
    assert analysis._selected_source(graph.procedure, (assignments[0], *group.original_nodes))[1] == (
        graph.node_id(assignments[0]), group.node_id)
    # The branch body sequence has the same guard but already owns the group.
    branch, = walk(analysis.routines[graph.procedure].execution, F.If_Construct)
    _, body = graph.nodes[graph.node_id(branch)].alternatives[0]
    with pytest.raises(CompilationError, match="subtrees overlap"):
        analysis._selected_source(graph.procedure, (body, *group.original_nodes))
