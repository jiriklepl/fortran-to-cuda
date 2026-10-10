"""Large original teams defer bounded proof work without acquiring member authority."""

import pytest
from fparser.common.readfortran import FortranStringReader
from fparser.two import Fortran2003 as F
from fparser.two.parser import ParserFactory
from fparser.two.utils import walk

from compiler.frontend import native_group_structure as native_groups
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError


def fixture(tmp_path, body, *, operations=12, tail="", specification=""):
    path = tmp_path / "joined_native.f90"
    path.write_text("module renamed_boundary\nimplicit none\ncontains\n"
        "subroutine apply(a,b,n,flag)\nreal(8),intent(inout)::a(-2:),b(-2:)\n"
        "integer,intent(in)::n\nlogical,intent(in)::flag\ninteger::i\n"
        + specification + "\ncontinue\n" + body + "\n" + tail
        + "\nend subroutine\nend module\n")
    return path, SourceEffects([path], operations=operations)


def worksharing(statement="a(i)=b(i)+1", *, end="!$omp end do nowait"):
    return "!$omp do\ndo i=-2,n\n" + statement + "\nenddo\n" + end + "\n"


def team(body):
    return "!$omp parallel private(i)\n" + body + "!$omp end parallel\n"


def graph_group(analysis):
    graph = analysis.structure("renamed_boundary::apply")
    assert graph.available, graph.reasons
    assert len(graph.native_groups) == 1
    return graph, next(iter(graph.native_groups.values()))


def test_only_raw_budget_failure_defers_complete_original_groups(tmp_path):
    _, analysis = fixture(tmp_path, team("if(flag) then\n" + worksharing() * 6 + "endif\n"))
    graph, group = graph_group(analysis)
    assert graph.version == 6
    assert graph.raw_structure_rejection == ("bounded structured source operation budget exhausted",)
    assert group.available, group.reason
    assert [unit.kind for unit in group.units] == ["condition"] + ["worksharing"] * 6
    assert group.control_count == 2
    assert graph.operation_count <= graph.operation_limit == 12
    assert all(unit.structure.available and unit.structure.operation_count <= 12 for unit in group.units)
    assert graph.nodes[group.node_id].kind == "boundary"
    assert graph.nodes[group.node_id].public()["deferred_native_group"] == group.public()
    original_ids = {id(node) for node in walk(analysis.routines[graph.procedure].execution)}
    assert all(id(node) in original_ids for node in graph.source_nodes(graph.root_id))
    assert graph.deferred_native_descendants((graph.root_id,)) == (group.node_id,)
    assert graph.deferred_native_descendants((group.node_id,)) == (group.node_id,)
    assert graph.deferred_native_descendants((graph.entry_id,)) == ()
    assert all(unit.guard == ("flag",) for unit in group.units[1:])
    assert group.units[0].guard == ()
    assert group.units[0].structure.nodes[group.units[0].selected_node_ids[0]].details["evaluation"] == "condition"
    public = group.public()
    assert 0 < public["scanned_source_nodes"] <= public["source_traversal_entries"] <= public["source_scan_limit"]
    assert 0 < public["scanned_control_items"] <= public["control_scan_limit"]


def test_small_original_team_keeps_existing_graph(tmp_path):
    _, analysis = fixture(tmp_path, team(worksharing()), operations=12)
    graph = analysis.structure("renamed_boundary::apply")
    assert graph.available
    assert not graph.native_groups
    assert not graph.raw_structure_rejection
    assert not any(node.kind == "boundary" for node in graph.nodes.values())
    assert graph.deferred_native_descendants((graph.root_id,)) == ()


def test_ancestor_search_retains_branch_alternatives_and_original_ast(tmp_path):
    _, analysis = fixture(tmp_path, "if(flag) then\n" + team(worksharing() * 6) + "else\na(-2)=0\nendif")
    graph, group = graph_group(analysis)
    branch = walk(analysis.routines[graph.procedure].execution, F.If_Construct)[0]
    branch_id = graph.node_id(branch)
    assert graph.deferred_native_descendants((branch_id,)) == (group.node_id,)
    assert graph.deferred_native_descendants((graph.root_id, branch_id, group.node_id)) == (group.node_id,)
    condition_id, _ = graph.nodes[branch_id].alternatives[0]
    assert graph.deferred_native_descendants((condition_id,)) == ()
    with pytest.raises(CompilationError, match="original source authority"):
        graph.deferred_native_descendants(("foreign",))


def test_ordinary_budget_failure_is_not_hidden(tmp_path):
    _, analysis = fixture(tmp_path, "a(-2)=b(-2)\n" * 13)
    graph = analysis.structure("renamed_boundary::apply")
    assert not graph.available
    assert not graph.native_groups
    assert graph.reasons == ("bounded structured source operation budget exhausted",)


def test_deferred_selection_requires_exact_complete_original_tuple(tmp_path):
    _, analysis = fixture(tmp_path, team(worksharing() * 6))
    graph, group = graph_group(analysis)
    assert graph.native_group_for_selection(group.original_nodes) is group
    assert graph.native_group_for_selection(list(group.original_nodes)) is group
    clones = []
    for node in group.original_nodes:
        # fparser's Comment pickling protocol does not support copy.copy.
        # These distinct wrappers deliberately retain the original payload.
        clone = object.__new__(type(node))
        clone.__dict__.update(node.__dict__)
        clones.append(clone)
    for selected in ((group.original_nodes[0],), (group.original_nodes[-1],),
                     (group.units[0].original_nodes[0],), group.original_nodes[1:],
                     tuple(reversed(group.original_nodes)),
                     tuple(clones)):
        assert graph.native_group_for_selection(selected) is None
    for member in group.original_nodes:
        with pytest.raises(CompilationError, match="original source authority"):
            graph.node_id(member)
    assert graph.source_nodes(group.node_id) == group.original_nodes


def test_attached_join_does_not_absorb_or_clone_following_loop(tmp_path):
    tail = "do i=-2,n\nb(i)=a(i)+2\nenddo\n"
    _, analysis = fixture(tmp_path, team(worksharing() * 6), tail=tail)
    routine = analysis.routines["renamed_boundary::apply"]
    original_text = str(routine.execution)
    original_nodes = tuple(walk(routine.execution))
    graph, group = graph_group(analysis)
    loops = walk(routine.execution, F.Block_Nonlabel_Do_Construct)
    following = loops[-1]
    join = next(node for node in walk(routine.execution, F.Comment) if native_groups.directive(node) == "end parallel")
    assert join in group.original_nodes
    assert all(node is not following for node in group.original_nodes)
    assert graph.node_id(following) in graph.nodes
    assert str(routine.execution) == original_text
    assert all(before is after for before, after in zip(original_nodes, walk(routine.execution), strict=True))


def test_raw_parser_attached_join_is_exposed_without_ast_mutation(tmp_path):
    path, _ = fixture(tmp_path, team(worksharing() * 6), tail="do i=-2,n\nb(i)=a(i)\nenddo")
    parsed = ParserFactory().create(std="f2008")(FortranStringReader(path.read_text(), ignore_comments=False))
    execution = walk(parsed, F.Execution_Part)[0]
    following = walk(execution, F.Block_Nonlabel_Do_Construct)[-1]
    join = next(node for node in walk(following, F.Comment) if native_groups.directive(node) == "end parallel")
    text, originals = str(execution), tuple(walk(execution))
    result = native_groups.grouped_original_sequence(execution.content, {})
    group = next(item for item in result if isinstance(item, native_groups.NativeGroupCandidate))
    assert any(node is join for node in group.original_nodes)
    assert all(node is not following for node in group.original_nodes)
    assert any(node is following for node in result)
    assert str(execution) == text
    assert all(before is after for before, after in zip(originals, walk(execution), strict=True))


@pytest.mark.parametrize(("body", "reason"), [
    (worksharing() * 13, "source-unit budget"),
    (worksharing("\n".join("a(i)=b(i)+1" for _ in range(13))), "unit skeleton unavailable"),
    ("if(flag) then\nendif\n" * 7 + worksharing() * 2, "control-node budget"),
])
def test_exhaustion_rejects_whole_group_without_partial_units(tmp_path, body, reason):
    _, analysis = fixture(tmp_path, team(body))
    _, group = graph_group(analysis)
    assert not group.available
    assert reason in group.reason
    assert not group.units


def test_unknown_calls_and_lifetime_nodes_remain_in_local_unit_skeleton(tmp_path):
    body = worksharing("call unknown(a)") + worksharing("allocate(temporary(2))") + worksharing() * 5
    _, analysis = fixture(tmp_path, team(body), specification="real(8),allocatable::temporary(:)")
    _, group = graph_group(analysis)
    assert group.available, group.reason
    first, second = group.units[:2]
    assert any(node.kind == "boundary" and "unresolved" in node.details["reason"]
               for node in first.structure.nodes.values())
    assert any(node.kind == "boundary" and "Allocate_Stmt" in node.details["reason"]
               for node in second.structure.nodes.values())
    assert group.public()["effect_authority"].startswith("requires registered")


def test_branch_condition_units_preserve_original_evaluation_order(tmp_path):
    body = ("if(flag) then\n" + worksharing() * 3 + "else if(b(-2)>0) then\n"
            + worksharing() * 3 + "else\n" + worksharing() + "endif\n")
    _, analysis = fixture(tmp_path, team(body))
    _, group = graph_group(analysis)
    assert group.available, group.reason
    first, second = [unit for unit in group.units if unit.kind == "condition"]
    assert first.guard == ()
    assert second.guard == (".not.(flag)",)
    assert tuple(guard.replace(" ", "") for guard in group.units[-1].guard) == (".not.(flag)", ".not.(b(-2)>0)")


def test_full_source_syntax_scan_has_an_explicit_bound(tmp_path, monkeypatch):
    _, analysis = fixture(tmp_path, team(worksharing() * 6))
    monkeypatch.setattr(native_groups, "NATIVE_GROUP_SCAN_LIMIT", 40)
    graph = analysis.structure("renamed_boundary::apply")
    assert not graph.available
    assert not graph.native_groups
    assert graph.reasons == ("bounded structured source operation budget exhausted",)


def test_deferred_graph_cannot_outlive_source_or_original_ast(tmp_path):
    path, analysis = fixture(tmp_path, team(worksharing() * 6))
    graph, group = graph_group(analysis)
    path.write_text(path.read_text().replace("+1", "+3"))
    with pytest.raises(CompilationError, match="changed"):
        analysis.structure(graph.procedure)
    fresh = SourceEffects([path], operations=12)
    fresh_graph, _ = graph_group(fresh)
    assert fresh_graph.identity != graph.identity
    assert fresh_graph.native_group_for_selection(group.original_nodes) is None


def test_missing_join_cannot_become_a_deferred_native_group(tmp_path):
    _, analysis = fixture(tmp_path, "!$omp parallel private(i)\n" + worksharing() * 6)
    graph = analysis.structure("renamed_boundary::apply")
    assert not graph.available
    assert not graph.native_groups
