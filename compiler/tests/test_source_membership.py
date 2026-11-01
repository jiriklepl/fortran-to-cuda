"""Immutable source selections retain exact walk membership and overlap rules."""

from copy import copy
from itertools import combinations_with_replacement

import pytest
from fparser.two.utils import Base, walk

from compiler.frontend import source_effects
from compiler.frontend.source_effects import SourceEffects
from compiler.frontend.source_membership import ExecutionMembership
from compiler.frontend.source_session import SourceSession
from compiler.ir import CompilationError

SOURCE = """module work
contains
subroutine step(a,n,flag)
real(8),intent(inout)::a(:)
integer,intent(in)::n
logical,intent(in)::flag
integer::i
a(1)=0
! ordinary documentation
do i=2,n
  if(flag) then
    a(i)=a(i-1)+1
  else
    a(i)=0
  endif
enddo
a(n)=2
end subroutine
end module
"""


def fixture(tmp_path):
    path = tmp_path / "source.f90"
    path.write_text(SOURCE)
    analysis = SourceEffects([path])
    return analysis, analysis.routines["work::step"]


def subtree(node):
    return {id(item) for item in walk(node) if hasattr(item, "items") or hasattr(item, "content")}


def test_membership_matches_original_walk_and_every_pair_of_subtrees(tmp_path):
    _, routine = fixture(tmp_path)
    index = ExecutionMembership.create(routine.execution)
    expected = {id(node): node for node in walk(routine.execution)}
    assert index.nodes == expected
    assert all(index.nodes[key] is node for key, node in expected.items())
    nodes = [node for node in expected.values() if isinstance(node, Base)]
    for first, second in combinations_with_replacement(nodes, 2):
        assert index.overlaps(((first,), (second,))) == bool(subtree(first) & subtree(second))


def test_same_selection_may_contain_overlapping_source_nodes(tmp_path):
    _, routine = fixture(tmp_path)
    index = ExecutionMembership.create(routine.execution)
    loop = next(node for node in walk(routine.execution) if type(node).__name__ == "Block_Nonlabel_Do_Construct")
    inside = next(node for node in walk(loop) if type(node).__name__ == "Assignment_Stmt")
    following = routine.execution.content[-1]
    assert index.overlaps(((loop, inside), (following,))) is False
    assert index.overlaps(((loop, inside), (inside,))) is True


@pytest.mark.parametrize("container", [list, tuple])
def test_shared_parser_nodes_disable_only_interval_shortcut(tmp_path, container):
    _, routine = fixture(tmp_path)
    original = routine.execution.content[-1]
    repeated = container((original, original))
    index = ExecutionMembership.create(repeated)
    assert index.nodes == {id(node): node for node in walk(repeated)}
    assert index.intervals is None
    assert index.overlaps(((original,), (original,))) is None


def test_foreign_nodes_and_specification_need_fresh_walk(tmp_path):
    _, routine = fixture(tmp_path)
    index = ExecutionMembership.create(routine.execution)
    assert index.overlaps(((copy(routine.execution),),)) is None
    assert index.overlaps(((routine.scope.node,),)) is None
    with pytest.raises(TypeError):
        index.nodes[0] = routine.execution


def outcome(analysis, selection):
    try:
        graph, identities = analysis._selected_source("work::step", selection)
        return graph.public(), identities
    except CompilationError as error:
        return str(error)


@pytest.mark.parametrize("selection_kind", ["siblings", "reverse", "repeat", "overlap", "foreign_comment", "empty", "ids"])
def test_sealed_selection_matches_fresh_walk(tmp_path, monkeypatch, selection_kind):
    analysis, routine = fixture(tmp_path)
    analysis.seal_session()
    graph = analysis.structure(routine.qualified)
    selected = tuple(routine.execution.content)
    if selection_kind == "reverse":
        selected = tuple(reversed(selected))
    elif selection_kind == "repeat":
        selected = (selected[-1], selected[-1])
    elif selection_kind == "overlap":
        selected = (graph.root_id, graph.node_id(selected[-1]))
    elif selection_kind == "foreign_comment":
        comment = next(node for node in selected if type(node).__name__ == "Comment")
        selected = (copy(comment), selected[-1])
    elif selection_kind == "empty":
        selected = ()
    elif selection_kind == "ids":
        selected = tuple(graph.nodes[graph.root_id].children)
    actual = outcome(analysis, selected)
    monkeypatch.setattr(SourceSession, "execution_overlaps", lambda *_: None)
    monkeypatch.setattr(SourceSession, "execution_nodes", lambda _, _analysis, current:
                        {id(node): node for node in walk(current.execution)})
    assert outcome(analysis, selected) == actual


def test_repeated_sealed_selections_do_not_walk_original_subtrees(tmp_path, monkeypatch):
    analysis, routine = fixture(tmp_path)
    analysis.seal_session()
    selected = tuple(routine.execution.content)
    expected = outcome(analysis, selected)
    assert not isinstance(expected, str)

    def unexpected(*_args, **_kwargs):
        pytest.fail("immutable source selection walked an original subtree again")

    monkeypatch.setattr(source_effects, "walk", unexpected)
    for current in (analysis, analysis.private_projection(), analysis.private_projection(routines=True)):
        assert outcome(current, selected) == expected


def test_replaced_registry_cannot_borrow_interval_authority(tmp_path):
    analysis, routine = fixture(tmp_path)
    analysis.seal_session()
    original_registry = analysis.routines
    replacement = dict(original_registry)
    with pytest.raises(CompilationError, match="original source registry is immutable"):
        analysis.routines = replacement
    assert analysis.routines is original_registry
    assert analysis._source_session.execution_overlaps(analysis, routine, ((routine.execution,),)) is False
    # Retain the consumer fence under deliberate internal root corruption.
    object.__setattr__(analysis, "routines", replacement)
    with pytest.raises(CompilationError, match="original registry changed"):
        analysis._source_session.execution_overlaps(analysis, routine, ((routine.execution,),))


def test_unsealed_syntax_changes_still_require_fresh_authority(tmp_path):
    analysis, routine = fixture(tmp_path)
    selected = tuple(routine.execution.content)
    assert not isinstance(outcome(analysis, selected), str)
    routine.execution.content[-1].items = ()
    with pytest.raises(CompilationError, match="original source-backed"):
        analysis._selected_source(routine.qualified, selected)


def test_generated_artifacts_are_identical_to_fresh_selection_walks(tmp_path, monkeypatch):
    from compiler.frontend.summary_cache import SummaryCache
    from compiler.tests.test_collective_source_scopes import generate

    monkeypatch.setattr(source_effects, "_DEFAULT_SUMMARY_CACHE", SummaryCache())
    _, indexed, indexed_report, _ = generate(tmp_path)
    monkeypatch.setattr(SourceSession, "execution_overlaps", lambda *_: None)
    monkeypatch.setattr(SourceSession, "execution_nodes", lambda _, _analysis, current:
                        {id(node): node for node in walk(current.execution)})
    monkeypatch.setattr(source_effects, "_DEFAULT_SUMMARY_CACHE", SummaryCache())
    _, original, original_report, _ = generate(tmp_path)
    assert indexed == original
    assert indexed_report == original_report
