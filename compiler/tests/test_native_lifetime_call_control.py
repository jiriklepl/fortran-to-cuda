"""Lifetime indexing retains the whole original single-line IF CALL."""
from copy import copy

import pytest

from compiler.frontend.native_lifetime_context import NativeLifetimeContext
from compiler.ir import CompilationError
from compiler.tests.test_native_record_lifetime import SOURCE, prove, setup


def guarded(predicate='enabled'):
    return SOURCE.replace('subroutine owner()',
        'subroutine owner(enabled)\nlogical,intent(in)::enabled').replace(
        'call update(points,input,1)', f'if({predicate})call update(points,input,1)')


@pytest.mark.parametrize('predicate', ['enabled', '.false.', '.true.'])
def test_exact_whole_if_call_retains_same_graph_and_storage(tmp_path, predicate):
    path, analysis, selected, graph, teams, storage, completion = prove(tmp_path, guarded(predicate))
    before = path.read_bytes()
    context = NativeLifetimeContext(analysis, graph)
    assert context.require_selection('work::owner', selected) is context
    assert type(selected[0]).__name__ == 'If_Stmt'
    assert storage._native_graph is graph
    assert all(team._native_graph is graph for team in teams)
    assert storage.public()['complete']
    assert not storage.public()['escapes']
    assert not context.public()['payload_access_authorized']
    assert not context.public()['entry_definition_authorized']
    assert completion.validate(analysis, 'work::owner', completion.selected_node_ids) is completion
    assert path.read_bytes() == before


@pytest.mark.parametrize('change', ['detached', 'foreign_guard', 'multiple', 'copied_graph'])
def test_context_cannot_substitute_or_drop_original_control(tmp_path, change):
    _, analysis, selected, graph, _, _, _ = prove(tmp_path, guarded())
    context = NativeLifetimeContext(analysis, graph)
    if change == 'copied_graph':
        with pytest.raises(CompilationError):
            NativeLifetimeContext(analysis, copy(graph))
        return
    if change == 'detached':
        wrong = (selected[0].items[1],)
    elif change == 'foreign_guard':
        other = tmp_path/'other'
        other.mkdir()
        wrong = setup(other, guarded('.not.enabled'))[2]
    else:
        wrong = (*selected, *selected)
    with pytest.raises(CompilationError):
        context.require_selection('work::owner', wrong)


def test_separately_complete_multicall_graph_does_not_become_one_call(tmp_path):
    text = SOURCE.replace('call update(points,input,1)',
                          'call update(points,input,1)\ncall update(points,input,1)')
    _, analysis, _, graph = setup(tmp_path, text)
    assert graph.retirement_effects_eligible
    with pytest.raises(CompilationError, match='one complete original CALL'):
        NativeLifetimeContext(analysis, graph)
