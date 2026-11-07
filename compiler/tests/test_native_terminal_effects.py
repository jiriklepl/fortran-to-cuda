"""Issued fatal calls stay observable in native graphs and normal-return joins."""
from collections import Counter
from copy import copy

import pytest

from compiler.frontend import descriptor_termination
from compiler.frontend.native_completion import prove_graph_joined_completion
from compiler.frontend.resource_reachability import (
    NativeTerminatingCall,
    ResourceReachabilityBudgetExceeded,
    prove_resource_reachability,
)
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError

SOURCE = '''module work
use iso_fortran_env,only:error_unit
implicit none
real(8)::a(8),later(8)
integer::number
interface fail
module procedure fail_message,fail_number
end interface
contains
subroutine owner()
integer::i
!$omp parallel do private(i) schedule(guided,5)
do i=1,8
if(a(i)<0)call fail(ACTUAL)
a(i)=a(i)+1
later(i)=a(i)
enddo
!$omp end parallel do
end subroutine
subroutine fail_message(message)
character(*),intent(in)::message
SPEC
BODY
TERMINAL
end subroutine
subroutine fail_number(value)
integer,intent(in)::value
write(error_unit,'(i0)')value
stop
end subroutine
end module
'''


def setup(tmp_path, *, seal=True, actual='"invalid value"', body="write(error_unit,'(a)')message",
          terminal='stop', spec='', source=SOURCE):
    path = tmp_path / 'source.f90'
    path.write_text(source.replace('ACTUAL', actual).replace('SPEC', spec).replace('BODY', body)
                    .replace('TERMINAL', terminal))
    analysis = SourceEffects([path])
    if seal:
        analysis.seal_session()
    selected = tuple(analysis.routines['work::owner'].execution.content)
    demanded = tuple(analysis.modules['work'].bindings[n] for n in ('a', 'later', 'number'))
    return path, analysis, selected, demanded


def issue(data):
    _path, analysis, selected, demanded = data
    return prove_resource_reachability(analysis, 'work::owner', selected, demanded)


@pytest.mark.parametrize('seal', [False, True])
@pytest.mark.parametrize('terminal', ['stop', 'stop 3', 'error stop', 'error stop 2'])
def test_original_diagnostics_terminal_and_guard_survive_graph_and_join(tmp_path, seal, terminal):
    data = setup(tmp_path, seal=seal, terminal=terminal)
    path, analysis, selected, demanded = data
    original = path.read_bytes()
    proof = issue(data)
    assert not proof.barriers, proof.public()['barriers']
    assert proof.validate(analysis, 'work::owner', selected, demanded) is proof
    graph = proof.native_graph.public()
    item, = graph['terminal_calls']
    assert item['kind'] == 'original_observable_terminal_call'
    assert item['instance'] == 0
    assert item['returns_to_caller'] is False
    events = item['termination']['original_observable_events']
    assert [event['kind'] for event in events] == ['original_external_diagnostic_write', 'original_termination']
    assert events[-1]['source'].lower() == terminal
    template, = proof.native_graph.templates
    if_node = next(node for node in _nodes(template.nodes) if node.kind == 'If_Stmt')
    terminal_node, = [item for node in _nodes((if_node,)) for item in node.items
                      if type(item) is NativeTerminatingCall]
    assert terminal_node.original is template.terminations[0].original
    completion, = [item for _name, _ids, item in proof._completion
                   if type(item).__name__ == 'NativeGraphCompletionProof']
    record = completion.public()
    assert record['schema_version'] == 2
    assert record['completion_scope'] == 'normal return only'
    assert record['termination_paths_return'] is False
    assert record['original_terminal_calls'][0]['termination'] == item['termination']
    assert record['original_observable_terminal_instances'] == graph['terminal_calls']
    for key in ('motion_authorized', 'internal_cuts_authorized', 'gpu_legality_established'):
        assert record[key] is False
    assert not graph['execution_authority']
    assert not proof.public()['execution_authorized']
    assert any(access.storage.resource == 'work::later' and access.kind == 'write' for access in proof.accesses)
    assert path.read_bytes() == original
    events.clear()
    assert len(proof.native_graph.public()['terminal_calls'][0]['termination']['original_observable_events']) == 2


def _nodes(nodes):
    from compiler.frontend.resource_reachability import NativeEffectNode
    for node in nodes:
        if isinstance(node, NativeEffectNode):
            yield node
            yield from _nodes(node.items)


@pytest.mark.parametrize('actual', ['number', '3', 'value=number'])
def test_numeric_terminal_actual_keeps_original_read_effect(tmp_path, actual):
    proof = issue(setup(tmp_path, actual=actual))
    assert not proof.barriers, proof.public()['barriers']
    terminal, = proof.native_graph.terminal_calls
    assert terminal[1].proof._callee.qualified == 'work::fail_number'
    if 'number' in actual:
        assert any(item.storage.resource == 'work::number' and item.kind == 'read' for item in proof.accesses)
    assert dict(proof.native_graph.instantiation_work)['call_instances'] == 1


@pytest.mark.parametrize(('body', 'spec', 'terminal'), [
    ('message="changed"', '', 'stop'),
    ('call unknown()', '', 'stop'),
    ('allocate(scratch(1))', 'real,allocatable::scratch(:)', 'stop'),
    ('write(6,"(a)")message', '', 'stop'),
    ('write(error_unit,"(a)")message', '', 'return'),
    ('write(error_unit,"(a)")message', '', 'if(.true.)stop'),
    ('stop\nwrite(error_unit,"(a)")message', '', 'stop'),
    ('write(error_unit,"(a)")message', 'integer,save::state', 'stop'),
])
def test_failed_full_terminal_issuer_preserves_conservative_barrier(tmp_path, body, spec, terminal):
    data = setup(tmp_path, body=body, spec=spec, terminal=terminal)
    proof = issue(data)
    assert proof.barriers
    assert not proof.native_graph.terminal_calls
    assert not proof.native_graph.completion_eligible
    assert not getattr(data[1], '_native_graph_completions', {})


def test_symbol_spelling_never_establishes_termination(tmp_path):
    source = SOURCE.replace('call fail(ACTUAL)', 'call error_stop(ACTUAL)')
    proof = issue(setup(tmp_path, source=source))
    assert proof.barriers
    assert not proof.native_graph.terminal_calls


def test_terminal_in_called_helper_retains_instance_and_observables(tmp_path):
    source = SOURCE.replace('if(a(i)<0)call fail(ACTUAL)', 'call helper(a(i))').replace(
        'subroutine fail_message(message)', 'subroutine helper(value)\nreal(8),intent(in)::value\n'
        'if(value<0)call fail(ACTUAL)\nend subroutine\nsubroutine fail_message(message)')
    proof = issue(setup(tmp_path, source=source))
    assert not proof.barriers, proof.public()['barriers']
    terminal, = proof.native_graph.public()['terminal_calls']
    assert terminal['instance'] == 1
    assert terminal['procedure'] == 'work::helper'
    completion, = [item for _name, _ids, item in proof._completion
                   if type(item).__name__ == 'NativeGraphCompletionProof']
    assert completion.public()['original_observable_terminal_instances'] == [terminal]


@pytest.mark.parametrize('replacement', ['copy', 'foreign', 'public'])
def test_terminal_consumer_cannot_borrow_replaced_proof(tmp_path, replacement):
    data = setup(tmp_path)
    proof = issue(data)
    terminal = proof.native_graph.terminal_calls[0][1]
    original = terminal.proof
    if replacement == 'copy':
        substitute = copy(original)
    elif replacement == 'public':
        substitute = original.public()
    else:
        other = SourceEffects([data[0]])
        other.seal_session()
        call = next(node for node in _walk(other.routines['work::owner'].execution)
                    if type(node).__name__ == 'Call_Stmt')
        substitute = descriptor_termination.prove_descriptor_termination(other, 'work::owner', call)
    object.__setattr__(terminal, 'proof', substitute)
    with pytest.raises(CompilationError):
        proof.validate(data[1], 'work::owner', data[2], data[3])


def _walk(node):
    yield node
    for child in getattr(node, 'content', getattr(node, 'items', ())) or ():
        if child is not None:
            if isinstance(child, (tuple, list)):
                for item in child:
                    yield from _walk(item)
            else:
                yield from _walk(child)


def test_removed_termination_registration_invalidates_graph_and_completion(tmp_path):
    data = setup(tmp_path)
    proof = issue(data)
    data[1]._descriptor_terminating_calls.clear()
    with pytest.raises(CompilationError):
        proof.validate(data[1], 'work::owner', data[2], data[3])


def test_changed_original_diagnostic_rejects_publication(tmp_path):
    data = setup(tmp_path)
    issue(data)
    data[0].write_text(data[0].read_text().replace('invalid value', 'other failure'))
    with pytest.raises(CompilationError, match='changed'):
        data[1].verify_publication()


def test_hard_termination_budget_failure_is_not_reinterpreted_as_a_call_barrier(tmp_path, monkeypatch):
    from compiler.frontend.call_bindings import OriginalCallBudgetExceeded
    data = setup(tmp_path)
    def exhausted(*_args, **_kwargs):
        raise OriginalCallBudgetExceeded('test termination budget')
    monkeypatch.setattr(descriptor_termination, 'prove_descriptor_termination', exhausted)
    with pytest.raises(ResourceReachabilityBudgetExceeded, match='termination budget'):
        issue(data)
    assert not getattr(data[1], '_native_effect_graphs', {})


@pytest.mark.parametrize('seal', [False, True])
def test_type_bound_call_remains_an_unknown_effect_without_trying_named_termination(tmp_path, seal, monkeypatch):
    from compiler.frontend.native_preservation import prove_native_preservation

    source = '''module work
implicit none
type :: receiver
contains
procedure :: update
end type
type(receiver) :: object
real(8) :: a(8),later(8)
integer :: number
contains
subroutine owner()
call object%update()
end subroutine
subroutine update(self)
class(receiver),intent(inout)::self
number=number+1
end subroutine
end module
'''
    data = setup(tmp_path, source=source, seal=seal)

    def wrong_issuer(*_args, **_kwargs):
        pytest.fail('An indirect designator cannot enter the named termination issuer')

    monkeypatch.setattr(descriptor_termination, 'prove_descriptor_termination', wrong_issuer)
    proof = issue(data)
    assert any(item.kind == 'call' and 'indirect' in item.reason for item in proof.barriers)
    assert not proof.native_graph.terminal_calls
    preserved = prove_native_preservation(data[1], 'work::owner', data[2], data[3])
    assert not preserved.complete
    assert {root for root, _reasons in preserved.invalidations} == {item.root for item in data[3]}


def test_serialized_graph_is_not_completion_authority(tmp_path):
    data = setup(tmp_path)
    proof = issue(data)
    with pytest.raises(CompilationError, match='issued native effect graph'):
        prove_graph_joined_completion(data[1], 'work::owner', data[2], proof.native_graph.public())


@pytest.mark.parametrize('mutation', ['policy', 'projection'])
def test_changed_supported_source_context_invalidates_terminal_graph(tmp_path, mutation):
    data = setup(tmp_path)
    proof = issue(data)
    analysis = data[1]
    if mutation == 'policy':
        analysis.operation_limit += 1
    else:
        analysis.project_routine('work::fail_message', copy(analysis.routines['work::fail_message']))
    with pytest.raises(CompilationError):
        proof.validate(analysis, 'work::owner', data[2], data[3])


def test_discarded_generation_terminal_handle_cannot_be_rebound_into_parent(tmp_path):
    from hashlib import sha256

    from compiler.driver.options import CompilerOptions
    from compiler.offload.config import OffloadConfig
    from compiler.scopes.generation_transaction import GenerationTransaction
    from compiler.scopes.source import ScopeBuilder
    path, _analysis, _selected, _demanded = setup(tmp_path)
    facts = {'schema_version': 1, 'participation': 'serial', 'captures': {},
             'sources': {str(path): sha256(path.read_bytes()).hexdigest()}}
    builder = ScopeBuilder([path], 'work::owner', facts=facts, options=CompilerOptions(),
                           config=OffloadConfig(policy='sections', scope_execution='graph'))
    analysis = builder.analysis
    selected = tuple(analysis.routines['work::owner'].execution.content)
    demanded = (analysis.modules['work'].bindings['a'],)
    with GenerationTransaction(builder):
        discarded = prove_resource_reachability(analysis, 'work::owner', selected, demanded)
        terminal = discarded.native_graph.terminal_calls[0][1]
    # Even a retained registry reference cannot supply the discarded arena's
    # original termination handle to a later graph.
    analysis._descriptor_terminating_calls = {('work::owner', id(terminal.original)): terminal.proof}
    with pytest.raises(CompilationError):
        terminal.validate(analysis)


def test_repeated_helper_instances_retain_each_terminal_occurrence(tmp_path):
    source = SOURCE.replace('if(a(i)<0)call fail(ACTUAL)', 'call helper(a(i))\ncall helper(later(i))').replace(
        'subroutine fail_message(message)', 'subroutine helper(value)\nreal(8),intent(in)::value\n'
        'if(value<0)call fail(ACTUAL)\nend subroutine\nsubroutine fail_message(message)')
    proof = issue(setup(tmp_path, source=source))
    assert not proof.barriers
    terminals = proof.native_graph.public()['terminal_calls']
    assert [item['instance'] for item in terminals] == [1, 2]
    assert terminals[0]['termination'] == terminals[1]['termination']
    assert dict(proof.native_graph.instantiation_work)['call_instances'] == 4


def test_terminal_occurrences_share_graph_work_budget_not_statement_limit(tmp_path):
    source = SOURCE.replace('if(a(i)<0)call fail(ACTUAL)', 'call helper(a(i))\n' * 16).replace(
        'subroutine fail_message(message)', 'subroutine helper(value)\nreal(8),intent(in)::value\n'
        + 'if(value<0)call fail(ACTUAL)\n' * 16 + 'end subroutine\nsubroutine fail_message(message)')
    proof = issue(setup(tmp_path, source=source))
    assert not proof.barriers
    graph = proof.native_graph
    assert dict(graph.instantiation_work)['call_instances'] == 272
    assert len(graph.terminal_calls) == 256
    assert Counter(instance for instance, _terminal in graph.terminal_calls) == {
        instance: 16 for instance in range(1, 17)}
    assert all(terminal.public()['returns_to_caller'] is False for _instance, terminal in graph.terminal_calls)


def test_failed_numeric_terminal_hint_still_scans_original_mutation(tmp_path):
    source = SOURCE.replace('write(error_unit,\'(i0)\')value\nstop',
                            'later(1)=value\nwrite(error_unit,\'(i0)\')value\nstop')
    proof = issue(setup(tmp_path, actual='number', source=source))
    assert proof.barriers
    assert not proof.native_graph.terminal_calls
    assert any(item.storage.resource == 'work::later' and item.kind == 'write' for item in proof.accesses)
