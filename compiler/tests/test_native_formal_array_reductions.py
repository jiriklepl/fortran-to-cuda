"""Exact incoming native CALL associations compose with original joined storage."""
from copy import copy
from dataclasses import fields

import pytest

from compiler.frontend.descriptor_teams import (
    DescriptorTeamProof,
    _FixedArrayAssociationRequired,
    prove_descriptor_team,
)
from compiler.frontend.native_completion import prove_graph_joined_completion
from compiler.frontend.native_fixed_array_association import prove_native_fixed_array_association
from compiler.frontend.resource_reachability import prove_native_effect_graph
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError

CALLER = 'formal_reduction::outer'
CALLEE = 'formal_reduction::reduce_mean'


def source(*, actual='mean', other_shape='4:6', formal='result(3)', intent='out',
           masked_action=None, extra='', repeated=True, guarded=True):
    def loops(masked):
        result = []
        for component in (1, 2, 3):
            action = f'result({component})=result({component})+values(i,j,k)'
            if masked:
                action = (masked_action or f'if(values(i,j,k)>0){action}')
            result.append(f'''!$omp do collapse(3)
do k=1,n
 do j=1,n
  do i=1,n
   {action}
  end do
 end do
end do
''')
        return ''.join(result)
    first = f'call reduce_mean({actual},values,n,masked)'
    if guarded:
        first = 'if(go) '+first
    second = 'call reduce_mean(other,values,n,masked)' if repeated else ''
    return f'''module formal_reduction
implicit none
real(8)::values(-2:3,-2:3,-2:3), mean(-1:1), other({other_shape})
contains
subroutine outer(n,masked,go)
integer,intent(in)::n
logical,intent(in)::masked,go
{first}
{second}
end subroutine
subroutine reduce_mean({formal.split('(')[0]},values,n,masked)
real(8),intent({intent})::{formal}
real(8),contiguous,intent(in)::values(-2:,-2:,-2:)
integer,intent(in)::n
logical,intent(in)::masked
integer::i,j,k
result=0
if(masked)then
!$omp parallel reduction(+:result)
{loops(True)}!$omp end parallel
else
!$omp parallel reduction(+:result)
{loops(False)}!$omp end parallel
end if
{extra}
result=result/max(1,n)
end subroutine
end module
'''


def setup(tmp_path, text=None):
    path = tmp_path/'source.f90'
    path.write_text(source() if text is None else text)
    analysis = SourceEffects([path])
    analysis.seal_session()
    selection = tuple(analysis.routines[CALLER].execution.content)
    graph = prove_native_effect_graph(analysis, CALLER, selection)
    return path, analysis, graph


def complete(analysis, graph):
    return tuple(prove_graph_joined_completion(analysis, item.procedure, item.originals, graph)
                 for item in graph.deferred_completion)


@pytest.mark.parametrize('intent', ['out', 'inout'])
@pytest.mark.parametrize('guarded', [False, True])
@pytest.mark.parametrize('repeated', [False, True])
def test_original_shaped_branches_repeated_joins_and_every_incoming_call(tmp_path, intent, guarded, repeated):
    path, analysis, graph = setup(tmp_path, source(intent=intent, guarded=guarded, repeated=repeated))
    original = path.read_bytes()
    assert graph.completion_eligible
    assert len(graph.instances) == 2+repeated
    assert len(graph.deferred_completion) == 2
    for proof in complete(analysis, graph):
        assert proof.validate(analysis, CALLEE, proof.selected_node_ids) is proof
        record = proof.public()
        assert record['schema_version'] == 5
        assert len(record['fixed_array_call_instances']) == 1+repeated
        assert {item['instance'] for item in record['fixed_array_call_instances']} == set(range(1, 2+repeated))
        assert record['completion_scope'] == 'normal return only'
        assert not record['gpu_legality_established']
        assert not record['internal_cuts_authorized']
        for team in proof._reduction_teams:
            row = team.public()
            assert row['version'] == 8
            assert not row.get('intrinsic_scalar_reductions')
            assert len(row['implicit_worksharing_endings']) == 3
            assert len(team._fixed_array_associations) == 1+repeated
            for association in team._fixed_array_associations:
                assert association.formal_bounds == (1, 3)
                assert association.element_count == 3
                assert association.actual_bounds in {(-1, 1), (4, 6)}
                assert association.index_offset == association.actual_bounds[0]-1
                assert not association.public()['native_write_authority']
                assert not association.public()['initialization_authorized']
            first = team._fixed_array_associations[0]
            assert (type(first._selection[0]).__name__ == 'If_Stmt') is guarded
    assert path.read_bytes() == original


def test_formal_team_without_call_receipt_never_publishes_storage(tmp_path):
    _, analysis, graph = setup(tmp_path)
    group = graph.deferred_completion[0]
    with pytest.raises(_FixedArrayAssociationRequired):
        prove_descriptor_team(analysis, CALLEE, group.originals)
    assert not getattr(analysis, '_descriptor_team_proofs', {})
    assert graph.completion_eligible
    assert len(complete(analysis, graph)) == 2


@pytest.mark.parametrize('actual', ['mean(:)', 'mean([1,0,-1])', 'mean+1'])
def test_nonwhole_actual_cannot_borrow_other_incoming_call_authority(tmp_path, actual):
    _, analysis, graph = setup(tmp_path, source(actual=actual))
    if graph.completion_eligible:
        with pytest.raises(CompilationError):
            complete(analysis, graph)
    assert not getattr(analysis, '_native_graph_completions', {})


@pytest.mark.parametrize('other_shape', ['2', '4'])
def test_every_incoming_call_needs_equal_original_extent(tmp_path, other_shape):
    _, analysis, graph = setup(tmp_path, source(other_shape=other_shape))
    if graph.completion_eligible:
        with pytest.raises(CompilationError):
            complete(analysis, graph)
    assert not getattr(analysis, '_native_graph_completions', {})


@pytest.mark.parametrize('mutation', ['remove', 'registered_copy', 'actual', 'selection', 'record', 'teams'])
def test_consumed_association_cannot_be_replaced_or_healed(tmp_path, mutation):
    _, analysis, graph = setup(tmp_path)
    proof = complete(analysis, graph)[0]
    team = proof._reduction_teams[0]
    association = team._fixed_array_associations[0]
    if mutation == 'remove':
        del analysis._native_fixed_array_associations[association.identity]
    elif mutation == 'registered_copy':
        analysis._native_fixed_array_associations[association.identity] = copy(association)
    elif mutation == 'actual':
        object.__setattr__(association, 'actual', association.formal)
    elif mutation == 'selection':
        object.__setattr__(association, '_selection', ())
    elif mutation == 'record':
        object.__setattr__(association, '_record', '{}')
    else:
        object.__setattr__(team, '_fixed_array_associations', team._fixed_array_associations[1:])
    with pytest.raises(CompilationError):
        proof.validate(analysis, CALLEE, proof.selected_node_ids)
    group = graph.deferred_completion[0]
    with pytest.raises(CompilationError):
        prove_graph_joined_completion(analysis, CALLEE, group.originals, graph)


def test_valid_unrelated_call_receipt_never_covers_current_graph(tmp_path, monkeypatch):
    import compiler.frontend.native_completion as completion
    text = source().replace('end subroutine\nsubroutine reduce_mean',
        'call reduce_mean(mean,values,n,masked)\nend subroutine\nsubroutine reduce_mean', 1)
    path = tmp_path/'source.f90'
    path.write_text(text)
    analysis = SourceEffects([path])
    analysis.seal_session()
    statements = tuple(analysis.routines[CALLER].execution.content)
    unrelated = prove_native_fixed_array_association(analysis, CALLER, (statements[-1],), 'result')
    graph = prove_native_effect_graph(analysis, CALLER, statements[:-1])
    monkeypatch.setattr(completion, '_derive_formal_array_calls', lambda *args: (unrelated,))
    with pytest.raises(CompilationError, match='incoming CALL'):
        complete(analysis, graph)
    assert not getattr(analysis, '_native_graph_completions', {})


@pytest.mark.parametrize('mutation', ['association_actual', 'association_record', 'team_receipts', 'instances',
                                    'template_calls', 'argument_formal', 'actual_type_binding', 'original_argument'])
def test_final_publication_fences_all_consumed_receipts_and_instance_coverage(tmp_path, monkeypatch, mutation):
    import sys
    _, analysis, graph = setup(tmp_path)
    verify = analysis.verify_publication
    changed = []
    def intercept():
        verify()
        if sys._getframe(1).f_code.co_name != '_publish_graph_completion':
            return
        proof, = analysis._native_graph_completions.values()
        changed.append(proof.identity)
        team = proof._reduction_teams[0]
        if mutation == 'team_receipts':
            object.__setattr__(team, '_fixed_array_associations', ())
        elif mutation == 'instances':
            object.__setattr__(graph, 'instances', graph.instances[:-1])
        elif mutation == 'template_calls':
            object.__setattr__(graph.instances[0].template, 'calls', ())
        elif mutation in {'argument_formal', 'actual_type_binding', 'original_argument'}:
            resolved = graph.instances[1].call.edge.resolved
            mapping = resolved.associations[0]
            if mutation == 'argument_formal':
                object.__setattr__(mapping, 'formal_binding', mapping.binding)
            elif mutation == 'actual_type_binding':
                object.__setattr__(mapping.actual_type, 'binding', mapping.formal_binding)
            else:
                original = resolved.original_arguments[0]
                object.__setattr__(original, fields(type(original))[0].name, object())
        else:
            association = team._fixed_array_associations[0]
            object.__setattr__(association, 'actual' if mutation == 'association_actual' else '_record',
                               association.formal if mutation == 'association_actual' else '{}')
    monkeypatch.setattr(analysis, 'verify_publication', intercept)
    with pytest.raises(CompilationError, match='attachments changed'):
        complete(analysis, graph)
    assert len(changed) == 1
    assert not analysis._native_graph_completions


def test_root_formal_without_original_incoming_call_remains_unsupported(tmp_path):
    _, analysis, _ = setup(tmp_path)
    graph = prove_native_effect_graph(analysis, CALLEE, tuple(analysis.routines[CALLEE].execution.content))
    assert graph.completion_eligible
    with pytest.raises(CompilationError, match='incoming CALL'):
        complete(analysis, graph)


@pytest.mark.parametrize('intent', ['in', ''])
def test_formal_reduction_requires_explicit_writable_intent(tmp_path, intent):
    text = source(intent=intent)
    if not intent:
        text = text.replace(',intent()', '')
    _, analysis, graph = setup(tmp_path, text)
    assert graph.completion_eligible
    with pytest.raises(CompilationError, match='OUT or INOUT'):
        complete(analysis, graph)


def test_dynamic_formal_shape_does_not_gain_fixed_storage_from_actual(tmp_path):
    _, analysis, graph = setup(tmp_path, source(formal='result(:)'))
    assert graph.completion_eligible
    with pytest.raises(CompilationError):
        complete(analysis, graph)


def test_unknown_callee_outside_join_still_blocks_the_complete_graph(tmp_path):
    _, analysis, graph = setup(tmp_path, source(extra='call unknown(result)'))
    assert not graph.completion_eligible
    with pytest.raises(CompilationError, match='barriers'):
        complete(analysis, graph)


def test_nested_worksharing_inside_one_formal_team_remains_unowned(tmp_path):
    text = source().replace('!$omp parallel reduction(+:result)\n',
                            '!$omp parallel reduction(+:result)\nif(n>0)then\n')
    text = text.replace('!$omp end parallel', 'end if\n!$omp end parallel')
    _, analysis, graph = setup(tmp_path, text)
    assert not graph.completion_eligible
    with pytest.raises(CompilationError, match='barriers'):
        complete(analysis, graph)


@pytest.mark.parametrize('attachment', [part.name for part in fields(DescriptorTeamProof)])
def test_standalone_team_validation_fences_final_source_fact_callback(tmp_path, monkeypatch, attachment):
    from compiler.frontend.source_facts import SourceFact

    _, analysis, graph = setup(tmp_path)
    team = complete(analysis, graph)[0]._reduction_teams[0]
    original = SourceFact.require
    changed = []
    def intercept(fact, proof):
        result = original(fact, proof)
        if proof is team:
            changed.append(True)
            object.__setattr__(proof, attachment, () if attachment == '_fixed_array_associations' else object())
        return result
    monkeypatch.setattr(SourceFact, 'require', intercept)
    with pytest.raises(CompilationError, match='attachments changed'):
        team.validate(analysis, CALLEE, team.selected_node_ids)
    assert changed == [True]


@pytest.mark.parametrize('mutation', ['association_actual', 'association_record', 'registry', 'arena'])
def test_standalone_team_validation_fences_last_dependency_check(tmp_path, monkeypatch, mutation):
    import compiler.frontend.descriptor_teams as teams

    _, analysis, graph = setup(tmp_path)
    team = complete(analysis, graph)[0]._reduction_teams[0]
    original = teams._require_fixed_array_associations
    calls = 0
    def intercept(owner, procedure, associations):
        nonlocal calls
        original(owner, procedure, associations)
        if associations is team._fixed_array_associations:
            calls += 1
            if calls == 2:
                if mutation == 'registry':
                    analysis._descriptor_team_proofs = analysis._descriptor_team_proofs.copy()
                elif mutation == 'arena':
                    object.__setattr__(analysis, '_source_derivation_arena', None)
                else:
                    proof = associations[0]
                    object.__setattr__(proof, 'actual' if mutation == 'association_actual' else '_record',
                                       proof.formal if mutation == 'association_actual' else '{}')
    monkeypatch.setattr(teams, '_require_fixed_array_associations', intercept)
    with pytest.raises(CompilationError, match='attachments changed'):
        team.validate(analysis, CALLEE, team.selected_node_ids)
    assert calls == 2
