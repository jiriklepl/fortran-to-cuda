"""Fixed array storage is native-only; completion also needs the exact effect graph."""
from copy import copy
from dataclasses import fields

import pytest

from compiler.frontend.descriptor_teams import prove_descriptor_team
from compiler.frontend.native_completion import NativeGraphCompletionProof, prove_graph_joined_completion
from compiler.frontend.native_group_structure import NativeGroupCandidate, grouped_original_sequence
from compiler.frontend.resource_reachability import prove_native_effect_graph
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError

PROC = 'array_reduction::run'


def source(*, declaration='real(8)::total(3)', clause='+:total', action='total(1)=total(1)+values(i)',
           prefix='', opening='parallel private(i)', end='end parallel', guarded=False, repeat=False,
           inner_end='end do', extra=''):
    loop = f'''!$omp do reduction({clause})
do i=1,n
{action}
end do
!$omp {inner_end}
'''
    body = ('if(flag)then\n' + loop + 'else\n' + loop + 'end if\n') if guarded else loop
    if repeat:
        body += loop
    return f'''module array_reduction
implicit none
real(8)::values(8)
{prefix}
contains
subroutine run(n,flag)
integer,intent(in)::n
logical,intent(in)::flag
integer::i
{declaration}
!$omp {opening}
{body}{extra}
!$omp {end}
end subroutine
end module
'''


def setup(tmp_path, text=None, *, sealed=True):
    path = tmp_path / 'source.f90'
    path.write_text(source() if text is None else text)
    analysis = SourceEffects([path])
    if sealed:
        analysis.seal_session()
    routine = analysis.routines[PROC]
    groups = [row for row in grouped_original_sequence(routine.execution.content, {})
              if isinstance(row, NativeGroupCandidate)]
    if len(groups) != 1:
        raise CompilationError("fixture lacks one exact complete original joined team")
    group, = groups
    return path, analysis, group.original_nodes


def issue(data):
    _, analysis, originals = data
    return prove_descriptor_team(analysis, PROC, originals)


def completed(data):
    _, analysis, originals = data
    graph = prove_native_effect_graph(analysis, PROC, originals)
    assert graph.completion_eligible
    proof = prove_graph_joined_completion(analysis, PROC, originals, graph)
    return graph, proof


@pytest.mark.parametrize('sealed', [False, True])
@pytest.mark.parametrize(('shape', 'lo', 'hi'), [('3', 1, 3), ('-2:5', -2, 5), ('1024', 1, 1024), ('2147483647', 1, 2147483647)])
def test_local_fixed_bounds_are_metadata_not_an_element_work_budget(tmp_path, sealed, shape, lo, hi):
    data = setup(tmp_path, source(declaration=f'real(8)::total({shape})'), sealed=sealed)
    path = data[0]
    original = path.read_bytes()
    record = issue(data).public()
    assert record['version'] == 7
    assert not record.get('intrinsic_scalar_reductions')
    reduction, = record['intrinsic_fixed_array_reductions']
    row, = reduction['bindings']
    assert (row['lower_bound'], row['upper_bound'], row['element_count'], row['byte_count']) == (
        lo, hi, hi-lo+1, (hi-lo+1)*8)
    assert row['binding']['rank'] == 1
    assert reduction['version'] == 1
    assert reduction['native_only']
    for key in ('payload_access_authorized', 'entry_definition_authority', 'gpu_legality_established', 'arithmetic_reassociation_authorized'):
        assert reduction[key] is False
    assert all(record[key] is False for key in ('execution_authorized', 'entry_definition_authority', 'payload_access_authorized', 'gpu_legality_established', 'internal_cuts_authorized'))
    assert path.read_bytes() == original


@pytest.mark.parametrize(('dtype', 'operator'), [('real(4)', '+'), ('real(8)', 'max'), ('integer(4)', 'iand'), ('integer(8)', 'min'), ('logical(4)', '.and.'), ('logical(8)', '.neqv.')])
def test_intrinsic_types_use_distinct_array_inventory(tmp_path, dtype, operator):
    record = issue(setup(tmp_path, source(declaration=f'{dtype}::total(3)', clause=operator+':total'))).public()
    assert record['intrinsic_fixed_array_reductions'][0]['operator'] == operator
    assert not record.get('intrinsic_scalar_reductions')


def test_original_parameter_bounds_and_mixed_scalar_inventory(tmp_path):
    data = setup(tmp_path, source(declaration='integer,parameter::lo=-2,hi=4\nreal(8)::total(lo:hi),scalar', clause='+:scalar,total'))
    record = issue(data).public()
    assert [row['name'] for row in record['intrinsic_scalar_reductions'][0]['bindings']] == ['scalar']
    row, = record['intrinsic_fixed_array_reductions'][0]['bindings']
    assert (row['lower_bound'], row['upper_bound']) == (-2, 4)
    assert data[1]._summary_authority()['descriptor_team_version'] == 2
    assert data[1]._summary_authority()['descriptor_team_fixed_array_reduction_version'] == 7


@pytest.mark.parametrize(('declaration', 'clause'), [
    ('real(8)::total(n)', '+:total'), ('real(8)::total(0)', '+:total'),
    ('real(8)::total(-2147483647:2147483647)', '+:total'), ('real(8)::total(2,2)', '+:total'),
    ('real(16)::total(3)', '+:total'), ('real(8),pointer::total(:)', '+:total'),
    ('real(8),allocatable::total(:)', '+:total'), ('real(8),target::total(3)', '+:total'),
    ('real(8),save::total(3)', '+:total'), ('real(8)::total(3)=0', '+:total'), ('real(8),volatile::total(3)', '+:total'),
    ('real(8),asynchronous::total(3)', '+:total'), ('integer,parameter::total(3)=1', '+:total'),
    ('real(8)::total(3)', '+:total(1)'), ('real(8)::total(3)', '+:total(:)'),
    ('real(8)::total(3)', '+:total,total'), ('real(8)::total(3)', 'iand:total'),
    ('logical(4)::total(3)', '+:total'), ('real(8)::total(3)', 'custom:total')])
def test_uncertain_shape_storage_type_and_items_decline(tmp_path, declaration, clause):
    with pytest.raises(CompilationError):
        issue(setup(tmp_path, source(declaration=declaration, clause=clause)))


@pytest.mark.parametrize('where', ['local', 'host', 'use'])
def test_user_reduction_namespace_cannot_claim_primitive_array_operator(tmp_path, where):
    text = source()
    udr = '!$omp declare reduction(+:real(8):omp_out=omp_out+omp_in)\n'
    if where == 'local':
        text = text.replace('real(8)::total(3)', udr+'real(8)::total(3)')
    elif where == 'host':
        text = text.replace('contains\n', udr+'contains\n', 1)
    else:
        text = 'module custom\n'+udr+'end module\n'+text
        text = text.replace('subroutine run(n,flag)\n', 'subroutine run(n,flag)\nuse custom\n')
    with pytest.raises(CompilationError, match='user reduction'):
        issue(setup(tmp_path, text))


def test_named_operator_shadowing_and_outer_private_reject(tmp_path):
    with pytest.raises(CompilationError, match='shadowed'):
        issue(setup(tmp_path, source(declaration='real(8),external::max\nreal(8)::total(3)', clause='max:total')))
    with pytest.raises(CompilationError, match='PRIVATE'):
        issue(setup(tmp_path, source(opening='parallel private(i,total)')))


@pytest.mark.parametrize('change', ['formal', 'module', 'common', 'equivalence', 'threadprivate'])
def test_nonlocal_or_associated_reduction_storage_has_no_local_authority(tmp_path, change):
    text = source()
    if change == 'formal':
        text = text.replace('run(n,flag)', 'run(n,flag,total)')
    elif change == 'module':
        text = text.replace('real(8)::total(3)\n', '').replace('contains\n', 'real(8)::total(3)\ncontains\n')
    elif change == 'common':
        text = text.replace('!$omp parallel', 'common /hidden/ total\n!$omp parallel', 1)
    elif change == 'equivalence':
        text = text.replace('real(8)::total(3)', 'real(8)::total(3),other(3)\nequivalence(total,other)')
    else:
        text = text.replace('!$omp parallel', '!$omp threadprivate(total)\n!$omp parallel', 1)
    with pytest.raises(CompilationError):
        issue(setup(tmp_path, text))


@pytest.mark.parametrize(('guarded', 'repeat', 'ending'), [(False, False, 'end do'), (True, True, 'end do'), (True, True, 'end do nowait')])
def test_strong_graph_retains_every_original_branch_loop_join_and_shared_effect(tmp_path, guarded, repeat, ending):
    data = setup(tmp_path, source(guarded=guarded, repeat=repeat, inner_end=ending))
    _, analysis, originals = data
    before = data[0].read_bytes()
    graph, completion = completed(data)
    record = completion.public()
    assert record['schema_version'] == 4
    assert record['completion_scope'] == 'normal return only'
    assert not record['motion_authorized']
    assert not record['internal_cuts_authorized']
    assert not record['gpu_legality_established']
    team, = completion._reduction_teams
    assert team is issue(data)
    assert record['fixed_array_reduction_teams'] == [team.public()]
    assert record['fixed_array_reduction_team_identity'] == team.identity
    assert len(team.public()['intrinsic_fixed_array_reductions']) == (3 if repeat else 1)
    assert completion.validate(analysis, PROC, team.selected_node_ids) is completion
    assert graph.deferred_completion
    assert data[0].read_bytes() == before
    for consumer in (analysis.joined_completion, analysis.numerical_joined_completion):
        with pytest.raises(CompilationError):
            consumer(PROC, originals)


@pytest.mark.parametrize('extra', ['!$omp task\n!$omp end task', '!$omp target\n!$omp end target', '!$omp single copyprivate(total)', '!$omp flush(total)'])
def test_unsupported_sync_or_async_never_gets_storage_or_completion(tmp_path, extra):
    data = setup(tmp_path, source(extra=extra))
    with pytest.raises(CompilationError):
        issue(data)
    with pytest.raises(CompilationError):
        completed(data)


def test_complete_join_cannot_be_cut_or_replaced_by_unknown_body_call(tmp_path):
    data = setup(tmp_path)
    with pytest.raises(CompilationError):
        prove_descriptor_team(data[1], PROC, data[2][:-1])
    text = source(action='call unknown(total)')
    data = setup(tmp_path, text)
    assert issue(data).public()['requires_complete_descriptor_body_and_call_rescan']
    graph = prove_native_effect_graph(data[1], PROC, data[2])
    assert not graph.completion_eligible
    with pytest.raises(CompilationError):
        prove_graph_joined_completion(data[1], PROC, data[2], graph)


@pytest.mark.parametrize('mutation', ['copy', 'delete_registry', 'replace_registry', 'drop_retained_team', 'change_record'])
def test_completion_retains_exact_registered_array_team_without_healing(tmp_path, mutation):
    data = setup(tmp_path)
    _, analysis, originals = data
    graph, completion = completed(data)
    team, = completion._reduction_teams
    if mutation == 'copy':
        completion = copy(completion)
    elif mutation == 'delete_registry':
        del analysis._descriptor_team_proofs[team.identity]
    elif mutation == 'replace_registry':
        analysis._descriptor_team_proofs[team.identity] = copy(team)
    elif mutation == 'drop_retained_team':
        object.__setattr__(completion, '_reduction_teams', ())
    else:
        object.__setattr__(completion, '_record', {})
    with pytest.raises(CompilationError):
        completion.validate(analysis, PROC, team.selected_node_ids)
    if mutation in {'delete_registry', 'replace_registry'}:
        with pytest.raises(CompilationError):
            prove_graph_joined_completion(analysis, PROC, originals, graph)


def test_copied_foreign_team_and_real_source_edit_remain_rejected(tmp_path):
    data = setup(tmp_path)
    team = issue(data)
    with pytest.raises(CompilationError):
        copy(team).validate(data[1], PROC, team.selected_node_ids)
    other = SourceEffects([data[0]])
    other.seal_session()
    with pytest.raises(CompilationError):
        team.validate(other, PROC, team.selected_node_ids)
    data[0].write_text(data[0].read_text().replace('total(1)+values(i)', 'total(1)-values(i)'))
    with pytest.raises(CompilationError, match='source|changed'):
        data[1].verify_publication()


def test_same_clause_shared_array_reduction_is_not_valid_data_sharing(tmp_path):
    with pytest.raises(CompilationError, match='SHARED'):
        issue(setup(tmp_path, source(opening='parallel reduction(+:total) shared(total) private(i)')))


def test_scalar_only_record_and_version_remain_unchanged(tmp_path):
    text = source(declaration='real(8)::total', action='total=total+values(i)')
    record = issue(setup(tmp_path, text)).public()
    assert record['version'] == 2
    assert len(record['intrinsic_scalar_reductions']) == 1
    assert 'intrinsic_fixed_array_reductions' not in record
    assert 'fixed_array_reduction_contract' not in record


def test_missing_parent_join_and_orphan_worksharing_never_complete(tmp_path):
    for text in (source().replace('!$omp end parallel', ''), source().replace('!$omp parallel private(i)', '').replace('!$omp end parallel', '')):
        with pytest.raises(CompilationError):
            issue(setup(tmp_path, text))


def test_conditional_predicate_cannot_read_changed_reduction_storage(tmp_path):
    text = source(guarded=True).replace('if(flag)then', 'if(total(1)>0)then')
    data = setup(tmp_path, text)
    with pytest.raises(CompilationError, match='private or changes'):
        completed(data)


@pytest.mark.parametrize('directive', ['task', 'target', 'task detach(event)', 'do', 'end parallel'])
@pytest.mark.parametrize('branch', ['if', 'else'])
def test_nested_new_directive_ownership_never_hides_unknown_async_or_unjoined_work(tmp_path, directive, branch):
    text = source(guarded=True)
    marker = 'if(flag)then' if branch == 'if' else 'else'
    text = text.replace(marker+'\n', marker+'\n!$omp '+directive+'\n', 1)
    try:
        data = setup(tmp_path, text)
    except CompilationError:
        return  # No complete original group exists to supply authority.
    with pytest.raises(CompilationError):
        issue(data)
    graph = prove_native_effect_graph(data[1], PROC, data[2])
    if graph.completion_eligible:
        with pytest.raises(CompilationError):
            prove_graph_joined_completion(data[1], PROC, data[2], graph)
    else:
        assert not graph.completion_eligible


def test_nested_unresolved_call_keeps_graph_effect_barrier(tmp_path):
    data = setup(tmp_path, source(guarded=True, action='call unknown(total)'))
    issue(data)
    graph = prove_native_effect_graph(data[1], PROC, data[2])
    assert not graph.completion_eligible
    with pytest.raises(CompilationError):
        prove_graph_joined_completion(data[1], PROC, data[2], graph)


def test_directives_after_exact_group_remain_unowned(tmp_path):
    text = source().replace('!$omp end parallel\n', '!$omp end parallel\n!$omp do\ndo i=1,n\ntotal(1)=values(i)\nenddo\n!$omp end do\n')
    data = setup(tmp_path, text)
    _, analysis, _ = data
    complete = tuple(analysis.routines[PROC].execution.content)
    graph = prove_native_effect_graph(analysis, PROC, complete)
    assert not graph.completion_eligible


@pytest.mark.parametrize('retag', [False, True])
def test_paired_array_field_removal_never_downgrades_an_issued_completion(tmp_path, retag):
    import hashlib
    import json

    from compiler.frontend.structured_effects import _freeze, _thaw
    data = setup(tmp_path)
    graph, completion = completed(data)
    record = _thaw(completion._record)
    del record['fixed_array_reduction_teams']
    object.__setattr__(completion, '_reduction_teams', ())
    object.__setattr__(completion, '_record', _freeze(record))
    if retag:
        identity = hashlib.sha256(json.dumps({'version': 2, 'native_graph': graph.identity,
            'procedure': PROC, 'selected': completion.selected_node_ids, 'record': record},
            sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        object.__setattr__(completion, 'identity', identity)
        data[1]._native_graph_completions[identity] = completion
    with pytest.raises(CompilationError):
        completion.validate(data[1], PROC, completion.selected_node_ids)


def test_registered_completion_copy_cannot_become_an_arena_handle(tmp_path):
    data = setup(tmp_path)
    graph, completion = completed(data)
    data[1]._native_graph_completions[completion.identity] = copy(completion)
    with pytest.raises(CompilationError, match='arena-issued'):
        prove_graph_joined_completion(data[1], PROC, data[2], graph)


def generation_fixture(tmp_path):
    from hashlib import sha256

    from compiler._legacy.config import LegacyCompilerOptions, LegacyOffloadConfig
    from compiler.scopes.source import ScopeBuilder
    path = tmp_path / 'source.f90'
    path.write_text(source())
    parent = ScopeBuilder([path], PROC, facts={'schema_version': 1, 'participation': 'serial',
        'sources': {str(path): sha256(path.read_bytes()).hexdigest()}, 'captures': {}},
        options=LegacyCompilerOptions(), config=LegacyOffloadConfig(policy='sections', scope_execution='graph'))
    routine = parent.analysis.routines[PROC]
    group, = [row for row in grouped_original_sequence(routine.execution.content, {})
              if isinstance(row, NativeGroupCandidate)]
    return parent, (path, parent.analysis, group.original_nodes)


@pytest.mark.parametrize('adopt', [False, True])
def test_parent_handle_and_registry_copy_survive_generation(tmp_path, adopt):
    from compiler.scopes.generation_transaction import GenerationTransaction
    parent, data = generation_fixture(tmp_path)
    _, analysis, _ = data
    _, completion = completed(data)
    analysis._native_graph_completions = analysis._native_graph_completions.copy()
    with GenerationTransaction(parent) as transaction:
        assert completion.validate(analysis, PROC, completion.selected_node_ids) is completion
        if adopt:
            transaction.adopt()
    assert completion.validate(analysis, PROC, completion.selected_node_ids) is completion


@pytest.mark.parametrize('adopt', [False, True])
def test_new_generation_completion_handle_requires_adoption(tmp_path, adopt):
    from compiler.scopes.generation_transaction import GenerationTransaction
    parent, data = generation_fixture(tmp_path)
    _, analysis, _ = data
    with GenerationTransaction(parent) as transaction:
        _, completion = completed(data)
        registry = analysis._native_graph_completions.copy()
        if adopt:
            transaction.adopt()
    if adopt:
        assert completion.validate(analysis, PROC, completion.selected_node_ids) is completion
    else:
        analysis._native_graph_completions = registry
        with pytest.raises(CompilationError, match='discarded generation'):
            completion.validate(analysis, PROC, completion.selected_node_ids)


def test_unsealed_array_completion_declines_without_changing_ordinary_diagnostics(tmp_path):
    data = setup(tmp_path, sealed=False)
    issue(data)  # Storage-only mutable diagnostic remains available.
    with pytest.raises(CompilationError, match='sealed'):
        completed(data)
    text = source(declaration='real(8)::total', action='total=total+values(i)').replace(
        '!$omp parallel private(i)', '!$omp parallel do private(i)').replace(
        '!$omp do reduction(+:total)\n', '').replace('!$omp end do\n', '').replace(
        '!$omp end parallel', '!$omp end parallel do')
    data = setup(tmp_path, text, sealed=False)
    _, completion = completed(data)
    assert completion.public()['schema_version'] == 2
    assert completion.validate(data[1], PROC, completion.selected_node_ids) is completion


def test_non_omp_executable_sentinel_keeps_its_host_blocker(tmp_path):
    data = setup(tmp_path, source(extra='!$acc wait'))
    graph = prove_native_effect_graph(data[1], PROC, data[2])
    assert any(item.reason == 'unclassified executable directive' for item in graph.host_blockers)


def test_guarded_scalar_team_is_not_given_new_nested_ownership(tmp_path):
    data = setup(tmp_path, source(declaration='real(8)::total', action='total=total+values(i)', guarded=True))
    record = issue(data).public()
    assert record['version'] == 2
    graph = prove_native_effect_graph(data[1], PROC, data[2])
    assert not graph.completion_eligible


@pytest.mark.parametrize('failure', ['ordinary', 'budget', 'integrity'])
def test_failed_nested_storage_issuer_cannot_suppress_directives(tmp_path, monkeypatch, failure):
    from compiler.frontend import descriptor_teams
    from compiler.frontend.resource_reachability import ResourceReachabilityBudgetExceeded
    data = setup(tmp_path, source(guarded=True))
    def fail(*_args, **_kwargs):
        if failure == 'budget':
            raise ResourceReachabilityBudgetExceeded('original storage budget')
        raise CompilationError('original storage '+failure)
    monkeypatch.setattr(descriptor_teams, 'prove_descriptor_team', fail)
    graph = prove_native_effect_graph(data[1], PROC, data[2])
    assert not graph.completion_eligible


@pytest.mark.parametrize('mutation', ['bound', 'namespace'])
def test_array_publication_rechecks_consumed_mutable_source_attachments(tmp_path, monkeypatch, mutation):
    from compiler.frontend.source_effects import Binding
    data = setup(tmp_path, sealed=False)
    original = Binding.public
    changed = False
    def intercept(self):
        nonlocal changed
        result = original(self)
        if self.name == 'total' and not changed:
            changed = True
            if mutation == 'bound':
                axis = self.shape_nodes[0]
                object.__setattr__(axis, 'items', (axis.items[0], axis.items[0]))
            else:
                self.declaring_scope.externals.add('max')
        return result
    monkeypatch.setattr(Binding, 'public', intercept)
    with pytest.raises(CompilationError, match='changed|bounds'):
        issue(data)
    assert changed


@pytest.mark.parametrize('attachment', [item.name for item in fields(NativeGraphCompletionProof)])
def test_final_completion_publication_cannot_issue_changed_attachment(tmp_path, monkeypatch, attachment):
    import sys

    data = setup(tmp_path)
    _, analysis, originals = data
    graph = prove_native_effect_graph(analysis, PROC, originals)
    verify = analysis.verify_publication
    changed = []

    def intercept():
        verify()
        if sys._getframe(1).f_code.co_name == '_publish_graph_completion':
            proof, = analysis._native_graph_completions.values()
            changed.append((proof, proof.identity))
            value = 'changed' if attachment == 'identity' else () if attachment == '_reduction_teams' else object()
            object.__setattr__(proof, attachment, value)

    monkeypatch.setattr(analysis, 'verify_publication', intercept)
    with pytest.raises(CompilationError, match='attachments changed'):
        prove_graph_joined_completion(analysis, PROC, originals, graph)
    assert len(changed) == 1
    proof, identity = changed[0]
    assert identity not in analysis._native_graph_completions
    arena = analysis.source_authority_context().derivation_arena(analysis)
    with pytest.raises(CompilationError, match='arena-issued'):
        arena.require_handle(('native_graph_completion', id(analysis), identity), proof)
    monkeypatch.setattr(analysis, 'verify_publication', verify)
    valid = prove_graph_joined_completion(analysis, PROC, originals, graph)
    assert valid is not proof
    assert valid.validate(analysis, PROC, valid.selected_node_ids) is valid


@pytest.mark.parametrize('attachment', [item.name for item in fields(NativeGraphCompletionProof)])
def test_public_completion_validation_has_final_attachment_fence(tmp_path, monkeypatch, attachment):
    data = setup(tmp_path)
    _, proof = completed(data)
    original = NativeGraphCompletionProof._require_handle
    calls = 0

    def late_handle_check(self, analysis):
        nonlocal calls
        original(self, analysis)
        if self is proof:
            calls += 1
            if calls == 2:
                object.__setattr__(self, attachment, object())

    monkeypatch.setattr(NativeGraphCompletionProof, '_require_handle', late_handle_check)
    with pytest.raises(CompilationError, match='attachments changed|publication authority changed'):
        proof.validate(data[1], PROC, proof.selected_node_ids)
    assert calls == 2


@pytest.mark.parametrize('authority', ['registry', 'session', 'context', 'arena'])
def test_public_completion_validation_retains_final_authority_attachments(tmp_path, monkeypatch, authority):
    data = setup(tmp_path)
    _, proof = completed(data)
    original = NativeGraphCompletionProof._require_handle
    calls = 0

    def late_handle_check(self, analysis):
        nonlocal calls
        original(self, analysis)
        if self is proof:
            calls += 1
            if calls == 2:
                name = {'registry': '_native_graph_completions', 'session': '_source_session',
                        'context': '_source_authority_context', 'arena': '_source_derivation_arena'}[authority]
                value = analysis._native_graph_completions.copy() if authority == 'registry' else None
                object.__setattr__(analysis, name, value)

    monkeypatch.setattr(NativeGraphCompletionProof, '_require_handle', late_handle_check)
    with pytest.raises(CompilationError, match='publication authority changed'):
        proof.validate(data[1], PROC, proof.selected_node_ids)
    assert calls == 2


@pytest.mark.parametrize('authority', ['registry', 'session', 'context', 'arena', 'raise'])
def test_final_completion_publication_preserves_authority_and_cleanup(tmp_path, monkeypatch, authority):
    import sys

    data = setup(tmp_path)
    _, analysis, originals = data
    graph = prove_native_effect_graph(analysis, PROC, originals)
    verify = analysis.verify_publication
    changed = []

    def intercept():
        verify()
        if sys._getframe(1).f_code.co_name != '_publish_graph_completion':
            return
        registry = analysis._native_graph_completions
        proof, = registry.values()
        changed.append((registry, proof.identity))
        if authority == 'raise':
            raise CompilationError('delegated publication failed')
        name = {'registry': '_native_graph_completions', 'session': '_source_session',
                'context': '_source_authority_context', 'arena': '_source_derivation_arena'}[authority]
        object.__setattr__(analysis, name, {} if authority == 'registry' else None)

    monkeypatch.setattr(analysis, 'verify_publication', intercept)
    with pytest.raises(CompilationError):
        prove_graph_joined_completion(analysis, PROC, originals, graph)
    assert len(changed) == 1
    registry, identity = changed[0]
    assert identity not in registry


def test_failed_completion_publication_keeps_foreign_registry_replacement(tmp_path, monkeypatch):
    import sys

    data = setup(tmp_path)
    _, analysis, originals = data
    graph = prove_native_effect_graph(analysis, PROC, originals)
    verify = analysis.verify_publication
    replacement = object()
    changed = []

    def intercept():
        verify()
        if sys._getframe(1).f_code.co_name == '_publish_graph_completion':
            proof, = analysis._native_graph_completions.values()
            changed.append(proof.identity)
            analysis._native_graph_completions[proof.identity] = replacement

    monkeypatch.setattr(analysis, 'verify_publication', intercept)
    with pytest.raises(CompilationError, match='registry changed'):
        prove_graph_joined_completion(analysis, PROC, originals, graph)
    assert len(changed) == 1
    assert analysis._native_graph_completions[changed[0]] is replacement
