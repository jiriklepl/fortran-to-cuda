"""Retired native islands retain exact graph-owned local record aliases."""
from copy import copy

import pytest

from compiler.frontend.native_completion import native_call_storage_structures, prove_native_call_retirement
from compiler.frontend.resource_reachability import prove_native_effect_graph
from compiler.frontend.source_effects import SourceEffects
from compiler.frontend.storage_mutation_effects import prove_storage_mutation_effects
from compiler.ir import CompilationError

SOURCE = '''module work
implicit none
type record
real(8)::flux=0.0_8,aux=0.0_8
end type
type(record),target::points(4)
real(8)::input(4)
contains
subroutine owner()
call update(points,input,1)
end subroutine
subroutine update(items,values,component)
type(record),target,intent(inout)::items(:)
real(8),intent(in)::values(:)
integer,intent(in)::component
type(record),pointer::p
integer::i
!$omp parallel do private(i,p) schedule(guided,5)
do i=1,size(items)
p=>items(i)
p%aux=values(i)
call helper(p%flux,p%aux,component)
enddo
!$omp end parallel do
end subroutine
subroutine helper(value,input,component)
real(8),intent(inout)::value
real(8),intent(in)::input
integer,intent(in)::component
select case(component)
case(1)
value=value+input
case default
value=value-input
end select
end subroutine
end module
'''


def setup(tmp_path, source=SOURCE, *, seal=True):
    path = tmp_path / 'record.f90'
    path.write_text(source)
    analysis = SourceEffects([path])
    if seal:
        analysis.seal_session()
    selected = tuple(analysis.routines['work::owner'].execution.content)
    native = prove_native_effect_graph(analysis, 'work::owner', selected)
    return path, analysis, selected, native


def prove(tmp_path, source=SOURCE, *, seal=True):
    path, analysis, selected, native = setup(tmp_path, source, seal=seal)
    assert native.retirement_effects_eligible, native.public()['barriers']
    teams = native_call_storage_structures(analysis, native)
    storage = prove_storage_mutation_effects(analysis, 'work::owner', selection=selected,
                                            joined_storage=teams, native_graph=native)
    assert storage.public()['complete'], storage.public()['frontiers']
    assert not storage.public()['escapes']
    completion = prove_native_call_retirement(analysis, 'work::owner', selected, native, storage)
    return path, analysis, selected, native, teams, storage, completion


@pytest.mark.parametrize('seal', [False, True])
def test_fixed_record_writes_have_only_native_retirement_authority(tmp_path, seal):
    _path, analysis, selected, native, teams, storage, completion = prove(tmp_path, seal=seal)
    assert len(teams) == 1
    assert teams[0]._native_graph is native
    assert storage._native_graph is native
    assert storage.public()['version'] == 20
    assert not storage.public()['mutations']
    assert not storage.public()['payload_access_authorized']
    assert completion.public()['requires_domain_retirement']
    assert completion.validate(analysis, 'work::owner', completion.selected_node_ids) is completion
    plain = prove_storage_mutation_effects(analysis, 'work::owner', selection=selected)
    assert not plain.public()['complete']
    assert plain.public()['escapes']
    analysis.verify_publication()


@pytest.mark.parametrize(('before', 'after'), [
    ('p=>items(i)', 'p=>items(i)\np=>items(i)'),
    ('private(i,p)', 'private(i)'),
    ('type(record),pointer::p', 'type(record),pointer,save::p'),
    ('enddo\n!$omp end parallel do', 'enddo\n!$omp end parallel do\np%aux=0.0_8'),
    ('p%aux=values(i)', 'nullify(p)'),
    ('p%aux=values(i)', 'call unknown(p)'),
    ('value=value+input', 'call hidden(value)'),
    ('p%aux=values(i)', '!$omp task\np%aux=values(i)\n!$omp end task'),
    ('real(8)::flux=0.0_8,aux=0.0_8', 'real(8)::flux=0.0_8,aux=0.0_8\nreal(8),allocatable::other(:)'),
    ('real(8)::flux=0.0_8,aux=0.0_8', 'real(8)::flux=0.0_8,aux=0.0_8\nreal(8),pointer::other'),
])
def test_unproved_pointer_lifetime_and_hidden_effects_reject(tmp_path, before, after):
    source = SOURCE.replace(before, after)
    try:
        _path, analysis, selected, native = setup(tmp_path, source)
        teams = native_call_storage_structures(analysis, native)
        storage = prove_storage_mutation_effects(analysis, 'work::owner', selection=selected,
                                                joined_storage=teams, native_graph=native)
        prove_native_call_retirement(analysis, 'work::owner', selected, native, storage)
    except CompilationError:
        return
    pytest.fail('unsupported record lifetime obtained native normal-return authority')


@pytest.mark.parametrize('change', ['copied_graph', 'foreign_graph', 'removed_call', 'replaced_call',
                                   'missing_team_graph', 'missing_storage_graph', 'copied_team'])
def test_retirement_consumption_keeps_exact_graph_and_original_associations(tmp_path, change):
    _path, analysis, selected, native, teams, storage, completion = prove(tmp_path)
    if change == 'copied_graph':
        object.__setattr__(storage, '_native_graph', copy(native))
    elif change == 'foreign_graph':
        other = tmp_path / 'other'
        other.mkdir()
        object.__setattr__(storage, '_native_graph', prove(other)[3])
    elif change in {'removed_call', 'replaced_call'}:
        call = next(call.edge.resolved for template in native.templates for call in template.calls)
        registry = analysis._original_call_associations
        key = next(key for key, value in registry.items() if value is call)
        if change == 'removed_call':
            del registry[key]
        else:
            registry[key] = copy(call)
    elif change == 'missing_team_graph':
        object.__setattr__(teams[0], '_native_graph', None)
    elif change == 'missing_storage_graph':
        object.__setattr__(storage, '_native_graph', None)
    else:
        object.__setattr__(storage, '_joined_storage', (copy(teams[0]),))
    with pytest.raises(CompilationError):
        completion.validate(analysis, 'work::owner', completion.selected_node_ids)


def test_changed_source_cannot_publish_issued_retirement(tmp_path):
    path, analysis, _selected, _native, _teams, _storage, _completion = prove(tmp_path)
    path.write_text(SOURCE.replace('value=value+input', 'call hidden(value)'))
    with pytest.raises(CompilationError, match='changed'):
        analysis.verify_publication()


def hierarchical_source():
    from compiler.tests.test_hierarchical_child_regions import SOURCE as HIERARCHY
    declarations = 'type record\nreal(8)::flux,aux\nend type\ntype(record),target::native_points(257)\n'
    worker = '''subroutine record_worker(points,values,n)
integer,intent(in)::n
real(8),intent(inout)::values(:)
type(record),target,intent(inout)::points(:)
call update(points,values,1)
end subroutine
'''
    updates = SOURCE[SOURCE.index('subroutine update('):SOURCE.index('end module')]
    updates = updates.replace('real(8),intent(in)::values(:)', 'real(8),intent(inout)::values(:)')
    updates = updates.replace('do i=1,size(items)', 'do i=1,size(values)')
    updates = updates.replace('p%aux=values(i)', 'p%flux=values(i)\np%aux=1.0_8')
    updates = updates.replace('call helper(p%flux,p%aux,component)',
                              'call helper(p%flux,p%aux,component)\nvalues(i)=p%flux')
    return (HIERARCHY.replace('implicit none\n', 'implicit none\n'+declarations, 1)
            .replace('if(change)call relocate(left,right)', 'call record_worker(native_points,right,n)')
            .replace('end module', worker+updates+'end module'))


@pytest.mark.parametrize('mode', ['retained', 'diagnostic_decline', 'scalar_formal'])
def test_authentic_hierarchy_emission_and_strict_transport(tmp_path, monkeypatch, mode):
    import json

    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans

    source = hierarchical_source()
    if mode == 'scalar_formal':
        source = source.replace('call update(points,values,1)', 'call update(points,values,n)')
    if mode != 'retained':
        from compiler.scopes import segments
        from compiler.tests.test_joined_boundary_lifecycle import emit

        def decline(*_args, **_kwargs):
            raise CompilationError('diagnostic complete-CALL admission decline')

        if mode == 'diagnostic_decline':
            monkeypatch.setattr(segments, 'native_call_fragment', decline)
        path, builder, middle, patches, outputs, report = emit(tmp_path, source)
        patch, = patches
        assert patch.lifecycle._native_graph is patch.completion._graph
        assert patch.lifecycle.public()['version'] == 20
        assert patch.validate(middle, require_source=True) is patch
        assert patch.completion.public()['requires_domain_retirement']
    else:
        from compiler.tests.test_hierarchical_child_regions import fixture

        path, builder = fixture(tmp_path, source)
        outputs, report = builder.run()
        scope, = report['scopes']
        member, = [row for row in scope['checked_execution_plan']['procedures']
                    if row['procedure'] == 'hierarchical::middle']
        operation, = [op for unit in member['units'] for op in unit['operations']['native_operations']
                       if op['kind'] == 'complete original native CALL']
        assert operation['resource_effects'] == {'hierarchical::right': ['read', 'write']}
        assert operation['preserves_original_source']
        assert 'argument::n' not in operation['host_metadata']
        assert not member['boundaries']
    generated = outputs[report['sources'][str(path)]['replacement']]
    assert 'call record_worker(native_points,right,n)' in generated
    assert 'type(record),target,intent(inout)::points(:)' in generated
    validate_checked_execution_plans(report)
    validate_checked_execution_plans(json.loads(outputs['scope-manifest.json']))
    assert builder.finish()
