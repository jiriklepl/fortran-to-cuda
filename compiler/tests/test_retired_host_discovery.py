"""Retired-host control discovery belongs to the existing native source graph."""
from dataclasses import replace

import pytest

from compiler.frontend.resource_reachability import prove_native_effect_graph
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError

SOURCE='''module work
implicit none
type base
integer::xi,yj,zk
end type
type,extends(base)::record
real(8)::flux
end type
real(8)::left(4,4,4),right(4,4,4)
type(record),allocatable::records(:)
contains
subroutine owner()
call apply(left,right,records,scale1)
call apply(left,right,records,scale2)
end subroutine
subroutine apply(x2,x,src,fun)
real(8),intent(inout)::x2(:,:,:)
real(8),intent(in)::x(:,:,:)
type(record),allocatable,intent(in)::src(:)
procedure(scale1)::fun
integer::i
if(allocated(src))then
do i=1,size(src)
associate(xi=>src(i)%xi,yj=>src(i)%yj,zk=>src(i)%zk)
x2(xi,yj,zk)=x2(xi,yj,zk)-src(i)%flux*fun(xi,yj,zk)*x(xi,yj,zk)
end associate
enddo
endif
end subroutine
pure real(8) function scale1(i,j,k)
integer,intent(in)::i,j,k
scale1=hypot(right(i,j,k),1.0_8)
end function
pure real(8) function scale2(i,j,k)
integer,intent(in)::i,j,k
scale2=right(i,j,k)
end function
end module
'''


def issue(tmp_path, source=SOURCE, *, seal=True, project_payload=False):
    path=tmp_path/'source.f90';path.write_text(source)
    analysis=SourceEffects([path])
    if seal:analysis.seal_session()
    selection=tuple(analysis.routines['work::owner'].execution.content)
    graph=prove_native_effect_graph(analysis,'work::owner',selection,project_payload=project_payload)
    return path,analysis,selection,graph


def test_one_graph_keeps_strong_payload_decline_and_independent_retired_host_view(tmp_path):
    _path,analysis,selection,graph=issue(tmp_path)
    assert not graph.completion_eligible and not graph.retirement_effects_eligible
    view=graph.retired_host_completion_view()
    assert view.validate(analysis,'work::owner',selection)is view
    public=view.public()
    assert public['available'],public['blockers']
    assert public['requires_quiescent_host_entry']
    assert public['requires_continuing_foreign_access_exclusion']
    assert public['future_activation_requires_quiescence_for_current_allocations']
    assert public['requires_full_owning_domain_retirement']
    assert public['requires_untouched_original_execution']
    assert public['completion_scope']=='normal return only'
    for flag in ['initial_quiescence_authorized','boundary_admission_authorized','payload_access_authorized',
                 'stable_storage_authorized','alias_authority','entry_definition_authority','motion_authority',
                 'cross_cut_retention_authorized','termination_guaranteed']:
        assert public[flag]is False
    instances=public['instances']
    assert [row['procedure']for row in instances]==[
        'work::owner','work::apply','work::scale1','work::apply','work::scale2']
    first,second=[row for row in instances if row['procedure']=='work::apply']
    assert first['procedure_bindings']!=second['procedure_bindings']
    assert {t.procedure for t in graph.templates}=={'work::owner','work::apply','work::scale1','work::scale2'}
    analysis.verify_publication()


def test_hidden_subscript_call_is_discovered_before_rejected_payload_metadata(tmp_path):
    source=SOURCE.replace('src(i)%xi','src(next_index())%xi',1).replace('end module','''integer function next_index()
call unknown_async()
next_index=1
end function
end module''')
    _path,_analysis,_selection,graph=issue(tmp_path,source)
    view=graph.retired_host_completion_view().public()
    assert not view['available']
    assert 'work::next_index'in [row['procedure']for row in view['instances']]
    assert any('unknown_async'in row['source']for row in view['blockers'])


@pytest.mark.parametrize('change',['policy','foreign_selection','callback_receipt'])
def test_projection_binds_source_policy_selection_and_callback_instance_receipts(tmp_path,change):
    _path,analysis,selection,graph=issue(tmp_path)
    view=graph.retired_host_completion_view()
    if change=='policy':
        analysis.depth_limit+=1
    elif change=='foreign_selection':
        other=tmp_path/'other';other.mkdir()
        selection=issue(other)[2]
    else:
        altered=list(graph.host_instances)
        altered[1]=replace(altered[1],procedure_bindings=altered[3].procedure_bindings)
        object.__setattr__(graph,'host_instances',tuple(altered))
    with pytest.raises(CompilationError):view.validate(analysis,'work::owner',selection)


def test_live_source_mutation_still_fails_original_publication(tmp_path):
    path,analysis,_selection,graph=issue(tmp_path)
    assert graph.retired_host_completion_view().public()['available']
    path.write_text(path.read_text().replace('scale2=right','scale2=2*right'))
    with pytest.raises(CompilationError):analysis.verify_publication()


def test_projection_requires_sealed_session(tmp_path):
    _path,_analysis,_selection,graph=issue(tmp_path,seal=False)
    with pytest.raises(CompilationError,match='SourceSession'):graph.retired_host_completion_view()


@pytest.mark.parametrize('body',[
    'call unknown()',
    '!$omp task\nscale2=right(i,j,k)\n!$omp end task',
    '!$omp target nowait\nscale2=right(i,j,k)\n!$omp end target',
    '!$omp do\nscale2=right(i,j,k)\n!$omp end do',
])
def test_pure_interface_never_hides_unknown_async_or_orphan_work(tmp_path,body):
    _path,_analysis,_selection,graph=issue(tmp_path,SOURCE.replace('scale2=right(i,j,k)',body))
    assert not graph.retired_host_completion_view().public()['available']


def test_primitive_field_access_does_not_execute_unrelated_finalizer(tmp_path):
    source=SOURCE.replace('real(8)::flux\nend type','real(8)::flux\ncontains\nfinal::finish\nend type')
    source=source.replace('end module','''subroutine finish(item)
type(record)::item
call unknown_finalizer()
end subroutine
end module''')
    _path,_analysis,_selection,graph=issue(tmp_path,source)
    assert graph.retired_host_completion_view().public()['available']
    assert 'work::finish'not in [row.procedure for row in graph.host_instances]


@pytest.mark.parametrize('operation',[
    'src(1)=src(2)',
    'deallocate(src)',
])
def test_whole_derived_lifetime_operations_need_separate_hidden_call_coverage(tmp_path,operation):
    source=SOURCE.replace('integer::i\nif(allocated(src))then','integer::i\n'+operation+'\nif(allocated(src))then')
    _path,_analysis,_selection,graph=issue(tmp_path,source)
    assert not graph.retired_host_completion_view().public()['available']


def test_local_derived_exit_lifetime_is_checked_even_when_unused(tmp_path):
    source=SOURCE.replace('integer::i\nif(allocated(src))then','integer::i\ntype(record)::unused\nif(allocated(src))then')
    _path,_analysis,_selection,graph=issue(tmp_path,source)
    assert not graph.retired_host_completion_view().public()['available']


def test_primitive_allocation_save_move_and_passive_pointer_mutation_have_no_payload_authority(tmp_path):
    source='''module work
implicit none
real(8),allocatable,target::a(:),b(:)
real(8),pointer::p(:)
contains
subroutine owner()
call native()
end subroutine
subroutine native()
real(8),allocatable,save::saved(:)
if(allocated(a))deallocate(a)
allocate(a(7))
a=1
saved=a
p=>a
call move_alloc(a,b)
nullify(p)
end subroutine
end module
'''
    _path,_analysis,_selection,graph=issue(tmp_path,source)
    public=graph.retired_host_completion_view().public()
    assert public['available'],public['blockers']
    assert not graph.completion_eligible


def test_ordinary_joined_parallel_do_uses_structural_completion_without_payload_geometry(tmp_path):
    source=SOURCE.replace('scale2=right(i,j,k)','''!$omp parallel do private(q)
do q=1,4
scratch(q)=right(q,j,k)
enddo
!$omp end parallel do
scale2=scratch(i)''').replace('pure real(8) function scale2(i,j,k)\ninteger,intent(in)::i,j,k',
        'pure real(8) function scale2(i,j,k)\ninteger,intent(in)::i,j,k\ninteger::q\nreal(8)::scratch(4)')
    _path,_analysis,_selection,graph=issue(tmp_path,source)
    public=graph.retired_host_completion_view().public()
    assert public['available'],public['blockers']


def test_source_only_mode_never_invokes_payload_or_joined_payload_issuers(tmp_path,monkeypatch):
    from compiler.frontend import resource_reachability as module
    path=tmp_path/'source.f90';path.write_text(SOURCE)
    analysis=SourceEffects([path]);analysis.seal_session()
    def forbidden(*_args,**_kwargs):
        pytest.fail('source-only discovery invoked payload typing/completion')
    monkeypatch.setattr(module._Scan,'storage',forbidden)
    monkeypatch.setattr(module,'resolve_original_call',forbidden)
    monkeypatch.setattr(module,'resolve_original_function',forbidden)
    monkeypatch.setattr(analysis,'joined_completion',forbidden)
    selection=tuple(analysis.routines['work::owner'].execution.content)
    graph=prove_native_effect_graph(analysis,'work::owner',selection,project_payload=False)
    assert graph.retired_host_completion_view().public()['available']
    assert not graph.completion_eligible and not graph.retirement_effects_eligible


def test_dynamic_procedure_actual_cannot_become_a_fixed_callback(tmp_path):
    source=SOURCE.replace('subroutine owner()','subroutine owner()\nprocedure(scale1),pointer::dynamic').replace(
        'call apply(left,right,records,scale1)','dynamic=>scale1\ncall apply(left,right,records,dynamic)')
    _path,_analysis,_selection,graph=issue(tmp_path,source)
    assert not graph.retired_host_completion_view().public()['available']


def test_unknown_defined_assignment_is_not_hidden_by_primitive_operands(tmp_path):
    source=SOURCE.replace('contains\nsubroutine owner()',"""interface assignment(=)
module procedure assign
end interface
contains
subroutine assign(x,y)
real(8),intent(out)::x
logical,intent(in)::y
call unknown_async()
end subroutine
subroutine owner()""").replace('scale2=right(i,j,k)','scale2=.true.')
    _path,_analysis,_selection,graph=issue(tmp_path,source)
    assert not graph.retired_host_completion_view().public()['available']


def test_selected_return_cannot_hide_enclosing_exit_finalization(tmp_path):
    source=SOURCE.replace('call apply(left,right,records,scale1)\ncall apply(left,right,records,scale2)',
                          'type(record)::unused\nreturn')
    _path,_analysis,_selection,graph=issue(tmp_path,source)
    assert not graph.retired_host_completion_view().public()['available']


def test_original_automatic_array_bound_calls_belong_to_the_same_discovery_graph(tmp_path):
    source=SOURCE.replace('integer::i\nif(allocated(src))then','integer::i\nreal(8)::scratch(lower_bound():4)\nif(allocated(src))then')
    source=source.replace('end module','pure integer function lower_bound()\nlower_bound=1\nend function\nend module')
    _path,_analysis,_selection,graph=issue(tmp_path,source)
    public=graph.retired_host_completion_view().public()
    assert public['available'],public['blockers']
    assert sum(row['procedure']=='work::lower_bound'for row in public['instances'])==2


@pytest.mark.parametrize('attributes',['intent(out)','value'])
def test_derived_entry_or_value_finalization_is_not_hidden_by_an_empty_callee(tmp_path,attributes):
    source='module work\nimplicit none\ntype record\ninteger::value\nend type\ntype(record)::item\ncontains\nsubroutine owner()\ncall native(item)\nend subroutine\nsubroutine native(value)\ntype(record),ATTRIBUTES::value\nend subroutine\nend module\n'.replace('ATTRIBUTES',attributes)
    _path,_analysis,_selection,graph=issue(tmp_path,source)
    assert not graph.retired_host_completion_view().public()['available']


def test_payload_and_control_graph_roles_have_distinct_ids_and_cache_purposes(tmp_path):
    _path,analysis,selection,weak=issue(tmp_path)
    strong=prove_native_effect_graph(analysis,'work::owner',selection,project_payload=True)
    assert strong.identity!=weak.identity and strong is not weak
    assert all(template._payload_requested for template in strong.templates)
    assert not any(template._payload_requested for template in weak.templates)
    assert not {id(t)for t in strong.templates}&{id(t)for t in weak.templates}
    assert prove_native_effect_graph(analysis,'work::owner',selection,project_payload=False)is weak
    assert strong.validate(analysis,'work::owner',selection)is strong
    for operation in [lambda:weak.validate(analysis,'work::owner',selection),weak.public]:
        with pytest.raises(CompilationError,match='control-only graph role'):operation()
    object.__setattr__(weak,'completion_eligible',True)
    object.__setattr__(weak,'retirement_effects_eligible',True)
    with pytest.raises(CompilationError,match='control-only graph role'):
        weak.validate(analysis,'work::owner',selection)


ROLE_SOURCE='module work\nimplicit none\nreal(8)::a(4)\ncontains\nsubroutine owner()\ncall worker()\nend subroutine\nsubroutine worker()\ninteger::i\n!$omp parallel do private(i)\ndo i=1,4\na(i)=real(i,8)\nenddo\n!$omp end parallel do\nend subroutine\nend module\n'


@pytest.mark.parametrize('consumer',['lifetime','storage','team','joined_completion','retirement_structures',
                                     'retirement_completion','private_joined_facts','private_group_inventory','reachability'])
def test_every_existing_native_authority_consumer_rejects_control_only_graph(tmp_path,consumer):
    from compiler.frontend.native_group_structure import grouped_original_sequence,NativeGroupCandidate
    from compiler.frontend.native_lifetime_context import NativeLifetimeContext
    from compiler.frontend.storage_mutation_effects import prove_storage_mutation_effects
    from compiler.frontend.descriptor_teams import prove_descriptor_team
    from compiler.frontend.native_completion import (prove_graph_joined_completion,native_call_storage_structures,
        prove_native_call_retirement,_graph_joined_facts,_native_call_groups)
    from compiler.frontend.resource_reachability import prove_resource_reachability
    _path,analysis,selection,weak=issue(tmp_path,ROLE_SOURCE)
    worker=analysis.routines['work::worker']
    candidate,=grouped_original_sequence(tuple(worker.execution.content),{})
    assert isinstance(candidate,NativeGroupCandidate)
    group=candidate.original_nodes
    if consumer=='lifetime':operation=lambda:NativeLifetimeContext(analysis,weak)
    elif consumer=='storage':operation=lambda:prove_storage_mutation_effects(analysis,'work::owner',selection=selection,joined_storage=(),native_graph=weak)
    elif consumer=='team':operation=lambda:prove_descriptor_team(analysis,'work::worker',group,native_graph=weak)
    elif consumer=='joined_completion':operation=lambda:prove_graph_joined_completion(analysis,'work::worker',group,weak)
    elif consumer=='retirement_structures':operation=lambda:native_call_storage_structures(analysis,weak)
    elif consumer=='retirement_completion':operation=lambda:prove_native_call_retirement(analysis,'work::owner',selection,weak,None)
    elif consumer=='private_joined_facts':operation=lambda:_graph_joined_facts(analysis,'work::worker',group,weak)
    elif consumer=='private_group_inventory':operation=lambda:_native_call_groups(weak)
    else:
        binding=analysis.modules['work'].bindings['a']
        proof=prove_resource_reachability(analysis,'work::owner',selection,(binding,))
        object.__setattr__(proof,'native_graph',weak)
        operation=lambda:proof.validate(analysis,'work::owner',selection,(binding,))
    with pytest.raises(CompilationError,match='control-only graph role'):operation()


def test_projection_cannot_be_relabelled_as_initial_quiescence_or_boundary_admission(tmp_path):
    import json
    _path,analysis,selection,graph=issue(tmp_path)
    view=graph.retired_host_completion_view()
    record=json.loads(view._record)
    record['initial_quiescence_authorized']=True
    record['boundary_admission_authorized']=True
    with pytest.raises(CompilationError):
        replace(view,_record=json.dumps(record)).validate(analysis,'work::owner',selection)


def test_global_control_policy_revision_cannot_reuse_a_template_or_projection(tmp_path,monkeypatch):
    from compiler.frontend import resource_reachability as module
    _path,analysis,selection,graph=issue(tmp_path)
    view=graph.retired_host_completion_view()
    monkeypatch.setattr(module,'HOST_CONTROL_VERSION',module.HOST_CONTROL_VERSION+1)
    with pytest.raises(CompilationError):view.validate(analysis,'work::owner',selection)
    with pytest.raises(CompilationError):
        prove_native_effect_graph(analysis,'work::owner',selection,project_payload=False)
