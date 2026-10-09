"""Original source control continues once after a reached resident escape."""

from hashlib import sha256

from compiler.driver.options import CompilerOptions
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_source_scopes import FACT


SOURCE = """module local_owner
implicit none
integer::visits=0
contains
subroutine step(a,b,out,n,escape)
real(8),intent(in)::a(-2:)
real(8),intent(inout)::b(-2:),out(-2:)
integer,intent(in)::n
logical,intent(in)::escape
integer::i
visits=visits+1
do i=-2,n-3
b(i)=2*a(i)+real(i,8)
enddo
if(escape) then
call opaque(b,n)
endif
do i=-2,n-3
out(i)=a(i)+b(i)
enddo
end subroutine
end module
"""

INTERNAL_SOURCE = SOURCE.replace('if(escape) then\ncall opaque(b,n)\nendif',
    'call manage()\ncall manage()').replace('end subroutine\nend module', '''contains
subroutine manage()
integer,save::times=0
integer::j
times=times+1
visits=visits+times
if(escape) then
call opaque(b,n)
endif
do j=-2,n-3
b(j)=b(j)+a(j)
enddo
end subroutine
end subroutine
end module''')

ASSOCIATE_SOURCE = SOURCE.replace('visits=visits+1',
    'associate(count=>visits)\ncount=count+1').replace('end subroutine\nend module',
    'end associate\nend subroutine\nend module')

GUARDED_CHILD_SOURCE = SOURCE.replace('subroutine step', '''subroutine adjust(x,n)
real(8),intent(inout)::x(-2:)
integer,intent(in)::n
integer::j
visits=visits+11
do j=-2,n-3
x(j)=x(j)+7
enddo
end subroutine
subroutine step''', 1).replace('if(escape) then\ncall opaque(b,n)\nendif', 'if(escape) call adjust(b,n)')

PROTECTED_SOURCE = '''module readonly_controls
integer,protected::coefficients(3)=[1,2,3]
end module
''' + SOURCE.replace('module local_owner\nimplicit none',
    'module local_owner\nuse readonly_controls,only:coefficients\nimplicit none').replace(
    '2*a(i)+real(i,8)', '2*a(i)+real(i,8)+real(coefficients(2),8)')

MODULE_SOURCE = '''module stage_child
implicit none
integer::child_visits=0
contains
subroutine adjust(x,n,escape)
real(8),intent(inout)::x(-4:)
integer,intent(in)::n
logical,intent(in)::escape
integer::j
child_visits=child_visits+1
do j=-4,n-5
x(j)=x(j)+real(j,8)
enddo
if(escape) then
call opaque(x,n)
endif
end subroutine
end module
''' + SOURCE.replace('module local_owner\nimplicit none',
    'module local_owner\nuse stage_child, only: adjust\nimplicit none').replace(
        'if(escape) then\ncall opaque(b,n)\nendif', 'call adjust(b,n,escape)\ncall adjust(b,n,.false.)')

PAYLOAD_SOURCE = SOURCE.replace('if(escape) then', 'if(b(-2)>10.d0) then')

STRIDED_MODULE_SOURCE = MODULE_SOURCE.replace('intent(inout)::x(-4:)',
    'contiguous,intent(inout)::x(-4:)').replace('b(i)=2*a(i)+real(i,8)', 'out(i)=2*a(i)+real(i,8)')

RETURN_MODULE_SOURCE = MODULE_SOURCE.replace('child_visits=child_visits+1',
    'child_visits=child_visits+1\nif (.not.escape) return').replace('call opaque(x,n)', 'return')

FIRST_GUARD_MODULE_SOURCE = MODULE_SOURCE.replace('child_visits=child_visits+1',
    'if(x(-4)>0.d0) child_visits=child_visits+1')

FIRST_GUARD_SOURCE = SOURCE.replace('visits=visits+1', 'if(b(-2)<0.d0) visits=visits+1')

ALLOCATION_GUARD_SOURCE = SOURCE.replace('integer::visits=0',
    'integer::visits=0\nreal(8),allocatable::spare(:)').replace('if(escape) then', 'if(allocated(spare)) then')

NATIVE_ATOMIC_SOURCE = SOURCE.replace('if(escape) then\ncall opaque(b,n)\nendif',
    '!$omp parallel private(i)\n' + ('!$omp do\ndo i=-2,n-3\n'
    'b(i)=b(i)+a(i)+b(i)+a(i)+sum(a)\nenddo\n!$omp end do\n')*24 + '!$omp end parallel')

GUARDED_NATIVE_SOURCE = SOURCE.replace('integer::visits=0', '''integer::visits=0
type cell
integer::coordinate
end type
type(cell),allocatable::metadata(:)''').replace('if(escape) then\ncall opaque(b,n)\nendif',
    'call guarded(b,metadata,transform)\ncall guarded(out,metadata,transform)').replace(
    'end subroutine\nend module', '''contains
pure real(8) function transform(x)
real(8),intent(in)::x
transform=x+2
end function
subroutine guarded(x,items,fun)
real(8),intent(inout)::x(:)
type(cell),allocatable,intent(in)::items(:)
procedure(transform)::fun
integer::j
if(allocated(items)) then
do j=1,size(items)
x(items(j)%coordinate)=fun(x(items(j)%coordinate))
enddo
endif
end subroutine
end subroutine
end module''')

NATIVE_METADATA_SOURCE = SOURCE.replace('integer::visits=0', '''integer::visits=0
type cell
integer::coordinate
real(8)::weight
end type
type(cell),allocatable,target::metadata(:)''').replace('if(escape) then\ncall opaque(b,n)\nendif', '''if(allocated(metadata)) then
do i=1,size(metadata)
b(metadata(i)%coordinate)=b(metadata(i)%coordinate)+metadata(i)%weight
enddo
endif''')

METADATA_ALIAS_SOURCE = NATIVE_METADATA_SOURCE.replace('type cell\ninteger::coordinate',
    'type,bind(C)::cell').replace('do i=1,size(metadata)', 'do i=1,n').replace('metadata(i)%coordinate', 'i-3')

NESTED_MODULE_SOURCE = '''module deep_child
implicit none
real(8)::weights(8)=1.d0
contains
subroutine leaf(y,n,escape)
real(8),intent(inout)::y(-6:)
integer,intent(in)::n
logical,intent(in)::escape
integer::k
do k=-6,n-7
y(k)=y(k)+weights(1)
enddo
if(escape) then
call opaque(y,n)
endif
end subroutine
end module
''' + MODULE_SOURCE.replace('module stage_child\nimplicit none',
    'module stage_child\nuse deep_child,only:leaf\nimplicit none').replace(
        'call opaque(x,n)', 'call leaf(x,n,escape)')


def emit(directory, source=SOURCE, *, captures=None):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/'owner.f90'
    path.write_text(source)
    facts = {'schema_version':1, 'participation':'serial',
             'sources':{str(path):sha256(path.read_bytes()).hexdigest()},
             'captures':{'argument::'+name:FACT for name in ('a','b','out')}}
    facts['captures'].update(captures or {})
    builder = ScopeBuilder([path], 'local_owner::step', facts=facts, options=CompilerOptions(),
                           config=OffloadConfig(policy='sections',scope_execution='reached'))
    outputs, report = builder.run()
    return path, outputs, report


def test_inactive_unknown_branch_is_a_reached_close_without_wrapper_copies(tmp_path):
    path, outputs, report = emit(tmp_path)
    assert path.read_text() == SOURCE
    owner, = report['scopes']
    assert len(owner['gpu_leaves']) == 2
    assert len(owner['planning_segments']) == 2
    assert len(owner['boundaries']) == 1
    assert owner['reopen_after_boundary'] is False
    text = outputs[report['sources'][str(path)]['replacement']]
    assert text.count('visits=visits+1') == 1
    assert text.count('call opaque(b,n)') == 1
    assert text.index('if(escape) then') < text.index('call opaque(b,n)')
    between = text[text.index('if(escape) then'):text.index('call opaque(b,n)')]
    assert 'fort_scope_close(' in between
    assert 'save' not in text.lower()
    assert 'subroutine fort_scope_owner_' not in text
    assert 'TARGET :: a' in text
    assert 'TARGET :: b(' in text
    assert report['implementation_variants']['generated_count'] == 2


def test_reached_return_closes_without_executing_the_prefix_again(tmp_path):
    source = SOURCE.replace('call opaque(b,n)', 'return')
    _, outputs, report = emit(tmp_path, source)
    text = next(text for name,text in outputs.items() if name.startswith('sources/'))
    assert text.count('visits=visits+1') == 1
    before = text[text.index('if(escape) then'):text.index('\nreturn')]
    assert 'fort_scope_close(' in before
    assert report['scopes'][0]['boundaries'][0]['reason'].endswith('Return_Stmt')


def test_original_internal_coordinator_shares_host_control_and_saved_state(tmp_path):
    path, outputs, report = emit(tmp_path, INTERNAL_SOURCE)
    owner, = report['scopes']
    child, = owner['internal_coordinators']
    assert child['procedure'] == 'local_owner::step::manage'
    assert child['control'] == 'host association; original declarations and body'
    assert len(child['planning_segments']) == 1
    assert len(child['boundaries']) == 1
    assert len(owner['internal_calls']) == 2
    assert len(owner['gpu_leaves']) == 3
    text = outputs[report['sources'][str(path)]['replacement']]
    assert text.count('subroutine manage()') == 1
    assert text.count('integer,save::times=0') == 1
    assert text.count('times=times+1') == 1
    assert text.count('call opaque(b,n)') == 1
    assert text.count(owner['owner'] + ' = 0') >= 2
    assert not owner['boundaries']
    assert 'integer(' not in text[text.index('contains\nsubroutine manage()'):].split('integer,save::times=0')[0]


def test_internal_formals_retain_a_native_boundary_at_first_payload_work(tmp_path):
    source = INTERNAL_SOURCE.replace('call manage()', 'call manage(n)').replace(
        'subroutine manage()', 'subroutine manage(m)\ninteger,intent(in)::m')
    _, _, report = emit(tmp_path, source)
    owner, = report['scopes']
    assert not owner['boundaries']
    child, = owner['internal_coordinators']
    assert not child['planning_segments']
    assert child['boundaries'][0]['reason'].startswith('reached original native operation')


def test_alternate_entry_cannot_jump_past_owner_control_initialization(tmp_path):
    source = SOURCE.replace('visits=visits+1', 'entry alternate(a,b,out,n,escape)\nvisits=visits+1')
    _, _, report = emit(tmp_path, source)
    assert all('native_continuation' not in item for item in report['scopes'])
    assert any('ordinary impure subroutine entry' in item['reason'] for item in report['boundaries'])


def test_single_line_unknown_call_closes_only_on_its_original_guard(tmp_path):
    source = SOURCE.replace('if(escape) then\ncall opaque(b,n)\nendif', 'if(escape) call opaque(b,n)')
    _, outputs, report = emit(tmp_path, source)
    boundary, = report['scopes'][0]['boundaries']
    assert boundary['conditional_close'].lower() == 'escape'
    text = next(text for name,text in outputs.items() if name.startswith('sources/'))
    condition = text.lower().index('if (escape) then')
    close = text.lower().index('fort_scope_close(', condition)
    native = text.lower().index('call opaque(b, n)', close)
    assert condition < close < native


def test_association_keeps_individual_fallbacks_and_original_prefix(tmp_path):
    _, outputs, report = emit(tmp_path, ASSOCIATE_SOURCE)
    owner, = report['scopes']
    assert len(owner['planning_segments']) == 2
    assert len(owner['boundaries']) == 1
    text = next(text for name, text in outputs.items() if name.startswith('sources/'))
    assert text.count('associate(count=>visits)') == 1
    assert text.count('count=count+1') == 1
    assert 'fort_scope_close(' in text[text.index('if(escape) then'):text.index('call opaque(b,n)')]


def test_single_line_supported_child_keeps_guard_before_descriptor_queries(tmp_path):
    _, outputs, report = emit(tmp_path, GUARDED_CHILD_SOURCE)
    assert not report['scopes'][0]['boundaries'], report['boundaries']
    assert any(item['procedure'] == 'local_owner::adjust' for item in report['borrowed_source_coordinators'])
    text = next(text for name, text in outputs.items() if name.startswith('sources/'))
    assert text.lower().count('if (escape) then') == 1
    assert 'if(escape) call adjust' not in text.lower()
    guard = text.lower().index('if (escape) then')
    query = text.lower().index('is_contiguous(b)', guard)
    call = text.lower().index('call fort_scope_coordinator_', query)
    assert guard < query < call


def test_protected_read_only_capture_uses_original_object_without_pointer_assignment(tmp_path):
    _, outputs, report = emit(tmp_path, PROTECTED_SOURCE, captures={'readonly_controls::coefficients': FACT})
    assert len(report['scopes'][0]['gpu_leaves']) == 2
    text = next(text for name, text in outputs.items() if name.startswith('sources/'))
    assert '=> coefficients' not in text.lower()
    assert 'c_loc(coefficients)' in text.lower()
    assert 'PROTECTED, TARGET :: coefficients' in text


def test_original_module_child_closes_only_at_reached_unknown_operation(tmp_path):
    path, outputs, report = emit(tmp_path, MODULE_SOURCE)
    owner, = report['scopes']
    child, = owner['module_coordinators']
    assert child['procedure'] == 'stage_child::adjust'
    assert child['resource_mappings'] == {'argument::x': 'argument::b'}
    assert len(child['calls']) == 2
    assert len(child['boundaries']) == 1
    assert not owner['boundaries']
    assert len(owner['gpu_leaves']) == 3
    text = outputs[report['sources'][str(path)]['replacement']]
    assert text.count('entry ' + child['entry']) == 1
    assert text.count('child_visits=child_visits+1') == 1
    assert text.count('call opaque(x,n)') == 1


def test_array_condition_publishes_before_original_header(tmp_path):
    path, outputs, report = emit(tmp_path, PAYLOAD_SOURCE)
    owner, = report['scopes']
    assert len(owner['planning_segments']) == 3
    assert len(owner['boundaries']) == 1
    text = outputs[report['sources'][str(path)]['replacement']]
    assert text.count('if(b(-2)>10.d0) then') == 1
    position = text.index('if(b(-2)>10.d0) then')
    assert 'fort_scope_host_begin' in text[:position]
    assert 'fort_scope_close(' in text[position:text.index('call opaque(b,n)')]


def test_array_elseif_stays_native_without_hoisting_payload_read(tmp_path):
    source = SOURCE.replace('call opaque(b,n)\nendif',
                            'call opaque(b,n)\nelse if(b(-2)>10.d0) then\ncall opaque(b,n)\nendif')
    path, outputs, report = emit(tmp_path, source)
    assert len(report['scopes'][0]['planning_segments']) == 2
    assert 'ELSEIF' in report['scopes'][0]['boundaries'][0]['reason']
    text = outputs[report['sources'][str(path)]['replacement']]
    assert 'if(escape) then\ncall opaque(b,n)\nelse if(b(-2)>10.d0) then' in text


def test_module_companion_different_mapping_has_explicit_native_boundary(tmp_path):
    source = MODULE_SOURCE.replace('call adjust(b,n,.false.)', 'call adjust(out,n,.false.)')
    _, _, report = emit(tmp_path, source)
    owner, = report['scopes']
    assert len(owner['module_coordinators']) == 1
    assert len(owner['module_coordinators'][0]['calls']) == 1
    assert len(owner['boundaries']) == 1


def test_module_companion_out_first_registration_cannot_reuse_caller_freshness(tmp_path):
    source = MODULE_SOURCE.replace('intent(inout)::x(-4:)', 'intent(out)::x(-4:)').replace(
        'x(j)=x(j)+real(j,8)', 'x(j)=real(j,8)')
    path, outputs, report = emit(tmp_path, source)
    text = outputs[report['sources'][str(path)]['replacement']]
    child_text = text[:text.index('module local_owner')]
    assert 'fort_scope_forget_definition' in child_text
    assert 'fort_layout, 0_c_int, fort_buffer_' in child_text
    assert 'fort_layout, 1_c_int, fort_buffer_' not in child_text


def test_module_companion_rejects_unproved_explicit_shape_storage_mapping(tmp_path):
    source = MODULE_SOURCE.replace('intent(inout)::x(-4:)', 'intent(inout)::x(-4:n-5)')
    _, _, report = emit(tmp_path, source)
    owner, = report['scopes']
    assert not owner['module_coordinators']
    assert len(owner['boundaries']) == 2


def test_nested_module_companion_forwards_hidden_resource_handles(tmp_path):
    path, outputs, report = emit(tmp_path, NESTED_MODULE_SOURCE, captures={'deep_child::weights': FACT})
    owner, = report['scopes']
    assert not owner['boundaries']
    assert {item['procedure'] for item in owner['module_coordinators']} == {
        'deep_child::leaf', 'stage_child::adjust'}
    assert 'deep_child::weights' in owner['ownership']['retained_resources']
    child = next(item for item in owner['module_coordinators'] if item['procedure'] == 'deep_child::leaf')
    text = outputs[report['sources'][str(path)]['replacement']]
    assert 'use deep_child, only: ' + child['entry'] in text


def test_module_companion_checks_original_actual_before_contiguous_association(tmp_path):
    path, outputs, report = emit(tmp_path, STRIDED_MODULE_SOURCE)
    owner, = report['scopes']
    child, = owner['module_coordinators']
    text = outputs[report['sources'][str(path)]['replacement']]
    caller = text[text.index('subroutine step'):]
    guard = caller.index('fort_actuals_contiguous = is_contiguous(b)')
    close = caller.index('fort_scope_close(', guard)
    call = caller.index('call '+child['entry'], close)
    assert guard < close < call


def test_borrowed_child_return_retains_outer_owner(tmp_path):
    path, outputs, report = emit(tmp_path, RETURN_MODULE_SOURCE)
    owner, = report['scopes']
    child, = owner['module_coordinators']
    assert not child['boundaries']
    assert not owner['boundaries']
    text = outputs[report['sources'][str(path)]['replacement']]
    assert 'if (.not.escape) return' in text


def test_original_allocation_guard_requires_no_payload_capture(tmp_path):
    path, outputs, report = emit(tmp_path, ALLOCATION_GUARD_SOURCE)
    owner, = report['scopes']
    assert len(owner['boundaries']) == 1
    assert 'opaque' in owner['boundaries'][0]['reason']
    assert 'local_owner::spare' not in owner['ownership']['retained_resources']
    text = outputs[report['sources'][str(path)]['replacement']]
    guard = text.index('if(allocated(spare)) then')
    assert 'fort_scope_close(' in text[guard:text.index('call opaque(b,n)')]
    assert 'c_loc(spare)' not in text


def test_complete_native_team_composes_bounded_units_without_flattening(tmp_path):
    path, outputs, report = emit(tmp_path, NATIVE_ATOMIC_SOURCE)
    owner, = report['scopes']
    assert not owner['boundaries'], owner['boundaries']
    assert len(owner['gpu_leaves']) == 2
    operations = [op for segment in owner['planning_segments']
                  for op in segment['operations']['native_operations'] if op.get('atomic_effects')]
    operation, = operations
    proof = operation['atomic_effects']
    assert len(proof['source_units']) == 24
    assert max(proof['unit_operation_counts']) <= 256
    assert sum(proof['unit_operation_counts']) > 256
    assert proof['effect_count'] <= 256
    text = outputs[report['sources'][str(path)]['replacement']]
    # The coherent path and the disabled-owner fallback each keep one whole
    # original team. Runtime agreement also checks they never both execute.
    assert text.count('!$omp parallel private(i)') == 2
    assert text.count('!$omp end parallel') == 2


def test_guarded_native_child_reuses_original_formals_and_callback(tmp_path):
    path, outputs, report = emit(tmp_path, GUARDED_NATIVE_SOURCE)
    owner, = report['scopes']
    assert not owner['boundaries']
    child, = owner['internal_coordinators']
    assert len(child['boundaries']) == 1
    assert not child['planning_segments']
    assert len(owner['internal_calls']) == 2
    assert all(not call['effect_summary_available'] for call in owner['internal_calls'])
    text = outputs[report['sources'][str(path)]['replacement']]
    guard = text.index('if(allocated(items)) then')
    work = text.index('do j=1,size(items)')
    assert guard < work < text.index('fort_scope_close(', work) < text.index('x(items(j)%coordinate)=')
    assert text.count('subroutine guarded(x,items,fun)') == 1
    assert 'c_loc(metadata)' not in text
    assert report['implementation_variants']['generated_count'] <= 3


def test_guarded_native_loop_payload_bounds_publish_before_header(tmp_path):
    source = GUARDED_NATIVE_SOURCE.replace('do j=1,size(items)', 'do j=1,min(size(items),int(x(1)))')
    path, outputs, report = emit(tmp_path, source)
    text = outputs[report['sources'][str(path)]['replacement']]
    guard = text.index('if(allocated(items)) then')
    assert guard < text.index('fort_scope_close(', guard) < text.index('do j=1,min(size(items),int(x(1)))')
    child, = report['scopes'][0]['internal_coordinators']
    assert 'scalar/descriptor-only' in child['boundaries'][0]['reason']


def test_native_callback_liveness_does_not_require_a_numerical_interface(tmp_path):
    _, _, report = emit(tmp_path/'independent', GUARDED_NATIVE_SOURCE)
    assert len(report['scopes'][0]['gpu_leaves']) == 2
    # A callback observing the original iterator keeps the producer native;
    # an indirect use must not disappear with the procedure dummy declaration.
    source = GUARDED_NATIVE_SOURCE.replace('transform=x+2', 'transform=x+real(i,8)')
    _, _, report = emit(tmp_path/'capture', source)
    assert len(report['scopes'][0]['gpu_leaves']) == 1
    assert any('live after' in row['reason'] for row in report['boundaries'])

    source = GUARDED_NATIVE_SOURCE.replace('step(a,b,out,n,escape)', 'step(a,b,out,n,escape,callback)').replace(
        'integer::i\nvisits', 'procedure(transform)::callback\ninteger::i\nvisits').replace(
        'call guarded(b,metadata,transform)', 'call guarded(b,metadata,callback)')
    _, _, report = emit(tmp_path/'dynamic', source)
    assert len(report['scopes'][0]['gpu_leaves']) == 1
    assert any('live after' in row['reason'] for row in report['boundaries'])


def test_native_guard_rejects_associations_that_may_copy_or_define_before_body(tmp_path):
    for index, declaration in enumerate(('real(8),contiguous,intent(inout)::x(:)',
                                         'real(8),intent(out)::x(:)', 'real(8),intent(inout)::x(8)',
                                         'real(8),intent(inout)::x(size(b):)')):
        _, _, report = emit(tmp_path/str(index), GUARDED_NATIVE_SOURCE.replace(
            'real(8),intent(inout)::x(:)', declaration))
        assert len(report['scopes'][0]['boundaries']) == 2
        assert not report['scopes'][0]['internal_coordinators']


def test_owner_prologue_precedes_publication_for_first_original_condition(tmp_path):
    for index, source in enumerate((FIRST_GUARD_SOURCE, FIRST_GUARD_MODULE_SOURCE)):
        path, outputs, report = emit(tmp_path/str(index), source)
        text = outputs[report['sources'][str(path)]['replacement']]
        owner, = report['scopes']
        if index:
            child, = owner['module_coordinators']
            assert text.index('entry '+child['entry']) < text.index('fort_reached_0: block')
        else:
            assert text.index(owner['owner']+' = 0') < text.index('fort_reached_0: block')


def test_native_metadata_has_an_opaque_alias_reservation_and_no_device_worker(tmp_path):
    path, outputs, report = emit(tmp_path, NATIVE_METADATA_SOURCE)
    owner, = report['scopes']
    assert report['reached_plan_schema_version'] == 1
    registrations = owner['resource_bindings']
    assert registrations['schema_version'] == 1
    identities = {item['resource']: item['registration_identity'] for item in registrations['resources']}
    assert len(set(identities.values())) == len(identities)
    assert identities['local_owner::metadata'] > 0
    assert not owner['boundaries'], owner['boundaries']
    assert len(owner['gpu_leaves']) == 2
    native = [op for segment in owner['planning_segments']
              for op in segment['operations']['native_operations'] if op['host_metadata']]
    operation, = native
    assert not operation['host_metadata']['local_owner::metadata']['device_capture']
    assert not operation['sections']['available']  # Indexed correction stays conservative and native.
    text = outputs[report['sources'][str(path)]['replacement']]
    assert 'storage_size(metadata, kind=c_size_t)' in text
    assert 'FORT_SCOPE_BYTES' in text
    assert 'c_loc(metadata)' in text
    assert 'b(metadata(i)%coordinate)' in text  # Original fallback still exists.


def test_source_branch_targets_cannot_enter_generated_owner_blocks(tmp_path):
    source = SOURCE.replace('visits=visits+1', 'if(escape) goto 100\nvisits=visits+1').replace(
        'do i=-2,n-3', '100 do i=-2,n-3', 1)
    original, outputs, report = emit(tmp_path, source)
    assert report['scope_count'] == 0
    assert not report['source_edits']
    assert any('source statement labels remain native' in boundary['reason'] for boundary in report['boundaries'])
    assert original.read_text() == source


def test_native_metadata_rejects_pointer_fields_and_array_components(tmp_path):
    for index, declaration in enumerate(('real(8),pointer::weight', 'real(8)::weight(1)')):
        _, _, report = emit(tmp_path/str(index), NATIVE_METADATA_SOURCE.replace('real(8)::weight', declaration))
        assert report['scopes'][0]['boundaries']


def test_native_metadata_index_reads_preserve_scalar_liveness(tmp_path):
    source = NATIVE_METADATA_SOURCE.replace('do i=1,size(metadata)\n'
        'b(metadata(i)%coordinate)=b(metadata(i)%coordinate)+metadata(i)%weight\nenddo',
        'b(-2)=b(-2)+metadata(i)%weight')
    _, _, report = emit(tmp_path, source)
    assert len(report['scopes'][0]['gpu_leaves']) == 1
    assert any('scalar is live after' in item['reason'] for item in report['boundaries'])


def test_lexical_child_and_forwarded_callback_keep_original_iterator_live(tmp_path):
    for index, call in enumerate(('call observe()', 'call opaque_callback(observe)')):
        source = SOURCE.replace('if(escape) then\ncall opaque(b,n)\nendif', call).replace(
            'end subroutine\nend module', '''contains
subroutine observe()
visits=visits+i
end subroutine
end subroutine
end module''')
        _, _, report = emit(tmp_path/str(index), source)
        assert any('scalar is live after' in item['reason'] for item in report['boundaries'])
        regions = report['inline_numerical_regions']['regions']
        assert all(region['written_resources'] != ['argument::b'] for region in regions)
