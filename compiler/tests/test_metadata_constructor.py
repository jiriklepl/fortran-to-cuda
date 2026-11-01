"""Once-only metadata construction cannot bypass hidden host effects."""
from copy import copy

import pytest

from compiler.frontend.metadata_constructor import original_metadata_constructor
from compiler.ir import CompilationError
from compiler.tests.test_opaque_library_state import fixture, issue

SOURCE = """module opaque_fixture
implicit none
type workspace
real(8),allocatable::payload(:)
end type
type state_type
integer::shape(3)=0
real(8)::lengths(3)=0
logical::configured=.false.
integer::count=0
type(workspace),allocatable::cpu
end type
contains
pure function make_state(shape,lengths) result(value)
integer,intent(in)::shape(3)
real(8),intent(in)::lengths(3)
type(state_type)::value
value%shape=shape
value%lengths=lengths
value%configured=.true.
end function
subroutine operation(d,a)
type(state_type),intent(inout)::d
real(8),intent(inout)::a(:)
if(.not.allocated(d%cpu))allocate(d%cpu)
a=a+1.0_8
end subroutine
subroutine advance(a)
real(8),intent(inout)::a(:)
type(state_type),save::cache
logical,save::called=.false.
integer::n=3
real(8)::length=1.0_8
if(.not.called)then
cache=make_state([n,n,n],[length,length+1.0_8,length*2.0_8])
called=.true.
endif
call operation(cache,a)
end subroutine
end module
"""


def proof(tmp_path, source=SOURCE):
    parts = fixture(tmp_path, source)
    _, analysis, routine, _, _, assignment, _, _ = parts
    return parts, original_metadata_constructor(analysis, routine, assignment, issue(parts))


def test_once_only_fixed_metadata_preserves_original_guard_and_no_authority(tmp_path):
    parts, issued = proof(tmp_path)
    _, analysis, routine, _, _, assignment, _, _ = parts
    assert issued.validate(analysis, routine, assignment) is issued
    assert original_metadata_constructor(analysis, routine, assignment, issue(parts)) is issued
    public = issued.public()
    assert public['result_fields'] == ['shape', 'lengths', 'configured']
    assert {x['resource'] for x in public['metadata_reads']} == {
        'opaque_fixture::advance::n', 'opaque_fixture::advance::length'}
    assert all(public[key] is False for key in public if key.endswith('_authorized'))
    public['definitions_authorized'] = True
    assert issued.public()['definitions_authorized'] is False
    with pytest.raises(CompilationError, match='exact compiler-issued'):
        copy(issued).validate(analysis, routine, assignment)
    with pytest.raises(CompilationError, match='exact compiler-issued'):
        issued.validate(analysis, routine, copy(assignment))
    del analysis._metadata_constructor_proofs[issued.identity]
    with pytest.raises(CompilationError, match='exact compiler-issued'):
        issued.validate(analysis, routine, assignment)


@pytest.mark.parametrize(('old', 'new'), [
    ('pure function make_state', 'function make_state'),
    ('value%configured=.true.', 'call hidden()\nvalue%configured=.true.'),
    ('value%configured=.true.', 'allocate(value%cpu)\nvalue%configured=.true.'),
    ('value%configured=.true.', 'value%cpu=workspace()\nvalue%configured=.true.'),
    ('integer::count=0', 'integer::count=hidden()'),
    ('integer::count=0', 'integer::count'),
    ('type(workspace),allocatable::cpu', 'type(workspace),pointer::cpu'),
    ('logical,save::called=.false.', 'logical,save::called=.true.'),
    ('called=.true.\nendif', 'called=.false.\nendif'),
    ('called=.true.\nendif', 'called=.true.\nelse\nprint *,1\nendif'),
    ('call operation(cache,a)', 'called=.false.\ncall operation(cache,a)'),
    ('if(.not.called)then', 'call hidden()\nif(.not.called)then'),
    ('cache=make_state', 'print *,1\ncache=make_state'),
    ('[n,n,n]', '[n,n]'),
    ('[n,n,n]', '[n,n,3_8]'),
    ('[n,n,n]', '[(n,n=1,3)]'),
    ('[length,length+1.0_8,length*2.0_8]', '[length,1.0,length*2.0_8]'),
    ('value%shape=shape', 'value%shape=shape\nvalue%shape=shape'),
    ('value%lengths=lengths', 'value%lengths=1.0_8'),
])
def test_hidden_effects_repetition_and_unproved_actuals_reject(tmp_path, old, new):
    with pytest.raises(CompilationError):
        proof(tmp_path, SOURCE.replace(old, new))



def test_promoted_scalar_expression_keeps_fixed_real8_metadata_and_no_authority(tmp_path):
    source = SOURCE.replace('[length,length+1.0_8,length*2.0_8]',
                            '[length,length+1.0,length*2.0_8]')
    parts, issued = proof(tmp_path, source)
    path, analysis, routine, _, _, assignment, _, _ = parts
    assert issued.validate(analysis, routine, assignment) is issued
    public = issued.public()
    lengths = next(item for item in public['constructor']['associations'] if item['formal'] == 'lengths')
    assert lengths['actual_type'] == {'type': 'real', 'kind': 8, 'rank': 1,
                                     'storage_resource': None, 'declaring_type': None}
    assert lengths['actual'].replace(' ', '').lower() == '[length,length+1.0,length*2.0_8]'
    assert public['constructor']['actual_evaluation'] == 'unchanged original call position and guards'
    assert public['result_fields'] == ['shape', 'lengths', 'configured']
    assert {item['resource'] for item in public['metadata_reads']} == {
        'opaque_fixture::advance::n', 'opaque_fixture::advance::length'}
    assert all(not value for name, value in public.items() if name.endswith('_authorized'))
    assert path.read_text() == source


def test_mutating_guard_ast_after_issuance_invalidates_proof(tmp_path):
    parts, issued = proof(tmp_path)
    _, analysis, routine, _, _, assignment, _, _ = parts
    from fparser.two.utils import walk
    guard = next(node for node in walk(routine.execution) if type(node).__name__ == 'If_Then_Stmt')
    guard.items = (guard.items[0], 'forged')
    with pytest.raises(CompilationError):
        issued.validate(analysis, routine, assignment)


def test_fixed_vector_actual_records_exact_original_read_extent(tmp_path):
    source = SOURCE.replace('real(8)::length=1.0_8', 'real(8)::length(3)=1.0_8')
    source = source.replace('[length,length+1.0_8,length*2.0_8]', 'length')
    _, issued = proof(tmp_path, source)
    reads = issued.public()['metadata_reads']
    assert next(item for item in reads if item['resource'].endswith('::length'))['bounds'] == [[1, 3]]


def test_visible_defined_assignment_rejects_even_pure_numeric_constructor(tmp_path):
    source = SOURCE.replace('contains\npure function', 'interface assignment(=)\n'
                            'module procedure assign_state\nend interface\ncontains\n'
                            'subroutine assign_state(a,b)\ntype(state_type),intent(out)::a\n'
                            'type(state_type),intent(in)::b\nend subroutine\npure function', 1)
    with pytest.raises(CompilationError, match='intrinsic assignment'):
        proof(tmp_path, source)


def test_namelist_group_cannot_hide_a_reset_of_the_once_only_flag(tmp_path):
    source = SOURCE.replace('real(8)::length=1.0_8', 'real(8)::length=1.0_8\nnamelist /reset_state/ called')
    source = source.replace('call operation(cache,a)', 'call operation(cache,a)\nread(*,nml=reset_state)')
    with pytest.raises(CompilationError, match='NAMELIST'):
        proof(tmp_path, source)
