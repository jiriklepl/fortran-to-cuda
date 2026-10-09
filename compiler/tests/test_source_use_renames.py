"""Unrestricted USE renames preserve the complete source export namespace."""

from compiler.frontend.source_effects import SourceEffects


LIBRARY = """module storage_owner
implicit none
private
integer,parameter,public::precision=8
real(8),public::field(8),factor=2
type,public::record
real(8)::values(8)
end type
public::apply
interface apply
module procedure worker
end interface
contains
subroutine worker(a)
real(8),intent(inout)::a(:)
a=a*factor+field
end subroutine
end module
"""


def analysis(tmp_path, client):
    left, right = tmp_path/'library.f90', tmp_path/'client.f90'
    left.write_text(LIBRARY)
    right.write_text(client)
    return SourceEffects([left, right])


def test_renamed_generic_kind_and_field_reexport_through_source_module(tmp_path):
    model = analysis(tmp_path, """module facade
use storage_owner, run=>apply, payload=>field, rk=>precision, box=>record
implicit none
private
public::run,payload,rk,factor,box
end module
module client
use facade
contains
subroutine step(a)
real(rk),intent(inout)::a(:)
type(box)::item
item%values=payload
call run(a)
a=a+factor+item%values
end subroutine
end module
""")
    routine = model.routines['client::step']
    assert routine.scope.bindings['a'].kind == 8
    assert model._binding(routine.scope, 'payload').root == 'storage_owner::field'
    assert model._binding(routine.scope, 'field') is None
    assert model._binding(routine.scope, 'factor').root == 'storage_owner::factor'
    assert model._candidates(routine.scope, 'run') == ['storage_owner::worker']
    assert model._candidates(routine.scope, 'worker') == []
    summary = model.summarize('client::step')
    assert summary['complete'], summary['reasons']
    assert any(op.get('resource') == 'storage_owner::field' for op in summary['ordered_effects'])


def test_another_unrestricted_use_can_restore_the_original_name(tmp_path):
    model = analysis(tmp_path, """module client
use storage_owner, payload=>field
use storage_owner
contains
subroutine step(a)
real(8),intent(inout)::a(:)
a=field+payload
end subroutine
end module
""")
    scope = model.routines['client::step'].scope
    assert model._binding(scope, 'field').root == model._binding(scope, 'payload').root
    assert model.summarize('client::step')['complete']


def test_only_import_can_restore_a_renamed_original_without_restoring_others(tmp_path):
    model = analysis(tmp_path, """module client
use storage_owner, payload=>field, rk=>precision
use storage_owner,only:field
contains
subroutine step(a)
real(rk),intent(inout)::a(:)
a=field+payload
end subroutine
end module
""")
    scope = model.routines['client::step'].scope
    assert model._binding(scope, 'field').root == 'storage_owner::field'
    assert model._binding(scope, 'precision') is None
    assert model.summarize('client::step')['complete']


def test_renamed_use_does_not_override_a_conflicting_unrestricted_import(tmp_path):
    model = analysis(tmp_path, """module conflicting
real(8)::payload(8)
end module
module client
use storage_owner,payload=>field
use conflicting
contains
subroutine step(a)
real(8),intent(inout)::a(:)
a=payload
end subroutine
end module
""")
    scope = model.routines['client::step'].scope
    assert model._binding(scope, 'payload') is None
    assert not model.summarize('client::step')['complete']


def test_renamed_source_and_contract_cache_identity_changes(tmp_path):
    client = """module client
use storage_owner,payload=>field
contains
subroutine step(a)
real(8),intent(inout)::a(:)
a=payload
end subroutine
end module
"""
    first = analysis(tmp_path, client).summarize('client::step')
    second = analysis(tmp_path, client.replace('payload=>field', 'renamed=>field').replace(
        'a=payload', 'a=renamed')).summarize('client::step')
    assert first['complete'] and second['complete']
    assert first['summary_identity'] != second['summary_identity']
