"""Entry obligations follow original definitions, branches and child mappings."""

import pytest

from compiler.tests.test_source_scopes import FACT, generate


SOURCE = '''module ordered_storage
implicit none
contains
subroutine fill(b)
real(8),intent(out)::b(-3:)
integer::i
do i=lbound(b,1),ubound(b,1)
b(i)=2
enddo
end subroutine
subroutine touch(b,indices)
real(8),intent(inout)::b(-3:)
integer,intent(in)::indices(:)
integer::i
do i=1,size(indices)
b(indices(i))=b(indices(i))+1
enddo
end subroutine
subroutine child(b,indices,flag)
real(8),intent(out)::b(-3:)
integer,intent(in)::indices(:)
logical,intent(in)::flag
call fill(b)
call touch(b,indices)
end subroutine
subroutine step(a,b,out,n)
real(8),intent(in)::a(:)
real(8),intent(inout)::b(-3:),out(-3:)
integer,intent(in)::n(:)
call child(b,n,.true.)
out=b+a
end subroutine
end module
'''


def build(directory, text=SOURCE):
    return generate(directory, text, scope_execution='reached', facts={
        'schema_version': 1, 'participation': 'serial', 'captures': {
            'argument::a': FACT, 'argument::n': FACT,
            'argument::b': {**FACT, 'initialized': 'none'},
            'argument::out': {**FACT, 'initialized': 'none'}}})


def child(manifest):
    return next(item for item in manifest['borrowed_source_coordinators']
                if item['procedure'] == 'ordered_storage::child')


def test_full_child_definition_satisfies_later_conservative_access(tmp_path):
    original, output, manifest = build(tmp_path)
    assert not manifest['scopes'][0]['boundaries'], manifest['boundaries']
    proof = child(manifest)['ordered_definitions']
    assert proof['required_whole'] == ['argument::indices']
    assert 'argument::b' in proof['whole_on_return']
    assert 'argument::b' in proof['possible_definition_changes']
    generated = (output/manifest['sources'][str(original)]['replacement']).read_text()
    assert 'fort_scope_plan_validate(fort_context)' in generated


def test_every_branch_can_restore_whole_storage(tmp_path):
    source = SOURCE.replace('call fill(b)\ncall touch',
                            'if(flag) then\ncall fill(b)\nelse\ncall fill(b)\nendif\ncall touch')
    _, _, manifest = build(tmp_path, source)
    assert not manifest['scopes'][0]['boundaries'], manifest['boundaries']
    assert child(manifest)['ordered_definitions']['required_whole'] == ['argument::indices']


@pytest.mark.parametrize('prefix', [
    'if(flag) then\ncall fill(b)\nendif',
    'call fill(b)\ncall partial(b)',
])
def test_missing_branch_or_later_out_event_does_not_reuse_old_definition(tmp_path, prefix):
    source = SOURCE.replace('subroutine touch', '''subroutine partial(b)
real(8),intent(out)::b(-3:)
b(-3)=7
end subroutine
subroutine touch''', 1).replace('call fill(b)\ncall touch', prefix+'\ncall touch')
    _, _, manifest = build(tmp_path, source)
    assert not any(item['procedure'] == 'ordered_storage::child'
                   for item in manifest['borrowed_source_coordinators'])
    assert any('incomplete definition' in item['reason']
               for item in manifest['source_coordinator_boundaries']
               if item['procedure'] == 'ordered_storage::child')


def test_prior_lexical_initialization_uses_reached_coverage_not_entry_capture(tmp_path):
    source = SOURCE.replace('intent(out)::b(-3:)\ninteger,intent(in)::indices',
                            'intent(inout)::b(-3:)\ninteger,intent(in)::indices')
    source = source.replace('call fill(b)\ncall touch', 'call touch')
    # A numerical operation keeps the child eligible while the native indirect
    # operation still requires whole storage on entry.
    source = source.replace('call touch(b,indices)\nend subroutine', 'call touch(b,indices)\nb=b*2\nend subroutine')
    source = source.replace('call child(b,n,.true.)', 'call fill(b)\ncall child(b,n,.true.)')
    _, _, manifest = build(tmp_path, source)
    assert not manifest['scopes'][0]['boundaries'], manifest['boundaries']
    assert child(manifest)['ordered_definitions']['required_whole'] == ['argument::b', 'argument::indices']
