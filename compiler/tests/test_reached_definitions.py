"""Entry obligations follow original definitions, branches and child mappings."""

import json

import pytest

from compiler._legacy.config import LegacyCompilerOptions, LegacyOffloadConfig
from compiler.ir import CompilationError
from compiler.scopes.procedures import BorrowedCoordinator
from compiler.scopes.source import ScopeBuilder
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


def legacy_child(original):
    """Retain the bounded coordinator's independent branch-meet proof oracle."""
    builder = ScopeBuilder([original], 'ordered_storage::step',
        facts=json.loads((original.parent/'captures.json').read_text()),
        options=LegacyCompilerOptions(),
        config=LegacyOffloadConfig(policy='sections', scope_execution='reached'))
    return BorrowedCoordinator(builder, 'ordered_storage::child', set())


def module_child(manifest):
    owner, = manifest['scopes']
    module, = [item for item in owner['module_coordinators']
               if item['procedure'] == 'ordered_storage::child']
    assert module['interface'] == 'original_module_entry_v1'
    assert module['resource_mappings'] == {'argument::b': 'argument::b', 'argument::indices': 'argument::n'}
    checked, = [item for item in owner['checked_execution_plan']['procedures']
                if item['procedure'] == module['procedure']]
    call, = checked['calls']
    assert call['caller'] == 'ordered_storage::step'
    assert all(actual == module['resource_mappings'][formal] for item in checked['units']
               for formal, actual in item['resources'].items())
    return checked


def test_full_child_definition_satisfies_later_conservative_access(tmp_path):
    original, output, manifest = build(tmp_path)
    assert not manifest['scopes'][0]['boundaries'], manifest['boundaries']
    proof = legacy_child(original).definition_summary.public()
    assert proof['required_whole'] == ['argument::indices']
    assert 'argument::b' in proof['whole_on_return']
    assert 'argument::b' in proof['possible_definition_changes']
    touch, = module_child(manifest)['units']
    assert touch['ordered_definitions']['required_whole'] == ['argument::b', 'argument::indices']
    generated = (output/manifest['sources'][str(original)]['replacement']).read_text()
    assert 'fort_scope_plan_validate(fort_context)' in generated


def test_every_branch_can_restore_whole_storage(tmp_path):
    source = SOURCE.replace('call fill(b)\ncall touch',
                            'if(flag) then\ncall fill(b)\nelse\ncall fill(b)\nendif\ncall touch')
    original, _, manifest = build(tmp_path, source)
    assert not manifest['scopes'][0]['boundaries'], manifest['boundaries']
    assert legacy_child(original).definition_summary.public()['required_whole'] == ['argument::indices']
    touch, = module_child(manifest)['units']
    assert touch['ordered_definitions']['required_whole'] == ['argument::b', 'argument::indices']


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
    original, _, manifest = build(tmp_path, source)
    assert not any(item['procedure'] == 'ordered_storage::child'
                   for item in manifest['borrowed_source_coordinators'])
    with pytest.raises(CompilationError, match='incomplete definition: argument::b'):
        legacy_child(original)
    units = module_child(manifest)['units']
    assert units[-1]['ordered_definitions']['required_whole'] == ['argument::b', 'argument::indices']
    if 'partial' in prefix:
        invalidation, touch = units
        assert invalidation['ordered_definitions']['possible_definition_changes'] == ['argument::b']
        assert invalidation['ordered_definitions']['whole_on_return'] == []
        assert invalidation['last_line'] < touch['first_line']


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
