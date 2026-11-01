"""Native scalar record arguments prove no payload layout or initialization."""
import pytest

from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError
from compiler.tests.test_descriptor_metadata_arrays import fixture, prove


def setup(tmp_path, *, formal='real(8),intent(inout)::value', body='value=value+1.0_8',
          field='real(8)::weight', action='call scalar(marks(i)%weight)', extra=''):
    path, _, selection = fixture(tmp_path, field=field, action=action)
    source = path.read_text().replace('subroutine update(a)',
        'subroutine scalar(value)\n' + formal + '\n' + extra + '\n' + body +
        '\nend subroutine\nsubroutine update(a)', 1)
    path.write_text(source)
    analysis = SourceEffects([path])
    return path, analysis, selection


def test_original_scalar_component_call_retains_complete_checked_closure(tmp_path):
    path, analysis, selection = setup(tmp_path)
    before = path.read_bytes()
    public = prove(analysis, selection).public()
    call, = public['native_metadata_call_descriptor_requirements']
    assert call['association']['procedure'] == 'metadata_steps::scalar'
    assert call['scalar_fields'] == [['weight']]
    assert public['source_call_closure']['metadata_steps::scalar']
    assert call['whole_original_closure_required']
    assert not any(call[key] for key in ('capture_authorized', 'payload_access_authorized',
                                       'entry_definition_authority', 'execution_authorized'))
    assert not public['entry_definition_authority']
    assert path.read_bytes() == before


@pytest.mark.parametrize(('formal', 'body', 'field', 'extra'), [
    ('real(8),pointer::value', 'value=1.0_8', 'real(8)::weight', ''),
    ('real(8),target::value', 'value=1.0_8', 'real(8)::weight', ''),
    ('real(8),allocatable::value', 'allocate(value)', 'real(8)::weight', ''),
    ('real(8),optional::value', 'value=1.0_8', 'real(8)::weight', ''),
    ('real(8)::value', 'call absent(value)', 'real(8)::weight', ''),
    ('real(8)::value', 'value=1.0_8', 'real(8),pointer::weight', ''),
    ('real(8)::value', 'value=1.0_8', 'real(8),allocatable::weight', ''),
    ('real(8)::value', 'value=1.0_8', 'real(8)::weight(3)', ''),
    ('real(8)::value', 'deallocate(scratch)', 'real(8)::weight', 'real(8),allocatable::scratch(:)'),
    ('real(8)::value', 'value=1.0_8', 'real(8)::weight', 'common /unsafe/ value'),
])
def test_descriptor_call_does_not_hide_alias_or_lifetime_effects(tmp_path, formal, body, field, extra):
    _, analysis, selection = setup(tmp_path, formal=formal, body=body, field=field, extra=extra)
    with pytest.raises(CompilationError):
        prove(analysis, selection)
    assert not getattr(analysis, '_descriptor_window_proofs', {})
