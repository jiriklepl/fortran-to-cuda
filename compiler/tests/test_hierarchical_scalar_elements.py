"""Original scalar element associations survive checked hierarchical calls."""
from copy import copy, deepcopy
from dataclasses import replace

import pytest

from compiler.ir import CompilationError
from compiler.tests.test_hierarchical_child_regions import SOURCE, fixture


SOURCE_ELEMENTS = (SOURCE.replace('real(8)::observed',
    'real(8)::observed\nreal(8)::ratios(0:1)=[2.0_8,3.0_8]\ninteger::selector=0')
    .replace('subroutine consumer(a,b,n)', 'subroutine consumer(a,b,n,scale)')
    .replace('real(8),intent(in)::b(:)', 'real(8),intent(in)::b(:)\nreal(8),intent(in)::scale')
    .replace('a(i)=2.0_8*b(i)', 'a(i)=scale*b(i)')
    .replace('call consumer(left,right,n)', 'call consumer(left,right,n,ratios(selector))'))


def emit(tmp_path, source=SOURCE_ELEMENTS):
    path, builder = fixture(tmp_path, source)
    outputs, report = builder.run()
    assert report['scope_count'] == 1, report.get('hierarchical_region_attempt', report.get('graph_owner_attempt'))
    owner = builder._conditional_return_draft.validate().owner
    middle = owner.members['hierarchical::middle']
    calls = [call for call in middle.return_calls if call.callee.routine.qualified == 'hierarchical::consumer']
    assert calls and all(call.scalar_element_coordinates is not None for call in calls)
    return path, builder, middle, calls, outputs, report


def test_original_table_argument_uses_registered_coordinates_and_existing_reservation(tmp_path):
    path, builder, middle, calls, outputs, report = emit(tmp_path)
    call, = calls
    assert call.validate(middle) is call
    assert call.scalar_element_coordinates.scalar_elements[0].binding.root == 'hierarchical::ratios'
    assert call.scalar_element_coordinates.public()['scalar_element_closure']['no_other_backing_reference']
    generated = outputs[report['sources'][str(path)]['replacement']]
    assert 'ratios(selector)' in generated
    assert 'fort_element_extents_' in generated
    assert 'hierarchical::ratios' in middle.owner.scalar_element_resources
    site, = call.callee.call_sites
    assert site['coordinate_proof_identity'] == call.scalar_element_coordinates.identity
    assert site['original_scalar_element_coordinates'] == call.scalar_element_coordinates.public()
    assert site['mapped_call_node_id'] is None
    assert builder.finish()


def test_unchanged_strict_reader_accepts_real_element_call_package(tmp_path):
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans
    _path, _builder, _middle, _calls, _outputs, report = emit(tmp_path)
    original = deepcopy(report)
    validate_checked_execution_plans(report)
    assert report == original


@pytest.mark.parametrize('bad', ['missing', 'copy', 'public', 'foreign', 'registry_removed'])
def test_coordinates_keep_exact_original_registered_ownership(tmp_path, bad):
    _path, builder, middle, calls, _outputs, _report = emit(tmp_path)
    call, = calls
    proof = call.scalar_element_coordinates
    if bad == 'registry_removed':
        builder.analysis._native_definition_proofs.pop(proof.identity)
        altered = call
    else:
        if bad == 'foreign':
            other = tmp_path / 'other'
            other.mkdir()
            value = emit(other)[3][0].scalar_element_coordinates
        else:
            value = {'missing': None, 'copy': copy(proof), 'public': proof.public()}[bad]
        altered = replace(call, scalar_element_coordinates=value)
    with pytest.raises(CompilationError):
        altered.validate(middle)


@pytest.mark.parametrize('old,new', [('ratios(selector)', 'ratios(1)'), ('[2.0_8,3.0_8]', '[2.0_8,4.0_8]'),
                                   ('ratios(0:1)', 'ratios(-1:0)')])
def test_changed_original_association_or_declaration_cannot_publish(tmp_path, old, new):
    path, builder, _middle, _calls, _outputs, _report = emit(tmp_path)
    assert old in path.read_text()
    path.write_text(path.read_text().replace(old, new))
    with pytest.raises(CompilationError, match='changed'):
        builder.finish()


def test_selector_write_inside_child_keeps_original_element_association(tmp_path):
    source = SOURCE_ELEMENTS.replace(
        'do i=1,n\na(i)=scale*b(i)', 'selector=1-selector\ndo i=1,n\na(i)=scale*b(i)')
    path, builder, _middle, calls, outputs, report = emit(tmp_path, source)
    assert calls[0].scalar_element_coordinates.public()['scalar_element_closure']['no_other_backing_reference']
    generated = outputs[report['sources'][str(path)]['replacement']]
    assert 'selector=1-selector' in generated
    assert 'ratios(selector)' in generated
    assert builder.finish()


@pytest.mark.parametrize('source', [
    SOURCE_ELEMENTS.replace('a(i)=scale*b(i)', 'ratios(0)=4.0_8\na(i)=scale*b(i)'),
    SOURCE_ELEMENTS.replace('intent(in)::scale', 'intent(inout)::scale').replace('a(i)=scale*b(i)', 'scale=4.0_8\na(i)=scale*b(i)'),
    SOURCE_ELEMENTS.replace('real(8)::ratios', 'real(8),target::ratios'),
], ids=['backing_write', 'dummy_write', 'target_storage'])
def test_backing_or_dummy_mutation_and_unproved_alias_storage_reject(tmp_path, source):
    _path, builder = fixture(tmp_path, source)
    _outputs, report = builder.run()
    # Other proved work may remain offloaded. The unsafe association cannot
    # become a checked child or acquire coordinate/alias-reservation authority.
    calls = [route for scope in report['scopes']
             for member in scope['checked_execution_plan']['storage_epoch'].get('return_contract', {}).get('members', [])
             for route in member['routes'] if route['kind'] == 'checked_child']
    assert not any(call['procedure'] == 'hierarchical::consumer' for call in calls)


def test_repeated_original_calls_keep_distinct_selector_and_coordinate_receipts(tmp_path):
    source = SOURCE_ELEMENTS.replace('call consumer(left,right,n,ratios(selector))',
        'call consumer(left,right,n,ratios(selector))\nselector=1-selector\ncall consumer(left,right,n,ratios(selector))')
    _path, builder, middle, calls, _outputs, _report = emit(tmp_path, source)
    assert len(calls) == 2
    assert calls[0].scalar_element_coordinates is not calls[1].scalar_element_coordinates
    assert calls[0].scalar_element_coordinates.identity != calls[1].scalar_element_coordinates.identity
    with pytest.raises(CompilationError):
        replace(calls[0], scalar_element_coordinates=calls[1].scalar_element_coordinates).validate(middle)
    assert builder.finish()


def test_backing_change_between_calls_stays_original_native_work(tmp_path):
    source = SOURCE_ELEMENTS.replace('call consumer(left,right,n,ratios(selector))',
        'call consumer(left,right,n,ratios(selector))\nratios(0)=5.0_8\ncall consumer(left,right,n,ratios(0))')
    path, builder, _middle, calls, outputs, report = emit(tmp_path, source)
    assert len(calls) == 2
    generated = outputs[report['sources'][str(path)]['replacement']]
    assert 'ratios(0)=5.0_8' in generated
    assert builder.finish()


@pytest.mark.parametrize('consumer', ['checked_call', 'publication'])
@pytest.mark.parametrize('change', ['missing', 'copy'])
def test_sealed_hierarchy_keeps_exact_nested_scalar_associations(tmp_path, consumer, change):
    source = SOURCE_ELEMENTS.replace('a(i)=scale*b(i)', 'a(i)=scaled_value(b(i),scale)')
    source = source.replace('end module\n', '''pure real(8) function scaled_value(value,scale)
real(8),intent(in)::value,scale
scaled_value=value*scale
end function
end module
''')
    _path, builder, middle, calls, _outputs, _report = emit(tmp_path, source)
    call, = calls
    proof = call.scalar_element_coordinates
    nested, = proof._element_calls
    assert nested.procedure == 'hierarchical::scaled_value'
    assert builder.finish()
    registry = proof._analysis._original_call_associations
    key = id(proof._analysis), id(nested._original)
    assert registry[key] is nested
    if change == 'missing':
        del registry[key]
    else:
        registry[key] = copy(nested)
    with pytest.raises(CompilationError):
        call.validate(middle) if consumer == 'checked_call' else builder.finish()
    assert registry.get(key) is not nested
