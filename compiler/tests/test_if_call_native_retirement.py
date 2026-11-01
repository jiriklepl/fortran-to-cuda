"""One original IF CALL keeps its predicate in existing retirement authorities."""
from dataclasses import replace

import pytest

from compiler.frontend.native_completion import native_call_storage_structures, prove_native_call_retirement
from compiler.frontend.resource_reachability import prove_native_effect_graph
from compiler.frontend.source_effects import SourceEffects
from compiler.frontend.storage_mutation_effects import prove_storage_mutation_effects
from compiler.ir import CompilationError
from compiler.tests.test_native_call_retirement import SOURCE, prove, setup

ABSTRACT = """abstract interface
subroutine boundary_interface(values)
real(8),intent(inout)::values(:)
end subroutine
end interface
"""


def conditional_source(predicate='flag'):
    return (SOURCE.replace('implicit none\n', 'implicit none\n'+ABSTRACT, 1)
        .replace('subroutine owner()', 'subroutine owner(flag)\nlogical,intent(in)::flag')
        .replace('call native_worker(left,right)', f'if({predicate})call native_worker(left,right)'))


@pytest.mark.parametrize('predicate', ['flag', '.false.', '.true.'])
def test_whole_if_predicate_and_callees_share_exact_registered_retirement(tmp_path, predicate):
    _path, analysis, selected, graph, _structures, storage, completion = prove(tmp_path, conditional_source(predicate))
    assert type(selected[0]).__name__ == 'If_Stmt'
    assert graph._selection is selected or graph._selection == selected
    assert storage._selection == selected
    _original_graph, identities = analysis._selected_source('work::owner', selected)
    assert completion.selected_node_ids == identities
    assert storage.public()['original_selection']['selected_node_ids'] == list(identities)
    assert storage.public()['complete'] and not storage.public()['escapes']
    assert completion.public()['completion_scope'] == 'normal return only'
    assert not completion.public()['payload_access_authorized']
    assert not completion.public()['stable_storage_authorized']
    assert 'boundary_interface' in analysis.modules['work'].procedure_arguments
    with pytest.raises(CompilationError):
        graph.validate(analysis, 'work::owner', (selected[0].items[1],))


@pytest.mark.parametrize('change', ['predicate_source', 'foreign_predicate', 'dropped_predicate'])
def test_changed_or_foreign_predicate_cannot_reuse_retirement_authority(tmp_path, change):
    path, analysis, selected, graph, _structures, storage, completion = prove(tmp_path, conditional_source())
    if change == 'predicate_source':
        path.write_text(path.read_text().replace('if(flag)', 'if(.not.flag)'))
        with pytest.raises(CompilationError, match='changed'):
            analysis.verify_publication()
        return
    if change == 'foreign_predicate':
        other = tmp_path/'other'
        other.mkdir()
        foreign = setup(other, conditional_source('.not.flag'))
        original = foreign[2]
    else:
        original = (selected[0].items[1],)
    with pytest.raises(CompilationError):
        graph.validate(analysis, 'work::owner', original)
    with pytest.raises(CompilationError):
        prove_native_call_retirement(analysis, 'work::owner', original, graph, storage)
    assert completion.validate(analysis, 'work::owner', completion.selected_node_ids) is completion


@pytest.mark.parametrize('replacement', [
    'call unknown(input)',
    '!$omp task\noutput(1)=input(1)\n!$omp end task',
])
def test_false_source_guard_cannot_hide_unknown_or_asynchronous_callee_work(tmp_path, replacement):
    source = conditional_source('.false.').replace('output(1)=identity_value(output(1))', replacement)
    _path, analysis, _selected, graph = setup(tmp_path, source)
    assert not graph.retirement_effects_eligible
    with pytest.raises(CompilationError):
        native_call_storage_structures(analysis, graph)


def test_unknown_predicate_is_not_peeled_away_before_graph_completion(tmp_path):
    _path, analysis, _selected, graph = setup(tmp_path, conditional_source('unknown_flag()'))
    assert not graph.retirement_effects_eligible
    with pytest.raises(CompilationError):
        native_call_storage_structures(analysis, graph)


def test_renamed_reexported_abstract_interface_cannot_fall_back_to_host_body(tmp_path):
    declarations = 'module interfaces\n'+ABSTRACT+'end module\n' + """module bridge
use interfaces,only:renamed=>boundary_interface
end module
"""
    source = declarations+SOURCE.replace('subroutine owner()', '''subroutine owner()
use bridge,only:native_worker=>renamed''')
    path = tmp_path/'shadow.f90'
    path.write_text(source)
    analysis = SourceEffects([path])
    analysis.seal_session()
    routine = analysis.routines['work::owner']
    assert analysis._candidates(routine.scope, 'native_worker') != ['work::native_worker']
    selected = tuple(routine.execution.content)
    graph = prove_native_effect_graph(analysis, 'work::owner', selected)
    assert not graph.retirement_effects_eligible
    with pytest.raises(CompilationError):
        native_call_storage_structures(analysis, graph)


def test_abstract_interface_does_not_authorize_procedure_dummy_dispatch(tmp_path):
    source = conditional_source().replace('call native_worker(left,right)', 'call callback_worker(callback_impl)')
    source = source.replace('end module', '''subroutine callback_worker(callback)
procedure(boundary_interface)::callback
call callback(left)
end subroutine
subroutine callback_impl(values)
real(8),intent(inout)::values(:)
values=1
end subroutine
end module''')
    _path, analysis, _selected, graph = setup(tmp_path, source)
    assert not graph.retirement_effects_eligible
    with pytest.raises(CompilationError):
        native_call_storage_structures(analysis, graph)


def test_hierarchy_retires_only_under_original_metadata_guard_and_preserves_reader(tmp_path):
    import json
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans
    from compiler.tests.test_joined_boundary_lifecycle import GROUP, JOINED, emit

    worker = '''subroutine saved_worker(values,n)
integer,intent(in)::n
real(8),intent(inout)::values(:)
real(8),allocatable,save::saved(:)
integer::i
if(.not.allocated(saved))saved=values
!$omp parallel do private(i)
do i=1,n
saved(i)=saved(i)+values(i)
enddo
!$omp end parallel do
!$omp do
do i=1,n
values(i)=saved(i)
enddo
!$omp end do nowait
end subroutine
'''
    source = (JOINED.replace('implicit none\n', 'implicit none\n'+ABSTRACT, 1)
        .replace(GROUP, 'if(change)call saved_worker(right,n)')
        .replace('end module', worker+'end module'))
    path, builder, middle, patches, outputs, report = emit(tmp_path, source)
    patch, = patches
    assert patch.conditional_guard.lower() == 'change'
    assert type(patch.originals[0]).__name__ == 'If_Stmt'
    assert patch.completion._call_storage is patch.lifecycle
    assert patch.validate(middle, require_source=True) is patch
    with pytest.raises(CompilationError):
        replace(patch, conditional_guard='.not.change').validate(middle, require_source=True)
    generated = outputs[report['sources'][str(path)]['replacement']]
    assert worker in generated
    assert generated.lower().replace(' ', '').count('callsaved_worker(right,n)') == 1
    validate_checked_execution_plans(report)
    validate_checked_execution_plans(json.loads(outputs['scope-manifest.json']))
    assert builder.finish()


def test_false_effectful_payload_predicate_closes_before_untouched_if(tmp_path):
    from compiler.tests.test_if_call_native_retirement_cuda import source
    from compiler.tests.test_joined_boundary_lifecycle import emit

    text = source().replace('if(change)call saved_worker(right,n)',
                            'if(probe_false())call saved_worker(right,n)')
    text = text.replace('end module', 'logical function probe_false()\nobserved=observed+right(4)\nprobe_false=.false.\nend function\nend module')
    _path, builder, middle, patches, _outputs, _report = emit(tmp_path, text)
    patch, = patches
    assert patch.conditional_guard is None
    assert type(patch.originals[0]).__name__ == 'If_Stmt'
    assert patch.completion._call_storage is patch.lifecycle
    rendered = patch.render(middle).lower().replace(' ', '')
    assert rendered.index('fort_scope_resident_close_v1(') < rendered.index('if(probe_false())')
    assert rendered.count('probe_false()') == 1
    with pytest.raises(CompilationError):
        replace(patch, conditional_guard=str(patch.originals[0].items[0])).validate(middle, require_source=True)
    assert patch.validate(middle, require_source=True) is patch
    assert builder.finish()
