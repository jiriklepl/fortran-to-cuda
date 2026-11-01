"""Whole original native calls retire storage without creating companion SAVE state."""
from copy import copy

import pytest

from compiler.frontend.native_completion import native_call_storage_structures, prove_native_call_retirement
from compiler.frontend.resource_reachability import prove_native_effect_graph, prove_resource_reachability
from compiler.frontend.source_effects import SourceEffects
from compiler.frontend.storage_mutation_effects import prove_storage_mutation_effects
from compiler.ir import CompilationError

SOURCE = '''module work
implicit none
real(8)::left(8),right(8)
contains
subroutine owner()
call native_worker(left,right)
end subroutine
subroutine native_worker(input,output)
real(8),intent(in)::input(8)
real(8),intent(out)::output(8)
real(8),allocatable,save::saved(:)
integer::i
if(.not.allocated(saved))saved=input
!$omp parallel do private(i)
do i=1,8
saved(i)=saved(i)+input(i)
enddo
!$omp end parallel do
!$omp do
do i=1,8
output(i)=saved(i)
enddo
!$omp end do nowait
output(1)=identity_value(output(1))
contains
function identity_value(x) result(value)
real(8),intent(in)::x
real(8)::value
value=x
end function
end subroutine
end module
'''


def setup(tmp_path, source=SOURCE):
    path = tmp_path / 'native.f90'
    path.write_text(source)
    analysis = SourceEffects([path])
    analysis.seal_session()
    selected = tuple(analysis.routines['work::owner'].execution.content)
    graph = prove_native_effect_graph(analysis, 'work::owner', selected)
    return path, analysis, selected, graph


def prove(tmp_path, source=SOURCE):
    path, analysis, selected, graph = setup(tmp_path, source)
    structures = native_call_storage_structures(analysis, graph)
    storage = prove_storage_mutation_effects(analysis, 'work::owner', selection=selected, joined_storage=structures)
    completion = prove_native_call_retirement(analysis, 'work::owner', selected, graph, storage)
    return path, analysis, selected, graph, structures, storage, completion


def test_retirement_discharges_typed_obligations_without_payload_eligibility(tmp_path):
    _path, analysis, selected, graph, structures, storage, completion = prove(tmp_path)
    assert not graph.completion_eligible
    assert graph.retirement_effects_eligible
    assert {item.public()['kind'] for _ordinal, item in graph.lifetime_obligations} == {
        'native_allocation_assignment_obligation', 'native_orphan_worksharing_obligation'}
    assert len(structures) == 2
    assert storage.public()['version'] == 19
    assert storage.public()['complete']
    assert len(storage.public()['mutations']) == 1
    assert not storage.public()['escapes']
    assert not storage.preserves_descriptors()
    assert completion._call_storage is storage
    record = completion.public()
    assert record['schema_version'] == 3
    assert record['completion_scope'] == 'normal return only'
    assert record['requires_domain_retirement']
    assert record['requires_serial_caller']
    assert not record['stable_storage_authorized']
    assert not record['payload_access_authorized']
    demanded = (analysis.modules['work'].bindings['left'], analysis.modules['work'].bindings['right'])
    payload = prove_resource_reachability(analysis, 'work::owner', selected, demanded)
    assert {item.kind for item in payload.barriers} >= {'allocation', 'completion'}
    assert not payload.native_graph.completion_eligible
    analysis.verify_publication()


def test_contained_function_is_owned_by_its_graph_template(tmp_path):
    _path, analysis, _selected, graph = setup(tmp_path)
    template, = [item for item in graph.templates if item.procedure.endswith('::identity_value')]
    assert template.procedure not in analysis.routines
    assert template._routine.qualified == template.procedure
    template.validate(analysis)


@pytest.mark.parametrize('change', ['copied_team', 'copied_worksharing', 'foreign_team'])
def test_exact_structure_receipts_are_required(tmp_path, change):
    _path, analysis, selected, graph = setup(tmp_path)
    structures = list(native_call_storage_structures(analysis, graph))
    if change == 'foreign_team':
        other = tmp_path / 'other'
        other.mkdir()
        foreign = setup(other)
        structures[0] = native_call_storage_structures(foreign[1], foreign[3])[0]
    else:
        index = 0 if change == 'copied_team' else 1
        structures[index] = copy(structures[index])
    with pytest.raises(CompilationError):
        prove_storage_mutation_effects(analysis, 'work::owner', selection=selected, joined_storage=tuple(structures))


def test_missing_orphan_structure_does_not_complete_storage_or_native_return(tmp_path):
    _path, analysis, selected, graph = setup(tmp_path)
    structures = native_call_storage_structures(analysis, graph)
    storage = prove_storage_mutation_effects(analysis, 'work::owner', selection=selected, joined_storage=structures[:1])
    assert not storage.public()['complete']
    with pytest.raises(CompilationError):
        prove_native_call_retirement(analysis, 'work::owner', selected, graph, storage)


@pytest.mark.parametrize('change', ['removed_receipt', 'copied_completion', 'source_file'])
def test_issued_retirement_keeps_source_and_registry_dependencies(tmp_path, change):
    path, analysis, _selected, _graph, structures, _storage, completion = prove(tmp_path)
    if change == 'removed_receipt':
        analysis._original_worksharing_storage.pop(structures[1].identity)
    elif change == 'copied_completion':
        completion = copy(completion)
    else:
        path.write_text(path.read_text().replace('saved(i)+input(i)', 'saved(i)-input(i)'))
    def validate():
        if change == 'source_file':
            analysis.verify_publication()
        else:
            completion.validate(analysis, 'work::owner', completion.selected_node_ids)

    with pytest.raises(CompilationError):
        validate()


def test_original_async_work_remains_unproved(tmp_path):
    source = SOURCE.replace('output(1)=identity_value(output(1))',
        '!$omp task\noutput(1)=identity_value(output(1))\n!$omp end task')
    _path, analysis, _selected, graph = setup(tmp_path, source)
    assert not graph.retirement_effects_eligible
    with pytest.raises(CompilationError):
        native_call_storage_structures(analysis, graph)


def test_unused_original_team_cannot_lend_authority_to_a_call(tmp_path):
    from compiler.frontend.descriptor_teams import prove_descriptor_team
    from compiler.frontend.native_group_structure import NativeGroupCandidate, grouped_original_sequence

    source = SOURCE.replace('end module', '''subroutine unused()
integer::i
!$omp parallel do private(i)
do i=1,8
left(i)=0
enddo
!$omp end parallel do
end subroutine
end module''')
    _path, analysis, selected, graph = setup(tmp_path, source)
    group, = [node for node in grouped_original_sequence(analysis.routines['work::unused'].execution.content, {})
              if isinstance(node, NativeGroupCandidate)]
    extra = prove_descriptor_team(analysis, 'work::unused', group.original_nodes)
    structures = (*native_call_storage_structures(analysis, graph), extra)
    with pytest.raises(CompilationError, match='not consumed'):
        prove_storage_mutation_effects(analysis, 'work::owner', selection=selected, joined_storage=structures)


def test_source_call_at_another_site_cannot_lend_its_storage_proof(tmp_path):
    _path, analysis, selected, _graph = setup(tmp_path, SOURCE.replace(
        'call native_worker(left,right)', 'call native_worker(left,right)\ncall native_worker(right,left)'))
    first = prove_native_effect_graph(analysis, 'work::owner', selected[:1])
    second = prove_native_effect_graph(analysis, 'work::owner', selected[1:])
    storage = prove_storage_mutation_effects(analysis, 'work::owner', selection=selected[:1],
        joined_storage=native_call_storage_structures(analysis, first))
    with pytest.raises(CompilationError):
        prove_native_call_retirement(analysis, 'work::owner', selected[1:], second, storage)


def test_orphan_worksharing_inside_an_original_team_keeps_participation_barrier(tmp_path):
    source = SOURCE.replace('call native_worker(left,right)', 'call team_worker()').replace('end module', '''subroutine team_worker()
!$omp parallel
call native_worker(left,right)
!$omp end parallel
end subroutine
end module''')
    _path, analysis, selected, graph = setup(tmp_path, source)
    structures = native_call_storage_structures(analysis, graph)
    storage = prove_storage_mutation_effects(analysis, 'work::owner', selection=selected, joined_storage=structures)
    with pytest.raises(CompilationError, match='participation'):
        prove_native_call_retirement(analysis, 'work::owner', selected, graph, storage)


def test_selected_root_orphan_group_is_not_whole_native_callee_authority(tmp_path):
    _path, analysis, _selected, _graph = setup(tmp_path)
    selected = tuple(analysis.routines['work::native_worker'].execution.content)
    graph = prove_native_effect_graph(analysis, 'work::native_worker', selected)
    assert not graph.retirement_effects_eligible
    with pytest.raises(CompilationError):
        native_call_storage_structures(analysis, graph)


def test_rolled_back_native_retirement_cannot_survive_its_generation(tmp_path):
    from hashlib import sha256

    from compiler.driver.options import CompilerOptions
    from compiler.offload.config import OffloadConfig
    from compiler.scopes.generation_transaction import GenerationTransaction
    from compiler.scopes.source import ScopeBuilder

    path = tmp_path / 'native.f90'
    path.write_text(SOURCE)
    facts = {'schema_version': 1, 'participation': 'serial',
             'sources': {str(path): sha256(path.read_bytes()).hexdigest()}, 'captures': {}}
    builder = ScopeBuilder([path], 'work::owner', facts=facts, options=CompilerOptions(),
                           config=OffloadConfig(policy='sections', scope_execution='graph'))
    analysis = builder.analysis
    selected = tuple(builder.entry.execution.content)
    with GenerationTransaction(builder):
        graph = prove_native_effect_graph(analysis, 'work::owner', selected)
        structures = native_call_storage_structures(analysis, graph)
        storage = prove_storage_mutation_effects(analysis, 'work::owner', selection=selected, joined_storage=structures)
        temporary = prove_native_call_retirement(analysis, 'work::owner', selected, graph, storage)
    with pytest.raises(CompilationError):
        temporary.validate(analysis, 'work::owner', temporary.selected_node_ids)


@pytest.mark.parametrize('plain', [False, True])
def test_authentic_hierarchy_keeps_original_saved_native_callee_and_strict_transport(tmp_path, plain):
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
    if plain:
        worker = worker.replace('!$omp parallel do private(i)', '!$omp parallel private(i)\n!$omp do schedule(dynamic,4)')
        worker = worker.replace('!$omp end parallel do', '!$omp end do\n!$omp do schedule(guided)\ndo i=1,n\nsaved(i)=saved(i)+values(i)\nenddo\n!$omp end do\n!$omp end parallel')
    source = JOINED.replace(GROUP, 'call saved_worker(right,n)').replace('end module', worker+'end module')
    path, builder, middle, patches, outputs, report = emit(tmp_path, source)
    patch, = patches
    assert patch.completion._call_storage is patch.lifecycle
    assert patch.validate(middle, require_source=True) is patch
    owner = builder._conditional_return_draft.validate().owner
    assert 'hierarchical::saved_worker' not in owner.members
    generated = outputs[report['sources'][str(path)]['replacement']]
    assert worker in generated
    assert 'call saved_worker(right,n)' in generated
    validate_checked_execution_plans(report)
    validate_checked_execution_plans(json.loads(outputs['scope-manifest.json']))
    assert builder.finish()


@pytest.mark.parametrize('change', ['removed', 'copied'])
def test_final_consumption_keeps_every_graph_owned_group_completion(tmp_path, change):
    _path, analysis, _selected, _graph, _structures, _storage, completion = prove(tmp_path)
    assert completion._call_joins
    joins = completion._call_joins
    if change == 'removed':
        altered = joins[:-1]
    else:
        name, identities, proof = joins[0]
        altered = ((name, identities, copy(proof)), *joins[1:])
    object.__setattr__(completion, '_call_joins', altered)
    with pytest.raises(CompilationError, match='completion receipt'):
        completion.validate(analysis, completion.procedure, completion.selected_node_ids)


PLAIN_TEAM = """!$omp parallel private(i)
!$omp do schedule(dynamic,4)
do i=1,8
saved(i)=saved(i)+input(i)
enddo
!$omp end do nowait
!$omp do schedule(guided)
do i=1,8
saved(i)=saved(i)+input(i)
enddo
!$omp end do
!$omp end parallel"""
COMBINED_TEAM = """!$omp parallel do private(i)
do i=1,8
saved(i)=saved(i)+input(i)
enddo
!$omp end parallel do"""


def test_whole_call_retains_plain_team_schedules_and_its_outer_join(tmp_path):
    source = SOURCE.replace(COMBINED_TEAM, PLAIN_TEAM)
    _path, analysis, _selected, graph, _structures, storage, completion = prove(tmp_path, source)
    assert storage.public()['complete']
    assert not graph.completion_eligible
    assert graph.retirement_effects_eligible
    group, = completion.public()['original_groups']
    assert group['graph_completion']['join'] == 'explicit original parallel end'
    assert not completion.public()['payload_access_authorized']
    from compiler.frontend.native_completion import prove_graph_joined_completion
    from compiler.frontend.native_group_structure import NativeGroupCandidate, grouped_original_sequence

    native_group, = [item for item in grouped_original_sequence(
        analysis.routines['work::native_worker'].execution.content, {}) if isinstance(item, NativeGroupCandidate)]
    with pytest.raises(CompilationError):
        prove_graph_joined_completion(analysis, 'work::native_worker', native_group.original_nodes, graph)
    analysis.verify_publication()


@pytest.mark.parametrize(('before', 'after'), [
    ('schedule(dynamic,4)', 'schedule(dynamic,i)'),
    ('schedule(dynamic,4)', 'schedule(dynamic,0)'),
    ('!$omp end parallel', '!$omp end parallel nowait'),
    ('!$omp do schedule(guided)', '!$omp task'),
])
def test_whole_call_does_not_relax_unknown_schedules_or_join_structure(tmp_path, before, after):
    source = SOURCE.replace(COMBINED_TEAM, PLAIN_TEAM).replace(before, after)
    with pytest.raises(CompilationError):
        prove(tmp_path, source)
