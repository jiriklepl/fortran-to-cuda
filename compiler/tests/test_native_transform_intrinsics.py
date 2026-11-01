"""Native matrix transforms read complete operands through original resolution."""
from copy import copy

import pytest

from compiler.frontend import resource_reachability as native
from compiler.frontend.descriptor_teams import prove_descriptor_team
from compiler.frontend.native_completion import prove_graph_joined_completion
from compiler.frontend.native_group_structure import NativeGroupCandidate, grouped_original_sequence
from compiler.frontend.source_effects import SourceEffects
from compiler.frontend.storage_mutation_effects import prove_storage_mutation_effects
from compiler.ir import CompilationError

SOURCE = """module work
implicit none
real(8)::a(2,3),b(3,3),observed
contains
subroutine owner()
b=matmul(transpose(a),a)
end subroutine
end module
"""


def setup(tmp_path, source=SOURCE):
    path = tmp_path / 'native.f90'
    path.write_text(source)
    analysis = SourceEffects([path])
    analysis.seal_session()
    originals = tuple(analysis.routines['work::owner'].execution.content)
    demands = tuple(analysis.modules['work'].bindings[name] for name in ('a', 'b', 'observed'))
    return path, analysis, originals, demands


@pytest.mark.parametrize('declarations', ['', 'intrinsic::matmul,transpose\n'])
@pytest.mark.parametrize('expression', ['matmul(transpose(a),a)', 'matmul(matrix_a=transpose(a),matrix_b=a)'])
def test_native_transforms_keep_operand_reads_and_result_write(tmp_path, declarations, expression):
    source = SOURCE.replace('subroutine owner()\n', 'subroutine owner()\n' + declarations)
    source = source.replace('matmul(transpose(a),a)', expression)
    _path, analysis, originals, demands = setup(tmp_path, source)
    proof = native.prove_resource_reachability(analysis, 'work::owner', originals, demands)
    assert not proof.barriers
    assert proof.native_graph.completion_eligible
    assert [access.storage.binding for access in proof.accesses if access.kind == 'read'] == [demands[0], demands[0]]
    assert [access.storage.binding for access in proof.accesses if access.kind == 'write'] == [demands[1]]
    assert not proof.calls
    assert not proof.public()['execution_authorized']
    assert proof.native_graph.public()['limits']['source_policy_version'] == native.VERSION


@pytest.mark.parametrize('transform', ['matmul', 'transpose'])
def test_source_function_named_like_transform_keeps_its_effects(tmp_path, transform):
    expression = 'matmul(a,a)' if transform == 'matmul' else 'transpose(a)'
    arguments = 'left,right' if transform == 'matmul' else 'left'
    declarations = 'real(8),intent(in)::left(2,2)' + (',right(2,2)' if transform == 'matmul' else '')
    helper = f'''function {transform}({arguments}) result(value)
{declarations}
real(8)::value(2,2)
observed=observed+1.0_8
value=left
end function
'''
    source = SOURCE.replace('a(2,3),b(3,3)', 'a(2,2),b(2,2)').replace('matmul(transpose(a),a)', expression)
    source = source.replace('end module\n', helper + 'end module\n')
    _path, analysis, originals, demands = setup(tmp_path, source)
    proof = native.prove_resource_reachability(analysis, 'work::owner', originals, demands)
    assert not proof.barriers
    assert any(call.callee == 'work::' + transform for call in proof.calls)
    assert any(access.kind == 'write' and access.storage.binding is demands[2] for access in proof.accesses)


@pytest.mark.parametrize('transform', ['matmul', 'transpose'])
def test_unresolved_external_transform_is_not_assumed_intrinsic(tmp_path, transform):
    source = SOURCE.replace('subroutine owner()\n', f'subroutine owner()\nreal(8),external::{transform}\n')
    _path, analysis, originals, demands = setup(tmp_path, source)
    proof = native.prove_resource_reachability(analysis, 'work::owner', originals, demands)
    assert proof.barriers
    assert not proof.native_graph.completion_eligible


def test_transform_operands_do_not_hide_source_function_writes(tmp_path):
    source = SOURCE.replace('transpose(a)', 'transpose(changed())').replace('end module\n', '''function changed() result(value)
real(8)::value(2,3)
observed=observed+1.0_8
value=a
end function
end module
''')
    _path, analysis, originals, demands = setup(tmp_path, source)
    proof = native.prove_resource_reachability(analysis, 'work::owner', originals, demands)
    assert not proof.barriers
    assert any(call.callee == 'work::changed' for call in proof.calls)
    assert any(access.kind == 'write' and access.storage.binding is demands[2] for access in proof.accesses)


def test_unknown_operand_call_stays_a_barrier(tmp_path):
    _path, analysis, originals, demands = setup(tmp_path, SOURCE.replace('transpose(a)', 'transpose(missing())'))
    proof = native.prove_resource_reachability(analysis, 'work::owner', originals, demands)
    assert proof.barriers
    assert not proof.native_graph.completion_eligible


@pytest.mark.parametrize('change', ['file', 'policy', 'copied_graph'])
def test_source_policy_and_issued_graph_ownership_remain_required(tmp_path, monkeypatch, change):
    path, analysis, originals, _demands = setup(tmp_path)
    graph = native.prove_native_effect_graph(analysis, 'work::owner', originals)
    if change == 'file':
        path.write_text(path.read_text().replace('transpose(a)', 'transpose(a*2.0_8)'))
    elif change == 'policy':
        monkeypatch.setattr(native, 'VERSION', native.VERSION + 1)
    else:
        graph = copy(graph)
    def validate():
        if change == 'file':
            analysis.verify_publication()
        else:
            graph.validate(analysis, 'work::owner', originals)

    with pytest.raises(CompilationError):
        validate()


def test_nonpure_native_callee_joins_and_has_complete_storage_without_gpu_lowering(tmp_path):
    source = '''module work
implicit none
real(8)::a(2,2,7),b(2,2,7),observed
contains
subroutine owner()
real(8)::g(2,2)
integer::i
!$omp parallel do private(g,i)
do i=1,7
call transform(g,a(:,:,i))
b(:,:,i)=g
enddo
!$omp end parallel do
end subroutine
subroutine transform(out,x)
real(8),intent(out)::out(2,2)
real(8),intent(in)::x(2,2)
out=matmul(transpose(x),x)
end subroutine
end module
'''
    _path, analysis, _originals, _demands = setup(tmp_path, source)
    group, = (node for node in grouped_original_sequence(analysis.routines['work::owner'].execution.content, {})
              if isinstance(node, NativeGroupCandidate))
    originals = tuple(group.original_nodes)
    graph = native.prove_native_effect_graph(analysis, 'work::owner', originals)
    assert graph.completion_eligible
    completion = prove_graph_joined_completion(analysis, 'work::owner', originals, graph)
    team = prove_descriptor_team(analysis, 'work::owner', originals)
    storage = prove_storage_mutation_effects(analysis, 'work::owner', selection=team.selected_node_ids, joined_storage=team)
    assert storage.public()['complete']
    assert not storage.public()['escapes']
    assert completion.public()['completion_scope'] == 'normal return only'
    assert not completion.public()['gpu_legality_established']
    assert completion.selected_node_ids == team.selected_node_ids
    analysis.verify_publication()


def test_matrix_transform_retirement_emits_and_passes_strict_saved_package_reader(tmp_path):
    import json

    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans

    from compiler.tests.test_joined_boundary_lifecycle import JOINED, emit

    source = JOINED.replace('value=value+1', 'real(8)::matrix(2,2),product(2,2)\nmatrix=value\nproduct=matmul(transpose(matrix),matrix)\nvalue=product(1,1)')
    path, builder, middle, patches, outputs, report = emit(tmp_path, source)
    patch, = patches
    assert patch.completion._graph.public()['limits']['source_policy_version'] == native.VERSION
    assert patch.lifecycle.public()['complete']
    assert not patch.lifecycle.public()['escapes']
    assert patch.validate(middle, require_source=True) is patch
    assert 'matmul(transpose(matrix),matrix)' in outputs[report['sources'][str(path)]['replacement']].lower()
    for name, record in (('report.json', report), ('scope-manifest.json', json.loads(outputs['scope-manifest.json']))):
        saved = tmp_path / name
        saved.write_text(json.dumps(record))
        validate_checked_execution_plans(json.loads(saved.read_text()))
    assert builder.finish()
