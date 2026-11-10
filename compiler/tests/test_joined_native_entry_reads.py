"""Reached CPU initializers preserve earlier GPU replicas across checked children."""

from copy import copy

import pytest
from fparser.two.utils import walk

from compiler.frontend.native_group_structure import NativeGroupCandidate, grouped_original_sequence
from compiler.frontend.reached_read_requirements import native_read_sections, prove_native_read_requirements
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError
from compiler.tests.test_native_call_graph import generate


def nest(statement):
    return 'do k=1,n\ndo j=1,n\ndo i=1,n\n' + statement + '\nenddo\nenddo\nenddo\n'


INITIALIZER = ('!$omp parallel private(i,j,k)\n'
               + ''.join('!$omp do\n' + nest(f'{name}(i,j,k)=x(i,j,k)*rho(stage)*dt')
                         + '!$omp end do nowait\n' for name in ('p', 'q', 'r'))
               + '!$omp end parallel\n')
SOURCE = '''module native_graph
implicit none
real(8)::a(-1:6,-1:6,-1:6),b(-1:6,-1:6,-1:6),c(-1:6,-1:6,-1:6)
real(8)::d(-1:6,-1:6,-1:6),e(-1:6,-1:6,-1:6),f(-1:6,-1:6,-1:6)
real(8),parameter::weights(1:3)=[0.5d0,1.5d0,2.5d0]
contains
subroutine advance(stage,n)
integer,intent(in)::stage,n
integer::i,j,k
''' + nest('b(i,j,k)=2*a(i,j,k)') + '''call velocity(a,b,c,d,e,f,stage,n,weights,0.25d0)
end subroutine
subroutine velocity(input,producer,x,p,q,r,stage,n,rho,dt)
real(8),contiguous,intent(in)::input(-1:,-1:,-1:),producer(-1:,-1:,-1:)
real(8),contiguous,intent(inout)::x(-1:,-1:,-1:),p(-1:,-1:,-1:),q(-1:,-1:,-1:),r(-1:,-1:,-1:)
real(8),intent(in)::rho(1:3),dt
integer,intent(in)::stage,n
integer::i,j,k
if(stage>1)then
''' + INITIALIZER + 'endif\n' + nest('x(i,j,k)=producer(i,j,k)+input(i,j,k)+p(i,j,k)+q(i,j,k)+r(i,j,k)') + '''end subroutine
end module
'''


def emit(directory, source=SOURCE):
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans

    none = {'storage': 'stable', 'initialized': 'none', 'escapes': False, 'allocation_changes': False}
    path, builder, outputs, manifest = generate(directory, source, capture_updates={
        'native_graph::' + name: dict(none) for name in ('b', 'c', 'd', 'e', 'f')})
    validate_checked_execution_plans(manifest)
    return path, builder, outputs, manifest


def proof_fixture(directory, source=SOURCE):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'source.f90'
    path.write_text(source)
    analysis = SourceEffects([path])
    routine = analysis.routines['native_graph::velocity']
    condition, = [node for node in walk(routine.execution) if type(node).__name__ == 'If_Construct']
    group, = [g for g in grouped_original_sequence(condition.content[1:-1], {}) if isinstance(g, NativeGroupCandidate)]
    selection = group.original_nodes
    completion = analysis.joined_completion(routine.qualified, selection)
    proof = prove_native_read_requirements(analysis, routine.qualified, selection, completion=completion)
    return analysis, routine, selection, proof


@pytest.mark.parametrize('change', ['copied_proof', 'foreign_source', 'missing_registry'])
def test_joined_entry_read_authority_requires_original_receipts(tmp_path, change):
    analysis, routine, selection, proof = proof_fixture(tmp_path)
    assert proof is not None
    assert native_read_sections(analysis, routine.qualified, selection, proof).available
    if change == 'copied_proof':
        proof = copy(proof)
    elif change == 'foreign_source':
        _, _, selection, _ = proof_fixture(tmp_path / 'foreign')
    else:
        analysis._joined_completions.clear()
    with pytest.raises(CompilationError):
        native_read_sections(analysis, routine.qualified, selection, proof)


def test_checked_child_initializer_admits_only_its_exact_original_reads(tmp_path):
    _, _, outputs, manifest = emit(tmp_path)
    owner, = manifest['scopes']
    child, = [p for p in owner['checked_execution_plan']['procedures'] if p['procedure'] == 'native_graph::velocity']
    operations = [o for u in child['units'] for o in u['operations']['native_operations']]
    joined, = [o for o in operations if o['kind'] == 'joined native OpenMP']
    assert joined['source_read_requirements'] is not None
    assert not joined['source_read_requirements']['prefix_definitions_authorized']
    assert joined['managed_resources'] == ['argument::p', 'argument::q', 'argument::r', 'argument::x']
    assert joined['immutable_inputs'][0]['formal_resource'] == 'argument::rho'
    source = '\n'.join(text for name, text in outputs.items() if name.startswith('sources/'))
    assert 'fort_scope_admit_source_reads_v1' in source


@pytest.mark.parametrize('change', ['conditional', 'write_then_read', 'scalar_bound', 'conditional_operand'])
def test_communication_envelopes_do_not_authorize_uncertain_entry_reads(tmp_path, change):
    if change == 'conditional':
        group = INITIALIZER.replace('p(i,j,k)=', 'if(x(i,j,k)>0) p(i,j,k)=')
    elif change == 'write_then_read':
        group = INITIALIZER.replace('p(i,j,k)=x(i,j,k)*rho(stage)*dt', 'x(i,j,k)=1.d0\np(i,j,k)=x(i,j,k)*rho(stage)*dt')
    elif change == 'scalar_bound':
        group = INITIALIZER.replace('!$omp parallel private(i,j,k)', '!$omp parallel private(i,j,k)\nn=n-1')
    else:
        group = INITIALIZER.replace('p(i,j,k)=x(i,j,k)*rho(stage)*dt', 'p(i,j,k)=merge(x(i,j,k),0.d0,stage>1)')
    source = SOURCE.replace(INITIALIZER, group).replace('integer,intent(in)::stage,n', 'integer::stage,n')
    try:
        _, _, _, proof = proof_fixture(tmp_path, source)
    except CompilationError:
        return  # Completion/geometry can reject the original group first.
    assert proof is None
