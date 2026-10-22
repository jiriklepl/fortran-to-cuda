"""Full section initialization and updates inside independent source loops."""

import pytest

from compiler.tests.test_source_scopes import FACT, generate

SOURCE = '''module planes
implicit none
contains
subroutine fill(to,c)
real(8),intent(out)::to(-9:,7:,2:)
real(8),intent(in)::c
integer::k
!$omp parallel do private(k)
do k=2,ubound(to,3)
to(:,:,k)=c
enddo
!$omp end parallel do
end subroutine
subroutine update(from,to)
real(8),intent(in)::from(-2:,5:,2:)
real(8),intent(inout)::to(-9:,7:,2:)
integer::k
!$omp parallel do private(k)
do k=2,ubound(to,3)
to(:,:,k)=to(:,:,k)+from(:,:,k)*2+real(k,kind=8)
enddo
!$omp end parallel do
end subroutine
subroutine step(a,b,out,n)
real(8),intent(in)::a(-3:,4:,2:)
real(8),intent(inout)::b(-3:,4:,2:),out(-3:,4:,2:)
integer,intent(in)::n
call fill(b,2.d0)
call update(a,b)
out=b
end subroutine
end module
'''


def build(directory, source=SOURCE):
    return generate(directory, source, scope_execution='reached', facts={
        'schema_version':1, 'participation':'serial', 'captures':{
            'argument::a':FACT, 'argument::b':{**FACT,'initialized':'none'},
            'argument::out':{**FACT,'initialized':'none'}}})


def test_plane_initialization_and_updates_keep_original_bounds(tmp_path):
    _, output, manifest = build(tmp_path)
    assert manifest['scope_count'] == 1, manifest['boundaries']
    assert not manifest['scopes'][0]['boundaries'], manifest['boundaries']
    assert {item['procedure'] for item in manifest['borrowed_source_coordinators']} == {'planes::update'}
    fill, = manifest['scopes'][0]['module_coordinators']
    assert fill['procedure'] == 'planes::fill'
    assert fill['interface'] == 'original_module_entry_v1'
    assert fill['resource_mappings'] == {'argument::to': 'argument::b'}
    plan = manifest['scopes'][0]['checked_execution_plan']
    child, = [item for item in plan['procedures'] if item['procedure'] == 'planes::fill']
    assert len(child['calls']) == 1
    assert child['calls'][0]['caller'] == 'planes::step'
    assert child['units'][0]['resources'] == fill['resource_mappings']
    regions = manifest['child_numerical_regions']
    assert {item['procedure'] for item in regions} == {'planes::fill', 'planes::update'}
    for item in regions:
        selected, = [region for region in item['regions'] if region['used']]
        assert selected['shared_computation'] in plan['gpu_leaves']
        if item['procedure'] == 'planes::fill':
            team = child['units'][0]['operations']['combined_numerical_team']
            assert team['completion']['available']
            assert team['completion']['retains_original_team_and_directives']
            assert team['completion']['proof_identity'] == selected['completion']['proof_identity']
    sources = '\n'.join(path.read_text() for path in (output/'regions').glob('*.source'))
    assert 'fort_section_index_' in sources
    assert 'fort_region_lb_to_1' in sources
    update, = [region for item in regions if item['procedure'] == 'planes::update'
               for region in item['regions'] if region['used']]
    assert any('size(from,1' in guard for guard in update['runtime_guards'])


@pytest.mark.parametrize('expression', ['to(-9,7,2)', 'to(:,:,2)', 'to(-8:,:,k)', 'sum(to(:,:,k))'])
def test_nonpointwise_output_reads_do_not_lose_snapshot_semantics(tmp_path, expression):
    source = SOURCE.replace('to(:,:,k)+from(:,:,k)*2+real(k,kind=8)', expression)
    _, _, manifest = build(tmp_path, source)
    assert all(item['procedure'] != 'planes::update' for item in manifest['child_numerical_regions'])
    assert any('nested array' in item['reason'] for item in manifest['boundaries'])
