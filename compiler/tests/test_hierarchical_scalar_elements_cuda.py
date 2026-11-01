"""Original scalar associations keep their elements through repeated GPU calls."""
import json
import os
import re

import pytest

from compiler.tests import test_graph_storage_epochs_cuda as base
from compiler.tests.test_hierarchical_child_regions_cuda import events_in, procedure, copy_bytes
from compiler.tests.test_hierarchical_scalar_elements import SOURCE_ELEMENTS, emit
from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans


CALL = 'call consumer(left,right,n,ratios(selector))'
SOURCE = (SOURCE_ELEMENTS
    .replace('real(8),intent(out)::b(:)', 'real(8),intent(inout)::b(:)')
    .replace('real(8),intent(out)::a(:)', 'real(8),intent(inout)::a(:)')
    .replace('b(i)=real(i,8)', 'b(i+2)=real(i,8)')
    .replace('do i=1,n\na(i)=scale*b(i)',
             'selector=1-selector\ndo i=1,n\na(i+2)=a(i+2)+scale*b(i+2)')
    .replace('observed=0.0_8', 'ratios(0)=2.0_8\nobserved=0.0_8')
    .replace('right(lbound(right,1)+i-1)', 'right(lbound(right,1)+i+1)')
    .replace('if(change)call relocate(left,right)\n' + CALL,
             CALL + '\nif(change)then\nratios(0)=5.0_8\nselector=0\n' + CALL
             + '\nendif\nselector=1\n' + CALL)
    .replace('allocate(left(-3:n-4),right(2:n+1))\nleft=0\nright=0',
             'allocate(left(-3:n),right(2:n+5))\nleft=-77.0_8\nright=-91.0_8\nleft(-1:n-2)=0.0_8'))

DRIVER = """program exercise
use hierarchical
use iso_fortran_env,only:error_unit
implicit none
integer::mode,case_index,n,i,factor
logical::change
do mode=0,1
change=mode==1
write(error_unit,'(A)')'ELEMENT_ZERO_BEGIN'
call advance(257,0,change)
write(error_unit,'(A)')'ELEMENT_ZERO_END'
if(allocated(left).or.allocated(right))error stop 'zero trip inspected storage'
do case_index=1,3
select case(case_index)
case(1)
n=1
case(2)
n=7
case default
n=257
end select
ratios=[2.0_8,3.0_8]
selector=0
write(error_unit,'(A,1X,I0,1X,I0)')'ELEMENT_CASE_BEGIN',mode,n
call advance(n,2,change)
write(error_unit,'(A,1X,I0,1X,I0)')'ELEMENT_CASE_END',mode,n
if(lbound(left,1)/=-3.or.ubound(left,1)/=n)error stop 'left bounds'
if(lbound(right,1)/=2.or.ubound(right,1)/=n+5)error stop 'right bounds'
if(any(left(-3:-2)/=-77).or.any(left(n-1:n)/=-77))error stop 'left halos'
if(any(right(2:3)/=-91).or.any(right(n+4:n+5)/=-91))error stop 'right halos'
factor=6
if(change)factor=11
do i=1,n
if(left(i-2)/=real(i,8))error stop 'producer values'
if(right(i+3)/=real(factor*i,8))error stop 'original element association'
enddo
if(observed/=real(n*(n+1)/2,8))error stop 'host read'
if(selector/=0.or.ratios(1)/=3)error stop 'selector/table tail'
if(change)then
if(ratios(0)/=5)error stop 'reached table write'
else
if(ratios(0)/=2)error stop 'untaken table write'
endif
write(*,'(7(I0,1X))')mode,n,selector,lbound(left,1),ubound(left,1),lbound(right,1),ubound(right,1)
write(*,'(*(ES26.17E3,1X))')observed,ratios,left,right
deallocate(left,right)
enddo
enddo
print *, 'SCALAR_ELEMENT_FIELDS_HALOS_ASSOCIATION_OK'
end program
"""


def issued(directory):
    original, builder, middle, calls, outputs, report = emit(directory, SOURCE)
    assert len(calls) == 3
    assert len({call.scalar_element_coordinates.identity for call in calls}) == 3
    assert all(call.validate(middle) is call for call in calls)
    assert builder.finish()
    manifest = json.loads(json.dumps(report))
    validate_checked_execution_plans(manifest)
    scope, = manifest['scopes']
    assert {name.split('#')[0] for name in scope['gpu_leaves']} == {
        'hierarchical::producer', 'hierarchical::consumer'}
    resource, = [row for row in scope['resource_bindings']['resources']
                 if row['resource'] == 'hierarchical::ratios']
    assert resource['role'] == 'opaque_scalar_element_alias_reservation'
    assert resource['managed_payload'] is resource['definition_authority'] is False
    assert all(call.scalar_element_coordinates.public()['scalar_element_closure']['version'] == 2 for call in calls)
    generated = outputs[manifest['sources'][str(original)]['replacement']]
    assert 'ratios(selector)' in generated and 'selector=1-selector' in generated
    assert 'fort_element_extents_' in generated
    assert 'fort_scope_buffer_valid_v1' in generated
    return original, outputs, manifest


def test_scalar_hierarchy_source_and_strict_reader_keep_original_associations(tmp_path):
    original, outputs, report = issued(tmp_path)
    package = tmp_path / 'package'
    package.mkdir()
    for name, text in outputs.items():
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    (tmp_path / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    assert original.read_text() == SOURCE


@pytest.fixture(scope='module')
def element_binaries(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        # Reuse the maintained compiler/linker harness; no authority is patched.
        patch.setattr(base, 'emit', issued)
        patch.setattr(base, 'DRIVER', DRIVER)
        build, candidate, native = base.epoch_binaries.__wrapped__(tmp_path_factory)
    manifest = json.loads((build / 'scope-manifest.json').read_text())
    provenance = {row['id']: row for row in manifest['runtime_provenance']['records']}
    resources = {row['resource']: row['registration_identity']
                 for row in manifest['scopes'][0]['resource_bindings']['resources']}
    return build, candidate, native, provenance, resources


@pytest.mark.cuda
@pytest.mark.parametrize('mode', ['traced', 'untraced', 'hidden'])
def test_original_hierarchical_scalar_elements_execute_without_payload_copies(element_binaries, mode):
    build, candidate, native, provenance, resources = element_binaries
    env = {**os.environ, 'OMP_NUM_THREADS': '4', 'OMP_DYNAMIC': 'FALSE'}
    env.pop('FORT_RUNTIME_TRACE', None)
    if mode != 'untraced':
        env['FORT_RUNTIME_TRACE'] = '1'
    if mode == 'hidden':
        env['CUDA_VISIBLE_DEVICES'] = ''
    reference = base.checked([str(native)], build, mode + '-native', env)
    result = base.checked([str(candidate)], build, mode + '-candidate', env)
    assert result.stdout == reference.stdout
    assert 'SCALAR_ELEMENT_FIELDS_HALOS_ASSOCIATION_OK' in result.stdout
    assert 'array temporary' not in reference.stderr.lower() + result.stderr.lower()
    zero = re.findall(r'ELEMENT_ZERO_BEGIN\n(.*?)ELEMENT_ZERO_END', result.stderr, re.S)
    assert len(zero) == 2
    assert all(not any(e['kind'] in {'domain_create', 'create', 'allocate', 'launch'} for e in events_in(text)) for text in zero)
    cases = re.findall(r'ELEMENT_CASE_BEGIN (\d+) (\d+)\n(.*?)ELEMENT_CASE_END \1 \2', result.stderr, re.S)
    assert len(cases) == 6
    ledger = []
    for changed, width, text in cases:
        events = events_in(text)
        launches = [e for e in events if e['kind'] == 'launch']
        copied = [e for e in events if e['kind'] in {'upload', 'download'}]
        row = {'changed': bool(int(changed)), 'n': int(width), 'events': events,
               'launches': [procedure(e, provenance) for e in launches],
               'upload_bytes': copy_bytes(events, 'upload'), 'download_bytes': copy_bytes(events, 'download')}
        ledger.append(row)
        (build / (mode + '-ledger.json')).write_text(json.dumps(ledger, indent=2) + '\n')
        if mode != 'traced':
            assert not launches and not copied
            continue
        expected = (['hierarchical::producer'] + ['hierarchical::consumer'] * (3 if changed == '1' else 2)) * 2
        assert row['launches'] == expected, text
        assert all(int(e['buffer']) in {resources['hierarchical::left'], resources['hierarchical::right']} for e in copied), text
        table = resources['hierarchical::ratios']
        assert not any(e['kind'] in {'allocate', 'upload', 'download', 'device_commit'}
                       and int(e.get('buffer', -1)) == table for e in events), text
        assert all(int(e['bytes']) % 8 == 0 for e in copied)
        assert row['download_bytes'] >= 32 * int(width), text
