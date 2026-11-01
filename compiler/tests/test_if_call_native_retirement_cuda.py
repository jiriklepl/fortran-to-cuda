"""An original false IF keeps residency; a true IF retires before one native CALL."""
import json
import re

import pytest

from compiler.tests import test_graph_storage_epochs_cuda as base
from compiler.tests.test_hierarchical_child_regions import SOURCE, fixture
from compiler.tests.test_hierarchical_child_regions_cuda import copy_bytes, events_in, procedure
from compiler.tests.test_if_call_native_retirement import ABSTRACT
from compiler.tests.test_joined_boundary_lifecycle_cuda import environment

WORKER = '''subroutine saved_worker(values,n)
integer,intent(in)::n
real(8),intent(inout)::values(:)
real(8),allocatable,save::saved(:)
integer::i
native_calls=native_calls+1
saved=values
!$omp parallel do private(i)
do i=3,n+2
saved(i)=saved(i)+1.0_8
enddo
!$omp end parallel do
values=saved
end subroutine
'''


def source():
    return (SOURCE.replace('real(8)::observed', 'real(8)::observed\ninteger::native_calls=0\n'+ABSTRACT)
        .replace('real(8),intent(out)::b(:)', 'real(8),intent(inout)::b(:)')
        .replace('real(8),intent(out)::a(:)', 'real(8),intent(inout)::a(:)')
        .replace('b(i)=real(i,8)', 'b(i+2)=real(i,8)')
        .replace('a(i)=2.0_8*b(i)', 'a(i+2)=2.0_8*b(i+2)')
        .replace('do i=1,n\nobserved=observed+right(lbound(right,1)+i-1)\nenddo\n', '')
        .replace('if(change)call relocate(left,right)', 'if(change)call saved_worker(right,n)')
        .replace('allocate(left(-3:n-4),right(2:n+1))', 'allocate(left(2:n+5),right(2:n+5))')
        .replace('left=0\nright=0', 'left=-101\nright=-99')
        .replace('integer::stage\n', 'integer::stage\nif(n<0)return\n')
        .replace('end module', WORKER+'end module'))


def sealed_emission(directory):
    from compiler.scopes.lexical import LexicalBoundaryPatch
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans
    from elmm_pipeline.local_adapter import _scope_package_build

    original, builder = fixture(directory, source())
    outputs, report = builder.run()
    assert report['scope_count'] == 1, report.get('graph_owner_attempt')
    owner = builder._conditional_return_draft.validate().owner
    middle = owner.members['hierarchical::middle']
    patch, = [patch for _first,_last,patch in middle.patches
              if isinstance(patch, LexicalBoundaryPatch) and patch.completion is not None]
    assert patch.conditional_guard.lower() == 'change'
    assert type(patch.originals[0]).__name__ == 'If_Stmt'
    assert patch.validate(middle, require_source=True) is patch
    assert patch.completion._call_storage is patch.lifecycle
    emitted = outputs[report['sources'][str(original)]['replacement']]
    assert WORKER in emitted
    assert emitted.lower().replace(' ', '').count('callsaved_worker(right,n)') == 1
    scope, = report['scopes']
    assert {name.split('#')[0] for name in scope['gpu_leaves']} == {
        'hierarchical::producer', 'hierarchical::consumer'}
    outputs = dict(outputs)
    validate_checked_execution_plans(report)
    package = directory/'source-package'
    package.mkdir()
    for name,text in outputs.items():
        path = package/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    manifest = json.loads((package/'scope-manifest.json').read_text())
    validate_checked_execution_plans(manifest)
    _scope_package_build({'artifact_root':str(package), 'scope_manifest':manifest,
                          'tool':'local', 'memory_model':'scoped'})
    assert builder.finish()
    (directory/'if-call-proof.json').write_text(json.dumps({
        'completion':patch.completion.public(), 'storage':patch.lifecycle.public(),
        'original_if':[str(node) for node in patch.originals]}, indent=2)+'\n')
    return original, outputs, manifest


def test_conditional_halo_source_preserves_whole_if_proofs_and_current_reader(tmp_path):
    sealed_emission(tmp_path)


DRIVER = '''program exercise
use hierarchical
use iso_fortran_env,only:error_unit
implicit none
integer::mode,case_index,n,i,lo,hi
logical::change
native_calls=0
write(error_unit,'(A)')'IFCALL_COLD_BEGIN'
call advance(-1,1,.true.)
write(error_unit,'(A)')'IFCALL_COLD_END'
if(allocated(left).or.allocated(right).or.native_calls/=0)error stop 'cold path touched storage'
write(error_unit,'(A)')'IFCALL_ZERO_BEGIN'
call advance(257,0,.true.)
write(error_unit,'(A)')'IFCALL_ZERO_END'
if(allocated(left).or.allocated(right).or.native_calls/=0)error stop 'zero path touched storage'
do mode=0,1
change=mode==1
do case_index=1,3
select case(case_index)
case(1)
n=1
case(2)
n=7
case default
n=257
end select
native_calls=0
write(error_unit,'(A,1X,I0,1X,I0)')'IFCALL_CASE_BEGIN',mode,n
call advance(n,1,change)
write(error_unit,'(A,1X,I0,1X,I0)')'IFCALL_CASE_END',mode,n
if(native_calls/=mode)error stop 'native conditional CALL count differs'
if(any(shape(left)/=[n+4]).or.any(shape(right)/=[n+4]))error stop 'shape changed'
if(lbound(left,1)/=2.or.lbound(right,1)/=2)error stop 'allocation origin lost'
lo=lbound(left,1)
hi=ubound(left,1)
if(any(left(lo:lo+1)/=-99).or.any(left(hi-1:hi)/=-99))error stop 'left halo changed'
if(any(right(lo:lo+1)/=-101).or.any(right(hi-1:hi)/=-101))error stop 'right halo changed'
do i=1,n
if(left(i+3)/=real(i+mode,8))error stop 'native modification lost'
if(right(i+3)/=2.0_8*real(i+mode,8))error stop 'GPU continuation differs'
enddo
write(*,'(6(I0,1X))')mode,n,lo,hi,native_calls,size(left)
write(*,'(*(ES26.17E3,1X))')left,right
deallocate(left,right)
enddo
enddo
print *, 'IFCALL_FIELDS_HALOS_GUARDS_AND_BOUNDS_OK'
end program
'''


@pytest.fixture(scope='module')
def binaries(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(base, 'emit', sealed_emission)
        patch.setattr(base, 'DRIVER', DRIVER)
        directory, candidate, native = base.epoch_binaries.__wrapped__(tmp_path_factory)
    manifest = json.loads((directory/'scope-manifest.json').read_text())
    provenance = {row['id']:row for row in manifest['runtime_provenance']['records']}
    return directory,candidate,native,provenance


@pytest.mark.cuda
@pytest.mark.parametrize('mode', ['traced','untraced','hidden'])
def test_false_guard_keeps_domain_and_true_guard_retires_once(binaries, mode):
    directory,candidate,native,provenance = binaries
    env = environment(mode)
    env['FORT_OFFLOAD_TRACE'] = '1'
    reference = base.checked([str(native)], directory, mode+'-native', env)
    result = base.checked([str(candidate)], directory, mode+'-candidate', env)
    assert result.stdout == reference.stdout
    assert 'IFCALL_FIELDS_HALOS_GUARDS_AND_BOUNDS_OK' in result.stdout
    if mode == 'untraced':
        assert 'FORT_SCOPED ' not in result.stderr
        return
    for name in ('COLD','ZERO'):
        text, = re.findall('IFCALL_'+name+'_BEGIN\\n(.*?)IFCALL_'+name+'_END', result.stderr, re.S)
        assert not any(event['kind'] in {'domain_create','register','launch'} for event in events_in(text))
    cases = re.findall(r'IFCALL_CASE_BEGIN (\d+) (\d+)\n(.*?)IFCALL_CASE_END \1 \2', result.stderr, re.S)
    assert len(cases) == 6
    ledger = []
    for flag,width,text in cases:
        events = events_in(text)
        launches = [i for i,event in enumerate(events) if event['kind']=='launch']
        n,changed = int(width),bool(int(flag))
        ledger.append({'changed':changed,'n':n,'events':events,
                       'upload_bytes':copy_bytes(events,'upload'),'download_bytes':copy_bytes(events,'download')})
        if mode == 'hidden':
            assert not launches and copy_bytes(events,'upload') == copy_bytes(events,'download') == 0
            continue
        assert [procedure(events[i],provenance) for i in launches] == [
            'hierarchical::producer','hierarchical::consumer']
        first,last = launches
        between = events[first+1:last]
        if changed:
            assert events[first]['domain'] != events[last]['domain']
            assert sum(event['kind']=='domain_retire' for event in between) == 1
            assert copy_bytes(between,'download') == 8*n
            assert copy_bytes(between,'upload') == 8*n
        else:
            assert events[first]['domain'] == events[last]['domain']
            assert not any(event['kind'] in {'domain_retire','domain_create'} for event in between)
            assert copy_bytes(between,'download') == copy_bytes(between,'upload') == 0
    (directory/(mode+'-ledger.json')).write_text(json.dumps(ledger,indent=2)+'\n')
