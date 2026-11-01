"""Original synchronous native teams retire and rearm actual child kernels."""
import json
import os
import re

import pytest

from compiler.tests import test_graph_storage_epochs_cuda as base
from compiler.tests.test_hierarchical_child_regions_cuda import copy_bytes, events_in, procedure
from compiler.tests.test_joined_boundary_lifecycle import GROUP, JOINED, emit


def halo_source(fatal=False):
    # Both physical origins are deliberately explicit and equal: the original
    # scalar CALL uses affine integer indexing, including after MOVE_ALLOC.
    source = (JOINED
              .replace('real(8),intent(out)::b(:)', 'real(8),intent(inout)::b(:)')
              .replace('real(8),intent(out)::a(:)', 'real(8),intent(inout)::a(:)')
              .replace('b(i)=real(i,8)', 'b(i+2)=real(i-2,8)' if fatal else 'b(i+2)=real(i,8)')
              .replace('a(i)=2.0_8*b(i)', 'a(i+2)=2.0_8*b(i+2)')
              .replace('right(lbound(right,1)+i-1)', 'right(lbound(right,1)+i+1)')
              .replace('call native_step(right(i+1))', 'call native_step(right(i+3))')
              .replace('allocate(left(-3:n-4),right(2:n+1))', 'allocate(left(2:n+5),right(2:n+5))')
              .replace('left=0\nright=0', 'left=-101\nright=-99'))
    assert source.count('call native_step(right(i+3))') == 1
    return source


def sealed_emission(directory, fatal=False):
    from elmm_pipeline.local_adapter import _scope_package_build
    original, builder, middle, patches, outputs, report = emit(directory, halo_source(fatal))
    patch, = patches
    assert patch.completion.public()['completion_scope'] == 'normal return only'
    assert patch.validate(middle, require_source=True) is patch
    assert patch.lifecycle.preserves_descriptors()
    assert builder.finish()
    scope, = report['scopes']
    assert {name.split('#')[0] for name in scope['gpu_leaves']} == {
        'hierarchical::producer', 'hierarchical::consumer'}
    returned = scope['checked_execution_plan']['storage_epoch']['return_contract']
    member, = [row for row in returned['members'] if row['procedure'] == 'hierarchical::middle']
    assert any(row['kind'] == 'reached_retirement' and row['rearm_allowed'] for row in member['routes'])
    assert any(row['kind'] == 'checked_child' and row['procedure'] == 'hierarchical::consumer'
               for row in member['routes'])
    transformed = outputs[report['sources'][str(original)]['replacement']]
    group = (GROUP.replace('right(lbound(right,1)+i-1)', 'right(lbound(right,1)+i+1)')
             .replace('call native_step(right(i+1))', 'call native_step(right(i+3))'))
    assert group in transformed
    assert 'write(error_unit' in transformed.lower() and '\nstop\n' in transformed.lower()
    package = directory / 'source-package'
    package.mkdir()
    for name, text in outputs.items():
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    manifest_path = package / 'scope-manifest.json'
    manifest_path.write_text(json.dumps(report, indent=2) + '\n')
    # Validate emitted JSON transport, including typed bindings and exact edits.
    manifest = json.loads(manifest_path.read_text())
    _scope_package_build({'artifact_root': str(package), 'scope_manifest': manifest,
                          'tool': 'local', 'memory_model': 'scoped'})
    (directory / 'native-boundary-proof.json').write_text(json.dumps({
        'completion': patch.completion.public(), 'storage': patch.lifecycle._joined_storage.public(),
        'original_group': group, 'fatal': fatal}, indent=2) + '\n')
    return original, outputs, manifest


@pytest.mark.parametrize('fatal', [False, True], ids=['normal', 'terminal'])
def test_halo_native_boundary_uses_original_sealed_proofs_and_strict_transport(tmp_path, fatal):
    sealed_emission(tmp_path, fatal)


DRIVER = """program exercise
use hierarchical
use iso_fortran_env,only:error_unit
use omp_lib,only:omp_get_max_threads
implicit none
integer::case_index,n,i,lo,hi
character(8)::argument
if(omp_get_max_threads()/=4)error stop 'four threads required'
call get_command_argument(1,argument)
if(trim(argument)=='fatal')then
write(error_unit,'(A)')'JOINED_FATAL_BEGIN'
call advance(7,1,.false.)
error stop 'JOINED_FATAL_RETURNED'
endif
write(error_unit,'(A)')'JOINED_ZERO_BEGIN'
call advance(257,0,.false.)
write(error_unit,'(A)')'JOINED_ZERO_END'
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
write(error_unit,'(A,1X,I0)')'JOINED_CASE_BEGIN',n
call advance(n,2,.false.)
write(error_unit,'(A,1X,I0)')'JOINED_CASE_END',n
if(any(shape(left)/=[n+4]).or.any(shape(right)/=[n+4]))error stop 'shape changed'
if(lbound(left,1)/=2.or.lbound(right,1)/=2)error stop 'allocation origin lost'
lo=lbound(left,1)
hi=ubound(left,1)
if(any(left(lo:lo+1)/=-101).or.any(left(hi-1:hi)/=-101))error stop 'left halo changed'
if(any(right(lo:lo+1)/=-99).or.any(right(hi-1:hi)/=-99))error stop 'right halo changed'
do i=1,n
if(left(i+3)/=real(i+1,8))error stop 'native producer modification lost'
if(right(i+3)/=2.0_8*real(i+1,8))error stop 'consumer output differs'
enddo
write(*,'(5(I0,1X))')n,lbound(left,1),ubound(left,1),lbound(right,1),ubound(right,1)
write(*,'(*(ES26.17E3,1X))')observed,left,right
deallocate(left,right)
enddo
print *, 'JOINED_NATIVE_FIELDS_HALOS_AND_BOUNDS_OK'
end program
"""


@pytest.fixture(scope='module')
def joined_boundary_binaries(tmp_path_factory):
    def build(fatal):
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(base, 'emit', lambda directory: sealed_emission(directory, fatal))
            patch.setattr(base, 'DRIVER', DRIVER)
            directory, candidate, native = base.epoch_binaries.__wrapped__(tmp_path_factory)
        manifest = json.loads((directory / 'scope-manifest.json').read_text())
        provenance = {row['id']: row for row in manifest['runtime_provenance']['records']}
        return directory, candidate, native, provenance
    return {'normal': build(False), 'terminal': build(True)}


def environment(mode):
    env = {**os.environ, 'OMP_NUM_THREADS': '4', 'OMP_THREAD_LIMIT': '4',
           'OMP_DYNAMIC': 'FALSE', 'OMP_MAX_ACTIVE_LEVELS': '1'}
    env.pop('FORT_RUNTIME_TRACE', None)
    # Preserve the controller's exact visible-device binding on enabled runs.
    if mode == 'hidden':
        env['CUDA_VISIBLE_DEVICES'] = ''
    if mode != 'untraced':
        env['FORT_RUNTIME_TRACE'] = '1'
    return env


@pytest.mark.cuda
@pytest.mark.parametrize('mode', ['traced', 'untraced', 'hidden'])
def test_original_joined_native_return_rearms_consumer_and_preserves_halos(joined_boundary_binaries, mode):
    build, candidate, native, provenance = joined_boundary_binaries['normal']
    env = environment(mode)
    reference = base.checked([str(native)], build, mode + '-native', env)
    result = base.checked([str(candidate)], build, mode + '-candidate', env)
    assert result.stdout == reference.stdout
    assert 'JOINED_NATIVE_FIELDS_HALOS_AND_BOUNDS_OK' in result.stdout
    assert 'negative input' not in result.stderr
    if mode == 'untraced':
        assert 'FORT_SCOPED ' not in result.stderr
        return
    zero, = re.findall(r'JOINED_ZERO_BEGIN\n(.*?)JOINED_ZERO_END', result.stderr, re.S)
    assert not any(event['kind'] in {'domain_create', 'register', 'launch'} for event in events_in(zero))
    cases = re.findall(r'JOINED_CASE_BEGIN (\d+)\n(.*?)JOINED_CASE_END \1', result.stderr, re.S)
    assert len(cases) == 3
    ledger = []
    for width, text in cases:
        n = int(width)
        events = events_in(text)
        launches = [i for i, event in enumerate(events) if event['kind'] == 'launch']
        ledger.append({'n': n, 'events': events, 'upload_bytes': copy_bytes(events, 'upload'),
                       'download_bytes': copy_bytes(events, 'download')})
        (build / (mode + '-ledger.json')).write_text(json.dumps(ledger, indent=2) + '\n')
        if mode == 'hidden':
            assert not launches and copy_bytes(events, 'upload') == copy_bytes(events, 'download') == 0
            continue
        assert [procedure(events[i], provenance) for i in launches] == [
            'hierarchical::producer', 'hierarchical::consumer'] * 2
        for stage in range(2):
            first, last = launches[2*stage:2*stage+2]
            between = events[first+1:last]
            assert events[first]['domain'] != events[last]['domain']
            assert any(event['kind'] == 'domain_retire' for event in between)
            assert copy_bytes(between, 'download') == 8*n
            assert copy_bytes(between, 'upload') == 8*n
        assert copy_bytes(events, 'upload') == 16*n
        assert copy_bytes(events, 'download') == 32*n


@pytest.mark.cuda
def test_original_guarded_stop_after_producer_never_runs_consumer_or_replays(joined_boundary_binaries):
    build, candidate, native, provenance = joined_boundary_binaries['terminal']
    env = environment('traced')
    reference = base.checked([str(native), 'fatal'], build, 'fatal-native', env)
    result = base.checked([str(candidate), 'fatal'], build, 'fatal-candidate', env)
    # Original STOP has no code and returns zero on this Fortran runtime.
    assert not result.stdout and result.stdout == reference.stdout
    for stderr in (reference.stderr, result.stderr):
        assert stderr.count('JOINED_FATAL_BEGIN') == 1
        assert stderr.count('negative input') == 1
        assert 'JOINED_FATAL_RETURNED' not in stderr
    prefix = result.stderr[:result.stderr.index('negative input')]
    events = events_in(prefix)
    assert [procedure(event, provenance) for event in events if event['kind'] == 'launch'] == [
        'hierarchical::producer']
    assert [procedure(event, provenance) for event in events_in(result.stderr) if event['kind'] == 'launch'] == [
        'hierarchical::producer']
    assert any(event['kind'] == 'domain_retire' for event in events)
    assert copy_bytes(events, 'download') == 8*7
    assert copy_bytes(events, 'upload') == 0
    (build / 'fatal-ledger.json').write_text(json.dumps({
        'events_before_diagnostic': events, 'diagnostic_count': 1,
        'producer_launches': 1, 'consumer_launches': 0,
        'source_stop_exit_status': result.returncode}, indent=2) + '\n')
