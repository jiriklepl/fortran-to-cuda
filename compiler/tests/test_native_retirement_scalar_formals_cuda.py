"""Original scalar OUT reduction feeds a retained GPU consumer after retirement."""
import json
import re
import shutil

import pytest

from compiler.tests import test_graph_storage_epochs_cuda as base
from compiler.tests.test_hierarchical_child_regions_cuda import copy_bytes, events_in, procedure
from compiler.tests.test_joined_boundary_lifecycle_cuda import environment
from compiler.tests.test_native_retirement_formal_arrays import fixture
from compiler.tests.test_native_retirement_scalar_formals import SOURCE

# Actual current scoped_runtime.cu trace operation names. Scratch leases alone
# are bookkeeping; scratch_allocate/free are the physical allocation traces.
PHYSICAL_GPU_TRACE_KINDS = frozenset({
    'initialize', 'initialize_reuse', 'wait', 'allocate', 'release', 'evict',
    'scratch_allocate', 'scratch_free', 'upload', 'download',
    'upload_coalesced', 'upload_requested', 'launch',
    'graph_submit', 'graph_kernel', 'graph_warmup_direct',
})


def physical_gpu_events(text):
    result = [event for event in events_in(text) if event['kind'] in PHYSICAL_GPU_TRACE_KINDS]
    for line in text.splitlines():
        if not line.startswith('FORT_SCOPED evidence '):
            continue
        event = json.loads(line.split(' ', 2)[2])
        kind = event.get('event')
        if kind in {'compact_transfer', 'graph_statistics'} or kind == 'transfer_statistics' and any(event.get(name, 0) for name in (
                'pinned_uploads', 'pinned_downloads', 'pinned_upload_bytes', 'pinned_download_bytes',
                'tiles', 'events', 'event_waits', 'staging_allocations', 'staging_device_bytes')) or kind == 'completion_statistics' and event.get('event_records', 0):
            result.append(event)
    return result


def source():
    text = SOURCE.replace('real(8)::mean(3)=0', 'real(8)::mean(3)=0\ninteger::native_calls=0')
    old = '''subroutine advance(n,go,masked)
integer,intent(in)::n
logical,intent(in)::go,masked
call middle(n,go,masked)
end subroutine'''
    new = '''subroutine advance(n,stages,go,masked)
integer,intent(in)::n,stages
logical,intent(in)::go,masked
integer::stage
if(n<0)return
do stage=1,stages
call middle(n,go,masked)
enddo
end subroutine'''
    assert text.count(old) == 1
    text = text.replace(old, new)
    replacements = [
        ('call consumer(output,u,v,w,n)', 'call consumer(output,u,v,w,n,tke)'),
        ('subroutine consumer(a,b,c,d,n)\ninteger,intent(in)::n',
         'subroutine consumer(a,b,c,d,n,factor)\ninteger,intent(in)::n\nreal(8),intent(in)::factor'),
        ('a(i,j,k)=b(i,j,k)+c(i,j,k)+d(i,j,k)', 'a(i,j,k)=b(i,j,k)+c(i,j,k)+d(i,j,k)+factor'),
        ('    tke = 0', '    native_calls=native_calls+1\n    tke = 0'),
    ]
    for original, replacement in replacements:
        assert text.count(original) == 1
        text = text.replace(original, replacement)
    return text


def sealed_emission(directory):
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans
    from elmm_pipeline.local_adapter import _scope_package_build

    from compiler.scopes.lexical import LexicalBoundaryPatch

    original, builder = fixture(directory, source())
    outputs, report = builder.run()
    assert report['scope_count'] == 1, report.get('graph_owner_attempt')
    owner = builder._conditional_return_draft.validate().owner
    middle = owner.members['forcing::middle']
    patch, = [patch for _first, _last, patch in middle.patches
              if isinstance(patch, LexicalBoundaryPatch) and patch.completion is not None]
    assert patch.conditional_guard.lower() == 'go'
    assert patch.completion._call_storage is patch.lifecycle
    assert patch.completion._reduction_teams == ()
    assert len(patch.lifecycle._joined_storage) == 2
    for team in patch.lifecycle._joined_storage:
        assert team.public()['native_lifetime_scalar_reductions']['tke']
    scope, = report['scopes']
    assert {name.split('#')[0] for name in scope['gpu_leaves']} == {'forcing::producer', 'forcing::consumer'}
    assert all(row['resource'] != 'parameters::prtype' for row in scope['checked_execution_plan']['storage_epoch']['resources'])
    assert 'parameters::prtype' not in builder.facts['captures']
    package = directory / 'source-package'
    for name, text in outputs.items():
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    manifest = json.loads((package / 'scope-manifest.json').read_text())
    validate_checked_execution_plans(manifest)
    _scope_package_build({'artifact_root': str(package), 'scope_manifest': manifest})
    assert builder.finish()
    return original, outputs, manifest


def test_scalar_retirement_source_and_strict_reader_preserve_guard_and_host_mask(tmp_path):
    sealed_emission(tmp_path)


DRIVER = '''program exercise
use forcing
use parameters
use iso_fortran_env,only:error_unit
implicit none
integer::flag,mask_mode,case_index,n,i,j,k,free_count
logical::go,masked,interior
real(8)::expected_tke,expected_output
native_calls=0
write(error_unit,'(A)')'SCALAR_COLD_BEGIN'
call advance(-1,1,.true.,.true.)
write(error_unit,'(A)')'SCALAR_COLD_END'
if(allocated(u).or.allocated(v).or.allocated(w).or.allocated(output).or.allocated(Prtype))error stop 'cold touched storage'
if(native_calls/=0)error stop 'cold called reduction'
write(error_unit,'(A)')'SCALAR_ZERO_BEGIN'
call advance(7,0,.true.,.true.)
write(error_unit,'(A)')'SCALAR_ZERO_END'
if(allocated(u).or.allocated(v).or.allocated(w).or.allocated(output).or.allocated(Prtype))error stop 'zero touched storage'
if(native_calls/=0)error stop 'zero called reduction'
do flag=0,1
go=flag==1
do mask_mode=0,1
masked=mask_mode==1
do case_index=1,2
n=1
if(case_index==2)n=7
Prnx=n
Prny=n
Prnz=n
n_full_im_U=0
n_full_im_V=0
n_full_im_W=0
if(masked)then
n_full_im_U=1
n_full_im_V=1
n_full_im_W=1
endif
allocate(u(-2:n+3,-2:n+3,-2:n+3),v(-2:n+3,-2:n+3,-2:n+3))
allocate(w(-2:n+3,-2:n+3,-2:n+3),output(-2:n+3,-2:n+3,-2:n+3))
allocate(Prtype(-2:n+3,-2:n+3,-2:n+3))
u=-101
v=-102
w=-103
output=-104
Prtype=-1
free_count=0
do k=1,n
do j=1,n
do i=1,n
if(mod(i+j+k,2)==1)Prtype(i,j,k)=1
if(Prtype(i,j,k)<=0)free_count=free_count+1
enddo
enddo
enddo
n_free_domain=real(n**3,8)
if(masked)n_free_domain=real(max(1,free_count),8)
native_calls=0
write(error_unit,'(A,3(1X,I0))')'SCALAR_CASE_BEGIN',flag,mask_mode,n
call advance(n,1,go,masked)
write(error_unit,'(A,3(1X,I0))')'SCALAR_CASE_END',flag,mask_mode,n
if(native_calls/=flag)error stop 'scalar reduction count differs'
if(any(shape(u)/=[n+6,n+6,n+6]).or.any(shape(v)/=[n+6,n+6,n+6]))error stop 'input shape changed'
if(any(shape(w)/=[n+6,n+6,n+6]).or.any(shape(output)/=[n+6,n+6,n+6]))error stop 'output shape changed'
if(any(lbound(u)/=-2).or.any(lbound(v)/=-2).or.any(lbound(w)/=-2).or.any(lbound(output)/=-2))error stop 'origin changed'
expected_tke=1
if(go)then
expected_tke=0
do k=1,n
do j=1,n
do i=1,n
if(.not.masked.or.Prtype(i,j,k)<=0)then
expected_tke=expected_tke+(u(i,j,k)+u(i-1,j,k)-2*mean(1))**2 &
                         +(v(i,j,k)+v(i,j-1,k)-2*mean(2))**2 &
                         +(w(i,j,k)+w(i,j,k-1)-2*mean(3))**2
endif
enddo
enddo
enddo
expected_tke=expected_tke/8
expected_tke=expected_tke/n_free_domain
endif
do k=-2,n+3
do j=-2,n+3
do i=-2,n+3
interior=i>=1.and.i<=n.and.j>=1.and.j<=n.and.k>=1.and.k<=n
if(interior)then
if(u(i,j,k)/=real(i,8).or.v(i,j,k)/=real(j,8).or.w(i,j,k)/=real(k,8))error stop 'producer field differs'
expected_output=real(i+j+k,8)+expected_tke
if(abs(output(i,j,k)-expected_output)>1e-11_8)error stop 'returned scalar GPU consumer differs'
if(Prtype(i,j,k)/=merge(1,-1,mod(i+j+k,2)==1))error stop 'host mask changed'
else
if(u(i,j,k)/=-101.or.v(i,j,k)/=-102.or.w(i,j,k)/=-103.or.output(i,j,k)/=-104)error stop 'halo changed'
if(Prtype(i,j,k)/=-1)error stop 'mask halo changed'
endif
enddo
enddo
enddo
write(*,'(4(I0,1X),ES26.17E3)')flag,mask_mode,n,native_calls,expected_tke
write(*,'(*(ES26.17E3,1X))')u,v,w,output,mean
write(*,'(*(I0,1X))')Prtype
deallocate(u,v,w,output,Prtype)
enddo
enddo
enddo
print *, 'SCALAR_FIELDS_RETURN_GUARDS_MASK_HALOS_AND_BOUNDS_OK'
end program
'''


@pytest.fixture(scope='module')
def binaries(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(base, 'emit', sealed_emission)
        patch.setattr(base, 'DRIVER', DRIVER)
        directory, candidate, native = base.epoch_binaries.__wrapped__(tmp_path_factory)
    manifest = json.loads((directory / 'scope-manifest.json').read_text())
    provenance = {row['id']: row for row in manifest['runtime_provenance']['records']}
    resources = {row['resource']: row['registration_identity']
                 for row in manifest['scopes'][0]['checked_execution_plan']['storage_epoch']['resources']}
    assert set(resources) == {'forcing::u', 'forcing::v', 'forcing::w', 'forcing::output'}
    return directory, candidate, native, provenance, resources


@pytest.mark.cuda
@pytest.mark.parametrize('mode', ['traced', 'untraced', 'hidden'])
def test_scalar_out_native_reduction_retires_only_when_reached_and_feeds_gpu(binaries, mode):
    directory, candidate, native, provenance, resources = binaries
    env = environment(mode)
    env['FORT_OFFLOAD_TRACE'] = '1'
    reference = base.checked([str(native)], directory, mode + '-native', env)
    result = base.checked([str(candidate)], directory, mode + '-candidate', env)
    assert result.stdout == reference.stdout
    assert 'SCALAR_FIELDS_RETURN_GUARDS_MASK_HALOS_AND_BOUNDS_OK' in result.stdout
    if mode == 'untraced':
        assert 'FORT_SCOPED ' not in result.stderr
        return
    for name in ('COLD', 'ZERO'):
        text, = re.findall('SCALAR_' + name + '_BEGIN\n(.*?)SCALAR_' + name + '_END', result.stderr, re.S)
        assert not any(event['kind'] in {'domain_create', 'register'} for event in events_in(text))
        assert not physical_gpu_events(text)
    if mode == 'hidden':
        assert not physical_gpu_events(result.stderr)
    cases = re.findall(r'SCALAR_CASE_BEGIN (\d+) (\d+) (\d+)\n(.*?)SCALAR_CASE_END \1 \2 \3', result.stderr, re.S)
    assert len(cases) == 8
    ledger = []
    for flag, masked, width, text in cases:
        events = events_in(text)
        launches = [index for index, event in enumerate(events) if event['kind'] == 'launch']
        n, changed = int(width), bool(int(flag))
        expected_regions = [{'resource': name, 'registration_identity': resources[name],
                             'lower': [1, 1, 1], 'extents': [n, n, n], 'bytes': 8 * n**3}
                            for name in ('forcing::u', 'forcing::v', 'forcing::w')]
        ledger.append({'changed': changed, 'masked': bool(int(masked)), 'n': n, 'events': events,
                       'expected_dirty_regions': expected_regions,
                       'upload_bytes': copy_bytes(events, 'upload'), 'download_bytes': copy_bytes(events, 'download')})
        if mode == 'hidden':
            assert not physical_gpu_events(text)
            assert copy_bytes(events, 'upload') == copy_bytes(events, 'download') == 0
            continue
        assert [procedure(events[index], provenance) for index in launches] == ['forcing::producer', 'forcing::consumer']
        first, last = launches
        between = events[first + 1:last]
        if changed:
            assert events[first]['domain'] != events[last]['domain']
            assert sum(event['kind'] == 'domain_retire' for event in between) == 1
            assert copy_bytes(between, 'download') == 3 * 8 * n**3
            assert copy_bytes(between, 'upload') == 3 * 8 * n**3
            for direction in ('upload', 'download'):
                copied = {}
                for event in between:
                    if event['kind'] == direction:
                        identity = int(event['buffer'])
                        copied[identity] = copied.get(identity, 0) + int(event['bytes'])
                assert copied == {row['registration_identity']: row['bytes'] for row in expected_regions}
        else:
            assert events[first]['domain'] == events[last]['domain']
            assert not any(event['kind'] in {'domain_retire', 'domain_create'} for event in between)
            assert copy_bytes(between, 'upload') == copy_bytes(between, 'download') == 0
    (directory / (mode + '-ledger.json')).write_text(json.dumps(ledger, indent=2) + '\n')


def test_native_scalar_driver_checks_complete_fields_and_mask_without_cuda(tmp_path):
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    assert fortran is not None, "Fortran toolchain required for native fixture qualification"
    original = tmp_path / "scalar.f90"
    original.write_text(source())
    driver = tmp_path / "driver.f90"
    driver.write_text(DRIVER)
    native = tmp_path / "native"
    env = environment("hidden")
    base.checked([fortran, "-fopenmp", "-fcheck=all", "-ffp-contract=off", "-J", str(tmp_path),
                  str(original), str(driver), "-o", str(native)], tmp_path, "native-compile", env)
    result = base.checked([str(native)], tmp_path, "native-fields", env)
    assert "SCALAR_FIELDS_RETURN_GUARDS_MASK_HALOS_AND_BOUNDS_OK" in result.stdout
    assert len(re.findall(r"SCALAR_CASE_BEGIN ", result.stderr)) == 8
