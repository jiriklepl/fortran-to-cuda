"""Guarded original damping uses readonly allocated profiles between GPU children."""
import json
import re
import shutil

import pytest

from compiler.tests import test_graph_storage_epochs_cuda as base
from compiler.tests.test_hierarchical_child_regions_cuda import copy_bytes, events_in, procedure
from compiler.tests.test_joined_boundary_lifecycle_cuda import environment
from compiler.tests.test_native_retirement_scalar_formals_cuda import physical_gpu_events
from compiler.tests.test_readonly_record_association_lifetime import SOURCE, fixture


def source():
    text = SOURCE.replace('real(8)::mean(3)=0', 'real(8)::mean(3)=0\ninteger::native_calls=0')
    text = text.replace(',masked', '').replace('::go,masked', '::go')
    original = '''subroutine advance(n,go)
integer,intent(in)::n
logical,intent(in)::go
call middle(n,go)
end subroutine'''
    replacement = '''subroutine advance(n,stages,go)
integer,intent(in)::n,stages
logical,intent(in)::go
integer::stage
if(n<0)return
do stage=1,stages
call middle(n,go)
enddo
end subroutine'''
    assert text.count(original) == 1
    text = text.replace(original, replacement)
    original = '    associate(prop => profile)'
    assert text.count(original) == 1
    return text.replace(original, '    native_calls=native_calls+1\n' + original)


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
              if isinstance(patch, LexicalBoundaryPatch) and patch.conditional_guard == 'go']
    assert patch.conditional_guard.lower() == 'go'
    assert patch.completion is None
    assert patch.lifecycle is not None
    storage = patch.lifecycle.public()
    aliases = storage['fixed_scalar_associations']
    assert len(aliases) == 1
    assert aliases[0]['kind'] == 'original_readonly_scalar_record_lexical_declaration'
    assert aliases[0]['declaration_only']
    assert aliases[0]['layout_authorized'] is False
    scope, = report['scopes']
    assert {name.split('#')[0] for name in scope['gpu_leaves']} == {'forcing::producer', 'forcing::consumer'}
    assert all(row['resource'] != 'forcing::profile' for row in scope['checked_execution_plan']['storage_epoch']['resources'])
    assert 'forcing::profile' not in builder.facts['captures']
    assert not storage['frontiers']
    assert not storage['escapes']
    package = directory / 'source-package'
    for name, text in outputs.items():
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    manifest = json.loads((package / 'scope-manifest.json').read_text())
    validate_checked_execution_plans(manifest)
    _scope_package_build({'artifact_root': str(package), 'scope_manifest': manifest})
    assert builder.finish()
    (directory / 'damping-boundary-proof.json').write_text(json.dumps({
        'completion': None, 'storage': storage,
        'conditional_guard': patch.conditional_guard,
        'return_contract': scope['checked_execution_plan']['storage_epoch']['return_contract'],
    }, indent=2) + '\n')
    return original, outputs, manifest


def test_readonly_damping_source_keeps_original_guard_and_strict_transport(tmp_path):
    sealed_emission(tmp_path)


DRIVER = '''program exercise
use forcing
use parameters
use iso_fortran_env,only:error_unit
implicit none
integer::flag,profile_mode,case_index,n,i,j,k
logical::go,interior
real(8)::eu,ev,ew,eo,scale
native_calls=0
write(error_unit,'(A)')'DAMP_COLD_BEGIN'
call advance(-1,1,.true.)
write(error_unit,'(A)')'DAMP_COLD_END'
if(allocated(u).or.allocated(v).or.allocated(w).or.allocated(output))error stop 'cold touched fields'
if(allocated(u2).or.allocated(v2).or.allocated(w2))error stop 'cold touched tendencies'
if(allocated(profile%coefficient).or.allocated(profile%coefficientW))error stop 'cold touched profiles'
if(allocated(profile%u_prof).or.allocated(profile%v_prof).or.allocated(profile%w_prof))error stop 'cold touched profiles'
if(native_calls/=0)error stop 'cold called damping'
write(error_unit,'(A)')'DAMP_ZERO_BEGIN'
call advance(7,0,.true.)
write(error_unit,'(A)')'DAMP_ZERO_END'
if(allocated(u).or.allocated(v).or.allocated(w).or.allocated(output))error stop 'zero touched fields'
if(allocated(u2).or.allocated(v2).or.allocated(w2))error stop 'zero touched tendencies'
if(allocated(profile%coefficient).or.allocated(profile%coefficientW))error stop 'zero touched profiles'
if(allocated(profile%u_prof).or.allocated(profile%v_prof).or.allocated(profile%w_prof))error stop 'zero touched profiles'
if(native_calls/=0)error stop 'zero called damping'
do flag=0,1
go=flag==1
do profile_mode=0,1
scale=real(profile_mode+1,8)
do case_index=1,2
n=1
if(case_index==2)n=7
Unx=n
Uny=n
Vnx=n
Vny=n
Wnx=n
Wny=n
allocate(u(-2:n+3,-2:n+3,-2:n+3),v(-2:n+3,-2:n+3,-2:n+3))
allocate(w(-2:n+3,-2:n+3,-2:n+3),output(-2:n+3,-2:n+3,-2:n+3))
allocate(u2(-2:n+3,-2:n+3,-2:n+3),v2(-2:n+3,-2:n+3,-2:n+3),w2(-2:n+3,-2:n+3,-2:n+3))
u=-101
v=-102
w=-103
u2=-111
v2=-112
w2=-113
output=-104
do k=1,n
do j=1,n
do i=1,n
u(i,j,k)=real(i+2*j+3*k,8)/8
v(i,j,k)=real(2*i-j+4*k,8)/16
w(i,j,k)=real(3*i+2*j-k,8)/32
enddo
enddo
enddo
profile%bottom=-41
profile%top=42
profile%kbottom=1
profile%ktop=n
profile%kWbottom=1
profile%kWtop=n
if(go)then
allocate(profile%coefficient(0:n+1),profile%coefficientW(0:n+1))
allocate(profile%u_prof(0:n+1),profile%v_prof(0:n+1),profile%w_prof(0:n+1))
do k=0,n+1
profile%coefficient(k)=scale/8
profile%coefficientW(k)=scale/16
profile%u_prof(k)=real(profile_mode,8)/2+real(k,8)/16
profile%v_prof(k)=-real(profile_mode,8)/4+real(k,8)/8
profile%w_prof(k)=3*real(profile_mode,8)/4-real(k,8)/32
enddo
endif
native_calls=0
write(error_unit,'(A,3(1X,I0))')'DAMP_CASE_BEGIN',flag,profile_mode,n
call advance(n,1,go)
write(error_unit,'(A,3(1X,I0))')'DAMP_CASE_END',flag,profile_mode,n
if(native_calls/=flag)error stop 'original damping count differs'
if(any(shape(u)/=[n+6,n+6,n+6]).or.any(shape(v)/=[n+6,n+6,n+6]))error stop 'velocity shape changed'
if(any(shape(w)/=[n+6,n+6,n+6]).or.any(shape(output)/=[n+6,n+6,n+6]))error stop 'output shape changed'
if(any(shape(u2)/=[n+6,n+6,n+6]).or.any(shape(v2)/=[n+6,n+6,n+6]))error stop 'tendency shape changed'
if(any(shape(w2)/=[n+6,n+6,n+6]))error stop 'tendency shape changed'
if(any(lbound(u)/=-2).or.any(lbound(v)/=-2).or.any(lbound(w)/=-2))error stop 'velocity origin changed'
if(any(lbound(u2)/=-2).or.any(lbound(v2)/=-2).or.any(lbound(w2)/=-2))error stop 'tendency origin changed'
if(any(lbound(output)/=-2))error stop 'output origin changed'
do k=-2,n+3
do j=-2,n+3
do i=-2,n+3
interior=i>=1.and.i<=n.and.j>=1.and.j<=n.and.k>=1.and.k<=n
if(interior)then
if(u(i,j,k)/=real(i+2*j+3*k,8)/8)error stop 'U changed'
if(v(i,j,k)/=real(2*i-j+4*k,8)/16)error stop 'V changed'
if(w(i,j,k)/=real(3*i+2*j-k,8)/32)error stop 'W changed'
eu=real(i,8)
ev=real(j,8)
ew=real(k,8)
if(go)then
eu=eu-scale/8*(u(i,j,k)-(real(profile_mode,8)/2+real(k,8)/16))
ev=ev-scale/8*(v(i,j,k)-(-real(profile_mode,8)/4+real(k,8)/8))
ew=ew-scale/16*(w(i,j,k)-(3*real(profile_mode,8)/4-real(k,8)/32))
endif
eo=eu+ev+ew
if(abs(u2(i,j,k)-eu)>1e-11_8.or.abs(v2(i,j,k)-ev)>1e-11_8)error stop 'damped UV differs'
if(abs(w2(i,j,k)-ew)>1e-11_8)error stop 'damped W differs'
if(abs(output(i,j,k)-eo)>1e-11_8)error stop 'damped GPU consumer differs'
else
if(u(i,j,k)/=-101.or.v(i,j,k)/=-102.or.w(i,j,k)/=-103)error stop 'velocity halo changed'
if(u2(i,j,k)/=-111.or.v2(i,j,k)/=-112.or.w2(i,j,k)/=-113)error stop 'tendency halo changed'
if(output(i,j,k)/=-104)error stop 'output halo changed'
endif
enddo
enddo
enddo
if(profile%bottom/=-41.or.profile%top/=42)error stop 'profile scalar changed'
if(profile%kbottom/=1.or.profile%ktop/=n.or.profile%kWbottom/=1.or.profile%kWtop/=n)error stop 'profile bounds changed'
if(allocated(profile%temperature_prof).or.allocated(profile%moisture_prof))error stop 'unused profile allocated'
if(go)then
if(lbound(profile%coefficient,1)/=0.or.ubound(profile%coefficient,1)/=n+1)error stop 'coefficient bounds changed'
if(lbound(profile%coefficientW,1)/=0.or.ubound(profile%coefficientW,1)/=n+1)error stop 'coefficientW bounds changed'
if(lbound(profile%u_prof,1)/=0.or.ubound(profile%u_prof,1)/=n+1)error stop 'U profile bounds changed'
if(lbound(profile%v_prof,1)/=0.or.ubound(profile%v_prof,1)/=n+1)error stop 'V profile bounds changed'
if(lbound(profile%w_prof,1)/=0.or.ubound(profile%w_prof,1)/=n+1)error stop 'W profile bounds changed'
do k=0,n+1
if(profile%coefficient(k)/=scale/8.or.profile%coefficientW(k)/=scale/16)error stop 'coefficient changed'
if(profile%u_prof(k)/=real(profile_mode,8)/2+real(k,8)/16)error stop 'U profile changed'
if(profile%v_prof(k)/=-real(profile_mode,8)/4+real(k,8)/8)error stop 'V profile changed'
if(profile%w_prof(k)/=3*real(profile_mode,8)/4-real(k,8)/32)error stop 'W profile changed'
enddo
else
if(allocated(profile%coefficient).or.allocated(profile%coefficientW))error stop 'false branch allocated coefficient'
if(allocated(profile%u_prof).or.allocated(profile%v_prof).or.allocated(profile%w_prof))error stop 'false branch allocated profile'
endif
write(*,'(4(I0,1X))')flag,profile_mode,n,native_calls
write(*,'(*(ES26.17E3,1X))')u,v,w,u2,v2,w2,output
if(go)then
write(*,'(*(ES26.17E3,1X))')profile%coefficient,profile%coefficientW,profile%u_prof,profile%v_prof,profile%w_prof
deallocate(profile%coefficient,profile%coefficientW,profile%u_prof,profile%v_prof,profile%w_prof)
endif
deallocate(u,v,w,u2,v2,w2,output)
enddo
enddo
enddo
print *, 'READONLY_DAMPING_FIELDS_PROFILES_GUARDS_HALOS_AND_BOUNDS_OK'
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
    assert set(resources) == {'forcing::u2', 'forcing::v2', 'forcing::w2', 'forcing::output'}
    return directory, candidate, native, provenance, resources


@pytest.mark.cuda
@pytest.mark.parametrize('mode', ['traced', 'untraced', 'hidden'])
def test_guarded_readonly_profiles_preserve_native_damping_once_and_gpu_consumer(binaries, mode):
    directory, candidate, native, provenance, resources = binaries
    env = environment(mode)
    env['FORT_OFFLOAD_TRACE'] = '1'
    reference = base.checked([str(native)], directory, mode + '-native', env)
    result = base.checked([str(candidate)], directory, mode + '-candidate', env)
    assert result.stdout == reference.stdout
    assert 'READONLY_DAMPING_FIELDS_PROFILES_GUARDS_HALOS_AND_BOUNDS_OK' in result.stdout
    if mode == 'untraced':
        assert 'FORT_SCOPED ' not in result.stderr
        return
    for name in ('COLD', 'ZERO'):
        text, = re.findall('DAMP_' + name + '_BEGIN\n(.*?)DAMP_' + name + '_END', result.stderr, re.S)
        assert not any(event['kind'] in {'domain_create', 'register'} for event in events_in(text))
        assert not physical_gpu_events(text)
    if mode == 'hidden':
        assert not physical_gpu_events(result.stderr)
    cases = re.findall(r'DAMP_CASE_BEGIN (\d+) (\d+) (\d+)\n(.*?)DAMP_CASE_END \1 \2 \3', result.stderr, re.S)
    assert len(cases) == 8
    ledger = []
    for flag, profile_mode, width, text in cases:
        events = events_in(text)
        launches = [index for index, event in enumerate(events) if event['kind'] == 'launch']
        n, changed = int(width), bool(int(flag))
        expected_regions = [{'resource': name, 'registration_identity': resources[name],
                             'lower': [1, 1, 1], 'extents': [n, n, n], 'bytes': 8 * n**3}
                            for name in ('forcing::u2', 'forcing::v2', 'forcing::w2')]
        ledger.append({'changed': changed, 'profile_mode': int(profile_mode), 'n': n, 'events': events,
                       'expected_dirty_regions': expected_regions,
                       'upload_bytes': copy_bytes(events, 'upload'), 'download_bytes': copy_bytes(events, 'download')})
        if mode == 'hidden':
            assert not physical_gpu_events(text)
            assert copy_bytes(events, 'upload') == copy_bytes(events, 'download') == 0
            continue
        assert [procedure(events[index], provenance) for index in launches] == ['forcing::producer', 'forcing::consumer']
        first, last = launches
        between = events[first + 1:last]
        assert copy_bytes(events, 'upload') == (3 * 8 * n**3 if changed else 0)
        assert copy_bytes(events, 'download') == 4 * 8 * n**3
        all_downloads = {}
        for event in events:
            if event['kind'] == 'download':
                identity = int(event['buffer'])
                all_downloads[identity] = all_downloads.get(identity, 0) + int(event['bytes'])
        assert all_downloads == {identity: 8 * n**3 for identity in resources.values()}
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


def test_native_damping_driver_checks_all_fields_and_unallocated_profiles(tmp_path):
    fortran = shutil.which('gfortran-15') or shutil.which('gfortran')
    assert fortran is not None, 'Fortran toolchain required for native fixture qualification'
    original = tmp_path / 'damping.f90'
    original.write_text(source())
    driver = tmp_path / 'driver.f90'
    driver.write_text(DRIVER)
    native = tmp_path / 'native'
    env = environment('hidden')
    base.checked([fortran, '-fopenmp', '-fcheck=all', '-ffp-contract=off', '-J', str(tmp_path),
                  str(original), str(driver), '-o', str(native)], tmp_path, 'native-compile', env)
    result = base.checked([str(native)], tmp_path, 'native-fields', env)
    assert 'READONLY_DAMPING_FIELDS_PROFILES_GUARDS_HALOS_AND_BOUNDS_OK' in result.stdout
    assert len(re.findall(r'DAMP_CASE_BEGIN ', result.stderr)) == 8
