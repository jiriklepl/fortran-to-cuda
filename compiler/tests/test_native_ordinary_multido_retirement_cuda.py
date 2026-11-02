"""Original shared FL and TARGET field reads join between two GPU children."""
import json
import re
import shutil

import pytest

from compiler.tests import test_graph_storage_epochs_cuda as base
from compiler.tests import test_native_ordinary_join_retirement as source_fixture
from compiler.tests.test_hierarchical_child_regions_cuda import copy_bytes, events_in, procedure
from compiler.tests.test_joined_boundary_lifecycle_cuda import environment
from compiler.tests.test_native_retirement_scalar_formals_cuda import physical_gpu_events
from compiler.tests.test_native_target_record_storage_reads import source as target_source

PASSES = 18


def fixture(directory, **kwargs):
    # Reuse the existing builder/include fixture; only locate its nested group.
    grouped = source_fixture.grouped_original_sequence
    def guarded_groups(nodes, pragmas):
        yield from grouped(nodes, pragmas)
        for node in nodes:
            if type(node).__name__ == 'If_Construct':
                yield from guarded_groups(node.content[1:-1], pragmas)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(source_fixture, 'grouped_original_sequence', guarded_groups)
        return source_fixture.fixture(directory, **kwargs)


def source():
    text = target_source()
    text = text.replace('real(8),allocatable::u(:,:,:),a(:,:,:),b(:,:,:)',
        'real(8),allocatable,target::u(:,:,:),a(:,:,:),b(:,:,:)\ninteger::native_calls=0,native_joins=0')
    text = text.replace('real(8)::fluxp\nend type', 'real(8)::fluxp\ninteger::tag=-31415\nend type')
    original = '''subroutine advance(n)
integer,intent(in)::n
call producer(a,u,n)
call diffusion(a,u,n)
call consumer(b,a,n)
end subroutine'''
    replacement = '''subroutine advance(n,stages,go)
integer,intent(in)::n,stages
logical,intent(in)::go
integer::stage
if(n<0)return
do stage=1,stages
call producer(a,u,n)
call diffusion(a,u,n,go)
call consumer(b,a,n)
enddo
end subroutine'''
    assert text.count(original) == 1
    text = text.replace(original, replacement)
    start = text.index('!$omp do schedule(runtime) collapse(3)')
    end = text.index('!$omp do\ndo ip=1,size(points)', start)
    body = text[start:end]
    text = text[:start] + body * PASSES + text[end:]
    text = text.replace('subroutine diffusion(a,u,n)\ninteger,intent(in)::n',
        'subroutine diffusion(a,u,n,go)\ninteger,intent(in)::n\nlogical,intent(in)::go')
    text = text.replace('if(.not.allocated(fl))', 'native_calls=native_calls+1\nif(go)then\nif(.not.allocated(fl))')
    text = text.replace('!$omp end parallel', '!$omp end parallel\nnative_joins=native_joins+1\nendif')
    assert all(line == line.rstrip() for line in text.splitlines())
    return text


def sealed_emission(directory):
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans
    from elmm_pipeline.local_adapter import _scope_package_build

    from compiler.scopes.lexical import LexicalBoundaryPatch

    original, builder, _routine, _group = fixture(directory, source=source(), include=True)
    outputs, report = builder.run()
    assert report['scope_count'] == 1, report.get('graph_owner_attempt')
    owner = builder._conditional_return_draft.validate().owner
    member = owner.members['scratch_team::diffusion']
    patch, = [p for _first, _last, p in member.patches
              if isinstance(p, LexicalBoundaryPatch) and p.completion is not None]
    completion = patch.completion
    assert completion._ordinary_join is not None
    assert completion._call_storage is None
    assert patch.lifecycle._native_completion is completion
    work = dict(completion._graph.source_work)
    assert work['statements'] == 258 > builder.analysis.operation_limit == 256
    assert work['syntax_nodes'] < 65536
    storage = patch.lifecycle.public()
    assert storage['complete']
    assert not storage['escapes']
    assert storage['native_lifetime_context']['whole_original_group_retirement_required']
    scope, = report['scopes']
    assert {n.split('#')[0] for n in scope['gpu_leaves']} == {'scratch_team::producer', 'scratch_team::consumer'}
    resources = {r['resource'] for r in scope['checked_execution_plan']['storage_epoch']['resources']}
    assert resources == {'scratch_team::a', 'scratch_team::u', 'scratch_team::b'}
    assert set(builder.facts['captures']) == resources
    package = directory / 'source-package'
    for name, content in outputs.items():
        target = package / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    manifest = json.loads((package / 'scope-manifest.json').read_text())
    validate_checked_execution_plans(manifest)
    _scope_package_build({'artifact_root': str(package), 'scope_manifest': manifest})
    transformed = outputs[report['sources'][str(original)]['replacement']]
    assert transformed.count('#include "divergence.inc"') == 1
    assert transformed.count('!$omp parallel private(i,j,k)') == 1
    assert transformed.count('!$omp end parallel') == 1
    assert builder.finish()
    (directory / 'joined-proof.json').write_text(json.dumps({
        'source_work': work, 'completion': completion.public(), 'storage': storage,
        'canonical_resources': sorted(resources), 'raw_include_preserved_once': True}, indent=2) + '\n')
    return original, outputs, manifest


def test_shared_local_target_record_source_and_strict_reader(tmp_path):
    sealed_emission(tmp_path)


DRIVER = '''program exercise
use scratch_team
use iso_fortran_env,only:error_unit
use omp_lib,only:omp_get_max_threads
implicit none
integer::flag,variant,case_index,n,i,j,k,ip
logical::go,interior
real(8)::expected_a,delta
real(8),pointer::alias_a(:,:,:)
if(omp_get_max_threads()/=4)error stop 'four threads required'
native_calls=0
native_joins=0
write(error_unit,'(A)')'FL_COLD_BEGIN'
call advance(-1,1,.true.)
write(error_unit,'(A)')'FL_COLD_END'
if(allocated(a).or.allocated(u).or.allocated(b).or.allocated(points))error stop 'cold touched storage'
if(native_calls/=0.or.native_joins/=0)error stop 'cold called native team'
write(error_unit,'(A)')'FL_ZERO_BEGIN'
call advance(7,0,.true.)
write(error_unit,'(A)')'FL_ZERO_END'
if(allocated(a).or.allocated(u).or.allocated(b).or.allocated(points))error stop 'zero touched storage'
if(native_calls/=0.or.native_joins/=0)error stop 'zero called native team'
do flag=0,1
go=flag==1
do variant=0,1
delta=real(variant+1,8)/4
do case_index=1,2
n=1
if(case_index==2)n=7
allocate(a(-2:n+3,-2:n+3,-2:n+3),u(-2:n+3,-2:n+3,-2:n+3))
allocate(b(-2:n+3,-2:n+3,-2:n+3))
a=-111
b=-222
do k=-2,n+3
do j=-2,n+3
do i=-2,n+3
u(i,j,k)=real(i+j+k,8)**2
enddo
enddo
enddo
alias_a=>a
if(go)then
allocate(points(1:n))
do ip=1,n
points(ip)%xi=ip
points(ip)%yj=1
points(ip)%zk=1
points(ip)%fluxp=delta
points(ip)%tag=-31415
enddo
endif
native_calls=0
native_joins=0
write(error_unit,'(A,3(1X,I0))')'FL_CASE_BEGIN',flag,variant,n
call advance(n,1,go)
write(error_unit,'(A,3(1X,I0))')'FL_CASE_END',flag,variant,n
if(native_calls/=1.or.native_joins/=flag)error stop 'original call replay or missing team join'
if(.not.associated(alias_a,a))error stop 'real alias association changed'
if(any(shape(a)/=[n+6,n+6,n+6]).or.any(shape(u)/=[n+6,n+6,n+6]))error stop 'shape changed'
if(any(shape(b)/=[n+6,n+6,n+6]).or.any(shape(alias_a)/=[n+6,n+6,n+6]))error stop 'alias shape changed'
if(any(lbound(a)/=-2).or.any(lbound(u)/=-2).or.any(lbound(b)/=-2))error stop 'origin changed'
if(any(lbound(alias_a)/=-2))error stop 'alias origin changed'
do k=-2,n+3
do j=-2,n+3
do i=-2,n+3
interior=i>=1.and.i<=n.and.j>=1.and.j<=n.and.k>=1.and.k<=n
if(u(i,j,k)/=real(i+j+k,8)**2)error stop 'velocity or halo changed'
if(interior)then
expected_a=real(i+j+k,8)
if(go)then
expected_a=expected_a+36
if(j==1.and.k==1)expected_a=expected_a+delta
endif
if(a(i,j,k)/=expected_a.or.alias_a(i,j,k)/=expected_a)error stop 'shared FL or real alias differs'
if(b(i,j,k)/=2*expected_a)error stop 'GPU consumer differs'
else
if(a(i,j,k)/=-111.or.alias_a(i,j,k)/=-111.or.b(i,j,k)/=-222)error stop 'field or alias halo changed'
endif
enddo
enddo
enddo
if(go)then
if(lbound(points,1)/=1.or.ubound(points,1)/=n)error stop 'record descriptor changed'
do ip=1,n
if(points(ip)%xi/=ip.or.points(ip)%yj/=1.or.points(ip)%zk/=1)error stop 'record index changed'
if(points(ip)%fluxp/=delta.or.points(ip)%tag/=-31415)error stop 'readonly record changed'
enddo
else
if(allocated(points))error stop 'false branch allocated record'
endif
write(*,'(5(I0,1X))')flag,variant,n,native_calls,native_joins
write(*,'(*(ES26.17E3,1X))')a,u,b,alias_a
if(go)then
do ip=1,n
write(*,'(4(I0,1X),ES26.17E3)')points(ip)%xi,points(ip)%yj,points(ip)%zk,points(ip)%tag,points(ip)%fluxp
enddo
deallocate(points)
endif
nullify(alias_a)
deallocate(a,u,b)
enddo
enddo
enddo
print *, 'JOINED_FL_TARGET_FIELDS_ALIAS_HALOS_BOUNDS_AND_ONCE_OK'
end program
'''


@pytest.fixture(scope='module')
def binaries(tmp_path_factory):
    original_checked = base.checked
    def emit(directory):
        return sealed_emission(directory)
    def checked(command, build, stem, env=None):
        if 'gfortran' in str(command[0]):
            command = [command[0], '-cpp', '-I', str(build.parent), *command[1:]]
        return original_checked(command, build, stem, env)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(base, 'emit', emit)
        patch.setattr(base, 'DRIVER', DRIVER)
        patch.setattr(base, 'checked', checked)
        directory, candidate, native = base.epoch_binaries.__wrapped__(tmp_path_factory)
    manifest = json.loads((directory / 'scope-manifest.json').read_text())
    provenance = {r['id']: r for r in manifest['runtime_provenance']['records']}
    resources = {r['resource']: r['registration_identity']
        for r in manifest['scopes'][0]['checked_execution_plan']['storage_epoch']['resources']}
    return directory, candidate, native, provenance, resources


@pytest.mark.cuda
@pytest.mark.parametrize('mode', ['traced', 'untraced', 'hidden'])
def test_original_multido_shared_fl_and_target_record_reaches_gpu_consumer(binaries, mode):
    directory, candidate, native, provenance, resources = binaries
    env = environment(mode)
    env['FORT_OFFLOAD_TRACE'] = '1'
    reference = base.checked([str(native)], directory, mode + '-native', env)
    result = base.checked([str(candidate)], directory, mode + '-candidate', env)
    assert result.stdout == reference.stdout
    assert 'JOINED_FL_TARGET_FIELDS_ALIAS_HALOS_BOUNDS_AND_ONCE_OK' in result.stdout
    if mode == 'untraced':
        assert 'FORT_SCOPED ' not in result.stderr
        return
    for name in ('COLD', 'ZERO'):
        text, = re.findall('FL_' + name + '_BEGIN\n(.*?)FL_' + name + '_END', result.stderr, re.S)
        assert not any(e['kind'] in {'domain_create', 'register'} for e in events_in(text))
        assert not physical_gpu_events(text)
    if mode == 'hidden':
        assert not physical_gpu_events(result.stderr)
    cases = re.findall(r'FL_CASE_BEGIN (\d+) (\d+) (\d+)\n(.*?)FL_CASE_END \1 \2 \3', result.stderr, re.S)
    assert len(cases) == 8
    ledger = []
    for flag, variant, width, text in cases:
        events = events_in(text)
        n, changed = int(width), bool(int(flag))
        ledger.append({'changed': changed, 'variant': int(variant), 'n': n, 'events': events,
            'upload_bytes': copy_bytes(events, 'upload'), 'download_bytes': copy_bytes(events, 'download')})
        if mode == 'hidden':
            assert not physical_gpu_events(text)
            assert copy_bytes(events, 'upload') == copy_bytes(events, 'download') == 0
            continue
        launches = [i for i, e in enumerate(events) if e['kind'] == 'launch']
        assert [procedure(events[i], provenance) for i in launches] == ['scratch_team::producer', 'scratch_team::consumer']
        first, last = launches
        between = events[first + 1:last]
        assert copy_bytes(events, 'upload') == (8 * n**3 if changed else 0)
        assert copy_bytes(events, 'download') == 3 * 8 * n**3
        all_downloads = {}
        for event in events:
            if event['kind'] == 'download':
                identity = int(event['buffer'])
                all_downloads[identity] = all_downloads.get(identity, 0) + int(event['bytes'])
        assert all_downloads == {identity: 8 * n**3 for identity in resources.values()}
        if changed:
            assert events[first]['domain'] != events[last]['domain']
            assert sum(e['kind'] == 'domain_retire' for e in between) == 1
            assert copy_bytes(between, 'download') == 2 * 8 * n**3
            assert copy_bytes(between, 'upload') == 8 * n**3
            for direction, names in [('download', ['scratch_team::a', 'scratch_team::u']), ('upload', ['scratch_team::a'])]:
                copied = {}
                for event in between:
                    if event['kind'] == direction:
                        identity = int(event['buffer'])
                        copied[identity] = copied.get(identity, 0) + int(event['bytes'])
                assert copied == {resources[name]: 8 * n**3 for name in names}
        else:
            assert events[first]['domain'] == events[last]['domain']
            assert not any(e['kind'] in {'domain_retire', 'domain_create'} for e in between)
            assert copy_bytes(between, 'upload') == copy_bytes(between, 'download') == 0
    (directory / (mode + '-ledger.json')).write_text(json.dumps(ledger, indent=2) + '\n')


def test_native_shared_fl_driver_checks_complete_fields_alias_and_records(tmp_path):
    compiler = shutil.which('gfortran-15') or shutil.which('gfortran')
    assert compiler is not None
    original, _builder, _routine, _group = fixture(tmp_path, source=source(), include=True)
    driver = tmp_path / 'driver.f90'
    driver.write_text(DRIVER)
    assert all(line == line.rstrip() and len(line) <= 132 for line in DRIVER.splitlines())
    native = tmp_path / 'native'
    env = environment('hidden')
    base.checked([compiler, '-cpp', '-I', str(tmp_path), '-fopenmp', '-fcheck=all', '-ffp-contract=off',
        '-J', str(tmp_path), str(original), str(driver), '-o', str(native)], tmp_path, 'native-compile', env)
    result = base.checked([str(native)], tmp_path, 'native-fields', env)
    assert 'JOINED_FL_TARGET_FIELDS_ALIAS_HALOS_BOUNDS_AND_ONCE_OK' in result.stdout
    assert len(re.findall(r'FL_CASE_BEGIN ', result.stderr)) == 8
