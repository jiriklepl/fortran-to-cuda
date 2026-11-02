"""Original CPU array reductions retain device replicas and opaque BYTE storage."""
import json
import re
import shutil
import subprocess
from hashlib import sha256
from pathlib import Path

import pytest

from compiler.driver.options import CompilerOptions
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.tests import test_graph_storage_epochs_cuda as base
from compiler.tests.test_hierarchical_child_regions_cuda import copy_bytes, events_in, procedure
from compiler.tests.test_joined_boundary_lifecycle_cuda import environment


def source():
    def loops(masked):
        return '\n'.join(f'''!$omp do collapse(3)
do k=4,5
do j=-1,0
do i=1,n
{'if(values(i,j,k)>0) ' if masked else ''}result({c})=result({c})+{c}*values(i,j,k)
enddo
enddo
enddo''' for c in (1, 2, 3))

    return '''module opaque_chain
implicit none
real(8),allocatable::values(:,:,:),output(:,:,:)
real(8)::mean(-1:1),seen_a,seen_b,seen_c
contains
subroutine advance(n,steps,go,masked)
integer,intent(in)::n,steps
logical,intent(in)::go,masked
integer::stage
do stage=1,steps
call producer(values,n)
if(go) call reduce_mean(mean,values,n,masked)
if(go) seen_a=mean(-1)
if(go) seen_b=mean(0)
if(go) seen_c=mean(1)
call consumer(output,values,n)
enddo
end subroutine
subroutine producer(a,n)
integer,intent(in)::n
real(8),intent(inout)::a(-2:,-1:,4:)
integer::i,j,k
do k=4,5
do j=-1,0
do i=1,n
a(i,j,k)=real(i-2+8*j+16*(k-4),8)
enddo
enddo
enddo
end subroutine
subroutine consumer(a,b,n)
integer,intent(in)::n
real(8),intent(inout)::a(-2:,-1:,4:)
real(8),intent(in)::b(-2:,-1:,4:)
integer::i,j,k
do k=4,5
do j=-1,0
do i=1,n
a(i,j,k)=2.0_8*b(i,j,k)
enddo
enddo
enddo
end subroutine
subroutine reduce_mean(result,values,n,masked)
real(8),intent(out)::result(3)
real(8),contiguous,intent(in)::values(-2:,-1:,4:)
integer,intent(in)::n
logical,intent(in)::masked
integer::i,j,k
result=0
if(masked)then
!$omp parallel reduction(+:result)
''' + loops(True) + '''
!$omp end parallel
else
!$omp parallel reduction(+:result)
''' + loops(False) + '''
!$omp end parallel
endif
result=result/max(1,n)
end subroutine
end module
'''


def sealed_emission(directory):
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans
    from elmm_pipeline.local_adapter import _scope_package_build

    directory.mkdir(parents=True, exist_ok=True)
    original = directory / 'original.f90'
    original.write_text(source())
    fact = {'storage': 'stable', 'initialized': 'whole', 'escapes': False, 'allocation_changes': False}
    facts = {'schema_version': 1, 'participation': 'serial', 'sources': {str(original): sha256(original.read_bytes()).hexdigest()},
             'captures': {'opaque_chain::' + name: dict(fact) for name in ('values', 'output')}}
    builder = ScopeBuilder([original], 'opaque_chain::advance', facts=facts, options=CompilerOptions(),
        config=OffloadConfig(policy='sections', scope_execution='graph'))
    outputs, report = builder.run()
    (directory / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    assert report['scope_count'] == 1, report.get('graph_owner_attempt')
    scope, = report['scopes']
    assert not scope['boundaries'], scope['boundaries']
    assert {p.split('#')[0] for p in scope['gpu_leaves']} == {'opaque_chain::producer', 'opaque_chain::consumer'}
    plan = scope['checked_execution_plan']
    writes, reads = [], []
    for proc in plan['procedures']:
        for unit in proc['units']:
            for operation in unit['operations']['native_operations']:
                for access in (operation.get('native_opaque_accesses') or {}).get('accesses', []):
                    (reads if access['mode'] == 'R' else writes).append(access)
    assert len(writes) == 1
    assert writes[0]['resource'] == 'opaque_chain::mean'
    assert writes[0]['permit']['bounds'] == [-1, 1]
    assert reads
    opaque = next(r for r in plan['storage_epoch']['resources'] if r['resource'] == 'opaque_chain::mean')
    assert opaque['device_replica_authorized'] is opaque['initialization_authorized'] is False
    assert all('reduce_mean' not in p['procedure'] for p in plan['procedures'])
    package = directory / 'source-package'
    package.mkdir()
    for name, data in outputs.items():
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data)
    manifest = json.loads((package / 'scope-manifest.json').read_text())
    validate_checked_execution_plans(manifest)
    _scope_package_build({'artifact_root': str(package), 'scope_manifest': manifest, 'tool': 'local', 'memory_model': 'scoped'})
    return original, outputs, manifest


def test_original_reduction_and_opaque_read_seal_one_strict_package(tmp_path):
    sealed_emission(tmp_path)


DRIVER = '''program exercise
use opaque_chain
use iso_fortran_env,only:error_unit
use omp_lib,only:omp_get_max_threads
implicit none
integer::n,case_index,go_mode,mask_mode,i,j,k
real(8)::v,total,expected(3)
real(8),allocatable::ev(:,:,:),eo(:,:,:)
logical::go,masked
character(16)::argument
if(omp_get_max_threads()/=4)error stop 'four threads required'
call get_command_argument(1,argument)
if(trim(argument)=='fault')then
allocate(values(-2:10,-1:2,4:7),output(-2:10,-1:2,4:7))
values=-99
output=-101
mean=-77
call advance(7,2,.true.,.false.)
error stop 'fatal native preparation returned'
endif
write(error_unit,'(A)')'OPAQUE_ZERO_BEGIN'
call advance(257,0,.true.,.true.)
write(error_unit,'(A)')'OPAQUE_ZERO_END'
if(allocated(values).or.allocated(output))error stop 'zero trip touched descriptors'
do go_mode=0,1
do mask_mode=0,1
go=go_mode==1
masked=mask_mode==1
do case_index=1,3
select case(case_index)
case(1)
n=1
case(2)
n=7
case default
n=257
end select
allocate(values(-2:n+3,-1:2,4:7),output(-2:n+3,-1:2,4:7),ev(-2:n+3,-1:2,4:7),eo(-2:n+3,-1:2,4:7))
values=-99
output=-101
ev=values
eo=output
mean=-77
seen_a=-78
seen_b=-79
seen_c=-80
total=0
do k=4,5
do j=-1,0
do i=1,n
v=real(i-2+8*j+16*(k-4),8)
ev(i,j,k)=v
eo(i,j,k)=2*v
if(.not.masked.or.v>0)total=total+v
enddo
enddo
enddo
expected=[total,2*total,3*total]/real(max(1,n),8)
write(error_unit,'(A,3(1X,I0))')'OPAQUE_BEGIN',go_mode,mask_mode,n
call advance(n,2,go,masked)
write(error_unit,'(A,3(1X,I0))')'OPAQUE_END',go_mode,mask_mode,n
if(any(lbound(values)/=[-2,-1,4]).or.any(ubound(values)/=[n+3,2,7]))error stop 'input bounds changed'
if(any(lbound(output)/=[-2,-1,4]).or.any(ubound(output)/=[n+3,2,7]))error stop 'output bounds changed'
if(any(values/=ev).or.any(output/=eo))error stop 'full fields or halos differ'
if(go)then
if(any(mean/=expected).or.any([seen_a,seen_b,seen_c]/=expected))error stop 'native reduction or following opaque read differs'
else
if(any(mean/=-77).or.any([seen_a,seen_b,seen_c]/=[-78.0_8,-79.0_8,-80.0_8]))error stop 'untaken native call executed'
endif
write(*,'(9(I0,1X))')go_mode,mask_mode,n,lbound(values),ubound(values)
write(*,'(*(ES26.17E3,1X))')mean,seen_a,seen_b,seen_c,values,output
deallocate(values,output,ev,eo)
enddo
enddo
enddo
print *, 'OPAQUE_ORIGINAL_REDUCTION_FIELDS_HALOS_AND_BOUNDS_OK'
end program
'''

OBSERVER = r'''#include <cstdio>
extern "C" void __opaque_chain_MOD_reduce_mean();
extern "C" void __opaque_chain_MOD_producer();
extern "C" void __opaque_chain_MOD_consumer();
extern "C" __attribute__((no_instrument_function)) void __cyg_profile_func_enter(void* f,void*) {
 if(f==reinterpret_cast<void*>(&__opaque_chain_MOD_reduce_mean)) std::fputs("OPAQUE_ORIGINAL_REDUCTION\n",stderr);
 if(f==reinterpret_cast<void*>(&__opaque_chain_MOD_producer)) std::fputs("OPAQUE_ORIGINAL_PRODUCER\n",stderr);
 if(f==reinterpret_cast<void*>(&__opaque_chain_MOD_consumer)) std::fputs("OPAQUE_ORIGINAL_CONSUMER\n",stderr);
}
extern "C" __attribute__((no_instrument_function)) void __cyg_profile_func_exit(void*,void*) {}
'''


def observer_object(directory):
    path = directory / 'observer.cpp'
    path.write_text(OBSERVER)
    obj = directory / 'observer.o'
    host = shutil.which('g++-14') or shutil.which('g++')
    assert host
    base.checked([host, '-std=c++17', '-Wall', '-Wextra', '-Werror', '-c', str(path), '-o', str(obj)], directory, 'compile-observer')
    return obj


def entries(text):
    rows = re.findall(r'OPAQUE_BEGIN (\d+) (\d+) (\d+)\n(.*?)OPAQUE_END \1 \2 \3', text, re.S)
    assert [(int(g), int(m), int(n)) for g, m, n, _ in rows] == [(g, m, n) for g in (0, 1) for m in (0, 1) for n in (1, 7, 257)]
    for go, _mask, _n, body in rows:
        assert body.splitlines().count('OPAQUE_ORIGINAL_REDUCTION') == 2 * int(go)
    zero, = re.findall(r'OPAQUE_ZERO_BEGIN\n(.*?)OPAQUE_ZERO_END', text, re.S)
    assert 'OPAQUE_ORIGINAL_' not in zero
    return rows


def test_original_four_thread_reduction_and_observer_fields(tmp_path):
    original, driver = tmp_path / 'original.f90', tmp_path / 'driver.f90'
    original.write_text(source())
    driver.write_text(DRIVER)
    observer = observer_object(tmp_path)
    fc = shutil.which('gfortran-15')
    assert fc
    binary = tmp_path / 'native'
    base.checked([fc, '-fopenmp', '-fcheck=all', '-ffp-contract=off', '-finstrument-functions', '-J', str(tmp_path),
                  str(original), str(driver), str(observer), '-lstdc++', '-o', str(binary)], tmp_path, 'compile-native')
    result = base.checked([str(binary)], tmp_path, 'native', environment('hidden'))
    assert 'OPAQUE_ORIGINAL_REDUCTION_FIELDS_HALOS_AND_BOUNDS_OK' in result.stdout
    entries(result.stderr)


API_OBSERVER = r'''#include "scoped_runtime.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
extern "C" int __real_fort_scope_register(fort_scope_t,std::uint64_t,std::uint64_t,const fort_scope_layout*,int,fort_buffer_t*);
extern "C" int __real_fort_scope_access_begin_v1(fort_scope_t,const fort_scope_plan_binding*,std::size_t,int,void**);
static bool fired=false;
static bool mode(const char* expected) {
 const char* value=std::getenv("FORT_C3_TEST_FAULT");
 return value&&std::strcmp(value,expected)==0;
}
extern "C" int __wrap_fort_scope_register(fort_scope_t c,std::uint64_t identity,std::uint64_t generation,
 const fort_scope_layout* layout,int initialized,fort_buffer_t* buffer) {
 if(layout&&layout->type==FORT_SCOPE_BYTES) {
  if(!fired&&mode("resource")) {
   fired=true;std::fputs("OPAQUE_PREWORK_RESOURCE\n",stderr);return FORT_SCOPE_RESOURCE;
  }
  const int status=__real_fort_scope_register(c,identity,generation,layout,initialized,buffer);
  if(status!=FORT_SCOPE_OK)return status;
  std::fprintf(stderr,"OPAQUE_BYTES_RANGE identity=%llu initialized=%d element_bytes=%zu extent=%zu\n",
    static_cast<unsigned long long>(identity),initialized,layout->element_bytes,layout->extents[0]);
  if(!fired&&mode("alias")) {
   fired=true;fort_buffer_t alias=0;
   const int rejected=__real_fort_scope_register(c,identity+0x100000000ULL,generation,layout,0,&alias);
   if(rejected!=FORT_SCOPE_ALIAS||alias!=0)std::abort();
   std::fputs("OPAQUE_REAL_ALIAS_REJECT\n",stderr);return rejected;
  }
  return status;
 }
 return __real_fort_scope_register(c,identity,generation,layout,initialized,buffer);
}
extern "C" int __wrap_fort_scope_access_begin_v1(fort_scope_t c,const fort_scope_plan_binding* b,
 std::size_t count,int device,void** pointers) {
 const int status=__real_fort_scope_access_begin_v1(c,b,count,device,pointers);
 if(status==FORT_SCOPE_OK&&device==0&&!fired&&mode("fatal")) {
  fired=true;std::fputs("OPAQUE_FATAL_AFTER_COMPLETED_PRODUCER\n",stderr);return FORT_SCOPE_EXECUTION;
 }
 return status;
}
'''


def symbol_evidence(directory, binary, stem):
    nm = shutil.which('nm')
    assert nm
    result = base.checked([nm, '-g', '--defined-only', str(binary)], directory, stem + '-symbols')
    for name in ('reduce_mean', 'producer', 'consumer'):
        assert len(re.findall(r'^[0-9a-fA-F]+ [Tt] __opaque_chain_MOD_' + name + '$', result.stdout, re.M)) == 1


@pytest.fixture(scope='module')
def binaries(tmp_path_factory):
    observer_dir = tmp_path_factory.mktemp('opaque_native_observer')
    observer = observer_object(observer_dir)
    checked = base.checked

    def instrument(command, build, stem, env=None):
        command = list(command)
        if Path(command[0]).name.startswith('gfortran'):
            command.insert(1, '-finstrument-functions')
            if '-c' not in command:
                command.extend([str(observer), '-lstdc++'])
        if stem == 'link-candidate':
            wrapper = build / 'opaque-api-observer.cpp'
            wrapper.write_text(API_OBSERVER)
            obj = build / 'opaque-api-observer.o'
            host = shutil.which('g++-14') or shutil.which('g++')
            checked([host, '-std=c++17', '-Wall', '-Wextra', '-Werror', '-I', str(build), '-c', str(wrapper),
                     '-o', str(obj)], build, 'compile-api-observer')
            command.extend([str(obj), '-Wl,--wrap=fort_scope_register', '-Wl,--wrap=fort_scope_access_begin_v1'])
        return checked(command, build, stem, env)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(base, 'emit', sealed_emission)
        patch.setattr(base, 'DRIVER', DRIVER)
        patch.setattr(base, 'checked', instrument)
        directory, candidate, native = base.epoch_binaries.__wrapped__(tmp_path_factory)
    symbol_evidence(directory, candidate, 'candidate')
    symbol_evidence(directory, native, 'native')
    manifest = json.loads((directory / 'scope-manifest.json').read_text())
    provenance = {row['id']: row for row in manifest['runtime_provenance']['records']}
    mean = next(row for row in manifest['scopes'][0]['checked_execution_plan']['storage_epoch']['resources']
                if row['resource'] == 'opaque_chain::mean')['registration_identity']
    return directory, candidate, native, provenance, mean


def hidden_cpu_completions(events, provenance, manifest):
    """Authenticate completed inline CPU workers, which bypass helper entry hooks."""
    plan = manifest['scopes'][0]['checked_execution_plan']
    members = {row['procedure']: row for row in plan['procedures']}
    result = []
    for event in events:
        if event['kind'] != 'host_commit':
            continue
        implementation = provenance.get(event.get('implementation'), {})
        if implementation.get('backend') != 'generated_cpu':
            continue
        procedure = provenance[event['procedure']]
        segment = provenance[event['segment']]
        operation = provenance[event['operation']]
        worker = provenance[implementation['procedure_id']]
        name = procedure['procedure']
        assert name in {'opaque_chain::producer', 'opaque_chain::consumer'}
        assert (implementation['kind'], procedure['kind'], segment['kind'], operation['kind'], worker['kind']) == (
            'implementation', 'procedure', 'segment', 'source_call', 'procedure')
        child, = [row for row in manifest['child_numerical_regions'] if row['procedure'] == name]
        region, = [row for row in child['regions'] if row['entry'] == implementation['procedure']]
        assert implementation['source'] == 'inline:' + region['source_identity']
        assert worker['procedure'] == implementation['procedure']
        assert worker['source'] == implementation['source']
        assert worker['numerical_identity'] == implementation['numerical_identity']
        assert implementation['region'] + 1 == region['region_id']
        assert operation['target'] == name + '#region' + str(region['region_id'])
        assert all(row['procedure'] == name for row in (segment, operation))
        assert all((row['source'], row['source_sha256']) == (procedure['source'], procedure['source_sha256'])
                   for row in (segment, operation, members[name]))
        assert plan['sources'][procedure['source']] == procedure['source_sha256']
        assert all((row['first_line'], row['last_line']) == (region['first_line'], region['last_line'])
                   for row in (segment, operation))
        assert event['provenance_version'] == '1'
        assert event['domain_role'] == 'execution'
        assert int(event['bytes']) == 0
        key = (name, event['context'], event['domain'], event['operation'], event['implementation'])
        if not result or result[-1] != key:
            result.append(key)
    assert [row[0] for row in result] == ['opaque_chain::producer', 'opaque_chain::consumer'] * 2
    return result


def retained_ledger(result, mode, directory, provenance, mean):
    manifest = None
    if mode == 'hidden':
        from elmm_pipeline.scoped_provenance import scoped_provenance

        manifest = json.loads((directory / 'scope-manifest.json').read_text())
        assert provenance == {row['id']: row for row in manifest['runtime_provenance']['records']}
        linked = scoped_provenance(result.stderr, [manifest])
        assert linked['available'] is True
        assert linked['legacy_events'] == 0
        assert linked['residency_domains']['complete'] is True
    rows = entries(result.stderr)
    zero, = re.findall(r'OPAQUE_ZERO_BEGIN\n(.*?)OPAQUE_ZERO_END', result.stderr, re.S)
    assert 'OPAQUE_BYTES_RANGE' not in zero
    assert not any(e['kind'] in {'domain_create', 'register', 'allocate', 'launch'} for e in events_in(zero))
    ledger = []
    for go, masked, width, body in rows:
        events = events_in(body)
        launches = [i for i, event in enumerate(events) if event['kind'] == 'launch']
        reservations = re.findall(r'OPAQUE_BYTES_RANGE identity=(\d+) initialized=(\d+) element_bytes=(\d+) extent=(\d+)', body)
        n = int(width)
        ledger.append({'go': int(go), 'masked': int(masked), 'n': n, 'events': events, 'reservations': reservations})
        (directory / (mode + '-ledger.json')).write_text(json.dumps(ledger, indent=2) + '\n')
        assert all(int(identity) == mean and initialized == '0' and size == '8' and extent == '3'
                   for identity, initialized, size, extent in reservations)
        if mode == 'hidden':
            assert not launches
            assert bool(reservations) is bool(int(go))
            assert copy_bytes(events, 'upload') == copy_bytes(events, 'download') == 0
            ledger[-1]['cpu_worker_completions'] = hidden_cpu_completions(events, provenance, manifest)
            (directory / (mode + '-ledger.json')).write_text(json.dumps(ledger, indent=2) + '\n')
            continue
        assert bool(reservations) is bool(int(go))
        assert 'OPAQUE_ORIGINAL_PRODUCER' not in body
        assert 'OPAQUE_ORIGINAL_CONSUMER' not in body
        assert [procedure(events[i], provenance) for i in launches] == ['opaque_chain::producer', 'opaque_chain::consumer'] * 2
        assert len({events[i]['domain'] for i in launches}) == 1
        assert not any(int(event.get('buffer', -1)) == mean and event['kind'] in
                       {'allocate', 'upload', 'download', 'device_commit', 'host_commit'} for event in events)
        for stage in range(2):
            first, last = launches[2*stage:2*stage+2]
            between = events[first+1:last]
            assert not any(event['kind'] == 'domain_retire' for event in between)
            assert copy_bytes(between, 'upload') == 0
            assert copy_bytes(between, 'download') == (32*n if int(go) else 0)
    return ledger


@pytest.mark.cuda
@pytest.mark.parametrize('mode', ['traced', 'untraced', 'hidden'])
def test_original_native_reduction_retains_managed_read_replica_and_opaque_storage(binaries, mode):
    directory, candidate, native, provenance, mean = binaries
    env = environment(mode)
    reference = base.checked([str(native)], directory, mode + '-native', env)
    result = base.checked([str(candidate)], directory, mode + '-candidate', env)
    assert result.stdout == reference.stdout
    assert 'OPAQUE_ORIGINAL_REDUCTION_FIELDS_HALOS_AND_BOUNDS_OK' in result.stdout
    entries(reference.stderr)
    entries(result.stderr)
    if mode == 'untraced':
        assert 'FORT_SCOPED ' not in result.stderr
        assert 'OPAQUE_ORIGINAL_PRODUCER' not in result.stderr
        assert 'OPAQUE_ORIGINAL_CONSUMER' not in result.stderr
    else:
        retained_ledger(result, mode, directory, provenance, mean)


@pytest.mark.cuda
@pytest.mark.parametrize('fault', ['resource', 'alias'])
def test_prework_reservation_failure_preserves_original_calls_fields_and_canaries(binaries, fault):
    directory, candidate, native, _provenance, _mean = binaries
    env = environment('traced')
    reference = base.checked([str(native)], directory, fault + '-native', env)
    result = base.checked([str(candidate)], directory, fault + '-candidate', {**env, 'FORT_C3_TEST_FAULT': fault})
    assert result.stdout == reference.stdout
    entries(result.stderr)
    marker = 'OPAQUE_PREWORK_RESOURCE' if fault == 'resource' else 'OPAQUE_REAL_ALIAS_REJECT'
    assert result.stderr.splitlines().count(marker) == 1


@pytest.mark.cuda
def test_fatal_host_acquisition_after_real_producer_does_not_execute_or_replay_source(binaries):
    directory, candidate, _native, provenance, _mean = binaries
    env = {**environment('traced'), 'FORT_C3_TEST_FAULT': 'fatal'}
    result = subprocess.run([str(candidate), 'fault'], cwd=directory, env=env, capture_output=True, text=True, timeout=30)
    (directory / 'fatal.command.json').write_text(json.dumps([str(candidate), 'fault']) + '\n')
    (directory / 'fatal.stdout').write_text(result.stdout)
    (directory / 'fatal.stderr').write_text(result.stderr)
    assert result.returncode != 0
    assert result.stderr.splitlines().count('OPAQUE_FATAL_AFTER_COMPLETED_PRODUCER') == 1
    assert 'shared scope operation failed' in result.stderr
    assert 'fatal native preparation returned' not in result.stderr
    assert 'OPAQUE_ORIGINAL_REDUCTION' not in result.stderr
    assert 'OPAQUE_ORIGINAL_PRODUCER' not in result.stderr
    assert 'OPAQUE_ORIGINAL_CONSUMER' not in result.stderr
    events = events_in(result.stderr)
    assert [procedure(e, provenance) for e in events if e['kind'] == 'launch'] == ['opaque_chain::producer']
    assert copy_bytes(events, 'download') == 32*7
