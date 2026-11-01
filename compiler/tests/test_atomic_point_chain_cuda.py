"""Exact original point reads preserve replicas across GPU/host/GPU work."""
import json
import os
import re
import shutil
import subprocess
from hashlib import sha256

import pytest

from compiler.tests.test_atomic_point_chain_source import emit_chain
from compiler.tests.test_source_scopes import run


@pytest.fixture(scope='module')
def compiled_point_chain(tmp_path_factory):
    build = tmp_path_factory.mktemp('atomic-point-chain')
    nvcc, host, fc = (shutil.which(name) for name in ('nvcc', 'g++-14', 'gfortran-15'))
    assert nvcc, 'qualified CUDA toolchain required'
    assert host, 'qualified GNU C++ toolchain required'
    assert fc, 'qualified GNU Fortran toolchain required'
    outputs, report, facts = emit_chain(build)
    assert report['scope_count'] == 1
    (build / 'facts.json').write_text(json.dumps(facts, indent=2) + '\n')
    for name, text in outputs.items():
        path = build / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    objects = []
    for role in ('common_runtime', 'shared_entry', 'original_source'):
        for index, artifact in enumerate(report['build_sources']):
            if artifact['role'] != role:
                continue
            path, target = build / artifact['path'], build / f'object-{index}.o'
            assert sha256(path.read_bytes()).hexdigest() == report['artifacts_sha256'][artifact['path']]
            command = ([nvcc, '-std=c++17', '-ccbin', host, '-arch=sm_86', '-Xcompiler=-fopenmp',
                        *artifact['numerical_contract']['required_cuda_options'], '-I', str(build)]
                       if artifact['language'] == 'cuda' else
                       [fc, '-fopenmp', '-fcheck=all', '-ffp-contract=off', '-J', str(build), '-I', str(build)])
            result = run([*command, '-c', str(path), '-o', str(target)], cwd=build, timeout=120)
            (build / f'compile-{index}.stdout').write_text(result.stdout)
            (build / f'compile-{index}.stderr').write_text(result.stderr)
            objects.append(str(target))
    driver = build / 'driver.f90'
    driver.write_text('''program point_chain
use atomic_owner
use iso_fortran_env,only:error_unit
implicit none
integer,parameter::n=7
real(8)::a(-2:n+4),b(-2:n+4),p(0:n+1)
integer::gate(-2:2),i,scenario,mode
logical::flag,produce
character(24)::argument
call get_command_argument(1,argument)
read(argument,*)scenario
produce=mod(scenario,2)==0
mode=4
if(scenario>2)mode=2
flag=.true.
gate=[-731,13,-733,-734,-735]
gate(-1)=mode
if(produce)gate(-1)=13
if(scenario==3)gate(2)=1
if(scenario==4)gate(2)=-1
do i=-2,n+4
a(i)=real(7*i*i-3*i+11,8)/64._8
b(i)=real(500-i,8)/32._8
enddo
do i=0,n+1
p(i)=real(3*i-7,8)/16._8
enddo
write(error_unit,'(A,1X,I0)')'POINT_CHAIN_BEGIN',scenario
call exercise(a,b,p,n,mode,flag,gate,produce)
write(error_unit,'(A,1X,I0)')'POINT_CHAIN_END',scenario
open(77,file='fields.bin',access='stream',form='unformatted',status='replace')
write(77)a,b,p,gate
close(77)
print *,'POINT_CHAIN_FULL_FIELDS_AND_CANARIES_OK'
end program
''')
    wrapper = build / 'observe.cpp'
    wrapper.write_text(r'''
#include "scoped_runtime.h"
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
static fort_buffer_t gate_buffer=0;
static bool injected=false;
extern "C" int __real_fort_scope_register(fort_scope_t,uint64_t,uint64_t,const fort_scope_layout*,int,fort_buffer_t*);
extern "C" int __real_fort_scope_admit_source_reads_v1(fort_scope_t,fort_buffer_t,const fort_scope_section*,size_t);
extern "C" int __real_fort_scope_access_begin_v1(fort_scope_t,const fort_scope_plan_binding*,size_t,int,void**);
extern "C" int __wrap_fort_scope_register(fort_scope_t context,uint64_t identity,uint64_t generation,
 const fort_scope_layout* layout,int initialized,fort_buffer_t* buffer){
 int status=__real_fort_scope_register(context,identity,generation,layout,initialized,buffer);
 if(status==FORT_SCOPE_OK&&layout->rank==1&&layout->element_bytes==4&&layout->extents[0]==5){
  gate_buffer=*buffer;
  std::fprintf(stderr,"POINT_BUFFER identity=%" PRIu64 " handle=%" PRIu64 " initialized=%d\n",identity,*buffer,initialized);
 }
 return status;
}
static int fault(fort_scope_t context,const char* at){
 const char* mode=std::getenv("FORT_ATOMIC_POINT_FAULT");
 if(!mode||injected)return FORT_SCOPE_OK;
 if((std::strcmp(at,"admit")==0&&std::strcmp(mode,"after_submit_admit_fatal")==0)||
    (std::strcmp(at,"acquire")==0&&std::strcmp(mode,"after_submit_acquire_fatal")==0)){
  // Test-only boundary: execute and complete the already accepted original
  // producer before simulating a fatal API status. No source work is replayed.
  int flushed=fort_scope_graph_flush_v1(context);
  int waited=flushed==FORT_SCOPE_OK?fort_scope_wait(context):flushed;
  std::fprintf(stderr,"POINT_PRODUCER_COMPLETED flush=%d wait=%d\n",flushed,waited);
  if(flushed!=FORT_SCOPE_OK||waited!=FORT_SCOPE_OK)std::abort();
  injected=true;return FORT_SCOPE_EXECUTION;
 }
 if(std::strcmp(at,"admit")==0&&std::strcmp(mode,"admit_resource")==0){injected=true;return FORT_SCOPE_RESOURCE;}
 if(std::strcmp(at,"admit")==0&&std::strcmp(mode,"admit_fatal")==0){injected=true;return FORT_SCOPE_EXECUTION;}
 if(std::strcmp(at,"acquire")==0&&std::strcmp(mode,"acquire_fatal")==0){injected=true;return FORT_SCOPE_EXECUTION;}
 return FORT_SCOPE_OK;
}
extern "C" int __wrap_fort_scope_admit_source_reads_v1(fort_scope_t context,fort_buffer_t buffer,
 const fort_scope_section* sections,size_t count){
 if(buffer==gate_buffer&&count){
  for(size_t i=0;i<count;++i)std::fprintf(stderr,"POINT_ADMIT lower=%zu upper=%zu\n",sections[i].lower[0],sections[i].upper[0]);
  int status=fault(context,"admit");
  if(status!=FORT_SCOPE_OK){std::fprintf(stderr,"POINT_INJECT admit status=%d\n",status);return status;}
 }
 return __real_fort_scope_admit_source_reads_v1(context,buffer,sections,count);
}
extern "C" int __wrap_fort_scope_access_begin_v1(fort_scope_t context,const fort_scope_plan_binding* bindings,
 size_t count,int device,void** devices){
 if(device==0){
  for(size_t i=0;i<count;++i)if(bindings[i].buffer==gate_buffer){
   const auto& access=bindings[i].access;
   for(size_t j=0;j<access.read_count;++j)std::fprintf(stderr,"POINT_HOST_READ lower=%zu upper=%zu\n",access.reads[j].lower[0],access.reads[j].upper[0]);
   int status=fault(context,"acquire");
   if(status!=FORT_SCOPE_OK){std::fprintf(stderr,"POINT_INJECT acquire status=%d\n",status);return status;}
  }
 }
 return __real_fort_scope_access_begin_v1(context,bindings,count,device,devices);
}
''')
    wrapped = build / 'observe.o'
    run([host, '-std=c++17', '-I', str(build), '-c', str(wrapper), '-o', str(wrapped)], cwd=build)
    candidate, native = build / 'candidate', build / 'native'
    run([fc, '-fopenmp', '-fcheck=all', '-ffp-contract=off', '-I', str(build), str(driver), *objects, str(wrapped),
         '-Wl,--wrap=fort_scope_register', '-Wl,--wrap=fort_scope_admit_source_reads_v1',
         '-Wl,--wrap=fort_scope_access_begin_v1', '-L/usr/local/cuda/lib64', '-Wl,-rpath,/usr/local/cuda/lib64',
         '-lcudart', '-lstdc++', '-o', str(candidate)], cwd=build)
    run([fc, '-fopenmp', '-fcheck=all', '-ffp-contract=off', '-J', str(build),
         str(build / 'point_chain.f90'), str(driver), '-o', str(native)], cwd=build)
    return build, candidate, native


def execute(compiled, scenario, mode=None):
    build, candidate, native = compiled
    work = build / (str(scenario) + '-' + (mode or 'normal'))
    work.mkdir()
    environment = {**os.environ, 'OMP_NUM_THREADS': '4', 'OMP_DYNAMIC': 'FALSE', 'OMP_SCHEDULE': 'static',
                   'FORT_RUNTIME_TRACE': '1'}
    assert environment.get('CUDA_VISIBLE_DEVICES') == '0'
    environment.pop('FORT_ATOMIC_POINT_FAULT', None)
    reference = run([str(native), str(scenario)], cwd=work, env=environment, timeout=30)
    expected = (work / 'fields.bin').read_bytes()
    (work / 'native.bin').write_bytes(expected)
    (work / 'fields.bin').unlink()
    if mode:
        environment['FORT_ATOMIC_POINT_FAULT'] = mode
    result = subprocess.run([str(candidate), str(scenario)], cwd=work, env=environment,
                            capture_output=True, text=True, timeout=30)
    for name, process in (('native', reference), ('candidate', result)):
        (work / (name + '.stdout')).write_text(process.stdout)
        (work / (name + '.stderr')).write_text(process.stderr)
    return work, expected, result


@pytest.mark.cuda
@pytest.mark.parametrize('scenario', [1, 2, 3, 4], ids=['host-first', 'device-first', 'host-nested', 'device-nested'])
def test_original_point_host_read_preserves_device_and_dormant_points(compiled_point_chain, scenario):
    work, expected, result = execute(compiled_point_chain, scenario)
    assert result.returncode == 0, result.stderr
    assert (work / 'fields.bin').read_bytes() == expected
    assert 'POINT_CHAIN_FULL_FIELDS_AND_CANARIES_OK' in result.stdout
    records = re.findall(r'POINT_BUFFER identity=(\d+) handle=\d+ initialized=(\d+)', result.stderr)
    assert records
    assert len({identity for identity, _ in records}) == 1
    assert all(initialized == '0' for _, initialized in records)
    identity = records[0][0]
    copies = [line for line in result.stderr.splitlines()
              if re.match(r'FORT_SCOPED (?:upload|download) buffer=' + identity + r'\b', line)]
    uploaded = sum(int(re.search(r'bytes=(\d+)', line)[1]) for line in copies if ' upload ' in line)
    downloaded = sum(int(re.search(r'bytes=(\d+)', line)[1]) for line in copies if ' download ' in line)
    assert (uploaded, downloaded) == ((0, 4) if scenario % 2 == 0 else (4, 0)), copies
    host_reads = re.findall(r'POINT_HOST_READ lower=(\d+) upper=(\d+)', result.stderr)
    assert host_reads == ([('1', '2')] if scenario <= 2 else [('1', '2'), ('4', '5')]), result.stderr
    assert all(pair in [('1', '2'), ('4', '5')] for pair in re.findall(r'POINT_ADMIT lower=(\d+) upper=(\d+)', result.stderr))
    if scenario <= 2:
        assert 'POINT_ADMIT lower=4' not in result.stderr
    launches = re.findall(r'^FORT_SCOPED launch ', result.stderr, re.M)
    assert len(launches) == (4 if scenario % 2 == 0 else 3), result.stderr


@pytest.mark.cuda
def test_safe_predicate_admission_decline_runs_original_team_once(compiled_point_chain):
    work, expected, result = execute(compiled_point_chain, 2, 'admit_resource')
    assert result.returncode == 0, result.stderr
    assert (work / 'fields.bin').read_bytes() == expected
    assert result.stderr.count('POINT_INJECT admit ') == 1
    assert 'POINT_CHAIN_END 2' in result.stderr


@pytest.mark.cuda
@pytest.mark.parametrize('mode', ['admit_fatal', 'acquire_fatal'])
def test_predicate_failure_with_pending_gpu_producer_cannot_replay_original_team(compiled_point_chain, mode):
    work, _expected, result = execute(compiled_point_chain, 2, mode)
    assert result.returncode != 0
    expected = ('atomic predicate source admission failed fatally' if mode == 'admit_fatal'
                else 'atomic predicate acquisition failed fatally')
    assert expected in result.stderr
    assert result.stderr.count('POINT_INJECT ') == 1
    assert not re.findall(r'^FORT_SCOPED launch ', result.stderr, re.M), result.stderr
    assert len(re.findall(r'^FORT_SCOPED queue_enqueue ', result.stderr, re.M)) == 1
    assert 'POINT_PRODUCER_COMPLETED' not in result.stderr
    assert 'POINT_CHAIN_END' not in result.stderr
    assert not (work / 'fields.bin').exists()


@pytest.mark.cuda
@pytest.mark.parametrize('mode', ['after_submit_admit_fatal', 'after_submit_acquire_fatal'])
def test_predicate_failure_after_completed_gpu_producer_cannot_replay_original_team(compiled_point_chain, mode):
    work, _expected, result = execute(compiled_point_chain, 2, mode)
    assert result.returncode != 0
    expected = ('atomic predicate source admission failed fatally' if mode == 'after_submit_admit_fatal'
                else 'atomic predicate acquisition failed fatally')
    assert expected in result.stderr
    assert result.stderr.count('POINT_INJECT ') == 1
    assert result.stderr.count('POINT_PRODUCER_COMPLETED flush=0 wait=0') == 1
    assert len(re.findall(r'^FORT_SCOPED launch ', result.stderr, re.M)) == 1, result.stderr
    assert result.stderr.index('FORT_SCOPED launch ') < result.stderr.index('POINT_PRODUCER_COMPLETED')
    assert result.stderr.index('POINT_PRODUCER_COMPLETED') < result.stderr.index('POINT_INJECT ')
    assert 'POINT_CHAIN_END' not in result.stderr
    assert not (work / 'fields.bin').exists()
