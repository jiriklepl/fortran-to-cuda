"""Original guarded SECTION copies retain resident producer and readonly input."""
from __future__ import annotations

import json
import os
import re
import shutil
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_source_scopes import FACT, run

_TEAM = '''!$omp parallel private(i)
if(flag)then
!$omp sections
!$omp section
do i=n+1,n+3
a(i)=a(i-n)
enddo
!$omp section
do i=-2,0
a(i)=a(i+n)
enddo
!$omp end sections nowait
endif
!$omp end parallel
'''
SOURCE = '''module conditional_chain
contains
subroutine step(a,p,out,n,flag)
real(8),intent(inout)::a(-2:),out(-2:)
real(8),intent(in)::p(-2:)
integer,intent(in)::n
logical,intent(in)::flag
integer::i
do i=-2,n+3
a(i)=2*p(i)+real(i,8)
enddo
''' + _TEAM + _TEAM + '''do i=-2,n+3
out(i)=a(i)+p(i)
enddo
end subroutine
end module
'''
DRIVER = '''program exercise
use conditional_chain
use iso_fortran_env,only:error_unit
implicit none
real(8),allocatable::a(:),p(:),out(:),ea(:),ep(:),eo(:)
integer::which,n,i,repeat,branch,pass
logical::flag
do which=1,5
select case(which)
case(1)
n=0
case(2)
n=1
case(3)
n=2
case(4)
n=3
case default
n=67
end select
allocate(a(-2:n+3),p(-2:n+3),out(-2:n+3),ea(-2:n+3),ep(-2:n+3),eo(-2:n+3))
do branch=1,2
flag=branch==1
do repeat=1,2
do i=-2,n+3
p(i)=real(11*i*i-5*i+repeat,8)/64._8
ea(i)=2*p(i)+real(i,8)
enddo
ep=p
a=-999._8
out=-777._8
if(flag)then
do pass=1,2
do i=n+1,n+3
ea(i)=ea(i-n)
enddo
do i=-2,0
ea(i)=ea(i+n)
enddo
enddo
endif
eo=ea+p
write(error_unit,'(A,3(1X,I0))')'SECTION_BEGIN',n,branch,repeat
call step(a,p,out,n,flag)
write(error_unit,'(A,3(1X,I0))')'SECTION_END',n,branch,repeat
if(any(a/=ea).or.any(out/=eo))error stop 'full fields or original halo sequence changed'
if(any(p/=ep))error stop 'readonly input changed'
enddo
enddo
deallocate(a,p,out,ea,ep,eo)
enddo
print *, 'SECTION_CHAIN_FIELDS_HALOS_READONLY_OK'
end program
'''


def emit_chain(tmp_path):
    path = tmp_path/'original.f90'
    path.write_text(SOURCE)
    builder = ScopeBuilder([path], 'conditional_chain::step', options=CompilerOptions(opt_level=0),
        config=OffloadConfig(policy='sections', scope_execution='graph', host_threads=4),
        facts={'schema_version': 1, 'participation': 'serial',
               'sources': {str(path): sha256(path.read_bytes()).hexdigest()},
               'captures': {'argument::'+name: dict(FACT) for name in ('a', 'p', 'out')}})
    outputs, report = builder.run()
    return outputs, report, path


def test_normal_builder_preserves_two_original_section_tasks_in_one_chain(tmp_path):
    outputs, report, _path = emit_chain(tmp_path)
    scope, = report['scopes']
    tasks = scope['source_device_tasks']
    assert len(tasks) == 2
    assert all(t['version'] == 3 and t['source_proof']['alternatives'] == [[0, 1], []] for t in tasks)
    assert len([name for name in outputs if name.startswith('fort_atomic_') and name.endswith('.cu')]) == 2
    (tmp_path/'scope-manifest.json').write_text(outputs['scope-manifest.json'])


@pytest.fixture(scope='module')
def compiled_sections_chain(tmp_path_factory):
    build = tmp_path_factory.mktemp('conditional-chain-generated')
    nvcc, host = shutil.which('nvcc'), shutil.which('g++-14') or shutil.which('g++')
    fc = shutil.which('gfortran-15') or shutil.which('gfortran')
    if not all((nvcc, host, fc)):
        pytest.skip('CUDA and Fortran compilation toolchains required')
    outputs, report, original = emit_chain(build)
    (build/'scope-manifest.json').write_text(outputs['scope-manifest.json'])
    for name, text in outputs.items():
        path = build/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    objects = []
    for role in ('common_runtime', 'shared_entry', 'original_source'):
        for index, artifact in enumerate(report['build_sources']):
            if artifact['role'] != role:
                continue
            path, target = build/artifact['path'], build/f'object-{index}.o'
            assert sha256(path.read_bytes()).hexdigest() == report['artifacts_sha256'][artifact['path']]
            if artifact['language'] == 'cuda':
                command = [nvcc, '-std=c++17', '-ccbin', host, '-arch=sm_86', '-Xcompiler=-fopenmp',
                           *artifact['numerical_contract']['required_cuda_options'], '-I', str(build)]
            else:
                command = [fc, '-fopenmp', '-fcheck=all', '-ffp-contract=off', '-J', str(build), '-I', str(build)]
            process = run([*command, '-c', str(path), '-o', str(target)], cwd=build)
            (build/f'compile-{index}.stdout').write_text(process.stdout)
            (build/f'compile-{index}.stderr').write_text(process.stderr)
            objects.append(str(target))
    driver = build/'driver.f90';driver.write_text(DRIVER)
    candidate, native = build/'candidate', build/'native'
    run([fc, '-fopenmp', '-fcheck=all', '-ffp-contract=off', '-I', str(build), str(driver), *objects,
         '-L/usr/local/cuda/lib64', '-Wl,-rpath,/usr/local/cuda/lib64', '-lcudart', '-lstdc++', '-o', str(candidate)], cwd=build)
    run([fc, '-fopenmp', '-fcheck=all', '-ffp-contract=off', '-J', str(build), str(original), str(driver), '-o', str(native)], cwd=build)
    return build, candidate, native


@pytest.mark.cuda
@pytest.mark.parametrize('hidden', [False, True], ids=['enabled', 'hidden'])
def test_original_sections_chain_parity_and_readonly_replica_retention(compiled_sections_chain, hidden):
    build, candidate, native = compiled_sections_chain
    environment = {**os.environ, 'OMP_NUM_THREADS': '4', 'OMP_DYNAMIC': 'FALSE', 'OMP_SCHEDULE': 'static',
                   'FORT_RUNTIME_TRACE': '1'}
    environment.pop('CUDA_VISIBLE_DEVICES', None)
    if hidden:
        environment['CUDA_VISIBLE_DEVICES'] = ''
    label = 'hidden' if hidden else 'enabled'
    reference = run([str(native)], cwd=build, env=environment)
    result = run([str(candidate)], cwd=build, env=environment)
    for role, process in (('native', reference), ('candidate', result)):
        (build/f'{label}-{role}.stdout').write_text(process.stdout)
        (build/f'{label}-{role}.stderr').write_text(process.stderr)
    assert result.stdout == reference.stdout
    assert 'SECTION_CHAIN_FIELDS_HALOS_READONLY_OK' in result.stdout
    cases = re.findall(r'SECTION_BEGIN (\d+) (\d+) (\d+)\n(.*?)SECTION_END \1 \2 \3', result.stderr, re.S)
    assert len(cases) == 20
    for size, branch, _repeat, events in cases:
        size, branch = int(size), int(branch)
        commits = [line for line in events.splitlines() if line.startswith('FORT_SCOPED external_commit ')]
        if hidden or branch == 2 or size in {1, 2}:
            assert not commits, events
        else:
            assert len(commits) == 2, events
            assert all(re.search(r'\bbytes=1\b', line) for line in commits), events
            uploads = [line for line in events.splitlines() if line.startswith('FORT_SCOPED upload ')]
            downloads = [line for line in events.splitlines() if line.startswith('FORT_SCOPED download ')]
            assert sum(int(re.search(r'\bbytes=(\d+)\b', line)[1]) for line in uploads) == 8*(size+6), events
            assert sum(int(re.search(r'\bbytes=(\d+)\b', line)[1]) for line in downloads) == 16*(size+6), events
            between = events.split(commits[0], 1)[1].split(commits[1], 1)[0]
            assert not re.search(r'^FORT_SCOPED (?:upload|download) ', between, re.M), between
