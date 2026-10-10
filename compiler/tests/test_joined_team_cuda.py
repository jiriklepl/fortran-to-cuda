"""Public mixed-team artifacts preserve native boundaries and complete fields."""

import json
import os
import re
import shutil
from hashlib import sha256
from pathlib import Path

import pytest

from compiler.tests.test_joined_team_source import teams
from compiler.tests.test_scoped_batch_sources_cuda import actual_profile
from compiler.tests.test_source_scopes import FACT, ROOT, generate, run


TEAM = '''!$omp parallel private(i,j) shared(a,b,out,n,m,flag) default(none)
!$omp do
do j=3,m+2
do i=-2,n-3
b(i,j)=2*a(i,j)+real(i-2*j,8)
enddo
enddo
!$omp end do
if(flag) then
!$omp do
do i=-2,n-3
b(i,3)=b(i,3)+sum(a)
enddo
!$omp end do
endif
!$omp do
do j=3,m+2
do i=-2,n-3
out(i,j)=a(i,j)+b(i,j)
enddo
enddo
!$omp end do
!$omp end parallel
'''
DECLARATIONS = '''real(8),intent(in)::a(-2:,3:)
real(8),intent(inout)::b(-2:,3:),out(-2:,3:)
integer,intent(in)::n,m
logical,intent(in)::flag
integer::i,j
'''
SOURCE = '''module joined_owner
implicit none
integer::visits=0
contains
subroutine step(a,b,out,n,m,flag)
''' + DECLARATIONS + 'visits=visits+1\n' + TEAM + '''visits=visits+11
end subroutine
end module
'''

INDIRECT_CORRECTION = '''!$omp do
do q=1,size(sites)
b(sites(q)%row,sites(q)%column)=b(sites(q)%row,sites(q)%column)+a(sites(q)%row,sites(q)%column)
enddo
!$omp end do
'''
INDIRECT_CASES = ('indirect_faces', 'indirect_budget')
NATIVE_SEGMENT_COUNT = 28


def indirect_source(label):
    declarations = '''type site_t
integer::row,column
end type
type(site_t),allocatable::sites(:)
'''
    # Allocation and initialization execute in the original module owner,
    # outside the compiled entry. Each reached native unit reads stable metadata.
    if label == 'indirect_faces':
        setup = '''subroutine setup_sites(n,m)
integer,intent(in)::n,m
integer::q
if(allocated(sites)) deallocate(sites)
if(n>0) then
allocate(sites(2*m))
do q=1,m
sites(q)%row=-2
sites(q)%column=q+2
sites(q+m)%row=n-3
sites(q+m)%column=q+2
enddo
else
allocate(sites(0))
endif
end subroutine
'''
    else:
        setup = '''subroutine setup_sites(n,m)
integer,intent(in)::n,m
integer::q,count
if(allocated(sites)) deallocate(sites)
count=min(33,(n+1)/2)
allocate(sites(count))
do q=1,count
sites(q)%row=-2+2*(q-1)
sites(q)%column=3
enddo
end subroutine
'''
    correction = '''!$omp do
do i=-2,n-3
b(i,3)=b(i,3)+sum(a)
enddo
!$omp end do
'''
    source = SOURCE.replace('integer::visits=0\n', 'integer::visits=0\n'+declarations)
    source = source.replace('contains\n', 'contains\n'+setup, 1)
    source = source.replace('integer::i,j\n', 'integer::i,j,q\n')
    source = source.replace('parallel private(i,j) shared(a,b,out,n,m,flag)',
                            'parallel private(i,j,q) shared(a,b,out,n,m,flag,sites)')
    assert source.count(correction) == 1
    return source.replace(correction, INDIRECT_CORRECTION)


def source_case(label):
    if label in INDIRECT_CASES:
        return indirect_source(label)
    source = SOURCE
    if label == 'many_segments':
        correction = '''if(flag) then
!$omp do
do i=-2,n-3
b(i,3)=b(i,3)+sum(a)
enddo
!$omp end do
endif
'''
        repeated = correction.replace('sum(a)', '+'.join(['sum(a)'] * 6)) * NATIVE_SEGMENT_COUNT
        assert source.count(correction) == 1
        return source.replace(correction, repeated)
    if label == 'module_child':
        child = ('module joined_child\nimplicit none\ncontains\nsubroutine child_step(a,b,out,n,m,flag)\n'
                 + DECLARATIONS + TEAM + 'end subroutine\nend module\n')
        caller = ('''module joined_owner
use joined_child,only:child_step
implicit none
integer::visits=0
contains
subroutine step(a,b,out,n,m,flag)
''' + DECLARATIONS + '''visits=visits+1
do j=3,m+2
do i=-2,n-3
b(i,j)=a(i,j)+7
enddo
enddo
call child_step(a,b,out,n,m,flag)
do j=3,m+2
do i=-2,n-3
out(i,j)=out(i,j)+5
enddo
enddo
visits=visits+11
end subroutine
end module
''')
        return child + caller
    return source


DRIVER = '''program verify
use joined_owner,only:step,visits
use iso_fortran_env,only:error_unit
implicit none
real(8),allocatable::a(:,:),b(:,:),out(:,:)
integer,parameter::sizes(3)=[0,19,4096]
integer::shape,repetition,n,m,low,ly,i,j,unit,ordinal
logical::flag
open(newunit=unit,file='fields.bin',access='stream',form='unformatted',status='replace')
ordinal=0
do shape=1,size(sizes)
n=sizes(shape)
m=7
low=-7-shape
ly=5-2*shape
allocate(a(low:low+n+7,ly:ly+m+3),b(low:low+n+7,ly:ly+m+3),out(low:low+n+7,ly:ly+m+3))
do repetition=1,2
ordinal=ordinal+1
do j=ly,ly+m+3
do i=low,low+n+7
a(i,j)=real(i-low+2*(j-ly)+3*repetition,8)*0.25d0
enddo
enddo
b=-99.d0
out=-101.d0
flag=repetition==2
write(error_unit,'(a,i0,a,i0,a,i0)') 'BEGIN_CALL ',ordinal,' n=',n,' m=',m
call step(a,b,out,n,m,flag)
write(error_unit,'(a,i0)') 'END_CALL ',ordinal
if(visits/=12*ordinal) error stop 'original prefix or suffix replayed'
if(any(b(low+n:low+n+7,:)/=-99.d0).or.any(b(:,ly+m:ly+m+3)/=-99.d0)) error stop 'intermediate halo changed'
if(any(out(low+n:low+n+7,:)/=-101.d0).or.any(out(:,ly+m:ly+m+3)/=-101.d0)) error stop 'output halo changed'
write(unit) n,m,low,ly,flag,visits,a,b,out
! NATIVE_CHILD
enddo
deallocate(a,b,out)
enddo
close(unit)
print *, 'FIELDS_OK'
end program
'''


def driver_case(label):
    driver = DRIVER
    if label == 'many_segments':
        # This is a reached-planning correctness fixture. A small third shape
        # avoids turning repeated original SUMs into an unrelated CPU workload.
        driver = driver.replace('sizes(3)=[0,19,4096]', 'sizes(3)=[0,19,65]')
    if label in INDIRECT_CASES:
        driver = driver.replace('use joined_owner,only:step,visits',
                                'use joined_owner,only:step,visits,setup_sites')
        driver = driver.replace('do repetition=1,2\n', 'call setup_sites(n,m)\ndo repetition=1,2\n')
    if label == 'module_child':
        driver = driver.replace('use joined_owner,only:step,visits',
                                'use joined_owner,only:step,visits\nuse joined_child,only:child_step')
        driver = driver.replace('! NATIVE_CHILD', '''write(error_unit,'(a,i0)') 'NATIVE_ABI_BEGIN ',ordinal
call child_step(a,b,out,n,m,flag)
write(error_unit,'(a,i0)') 'NATIVE_ABI_END ',ordinal
if(any(b(low+n:low+n+7,:)/=-99.d0).or.any(b(:,ly+m:ly+m+3)/=-99.d0)) error stop 'native child b halo changed'
if(any(out(low+n:low+n+7,:)/=-101.d0).or.any(out(:,ly+m:ly+m+3)/=-101.d0)) error stop 'native child out halo changed'
write(unit) a,b,out''')
    return driver


@pytest.fixture(scope='module')
def joined_binaries(tmp_path_factory):
    nvcc = shutil.which('nvcc')
    host = shutil.which('g++-14') or shutil.which('g++')
    fortran = shutil.which('gfortran-15') or shutil.which('gfortran')
    if not all((nvcc, host, fortran)):
        pytest.skip('CUDA/C++/Fortran toolchains unavailable')
    directory = tmp_path_factory.mktemp('joined_team_cuda')
    checkout = directory/'independent-compiler'
    shutil.copytree(ROOT/'compiler', checkout/'compiler', ignore=shutil.ignore_patterns(
        '__pycache__', '.*cache', 'CODE_MAP.md', 'MEMORY_MODEL_PLAN.md'))
    architecture = actual_profile(directory, nvcc, host)['hardware']['compute_capability'].replace('.', '')
    cuda_library = Path(nvcc).resolve().parent.parent/'lib64'
    cuda_flags = [nvcc, '-O2', '-std=c++17', '-ccbin', host, '-arch=sm_'+architecture, '-Xcompiler=-fopenmp']
    fortran_flags = [fortran, '-O3', '-cpp', '-std=f2018', '-fopenmp', '-fcheck=all,array-temps',
                     '-finit-logical=true', '-ffree-line-length-none']
    cache, reuse, binaries = {}, [], {}
    for label in ('plain', 'zero_budget', 'module_child', *INDIRECT_CASES, 'many_segments'):
        case = directory/label
        facts = {'schema_version': 1, 'participation': 'serial',
                 'captures': {'argument::'+name: FACT for name in ('a', 'b', 'out')}}
        if label == 'zero_budget':
            facts['device_budget_bytes'] = 0
        original, output, manifest = generate(case, source_case(label), entry='joined_owner::step',
            facts=facts, checkout=checkout, scope_execution='reached')
        (team, operations), = teams(manifest)
        assert len(team['numerical_units']) == 2
        assert not manifest['scopes'][0]['boundaries']
        if label in INDIRECT_CASES:
            inspected = [item for item in operations['native_operations'] if item.get('indirect_inspector')]
            native, = inspected
            assert native['indirect_inspector']['available']
            assert native['indirect_inspector']['rectangle_limit'] == 32
            assert native['indirect_inspector']['metadata_reservations'] == ['joined_owner::sites']
            assert native['effective_sections']['availability'] == 'checked when reached'
            assert not native['indirect_inspector']['proves_scatter_independence']
        else:
            assert any(item['kind'] == 'original native worksharing' and item['sections']['available']
                       for item in operations['native_operations'])
        if label == 'module_child':
            companion, = manifest['scopes'][0]['module_coordinators']
            assert companion['procedure'] == 'joined_child::child_step'
        if label == 'many_segments':
            assert team['budgets']['expanded_owner_effects'] > 256
            assert team['budgets']['procedure_source_operations'] <= 256
            assert team['budgets']['largest_reached_native_effects'] <= 256
            assert source_case(label).count('if(flag) then') == NATIVE_SEGMENT_COUNT
        build = case/'build'
        build.mkdir()
        headers = tuple(sorted({(path.name, sha256(path.read_bytes()).hexdigest())
                                for path in output.rglob('*') if path.suffix in {'.h', '.hpp', '.cuh'}}))
        objects = []
        for role in ('common_runtime', 'shared_entry', 'original_source'):
            for item in manifest['build_sources']:
                if item['role'] != role:
                    continue
                source = output/item['path']
                digest = sha256(source.read_bytes()).hexdigest()
                assert digest == manifest['artifacts_sha256'][item['path']]
                key = (digest, headers, tuple(cuda_flags), manifest['runtime']['runtime_id'])
                cached = item['language'] == 'cuda' and key in cache
                target = cache[key] if cached else build/(str(len(objects))+'.o')
                if not cached:
                    flags = cuda_flags if item['language'] == 'cuda' else fortran_flags
                    run([*flags, '-I', str(output), '-I', str(build), '-c', str(source), '-o', str(target)], cwd=build)
                    if item['language'] == 'cuda':
                        cache[key] = target
                reuse.append({'case': label, 'source': item['path'], 'sha256': digest, 'reused': cached,
                              'object': str(target), 'headers': headers if item['language'] == 'cuda' else ()})
                objects.append(str(target))
        driver = build/'driver.f90'
        driver.write_text(driver_case(label))
        candidate = build/'candidate'
        run([*fortran_flags, '-I', str(build), str(driver), *objects, '-L'+str(cuda_library),
             '-Wl,-rpath,'+str(cuda_library), '-lcudart', '-lstdc++', '-o', str(candidate)], cwd=build)
        native_build = case/'native'
        native_build.mkdir()
        native = native_build/'native'
        run([*fortran_flags, str(original), str(driver), '-o', str(native)], cwd=native_build)
        environment = {**os.environ, 'OMP_NUM_THREADS': '4', 'OMP_DYNAMIC': 'FALSE'}
        reference = run([str(native)], cwd=native_build, env=environment)
        assert 'FIELDS_OK' in reference.stdout
        binaries[label] = (candidate, build, (native_build/'fields.bin').read_bytes(), manifest)
    (directory/'object-reuse.json').write_text(json.dumps(reuse, indent=2)+'\n')
    return binaries


def execute(compiled, label, *, threads=4):
    candidate, build, expected, manifest = compiled[label]
    result = run([str(candidate)], cwd=build, env={**os.environ, 'OMP_NUM_THREADS': str(threads),
        'OMP_DYNAMIC': 'FALSE', 'FORT_RUNTIME_TRACE': '1'})
    (build/('threads-'+str(threads)+'.stdout')).write_text(result.stdout)
    (build/('threads-'+str(threads)+'.stderr')).write_text(result.stderr)
    assert (build/'fields.bin').read_bytes() == expected
    assert 'FIELDS_OK' in result.stdout
    assert 'array temporary' not in result.stderr.lower()
    return result, manifest


def calls(stderr):
    result = []
    for part in stderr.split('BEGIN_CALL ')[1:]:
        header, body = part.split('\n', 1)
        ordinal, n, m = map(int, re.fullmatch(r'(\d+) n=(\d+) m=(\d+)', header).groups())
        result.append((ordinal, n, m, body.split('END_CALL ', 1)[0]))
    assert len(result) == 6
    return result


def transfer_bytes(trace, operation):
    return sum(int(value) for value in re.findall(r'^FORT_SCOPED '+operation+r'\b[^\n]*\bbytes=(\d+)', trace, re.M))


@pytest.mark.cuda
def test_mixed_original_team_preserves_fields_and_only_mirrors_boundary_sections(joined_binaries):
    result, _ = execute(joined_binaries, 'plain')
    validations = [json.loads(line.removeprefix('FORT_SCOPED evidence '))
                   for line in result.stderr.splitlines() if line.startswith('FORT_SCOPED evidence ')]
    definitions = [row for row in validations if row['event'] == 'definition_validation']
    assert definitions and all(row['status'] == 0 for row in definitions)
    for ordinal, n, m, trace in calls(result.stderr):
        assert trace.count('FORT_SCOPED launch ') == (2 if n else 0)
        # One immutable input upload spans producer and consumer. Only the
        # corrected plane is uploaded again; complete outputs are published.
        assert transfer_bytes(trace, 'upload') == n*m*8 + (n*8 if ordinal % 2 == 0 else 0)
        assert transfer_bytes(trace, 'download') == 2*n*m*8


@pytest.mark.cuda
@pytest.mark.parametrize('label,threads', [('plain', 2), ('zero_budget', 4)])
def test_original_team_mismatch_or_zero_budget_falls_back_once(joined_binaries, label, threads):
    result, _ = execute(joined_binaries, label, threads=threads)
    for _, _, _, trace in calls(result.stderr):
        assert 'FORT_SCOPED launch ' not in trace
        assert transfer_bytes(trace, 'upload') == transfer_bytes(trace, 'download') == 0


@pytest.mark.cuda
def test_original_module_child_native_abi_keeps_controls_absent(joined_binaries):
    result, manifest = execute(joined_binaries, 'module_child')
    assert manifest['scopes'][0]['module_coordinators']
    assert 'FORT_SCOPED launch ' in result.stderr
    blocks = result.stderr.split('NATIVE_ABI_BEGIN ')[1:]
    assert len(blocks) == 6
    for block in blocks:
        assert 'FORT_SCOPED ' not in block.split('NATIVE_ABI_END ', 1)[0]


@pytest.mark.cuda
def test_native_indirect_opposite_faces_keep_exact_communication_and_complete_fields(joined_binaries):
    result, _ = execute(joined_binaries, 'indirect_faces')
    for ordinal, n, m, trace in calls(result.stderr):
        assert trace.count('FORT_SCOPED launch ') == (2 if n else 0)
        corrected = 2*m*8 if n and ordinal % 2 == 0 else 0
        assert transfer_bytes(trace, 'upload') == n*m*8 + corrected
        assert transfer_bytes(trace, 'download') == 2*n*m*8


@pytest.mark.cuda
def test_overbudget_indirect_inspector_publishes_gpu_prefix_and_continues_once(joined_binaries):
    result, _ = execute(joined_binaries, 'indirect_budget')
    reached_exhaustion = False
    for ordinal, n, m, trace in calls(result.stderr):
        count = min(33, (n+1)//2)
        exhausted = ordinal % 2 == 0 and count > 32
        reached_exhaustion |= exhausted
        assert trace.count('FORT_SCOPED launch ') == (1 if exhausted else 2 if n else 0)
        # A failed exact union publishes the first GPU producer. The original
        # indirect correction and consumer then run once, with no GPU replay.
        corrected = count*8 if ordinal % 2 == 0 and not exhausted else 0
        assert transfer_bytes(trace, 'upload') == n*m*8 + corrected
        assert transfer_bytes(trace, 'download') == (n*m*8 if exhausted else 2*n*m*8)
    assert reached_exhaustion


@pytest.mark.cuda
def test_many_reached_native_segments_keep_residency_and_complete_fields(joined_binaries):
    result, manifest = execute(joined_binaries, 'many_segments')
    (team, operations), = teams(manifest)
    assert team['budgets']['expanded_owner_effects'] > 256
    corrections = [item for item in operations['native_operations']
                   if item['kind'] == 'original native worksharing' and 'argument::a' in item['resources']]
    assert len(corrections) >= NATIVE_SEGMENT_COUNT
    for ordinal, n, m, trace in calls(result.stderr):
        assert trace.count('FORT_SCOPED launch ') == (2 if n else 0)
        # All reached corrections reuse the same host-current plane. Its one
        # mirror upload occurs when the final GPU consumer needs it again.
        assert transfer_bytes(trace, 'upload') == n*m*8 + (n*8 if ordinal % 2 == 0 else 0)
        assert transfer_bytes(trace, 'download') == 2*n*m*8
        definitions = [json.loads(line.removeprefix('FORT_SCOPED evidence '))
                       for line in trace.splitlines() if line.startswith('FORT_SCOPED evidence ')]
        assert all(row['status'] == 0 for row in definitions if row['event'] == 'definition_validation')
