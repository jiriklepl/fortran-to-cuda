"""Original indirect native work sees earlier GPU definitions exactly once."""

import os
import shutil

import pytest

from compiler.tests.test_reached_definitions import SOURCE, build
from compiler.tests.test_scoped_batch_sources_cuda import actual_profile
from compiler.tests.test_source_scopes import run


DRIVER = '''program verify
use ordered_storage,only:step,visits
implicit none
real(8),allocatable::a(:),b(:),out(:)
integer,parameter::sizes(3)=[0,8,257]
integer,allocatable::indices(:)
integer::shape,iteration,n,i,unit
open(newunit=unit,file='fields.bin',access='stream',form='unformatted',status='replace')
do shape=1,size(sizes)
 n=sizes(shape)
 allocate(a(-7:n-8),b(-7:n-8),out(-7:n-8),indices(n))
 do iteration=1,3
  do i=-7,n-8
   a(i)=real(i+8,8)*0.25d0+iteration
  enddo
  indices=-3
  b=-99.d0
  out=-101.d0
  call step(a,b,out,indices)
  write(unit)a,b,out,visits
 enddo
 deallocate(a,b,out,indices)
enddo
close(unit)
end program
'''


@pytest.fixture(scope='module', params=['child', 'prior', 'partial'], ids=str)
def binaries(tmp_path_factory, request):
    nvcc, host = shutil.which('nvcc'), shutil.which('g++-14') or shutil.which('g++')
    fortran = shutil.which('gfortran-15') or shutil.which('gfortran')
    if not all((nvcc, host, fortran)):
        pytest.skip('CUDA/C++/Fortran toolchain unavailable')
    directory = tmp_path_factory.mktemp('ordered_definitions')
    architecture = actual_profile(directory, nvcc, host)['hardware']['compute_capability'].replace('.', '')
    source = SOURCE.replace('implicit none', 'implicit none\ninteger::visits=0', 1)
    source = source.replace('call touch(b,indices)', 'call touch(b,indices)\nvisits=visits+1\nb=b*2')
    if request.param != 'child':
        source = source.replace('intent(out)::b(-3:)', 'intent(inout)::b(-3:)')
        source = source.replace('call fill(b)\ncall touch', 'call touch')
        source = source.replace('call child(b,n,.true.)', 'call fill(b)\ncall child(b,n,.true.)')
    if request.param == 'partial':
        # Legal original storage is fully initialized by the driver. The
        # compiler has only partial coverage after this GPU prefix, so its
        # conservative child obligation must publish/close before native work.
        source = source.replace('ubound(b,1)', 'ubound(b,1)-1')
    original, output, manifest = build(directory/'case', source)
    assert not manifest['scopes'][0]['boundaries'], manifest['boundaries']
    objects = []
    flags = [fortran, '-O3', '-std=f2018', '-fopenmp', '-fcheck=all,array-temps']
    for role in ('common_runtime', 'shared_entry', 'original_source'):
        for item in manifest['build_sources']:
            if item['role'] != role:
                continue
            artifact = output/item['path']
            target = artifact.with_suffix('.o')
            compiler = ([nvcc, '-O2', '-std=c++17', '-ccbin', host, '-arch=sm_'+architecture, '-Xcompiler=-fopenmp']
                        if item['language'] == 'cuda' else flags)
            run([*compiler, '-I', str(output), '-c', str(artifact), '-o', str(target)], cwd=output)
            objects.append(str(target))
    driver = output/'driver.f90'; driver.write_text(DRIVER)
    candidate, native = output/'candidate', output/'native'
    run([*flags, str(driver), *objects, '-L/usr/local/cuda/lib64', '-Wl,-rpath,/usr/local/cuda/lib64',
         '-lcudart', '-lstdc++', '-o', str(candidate)], cwd=output)
    run([*flags, str(original), str(driver), '-o', str(native)], cwd=output)
    run([str(native)], cwd=output, env={**os.environ, 'OMP_NUM_THREADS': '4', 'OMP_DYNAMIC': 'FALSE'})
    return candidate, output, (output/'fields.bin').read_bytes()


@pytest.mark.cuda
@pytest.mark.parametrize('disabled', [False, True])
def test_reached_definition_coverage_and_native_publication(binaries, disabled):
    candidate, directory, expected = binaries
    environment = {**os.environ, 'OMP_NUM_THREADS': '4', 'OMP_DYNAMIC': 'FALSE', 'FORT_RUNTIME_TRACE': '1'}
    if disabled:
        environment['CUDA_VISIBLE_DEVICES'] = '-1'
    result = run([str(candidate)], cwd=directory, env=environment)
    assert (directory/'fields.bin').read_bytes() == expected
    assert 'array temporary' not in result.stderr.lower()
    assert ('FORT_SCOPED launch ' in result.stderr) is (not disabled)
