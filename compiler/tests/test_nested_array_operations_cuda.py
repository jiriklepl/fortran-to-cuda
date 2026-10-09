"""Complete native/GPU fields for original plane initialization and updates."""

import os
import shutil

import pytest

from compiler.tests.test_nested_array_operations import SOURCE, build
from compiler.tests.test_scoped_batch_sources_cuda import actual_profile
from compiler.tests.test_source_scopes import run


DRIVER = '''program verify
use planes,only:step,visits
implicit none
real(8),allocatable::a(:,:,:),b(:,:,:),out(:,:,:)
integer,parameter::sizes(3)=[0,8,257]
integer::shape,iteration,n,i,j,k,unit
open(newunit=unit,file='fields.bin',access='stream',form='unformatted',status='replace')
do shape=1,size(sizes)
 n=sizes(shape)
 allocate(a(-7:n-8,-4:0,6:8),b(-7:n-8,-4:0,6:8),out(-7:n-8,-4:0,6:8))
 do iteration=1,3
  do k=6,8
   do j=-4,0
    do i=-7,n-8
     a(i,j,k)=real(i+8+3*j+5*k,8)*0.25d0+iteration
    enddo
   enddo
  enddo
  b=-99.d0
  out=-101.d0
  call step(a,b,out,n)
  write(unit)a,b,out,visits
 enddo
 deallocate(a,b,out)
enddo
close(unit)
end program
'''


@pytest.fixture(scope='module', params=[False, True], ids=['explicit-end', 'implicit-end'])
def binaries(tmp_path_factory, request):
    nvcc,host = shutil.which('nvcc'),shutil.which('g++-14') or shutil.which('g++')
    fortran = shutil.which('gfortran-15') or shutil.which('gfortran')
    if not all((nvcc,host,fortran)):
        pytest.skip('CUDA/C++/Fortran toolchain unavailable')
    directory=tmp_path_factory.mktemp('nested_sections')
    architecture=actual_profile(directory,nvcc,host)['hardware']['compute_capability'].replace('.','')
    source = SOURCE.replace('implicit none', 'implicit none\ninteger::visits=0', 1)
    # The statement following an optional combined end runs once, outside the
    # original team. Verify its observable state as well as all array values.
    source = source.replace('!$omp end parallel do\nend subroutine',
                            ('!$omp end parallel do\n' if not request.param else '')+'visits=visits+1\nend subroutine')
    original,output,manifest=build(directory/'case', source)
    assert not manifest['scopes'][0]['boundaries']
    objects=[]
    flags=[fortran,'-O3','-std=f2018','-fopenmp','-fcheck=all,array-temps']
    for role in ('common_runtime','shared_entry','original_source'):
        for item in manifest['build_sources']:
            if item['role'] != role:
                continue
            artifact=output/item['path']
            target=artifact.with_suffix('.o')
            compiler=([nvcc,'-O2','-std=c++17','-ccbin',host,'-arch=sm_'+architecture,'-Xcompiler=-fopenmp']
                      if item['language']=='cuda' else flags)
            run([*compiler,'-I',str(output),'-c',str(artifact),'-o',str(target)],cwd=output)
            objects.append(str(target))
    driver=output/'driver.f90';driver.write_text(DRIVER)
    candidate,native=output/'candidate',output/'native'
    run([*flags,str(driver),*objects,'-L/usr/local/cuda/lib64','-Wl,-rpath,/usr/local/cuda/lib64',
         '-lcudart','-lstdc++','-o',str(candidate)],cwd=output)
    run([*flags,str(original),str(driver),'-o',str(native)],cwd=output)
    environment={**os.environ,'OMP_NUM_THREADS':'4','OMP_DYNAMIC':'FALSE'}
    run([str(native)],cwd=output,env=environment)
    return candidate,output,(output/'fields.bin').read_bytes()


@pytest.mark.cuda
@pytest.mark.parametrize('disabled',[False,True])
def test_full_sections_preserve_shapes_bounds_and_complete_fields(binaries,disabled):
    candidate,directory,expected=binaries
    environment={**os.environ,'OMP_NUM_THREADS':'4','OMP_DYNAMIC':'FALSE','FORT_RUNTIME_TRACE':'1'}
    if disabled:
        environment['CUDA_VISIBLE_DEVICES']='-1'
    result=run([str(candidate)],cwd=directory,env=environment)
    assert (directory/'fields.bin').read_bytes()==expected
    assert 'array temporary' not in result.stderr.lower()
    assert ('FORT_SCOPED launch ' in result.stderr) is (not disabled)
