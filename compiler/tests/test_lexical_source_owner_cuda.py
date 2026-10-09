"""Lexical escape publishes a GPU prefix and never replays its original body."""

import os
import re
import shutil

import pytest

from compiler.tests.test_lexical_source_owner import (
    SOURCE, INTERNAL_SOURCE, ASSOCIATE_SOURCE, GUARDED_CHILD_SOURCE, PROTECTED_SOURCE,
    MODULE_SOURCE, NESTED_MODULE_SOURCE, STRIDED_MODULE_SOURCE, RETURN_MODULE_SOURCE, PAYLOAD_SOURCE,
    ALLOCATION_GUARD_SOURCE, NATIVE_ATOMIC_SOURCE, GUARDED_NATIVE_SOURCE, emit,
    FIRST_GUARD_SOURCE, FIRST_GUARD_MODULE_SOURCE,
    NATIVE_METADATA_SOURCE, METADATA_ALIAS_SOURCE,
)
from compiler.tests.test_source_scopes import FACT
from compiler.tests.test_scoped_batch_sources_cuda import actual_profile
from compiler.tests.test_source_scopes import run


DRIVER = """subroutine opaque(b,n)
use local_owner,only:visits
integer,intent(in)::n
real(8),intent(inout)::b(-2:n-3)
if(n>0) b(-2)=b(-2)+17.d0
visits=visits+100
end subroutine
program verify
use local_owner,only:step,visits
implicit none
real(8),allocatable::a(:),b(:),out(:)
integer,parameter::sizes(3)=[0,8,4096]
integer::shape,iteration,n,i,unit
logical::escape
open(newunit=unit,file='fields.bin',access='stream',form='unformatted',status='replace')
do shape=1,size(sizes)
 n=sizes(shape)
 allocate(a(-7:n+2),b(-7:n+2),out(-7:n+2))
 do iteration=1,4
  do i=lbound(a,1),ubound(a,1)
   a(i)=real(i+8,8)*0.25d0+iteration
  enddo
  b=-99.d0
  out=-101.d0
  escape=mod(iteration,2)==0
  call step(a,b,out,n,escape)
  write(unit)visits,a,b,out
 enddo
 deallocate(a,b,out)
enddo
close(unit)
end program
"""


@pytest.fixture(scope='module', params=('unknown_call','return','internal','single_if','internal_openmp',
                                     'associate','guarded_child','protected','module','nested_module','strided_module',
                                     'return_module','payload','allocation_guard','native_atomic','guarded_native',
                                     'first_guard','first_module_guard','native_metadata','metadata_alias',
                                     'configured_native','guarded_empty','guarded_reverse'))
def lexical_binary(tmp_path_factory, request):
    nvcc, host = shutil.which('nvcc'), shutil.which('g++-14') or shutil.which('g++')
    fortran = shutil.which('gfortran-15') or shutil.which('gfortran')
    if not all((nvcc,host,fortran)):
        pytest.skip('CUDA/C++/Fortran toolchain unavailable')
    directory = tmp_path_factory.mktemp('lexical_escape')
    architecture = actual_profile(directory,nvcc,host)['hardware']['compute_capability'].replace('.','')
    source = {'unknown_call':SOURCE, 'return':SOURCE.replace('call opaque(b,n)','return'),
              'internal':INTERNAL_SOURCE,
              'associate':ASSOCIATE_SOURCE, 'guarded_child':GUARDED_CHILD_SOURCE,
              'protected':PROTECTED_SOURCE,
              'module':MODULE_SOURCE,
              'nested_module':NESTED_MODULE_SOURCE,
              'strided_module':STRIDED_MODULE_SOURCE,
              'return_module':RETURN_MODULE_SOURCE,
              'allocation_guard':ALLOCATION_GUARD_SOURCE,
              'native_atomic':NATIVE_ATOMIC_SOURCE,
              'guarded_native':GUARDED_NATIVE_SOURCE,
              'guarded_empty':GUARDED_NATIVE_SOURCE,
              'guarded_reverse':GUARDED_NATIVE_SOURCE.replace('do j=1,size(items)', 'do j=size(items),1,-1'),
              'first_guard':FIRST_GUARD_SOURCE,
              'first_module_guard':FIRST_GUARD_MODULE_SOURCE,
              'native_metadata':NATIVE_METADATA_SOURCE,
              'metadata_alias':METADATA_ALIAS_SOURCE,
              'configured_native':SOURCE,
              'payload':PAYLOAD_SOURCE,
              'single_if':SOURCE.replace('if(escape) then\ncall opaque(b,n)\nendif','if(escape) call opaque(b,n)'),
              'internal_openmp':INTERNAL_SOURCE.replace('do j=-2,n-3','!$omp parallel do private(j)\ndo j=-2,n-3').replace(
                  'enddo\nend subroutine\nend subroutine','enddo\n!$omp end parallel do\nend subroutine\nend subroutine')
              }[request.param]
    captures = {'readonly_controls::coefficients': FACT} if request.param == 'protected' else None
    if request.param == 'nested_module':
        captures = {'deep_child::weights': FACT}
    if request.param == 'configured_native':
        from compiler.driver.options import CompilerOptions
        from compiler.offload.config import OffloadConfig
        from compiler.scopes.source import ScopeBuilder
        from compiler.tests.test_configured_sources import configured_native_owner
        original, document, facts = configured_native_owner(directory)
        outputs, report = ScopeBuilder([original], 'local_owner::step', facts=facts, options=CompilerOptions(),
            config=OffloadConfig(policy='sections', scope_execution='reached'), analysis_sources=document).run()
        assert not report['scopes'][0]['boundaries']
    else:
        original, outputs, report = emit(directory,source,captures=captures)
    for name,text in outputs.items():
        path = directory/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(text)
    objects=[]
    flags=[fortran,'-O3','-cpp','-std=f2018','-fopenmp','-fcheck=all,array-temps']
    for role in ('common_runtime','shared_entry','original_source'):
        for item in report['build_sources']:
            if item['role'] != role:
                continue
            artifact=directory/item['path']
            target=artifact.with_suffix('.o')
            compiler=([nvcc,'-O2','-std=c++17','-ccbin',host,'-arch=sm_'+architecture,'-Xcompiler=-fopenmp']
                      if item['language']=='cuda' else flags)
            run([*compiler,'-I',str(directory),'-c',str(artifact),'-o',str(target)],cwd=directory)
            objects.append(str(target))
    driver=directory/'driver.f90'
    driver_source = DRIVER
    if request.param == 'native_atomic':
        driver_source = driver_source.replace('sizes(3)=[0,8,4096]', 'sizes(3)=[0,8,67]')
    if request.param in {'module', 'nested_module', 'strided_module', 'return_module', 'first_module_guard'}:
        driver_source = driver_source.replace('program verify\n',
            'program verify\nuse stage_child,only:adjust,child_visits\n', 1).replace(
                'write(unit)visits,a,b,out',
                'call adjust(out,n,escape)\nwrite(unit)visits,child_visits,a,b,out')
    if request.param == 'strided_module':
        driver_source = driver_source.replace('b(-7:n+2)', 'b(-7:2*(n+10)-8)').replace(
            'call step(a,b,out,n,escape)', 'call step(a,b(::2),out,n,escape)')
    if request.param == 'allocation_guard':
        driver_source = driver_source.replace('use local_owner,only:step,visits',
            'use local_owner,only:step,visits,spare').replace('call step(a,b,out,n,escape)',
            'if(escape) allocate(spare(0))\ncall step(a,b,out,n,escape)\nif(allocated(spare)) deallocate(spare)')
    if request.param in {'guarded_native', 'guarded_empty', 'guarded_reverse'}:
        driver_source = driver_source.replace('use local_owner,only:step,visits',
            'use local_owner,only:step,visits,metadata').replace('call step(a,b,out,n,escape)',
            'if(escape) then\nallocate(metadata(2))\nmetadata%coordinate=1\nendif\n'
            'call step(a,b,out,n,escape)\nif(allocated(metadata)) deallocate(metadata)')
        if request.param in {'guarded_empty', 'guarded_reverse'}:
            driver_source = driver_source.replace('allocate(metadata(2))',
                'allocate(metadata(merge(0,2,iteration==2)))')
    if request.param == 'native_metadata':
        driver_source = driver_source.replace('use local_owner,only:step,visits',
            'use local_owner,only:step,visits,metadata').replace('call step(a,b,out,n,escape)',
            'if(escape) then\nallocate(metadata(2))\nmetadata%coordinate=-2\nmetadata%weight=3.5d0\nendif\n'
            'call step(a,b,out,n,escape)\nif(allocated(metadata)) deallocate(metadata)')
    if request.param == 'metadata_alias':
        driver_source = driver_source.replace('program verify\n', 'program verify\nuse iso_c_binding\n').replace(
            'use local_owner,only:step,visits', 'use local_owner,only:step,visits,metadata').replace(
            'real(8),allocatable::a(:),b(:),out(:)', 'real(8),allocatable::a(:),out(:)\nreal(8),pointer::b(:)').replace(
            'allocate(a(-7:n+2),b(-7:n+2),out(-7:n+2))',
            'allocate(a(-7:n+2),metadata(n+10),out(-7:n+2))\ncall c_f_pointer(c_loc(metadata),b,[n+10])').replace(
            'deallocate(a,b,out)', 'nullify(b)\ndeallocate(a,metadata,out)')
    driver.write_text(driver_source)
    candidate,native=directory/'candidate',directory/'native'
    run([*flags,str(driver),*objects,'-L/usr/local/cuda/lib64','-Wl,-rpath,/usr/local/cuda/lib64',
         '-lcudart','-lstdc++','-o',str(candidate)],cwd=directory)
    run([*flags,str(original),str(driver),'-o',str(native)],cwd=directory)
    env={**os.environ,'OMP_NUM_THREADS':'4','OMP_DYNAMIC':'FALSE'}
    run([str(native)],cwd=directory,env=env)
    return candidate,directory,(directory/'fields.bin').read_bytes(),request.param


@pytest.mark.cuda
@pytest.mark.parametrize('disabled',[False,True])
def test_escape_after_gpu_prefix_preserves_complete_storage_and_side_effects(lexical_binary,disabled):
    candidate,directory,expected,case=lexical_binary
    env={**os.environ,'OMP_NUM_THREADS':'4','OMP_DYNAMIC':'FALSE','FORT_RUNTIME_TRACE':'1'}
    if disabled:
        env['CUDA_VISIBLE_DEVICES']='-1'
    result=run([str(candidate)],cwd=directory,env=env)
    assert (directory/'fields.bin').read_bytes()==expected
    assert ('array temporary' in result.stderr.lower()) is (case == 'strided_module')
    assert ('FORT_SCOPED launch ' in result.stderr) is (not disabled)
    if case == 'metadata_alias' and not disabled:
        # Registration detects the overlapping host-only metadata range after
        # the producer. Publish that prefix once and leave the consumer native;
        # the runtime's last-error string is not printed by this safe fallback.
        assert result.stderr.count('FORT_SCOPED launch ') == 8
        downloaded = sum(map(int, re.findall(r'FORT_SCOPED download .*bytes=(\d+)', result.stderr)))
        assert downloaded == 4*(8+4096)*8
    if case == 'configured_native' and not disabled:
        # Both numerical regions run, across the unchanged original team. The
        # native read/write publishes b; its consumer uploads those new values.
        assert result.stderr.count('FORT_SCOPED launch ') == 16
    if case in {'guarded_empty', 'guarded_reverse'} and not disabled:
        # The allocated empty metadata loop retains the producer for its
        # consumer. The nonempty correction publishes once and stays native.
        assert result.stderr.count('FORT_SCOPED launch ') == 14
    if case == 'return_module' and not disabled:
        # Two child calls, including their original early returns, must retain
        # the producer's values until the outer consumer. Only a is uploaded;
        # b and out are published once at the end of each owning invocation.
        uploaded = sum(map(int, re.findall(r'FORT_SCOPED upload .*bytes=(\d+)', result.stderr)))
        downloaded = sum(map(int, re.findall(r'FORT_SCOPED download .*bytes=(\d+)', result.stderr)))
        assert uploaded == 4*(8+4096)*8
        assert downloaded == 2*uploaded
