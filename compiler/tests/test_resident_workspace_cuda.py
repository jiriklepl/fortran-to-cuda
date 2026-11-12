"""Integrated lifetime milestone: queued private work survives source returns."""
from pathlib import Path
import os
import re
import shutil
import subprocess

import pytest

RUNTIME = Path(__file__).resolve().parents[1] / "runtime"

CUDA = r'''
#include "scoped_runtime.h"
#include <cuda_runtime.h>
extern "C" int available() {
    int n=0; return cudaGetDeviceCount(&n)==cudaSuccess && n>0;
}
__global__ void fill(double *a, size_t n, double value) {
    for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x) a[i]=value;
}
__global__ void consume(const double *a, double *b, size_t n) {
    for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x) b[i]+=a[i];
}
int worker(void *user,const fort_scope_external_view_v1 *v,size_t n,void *stream,int *started) {
    *started=0;
    if(n!=2 || v[0].bytes!=v[1].bytes) return FORT_SCOPE_ARGUMENT;
    *started=1;
    fill<<<17,128,0,static_cast<cudaStream_t>(stream)>>>(static_cast<double *>(v[0].device),v[0].bytes/8,*static_cast<double *>(user));
    if(cudaGetLastError()!=cudaSuccess) return FORT_SCOPE_EXECUTION;
    consume<<<17,128,0,static_cast<cudaStream_t>(stream)>>>(static_cast<double *>(v[0].device),static_cast<double *>(v[1].device),v[0].bytes/8);
    return cudaGetLastError()==cudaSuccess ? FORT_SCOPE_OK : FORT_SCOPE_EXECUTION;
}
extern "C" int enqueue(int64_t context,int64_t work,int64_t output,double value) {
    fort_scope_access overwrite{FORT_SCOPE_WRITE_ALL|FORT_SCOPE_OVERWRITE_ALL,0,nullptr,0,nullptr,0,nullptr};
    fort_scope_access update{FORT_SCOPE_READ_ALL|FORT_SCOPE_WRITE_ALL,0,nullptr,0,nullptr,0,nullptr};
    fort_scope_external_binding_v1 bindings[]={{work,FORT_SCOPE_REAL64,3,8,overwrite},{output,FORT_SCOPE_REAL64,3,8,update}};
    fort_scope_external_call_v1 call{1,0,bindings,2,0,0};
    fort_scope_external_report_v1 report{};
    return fort_scope_external_execute_v1(context,&call,worker,&value,&report);
}
'''

FORTRAN = r'''
program test
  use iso_c_binding
  use fort_scoped_memory
  use fort_resident_domain
  implicit none
  interface
    integer(c_int) function available() bind(C)
      import c_int
    end function
    integer(c_int) function enqueue(context,work,output,value) bind(C)
      import c_int,c_int64_t,c_double
      integer(c_int64_t),value :: context,work,output
      real(c_double),value :: value
    end function
  end interface
  type(resident_domain) :: owner
  type(fort_scope_stats) :: stats
  real(c_double),pointer,contiguous :: work(:,:,:),other(:,:,:)
  real(c_double),target :: output(-2:254,4:16,-7:-5)=0
  type(c_ptr) :: address
  integer(c_int64_t) :: handle,original,result_handle,other_handle
  integer :: i
  character(20) :: mode
  if (available()==0) stop 77
  call get_command_argument(1,mode)
  call owner%begin_epoch(0_c_int,8388608_c_size_t)
  result_handle=owner%bind_real64(output,lbound(output),.true.)
  if (trim(mode)=='shape') then
    handle=owner%bind_real64(output(:,:,-7:-6),lbound(output),.true.)
    error stop 'shape change accepted'
  endif
  do i=1,40
    call owner%acquire_work_real64_3(17_c_int64_t,lbound(output),ubound(output),work)
    if (any(lbound(work)/=lbound(output)).or.any(ubound(work)/=ubound(output))) error stop 'bounds'
    handle=owner%bind_real64(work,lbound(work),.false.,persistent=.true.)
    if (i==1) then
      original=handle; address=c_loc(work)
    endif
    if (handle/=original.or..not.c_associated(address,c_loc(work))) error stop 'not reused'
    if (trim(mode)=='overlap') then
      call owner%acquire_work_real64_3(17_c_int64_t,lbound(output),ubound(output),other)
      error stop 'overlap accepted'
    endif
    call resident_require(enqueue(owner%context,handle,result_handle,real(i,c_double)))
    call owner%release_work_real64_3(17_c_int64_t,work)
    if (associated(work)) error stop 'borrow not revoked'
  enddo
  call resident_require(fort_scope_stats_get(owner%context,stats))
  if (stats%allocations/=2.or.stats%uploads/=1.or.stats%downloads/=0.or.stats%waits/=0) then
    print *,stats
    error stop 'unexpected allocation, payload movement or synchronization'
  endif
  ! Different layout and different site cannot alias the old allocation.
  call owner%acquire_work_real64_3(17_c_int64_t,[1,1,1],[5,3,2],other)
  if (c_associated(address,c_loc(other))) error stop 'shape alias'
  call owner%release_work_real64_3(17_c_int64_t,other)
  call owner%acquire_work_real64_3(18_c_int64_t,lbound(output),ubound(output),other)
  other_handle=owner%bind_real64(other,lbound(other),.false.,persistent=.true.)
  if (other_handle==original) error stop 'site alias'
  call owner%release_work_real64_3(18_c_int64_t,other)
  call owner%end_epoch()
  if (any(output/=820)) error stop 'queued consumer lost previous values'
  call resident_require(fort_scope_stats_get(owner%context,stats))
  if (stats%downloads/=1.or.stats%download_bytes/=size(output)*8) error stop 'private workspace downloaded'
  call owner%begin_epoch(0_c_int,8388608_c_size_t)
  call owner%acquire_work_real64_3(17_c_int64_t,lbound(output),ubound(output),work)
  handle=owner%bind_real64(work,lbound(work),.false.,persistent=.true.)
  if (handle/=original) error stop 'workspace lost at epoch boundary'
  result_handle=owner%bind_real64(output,lbound(output),.true.)
  call resident_require(enqueue(owner%context,handle,result_handle,2.0_c_double))
  call owner%release_work_real64_3(17_c_int64_t,work)
  call owner%close()
  if (any(output/=822).or.owner%active.or.owner%domain/=0) error stop 'close ordering'
  print *, 'PASS domain-owned workspace'
end program
'''


@pytest.mark.cuda
def test_domain_owned_workspace(tmp_path):
    nvcc = shutil.which("nvcc")
    host = shutil.which("g++-14") or shutil.which("g++")
    fc = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not all((nvcc, host, fc)):
        pytest.skip("CUDA/Fortran toolchain unavailable")
    cuda, source = tmp_path / "worker.cu", tmp_path / "main.f90"
    cuda.write_text(CUDA)
    source.write_text(FORTRAN)

    def run(command):
        result = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True, timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr

    objects = []
    for path in (cuda, RUNTIME / "scoped_runtime.cu"):
        obj = tmp_path / (path.stem + ".o")
        run([nvcc, "-std=c++17", "-arch=sm_86", "-ccbin", host, "-I", str(RUNTIME),
             "-c", str(path), "-o", str(obj)])
        objects.append(str(obj))
    for path in (RUNTIME / "scoped_memory.f90", RUNTIME / "resident_domain.f90", source):
        obj = tmp_path / (path.stem + ".o")
        run([fc, "-ffree-line-length-none", "-fcheck=all", "-c", str(path), "-o", str(obj)])
        objects.append(str(obj))
    binary = tmp_path / "run"
    run([nvcc, "-ccbin", host, *objects, "-lgfortran", "-o", str(binary)])
    env = {**os.environ, "FORT_RUNTIME_TRACE": "1"}
    result = subprocess.run([str(binary)], env=env, text=True, capture_output=True, timeout=30)
    if result.returncode == 77:
        pytest.skip("CUDA device unavailable")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS domain-owned workspace" in result.stdout
    downloads = re.findall(r"FORT_SCOPED download .*?bytes=(\d+)", result.stderr)
    assert [int(n) for n in downloads] == [257 * 13 * 3 * 8] * 2
    rejected = subprocess.run([str(binary), "overlap"], text=True, capture_output=True, timeout=30)
    assert rejected.returncode != 0
    assert "overlapping use of a compiler workspace slot" in rejected.stderr
    rejected = subprocess.run([str(binary), "shape"], text=True, capture_output=True, check=False, timeout=30)
    assert rejected.returncode != 0
    assert "storage shape changed inside an epoch" in rejected.stderr
    assert re.search(r"Registered extents:\s+257\s+13\s+3", rejected.stderr)
    assert re.search(r"Requested extents:\s+257\s+13\s+2", rejected.stderr)
