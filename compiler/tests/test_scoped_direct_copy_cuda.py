"""DIRECT uses exact DMA operations and preserves sparse mirrors and host holes."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

RUNTIME = Path(__file__).resolve().parents[1] / 'runtime'
CLIENT = r'''
#include "scoped_runtime.h"
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
void check(int s){if(s){std::fprintf(stderr,"%d: %s\n",s,fort_scope_error());std::abort();}}
void ck(cudaError_t s){if(s!=cudaSuccess){std::fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(s));std::abort();}}
constexpr size_t X=134,Y=70,Z=70,N=X*Y*Z;
__global__ void fill(double *p){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<N)p[i]=1000000.0+i;}
__global__ void fill_bytes(unsigned char *p,size_t n){size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n)p[i]=i%251+1;}
void odd_byte_stripes(){
  constexpr size_t a=135,b=71,d=71,n=a*b*d;
  std::vector<unsigned char> host(n,0),got(n);fort_scope_t c=0;fort_buffer_t buffer=0;
  check(fort_scope_create(0,&c));size_t shape[]={a,b,d};int64_t lower[]={-5,-9,100};
  fort_scope_layout layout{3,FORT_SCOPE_BYTES,1,host.data(),shape,lower,1};check(fort_scope_register(c,1,1,&layout,1,&buffer));
  int previous=0;void *raw=nullptr;check(fort_scope_gpu_enter(c,&previous,&raw));auto stream=static_cast<cudaStream_t>(raw);
  fort_scope_access overwrite_all{};overwrite_all.flags=FORT_SCOPE_WRITE_ALL|FORT_SCOPE_OVERWRITE_ALL;
  unsigned char *device=nullptr;check(fort_scope_device_begin(c,buffer,&overwrite_all,reinterpret_cast<void **>(&device)));
  fill_bytes<<<(n+255)/256,256,0,stream>>>(device,n);ck(cudaGetLastError());check(fort_scope_note_launch(c));check(fort_scope_device_end(c,buffer));
  size_t lo[20][3],hi[20][3];fort_scope_section sections[20];
  for(size_t k=0;k<20;++k){lo[k][0]=k*6+1;lo[k][1]=lo[k][2]=0;hi[k][0]=k*6+2;hi[k][1]=b;hi[k][2]=d;sections[k]={lo[k],hi[k]};}
  fort_scope_access reads{0,20,sections,0,nullptr,0,nullptr};check(fort_scope_host_begin(c,buffer,&reads));check(fort_scope_host_end(c,buffer));
  for(size_t i=0;i<n;++i){size_t x=i%a;bool selected=x>=1&&x<=115&&(x-1)%6==0;assert(host[i]==(selected?i%251+1:0));}
  fort_scope_stats first{};check(fort_scope_stats_get(c,&first));assert(first.downloads==20&&first.download_bytes==20*b*d);
  fort_scope_access writes{0,0,nullptr,20,sections,20,sections};check(fort_scope_host_begin(c,buffer,&writes));
  for(size_t i=0;i<n;++i){size_t x=i%a;if(x>=1&&x<=115&&(x-1)%6==0)host[i]=255-i%251;}
  check(fort_scope_host_end(c,buffer));check(fort_scope_device_begin(c,buffer,&reads,reinterpret_cast<void **>(&device)));check(fort_scope_device_end(c,buffer));check(fort_scope_wait(c));
  ck(cudaMemcpy(got.data(),device,n,cudaMemcpyDeviceToHost));
  for(size_t i=0;i<n;++i){size_t x=i%a;bool selected=x>=1&&x<=115&&(x-1)%6==0;assert(got[i]==(selected?255-i%251:i%251+1));}
  fort_scope_stats final{};check(fort_scope_stats_get(c,&final));assert(final.uploads==20&&final.upload_bytes==20*b*d);
  check(fort_scope_gpu_leave(c,previous));check(fort_scope_close(c));
}
bool halo(size_t x,size_t y,size_t z){return x<3||x>=131||y<3||y>=67||z<2||z>=68;}
fort_scope_stats stats(fort_scope_t c){fort_scope_stats s{};check(fort_scope_stats_get(c,&s));return s;}
int main(int argc,char **argv){
  assert(argc==2);int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess||!devices)return 77;
  if(!std::strcmp(argv[1],"bytes")){odd_byte_stripes();return 0;}
  bool budget=!std::strcmp(argv[1],"budget");
  std::vector<double> host(N,-7),got(N);fort_scope_t c=0;fort_buffer_t b=0;
  check(fort_scope_create(0,&c));if(budget)check(fort_scope_set_device_budget(c,N*sizeof(double)));
  size_t shape[]={X,Y,Z};int64_t lower[]={-10,2,-100};fort_scope_layout layout{3,FORT_SCOPE_REAL64,8,host.data(),shape,lower,1};
  check(fort_scope_register(c,1,1,&layout,1,&b));
  int previous=0;void *raw=nullptr;check(fort_scope_gpu_enter(c,&previous,&raw));auto stream=static_cast<cudaStream_t>(raw);
  fort_scope_access write_all{};write_all.flags=FORT_SCOPE_WRITE_ALL|FORT_SCOPE_OVERWRITE_ALL;
  double *device=nullptr;check(fort_scope_device_begin(c,b,&write_all,reinterpret_cast<void **>(&device)));
  fill<<<(N+255)/256,256,0,stream>>>(device);ck(cudaGetLastError());check(fort_scope_note_launch(c));check(fort_scope_device_end(c,b));
  size_t lo[]={3,3,2},hi[]={131,67,68};fort_scope_section interior{lo,hi};
  fort_scope_access overwrite{0,0,nullptr,1,&interior,1,&interior};
  check(fort_scope_host_begin(c,b,&overwrite));
  for(size_t z=2;z<68;++z)for(size_t y=3;y<67;++y)for(size_t x=3;x<131;++x)host[x+X*(y+Y*z)]=-42;
  check(fort_scope_host_end(c,b));
  check(fort_scope_observe_v1(c,&b,1));auto first=stats(c);
  const size_t bytes=(N-size_t(128)*64*66)*sizeof(double);
  assert(first.download_bytes==bytes && first.downloads==6);
  for(size_t z=0;z<Z;++z)for(size_t y=0;y<Y;++y)for(size_t x=0;x<X;++x){size_t i=x+X*(y+Y*z);assert(host[i]==(halo(x,y,z)?1000000.0+i:-42));}
  fort_scope_access read_all{};read_all.flags=FORT_SCOPE_READ_ALL;
  // Upload the newer interior once, then both mirrors survive arbitrarily
  // interleaved read-only consumers without extra transfers.
  check(fort_scope_device_begin(c,b,&read_all,reinterpret_cast<void **>(&device)));check(fort_scope_device_end(c,b));check(fort_scope_wait(c));
  auto mirrored=stats(c);
  for(int n=0;n<20;++n){check(fort_scope_host_begin(c,b,&read_all));check(fort_scope_host_end(c,b));check(fort_scope_device_begin(c,b,&read_all,reinterpret_cast<void **>(&device)));check(fort_scope_device_end(c,b));}
  auto after=stats(c);assert(after.uploads==mirrored.uploads&&after.downloads==mirrored.downloads);
  size_t lows[6][3]={{0,0,0},{131,0,0},{3,0,0},{3,67,0},{3,3,0},{3,3,68}};
  size_t highs[6][3]={{3,Y,Z},{X,Y,Z},{131,3,Z},{131,Y,Z},{131,67,2},{131,67,Z}};
  fort_scope_section faces[6];for(int n=0;n<6;++n)faces[n]={lows[n],highs[n]};
  fort_scope_access halo_write{0,0,nullptr,6,faces,6,faces};check(fort_scope_host_begin(c,b,&halo_write));
  for(size_t z=0;z<Z;++z)for(size_t y=0;y<Y;++y)for(size_t x=0;x<X;++x)if(halo(x,y,z)){size_t i=x+X*(y+Y*z);host[i]=2000000.0+i;}
  check(fort_scope_host_end(c,b));auto before=stats(c);
  check(fort_scope_device_begin(c,b,&read_all,reinterpret_cast<void **>(&device)));check(fort_scope_device_end(c,b));check(fort_scope_wait(c));
  auto uploaded=stats(c);assert(uploaded.upload_bytes-before.upload_bytes==bytes);assert(uploaded.uploads-before.uploads==6);
  ck(cudaMemcpy(got.data(),device,N*sizeof(double),cudaMemcpyDeviceToHost));assert(got==host);
  fort_scope_scratch_stats_v1 scratch{};check(fort_scope_scratch_stats_get_v1(c,&scratch));
  assert(!scratch.capacity_bytes&&!scratch.allocations&&!scratch.reuses);
  fort_scope_transfer_stats transport{};check(fort_scope_transfer_stats_get_v1(c,&transport));
  assert(!transport.staging_allocations&&!transport.packed_bytes&&!transport.unpacked_bytes&&!transport.events&&!transport.event_waits);
  check(fort_scope_gpu_leave(c,previous));check(fort_scope_close(c));std::puts("exact direct copies accepted");
}
'''


@pytest.fixture(scope='module')
def direct_executable(tmp_path_factory):
    nvcc = shutil.which('nvcc')
    host = shutil.which('g++-14') or shutil.which('g++')
    if not nvcc or not host:
        pytest.skip('CUDA toolkit unavailable')
    directory = tmp_path_factory.mktemp('scoped_direct_copy')
    client = directory / 'client.cu'
    client.write_text(CLIENT)
    target = directory / 'run'
    result = subprocess.run([nvcc, '-O3', '-std=c++17', '-ccbin', host, '-arch=sm_86',
                             '-DFORT_SCOPE_TEST_FAULTS', '-I', str(RUNTIME), str(client),
                             str(RUNTIME / 'scoped_runtime.cu'), '-o', str(target)],
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    return target


@pytest.mark.cuda
@pytest.mark.parametrize('mode', ['normal', 'bytes', 'budget'])
def test_direct_transport_exact_mirrors_without_staging(direct_executable, mode):
    result = subprocess.run([str(direct_executable), mode], capture_output=True, text=True,
                            env={**os.environ, 'FORT_RUNTIME_TRACE': '1'}, timeout=30)
    if result.returncode == 77:
        pytest.skip('CUDA device unavailable')
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"event":"compact_transfer"' not in result.stderr
