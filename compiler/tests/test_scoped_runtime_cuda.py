"""Actual CUDA acceptance using separately compiled public-ABI consumers."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

RUNTIME = Path(__file__).resolve().parents[1] / "runtime"

PRODUCER = r'''
#include "scoped_runtime.h"
#include <cuda_runtime.h>
namespace {
__global__ void kernel(const double *a, double *b) {
    const int i=threadIdx.x;
    if (i<8) b[i]=2*a[i];
}
}
extern "C" int producer(fort_scope_t context, fort_buffer_t a, fort_buffer_t b) {
    fort_scope_access read{}; read.flags=FORT_SCOPE_READ_ALL;
    fort_scope_access write{}; write.flags=FORT_SCOPE_WRITE_ALL|FORT_SCOPE_OVERWRITE_ALL;
    void *pa=nullptr, *pb=nullptr, *stream=nullptr;
    int previous=0, status=0;
    if ((status=fort_scope_device_begin(context,a,&read,&pa))) return status;
    if ((status=fort_scope_device_begin(context,b,&write,&pb))) return status;
    if ((status=fort_scope_gpu_enter(context,&previous,&stream))) return status;
    kernel<<<1,32,0,static_cast<cudaStream_t>(stream)>>>(static_cast<double *>(pa),static_cast<double *>(pb));
    if (cudaGetLastError()!=cudaSuccess) return FORT_SCOPE_EXECUTION;
    if ((status=fort_scope_note_launch(context))) return status;
    if ((status=fort_scope_device_end(context,a))) return status;
    if ((status=fort_scope_device_end(context,b))) return status;
    return fort_scope_gpu_leave(context,previous);
}
'''

CONSUMER = r'''
#include "scoped_runtime.h"
#include <cuda_runtime.h>
namespace {
__global__ void kernel(const double *a, const double *b, double *out) {
    const int i=threadIdx.x;
    if (i<8) out[i]=a[i]+b[i];
}
}
extern "C" int consumer(fort_scope_t context, fort_buffer_t a, fort_buffer_t b, fort_buffer_t output) {
    fort_scope_access read{}; read.flags=FORT_SCOPE_READ_ALL;
    fort_scope_access write{}; write.flags=FORT_SCOPE_WRITE_ALL|FORT_SCOPE_OVERWRITE_ALL;
    void *pa=nullptr, *pb=nullptr, *po=nullptr, *stream=nullptr;
    int previous=0, status=0;
    if ((status=fort_scope_device_begin(context,a,&read,&pa))) return status;
    if ((status=fort_scope_device_begin(context,b,&read,&pb))) return status;
    if ((status=fort_scope_device_begin(context,output,&write,&po))) return status;
    if ((status=fort_scope_gpu_enter(context,&previous,&stream))) return status;
    kernel<<<1,32,0,static_cast<cudaStream_t>(stream)>>>(static_cast<double *>(pa),static_cast<double *>(pb),static_cast<double *>(po));
    if (cudaGetLastError()!=cudaSuccess) return FORT_SCOPE_EXECUTION;
    if ((status=fort_scope_note_launch(context))) return status;
    if ((status=fort_scope_device_end(context,a))) return status;
    if ((status=fort_scope_device_end(context,b))) return status;
    if ((status=fort_scope_device_end(context,output))) return status;
    return fort_scope_gpu_leave(context,previous);
}
'''

MAIN = r'''
#include "scoped_runtime.h"
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
extern "C" int producer(fort_scope_t,fort_buffer_t,fort_buffer_t);
extern "C" int consumer(fort_scope_t,fort_buffer_t,fort_buffer_t,fort_buffer_t);
void check(int status) {
    if (status) { std::fprintf(stderr,"status=%d: %s\n",status,fort_scope_error()); std::abort(); }
}
__global__ void interior(double *array) {
    int n=blockIdx.x*blockDim.x+threadIdx.x;
    if (n>=256) return;
    const int i=n%4, j=n/4%4, k=n/16%4, l=n/64;
    if (i>=1 && i<3 && j>=1 && j<3 && k>=1 && k<3 && l==1) array[n]+=1000;
}
void chain() {
    double a[8], b[8]={}, out[8]={};
    for (int i=0;i<8;++i) a[i]=i+1;
    fort_scope_t context=0;
    check(fort_scope_create(0,&context));
    size_t extents[1]={8}; int64_t bounds[1]={-2};
    fort_scope_layout layout{1,FORT_SCOPE_REAL64,8,a,extents,bounds,1};
    fort_buffer_t ha=0,hb=0,ho=0;
    check(fort_scope_register(context,1,1,&layout,1,&ha));
    layout.host=b; check(fort_scope_register(context,2,1,&layout,0,&hb));
    layout.host=out; check(fort_scope_register(context,3,1,&layout,0,&ho));
    check(producer(context,ha,hb));
    check(consumer(context,ha,hb,ho));
    fort_scope_stats stats{};
    check(fort_scope_stats_get(context,&stats));
    assert(stats.allocations==3 && stats.uploads==1 && stats.downloads==0 && stats.launches==2);
    fort_scope_access read{}; read.flags=FORT_SCOPE_READ_ALL;
    check(fort_scope_host_begin(context,hb,&read));
    for (int i=0;i<8;++i) assert(b[i]==2*a[i]);
    check(fort_scope_host_end(context,hb));
    check(consumer(context,ha,hb,ho));
    check(fort_scope_stats_get(context,&stats));
    assert(stats.uploads==1 && stats.downloads==1 && stats.launches==3);
    fort_scope_access update{}; update.flags=FORT_SCOPE_READ_ALL|FORT_SCOPE_WRITE_ALL;
    check(fort_scope_host_begin(context,hb,&update));
    for (int i=0;i<8;++i) b[i]*=10; // Original native intermediate transform.
    check(fort_scope_host_end(context,hb));
    check(consumer(context,ha,hb,ho));
    check(fort_scope_stats_get(context,&stats));
    assert(stats.uploads==2 && stats.downloads==1 && stats.launches==4);
    assert(stats.upload_bytes==128 && stats.download_bytes==64);
    check(fort_scope_close(context));
    for (int i=0;i<8;++i) assert(out[i]==21*a[i]);
    assert(fort_scope_wait(context)==FORT_SCOPE_STALE);
    std::puts("independent modules: correct; shared uploads=1; CPU read reuploads=0; transform uploads=1");
}
void sections() {
    double host[256];
    for (int i=0;i<256;++i) host[i]=i;
    size_t extents[4]={4,4,4,4}; int64_t bounds[4]={-2,-3,-4,-5};
    fort_scope_layout layout{4,FORT_SCOPE_REAL64,8,host,extents,bounds,1};
    fort_scope_t context=0; fort_buffer_t buffer=0;
    check(fort_scope_create(0,&context));
    check(fort_scope_register(context,1,1,&layout,1,&buffer));
    size_t lo[4]={1,1,1,1}, hi[4]={3,3,3,2};
    fort_scope_section section{lo,hi};
    fort_scope_access work{}; work.read_count=work.write_count=1; work.reads=work.writes=&section;
    void *pointer=nullptr, *stream=nullptr; int previous=0;
    check(fort_scope_device_begin(context,buffer,&work,&pointer));
    check(fort_scope_gpu_enter(context,&previous,&stream));
    interior<<<2,128,0,static_cast<cudaStream_t>(stream)>>>(static_cast<double *>(pointer));
    assert(cudaGetLastError()==cudaSuccess);
    check(fort_scope_note_launch(context));
    check(fort_scope_device_end(context,buffer));
    check(fort_scope_gpu_leave(context,previous));
    fort_scope_access read{}; read.flags=FORT_SCOPE_READ_ALL;
    check(fort_scope_host_begin(context,buffer,&read));
    check(fort_scope_host_end(context,buffer));
    for (int n=0;n<256;++n) {
        int i=n%4,j=n/4%4,k=n/16%4,l=n/64;
        bool inside=i>=1 && i<3 && j>=1 && j<3 && k>=1 && k<3 && l==1;
        assert(host[n]==n+(inside?1000:0));
    }
    fort_scope_stats stats{};
    check(fort_scope_stats_get(context,&stats));
    assert(stats.upload_bytes==64 && stats.download_bytes==64 && stats.uploads==1 && stats.downloads==1);
    check(fort_scope_close(context));
    std::puts("4D pitched copies: correct; H2D=64 bytes; D2H=64 bytes; one copy each");
}
int main(int argc,char **argv) {
    if (argc>1 && !std::strcmp(argv[1],"lazy")) {
        fort_scope_t context=0; check(fort_scope_create(0,&context));
        double a[8]={}; size_t shape[1]={8}; int64_t bounds[1]={0};
        fort_scope_layout layout{1,FORT_SCOPE_REAL64,8,a,shape,bounds,1}; fort_buffer_t buffer=0;
        check(fort_scope_register(context,1,1,&layout,1,&buffer));
        fort_scope_access read{}; read.flags=FORT_SCOPE_READ_ALL;
        check(fort_scope_host_begin(context,buffer,&read)); check(fort_scope_host_end(context,buffer));
        size_t empty_shape[2]={static_cast<size_t>(-1),0}; int64_t empty_bounds[2]={-2,-2};
        fort_scope_layout empty_layout{2,FORT_SCOPE_REAL64,8,nullptr,empty_shape,empty_bounds,1};
        fort_buffer_t empty_buffer=0; void *empty_pointer=nullptr;
        check(fort_scope_register(context,2,1,&empty_layout,1,&empty_buffer));
        check(fort_scope_device_begin(context,empty_buffer,&read,&empty_pointer));
        assert(!empty_pointer); check(fort_scope_device_end(context,empty_buffer));
        fort_scope_stats stats{}; check(fort_scope_stats_get(context,&stats));
        assert(stats.allocations==0 && stats.uploads==0 && stats.downloads==0 && stats.waits==0);
        check(fort_scope_close(context)); std::puts("native-only scope: no CUDA initialization"); return 0;
    }
    int count=0;
    if (cudaGetDeviceCount(&count)!=cudaSuccess || count<1) {
        std::fprintf(stderr,"CUDA device unavailable\n"); return 77;
    }
    if (argc>1 && !std::strcmp(argv[1],"probe")) return 0;
    if (argc>1 && !std::strcmp(argv[1],"sections")) sections(); else chain();
}
'''


@pytest.fixture(scope="module")
def cuda_runtime_executable(tmp_path_factory):
    nvcc = shutil.which("nvcc")
    host = shutil.which("g++-14") or shutil.which("g++")
    if not nvcc or not host:
        pytest.skip("CUDA toolkit unavailable")
    directory = tmp_path_factory.mktemp("scoped_cuda")
    objects = []
    for name, code in (("producer", PRODUCER), ("consumer", CONSUMER), ("main", MAIN), ("runtime", None)):
        source = directory / (name + ".cu")
        if code is not None:
            source.write_text(code)
        else:
            source = RUNTIME / "scoped_runtime.cu"
        target = directory / (name + ".o")
        command = [nvcc, "-std=c++17", "-ccbin", host, "-arch=sm_86", "-I", str(RUNTIME),
                   "-c", str(source), "-o", str(target)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
        objects.append(str(target))
    result = subprocess.run([nvcc, "-ccbin", host, *objects, "-o", str(directory / "run")],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return directory / "run"


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["chain", "sections", "lazy"])
def test_scoped_cuda_public_consumers(cuda_runtime_executable, mode):
    result = subprocess.run([str(cuda_runtime_executable), mode], capture_output=True, text=True, timeout=30)
    if result.returncode == 77:
        pytest.skip(result.stderr.strip())
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.cuda
def test_native_only_and_empty_scope_with_cuda_devices_hidden(cuda_runtime_executable):
    import os

    result = subprocess.run([str(cuda_runtime_executable), "lazy"], capture_output=True, text=True,
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "FORT_RUNTIME_TRACE": "1"}, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "initialize" not in result.stderr
