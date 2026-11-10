"""Milestone coverage for scalar reductions over coherent device allocations."""
from pathlib import Path
import shutil
import subprocess

import pytest

RUNTIME = Path(__file__).resolve().parents[1] / "runtime"
PROGRAM = r'''
#include "reductions.h"
#include <cuda_runtime.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>
void check(int status) {
    if(status) { std::fprintf(stderr,"status=%d: %s\n",status,fort_scope_error()); std::abort(); }
}
__global__ void overwrite(double *values,size_t count) {
    for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=gridDim.x*blockDim.x)
        values[i]=i==count-1?81.0:-double(i);
}
int produce(void *,const fort_scope_external_view_v1 *views,size_t count,void *stream,int *started) {
    *started=0;if(count!=1) return FORT_SCOPE_ARGUMENT;
    *started=1;
    overwrite<<<13,256,0,static_cast<cudaStream_t>(stream)>>>(static_cast<double *>(views[0].device),views[0].bytes/8);
    return cudaGetLastError()==cudaSuccess?FORT_SCOPE_OK:FORT_SCOPE_EXECUTION;
}
int main() {
    int devices=0;
    if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices) return 77;
    const double inf=std::numeric_limits<double>::infinity(),nan=std::numeric_limits<double>::quiet_NaN();
    for(size_t n:{size_t(1),size_t(257),size_t(1048579)}) {
        std::vector<double> values(n,-7.0);values.back()=-2.0;
        size_t extent[1]={n};int64_t lower[1]={-9};fort_scope_t c=0;fort_buffer_t b=0;
        check(fort_scope_create(0,&c));
        fort_scope_layout layout{1,FORT_SCOPE_REAL64,8,values.data(),extent,lower,1};
        check(fort_scope_register(c,1,1,&layout,1,&b));
        double maximum=0;int32_t any_nan=-1;fort_scope_external_report_v1 report{};
        for(int repeat=0;repeat<2;++repeat) {
            check(fort_scope_max_real64_v1(c,b,1,n,-inf,&maximum,&any_nan,&report));
            assert(maximum==-2 && any_nan==0 && report.started && report.committed);
        }
        fort_scope_stats stats{};check(fort_scope_stats_get(c,&stats));
        assert(stats.uploads==1 && stats.downloads==0);
        check(fort_scope_observe_v1(c,&b,1));
        assert(values.back()==-2);check(fort_scope_stats_get(c,&stats));assert(stats.downloads==0);
        maximum=13;any_nan=5;
        assert(fort_scope_max_real64_v1(c,b,1,n+1,0,&maximum,&any_nan,&report)==FORT_SCOPE_ARGUMENT);
        assert(!report.started && !report.committed && maximum==13 && any_nan==5);
        fort_scope_access write{FORT_SCOPE_WRITE_ALL|FORT_SCOPE_OVERWRITE_ALL,0,nullptr,0,nullptr,0,nullptr};
        fort_scope_external_binding_v1 binding{b,FORT_SCOPE_REAL64,1,8,write};
        fort_scope_external_call_v1 call{1,0,&binding,1,0,0};
        check(fort_scope_external_execute_v1(c,&call,produce,nullptr,&report));
        check(fort_scope_max_real64_v1(c,b,1,n,0,&maximum,&any_nan,&report));
        assert(maximum==81 && !any_nan && values.back()==-2); // scalar only, queued producer honored
        check(fort_scope_stats_get(c,&stats));assert(stats.uploads==1 && stats.downloads==0);
        check(fort_scope_observe_v1(c,&b,1));assert(values.back()==81);
        fort_scope_access host_write{FORT_SCOPE_WRITE_ALL|FORT_SCOPE_OVERWRITE_ALL,0,nullptr,0,nullptr,0,nullptr};
        check(fort_scope_host_begin(c,b,&host_write));values.assign(n,nan);check(fort_scope_host_end(c,b));
        check(fort_scope_max_real64_v1(c,b,1,n,-inf,&maximum,&any_nan,&report));
        assert(maximum==-inf && any_nan==1);
        check(fort_scope_host_begin(c,b,&host_write));values.back()=inf;check(fort_scope_host_end(c,b));
        check(fort_scope_max_real64_v1(c,b,1,n,-inf,&maximum,&any_nan,&report));
        assert(maximum==inf && any_nan==(n>1));
        check(fort_scope_close(c));
    }
    fort_scope_t c=0;fort_buffer_t b=0;double value=3,maximum=9;int32_t any_nan=7;
    size_t extent=1;int64_t lower=1;fort_scope_external_report_v1 report{};
    check(fort_scope_create(0,&c));check(fort_scope_set_device_budget(c,8));
    fort_scope_layout layout{1,FORT_SCOPE_REAL64,8,&value,&extent,&lower,1};
    check(fort_scope_register(c,2,1,&layout,1,&b));
    assert(fort_scope_max_real64_v1(c,b,1,1,0,&maximum,&any_nan,&report)==FORT_SCOPE_RESOURCE);
    assert(!report.started && !report.committed && maximum==9 && any_nan==7);
    check(fort_scope_close(c));
    std::puts("PASS resident max: ordering, replicas, NaN/infinity, irregular sizes, rejection and budget");
}
'''


@pytest.mark.cuda
def test_resident_maximum(tmp_path):
    nvcc, host = shutil.which("nvcc"), shutil.which("g++-14") or shutil.which("g++")
    if not nvcc or not host:
        pytest.skip("CUDA toolkit unavailable")
    source = tmp_path / "main.cu"
    source.write_text(PROGRAM)
    objects = []
    for path in (source, RUNTIME / "reductions.cu", RUNTIME / "scoped_runtime.cu"):
        target = tmp_path / (path.stem + ".o")
        built = subprocess.run([nvcc, "-std=c++17", "-arch=sm_86", "-ccbin", host, "-I", str(RUNTIME),
                                "-c", str(path), "-o", str(target)], capture_output=True, text=True, timeout=180)
        assert built.returncode == 0, built.stdout + built.stderr
        objects.append(str(target))
    binary = tmp_path / "run"
    linked = subprocess.run([nvcc, "-ccbin", host, *objects, "-o", str(binary)],
                            capture_output=True, text=True, timeout=60)
    assert linked.returncode == 0, linked.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
    if result.returncode == 77:
        pytest.skip("CUDA device unavailable")
    assert result.returncode == 0, result.stdout + result.stderr
