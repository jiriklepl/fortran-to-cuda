"""Real delayed GPU work proves independent host tasks return before completion."""
from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from compiler.tests.test_scoped_runtime import RUNTIME

CLIENT = r'''
#include "scoped_runtime.h"
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
void check(int s) { if(s) { std::fprintf(stderr,"%d: %s\n",s,fort_scope_error()); std::abort(); } }
void cuda(cudaError_t s) { if(s!=cudaSuccess) { std::fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(s)); std::abort(); } }
__global__ void delayed(double *out,double value,unsigned long long cycles) {
    const auto start=clock64(); while(clock64()-start<cycles) {}
    out[0]=value;
}
__global__ void delayed_read(const double *input,double *out,unsigned long long cycles) {
    const auto start=clock64(); while(clock64()-start<cycles) {}
    out[0]=input[0];
}
fort_scope_access read_all() { fort_scope_access a{}; a.flags=FORT_SCOPE_READ_ALL; return a; }
fort_scope_access overwrite_all() { fort_scope_access a{}; a.flags=FORT_SCOPE_WRITE_ALL|FORT_SCOPE_OVERWRITE_ALL; return a; }
struct Fixture {
    fort_scope_t context=0; fort_buffer_t a=0,x=0;
    double host_a[8]{},*host_x=nullptr,*pa=nullptr,*px=nullptr;
    cudaStream_t stream=nullptr; cudaEvent_t marker=nullptr; int previous=0;
    unsigned long long cycles=0;
    Fixture() {
        cuda(cudaHostAlloc(reinterpret_cast<void **>(&host_x),8*sizeof(double),cudaHostAllocDefault));
        for(int i=0;i<8;++i) host_x[i]=10+i;
        check(fort_scope_create(0,&context));
        size_t extents[1]={8}; int64_t bounds[1]={-4};
        fort_scope_layout layout{1,FORT_SCOPE_REAL64,8,host_a,extents,bounds,1};
        check(fort_scope_register(context,1,1,&layout,1,&a));
        layout.host=host_x; check(fort_scope_register(context,2,1,&layout,1,&x));
        void *raw=nullptr; check(fort_scope_gpu_enter(context,&previous,&raw)); stream=static_cast<cudaStream_t>(raw);
        int khz=0; cuda(cudaDeviceGetAttribute(&khz,cudaDevAttrClockRate,0)); cycles=static_cast<unsigned long long>(khz)*40;
        cuda(cudaEventCreateWithFlags(&marker,cudaEventDisableTiming));
    }
    fort_scope_stats stats() { fort_scope_stats s{}; check(fort_scope_stats_get(context,&s)); return s; }
    fort_scope_completion_stats_v1 completions() {
        fort_scope_completion_stats_v1 s{}; check(fort_scope_completion_stats_get_v1(context,&s)); return s;
    }
    void start_delay() {
        auto write=overwrite_all(); check(fort_scope_device_begin(context,a,&write,reinterpret_cast<void **>(&pa)));
        delayed<<<1,1,0,stream>>>(pa,1,cycles); cuda(cudaGetLastError());
        check(fort_scope_note_launch(context)); check(fort_scope_device_end(context,a));
    }
    void mark() { cuda(cudaEventRecord(marker,stream)); }
    void pending() { assert(cudaEventQuery(marker)==cudaErrorNotReady); }
    void close() {
        check(fort_scope_gpu_leave(context,previous)); check(fort_scope_close(context));
        cuda(cudaEventDestroy(marker)); cuda(cudaFreeHost(host_x));
    }
};
int main(int argc,char **argv) {
    assert(argc==2); const char *mode=argv[1];
    if(!std::strcmp(mode,"host_only")) {
        fort_scope_t c=0; fort_buffer_t b=0; double host[8]{};
        size_t extent[1]={8}; int64_t lower[1]={0}; fort_scope_layout layout{1,FORT_SCOPE_REAL64,8,host,extent,lower,1};
        check(fort_scope_create(0,&c)); check(fort_scope_register(c,1,1,&layout,1,&b));
        for(int i=0;i<5;++i) { auto read=read_all(); check(fort_scope_host_begin(c,b,&read)); check(fort_scope_host_end(c,b)); }
        fort_scope_completion_stats_v1 s{}; check(fort_scope_completion_stats_get_v1(c,&s));
        assert(!s.created_events && !s.event_records && !s.event_waits && !s.stream_waits && s.independent_host_tasks==5);
        check(fort_scope_close(c)); return 0;
    }
    int devices=0; if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices) return 77;
    Fixture f;
    if(!std::strcmp(mode,"independent")) {
        f.start_delay(); f.mark(); const auto before=f.stats().waits;
        auto access=read_all(); access.flags|=FORT_SCOPE_WRITE_ALL;
        check(fort_scope_host_begin(f.context,f.x,&access));
        f.host_x[0]=99; check(fort_scope_host_end(f.context,f.x));
        assert(f.stats().waits==before); f.pending();
        assert(f.completions().independent_host_tasks==1);
    } else if(!std::strcmp(mode,"overlap")) {
        f.start_delay();
        auto read=read_all(),write=overwrite_all();
        check(fort_scope_device_begin(f.context,f.x,&read,reinterpret_cast<void **>(&f.px)));
        check(fort_scope_device_begin(f.context,f.a,&write,reinterpret_cast<void **>(&f.pa)));
        delayed_read<<<1,1,0,f.stream>>>(f.px,f.pa,2*f.cycles); cuda(cudaGetLastError());
        check(fort_scope_note_launch(f.context));
        check(fort_scope_device_end(f.context,f.x)); check(fort_scope_device_end(f.context,f.a));
        f.mark(); const auto before=f.stats().waits;
        check(fort_scope_host_begin(f.context,f.x,&write));
        assert(f.stats().waits==before+1); f.pending();
        f.host_x[0]=99; check(fort_scope_host_end(f.context,f.x));
        check(fort_scope_observe_v1(f.context,&f.a,1));
        assert(f.host_a[0]==10); // The host edit followed the pending DMA read.
        check(fort_scope_device_begin(f.context,f.x,&read,reinterpret_cast<void **>(&f.px)));
        check(fort_scope_device_end(f.context,f.x)); check(fort_scope_wait(f.context));
        double copied=0; cuda(cudaMemcpy(&copied,f.px,sizeof(double),cudaMemcpyDeviceToHost)); assert(copied==99);
    } else if(!std::strcmp(mode,"disjoint") || !std::strcmp(mode,"read_mirror")) {
        f.start_delay();
        size_t lo[1]={2},hi[1]={6}; fort_scope_section middle{lo,hi};
        fort_scope_access reads{0,1,&middle,0,nullptr,0,nullptr};
        check(fort_scope_device_begin(f.context,f.x,&reads,reinterpret_cast<void **>(&f.px)));
        check(fort_scope_device_end(f.context,f.x)); f.mark();
        const auto before=f.stats();
        if(!std::strcmp(mode,"disjoint")) {
            size_t first_lo[1]={0},first_hi[1]={2}; fort_scope_section first{first_lo,first_hi};
            fort_scope_access writes{0,0,nullptr,1,&first,1,&first};
            check(fort_scope_host_begin(f.context,f.x,&writes));
            f.host_x[0]=99; f.host_x[1]=98;
        } else { auto all=read_all(); check(fort_scope_host_begin(f.context,f.x,&all)); }
        check(fort_scope_host_end(f.context,f.x));
        assert(f.stats().waits==before.waits); f.pending();
        check(fort_scope_wait(f.context));
        check(fort_scope_device_begin(f.context,f.x,&reads,reinterpret_cast<void **>(&f.px)));
        check(fort_scope_device_end(f.context,f.x)); assert(f.stats().uploads==before.uploads);
        double copied=0; cuda(cudaMemcpy(&copied,f.px+2,sizeof(double),cudaMemcpyDeviceToHost)); assert(copied==12);
    } else if(!std::strcmp(mode,"downloads")) {
        auto write=overwrite_all();
        fort_scope_plan_binding bindings[2]={{f.a,write},{f.x,write}}; void *pointers[2]{};
        check(fort_scope_access_begin_v1(f.context,bindings,2,1,pointers));
        delayed<<<1,1,0,f.stream>>>(static_cast<double *>(pointers[0]),11,f.cycles);
        delayed<<<1,1,0,f.stream>>>(static_cast<double *>(pointers[1]),22,0); cuda(cudaGetLastError());
        check(fort_scope_note_launch(f.context)); check(fort_scope_note_launch(f.context));
        check(fort_scope_device_end(f.context,f.a)); check(fort_scope_device_end(f.context,f.x));
        const auto before=f.completions();
        auto read=read_all(); bindings[0].access=bindings[1].access=read;
        check(fort_scope_access_begin_v1(f.context,bindings,2,0,nullptr));
        assert(f.host_a[0]==11 && f.host_x[0]==22);
        assert(f.completions().event_records==before.event_records+1);
        assert(f.completions().event_waits==before.event_waits+1);
        assert(f.completions().stream_waits==before.stream_waits);
        check(fort_scope_host_end(f.context,f.a)); check(fort_scope_host_end(f.context,f.x));
    } else if(!std::strcmp(mode,"reuse")) {
        auto read=read_all(),write=overwrite_all();
        for(int round=0;round<3;++round) {
            check(fort_scope_device_begin(f.context,f.x,&read,reinterpret_cast<void **>(&f.px)));
            check(fort_scope_device_end(f.context,f.x)); check(fort_scope_wait(f.context));
            check(fort_scope_host_begin(f.context,f.x,&write)); f.host_x[0]=round;
            check(fort_scope_host_end(f.context,f.x));
        }
        const auto s=f.completions(); assert(s.event_records==3 && s.created_events==1 && s.reused_events==2);
    } else return 2;
    f.close(); std::puts("dependent host completion accepted");
}
'''


@pytest.fixture(scope="module")
def cuda_completion_executable(tmp_path_factory):
    nvcc = shutil.which("nvcc")
    host = shutil.which("g++-14") or shutil.which("g++")
    if not nvcc or not host:
        pytest.skip("CUDA toolkit unavailable")
    directory = tmp_path_factory.mktemp("scoped_host_completion_cuda")
    source = directory / "client.cu"
    source.write_text(CLIENT)
    objects = []
    for name, path in [("runtime", RUNTIME / "scoped_runtime.cu"), ("client", source)]:
        target = directory / (name + ".o")
        result = subprocess.run([nvcc, "-std=c++17", "-ccbin", host, "-arch=sm_86", "-I", str(RUNTIME),
                                 "-DFORT_SCOPE_TEST_FAULTS", "-c", str(path), "-o", str(target)],
                                capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
        objects.append(str(target))
    target = directory / "run"
    result = subprocess.run([nvcc, "-ccbin", host, *objects, "-o", str(target)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return target


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["independent", "overlap", "disjoint", "read_mirror", "downloads", "reuse"])
def test_host_completion_with_pending_real_gpu_work(cuda_completion_executable, mode):
    result = subprocess.run([str(cuda_completion_executable), mode], capture_output=True, text=True, timeout=15)
    if result.returncode == 77:
        pytest.skip("CUDA device unavailable")
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.cuda
def test_host_only_has_no_cuda_initialization(cuda_completion_executable):
    result = subprocess.run([str(cuda_completion_executable), "host_only"], capture_output=True, text=True,
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "FORT_RUNTIME_TRACE": "1"}, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FORT_SCOPED initialize" not in result.stderr
