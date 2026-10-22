"""Actual CUDA execution validates ordered effects, parameter updates and cleanup."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from compiler.tests.test_scoped_runtime import RUNTIME

CLIENT = r'''
#include "scoped_runtime.h"
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <utility>
#include <vector>
void check(int status) {
    if(status) { std::fprintf(stderr,"status=%d: %s\n",status,fort_scope_error()); std::abort(); }
}
__global__ void produce(const double *a,double *b,int n,int first,int last,double add) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n && i>=first && i<last) b[i]=a[i]+add;
}
__global__ void consume(const double *b,double *out,int n,int first,int last,double factor) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n && i>=first && i<last) out[i]=factor*b[i];
}
__global__ void alternate(const double *b,double *out,int n,int first,int last,double factor) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n && i>=first && i<last) out[i]=factor*b[i]+1;
}
struct Fixture {
    fort_scope_t context=0; fort_buffer_t a=0,b=0,out=0,other=0;
    double host_a[8],host_b[8]{},host_out[8]{},host_other[8]{};
    double *pa=nullptr,*pb=nullptr,*po=nullptr;
    size_t extent[1]={8},lo[1]={0},hi[1]={8}; int64_t lower[1]={-3};
    fort_scope_section section{lo,hi};
    fort_scope_plan_binding producer[2]{},consumer[2]{};
    int n=8,first=0,last=8; double add=2,factor=3;
    void *produce_args[6]={&pa,&pb,&n,&first,&last,&add};
    void *consume_args[6]={&pb,&po,&n,&first,&last,&factor};
    fort_scope_graph_task_v1 tasks[2]{};
    Fixture(bool partial=false) {
        check(fort_scope_create(0,&context));
        for(int i=0;i<8;++i) host_a[i]=i;
        for(auto item:{std::pair<double *,fort_buffer_t *>{host_a,&a},{host_b,&b},{host_out,&out}}) {
            const uint64_t identity=item.second==&a?1:item.second==&b?2:3;
            fort_scope_layout layout{1,FORT_SCOPE_REAL64,8,item.first,extent,lower,1};
            check(fort_scope_register(context,identity,1,&layout,identity==1,item.second));
        }
        check(fort_scope_device_reserve_v1(context,a,reinterpret_cast<void **>(&pa)));
        check(fort_scope_device_reserve_v1(context,b,reinterpret_cast<void **>(&pb)));
        check(fort_scope_device_reserve_v1(context,out,reinterpret_cast<void **>(&po)));
        if(partial) { first=2; last=6; lo[0]=2; hi[0]=6; }
        fort_scope_access reads{0,1,&section,0,nullptr,0,nullptr};
        fort_scope_access writes{0,0,nullptr,1,&section,1,&section};
        producer[0]={a,reads}; producer[1]={b,writes};
        consumer[0]={b,reads}; consumer[1]={out,writes};
        tasks[0]={{1,0,reinterpret_cast<const void *>(produce),{1,1,1},{32,1,1},0,produce_args,6},producer,2};
        tasks[1]={{1,0,reinterpret_cast<const void *>(consume),{1,1,1},{32,1,1},0,consume_args,6},consumer,2};
    }
    void run() { check(fort_scope_graph_execute_v1(context,17,tasks,2)); }
    fort_scope_graph_stats_v1 graph() {
        fort_scope_graph_stats_v1 result{}; check(fort_scope_graph_stats_get_v1(context,&result)); return result;
    }
    fort_scope_stats stats() {
        fort_scope_stats result{}; check(fort_scope_stats_get(context,&result)); return result;
    }
    void observe(fort_buffer_t handle) { check(fort_scope_observe_v1(context,&handle,1)); }
    void rebind() {
        fort_scope_layout layout{1,FORT_SCOPE_REAL64,8,host_other,extent,lower,1};
        check(fort_scope_register(context,4,1,&layout,0,&other));
        auto *previous=po;
        check(fort_scope_device_reserve_v1(context,other,reinterpret_cast<void **>(&po)));
        assert(previous!=po); consumer[1].buffer=other;
    }
};
int main(int argc,char **argv) {
    assert(argc==2); const char *mode=argv[1];
    if(!std::strcmp(mode,"no_device")) {
        fort_scope_t c=0; check(fort_scope_create(0,&c));
        check(fort_scope_graph_submit_v1(c,0,nullptr,0));
        check(fort_scope_graph_execute_v1(c,0,nullptr,0));
        check(fort_scope_observe_v1(c,nullptr,0));
        fort_scope_graph_stats_v1 stats{}; check(fort_scope_graph_stats_get_v1(c,&stats));
        assert(!stats.creations && !stats.submissions); check(fort_scope_close(c)); return 0;
    }
    int devices=0; if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices) return 77;
    const bool partial=!std::strcmp(mode,"partial");
    Fixture f(partial);
    if(!std::strcmp(mode,"long_group")) {
        std::vector<fort_scope_graph_task_v1> tasks(40,f.tasks[0]); tasks.push_back(f.tasks[1]);
        check(fort_scope_graph_execute_v1(f.context,19,tasks.data(),tasks.size()));
        f.add=4; f.factor=5;
        check(fort_scope_graph_execute_v1(f.context,19,tasks.data(),tasks.size()));
        check(fort_scope_graph_execute_v1(f.context,19,tasks.data(),tasks.size()));
        f.observe(f.out);
        for(int i=0;i<8;++i) assert(f.host_out[i]==5*(i+4));
        const auto stats=f.graph();
        assert(stats.creations==1 && stats.replays==1 && stats.logical_kernels==82 && stats.parameter_updates==41);
        check(fort_scope_close(f.context)); return 0;
    }
    if(!std::strcmp(mode,"invalid")) {
        auto first=f.tasks[0]; f.tasks[0]=f.tasks[1]; f.tasks[1]=first;
        assert(fort_scope_graph_execute_v1(f.context,17,f.tasks,2)==FORT_SCOPE_UNINITIALIZED);
        assert(!f.graph().creations && !f.stats().upload_bytes);
        check(fort_scope_close(f.context)); return 0;
    }
    if(!std::strcmp(mode,"create_failure")) {
        f.run(); f.rebind(); // First reach is direct; fail actual second-reach creation.
        setenv("FORT_SCOPE_TEST_FAIL_GRAPH_CREATE","1",1);
        assert(fort_scope_graph_execute_v1(f.context,17,f.tasks,2)==FORT_SCOPE_RESOURCE);
        unsetenv("FORT_SCOPE_TEST_FAIL_GRAPH_CREATE");
        fort_scope_access read{}; read.flags=FORT_SCOPE_READ_ALL;
        assert(fort_scope_host_begin(f.context,f.other,&read)==FORT_SCOPE_UNINITIALIZED);
        assert(!f.graph().submissions && f.stats().launches==2);
        f.run(); f.observe(f.other);
        for(int i=0;i<8;++i) assert(f.host_other[i]==3*(i+2));
        check(fort_scope_close(f.context)); return 0;
    }
    if(!std::strcmp(mode,"direct_failure")) {
        f.tasks[1].kernel.block[0]=2048; // Valid descriptor, invalid hardware launch.
        assert(fort_scope_graph_execute_v1(f.context,17,f.tasks,2)==FORT_SCOPE_EXECUTION);
        (void)cudaGetLastError();
        assert(cudaDeviceSynchronize()==cudaSuccess);
        double prefix[8]{};
        assert(cudaMemcpy(prefix,f.pb,sizeof(prefix),cudaMemcpyDeviceToHost)==cudaSuccess);
        for(int i=0;i<8;++i) assert(prefix[i]==i+2); // First launch executed exactly once.
        assert(fort_scope_close(f.context)==FORT_SCOPE_EXECUTION);
        assert(fort_scope_graph_execute_v1(f.context,17,f.tasks,2)==FORT_SCOPE_EXECUTION);
        check(fort_scope_abandon(f.context)); return 0;
    }
    f.run();
    assert(!f.graph().creations && !f.graph().submissions && f.stats().launches==2);
    assert(f.stats().upload_bytes==(partial?4u:8u)*sizeof(double));
    assert(!f.stats().download_bytes);
    if(!std::strcmp(mode,"one_shot")) {
        f.observe(f.out);
        for(int i=0;i<8;++i) assert(f.host_out[i]==3*(i+2));
        check(fort_scope_close(f.context)); return 0;
    }
    if(!std::strcmp(mode,"forget") || !std::strcmp(mode,"forget_partial")) {
        const auto before=f.stats();
        if(!std::strcmp(mode,"forget_partial")) {
            size_t lo[1]={2},hi[1]={6}; fort_scope_section removed{lo,hi};
            check(fort_scope_forget_sections_v1(f.context,f.b,&removed,1));
        } else check(fort_scope_forget_definition(f.context,f.b));
        assert(f.stats().download_bytes==before.download_bytes && f.stats().waits==before.waits);
        fort_scope_access read{}; read.flags=FORT_SCOPE_READ_ALL;
        assert(fort_scope_host_begin(f.context,f.b,&read)==FORT_SCOPE_UNINITIALIZED);
        check(fort_scope_close(f.context));
        for(int i=0;i<8;++i) {
            assert(f.host_b[i]==i+2);
            assert(f.host_out[i]==3*(i+2));
        }
        return 0;
    }
    if(!std::strcmp(mode,"launch_failure")) {
        setenv("FORT_SCOPE_TEST_FAIL_GRAPH_LAUNCH","1",1);
        assert(fort_scope_graph_execute_v1(f.context,17,f.tasks,2)==FORT_SCOPE_EXECUTION);
        unsetenv("FORT_SCOPE_TEST_FAIL_GRAPH_LAUNCH");
        assert(fort_scope_close(f.context)==FORT_SCOPE_EXECUTION);
        assert(fort_scope_graph_execute_v1(f.context,17,f.tasks,2)==FORT_SCOPE_EXECUTION);
        check(fort_scope_abandon(f.context)); return 0;
    }
    f.run(); // Second reach constructs; subsequent matching reach replays.
    assert(f.graph().creations==1 && f.graph().submissions==1 && f.stats().launches==4);
    if(!std::strcmp(mode,"update_failure")) {
        setenv("FORT_SCOPE_TEST_FAIL_GRAPH_UPDATE","1",1);
        assert(fort_scope_graph_execute_v1(f.context,17,f.tasks,2)==FORT_SCOPE_RESOURCE);
        unsetenv("FORT_SCOPE_TEST_FAIL_GRAPH_UPDATE");
        assert(f.graph().submissions==1 && f.stats().launches==4);
    }
    if(!std::strcmp(mode,"rebind")) f.rebind();
    if(!std::strcmp(mode,"topology")) f.tasks[1].kernel.function=reinterpret_cast<const void *>(alternate);
    f.add=4; f.factor=5;
    f.tasks[0].kernel.grid[0]=2; f.tasks[1].kernel.block[0]=64;
    f.run();
    if(!std::strcmp(mode,"topology")) f.run(); // New topology also warms directly once.
    // Changing argument storage after submission must not change enqueued work.
    f.add=f.factor=999;
    f.observe(f.other?f.other:f.out);
    auto *result=f.other?f.host_other:f.host_out;
    for(int i=0;i<8;++i) {
        const double expected=(i>=f.first && i<f.last)?5*(i+4)+(!std::strcmp(mode,"topology")?1:0):0;
        assert(result[i]==expected);
    }
    const auto graph=f.graph();
    assert(graph.submissions==2 && graph.logical_kernels==4);
    assert(f.stats().launches==(!std::strcmp(mode,"topology")?8u:6u));
    if(!std::strcmp(mode,"topology")) assert(graph.creations==2 && graph.rebuilds==1 && graph.destructions==1);
    else assert(graph.creations==1 && graph.replays==1 && graph.parameter_updates==2);
    if(partial) {
        fort_scope_access read{}; read.flags=FORT_SCOPE_READ_ALL;
        assert(fort_scope_host_begin(f.context,f.out,&read)==FORT_SCOPE_UNINITIALIZED);
    }
    // Dead intermediate and old output need no publication at retirement.
    const auto before=f.stats().download_bytes;
    check(fort_scope_retire_v1(f.context,f.b));
    assert(f.stats().download_bytes==before);
    check(fort_scope_close(f.context));
    std::puts("CUDA graph replay accepted");
}
'''


@pytest.fixture(scope="module")
def cuda_graph_executable(tmp_path_factory):
    prepared = os.environ.get("FORT_TEST_GRAPH_PREBUILT")
    if prepared:
        return Path(prepared)
    nvcc = shutil.which("nvcc")
    host = shutil.which("g++-14") or shutil.which("g++")
    if not nvcc or not host:
        pytest.skip("CUDA toolkit unavailable")
    directory = tmp_path_factory.mktemp("scoped_graph_cuda")
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
@pytest.mark.parametrize("mode", ["replay", "rebind", "topology", "partial", "invalid", "create_failure",
                                 "update_failure", "launch_failure", "direct_failure", "one_shot", "long_group", "forget", "forget_partial"])
def test_real_cuda_graph_execution(cuda_graph_executable, mode):
    result = subprocess.run([str(cuda_graph_executable), mode], capture_output=True, text=True,
                            env={**os.environ, "FORT_RUNTIME_TRACE": "1"}, timeout=30)
    if result.returncode == 77:
        pytest.skip("CUDA device unavailable")
    assert result.returncode == 0, result.stdout + result.stderr
    records = [json.loads(line.removeprefix("FORT_SCOPED evidence ")) for line in result.stderr.splitlines()
               if line.startswith("FORT_SCOPED evidence ")]
    graphs = [record for record in records if record["event"] == "graph_statistics"]
    if mode != "invalid":
        assert len(graphs) == 1
        assert graphs[0]["complete"] == (mode not in {"launch_failure", "direct_failure"})
        assert graphs[0]["live_graphs"] == 0
        assert graphs[0]["creations"] == graphs[0]["destructions"]
    if mode == "one_shot":
        assert "FORT_SCOPED graph_warmup_direct bytes=2 " in result.stderr
        assert "FORT_SCOPED graph_submit " not in result.stderr
        assert graphs[0]["creations"] == graphs[0]["instantiations"] == 0
        queue, = [record for record in records if record["event"] == "command_queue_statistics"]
        assert queue["direct_submissions"] == 1
        assert queue["direct_kernels"] == 2
        assert queue["graph_resource_fallbacks"] == 0


@pytest.mark.cuda
def test_graph_noop_and_metadata_queries_need_no_device(cuda_graph_executable):
    result = subprocess.run([str(cuda_graph_executable), "no_device"], capture_output=True, text=True,
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "FORT_RUNTIME_TRACE": "1"}, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FORT_SCOPED initialize" not in result.stderr
