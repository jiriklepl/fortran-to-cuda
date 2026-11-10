"""Milestone: capture order, replica preservation, two-slot backpressure and join."""
from pathlib import Path
import shutil
import subprocess

import pytest

RUNTIME = Path(__file__).resolve().parents[1] / "runtime"
PROGRAM = r'''
#include "snapshots.h"
#include <cuda_runtime.h>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <future>
#include <thread>
#include <vector>
void check(int s) { if(s) { std::fprintf(stderr,"status=%d %s\n",s,fort_scope_error());std::abort(); } }
__global__ void fill(double *x,size_t n,double value) {
    for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x) x[i]=value;
}
int produce(void *user,const fort_scope_external_view_v1 *v,size_t,void *stream,int *started) {
    *started=1;
    fill<<<7,128,0,static_cast<cudaStream_t>(stream)>>>(static_cast<double *>(v[0].device),v[0].bytes/8,*static_cast<double *>(user));
    return cudaGetLastError()==cudaSuccess?FORT_SCOPE_OK:FORT_SCOPE_EXECUTION;
}
std::atomic<bool> release_writer{false},entered{false};std::atomic<unsigned> writes{0};
struct Expected { double value;bool fail; };
int write_snapshot(void *user,const void *p,size_t bytes) {
    const auto &expected=*static_cast<Expected *>(user);entered=true;
    while(!release_writer.load()) std::this_thread::yield();
    const auto *x=static_cast<const double *>(p);
    for(size_t i=0;i<bytes/8;++i) assert(x[i]==expected.value);
    assert(++writes==unsigned(expected.value));
    return expected.fail ? FORT_SCOPE_EXECUTION : FORT_SCOPE_OK;
}
int main() {
    int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices) return 77;
    constexpr size_t n=65537,bytes=n*8;
    fort_scope_domain_t domain=0;fort_scope_t context=0;fort_buffer_t buffer=0;
    check(fort_scope_domain_create_v1(0,&domain));check(fort_scope_create_in_domain_v1(domain,&context));
    check(fort_scope_set_device_budget(context,3*bytes));
    void *queue=nullptr;
    assert(fort_snapshot_create_v1(domain,bytes,bytes,&queue)==FORT_SCOPE_RESOURCE && !queue);
    assert(fort_snapshot_create_v1(domain,2*bytes,4*bytes,&queue)==FORT_SCOPE_RESOURCE && !queue);
    check(fort_snapshot_create_v1(domain,bytes,2*bytes,&queue));
    std::vector<double> host(n,-1);size_t extent=n;int64_t lower=-2;
    fort_scope_layout layout{1,FORT_SCOPE_REAL64,8,host.data(),&extent,&lower,1};
    check(fort_scope_register(context,1,1,&layout,1,&buffer));
    fort_scope_access overwrite{FORT_SCOPE_WRITE_ALL|FORT_SCOPE_OVERWRITE_ALL,0,nullptr,0,nullptr,0,nullptr};
    fort_scope_external_binding_v1 output{buffer,FORT_SCOPE_REAL64,1,8,overwrite};
    fort_scope_external_call_v1 call{1,0,&output,1,0,0};fort_scope_external_report_v1 report{};
    fort_scope_access read{FORT_SCOPE_READ_ALL,0,nullptr,0,nullptr,0,nullptr};
    fort_scope_external_binding_v1 input{buffer,FORT_SCOPE_REAL64,1,8,read};
    Expected expected[3]={{1,false},{2,false},{3,false}};
    uint32_t first=0,second=0;
    check(fort_snapshot_acquire_v1(queue,&first));
    size_t wrong=bytes-8;
    assert(fort_snapshot_submit_v1(queue,first,context,&input,&wrong,1,write_snapshot,&expected[0])==FORT_SCOPE_ARGUMENT);
    double value=1;check(fort_scope_external_execute_v1(context,&call,produce,&value,&report));
    check(fort_snapshot_submit_v1(queue,first,context,&input,&bytes,1,write_snapshot,&expected[0]));
    assert(fort_snapshot_cancel_v1(queue,first)==FORT_SCOPE_STATE);
    check(fort_snapshot_acquire_v1(queue,&second));assert(first!=second);
    value=2;check(fort_scope_external_execute_v1(context,&call,produce,&value,&report));
    check(fort_snapshot_submit_v1(queue,second,context,&input,&bytes,1,write_snapshot,&expected[1]));
    value=3;check(fort_scope_external_execute_v1(context,&call,produce,&value,&report));
    check(fort_scope_wait(context));
    while(!entered.load()) std::this_thread::yield();
    auto blocked=std::async(std::launch::async,[&]{uint32_t slot=0;check(fort_snapshot_acquire_v1(queue,&slot));return slot;});
    assert(blocked.wait_for(std::chrono::milliseconds(25))==std::future_status::timeout);
    release_writer=true;
    uint32_t third=blocked.get();
    check(fort_snapshot_submit_v1(queue,third,context,&input,&bytes,1,write_snapshot,&expected[2]));
    check(fort_snapshot_wait_v1(queue));assert(writes==3);
    fort_scope_stats stats{};check(fort_scope_stats_get(context,&stats));
    assert(stats.uploads==0 && stats.downloads==0 && host[0]==-1); // no publication of live fields
    check(fort_scope_observe_v1(context,&buffer,1));assert(host[0]==3 && host.back()==3);
    check(fort_scope_stats_get(context,&stats));assert(stats.downloads==1 && stats.uploads==0);
    // Host read preserved the device replica. Another snapshot uploads nothing.
    writes=2;check(fort_snapshot_acquire_v1(queue,&third));
    check(fort_snapshot_submit_v1(queue,third,context,&input,&bytes,1,write_snapshot,&expected[2]));
    check(fort_snapshot_destroy_v1(queue));assert(writes==3);
    check(fort_scope_stats_get(context,&stats));assert(stats.uploads==0 && stats.downloads==1);
    fort_scope_scratch_stats_v1 scratch{};check(fort_scope_scratch_stats_get_v1(context,&scratch));
    assert(scratch.peak_total_device_bytes==3*bytes);
    check(fort_scope_close(context));check(fort_scope_domain_close_v1(domain));

    // Writer failure is reported at wait/destroy and poisons the live execution.
    check(fort_scope_domain_create_v1(0,&domain));check(fort_scope_create_in_domain_v1(domain,&context));
    check(fort_scope_register(context,1,1,&layout,1,&buffer));input.buffer=buffer;
    check(fort_snapshot_create_v1(domain,bytes,2*bytes,&queue));
    Expected failed{3,true};writes=2;
    check(fort_snapshot_acquire_v1(queue,&first));
    check(fort_snapshot_submit_v1(queue,first,context,&input,&bytes,1,write_snapshot,&failed));
    assert(fort_snapshot_wait_v1(queue)==FORT_SCOPE_EXECUTION);
    assert(fort_snapshot_acquire_v1(queue,&first)==FORT_SCOPE_EXECUTION);
    assert(fort_scope_wait(context)==FORT_SCOPE_EXECUTION);
    assert(fort_snapshot_destroy_v1(queue)==FORT_SCOPE_EXECUTION);
    check(fort_scope_abandon(context));check(fort_scope_domain_abandon_v1(domain));
    std::puts("PASS snapshots: immutable generations, read replicas, two slots, backpressure, ordered writer, final join, budget, failure");
}
'''


@pytest.mark.cuda
def test_resident_snapshots(tmp_path):
    nvcc = shutil.which("nvcc")
    host = shutil.which("g++-14") or shutil.which("g++")
    if not nvcc or not host:
        pytest.skip("CUDA toolkit unavailable")
    source = tmp_path / "main.cu"
    source.write_text(PROGRAM)
    objects = []
    for path in (source, RUNTIME / "snapshots.cu", RUNTIME / "scoped_runtime.cu"):
        target = tmp_path / (path.stem + ".o")
        result = subprocess.run([nvcc, "-std=c++17", "-arch=sm_86", "-ccbin", host, "-I", str(RUNTIME),
                                 "-c", str(path), "-o", str(target)], capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr
        objects.append(str(target))
    binary = tmp_path / "run"
    result = subprocess.run([nvcc, "-ccbin", host, "-Xcompiler=-pthread", *objects, "-o", str(binary)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=40)
    if result.returncode == 77:
        pytest.skip("CUDA device unavailable")
    assert result.returncode == 0, result.stdout + result.stderr
