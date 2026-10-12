"""Exact copy batching through public context coherence, with legacy fallbacks."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from compiler.emission.common.resources import read_scoped_runtime

RUNTIME = Path(__file__).resolve().parents[1] / "runtime"
PROGRAM = r'''
#include "scoped_runtime.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#ifndef FORT_SCOPE_CPU_TEST
#include <cuda_runtime.h>
#endif
void check(int status) {
    if (status) { std::fprintf(stderr,"runtime status %d: %s\n",status,fort_scope_error()); std::abort(); }
}
constexpr size_t nx=8,ny=7,nz=6,total=nx*ny*nz;
bool selected(size_t n) {
    const size_t x=n%nx,y=n/nx%ny,z=n/(nx*ny);
    return (x>=1 && x<6 && y>=1 && y<5 && z>=2 && z<4) ||
           (x>=2 && x<5 && y>=2 && y<4 && (z==1 || z==4));
}
#ifndef FORT_SCOPE_CPU_TEST
__global__ void fill(double *p) { for (size_t n=threadIdx.x;n<total;n+=blockDim.x) p[n]=-7777; }
__global__ void update(double *p) { for (size_t n=threadIdx.x;n<total;n+=blockDim.x) {
    const size_t x=n%nx,y=n/nx%ny,z=n/(nx*ny);
    if ((x>=1 && x<6 && y>=1 && y<5 && z>=2 && z<4) ||
        (x>=2 && x<5 && y>=2 && y<4 && (z==1 || z==4))) p[n]+=1;
} }
#endif
void execute(fort_scope_t context,double *pointer,bool initialize) {
#ifdef FORT_SCOPE_CPU_TEST
    for (size_t n=0;n<total;++n) {
        if (initialize) pointer[n]=-7777;
        else if (selected(n)) pointer[n]+=1;
    }
#else
    int previous=0;void *stream=nullptr;
    check(fort_scope_gpu_enter(context,&previous,&stream));
    if (initialize) fill<<<1,128,0,static_cast<cudaStream_t>(stream)>>>(pointer);
    else update<<<1,128,0,static_cast<cudaStream_t>(stream)>>>(pointer);
    assert(cudaGetLastError()==cudaSuccess);
    check(fort_scope_note_launch(context));
    check(fort_scope_gpu_leave(context,previous));
#endif
}
void verify_device(double *pointer,unsigned generation) {
    std::vector<double> observed(total);
#ifdef FORT_SCOPE_CPU_TEST
    std::memcpy(observed.data(),pointer,total*sizeof(double));
#else
    assert(cudaDeviceSynchronize()==cudaSuccess);
    assert(cudaMemcpy(observed.data(),pointer,total*sizeof(double),cudaMemcpyDeviceToHost)==cudaSuccess);
#endif
    for (size_t n=0;n<total;++n)
        assert(observed[n]==(selected(n) ? double(generation*1000+n+1) : -7777.0));
}
void coherence(bool fail,bool selected_plan) {
    std::vector<double> host(total,-7777);
    size_t extents[3]={nx,ny,nz};int64_t lower[3]={-2,-3,-4};
    fort_scope_layout layout{3,FORT_SCOPE_REAL64,sizeof(double),host.data(),extents,lower,1};
    fort_scope_t context=0;fort_buffer_t buffer=0;void *pointer=nullptr;
    check(fort_scope_create(0,&context));check(fort_scope_register(context,1,1,&layout,1,&buffer));
    fort_scope_access replace{};replace.flags=FORT_SCOPE_WRITE_ALL|FORT_SCOPE_OVERWRITE_ALL;
    if (selected_plan) {
        // Synthetic costs select the initial GPU worker. Subsequent coherence
        // accesses must retain the profile's individually submitted copies.
        check(fort_scope_plan_reset(context));
        fort_scope_plan_binding binding{buffer,replace};
        check(fort_scope_plan_add(context,FORT_SCOPE_PLAN_WORKER,1,&binding,1,1e6,0,1));
        check(fort_scope_plan_validate(context));
        fort_scope_plan_costs costs{};costs.version=FORT_SCOPE_PLANNING_ABI_VERSION;costs.valid=1;
        costs.max_allocation_bytes=1<<24;costs.cpu_flops=costs.cpu_bandwidth=1e6;
        costs.gpu_flops=costs.gpu_bandwidth=costs.h2d_bandwidth=costs.d2h_bandwidth=1e12;
        costs.create_seconds=costs.register_seconds=costs.host_access_seconds=costs.device_access_seconds=1e-9;
        costs.gpu_setup_seconds=costs.cold_driver_startup_seconds=costs.allocation_seconds=costs.release_seconds=1e-9;
        costs.wait_seconds=costs.launch_enqueue_seconds=costs.planning_operation_seconds=1e-9;
        fort_scope_plan_decision decision{};check(fort_scope_plan_select(context,&costs,1,&decision));
        assert(decision.gpu_units==1);
        int gpu=0;check(fort_scope_plan_next(context,1,&binding,1,&gpu));assert(gpu);
    }
    check(fort_scope_device_begin(context,buffer,&replace,&pointer));
    execute(context,static_cast<double *>(pointer),true);check(fort_scope_device_end(context,buffer));
    size_t lo[3][3]={{1,1,2},{2,2,1},{2,2,4}},hi[3][3]={{6,5,4},{5,4,2},{5,4,5}};
    fort_scope_section sections[3]={{lo[0],hi[0]},{lo[1],hi[1]},{lo[2],hi[2]}};
    fort_scope_access work{};work.read_count=work.write_count=3;work.reads=work.writes=sections;
    for (unsigned generation=1;generation<=2;++generation) {
        check(fort_scope_host_begin(context,buffer,&replace));
        for (size_t n=0;n<total;++n) host[n]=generation*1000+n;
        check(fort_scope_host_end(context,buffer));
        if (fail && generation==2) {
            assert(setenv("FORT_SCOPE_TEST_FAIL_DIRECT_COPY_BATCH","1",1)==0);
            assert(fort_scope_device_begin(context,buffer,&work,&pointer)==FORT_SCOPE_EXECUTION);
            assert(fort_scope_device_begin(context,buffer,&work,&pointer)==FORT_SCOPE_EXECUTION);
            assert(fort_scope_close(context)==FORT_SCOPE_EXECUTION);
            check(fort_scope_abandon(context));
            std::puts("failed batch poisoned the owner without prefix replay");return;
        }
        check(fort_scope_device_begin(context,buffer,&work,&pointer));
        execute(context,static_cast<double *>(pointer),false);check(fort_scope_device_end(context,buffer));
        verify_device(static_cast<double *>(pointer),generation);
        // An original native partial writer must publish and invalidate only
        // its changed cell while every other device-produced value survives.
        size_t cell_lo[3]={1,1,2},cell_hi[3]={2,2,3};fort_scope_section cell{cell_lo,cell_hi};
        fort_scope_access native{};native.read_count=native.write_count=1;native.reads=native.writes=&cell;
        check(fort_scope_host_begin(context,buffer,&native));
        const size_t changed=1+nx*(1+ny*2);host[changed]+=100;
        check(fort_scope_host_end(context,buffer));
        fort_scope_access read{};read.flags=FORT_SCOPE_READ_ALL;
        check(fort_scope_host_begin(context,buffer,&read));check(fort_scope_host_end(context,buffer));
        for (size_t n=0;n<total;++n)
            assert(host[n]==generation*1000+n+(selected(n)?1:0)+(n==changed?100:0));
    }
    fort_scope_stats stats{};check(fort_scope_stats_get(context,&stats));
    assert(stats.uploads==6 && stats.upload_bytes==832 && stats.download_bytes==832);
    check(fort_scope_close(context));
    std::puts("complete arrays, exact holes, changed host values and native partial writes agree");
}
void overbudget() {
    std::vector<double> host(8*7*6*65,1);
    size_t extents[4]={8,7,6,65};int64_t lower[4]={-1,-2,-3,-4};
    fort_scope_layout layout{4,FORT_SCOPE_REAL64,sizeof(double),host.data(),extents,lower,1};
    fort_scope_t context=0;fort_buffer_t buffer=0;void *pointer=nullptr;
    check(fort_scope_create(0,&context));check(fort_scope_register(context,1,1,&layout,1,&buffer));
    size_t lo[4]={1,1,1,0},hi[4]={6,5,4,65};fort_scope_section section{lo,hi};
    fort_scope_access read{};read.read_count=1;read.reads=&section;
    check(fort_scope_device_begin(context,buffer,&read,&pointer));check(fort_scope_device_end(context,buffer));
    fort_scope_stats stats{};check(fort_scope_stats_get(context,&stats));
    assert(stats.uploads==65 && stats.upload_bytes==65*5*4*3*8);
    check(fort_scope_close(context));std::puts("65 operations retained the original exact copy path");
}
int main(int argc,char **argv) {
    if (argc>1 && !std::strcmp(argv[1],"lazy")) {
        fort_scope_t context=0;check(fort_scope_create(0,&context));check(fort_scope_close(context));return 0;
    }
#ifndef FORT_SCOPE_CPU_TEST
    int count=0;if (cudaGetDeviceCount(&count)!=cudaSuccess || !count) return 77;
    int version=0;assert(cudaDriverGetVersion(&version)==cudaSuccess);
    std::printf("copy batch capability: compile=%d driver=%d\n",CUDART_VERSION,version);
    if (argc>1 && !std::strcmp(argv[1],"failure") && (CUDART_VERSION<13040 || version<13040)) return 77;
#endif
    if (argc>1 && !std::strcmp(argv[1],"overbudget")) overbudget();
    else coherence(argc>1 && !std::strcmp(argv[1],"failure"),argc>1 && !std::strcmp(argv[1],"selected"));
}
'''


@pytest.fixture(scope="module")
def copy_programs(tmp_path_factory):
    directory = tmp_path_factory.mktemp("exact_copy_batches")
    source = directory / "main.cu"
    source.write_text(PROGRAM)
    host = shutil.which("g++-14") or shutil.which("g++")
    if not host:
        pytest.skip("C++ compiler unavailable")
    cpu = directory / "cpu"
    command = [host, "-O1", "-std=c++17", "-pthread", "-DFORT_SCOPE_CPU_TEST", "-x", "c++",
               "-I", str(RUNTIME), str(source), str(RUNTIME / "scoped_runtime.cu"), "-o", str(cpu)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    return directory, source, host, cpu


@pytest.fixture(scope="module")
def cuda_copy_program(copy_programs):
    directory, source, host, _cpu = copy_programs
    nvcc = shutil.which("nvcc")
    if not nvcc:
        pytest.skip("CUDA toolkit unavailable")
    binary = directory / "cuda"
    result = subprocess.run([nvcc, "-O2", "-std=c++17", "-ccbin", host,
                             "-arch=" + os.environ.get("FORT_TEST_CUDA_ARCH", "native"),
                             "-DFORT_SCOPE_TEST_FAULTS", "-I", str(RUNTIME), str(source),
                             str(RUNTIME / "scoped_runtime.cu"), "-o", str(binary)],
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    return binary


def run(binary, mode, **environment):
    result = subprocess.run([str(binary), mode], capture_output=True, text=True, timeout=30,
                            env={**os.environ, "FORT_RUNTIME_TRACE": "1", **environment})
    if result.returncode == 77:
        pytest.skip("CUDA device or required batch backend unavailable")
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.mark.parametrize("mode", ["coherence", "selected", "overbudget", "lazy"])
def test_reference_backend_preserves_exact_transfer_semantics(copy_programs, mode):
    result = run(copy_programs[3], mode)
    assert "copy_batch" not in result.stderr


def test_public_batch_copy_requirements_and_legacy_cost_accounting():
    _sources, manifest = read_scoped_runtime()
    public = manifest["scope_transfers"]["direct_copy_batches"]
    assert public["operation_limit"] == 64
    assert public["minimum_cuda_runtime"] == public["minimum_cuda_driver"] == 13040
    assert "physical operations" in public["legacy_statistics"]
    assert "disabled for cost-selected plans" in public["placement_costs"]


@pytest.mark.cuda
def test_cuda_context_batches_preserve_holes_host_edits_and_partial_writes(cuda_copy_program):
    result = run(cuda_copy_program, "coherence")
    batches = [line for line in result.stderr.splitlines() if line.startswith("FORT_SCOPED copy_batch ")]
    capability = re.search(r"copy batch capability: compile=(\d+) driver=(\d+)", result.stdout)
    assert capability
    if min(map(int, capability.groups())) < 13040:
        pytest.skip("CUDA runtime/driver is older than 13.4")
    assert len(batches) == 4
    assert all("copy_submissions=1" in line for line in batches)
    # Each generation uploads three rectangles. Removing the native-written
    # cell splits copyback into six exact physical operations.
    assert sum(int(re.search(r"copy_operations=(\d+)", line)[1]) for line in batches) == 18
    assert sum(int(re.search(r" bytes=(\d+)", line)[1]) for line in batches) == 1648


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["coherence", "selected", "overbudget", "lazy"])
def test_driver_budget_and_native_fallbacks_do_not_submit_batches(cuda_copy_program, mode):
    environment = {"FORT_SCOPE_TEST_DIRECT_COPY_DRIVER": "13030"} if mode == "coherence" else {}
    if mode == "lazy":
        environment["CUDA_VISIBLE_DEVICES"] = ""
    result = run(cuda_copy_program, mode, **environment)
    assert "copy_batch" not in result.stderr
    if mode == "lazy":
        assert "initialize" not in result.stderr


@pytest.mark.cuda
def test_failed_batch_after_a_gpu_prefix_is_terminal(cuda_copy_program):
    result = run(cuda_copy_program, "failure")
    assert "failed batch poisoned the owner without prefix replay" in result.stdout
    assert result.stderr.count("FORT_SCOPED copy_batch ") == 2
