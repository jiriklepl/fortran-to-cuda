"""Independent public batch clients run real CUDA without generated-source inspection."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

RUNTIME = Path(__file__).resolve().parents[1] / "runtime"

CLIENT = r'''
#include "scoped_runtime.h"
#include "scoped_entry.hpp"
#include "staging.hpp"
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <vector>
void check(int status) {
    if (status) { std::fprintf(stderr,"status=%d: %s\n",status,fort_scope_error()); std::abort(); }
}
fort_scope_plan_costs base_costs() {
    fort_scope_plan_costs c{}; c.version=1; c.valid=1; c.max_allocation_bytes=64ULL<<20;
    c.cpu_flops=c.cpu_bandwidth=1e8; c.gpu_flops=c.gpu_bandwidth=c.h2d_bandwidth=c.d2h_bandwidth=1e12;
    c.h2d_latency=c.d2h_latency=c.create_seconds=c.register_seconds=c.host_access_seconds=c.device_access_seconds=
    c.gpu_setup_seconds=c.cold_driver_startup_seconds=c.allocation_seconds=c.release_seconds=c.wait_seconds=
    c.launch_enqueue_seconds=c.planning_operation_seconds=1e-9; return c;
}
fort_scope_batch_costs transfer_costs() {
    fort_scope_batch_costs c{}; c.version=1; c.valid=1; c.async_engine_count=2; c.max_slot_bytes=16ULL<<20;
    for (int i=0;i<4;++i) c.staging_cold_seconds[i]=c.staging_reuse_seconds[i]=1e-9;
    c.event_record_seconds=c.event_wait_seconds=c.ready_event_seconds=c.preparation_operation_seconds=
    c.pack_row_seconds=c.unpack_row_seconds=c.pinned_h2d_latency=c.pinned_d2h_latency=1e-9;
    c.pack_bytes_per_second=c.unpack_bytes_per_second=c.pinned_h2d_bandwidth=c.pinned_d2h_bandwidth=1e12;
    return c;
}
__global__ void producer(const double *a,double *middle,int nx,int ny,int halo,int iterations,
                         int begin,int count,bool reverse,bool stencil) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    const int plane=nx*ny;
    if(i>=plane*count) return;
    const int ordinal=begin+i/plane;
    const int z=halo+(reverse ? iterations-1-ordinal : ordinal);
    const int index=i%plane+plane*z;
    middle[index]=stencil ? a[index-plane]+a[index]+a[index+plane] : 2*a[index];
}
__global__ void consumer(const double *middle,double *out,int nx,int ny,int halo,int iterations,
                         int begin,int count,bool reverse) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x, plane=nx*ny;
    if(i>=plane*count) return;
    const int ordinal=begin+i/plane, z=halo+(reverse ? iterations-1-ordinal : ordinal);
    const int x=i%nx,y=i/nx%ny,index=i%plane+plane*z;
    out[index]=middle[index]+(x-2)+10*(y-4)+100*(z-7);
}
struct Work {
    fort_buffer_t a,middle,out; int nx=32,ny=17,iterations=33; bool reverse=false,stencil=false,fail=false;
    int calls=0; std::set<void*> streams; double *host_out=nullptr;
};
int worker(const fort_scope_batch_window *window,void *opaque,uint64_t *launches) {
    auto &w=*static_cast<Work*>(opaque); ++w.calls; w.streams.insert(window->stream);
    const auto *a=fort_scoped::batch_view(*window,w.a),*middle=fort_scoped::batch_view(*window,w.middle),
               *out=fort_scoped::batch_view(*window,w.out);
    assert(window->version==1 && a && middle && out);
    assert(a->layout.extents[0]==size_t(w.nx) && a->layout.extents[1]==size_t(w.ny) &&
           a->layout.extents[2]==size_t(w.iterations+2));
    assert(a->layout.lower_bounds[0]==-2 && a->layout.lower_bounds[1]==-4 && a->layout.lower_bounds[2]==-7);
    auto stream=static_cast<cudaStream_t>(window->stream);
    const int threads=w.nx*w.ny*int(window->count);
    producer<<<(threads+127)/128,128,0,stream>>>(static_cast<double*>(a->device),static_cast<double*>(middle->device),
        w.nx,w.ny,1,w.iterations,int(window->begin),int(window->count),w.reverse,w.stencil);
    *launches=1;
    if (w.fail || cudaGetLastError()!=cudaSuccess) return FORT_SCOPE_EXECUTION;
    consumer<<<(threads+127)/128,128,0,stream>>>(static_cast<double*>(middle->device),static_cast<double*>(out->device),
        w.nx,w.ny,1,w.iterations,int(window->begin),int(window->count),w.reverse);
    *launches=2;
    if (w.host_out) {
        // These host halo writes must survive exact interior copyback.
        for (int i=0;i<w.nx*w.ny;++i) w.host_out[i]=42;
    }
    return cudaGetLastError()==cudaSuccess ? FORT_SCOPE_OK : FORT_SCOPE_EXECUTION;
}
void direct(fort_scope_t context,Work &work) {
    size_t lo[3]={0,0,1},hi[3]={size_t(work.nx),size_t(work.ny),size_t(work.iterations+1)};
    size_t input_lo[3]={0,0,work.stencil ? 0U : 1U},
           input_hi[3]={size_t(work.nx),size_t(work.ny),size_t(work.iterations+(work.stencil ? 2 : 1))};
    fort_scope_section interior{lo,hi},input{input_lo,input_hi};
    fort_scope_access read{},write{}; read.read_count=1; read.reads=&input;
    write.write_count=write.overwrite_count=1; write.writes=write.overwrites=&interior;
    void *pointers[3]{},*stream=nullptr; int previous=0;
    check(fort_scope_device_begin(context,work.a,&read,&pointers[0]));
    check(fort_scope_device_begin(context,work.middle,&write,&pointers[1]));
    check(fort_scope_device_begin(context,work.out,&write,&pointers[2]));
    check(fort_scope_gpu_enter(context,&previous,&stream));
    const int plane=work.nx*work.ny,threads=plane*work.iterations;
    producer<<<(threads+127)/128,128,0,static_cast<cudaStream_t>(stream)>>>(static_cast<double*>(pointers[0]),
        static_cast<double*>(pointers[1]),work.nx,work.ny,1,work.iterations,0,work.iterations,work.reverse,work.stencil);
    check(fort_scope_note_launch(context));
    consumer<<<(threads+127)/128,128,0,static_cast<cudaStream_t>(stream)>>>(static_cast<double*>(pointers[1]),
        static_cast<double*>(pointers[2]),work.nx,work.ny,1,work.iterations,0,work.iterations,work.reverse);
    assert(cudaGetLastError()==cudaSuccess); check(fort_scope_note_launch(context));
    check(fort_scope_device_end(context,work.a)); check(fort_scope_device_end(context,work.middle));
    check(fort_scope_device_end(context,work.out)); check(fort_scope_gpu_leave(context,previous));
}
void run(const char *mode,int iterations=33) {
    const bool preview=!std::strcmp(mode,"preview"),budget=!std::strcmp(mode,"budget"),
               allocation=!std::strcmp(mode,"allocation"),record=!std::strcmp(mode,"record"),
               failed_wait=!std::strcmp(mode,"wait"),callback=!std::strcmp(mode,"callback"),
               dirty=!std::strcmp(mode,"dirty"),slow=!std::strcmp(mode,"slow"),automatic=!std::strcmp(mode,"auto"),
               cross=!std::strcmp(mode,"cross"),native=!std::strcmp(mode,"native"),pinned=!std::strcmp(mode,"pinned");
    Work work; work.iterations=iterations; work.reverse=!std::strcmp(mode,"reverse");
    work.stencil=!std::strcmp(mode,"halo"); work.fail=callback;
    const size_t plane=work.nx*work.ny,items=plane*(iterations+2),bytes=items*sizeof(double);
    std::vector<double> a(items),middle(items,-101),out(items,-202);
    for(size_t i=0;i<items;++i) a[i]=double(i+1);
    work.host_out=work.stencil ? out.data() : nullptr;
    fort_scope_t context=0; check(fort_scope_create(0,&context));
    check(fort_scope_set_transfers(context,automatic ? FORT_SCOPE_TRANSFERS_AUTO : pinned ? FORT_SCOPE_TRANSFERS_PINNED : FORT_SCOPE_TRANSFERS_PIPELINED));
    auto costs=base_costs(); auto transfers=transfer_costs();
    if(pinned) check(fort_scope_set_transfer_costs_v1(context,&transfers,1));
    size_t extents[3]={size_t(work.nx),size_t(work.ny),size_t(iterations+2)}; int64_t lower[3]={-2,-4,-7};
    fort_scope_layout layout{3,FORT_SCOPE_REAL64,8,a.data(),extents,lower,1};
    check(fort_scope_register(context,1,1,&layout,1,&work.a)); layout.host=middle.data();
    check(fort_scope_register(context,2,1,&layout,0,&work.middle)); layout.host=out.data();
    check(fort_scope_register(context,3,1,&layout,1,&work.out));
    const size_t origin=work.reverse ? size_t(iterations) : 1;
    size_t lo[3]={0,0,origin},hi[3]={size_t(work.nx),size_t(work.ny),origin+1};
    size_t read_lo[3]={0,0,work.stencil ? origin-1 : origin},read_hi[3]={size_t(work.nx),size_t(work.ny),work.stencil ? origin+2 : origin+1};
    fort_scope_section slab{lo,hi},input_slab{read_lo,read_hi};
    fort_scope_access read{},input_read{},write{}; read.read_count=input_read.read_count=1;
    read.reads=&slab; input_read.reads=&input_slab; write.write_count=write.overwrite_count=1; write.writes=write.overwrites=&slab;
    const int64_t step=work.reverse ? -1 : 1;
    fort_scope_batch_binding first[2]={{work.a,2,step,input_read},{work.middle,2,step,write}},
                             second[2]={{work.middle,2,step,read},{work.out,2,step,write}},
                             forgotten{work.middle,FORT_SCOPE_BATCH_FIXED_AXIS,0,{}};
    fort_scope_batch_unit units[3]={{FORT_SCOPE_PLAN_FORGET,0,&forgotten,1,0,0},
        {FORT_SCOPE_PLAN_WORKER,101,first,2,1e8,double(bytes)}, {FORT_SCOPE_PLAN_WORKER,102,second,2,1e8,double(bytes)}};
    size_t full_lo[3]={0,0,1},full_hi[3]={size_t(work.nx),size_t(work.ny),size_t(iterations+1)};
    fort_scope_section full{full_lo,full_hi}; fort_scope_access full_read{},full_write{};
    full_read.read_count=1; full_read.reads=&full;
    full_write.write_count=full_write.overwrite_count=1; full_write.writes=full_write.overwrites=&full;
    fort_scope_plan_binding exported{work.out,full_read};
    fort_scope_batch batch{1,FORT_SCOPE_AUTO,size_t(iterations),units,3,&exported,1};
    check(fort_scope_plan_reset_mode(context,FORT_SCOPE_PLAN_CONTINUE));
    fort_scope_plan_binding planned_forget{work.middle,{}};
    check(fort_scope_plan_add(context,FORT_SCOPE_PLAN_FORGET,0,&planned_forget,1,0,0,0));
    fort_scope_access full_input=full_read; fort_scope_section whole_input{{}, {}};
    size_t whole_lo[3]={0,0,0},whole_hi[3]={size_t(work.nx),size_t(work.ny),size_t(iterations+2)};
    if(work.stencil) { whole_input={whole_lo,whole_hi}; full_input.reads=&whole_input; }
    fort_scope_plan_binding planned_a[2]={{work.a,full_input},{work.middle,full_write}},
                            planned_b[2]={{work.middle,full_read},{work.out,full_write}};
    check(fort_scope_plan_add(context,1,101,planned_a,2,1e8,double(bytes),1));
    check(fort_scope_plan_add(context,1,102,planned_b,2,1e8,double(bytes),1));
    check(fort_scope_plan_validate(context)); fort_scope_plan_decision decision{};
    check(fort_scope_plan_select(context,&costs,native ? 0 : 1,&decision));
    assert(decision.gpu_units==(native ? 0U : 2U));
    if(pinned) {
        int chosen=0; check(fort_scope_plan_next(context,101,planned_a,2,&chosen)); assert(chosen);
        check(fort_scope_plan_next(context,102,planned_b,2,&chosen)); assert(chosen);
        direct(context,work); check(fort_scope_close(context));
        for(size_t i=0;i<plane;++i) assert(middle[i]==-101 && out[i]==-202 &&
            middle[items-plane+i]==-101 && out[items-plane+i]==-202);
        assert(out[plane]==2*a[plane]-2-40-600); return;
    }
    if(dirty) {
        fort_scope_access update{}; update.flags=FORT_SCOPE_READ_ALL|FORT_SCOPE_WRITE_ALL;
        void *device=nullptr; check(fort_scope_device_begin(context,work.a,&update,&device));
        check(fort_scope_wait(context));
        std::vector<double> newer=a; for(double &value:newer) value+=700;
        assert(cudaMemcpy(device,newer.data(),bytes,cudaMemcpyHostToDevice)==cudaSuccess);
        check(fort_scope_device_end(context,work.a));
    }
    if(cross) { read_lo[2]=0; read_hi[2]=2; second[0].access=input_read; }
    if(slow || automatic) transfers.pinned_h2d_latency=2;
    fort_scope_batch_report report{},again{};
    check(fort_scope_batch_execute_v1(context,&batch,&costs,&transfers,-1,nullptr,nullptr,&report));
    check(fort_scope_batch_execute_v1(context,&batch,&costs,&transfers,-1,nullptr,nullptr,&again));
    assert(!report.applied && report.preparation_operations==again.preparation_operations && report.preparation_operations>0);
    if(preview || native || cross || automatic) {
        assert(work.calls==0);
        if(preview) assert(report.available && report.selected_transfers==FORT_SCOPE_TRANSFERS_PIPELINED);
        if(native) assert(report.reason==FORT_SCOPE_BATCH_PLACEMENT);
        if(cross) assert(report.reason==FORT_SCOPE_BATCH_UNSUPPORTED_CHAIN);
        if(automatic) assert(report.reason==FORT_SCOPE_BATCH_NO_ADVANTAGE);
        fort_scope_stats actual{}; check(fort_scope_stats_get(context,&actual));
        assert(actual.allocations==0 && actual.uploads==0 && actual.launches==0);
        if(!preview) {
            check(fort_scope_batch_execute_v1(context,&batch,&costs,&transfers,1,nullptr,nullptr,&again));
            assert(!again.applied && again.reason==report.reason);
        }
        int chosen=-1;
        check(fort_scope_plan_next(context,101,planned_a,2,&chosen)); assert(chosen==(native ? 0 : 1));
        check(fort_scope_plan_next(context,102,planned_b,2,&chosen)); assert(chosen==(native ? 0 : 1));
        check(fort_scope_close(context)); return;
    }
    fort_staging::Result occupied;
    if(budget) { occupied=fort_staging::acquire(fort_staging::pinned_limit/2,fort_staging::Role::Compact,false); assert(occupied.slots); }
    if(allocation) setenv("FORT_SCOPE_TEST_FAIL_ALLOC_AFTER","1",1);
    if(record) setenv("FORT_SCOPE_TEST_FAIL_BATCH_RECORD","1",1);
    if(failed_wait) setenv("FORT_SCOPE_TEST_FAIL_BATCH_WAIT","1",1);
    const int status=fort_scope_batch_execute_v1(context,&batch,&costs,&transfers,1,worker,&work,&report);
    if(callback || record || failed_wait) {
        assert(status==FORT_SCOPE_EXECUTION && report.applied && work.calls>0);
        for(double value:out) assert(value==-202);
        assert(fort_scope_host_begin(context,work.out,&full_read)==FORT_SCOPE_EXECUTION);
        assert(fort_scope_close(context)==FORT_SCOPE_EXECUTION); check(fort_scope_abandon(context));
        size_t freed=0; assert(fort_staging::trim_cache(freed)==cudaSuccess);
        assert(fort_staging::usage().reserved==0); return;
    }
    check(status);
    if(budget || allocation) {
        assert(!report.applied && !work.calls && report.reason==(budget ? FORT_SCOPE_BATCH_BUDGET : FORT_SCOPE_BATCH_ALLOCATION));
        unsetenv("FORT_SCOPE_TEST_FAIL_ALLOC_AFTER");
        int chosen=0; check(fort_scope_plan_next(context,101,planned_a,2,&chosen)); assert(chosen);
        check(fort_scope_plan_next(context,102,planned_b,2,&chosen)); assert(chosen);
        direct(context,work);
    } else {
        assert(report.applied && report.selected_transfers==FORT_SCOPE_TRANSFERS_PIPELINED);
        assert(work.streams.size()==2 && work.calls==int(report.batches) && report.completed_batches==report.batches);
        assert(report.actual_launches==2*report.batches && report.actual_download_bytes==plane*iterations*8);
        assert(report.actual_upload_bytes==(dirty ? 0 : work.stencil ? items*8 : plane*iterations*8));
        if(work.stencil) assert(report.prefix_upload_bytes==items*8);
        if(slow) assert(report.execution_seconds>report.baseline_seconds);
        int chosen=0; assert(fort_scope_plan_next(context,101,planned_a,2,&chosen)==FORT_SCOPE_STATE);
        fort_scope_stats before{},after{}; check(fort_scope_stats_get(context,&before)); check(fort_scope_wait(context));
        check(fort_scope_stats_get(context,&after)); assert(before.waits==after.waits);
        fort_scope_transfer_stats actual{}; check(fort_scope_transfer_stats_get_v1(context,&actual));
        assert(actual.staging_device_bytes==0 && actual.process_peak_bytes<=fort_staging::pinned_limit);
    }
    check(fort_scope_close(context));
    for(int z=1;z<=iterations;++z) for(int y=0;y<work.ny;++y) for(int x=0;x<work.nx;++x) {
        const size_t index=x+work.nx*y+plane*z;
        const double expected=work.stencil ? a[index-plane]+a[index]+a[index+plane] : 2*a[index];
        assert(middle[index]==expected && out[index]==expected+(x-2)+10*(y-4)+100*(z-7));
    }
    for(size_t i=0;i<plane;++i) {
        assert(middle[i]==-101 && middle[items-plane+i]==-101);
        assert(out[i]==(work.stencil ? 42 : -202) && out[items-plane+i]==-202);
    }
    size_t freed=0; assert(fort_staging::release(std::move(occupied.slots),freed)==cudaSuccess);
    assert(fort_staging::trim_cache(freed)==cudaSuccess); assert(fort_staging::usage().reserved==0);
}
int main(int argc,char **argv) {
    const char *mode=argc>1 ? argv[1] : "chain";
    if(std::strcmp(mode,"preview")) {
        int devices=0; if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices) return 77;
    }
    if(!std::strcmp(mode,"shapes")) { run("chain",5); run("reverse",37); run("halo",11); }
    else run(mode);
    std::puts("public batch: full-layout bounds, exact sections and replay safety accepted");
}
'''


@pytest.fixture(scope="module")
def cuda_batch_executable(tmp_path_factory):
    nvcc = shutil.which("nvcc")
    host = shutil.which("g++-14") or shutil.which("g++")
    if not nvcc or not host:
        pytest.skip("CUDA toolkit unavailable")
    directory = tmp_path_factory.mktemp("public_batch_cuda")
    client = directory / "client.cu"
    client.write_text(CLIENT)
    objects = []
    for name, source in (("client", client), ("runtime", RUNTIME / "scoped_runtime.cu")):
        target = directory / (name + ".o")
        command = [nvcc, "-std=c++17", "-ccbin", host, "-arch=sm_86", "-I", str(RUNTIME),
                   "-DFORT_SCOPE_TEST_FAULTS", "-c", str(source), "-o", str(target)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
        objects.append(str(target))
    target = directory / "run"
    result = subprocess.run([nvcc, "-ccbin", host, *objects, "-o", str(target)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return target


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["chain", "reverse", "halo", "dirty", "slow", "auto", "native", "cross",
                                 "budget", "allocation", "record", "wait", "callback", "pinned", "shapes"])
def test_real_cuda_public_batch(cuda_batch_executable, mode):
    result = subprocess.run([str(cuda_batch_executable), mode], capture_output=True, text=True, timeout=30)
    if result.returncode == 77:
        pytest.skip("CUDA device unavailable")
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.cuda
def test_preview_has_no_cuda_when_devices_are_hidden(cuda_batch_executable):
    result = subprocess.run([str(cuda_batch_executable), "preview"], capture_output=True, text=True,
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "FORT_RUNTIME_TRACE": "1"}, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FORT_SCOPED initialize" not in result.stderr
