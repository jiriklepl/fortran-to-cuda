"""Deferred CUDA tasks cross entries, own values and drain dependency closures."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from compiler.tests.test_scoped_graph_replay_cuda import CLIENT as GRAPH_CLIENT
from compiler.tests.test_scoped_runtime import RUNTIME

QUEUE_PREFIX = GRAPH_CLIENT.split("int main(", 1)[0].replace(
    "check(fort_scope_create(0,&context));",
    "check(fort_scope_create(0,&context)); "
    "if(const auto *budget=std::getenv(\"FORT_SCOPE_QUEUE_TEST_BUDGET\")) "
    "check(fort_scope_set_device_budget(context,std::strtoull(budget,nullptr,10)));",
)

CLIENT = QUEUE_PREFIX + r'''
#include <string>
__global__ void delayed_copy(const double *input,double *output,unsigned long long cycles) {
    const int i=threadIdx.x; if(i>=8) return;
    const auto start=clock64(); while(clock64()-start<cycles) {}
    output[i]=input[i];
}
fort_scope_graph_queue_stats_v1 queued(Fixture &f) {
    fort_scope_graph_queue_stats_v1 s{}; check(fort_scope_graph_queue_stats_get_v1(f.context,&s)); return s;
}
fort_scope_residency_stats_v1 residency(Fixture &f) {
    fort_scope_residency_stats_v1 s{}; check(fort_scope_residency_stats_get_v1(f.context,&s)); return s;
}
void enqueue(Fixture &f,size_t i) {
    size_t sizes[6]={sizeof(double *),sizeof(double *),sizeof(int),sizeof(int),sizeof(int),sizeof(double)};
    check(fort_scope_graph_enqueue_v1(f.context,100+i,&f.tasks[i],sizes));
}
void chain(Fixture &f) { enqueue(f,0); enqueue(f,1); }
int main(int argc,char **argv) {
    assert(argc==2); const std::string mode=argv[1];
    int devices=0; if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices) return 77;
    if(mode=="queue_off") setenv("FORT_SCOPE_GRAPH_QUEUE","0",1);
    if(mode=="graphs_off") setenv("FORT_SCOPE_CUDA_GRAPHS","0",1);
    if(mode=="long_mirrors" || mode=="pressure") setenv("FORT_SCOPE_QUEUE_TEST_BUDGET","192",1);
    Fixture f;
    if(mode=="stable_parameters") {
        for(int round=0;round<6;++round) {
            if(round==3) f.add=4;
            if(round==4) f.tasks[0].kernel.grid[0]=2;
            if(round==5) f.rebind();
            chain(f);
            if(round==5) f.observe(f.other); else f.observe(f.out);
            const auto *out=round==5?f.host_other:f.host_out;
            for(int i=0;i<8;++i) assert(out[i]==3*(i+(round<3?2:4)));
            const auto updates=f.graph().parameter_updates;
            assert(updates==(round<3?0u:unsigned(round-2)));
        }
    } else if(mode=="trace_origins") {
        double *other=nullptr;
        fort_scope_layout layout{1,FORT_SCOPE_REAL64,8,f.host_other,f.extent,f.lower,1};
        check(fort_scope_register(f.context,4,1,&layout,0,&f.other));
        check(fort_scope_device_reserve_v1(f.context,f.other,reinterpret_cast<void **>(&other)));
        const std::string owner(64,'a'),procedure(64,'b'),segment(64,'c'),operation(64,'d'),boundary(64,'e');
        auto position=[&](char id) {
            const std::string implementation(64,id);
            fort_scope_trace_set_v1(f.context,FORT_SCOPE_TRACE_ALL,owner.c_str(),procedure.c_str(),
                segment.c_str(),operation.c_str(),implementation.c_str(),boundary.c_str(),nullptr);
        };
        for(int round=0;round<3;++round) {
            position('1'+3*round); enqueue(f,0);
            position('2'+3*round);
            {
                double add=10+round; void *args[6]={&f.pa,&other,&f.n,&f.first,&f.last,&add};
                auto task=f.tasks[0]; task.kernel.arguments=args;
                fort_scope_plan_binding bindings[2]={f.producer[0],{f.other,f.producer[1].access}};
                task.bindings=bindings;
                size_t sizes[6]={8,8,4,4,4,8};
                check(fort_scope_graph_enqueue_v1(f.context,200,&task,sizes));
            }
            position('3'+3*round); enqueue(f,1);
            position('f');
            f.observe(f.out); // Selective conflict flush: producer+consumer, middle task stays queued.
            assert(queued(f).pending_tasks==1);
            for(int i=0;i<8;++i) assert(f.host_out[i]==3*(i+2));
            f.observe(f.other);
            for(int i=0;i<8;++i) assert(f.host_other[i]==i+10+round);
            fort_scope_trace_state_v1 current{};
            fort_scope_trace_set_v1(f.context,0,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,&current);
            assert(std::string(current.implementation)==std::string(64,'f'));
            assert(!queued(f).pending_tasks && f.stats().launches==3u*(round+1));
        }
    } else if(mode=="plan_suffix") {
        check(fort_scope_task_graph_enable_v1(f.context));
        check(fort_scope_plan_reset_mode(f.context,FORT_SCOPE_PLAN_CONTINUE));
        check(fort_scope_plan_add(f.context,FORT_SCOPE_PLAN_WORKER,100,f.producer,2,1e8,128,1));
        check(fort_scope_plan_add(f.context,FORT_SCOPE_PLAN_WORKER,101,f.consumer,2,1e8,128,1));
        fort_scope_plan_costs costs{}; costs.version=FORT_SCOPE_PLANNING_ABI_VERSION; costs.valid=1; costs.max_allocation_bytes=1u<<26;
        costs.cpu_flops=costs.cpu_bandwidth=1e8;
        costs.gpu_flops=costs.gpu_bandwidth=costs.h2d_bandwidth=costs.d2h_bandwidth=1e12;
        costs.create_seconds=costs.register_seconds=costs.host_access_seconds=costs.device_access_seconds=1e-9;
        costs.gpu_setup_seconds=costs.cold_driver_startup_seconds=costs.allocation_seconds=costs.release_seconds=1e-9;
        costs.wait_seconds=costs.launch_enqueue_seconds=costs.planning_operation_seconds=1e-9;
        fort_scope_plan_decision decision{};
        check(fort_scope_plan_validate(f.context));
        check(fort_scope_plan_select(f.context,&costs,1,&decision));
        assert(decision.available && decision.gpu_units==2);
        int gpu=0; check(fort_scope_plan_next(f.context,100,f.producer,2,&gpu)); assert(gpu);
        enqueue(f,0); assert(queued(f).pending_tasks==1 && !f.stats().launches);
        assert(fort_scope_close(f.context)==FORT_SCOPE_STATE);
        check(fort_scope_plan_abandon_remaining_v1(f.context));
        assert(queued(f).pending_tasks==1 && !f.stats().launches && !f.stats().upload_bytes);
        // No future output definition was admitted, and its failed read does
        // not launch or discard the independent accepted producer.
        fort_scope_access read{}; read.flags=FORT_SCOPE_READ_ALL;
        assert(fort_scope_host_begin(f.context,f.out,&read)==FORT_SCOPE_UNINITIALIZED);
        assert(queued(f).pending_tasks==1);
        f.observe(f.b); assert(f.stats().launches==1 && !queued(f).pending_tasks);
        fort_scope_access write{}; write.flags=FORT_SCOPE_WRITE_ALL|FORT_SCOPE_OVERWRITE_ALL;
        check(fort_scope_host_begin(f.context,f.out,&write));
        for(int i=0;i<8;++i) { assert(f.host_b[i]==i+2); f.host_out[i]=3*f.host_b[i]; }
        check(fort_scope_host_end(f.context,f.out));
        assert(f.stats().upload_bytes==64 && f.stats().download_bytes==64);
        // The native read preserved B's GPU mirror for a later GPU consumer.
        enqueue(f,1); f.observe(f.out);
        for(int i=0;i<8;++i) assert(f.host_out[i]==3*(i+2));
        assert(f.stats().launches==2 && f.stats().upload_bytes==64 && f.stats().download_bytes==128);
    } else if(mode=="pressure") {
        enqueue(f,0); assert(!f.stats().launches);
        fort_scope_layout layout{1,FORT_SCOPE_REAL64,8,f.host_other,f.extent,f.lower,1};
        check(fort_scope_register(f.context,4,1,&layout,0,&f.other));
        fort_buffer_t group[3]={f.a,f.out,f.other}; void *pointers[3]{};
        check(fort_scope_device_reserve_many_v1(f.context,group,3,pointers));
        f.pa=static_cast<double *>(pointers[0]); f.po=static_cast<double *>(pointers[1]);
        // Pending producer arguments pinned their allocations until submission.
        // The only unrequested victim is its dirty output B.
        assert(f.stats().launches==1 && !queued(f).pending_tasks);
        assert(residency(f).pressure_flushes==1 && residency(f).dirty_evictions==1);
        for(int i=0;i<8;++i) assert(f.host_b[i]==i+2);
        assert(f.stats().upload_bytes==64 && f.stats().download_bytes==64);
        fort_buffer_t consume_group[2]={f.b,f.out};
        check(fort_scope_device_reserve_many_v1(f.context,consume_group,2,pointers));
        f.pb=static_cast<double *>(pointers[0]); f.po=static_cast<double *>(pointers[1]);
        enqueue(f,1); f.observe(f.out);
        for(int i=0;i<8;++i) assert(f.host_out[i]==3*(i+2));
        // Replay the producer graph after both of its allocation pointers have
        // changed. No cached graph may retain an evicted pointer argument.
        fort_buffer_t produce_group[2]={f.a,f.b};
        check(fort_scope_device_reserve_many_v1(f.context,produce_group,2,pointers));
        f.pa=static_cast<double *>(pointers[0]); f.pb=static_cast<double *>(pointers[1]); f.add=5;
        enqueue(f,0); f.observe(f.b);
        for(int i=0;i<8;++i) assert(f.host_b[i]==i+5 && f.host_out[i]==3*(i+2));
        assert(f.graph().creations==1 && !f.graph().replays && queued(f).direct_kernels==2);
        assert(f.stats().allocated_bytes==192 && f.stats().peak_device_bytes==192);
        assert(f.stats().upload_bytes==192 && f.stats().download_bytes==192);
    } else if(mode=="long_mirrors") {
        const auto allocations=f.stats().allocations;
        fort_scope_access read{}; read.flags=FORT_SCOPE_READ_ALL;
        for(int round=0;round<16;++round) {
            check(fort_scope_host_begin(f.context,f.a,&read)); check(fort_scope_host_end(f.context,f.a));
            enqueue(f,0); check(fort_scope_graph_flush_v1(f.context));
            check(fort_scope_host_begin(f.context,f.a,&read));
            for(int i=0;i<8;++i) assert(f.host_a[i]==i);
            check(fort_scope_host_end(f.context,f.a));
            assert(f.stats().upload_bytes==64 && !f.stats().download_bytes && f.stats().allocations==allocations);
        }
        f.observe(f.b); assert(f.stats().download_bytes==64);
        for(int round=0;round<16;++round) {
            check(fort_scope_host_begin(f.context,f.b,&read)); check(fort_scope_host_end(f.context,f.b));
            enqueue(f,1); check(fort_scope_graph_flush_v1(f.context));
            check(fort_scope_host_begin(f.context,f.b,&read));
            for(int i=0;i<8;++i) assert(f.host_b[i]==i+2);
            check(fort_scope_host_end(f.context,f.b));
            assert(f.stats().upload_bytes==64 && f.stats().download_bytes==64 && f.stats().allocations==allocations);
        }
        // The output of the read probes is dead. Its retirement must not add
        // publication traffic to the two preserved read-only mirrors.
        check(fort_scope_retire_v1(f.context,f.out));
        assert(f.stats().upload_bytes==64 && f.stats().download_bytes==64 && f.stats().allocations==allocations);
        assert(f.graph().creations==2 && f.graph().replays==28 && queued(f).accepted_tasks==32);
        assert(queued(f).direct_submissions==2 && queued(f).direct_kernels==2);
    } else if(mode=="mirrors") {
        enqueue(f,0); f.observe(f.b);
        auto read=fort_scope_access{}; read.flags=FORT_SCOPE_READ_ALL;
        const auto original=f.stats();
        check(fort_scope_host_begin(f.context,f.b,&read)); check(fort_scope_host_end(f.context,f.b));
        enqueue(f,1); check(fort_scope_graph_flush_v1(f.context));
        assert(f.stats().upload_bytes==original.upload_bytes && f.stats().download_bytes==original.download_bytes);
        f.observe(f.out);
        for(int i=0;i<8;++i) assert(f.host_out[i]==3*(i+2));
        size_t lo[1]={0},hi[1]={2}; fort_scope_section first{lo,hi};
        fort_scope_access writes{0,0,nullptr,1,&first,1,&first};
        check(fort_scope_host_begin(f.context,f.b,&writes)); f.host_b[0]=-1; f.host_b[1]=-2;
        check(fort_scope_host_end(f.context,f.b));
        const auto before=f.stats(); enqueue(f,1); check(fort_scope_graph_flush_v1(f.context));
        assert(f.stats().upload_bytes==before.upload_bytes+2*sizeof(double));
        check(fort_scope_host_begin(f.context,f.b,&read)); check(fort_scope_host_end(f.context,f.b));
        assert(f.stats().download_bytes==before.download_bytes);
        f.observe(f.out); assert(f.host_out[0]==-3 && f.host_out[1]==-6);
    } else if(mode=="host_mirror") {
        auto write=fort_scope_access{}; write.flags=FORT_SCOPE_WRITE_ALL|FORT_SCOPE_OVERWRITE_ALL;
        auto read=fort_scope_access{}; read.flags=FORT_SCOPE_READ_ALL;
        check(fort_scope_host_begin(f.context,f.a,&write));
        for(int i=0;i<8;++i) f.host_a[i]=100+i;
        check(fort_scope_host_end(f.context,f.a));
        enqueue(f,0); check(fort_scope_graph_flush_v1(f.context));
        const auto before=f.stats(); assert(before.upload_bytes==64 && !before.download_bytes);
        check(fort_scope_host_begin(f.context,f.a,&read));
        for(int i=0;i<8;++i) assert(f.host_a[i]==100+i);
        check(fort_scope_host_end(f.context,f.a));
        assert(f.stats().upload_bytes==before.upload_bytes && f.stats().download_bytes==before.download_bytes);
        f.observe(f.b);
        for(int i=0;i<8;++i) assert(f.host_b[i]==102+i);
    } else if(mode=="selective") {
        enqueue(f,0);
        double *other=nullptr;
        fort_scope_layout layout{1,FORT_SCOPE_REAL64,8,f.host_other,f.extent,f.lower,1};
        check(fort_scope_register(f.context,4,1,&layout,0,&f.other));
        check(fort_scope_device_reserve_v1(f.context,f.other,reinterpret_cast<void **>(&other)));
        { // Every argument and coordinate here expires before observation.
            double add=10; void *args[6]={&f.pa,&other,&f.n,&f.first,&f.last,&add};
            auto task=f.tasks[0]; task.kernel.arguments=args;
            fort_scope_plan_binding bindings[2]={f.producer[0],{f.other,f.producer[1].access}};
            task.bindings=bindings;
            size_t sizes[6]={8,8,4,4,4,8};
            check(fort_scope_graph_enqueue_v1(f.context,200,&task,sizes));
        }
        enqueue(f,1); assert(!f.stats().launches && queued(f).pending_tasks==3);
        f.observe(f.out);
        assert(queued(f).pending_tasks==1 && queued(f).submissions==1 && f.stats().launches==2);
        for(int i=0;i<8;++i) assert(f.host_out[i]==3*(i+2) && f.host_other[i]==0);
        f.observe(f.other);
        for(int i=0;i<8;++i) assert(f.host_other[i]==i+10);
        assert(queued(f).submissions==2 && !queued(f).pending_tasks && f.stats().upload_bytes==64);
    } else if(mode=="war") {
        int previous=0; void *raw=nullptr; check(fort_scope_gpu_enter(f.context,&previous,&raw));
        int khz=0; assert(cudaDeviceGetAttribute(&khz,cudaDevAttrClockRate,0)==cudaSuccess);
        unsigned long long cycles=static_cast<unsigned long long>(khz)*40;
        void *args[3]={&f.pa,&f.pb,&cycles}; size_t sizes[3]={sizeof(f.pa),sizeof(f.pb),sizeof(cycles)};
        auto task=f.tasks[0]; task.kernel.function=reinterpret_cast<const void *>(delayed_copy);
        task.kernel.arguments=args; task.kernel.argument_count=3;
        check(fort_scope_graph_enqueue_v1(f.context,300,&task,sizes));
        fort_scope_access overwrite{}; overwrite.flags=FORT_SCOPE_WRITE_ALL|FORT_SCOPE_OVERWRITE_ALL;
        check(fort_scope_host_begin(f.context,f.a,&overwrite));
        cudaEvent_t marker=nullptr; assert(cudaEventCreateWithFlags(&marker,cudaEventDisableTiming)==cudaSuccess);
        assert(cudaEventRecord(marker,static_cast<cudaStream_t>(raw))==cudaSuccess);
        assert(cudaEventQuery(marker)==cudaErrorNotReady); // The kernel is still running.
        for(int i=0;i<8;++i) f.host_a[i]=100+i;
        check(fort_scope_host_end(f.context,f.a)); f.observe(f.b);
        for(int i=0;i<8;++i) assert(f.host_b[i]==i);
        assert(cudaEventDestroy(marker)==cudaSuccess); check(fort_scope_gpu_leave(f.context,previous));
        enqueue(f,0); f.observe(f.b);
        for(int i=0;i<8;++i) assert(f.host_b[i]==102+i);
    } else if(mode=="entry_observe" || mode=="entry_failure") {
        if(mode=="entry_failure") { chain(f); check(fort_scope_observe_all_v1(f.context)); }
        chain(f); assert(queued(f).pending_tasks==2);
        if(mode=="entry_failure") setenv("FORT_SCOPE_TEST_FAIL_GRAPH_LAUNCH","1",1);
        const int status=fort_scope_observe_all_v1(f.context);
        if(mode=="entry_failure") {
            assert(status==FORT_SCOPE_EXECUTION);
            assert(fort_scope_observe_all_v1(f.context)==FORT_SCOPE_EXECUTION);
            assert(fort_scope_graph_flush_v1(f.context)==FORT_SCOPE_EXECUTION);
            assert(fort_scope_graph_queue_cancel_v1(f.context)==FORT_SCOPE_EXECUTION);
            check(fort_scope_abandon(f.context)); return 0;
        }
        check(status);
        assert(!queued(f).pending_tasks && f.stats().launches==2);
        for(int i=0;i<8;++i) assert(f.host_b[i]==i+2 && f.host_out[i]==3*(i+2));
        assert(f.stats().upload_bytes==64 && f.stats().download_bytes==128);
        fort_scope_access read{}; read.flags=FORT_SCOPE_READ_ALL; void *pointer=nullptr;
        check(fort_scope_device_begin(f.context,f.b,&read,&pointer));
        check(fort_scope_device_end(f.context,f.b)); assert(pointer==f.pb);
        check(fort_scope_observe_all_v1(f.context));
        assert(f.stats().upload_bytes==64 && f.stats().download_bytes==128);
    } else {
        if(mode=="resource_fallback" || mode=="launch_failure") {
            chain(f); f.observe(f.out); // Warm directly before exercising actual graph APIs.
            assert(!f.graph().creations && queued(f).direct_submissions==1);
        }
        const int repetitions=mode=="repeated"?3:1;
        for(int round=0;round<repetitions;++round) { f.add=2+round; chain(f); }
        // Mutating all original metadata/values after acceptance cannot alter
        // captured kernel parameters or effect footprints.
        f.add=f.factor=999; f.lo[0]=4; f.hi[0]=5; f.first=4; f.last=5;
        if(mode!="queue_off") {
            assert(f.stats().launches==((mode=="resource_fallback" || mode=="launch_failure")?2u:0u));
            assert(queued(f).pending_tasks==2u*repetitions);
        }
        double host_x[8]{}; size_t extent[1]={8}; int64_t lower[1]={0}; fort_buffer_t x=0;
        fort_scope_layout layout{1,FORT_SCOPE_REAL64,8,host_x,extent,lower,1};
        check(fort_scope_register(f.context,99,1,&layout,1,&x));
        fort_scope_access native{}; native.flags=FORT_SCOPE_WRITE_ALL|FORT_SCOPE_OVERWRITE_ALL;
        check(fort_scope_host_begin(f.context,x,&native)); host_x[0]=77; check(fort_scope_host_end(f.context,x));
        if(mode!="queue_off") assert(queued(f).pending_tasks==2u*repetitions);
        if(mode=="resource_fallback") setenv("FORT_SCOPE_TEST_FAIL_GRAPH_CREATE","1",1);
        if(mode=="launch_failure") setenv("FORT_SCOPE_TEST_FAIL_GRAPH_LAUNCH","1",1);
        const int status=fort_scope_observe_v1(f.context,&f.out,1);
        if(mode=="launch_failure") {
            assert(status==FORT_SCOPE_EXECUTION);
            assert(fort_scope_graph_flush_v1(f.context)==FORT_SCOPE_EXECUTION);
            assert(fort_scope_graph_queue_cancel_v1(f.context)==FORT_SCOPE_EXECUTION);
            check(fort_scope_abandon(f.context)); return 0;
        }
        check(status);
        for(int i=0;i<8;++i) assert(f.host_out[i]==3*(i+1+repetitions));
        const unsigned previous=mode=="resource_fallback"?1u:0u;
        assert(!queued(f).pending_tasks && f.stats().launches==2u*(repetitions+previous));
        assert(f.stats().upload_bytes==64 && f.stats().download_bytes==64u*(1u+previous));
        if(mode=="queue_off") assert(queued(f).submissions==2 && !f.graph().submissions);
        else assert(queued(f).submissions==1u+previous);
        if(mode=="graphs_off" || mode=="resource_fallback") {
            assert(!f.graph().creations && queued(f).direct_submissions==1u+previous && queued(f).direct_kernels==2u*(1u+previous));
            if(mode=="resource_fallback") assert(queued(f).graph_resource_fallbacks==1);
        }
    }
    check(fort_scope_close(f.context)); std::puts("deferred graph queue accepted");
}
'''


@pytest.fixture(scope="module")
def cuda_queue_executable(tmp_path_factory):
    prepared = os.environ.get("FORT_TEST_QUEUE_PREBUILT")
    if prepared:
        return Path(prepared)
    nvcc = shutil.which("nvcc")
    host = shutil.which("g++-14") or shutil.which("g++")
    if not nvcc or not host:
        pytest.skip("CUDA toolkit unavailable")
    directory = tmp_path_factory.mktemp("scoped_graph_queue_cuda")
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
@pytest.mark.parametrize("mode", ["owned", "repeated", "selective", "war", "queue_off", "graphs_off",
                                 "resource_fallback", "launch_failure", "mirrors", "host_mirror", "long_mirrors", "pressure",
                                 "plan_suffix", "entry_observe", "entry_failure", "stable_parameters"])
def test_deferred_real_cuda_queue(cuda_queue_executable, mode):
    result = subprocess.run([str(cuda_queue_executable), mode], capture_output=True, text=True, timeout=20)
    if result.returncode == 77:
        pytest.skip("CUDA device unavailable")
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.cuda
@pytest.mark.parametrize("graphs", [False, True])
@pytest.mark.parametrize("allocation_failure", [None, "enqueue", "flush"])
def test_queued_launches_keep_original_provenance(cuda_queue_executable, graphs, allocation_failure):
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("FORT_")}
    environment.update(FORT_RUNTIME_TRACE="1", FORT_SCOPE_CUDA_GRAPHS="1" if graphs else "0")
    if allocation_failure:
        environment["FORT_SCOPE_TEST_FAIL_TRACE_ALLOC"] = allocation_failure
    result = subprocess.run([str(cuda_queue_executable), "trace_origins"], capture_output=True,
                            text=True, timeout=20, env=environment)
    if result.returncode == 77:
        pytest.skip("CUDA device unavailable")
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [(match[1], dict(re.findall(r"(\w+)=([^\s]+)", match[2])))
            for line in result.stderr.splitlines()
            if (match := re.match(r"FORT_SCOPED (launch|graph_kernel|graph_submit|download) (.*)", line))]
    launched = [row for event, row in rows if event in {"launch", "graph_kernel"}]
    expected = ["unknown"] * 9 if allocation_failure else [char * 64 for char in "132465798"]
    assert [row["implementation"] for row in launched] == expected
    assert all(row["bytes"] == "0" for row in launched)
    if not allocation_failure:
        assert all(row["owner"] == "a" * 64 and row["procedure"] == "b" * 64
                   and row["operation"] == "d" * 64 for row in launched)
    else:
        assert all(row["owner"] == row["procedure"] == row["operation"] == "unknown" for row in launched)
    assert sum(event == "launch" for event, _ in rows) == (3 if graphs else 9)
    assert sum(event == "graph_kernel" for event, _ in rows) == (6 if graphs else 0)
    assert sum(event == "graph_submit" for event, _ in rows) == (4 if graphs else 0)
    assert all(row["implementation"] == "f" * 64 for event, row in rows if event == "download")
