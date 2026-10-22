"""An original native unit retains real queued GPU producers and later choices."""
from __future__ import annotations

import shutil
import subprocess

import pytest

from compiler.tests.test_scoped_graph_replay_cuda import CLIENT as GRAPH_CLIENT
from compiler.tests.test_scoped_runtime import RUNTIME

CLIENT = GRAPH_CLIENT.split("int main(", 1)[0] + r'''
#include <string>
void enqueue(Fixture &f,size_t i) {
    size_t sizes[6]={sizeof(double *),sizeof(double *),sizeof(int),sizeof(int),sizeof(int),sizeof(double)};
    check(fort_scope_graph_enqueue_v1(f.context,100+i,&f.tasks[i],sizes));
}
fort_scope_graph_queue_stats_v1 queued(Fixture &f) {
    fort_scope_graph_queue_stats_v1 s{}; check(fort_scope_graph_queue_stats_get_v1(f.context,&s)); return s;
}
fort_scope_plan_costs costs() {
    fort_scope_plan_costs c{}; c.version=FORT_SCOPE_PLANNING_ABI_VERSION; c.valid=1; c.max_allocation_bytes=1u<<26;
    c.cpu_flops=c.cpu_bandwidth=1e8;
    c.gpu_flops=c.gpu_bandwidth=c.h2d_bandwidth=c.d2h_bandwidth=1e12;
    c.create_seconds=c.register_seconds=c.host_access_seconds=c.device_access_seconds=1e-9;
    c.gpu_setup_seconds=c.cold_driver_startup_seconds=c.allocation_seconds=c.release_seconds=1e-9;
    c.wait_seconds=c.launch_enqueue_seconds=c.planning_operation_seconds=1e-9;
    return c;
}
int main(int argc,char **argv) {
    assert(argc==2); const std::string mode=argv[1];
    int devices=0; if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices) return 77;
    Fixture f;
    double original[8]{}; fort_buffer_t x=0;
    fort_scope_layout layout{1,FORT_SCOPE_REAL64,8,original,f.extent,f.lower,1};
    check(fort_scope_register(f.context,99,1,&layout,1,&x));
    enqueue(f,0); assert(queued(f).pending_tasks==1 && !f.stats().launches);
    check(fort_scope_task_graph_enable_v1(f.context));
    check(fort_scope_plan_reset_mode(f.context,FORT_SCOPE_PLAN_CONTINUE));
    fort_scope_access read{}; read.flags=FORT_SCOPE_READ_ALL;
    fort_scope_access write{}; write.flags=FORT_SCOPE_WRITE_ALL|FORT_SCOPE_OVERWRITE_ALL;
    fort_scope_access edit{}; edit.flags=FORT_SCOPE_READ_ALL|FORT_SCOPE_WRITE_ALL;
    fort_scope_plan_binding expected{mode=="independent"?x:f.b,mode=="write"?edit:read};
    check(fort_scope_plan_team_entry_v1(f.context));
    check(fort_scope_plan_add(f.context,FORT_SCOPE_PLAN_WORKER,200,&expected,1,1e9,64,0));
    check(fort_scope_plan_add(f.context,FORT_SCOPE_PLAN_WORKER,201,f.consumer,2,1e9,128,1));
    check(fort_scope_plan_validate(f.context));
    auto c=costs(); fort_scope_plan_decision decision{};
    check(fort_scope_plan_select(f.context,&c,1,&decision));
    assert(decision.cpu_units==1 && decision.gpu_units==1);
    fort_scope_native_worker_v1 worker{200,&expected,1};
    fort_scope_plan_binding native{expected.buffer,mode=="independent"?write:expected.access};
    int applied=99;
    if(mode=="late_failure") {
        fort_scope_plan_binding bad[2]={native,{f.out,read}};
        assert(fort_scope_plan_native_begin_v1(f.context,&worker,1,bad,2,&applied)==FORT_SCOPE_UNINITIALIZED);
        assert(!applied && queued(f).pending_tasks==1 && !f.stats().launches && !f.stats().download_bytes);
    }
    if(mode=="mixed") {
        fort_scope_native_worker_v1 both[2]={worker,{201,f.consumer,2}};
        check(fort_scope_plan_native_begin_v1(f.context,both,2,&native,1,&applied));
        assert(!applied && queued(f).pending_tasks==1 && !f.stats().launches);
    }
    check(fort_scope_plan_native_begin_v1(f.context,&worker,1,&native,1,&applied));
    assert(applied==1);
    if(mode=="independent") {
        assert(queued(f).pending_tasks==1 && !f.stats().launches && !f.stats().download_bytes);
        for(int i=0;i<8;++i) original[i]=100+i;
    } else {
        assert(!queued(f).pending_tasks && f.stats().launches==1);
        assert(f.stats().upload_bytes==64 && f.stats().download_bytes==64);
        for(int i=0;i<8;++i) {
            assert(f.host_b[i]==i+2);
            if(mode=="write") f.host_b[i]+=10;
        }
    }
    check(fort_scope_host_end(f.context,native.buffer));
    fort_scope_plan_report report{}; check(fort_scope_plan_report_v2(f.context,&report));
    assert(!report.available && !report.owner_available);
    int gpu=0; check(fort_scope_plan_next(f.context,201,f.consumer,2,&gpu)); assert(gpu);
    enqueue(f,1); f.observe(f.out);
    for(int i=0;i<8;++i) {
        assert(f.host_out[i]==3*(i+(mode=="write"?12:2)));
        if(mode=="independent") assert(original[i]==100+i);
    }
    assert(f.stats().launches==2 && !queued(f).pending_tasks);
    assert(f.stats().upload_bytes==(mode=="write"?128u:64u));
    assert(f.stats().download_bytes==(mode=="independent"?64u:128u));
    assert(queued(f).submissions==(mode=="independent"?1u:2u));
    check(fort_scope_close(f.context));
    std::puts("original native unit handoff accepted");
}
'''


@pytest.fixture(scope="module")
def native_unit_cuda_executable(tmp_path_factory):
    nvcc = shutil.which("nvcc")
    host = shutil.which("g++-14") or shutil.which("g++")
    if not nvcc or not host:
        pytest.skip("CUDA toolkit unavailable")
    directory = tmp_path_factory.mktemp("scoped_native_unit_cuda")
    source = directory / "client.cu"
    source.write_text(CLIENT)
    objects = []
    for name, path in [("runtime", RUNTIME / "scoped_runtime.cu"), ("client", source)]:
        target = directory / (name + ".o")
        result = subprocess.run([
            nvcc, "-std=c++17", "-ccbin", host, "-arch=sm_86", "-I", str(RUNTIME),
            "-c", str(path), "-o", str(target),
        ], capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
        objects.append(str(target))
    target = directory / "run"
    result = subprocess.run([nvcc, "-ccbin", host, *objects, "-o", str(target)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return target


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["read", "write", "independent", "mixed", "late_failure"])
def test_current_native_unit_real_cuda(native_unit_cuda_executable, mode):
    result = subprocess.run([str(native_unit_cuda_executable), mode], capture_output=True,
                            text=True, timeout=20)
    if result.returncode == 77:
        pytest.skip("CUDA device unavailable")
    assert result.returncode == 0, result.stdout + result.stderr
