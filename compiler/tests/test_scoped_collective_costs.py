"""Public team calibration is metadata-only and obeys the actual original team."""

import shutil
import subprocess
from pathlib import Path

import pytest

RUNTIME = Path(__file__).resolve().parents[1] / "runtime"
SOURCE = r"""
#include "scoped_runtime.h"
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
void check(int s) { if(s) {std::fprintf(stderr,"%s\n",fort_scope_error());std::abort();} }
void require(bool v) { if(!v) std::abort(); }
fort_scope_team_costs team() {
 fort_scope_team_costs t{}; t.version=1;t.valid=1;t.cpu_threads=4;t.expected_omp_level=1;
 t.protocol_id=FORT_SCOPE_TEAM_PROTOCOL_ID;t.native_cpu_flops=1e12;t.native_cpu_bandwidth=1e12;
 t.owner_seconds=1e-9;t.descriptor_seconds=1e-9;t.entry_seconds=2e-9;
 t.cpu_worker_seconds=3e-9;t.gpu_worker_seconds=4e-9;t.native_call_seconds=5e-9;t.native_worker_seconds=6e-9;
 return t;
}
fort_scope_plan_costs prices() {
 fort_scope_plan_costs c{};c.version=1;c.valid=1;c.max_allocation_bytes=1<<20;
 c.cpu_flops=c.cpu_bandwidth=1e6;c.gpu_flops=c.gpu_bandwidth=1e9;
 c.h2d_bandwidth=c.d2h_bandwidth=1e12;
 c.h2d_latency=c.d2h_latency=c.create_seconds=c.register_seconds=c.host_access_seconds=
 c.device_access_seconds=c.gpu_setup_seconds=c.cold_driver_startup_seconds=c.allocation_seconds=
 c.release_seconds=c.wait_seconds=c.launch_enqueue_seconds=c.planning_operation_seconds=1e-12;
 return c;
}
void inspect(bool compatible,bool ready_expected,bool slow_native=false) {
 fort_scope_t ctx=0;check(fort_scope_create(0,&ctx));auto t=team();
 if(slow_native)t.native_cpu_flops=1e3;
 check(fort_scope_set_team_costs_v1(ctx,&t,compatible));int ready=-1;
 check(fort_scope_team_costs_ready_v1(ctx,&ready));require(bool(ready)==ready_expected);
 check(fort_scope_plan_reset(ctx));check(fort_scope_plan_team_entry_v1(ctx));
 check(fort_scope_plan_add(ctx,FORT_SCOPE_PLAN_WORKER,101,nullptr,0,1e8,0,1));
 check(fort_scope_plan_validate(ctx));
 auto c=prices();
 fort_scope_plan_decision d{};check(fort_scope_plan_select(ctx,&c,-1,&d));
 require(bool(d.available)==ready_expected);
 if(ready_expected) {
   require(bool(d.gpu_units)==slow_native);
   const double expected=1e8/t.native_cpu_flops+t.native_worker_seconds;
   require(std::abs(d.native_seconds-expected)<1e-10);
 }
 fort_scope_stats stats{};check(fort_scope_stats_get(ctx,&stats));
 require(stats.upload_bytes==0&&stats.download_bytes==0&&stats.launches==0&&stats.allocations==0);
 check(fort_scope_close(ctx));
}
void cached_preview(fort_scope_t ctx,bool available,int install=-1) {
 auto c=prices();fort_scope_plan_decision d{};
 check(fort_scope_plan_select(ctx,&c,install,&d));
 require(bool(d.available)==available);require(bool(d.gpu_units)==available);
 if(install==1) {int gpu=0;check(fort_scope_plan_next(ctx,102,nullptr,0,&gpu));require(gpu==1);}
 fort_scope_stats stats{};check(fort_scope_stats_get(ctx,&stats));
 require(!stats.upload_bytes&&!stats.download_bytes&&!stats.launches&&!stats.allocations);
}
int main() {
 omp_set_dynamic(0);omp_set_max_active_levels(2);
 inspect(true,false); // A serial caller cannot authorize a persistent team.
 #pragma omp parallel num_threads(4)
 { #pragma omp master
   {inspect(true,true);inspect(true,true,true);inspect(false,false);}
 }
 #pragma omp parallel num_threads(3)
 { #pragma omp master
   {inspect(true,false);}
 }
 fort_scope_t retained=0;
 #pragma omp parallel num_threads(4)
 { #pragma omp master
   {check(fort_scope_create(0,&retained));auto t=team();t.native_cpu_flops=1e3;
    check(fort_scope_set_team_costs_v1(retained,&t,1));check(fort_scope_plan_reset(retained));
    check(fort_scope_plan_team_entry_v1(retained));
    check(fort_scope_plan_add(retained,FORT_SCOPE_PLAN_WORKER,102,nullptr,0,1e8,0,1));
    check(fort_scope_plan_validate(retained));cached_preview(retained,true);}
 }
 cached_preview(retained,false);
 #pragma omp parallel num_threads(3)
 { #pragma omp master
   {cached_preview(retained,false);}
 }
 #pragma omp parallel num_threads(1)
 { #pragma omp parallel num_threads(4)
   { #pragma omp master
     {cached_preview(retained,false);}
   }
 }
 #pragma omp parallel num_threads(4)
 { #pragma omp master
   {cached_preview(retained,true);cached_preview(retained,true,1);}
 }
 check(fort_scope_close(retained));
 std::puts("TEAM_COSTS_OK");
}
""".replace("{ #pragma", "{\n #pragma")


@pytest.fixture(scope="module")
def team_costs_binary(tmp_path_factory):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("C++ OpenMP compiler unavailable")
    directory = tmp_path_factory.mktemp("collective_costs")
    source = directory / "check.cpp"
    source.write_text(SOURCE)
    binary = directory / "check"
    result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-fopenmp",
            "-DFORT_SCOPE_CPU_TEST",
            "-I",
            str(RUNTIME),
            str(source),
            "-x",
            "c++",
            str(RUNTIME / "scoped_runtime.cu"),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return binary


def test_actual_team_readiness_and_original_native_counterfactual_rates(team_costs_binary):
    result = subprocess.run([str(team_costs_binary)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "TEAM_COSTS_OK" in result.stdout
