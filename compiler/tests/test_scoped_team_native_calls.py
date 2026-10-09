"""Original collective native-call markers invalidate metadata-only queries."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

RUNTIME = Path(__file__).resolve().parents[1] / "runtime"
SOURCE = r'''
#include "scoped_runtime.h"
#include <omp.h>
#include <cassert>
#include <cmath>
int main() {
    omp_set_dynamic(0);
    #pragma omp parallel num_threads(4)
    {
        #pragma omp master
        {
            fort_scope_t context=0;
            assert(!fort_scope_create(0,&context));
            fort_scope_team_costs team{};
            team.version=FORT_SCOPE_TEAM_ABI_VERSION; team.valid=1;
            team.cpu_threads=4; team.expected_omp_level=1;
            team.protocol_id=FORT_SCOPE_TEAM_PROTOCOL_ID;
            team.native_cpu_flops=team.native_cpu_bandwidth=1000;
            team.native_call_seconds=.5;
            assert(!fort_scope_set_team_costs_v1(context,&team,1));
            assert(fort_scope_plan_team_native_call_v1(context)==FORT_SCOPE_STATE);
            assert(!fort_scope_plan_reset(context));
            assert(!fort_scope_plan_team_entry_v1(context));
            assert(!fort_scope_plan_add(context,FORT_SCOPE_PLAN_WORKER,101,nullptr,0,1000,0,1));
            assert(!fort_scope_plan_validate(context));
            assert(!fort_scope_plan_validate(context));
            fort_scope_plan_costs costs{};
            costs.version=FORT_SCOPE_PLANNING_ABI_VERSION; costs.valid=1;
            costs.max_allocation_bytes=1<<20;
            costs.cpu_flops=costs.cpu_bandwidth=100;
            costs.gpu_flops=1000/.6; costs.gpu_bandwidth=10000;
            costs.h2d_bandwidth=costs.d2h_bandwidth=1000;
            costs.create_seconds=costs.register_seconds=costs.host_access_seconds=
                costs.device_access_seconds=costs.gpu_setup_seconds=costs.cold_driver_startup_seconds=
                costs.allocation_seconds=costs.release_seconds=costs.wait_seconds=
                costs.launch_enqueue_seconds=costs.planning_operation_seconds=1e-12;
            fort_scope_plan_decision before{},after{};
            assert(!fort_scope_plan_select(context,&costs,-1,&before));
            assert(before.available && before.gpu_units==1);
            assert(!fort_scope_plan_team_native_call_v1(context));
            assert(!fort_scope_plan_validate(context));
            assert(!fort_scope_plan_select(context,&costs,1,&after));
            assert(after.available && !after.gpu_units);
            assert(std::abs(after.native_seconds-1)<1e-12);
            assert(fort_scope_plan_team_native_call_v1(context)==FORT_SCOPE_STATE);
            fort_scope_stats stats{};
            assert(!fort_scope_stats_get(context,&stats));
            assert(!stats.uploads && !stats.downloads && !stats.allocations && !stats.launches);
            assert(!fort_scope_close(context));
        }
    }
}
'''


def test_public_native_call_marker_invalidates_proof_and_preview_without_execution(tmp_path):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("C++ OpenMP compiler unavailable")
    source, binary = tmp_path / "native_calls.cpp", tmp_path / "native_calls"
    source.write_text(SOURCE)
    built = subprocess.run(
        [compiler, "-std=c++17", "-O2", "-fopenmp", "-DFORT_SCOPE_CPU_TEST", "-I", str(RUNTIME),
         str(source), "-x", "c++", str(RUNTIME / "scoped_runtime.cu"), "-o", str(binary)],
        capture_output=True, text=True, timeout=90,
    )
    assert built.returncode == 0, built.stdout + built.stderr
    run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=15,
                         env={**os.environ, "FORT_RUNTIME_TRACE": "1"})
    assert run.returncode == 0, run.stdout + run.stderr
    validations = [json.loads(line.split("FORT_SCOPED evidence ", 1)[1])
                   for line in run.stderr.splitlines() if line.startswith("FORT_SCOPED evidence ")]
    validations = [row for row in validations if row["event"] == "definition_validation"]
    assert [row["cache_hit"] for row in validations] == [False, True, False]
    assert validations[0]["query_generation"] == validations[1]["query_generation"]
    assert validations[2]["query_generation"] > validations[1]["query_generation"]
