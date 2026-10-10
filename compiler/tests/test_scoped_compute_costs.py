"""Versioned total costs separate original, coordinated CPU and GPU execution."""
from __future__ import annotations

import ctypes as c
import json
import math
from pathlib import Path
import shutil
import subprocess

import pytest

from compiler.tests.test_scoped_planning_runtime import (
    Binding, Scope, TOKEN, access, costs, planning, runtime, select,
)

NATIVE, CPU, GPU = 1, 2, 4


class Compute(c.Structure):
    _fields_ = [("version", c.c_uint32), ("known_backends", c.c_uint32),
                ("native_fortran_seconds", c.c_double), ("generated_cpu_seconds", c.c_double),
                ("gpu_seconds", c.c_double)]


@pytest.fixture
def compute_api(planning):
    planning.fort_scope_plan_add_compute_costs_v3.argtypes = [
        TOKEN, c.c_uint32, TOKEN, c.POINTER(Binding), c.c_size_t, c.c_double, c.c_double,
        c.c_int, c.POINTER(Compute)]
    return planning


def add(scope, values, *, kind=1, unit=101, bindings=None, count=0, gpu=1):
    return scope.lib.fort_scope_plan_add_compute_costs_v3(scope.handle, kind, unit, bindings, count,
                                                        100., 0., gpu, c.byref(values))


def consume(scope, expected):
    choice = c.c_int(-1)
    scope.check(scope.lib.fort_scope_plan_next(scope.handle, 101, None, 0, c.byref(choice)))
    assert choice.value == expected


@pytest.mark.parametrize("native,cpu,gpu,expected", [
    (10., 100., 1., 1), (1., 100., 2., 0), (10., .01, 1., 1),
])
def test_original_counterfactual_and_generated_worker_have_distinct_costs(compute_api, native, cpu, gpu, expected):
    scope = Scope(compute_api)
    scope.check(compute_api.fort_scope_plan_reset(scope.handle))
    values = Compute(1, NATIVE | CPU | GPU, native, cpu, gpu)
    scope.check(add(scope, values))
    # Evidence is copied into the query; caller storage is not retained.
    values.native_fortran_seconds = values.generated_cpu_seconds = 0
    values.gpu_seconds = 1000
    decision = select(scope, costs())
    assert decision.available and decision.gpu_units == expected
    assert decision.native_seconds == pytest.approx(native)
    assert decision.estimated_seconds == pytest.approx(gpu if expected else native, abs=1e-6)
    assert scope.stats().allocations == scope.stats().launches == 0
    consume(scope, expected)
    scope.close()


@pytest.mark.parametrize("missing", [NATIVE, CPU, GPU])
def test_missing_backend_estimate_preserves_definitions_but_declines_auto(compute_api, missing):
    scope = Scope(compute_api)
    scope.check(compute_api.fort_scope_plan_reset(scope.handle))
    scope.check(add(scope, Compute(1, 7 ^ missing, 10., 10., 1.)))
    scope.check(compute_api.fort_scope_plan_validate(scope.handle))
    decision = select(scope, costs())
    assert not decision.available and not decision.gpu_units
    assert scope.stats().allocations == scope.stats().upload_bytes == scope.stats().launches == 0
    scope.close()


@pytest.mark.parametrize("preparation,known,available,expected", [
    (0., NATIVE | CPU, True, 1), (20., NATIVE | CPU, True, 0),
    (0., NATIVE, False, 0),
])
def test_native_inspection_is_extra_candidate_work(compute_api, preparation, known, available, expected):
    scope = Scope(compute_api)
    scope.check(compute_api.fort_scope_plan_reset(scope.handle))
    scope.check(add(scope, Compute(1, 7, 10., 10., 1.)))
    scope.check(add(scope, Compute(1, known, 0., preparation, 0.), kind=0, unit=0, gpu=0))
    scope.check(compute_api.fort_scope_plan_validate(scope.handle))
    decision = select(scope, costs())
    assert bool(decision.available) == available
    assert decision.gpu_units == expected
    assert scope.stats().allocations == scope.stats().launches == 0
    consume(scope, expected)
    scope.close()


@pytest.mark.parametrize("field,bad", [
    ("version", 0), ("version", 2), ("known_backends", 8),
    *[(field, bad) for field in ("native_fortran_seconds", "generated_cpu_seconds", "gpu_seconds")
      for bad in (-1., math.inf, math.nan)],
])
def test_invalid_total_payload_is_rejected_before_recording(compute_api, field, bad):
    scope = Scope(compute_api)
    scope.check(compute_api.fort_scope_plan_reset(scope.handle))
    values = Compute(1, 7, 10., 10., 1.)
    setattr(values, field, bad)
    assert add(scope, values) != 0
    scope.check(add(scope, Compute(1, 7, 10., 10., 1.)))
    scope.check(compute_api.fort_scope_plan_validate(scope.handle))
    decision = select(scope, costs())
    assert decision.available and decision.gpu_units == 1 and decision.cpu_units == 0
    assert scope.stats().allocations == scope.stats().launches == 0
    consume(scope, 1)
    scope.close()


def test_missing_continuation_cost_does_not_replay_gpu_prefix(compute_api):
    compute_api.fort_scope_plan_reset_mode.argtypes = [TOKEN, c.c_uint32]
    scope = Scope(compute_api)
    host = (c.c_double * 8)(*range(8))
    status, handle = scope.register(host, [8])
    scope.check(status)
    effects = access(flags=3)
    device = scope.gpu(handle, effects)
    for index in range(8):
        device[index] += 10
    scope.gpu_end(handle)
    scope.check(compute_api.fort_scope_wait(scope.handle))
    scope.check(compute_api.fort_scope_plan_reset_mode(scope.handle, 1))
    bindings = (Binding * 1)(Binding(handle, effects))
    scope.check(add(scope, Compute(1, NATIVE | GPU, .01, 0., .001), bindings=bindings, count=1))
    scope.check(compute_api.fort_scope_plan_validate(scope.handle))
    decision = select(scope, costs())
    assert not decision.available and not decision.gpu_units
    choice = c.c_int(-1)
    scope.check(compute_api.fort_scope_plan_next(scope.handle, 101, bindings, 1, c.byref(choice)))
    assert choice.value == 0
    scope.cpu_begin(handle, effects)
    for index in range(8):
        host[index] += 1
    scope.cpu_end(handle)
    scope.close()
    assert list(host) == [index + 11 for index in range(8)]


DRIVER = r'''
#include "scoped_planning.hpp"
#include "scoped_entry.hpp"
#include "view_entry.hpp"
#include <iomanip>
#include <iostream>
using namespace fort_scoped::planning;
fort_scope_plan_costs costs() {
    fort_scope_plan_costs c{}; c.version=1; c.valid=1; c.max_allocation_bytes=1<<20;
    c.cpu_flops=c.cpu_bandwidth=c.gpu_flops=c.gpu_bandwidth=c.h2d_bandwidth=c.d2h_bandwidth=1e12;
    c.create_seconds=c.register_seconds=c.host_access_seconds=c.device_access_seconds=1e-9;
    c.gpu_setup_seconds=c.cold_driver_startup_seconds=c.allocation_seconds=c.release_seconds=1e-9;
    c.wait_seconds=c.launch_enqueue_seconds=c.planning_operation_seconds=1e-9;
    return c;
}
Operation op(uint32_t kind, uint32_t known, double native, double cpu, double gpu) {
    Operation value{kind,kind==FORT_SCOPE_PLAN_WORKER?1ULL:0ULL,{},100,0,kind==FORT_SCOPE_PLAN_WORKER};
    value.compute_costs=fort_scope_compute_costs_v1{1,known,native,cpu,gpu}; return value;
}
int main(int argc,char **argv) {
    if(argc!=2) return 2;
    const std::string scenario=argv[1]; Inputs in; auto c=costs();
    in.operations={op(FORT_SCOPE_PLAN_WORKER,7,10,20,1)};
    if(scenario=="direct") {
        auto &worker=in.operations[0]; worker.cpu_numerical_seconds=worker.gpu_numerical_seconds=100;
        worker.flops=1e18; worker.memory_bytes=1e18;
        auto native=op(FORT_SCOPE_PLAN_NATIVE,3,2,5,0);
        uint64_t work=0; auto state=detail::initial(in,c); const double before=state.time;
        detail::execute(state,native,false,in,c,work);
        std::cout<<std::setprecision(17)<<"{\"original\":"<<detail::native_compute(worker,in,c)
                 <<",\"cpu\":"<<detail::compute(worker,c,false)<<",\"gpu\":"<<detail::compute(worker,c,true)
                 <<",\"coordinated_native\":"<<state.time-before<<"}\n"; return 0;
    }
    if(scenario=="unknown_preparation") in.operations.push_back(op(FORT_SCOPE_PLAN_NATIVE,1,0,0,0));
    else if(scenario=="known_preparation") in.operations.push_back(op(FORT_SCOPE_PLAN_NATIVE,3,0,20,0));
    else if(scenario=="continuation" || scenario=="continuation_native" || scenario=="continuation_margin") {
        in.continuation=true; in.charge_create=false; in.registrations_incurred=0;
        in.operations[0].compute_costs=fort_scope_compute_costs_v1{1,7,.01,100,1};
        c.gpu_setup_seconds=1; c.launch_enqueue_seconds=.1;
        if(scenario=="continuation_native") in.operations[0].gpu_available=false;
        if(scenario=="continuation_margin") in.operations[0].compute_costs->gpu_seconds=85;
    } else if(scenario=="team") {
        fort_scope_team_costs t{}; t.version=1;t.valid=1;t.cpu_threads=4;t.expected_omp_level=1;
        t.protocol_id=FORT_SCOPE_TEAM_PROTOCOL_ID;t.native_cpu_flops=t.native_cpu_bandwidth=1;
        t.native_worker_seconds=3;t.cpu_worker_seconds=4;t.gpu_worker_seconds=5; in.team_costs=t;
        const auto &worker=in.operations[0]; auto state=detail::initial(in,c); uint64_t work=0;
        const double before=state.time; detail::execute(state,worker,false,in,c,work);
        std::cout<<std::setprecision(17)<<"{\"original\":"<<detail::native_compute(worker,in,c)
                 <<",\"cpu\":"<<state.time-before<<"}\n"; return 0;
    } else if(scenario=="legacy") {
        in.operations[0].compute_costs.reset(); in.operations[0].cpu_numerical_seconds=3;
        in.operations[0].gpu_numerical_seconds=1;
    } else return 3;
    const auto proof=validate_definitions(in); const auto result=select(in,c);
    double required=0;
    evidence(in,c,result,EvidenceSink{&required,[](void *data,const EvidenceEvent &event) {
        if(std::string(event.event)=="gate") *static_cast<double *>(data)=event.required_saving;
    }});
    std::cout<<std::setprecision(17)<<"{\"available\":"<<result.decision.available
             <<",\"gpu\":"<<result.decision.gpu_units<<",\"reason\":\""<<result.reason
             <<"\",\"execution\":"<<result.report.execution_seconds
             <<",\"native_execution\":"<<result.report.native_execution_seconds
             <<",\"native\":"<<result.decision.native_seconds
             <<",\"required_saving\":"<<required
             <<",\"common_excluded\":"<<result.native_common_compute_excluded
             <<",\"definitions\":"<<proof.status<<"}\n";
}
'''


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("g++ unavailable")
    directory = tmp_path_factory.mktemp("backend_compute_model")
    source, binary = directory / "driver.cpp", directory / "driver"
    source.write_text(DRIVER)
    runtime_dir = Path(__file__).resolve().parents[1] / "runtime"
    result = subprocess.run([compiler, "-std=c++17", "-O0", "-Wall", "-Wextra", "-Werror",
                             "-I", str(runtime_dir), str(source), "-o", str(binary)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr

    def run(scenario):
        result = subprocess.run([str(binary), scenario], capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)
    return run


def test_total_costs_are_used_once_and_do_not_add_legacy_roofline(model):
    assert model("direct") == {"original": 10, "cpu": 20, "gpu": 1, "coordinated_native": 5}
    assert model("team") == {"original": 13, "cpu": 24}


def test_unknown_native_preparation_is_not_common_excluded(model):
    unknown = model("unknown_preparation")
    assert unknown["definitions"] == 0
    assert not unknown["available"] and not unknown["gpu"] and not unknown["common_excluded"]
    assert unknown["reason"] == "coordinated_host_compute_unavailable"
    known = model("known_preparation")
    assert known["available"] and not known["gpu"] and not known["common_excluded"]
    assert known["native"] == 10 and known["native_execution"] == pytest.approx(10, abs=1e-6)


def test_continuation_uses_coordinated_cpu_for_startup_and_native_baseline(model):
    selected = model("continuation")
    assert selected["available"] and selected["gpu"]
    assert selected["required_saving"] == pytest.approx(20)
    native = model("continuation_native")
    assert native["available"] and not native["gpu"]
    assert native["execution"] == pytest.approx(100, abs=1e-6)
    assert native["native_execution"] == pytest.approx(100, abs=1e-6)
    borderline = model("continuation_margin")
    assert borderline["available"] and not borderline["gpu"]
    assert borderline["reason"] == "native_20_percent_margin_not_met"


def test_legacy_additional_cost_contract_is_unchanged(model):
    result = model("legacy")
    assert result["available"] and result["gpu"]
    assert result["native"] == pytest.approx(3, abs=1e-6)


def test_cpp_numerical_and_view_entry_helpers_accept_total_costs(tmp_path):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("g++ unavailable")
    source = tmp_path / "entries.cpp"
    source.write_text('''#include "view_entry.hpp"
int record(fort_scope_t context,const fort_scope_compute_costs_v1 &costs) {
    fort_scoped::AccessBatch<2> full(context);
    fort_scoped::ViewAccessBatch<2> views(context);
    return full.record_compute(FORT_SCOPE_PLAN_WORKER,1,1,1,1,costs) +
        views.record_compute(FORT_SCOPE_PLAN_WORKER,2,1,1,1,costs);
}
''')
    runtime_dir = Path(__file__).resolve().parents[1] / "runtime"
    result = subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-fsyntax-only",
                             "-I", str(runtime_dir), str(source)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_fortran_compute_payload_and_v3_call_match_c_abi(tmp_path):
    fortran = shutil.which("gfortran") or shutil.which("gfortran-15")
    cc = shutil.which("cc")
    if not fortran or not cc:
        pytest.skip("native Fortran/C toolchain unavailable")
    runtime_dir = Path(__file__).resolve().parents[1] / "runtime"
    stub = tmp_path / "stub.c"
    stub.write_text('''#include "scoped_runtime.h"
int fort_scope_plan_add_compute_costs_v3(fort_scope_t context,uint32_t kind,uint64_t unit,
    const fort_scope_plan_binding *bindings,size_t count,double flops,double memory,int gpu,
    const fort_scope_compute_costs_v1 *costs) {
    return context==7 && kind==FORT_SCOPE_PLAN_WORKER && unit==9 && !bindings && count==0 &&
        flops==3 && memory==4 && gpu==1 && costs && costs->version==FORT_SCOPE_COMPUTE_ABI_VERSION &&
        costs->known_backends==FORT_SCOPE_COMPUTE_ALL && costs->native_fortran_seconds==11 &&
        costs->generated_cpu_seconds==12 && costs->gpu_seconds==13 ? 42 : 1;
}
''')
    source = tmp_path / "caller.f90"
    source.write_text('''program caller
use iso_c_binding
use fort_scoped_memory
implicit none
type(fort_scope_compute_costs_v1) :: costs
integer(c_int) :: status
costs=fort_scope_compute_costs_v1(FORT_SCOPE_COMPUTE_ABI_VERSION,FORT_SCOPE_COMPUTE_ALL, &
    11.0_c_double,12.0_c_double,13.0_c_double)
if(c_sizeof(costs)/=32_c_size_t) stop 1
status=fort_scope_plan_add_compute_costs_v3(7_c_int64_t,FORT_SCOPE_PLAN_WORKER,9_c_int64_t, &
    c_null_ptr,0_c_size_t,3.0_c_double,4.0_c_double,1_c_int,costs)
if(status/=42) stop 2
end program
''')
    result = subprocess.run([cc, "-std=c11", "-I", str(runtime_dir), "-c", str(stub),
                             "-o", str(tmp_path / "stub.o")],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([fortran, "-std=f2018", "-J", str(tmp_path),
                             str(runtime_dir / "scoped_memory.f90"), str(source),
                             str(tmp_path / "stub.o"), "-o", str(tmp_path / "caller")],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(tmp_path / "caller")], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
