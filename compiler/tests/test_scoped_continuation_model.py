"""Continuation cost ranks include publication liability without executing it."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

RUNTIME = Path(__file__).resolve().parents[1] / "runtime"
DRIVER = r'''
#include "scoped_planning.hpp"
#include <iomanip>
#include <iostream>
#include <string>
using namespace fort_scoped::planning;
fort_scope_plan_costs costs() {
    fort_scope_plan_costs c{};
    c.version=1; c.valid=1; c.max_allocation_bytes=1<<20;
    c.cpu_flops=1; c.cpu_bandwidth=1e12; c.gpu_flops=1000; c.gpu_bandwidth=1e12;
    c.h2d_bandwidth=c.d2h_bandwidth=1e12; c.h2d_latency=1e-6; c.d2h_latency=.5;
    c.create_seconds=.02; c.register_seconds=.01;
    c.host_access_seconds=c.device_access_seconds=c.gpu_setup_seconds=1e-6;
    c.cold_driver_startup_seconds=c.allocation_seconds=c.release_seconds=c.wait_seconds=1e-6;
    c.launch_enqueue_seconds=1e-6; c.planning_operation_seconds=1e-12;
    return c;
}
Resource resource(fort_buffer_t handle) {
    Resource b; b.handle=handle; b.element_bytes=8; b.bytes=64; b.extents={8};
    b.initialized=b.device_current={Box{{0},{8}}}; b.allocated=true; return b;
}
int main(int argc,char **argv) {
    if(argc!=2) return 2;
    auto c=costs(); Inputs in; in.continuation=true; in.charge_create=false;
    in.registrations_incurred=0; in.device_ready=in.driver_initialized=true;
    in.resources={resource(1)};
    Region full=in.resources[0].initialized;
    in.operations={{FORT_SCOPE_PLAN_WORKER,1,{{1,{full,full,{}}}},1,0,true}};
    const std::string scenario=argv[1];
    if(scenario=="overwrite_credit") {
        in.operations[0].bindings[0].effects={{},full,full};
        in.operations[0].flops=.001; in.operations[0].gpu_available=false;
    } else if(scenario=="read_credit") {
        in.operations[0].bindings[0].effects={full,{},{}};
        in.operations[0].gpu_available=false;
    } else if(scenario=="output_liability") {
        in.device_ready=false; in.resources[0].allocated=false;
        in.resources[0].initialized.clear(); in.resources[0].device_current.clear();
        in.operations[0].bindings[0].effects={{},full,full}; c.d2h_latency=2;
    } else if(scenario=="margin_below" || scenario=="margin_above") {
        c.gpu_flops=1/(scenario=="margin_below" ? .81 : .79);
    } else if(scenario=="lifecycle_first" || scenario=="lifecycle_later" || scenario=="lifecycle_new") {
        in.operations[0].gpu_available=false;
        if(scenario=="lifecycle_first") { in.charge_create=true; in.registrations_incurred=1; }
        if(scenario=="lifecycle_new") in.registrations_incurred=1;
    } else if(scenario=="unknown_work") {
        in.operations[0].flops=std::numeric_limits<double>::infinity();
    } else if(scenario=="common_native") {
        in.operations.insert(in.operations.begin(),Operation{});
    } else if(scenario=="expensive_native_hooks") {
        in.device_ready=false; in.resources[0].allocated=false;
        in.resources[0].host_current=full; in.resources[0].device_current.clear();
        in.operations[0].flops=.1; c.host_access_seconds=1;
        c.gpu_setup_seconds=.2; c.launch_enqueue_seconds=.01; c.d2h_latency=.01;
    } else return 3;
    const auto r=select(in,c); const auto &p=r.report;
    std::cout<<std::setprecision(17)<<"{\"available\":"<<r.decision.available
      <<",\"gpu\":"<<r.decision.gpu_units<<",\"reason\":\""<<r.reason
      <<"\",\"execution\":"<<p.execution_seconds<<",\"native_execution\":"<<p.native_execution_seconds
      <<",\"entry_terminal\":"<<p.entry_terminal.seconds<<",\"terminal\":"<<p.terminal.seconds
      <<",\"native_terminal\":"<<p.native_terminal.seconds<<",\"ranking\":"<<p.ranking_seconds
      <<",\"native_ranking\":"<<p.native_ranking_seconds
      <<",\"prefix_downloads\":"<<r.decision.download_bytes
      <<",\"terminal_downloads\":"<<p.terminal.download_bytes
      <<",\"entry_downloads\":"<<p.entry_terminal.download_bytes
      <<",\"common_native\":"<<p.native_common_compute_excluded<<"}\n";
}
'''


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("g++ unavailable")
    directory = tmp_path_factory.mktemp("continuation_costs")
    source, executable = directory / "driver.cpp", directory / "driver"
    source.write_text(DRIVER)
    built = subprocess.run([compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                            "-I", str(RUNTIME), str(source), "-o", str(executable)],
                           capture_output=True, text=True, timeout=60)
    assert built.returncode == 0, built.stderr

    def run(scenario):
        result = subprocess.run([str(executable), scenario], capture_output=True, text=True, timeout=20)
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)
    return run


def test_native_overwrite_retires_dirty_publication_with_signed_credit(model):
    result = model("overwrite_credit")
    assert result["available"]
    assert not result["gpu"]
    assert result["entry_downloads"] == 64
    assert result["prefix_downloads"] == result["terminal_downloads"] == 0
    assert result["ranking"] < 0
    assert result["ranking"] == pytest.approx(result["execution"] + result["terminal"] - result["entry_terminal"])


def test_native_read_moves_publication_into_execution_only_once(model):
    result = model("read_credit")
    assert result["available"]
    assert not result["gpu"]
    assert result["prefix_downloads"] == result["entry_downloads"] == 64
    assert result["terminal_downloads"] == 0
    assert result["execution"] > 1.5
    assert result["ranking"] == pytest.approx(1, abs=5e-6)


def test_new_gpu_output_publication_liability_can_choose_cpu(model):
    result = model("output_liability")
    assert result["available"]
    assert not result["gpu"]
    assert result["execution"] > 1
    assert result["entry_terminal"] == result["terminal"] == 0


@pytest.mark.parametrize(("scenario", "gpu"), [("margin_below", 0), ("margin_above", 1)])
def test_margin_compares_same_dirty_cpu_continuation(model, scenario, gpu):
    result = model(scenario)
    assert result["available"]
    assert result["gpu"] == gpu
    assert result["entry_terminal"] > .5
    assert result["native_execution"] > 1.5
    if gpu:
        assert result["prefix_downloads"] == 0
        assert result["terminal_downloads"] == 64
        assert result["ranking"] <= result["native_ranking"] - .2


def test_lifecycle_cost_counts_only_incurred_create_and_registrations(model):
    first = model("lifecycle_first")["execution"]
    later = model("lifecycle_later")["execution"]
    new = model("lifecycle_new")["execution"]
    assert first-later == pytest.approx(.03)
    assert new-later == pytest.approx(.01)


def test_unknown_work_returns_no_estimate_and_common_native_cost_stays_explicit(model):
    unknown = model("unknown_work")
    assert not unknown["available"]
    assert unknown["reason"] == "unknown_or_overflowed_work"
    assert model("common_native")["common_native"]


def test_fresh_startup_bound_includes_mandatory_native_hooks(model):
    result = model("expensive_native_hooks")
    assert result["available"]
    assert result["gpu"] == 1
    assert result["native_execution"] > 1.1
    assert result["execution"] + result["terminal"] < .23
