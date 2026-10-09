"""Persistent-team placement prices original and generated work separately."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest

RUNTIME = Path(__file__).resolve().parents[1] / "runtime"

DRIVER = r'''
#include "scoped_planning.hpp"
#include <iomanip>
#include <iostream>
using namespace fort_scoped::planning;

fort_scope_plan_costs costs() {
    fort_scope_plan_costs c{};
    c.version=FORT_SCOPE_PLANNING_ABI_VERSION; c.valid=1;
    c.max_allocation_bytes=1<<20;
    c.cpu_flops=100; c.cpu_bandwidth=1000;
    c.gpu_flops=1000; c.gpu_bandwidth=10000;
    c.h2d_bandwidth=c.d2h_bandwidth=1000;
    c.create_seconds=2; c.register_seconds=3;
    c.host_access_seconds=5; c.device_access_seconds=7;
    c.gpu_setup_seconds=11; c.cold_driver_startup_seconds=13;
    c.allocation_seconds=17; c.release_seconds=19;
    c.wait_seconds=23; c.launch_enqueue_seconds=29;
    c.planning_operation_seconds=1e-12;
    return c;
}
fort_scope_team_costs team() {
    fort_scope_team_costs t{};
    t.version=FORT_SCOPE_TEAM_ABI_VERSION; t.valid=1;
    t.cpu_threads=4; t.expected_omp_level=1;
    t.protocol_id=FORT_SCOPE_TEAM_PROTOCOL_ID;
    t.native_cpu_flops=1000; t.native_cpu_bandwidth=10000;
    t.owner_seconds=31; t.descriptor_seconds=37; t.entry_seconds=41;
    t.cpu_worker_seconds=43; t.gpu_worker_seconds=47;
    t.native_call_seconds=53; t.native_worker_seconds=59;
    return t;
}
Resource resource(fort_buffer_t handle) {
    Resource b; b.handle=handle; b.element_bytes=8; b.bytes=32;
    b.extents={4}; b.initialized=b.host_current={Box{{0},{4}}};
    return b;
}
Operation entry() { return {FORT_SCOPE_PLAN_TEAM_ENTRY,0,{},0,0,false}; }
Operation native_call() { return {FORT_SCOPE_PLAN_TEAM_NATIVE_CALL,0,{},0,0,false}; }
Operation worker(uint64_t id, double flops, double memory=0) {
    return {FORT_SCOPE_PLAN_WORKER,id,{},flops,memory,true};
}

int main(int argc,char **argv) {
    if (argc!=2) return 2;
    const std::string scenario=argv[1]; auto c=costs(); auto t=team(); Inputs in;
    in.team_costs=t;
    in.operations={entry(),worker(1,200),entry(),worker(2,400)};
    in.resources={resource(1),resource(2)};
    std::cout << std::setprecision(17);
    if (scenario=="accounting") {
        uint64_t work=0; auto s=detail::initial(in,c);
        const double initial=s.time;
        detail::execute(s,in.operations[0],false,in,c,work);
        const double first_entry=s.time-initial;
        detail::execute(s,in.operations[1],false,in,c,work);
        const double cpu_worker=s.time-initial-first_entry;
        const double previous=s.time;
        detail::execute(s,in.operations[2],false,in,c,work);
        const double second_entry=s.time-previous;
        const double before_gpu=s.time;
        detail::execute(s,in.operations[3],true,in,c,work);
        const double gpu_worker=s.time-before_gpu;
        const double before_native=s.time;
        const Operation native{FORT_SCOPE_PLAN_NATIVE,0,{},100,0,false};
        detail::execute(s,native,false,in,c,work);
        const double native_compute=s.time-before_native;
        const double before_call=s.time;
        detail::execute(s,native_call(),false,in,c,work);
        const double native_protocol=s.time-before_call;
        const double before_preparation=s.time;
        detail::execute(s,{FORT_SCOPE_PLAN_NATIVE,0,{},0,0,false},false,in,c,work);
        const double preparation=s.time-before_preparation;
        auto metadata=detail::initial(in,c); const double before_metadata=metadata.time;
        detail::execute(metadata,{FORT_SCOPE_PLAN_NATIVE,0,{{1,{{Box{{0},{4}}},{},{}}}},0,0,false},
                        false,in,c,work);
        std::cout << "{\"initial\":" << initial << ",\"entry\":" << first_entry
                  << ",\"second_entry\":" << second_entry << ",\"cpu_worker\":" << cpu_worker
                  << ",\"gpu_worker\":" << gpu_worker << ",\"native_call\":" << native_protocol
                  << ",\"native_compute\":" << native_compute << ",\"preparation\":" << preparation
                  << ",\"metadata\":" << metadata.time-before_metadata
                  << ",\"native_worker\":" << detail::native_compute(in.operations[1],in,c)
                  << ",\"native_entry\":" << detail::native_compute(entry(),in,c)
                  << ",\"native_marker\":" << detail::native_compute(native_call(),in,c)
                  << ",\"launches\":" << s.count.launches << ",\"operations\":" << work << "}\n";
        return 0;
    }
    if (scenario=="original_native" || scenario=="startup_original_native") {
        // The whole-native fallback bypasses generated entry/worker protocols
        // and per-array hooks, even when generated CPU throughput is slower.
        c.cpu_flops=1; t.native_worker_seconds=0;
        if (scenario=="startup_original_native") t.native_cpu_flops=1e9;
        else c.gpu_setup_seconds=c.cold_driver_startup_seconds=c.launch_enqueue_seconds=1e-6;
        in.team_costs=t;
    } else if (scenario=="below_margin" || scenario=="above_margin" ||
               scenario=="preparation_only" || scenario=="native_protocol") {
        in.resources.clear(); in.operations={entry(),worker(1,1000)};
        for (double *v : {&c.create_seconds,&c.register_seconds,&c.host_access_seconds,&c.device_access_seconds,
                          &c.gpu_setup_seconds,&c.cold_driver_startup_seconds,&c.allocation_seconds,
                          &c.release_seconds,&c.wait_seconds,&c.launch_enqueue_seconds}) *v=1e-12;
        t.owner_seconds=t.descriptor_seconds=t.entry_seconds=t.cpu_worker_seconds=
            t.gpu_worker_seconds=t.native_call_seconds=t.native_worker_seconds=0;
        t.native_cpu_flops=1000; in.team_costs=t;
        c.gpu_flops=1000/(scenario=="below_margin" ? .799 : .801);
        if (scenario=="preparation_only" || scenario=="native_protocol") {
            c.gpu_flops=1000/.6; in.team_costs->native_call_seconds=.5;
            in.operations.insert(in.operations.begin()+1,{FORT_SCOPE_PLAN_NATIVE,0,{},0,0,false});
            if (scenario=="native_protocol") in.operations.insert(in.operations.begin()+1,native_call());
        }
    } else if (scenario=="whole_owner_15_percent" || scenario=="whole_owner_21_percent" ||
               scenario=="whole_owner_known_native") {
        in.resources.clear();
        in.operations={entry(),worker(1,500),native_call(),
            {FORT_SCOPE_PLAN_NATIVE,0,{},scenario=="whole_owner_known_native" ? 500.0 : 0.0,0,false},
            entry(),worker(2,500)};
        for (double *v : {&c.create_seconds,&c.register_seconds,&c.host_access_seconds,&c.device_access_seconds,
                          &c.gpu_setup_seconds,&c.cold_driver_startup_seconds,&c.allocation_seconds,
                          &c.release_seconds,&c.wait_seconds,&c.launch_enqueue_seconds}) *v=1e-12;
        t.owner_seconds=t.descriptor_seconds=t.entry_seconds=t.cpu_worker_seconds=
            t.gpu_worker_seconds=t.native_call_seconds=t.native_worker_seconds=0;
        t.native_cpu_flops=1000; in.team_costs=t;
        c.cpu_flops=10; // Local counterfactuals use much slower generated CPU workers.
        c.gpu_flops=1000/(scenario=="whole_owner_15_percent" ? .85 : .79);
        c.planning_operation_seconds=1e-5;
    } else if (scenario=="memory_roof") {
        in.operations={entry(),worker(1,200,40000)};
        c.cpu_flops=1; t.native_worker_seconds=0; in.team_costs=t;
    } else if (scenario=="missing_team") {
        in.team_costs.reset();
    } else if (scenario=="wrong_protocol") {
        ++in.team_costs->protocol_id;
    } else if (scenario=="continuation") {
        in.continuation=true;
    } else if (scenario=="marker_binding") {
        in.operations[0].bindings={{1,{}}};
    } else if (scenario=="native_marker_missing_team") {
        in.team_costs.reset(); in.operations[0]=native_call();
    } else if (scenario.compare(0,14,"native_marker_")==0) {
        in.operations[0]=native_call();
        if (scenario=="native_marker_binding") in.operations[0].bindings={{1,{}}};
        else if (scenario=="native_marker_work") in.operations[0].flops=1;
        else if (scenario=="native_marker_memory") in.operations[0].memory_bytes=1;
        else if (scenario=="native_marker_unit") in.operations[0].unit=1;
        else if (scenario=="native_marker_gpu") in.operations[0].gpu_available=true;
        else return 3;
    } else if (scenario=="undefined") {
        in.resources[0].initialized.clear(); in.resources[0].host_current.clear();
        in.operations[1].bindings={{1,{{Box{{0},{4}}},{},{}}}};
        in.operations[1].flops=1e-9;
        in.operations[3].flops=1e-9;
        in.team_costs->native_worker_seconds=0;
    } else if (scenario=="definition_events") {
        const Region full={Box{{0},{4}}};
        in.operations={entry(), native_call(), {FORT_SCOPE_PLAN_FORGET,0,{{1,{}}},0,0,false},
            {FORT_SCOPE_PLAN_WORKER,1,{{1,{{},full,full}}},200,0,true},entry(),
            {FORT_SCOPE_PLAN_WORKER,2,{{1,{full,{},{}}}},400,0,true}};
    } else return 3;
    const auto proof=validate_definitions(in);
    const auto r=select(in,c);
    std::vector<EvidenceEvent> events;
    evidence(in,c,r,{&events,[](void *context,const EvidenceEvent &event) {
        static_cast<std::vector<EvidenceEvent> *>(context)->push_back(event);
    }});
    size_t whole_gates=0,local_gates=0;
    double whole_seconds=0,whole_native=0,whole_saving=0;
    bool whole_accepted=false;
    for (const auto &event : events) if (std::string(event.event)=="gate") {
        if (event.phase && std::string(event.phase)=="whole_owner") {
            ++whole_gates; whole_seconds=event.seconds; whole_native=event.counterfactual_seconds;
            whole_saving=event.required_saving; whole_accepted=event.accepted;
        } else ++local_gates;
    }
    double all_gpu_without_planning=0;
    if (scenario.compare(0,12,"whole_owner_")==0) {
        uint64_t ignored=0;
        all_gpu_without_planning=detail::simulate(in,c,{true,true},ignored).time;
    }
    std::cout << "{\"available\":" << r.decision.available << ",\"gpu_units\":" << r.decision.gpu_units
              << ",\"reason\":\"" << r.reason << "\",\"native_seconds\":" << r.decision.native_seconds
              << ",\"seconds\":" << r.decision.estimated_seconds
              << ",\"startup_shortcut\":" << r.native_startup_shortcut
              << ",\"common_native_excluded\":" << r.native_common_compute_excluded
              << ",\"planning_seconds\":" << r.decision.simulated_operations*c.planning_operation_seconds
              << ",\"all_gpu_without_planning\":" << all_gpu_without_planning
              << ",\"whole_gates\":" << whole_gates << ",\"local_gates\":" << local_gates
              << ",\"whole_seconds\":" << whole_seconds << ",\"whole_native\":" << whole_native
              << ",\"whole_saving\":" << whole_saving << ",\"whole_accepted\":" << whole_accepted
              << ",\"proof_status\":" << proof.status << ",\"proof_operation\":" << proof.operation
              << ",\"proof_reason\":\"" << proof.reason << "\"}\n";
}
'''


@pytest.fixture(scope="module")
def team_model(tmp_path_factory):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("g++ unavailable")
    directory = tmp_path_factory.mktemp("scoped_team_planning")
    source = directory / "driver.cpp"
    binary = directory / "driver"
    source.write_text(DRIVER)
    built = subprocess.run(
        [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-I", str(RUNTIME),
         str(source), "-o", str(binary)], capture_output=True, text=True, timeout=60,
    )
    assert built.returncode == 0, built.stdout + built.stderr

    def run(scenario):
        result = subprocess.run([str(binary), scenario], capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stdout + result.stderr
        return json.loads(result.stdout)

    return run


def test_protocol_costs_are_exclusive_and_charged_at_their_execution_points(team_model):
    result = team_model("accounting")
    assert result["initial"] == 2 + 2 * 3 + 31 + 2 * 37
    assert result["entry"] == result["second_entry"] == 41
    assert result["cpu_worker"] == 200 / 100 + 43
    assert result["gpu_worker"] == pytest.approx(11 + 13 + 400 / 1000 + 47 + 29)
    assert result["native_call"] == 53
    assert result["native_compute"] == pytest.approx(100 / 1000)
    assert result["preparation"] == 0
    assert result["metadata"] == 5
    assert result["native_worker"] == 200 / 1000 + 59
    assert result["native_entry"] == 0
    assert result["native_marker"] == 0
    assert result["launches"] == 1
    assert result["operations"] == 8


@pytest.mark.parametrize("scenario", ["original_native", "startup_original_native"])
def test_whole_native_uses_original_rates_without_generated_protocols(team_model, scenario):
    result = team_model(scenario)
    rate = 1e9 if scenario.startswith("startup") else 1000
    native = 600 / rate
    assert result["available"] == 1
    assert result["gpu_units"] == 0
    assert result["native_seconds"] == pytest.approx(native)
    assert result["seconds"] == pytest.approx(2 + 2 * 3 + 31 + 2 * 37 + native)
    assert result["startup_shortcut"] == scenario.startswith("startup")


@pytest.mark.parametrize("scenario,expected_gpu", [("below_margin", 1), ("above_margin", 0)])
def test_twenty_percent_gate_uses_original_native_rate(team_model, scenario, expected_gpu):
    result = team_model(scenario)
    assert result["available"] == 1
    assert result["native_seconds"] == 1
    assert result["gpu_units"] == expected_gpu
    if not expected_gpu:
        assert result["reason"] == "native_20_percent_margin_not_met"


@pytest.mark.parametrize("scenario,expected_gpu", [("whole_owner_15_percent", 0), ("whole_owner_21_percent", 2)])
def test_multiple_entry_intervals_need_twenty_percent_whole_original_native_saving(team_model, scenario, expected_gpu):
    result = team_model(scenario)
    assert result["available"] == 1
    assert result["native_seconds"] == 1
    assert result["common_native_excluded"]
    assert result["gpu_units"] == expected_gpu
    assert result["whole_gates"] == bool(expected_gpu)
    if expected_gpu:
        assert result["local_gates"] == 2
        assert result["whole_accepted"]
        assert result["whole_seconds"] == result["seconds"]
        assert result["whole_native"] == 1
        assert result["whole_saving"] == pytest.approx(.2)
        assert result["seconds"] == pytest.approx(result["all_gpu_without_planning"] + result["planning_seconds"])
    else:
        assert result["reason"] == "native_20_percent_margin_not_met"
        assert result["seconds"] == pytest.approx(1 + result["planning_seconds"])


def test_whole_owner_margin_includes_known_common_native_computation(team_model):
    result = team_model("whole_owner_known_native")
    assert result["available"] == 1
    assert result["native_seconds"] == 1.5
    assert not result["common_native_excluded"]
    assert result["gpu_units"] == result["whole_gates"] == 0
    assert result["reason"] == "native_20_percent_margin_not_met"


def test_original_native_roofline_uses_the_slower_memory_bound(team_model):
    result = team_model("memory_roof")
    assert result["available"] == 1
    assert result["native_seconds"] == max(200 / 1000, 40000 / 10000)
    assert result["gpu_units"] == 0


@pytest.mark.parametrize("scenario,expected_gpu", [("preparation_only", 1), ("native_protocol", 0)])
def test_native_call_protocol_changes_placement_only_at_original_calls(team_model, scenario, expected_gpu):
    result = team_model(scenario)
    assert result["available"] == 1
    assert result["native_seconds"] == 1
    assert result["gpu_units"] == expected_gpu


@pytest.mark.parametrize("scenario", ["missing_team", "wrong_protocol", "continuation", "native_marker_missing_team"])
def test_unavailable_collective_costs_preserve_placement_independent_definition_proof(team_model, scenario):
    result = team_model(scenario)
    assert result["proof_status"] == 0
    assert result["available"] == result["gpu_units"] == 0
    assert result["reason"] == "collective_synchronization_calibration_unavailable"


def test_collective_marker_cannot_hide_effects(team_model):
    result = team_model("marker_binding")
    assert result["proof_status"] != 0
    assert result["proof_reason"] == result["reason"] == "invalid_collective_entry_marker"
    assert result["available"] == result["gpu_units"] == 0


@pytest.mark.parametrize("field", ["binding", "work", "memory", "unit", "gpu"])
def test_native_call_marker_cannot_hide_effects_work_or_choices(team_model, field):
    result = team_model("native_marker_" + field)
    assert result["proof_status"] != 0
    assert result["proof_reason"] == result["reason"] == "invalid_collective_native_call_marker"
    assert result["available"] == result["gpu_units"] == 0


def test_native_startup_shortcut_still_rejects_undefined_read_after_entry_marker(team_model):
    result = team_model("undefined")
    assert result["proof_status"] != 0
    assert result["proof_operation"] == 1
    assert result["proof_reason"] == result["reason"] == "uninitialized_read"
    assert result["available"] == result["gpu_units"] == 0


def test_markers_do_not_change_ordered_nested_definition_events(team_model):
    result = team_model("definition_events")
    assert result["proof_status"] == 0
    assert result["available"] == 1
