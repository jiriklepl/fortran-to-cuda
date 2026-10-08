"""Complete-schedule costs preserve the runtime's physical coherence model."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

RUNTIME = Path(__file__).resolve().parents[1] / "runtime"

DRIVER = r'''
#include "scoped_planning.hpp"
#include <cstdlib>
#include <iomanip>
#include <iostream>

namespace {
size_t heap_live=0, heap_peak=0;
struct alignas(std::max_align_t) Allocation { size_t bytes; };
}
void *operator new(size_t bytes) {
    if (bytes > std::numeric_limits<size_t>::max()-sizeof(Allocation)) throw std::bad_alloc{};
    auto *allocation=static_cast<Allocation *>(std::malloc(sizeof(Allocation)+bytes));
    if (!allocation) throw std::bad_alloc{};
    allocation->bytes=bytes; heap_live+=bytes; heap_peak=std::max(heap_peak,heap_live);
    return allocation+1;
}
void operator delete(void *pointer) noexcept {
    if (!pointer) return;
    auto *allocation=static_cast<Allocation *>(pointer)-1;
    heap_live-=allocation->bytes; std::free(allocation);
}
void operator delete(void *pointer,size_t) noexcept { ::operator delete(pointer); }
void *operator new[](size_t bytes) { return ::operator new(bytes); }
void operator delete[](void *pointer) noexcept { ::operator delete(pointer); }
void operator delete[](void *pointer,size_t) noexcept { ::operator delete(pointer); }
using namespace fort_scoped::planning;

fort_scope_plan_costs costs() {
    fort_scope_plan_costs c{};
    c.version=1; c.valid=1; c.max_allocation_bytes=size_t(1)<<30;
    c.cpu_flops=1000; c.cpu_bandwidth=1e9; c.gpu_flops=1e6; c.gpu_bandwidth=1e10;
    c.h2d_latency=c.d2h_latency=1e-6; c.h2d_bandwidth=c.d2h_bandwidth=1e9;
    c.create_seconds=c.register_seconds=c.host_access_seconds=c.device_access_seconds=1e-9;
    c.gpu_setup_seconds=c.cold_driver_startup_seconds=c.allocation_seconds=c.release_seconds=1e-6;
    c.wait_seconds=1e-6; c.launch_enqueue_seconds=1e-5; c.planning_operation_seconds=1e-12;
    return c;
}
Resource resource(fort_buffer_t handle, std::vector<size_t> extents) {
    Resource b; b.handle=handle; b.element_bytes=8; b.extents=extents; b.bytes=8;
    for (size_t n: extents) b.bytes*=n;
    Box full{std::vector<size_t>(extents.size(),0),extents};
    b.initialized=b.host_current={full}; return b;
}
Effects rw(const Region &r) { return {r,r,{}}; }
Operation worker(uint64_t unit, fort_buffer_t handle, Effects e, double flops=1000) {
    return {FORT_SCOPE_PLAN_WORKER,unit,{{handle,e}},flops,0,true};
}
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    std::string scenario=argv[1]; auto c=costs(); Inputs in;
    auto b=resource(1,{512}); Region full=b.initialized;
    in.resources={b}; in.operations={worker(1,1,rw(full))};
    if (scenario=="four") {
        c.launch_enqueue_seconds=.01;
        in.operations={worker(1,1,rw(full),1),worker(2,1,rw(full)),worker(3,1,rw(full)),worker(4,1,rw(full),1)};
    } else if (scenario=="read_mirror" || scenario=="write_invalidation") {
        Region face={Box{{0},{1}}}; Effects host{face,{},{}};
        if (scenario=="write_invalidation") host.writes=face;
        Operation native{FORT_SCOPE_PLAN_NATIVE,0,{{1,host}},0,0,false};
        in.operations={worker(1,1,rw(full)),native,worker(2,1,rw(full))};
    } else if (scenario=="opposite_faces" || scenario=="partial_initialized") {
        b=resource(1,{8,4,6}); Region faces={Box{{0,0,0},{8,4,1}},Box{{0,0,5},{8,4,6}}};
        if (scenario=="partial_initialized") b.initialized=b.host_current=faces;
        in.resources={b}; in.operations={worker(1,1,rw(faces))};
    } else if (scenario=="physical_copies") {
        b=resource(1,{4,4,5,3,4}); Region section={Box{{1,1,1,0,0},{2,3,4,2,3}}};
        in.resources={b}; in.operations={worker(1,1,rw(section))};
    } else if (scenario=="forget") {
        Operation forget{FORT_SCOPE_PLAN_FORGET,0,{{1,{}}},0,0,false};
        Effects overwrite{{},full,full};
        in.operations={worker(1,1,overwrite),forget,worker(2,1,overwrite)};
    } else if (scenario=="retained_peak") {
        in.resources.push_back(resource(2,{1024}));
        in.operations.push_back(worker(2,2,rw(in.resources[1].initialized)));
    } else if (scenario=="device_budget" || scenario=="allocation_range") {
        b=resource(1,{size_t(1)<<20}); in.resources={b};
        Region tiny={Box{{0},{1}}}; in.operations={worker(1,1,rw(tiny))};
        if (scenario=="device_budget") in.device_budget=b.bytes-1;
        else c.max_allocation_bytes=b.bytes-1;
    } else if (scenario=="cold" || scenario=="warm") {
        c.cold_driver_startup_seconds=2; in.driver_initialized=scenario=="warm";
    } else if (scenario=="startup_small_native") {
        in.operations[0].flops=.001; in.query_construction_operations=7;
    } else if (scenario=="common_native_compute") {
        in.operations.insert(in.operations.begin(), Operation{FORT_SCOPE_PLAN_NATIVE,0,{},1000000,0,false});
    } else if (scenario=="below_margin" || scenario=="above_margin") {
        in.resources.clear(); in.operations={{FORT_SCOPE_PLAN_WORKER,1,{},1000,0,true}};
        c.gpu_flops=1000/(scenario=="below_margin" ? .81 : .79);
    } else if (scenario=="whole_group" || scenario=="many_units") {
        in.operations.clear(); const int count=scenario=="whole_group" ? 6 : 24;
        for (int k=0;k<count;++k) in.operations.push_back(worker(k+1,1,rw(full)));
    } else if (scenario=="retained_state") {
        in.resources.clear(); in.operations.clear();
        for (int k=1;k<=128;++k) in.resources.push_back(resource(k,{16,4,4,4,4,4}));
        for (int k=0;k<32;++k) {
            in.operations.push_back(worker(k+1,1,rw(in.resources[0].initialized)));
            in.operations.push_back(Operation{}); in.operations.push_back(Operation{});
        }
        in.operations.insert(in.operations.end(),96,Operation{});
    } else if (scenario=="native_only") {
        in.operations[0].gpu_available=false;
    } else if (scenario=="existing_device_native") {
        in.resources[0].allocated=true; in.resources[0].device_current=full;
        in.resources[0].host_current.clear(); in.operations[0].gpu_available=false;
    } else if (scenario=="query_cost") {
        in.query_construction_operations=1000000000000ULL;
    } else if (scenario=="empty") {
        in.resources[0].extents={0,std::numeric_limits<size_t>::max()}; in.resources[0].bytes=0;
        in.resources[0].initialized.clear(); in.resources[0].host_current.clear(); in.operations.clear();
    } else if (scenario=="thin") {
        c.h2d_latency=c.d2h_latency=.01; in.operations[0].flops=1;
    } else if (scenario=="unknown_work") {
        in.operations[0].flops=0;
    } else if (scenario=="overflowed_work") {
        in.operations[0].flops=double(std::numeric_limits<uint64_t>::max());
    } else if (scenario=="missing_costs") {
        c.valid=0;
    } else if (scenario=="invalid_costs") {
        c.cpu_bandwidth=0;
    } else if (scenario=="zero_transfer_latency") {
        c.h2d_latency=c.d2h_latency=0;
    } else if (scenario=="negative_h2d_latency") {
        c.h2d_latency=-1e-6;
    } else if (scenario=="negative_d2h_latency") {
        c.d2h_latency=-1e-6;
    } else if (scenario=="uninitialized") {
        in.resources[0].initialized.clear(); in.resources[0].host_current.clear();
    } else if (scenario=="record_budget") {
        in.operations.assign(257, Operation{});
    } else if (scenario=="worker_budget") {
        in.operations.clear();
        for (int k=0;k<65;++k) in.operations.push_back(worker(k+1,1,rw(full)));
    } else if (scenario=="overflow") {
        in.resources[0].extents={std::numeric_limits<size_t>::max(),2};
        in.resources[0].initialized.clear(); in.resources[0].host_current.clear();
    } else if (scenario!="basic") return 3;
    const auto input_heap=heap_live; heap_peak=input_heap;
    auto r=select(in,c); auto &d=r.decision;
    const auto metadata_peak=heap_peak-input_heap;
    std::cout << std::setprecision(17) << "{\"available\":" << d.available << ",\"choices\":[";
    for(size_t k=0;k<r.gpu_workers.size();++k) { if(k)std::cout<<','; std::cout<<(r.gpu_workers[k]?1:0); }
    std::cout << "],\"reason\":\"" << r.reason << "\",\"candidates\":" << d.candidates
              << ",\"operations\":" << d.simulated_operations << ",\"upload_bytes\":" << d.upload_bytes
              << ",\"download_bytes\":" << d.download_bytes << ",\"uploads\":" << d.uploads
              << ",\"downloads\":" << d.downloads << ",\"launches\":" << d.launches
              << ",\"waits\":" << d.waits << ",\"allocations\":" << d.allocations
              << ",\"peak\":" << d.peak_device_bytes << ",\"seconds\":" << d.estimated_seconds
              << ",\"native_seconds\":" << d.native_seconds
              << ",\"native_common_excluded\":" << r.native_common_compute_excluded
              << ",\"metadata_peak\":" << metadata_peak << "}\n";
}
'''


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("g++ unavailable")
    directory = tmp_path_factory.mktemp("scoped_planning")
    source, executable = directory / "driver.cpp", directory / "driver"
    source.write_text(DRIVER)
    built = subprocess.run(
        [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-I", str(RUNTIME),
         str(source), "-o", str(executable)], capture_output=True, text=True, timeout=60,
    )
    assert built.returncode == 0, built.stderr

    def run(scenario):
        result = subprocess.run([str(executable), scenario], capture_output=True, text=True, timeout=20)
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    return run


def test_four_unit_sequence_prefers_middle_gpu_interval(model):
    result = model("four")
    assert result["available"] == 1
    assert result["choices"] == [0, 1, 1, 0]
    assert result["launches"] == 2
    assert result["uploads"] == result["downloads"] == result["allocations"] == 1
    assert result["upload_bytes"] == result["download_bytes"] == result["peak"] == 4096
    assert result["seconds"] < result["native_seconds"] * .02


def test_native_read_mirror_retains_gpu_validity(model):
    result = model("read_mirror")
    assert result["choices"] == [1, 1]
    assert result["uploads"] == 1
    assert result["upload_bytes"] == 4096
    assert result["downloads"] == 2
    assert result["download_bytes"] == 4096 + 8
    assert result["native_common_excluded"] == 1


def test_native_write_invalidates_only_written_gpu_section(model):
    result = model("write_invalidation")
    assert result["choices"] == [1, 1]
    assert result["uploads"] == 2
    assert result["upload_bytes"] == 4096 + 8
    assert result["download_bytes"] == 4096 + 8


@pytest.mark.parametrize("scenario", ["opposite_faces", "partial_initialized"])
def test_opposite_faces_do_not_become_full_volume_transfer(model, scenario):
    result = model(scenario)
    assert result["choices"] == [1]
    assert result["upload_bytes"] == result["download_bytes"] == 512
    assert result["uploads"] == result["downloads"] == 2
    assert result["peak"] == 1536


def test_transfer_latency_counts_physical_copy_calls(model):
    result = model("physical_copies")
    assert result["choices"] == [1]
    assert result["upload_bytes"] == result["download_bytes"] == 288
    assert result["uploads"] == result["downloads"] == 6
    assert result["peak"] == 8 * 4 * 4 * 5 * 3 * 4


def test_definition_forget_discards_values_and_retains_allocation(model):
    result = model("forget")
    assert result["choices"] == [1, 1]
    assert result["uploads"] == 0
    assert result["downloads"] == result["allocations"] == 1
    assert result["download_bytes"] == result["peak"] == 4096
    assert result["waits"] == 3


def test_peak_counts_full_retained_allocations(model):
    result = model("retained_peak")
    assert result["choices"] == [1, 1]
    assert result["allocations"] == 2
    assert result["peak"] == (512 + 1024) * 8


@pytest.mark.parametrize("scenario", ["device_budget", "allocation_range"])
def test_tiny_transfer_does_not_hide_full_allocation_budget(model, scenario):
    result = model(scenario)
    assert result["available"] == 1
    assert result["choices"] == [0]
    assert result["allocations"] == result["upload_bytes"] == result["download_bytes"] == 0


def test_cold_initialization_is_charged_until_device_is_known_initialized(model):
    cold, warm = model("cold"), model("warm")
    assert cold["choices"] == [0]
    assert cold["candidates"] == 1
    assert cold["reason"] == "native_gpu_startup_lower_bound"
    assert warm["choices"] == [1]
    assert warm["candidates"] > 1


def test_proven_startup_lower_bound_returns_native_before_candidate_search(model):
    result = model("startup_small_native")
    assert result["available"] == 1
    assert result["choices"] == [0]
    assert result["candidates"] == 1
    # One numerical operation and one publication simulation, plus seven
    # caller-reported reset/record construction operations.
    assert result["operations"] == 9
    assert result["reason"] == "native_gpu_startup_lower_bound"
    assert result["seconds"] > result["native_seconds"]
    assert model("basic")["choices"] == [1]


def test_gpu_requires_twenty_percent_estimated_advantage(model):
    assert model("below_margin")["choices"] == [0]
    assert model("above_margin")["choices"] == [1]


def test_interval_margin_does_not_claim_twenty_percent_whole_application_speedup(model):
    result = model("common_native_compute")
    assert result["choices"] == [1]
    assert result["native_seconds"] == 1001
    # The known fixed CPU helper remains in both complete schedules. Its
    # common cost cannot be counted as a GPU benefit or removed by placement.
    assert 1000 < result["seconds"] < 1001
    assert result["seconds"] / result["native_seconds"] > .99
    assert result["native_common_excluded"] == 0


@pytest.mark.parametrize(("scenario", "count"), [("whole_group", 6), ("many_units", 24), ("retained_state", 32)])
def test_complete_block_alternative_survives_bounded_frontier(model, scenario, count):
    result = model(scenario)
    assert result["available"] == 1
    assert result["choices"] == [1] * count
    assert result["launches"] == count
    assert result["allocations"] == result["uploads"] == result["downloads"] == 1
    assert 1 < result["candidates"] <= 128
    assert result["operations"] > count
    if scenario == "retained_state":
        # Hundreds of source positions must not each retain their own full
        # coherent frontier; candidate metadata also excludes finished states.
        assert result["metadata_peak"] < 16 * 1024**2


@pytest.mark.parametrize("scenario", ["native_only", "thin"])
def test_valid_all_native_choice_has_no_cuda_work(model, scenario):
    result = model(scenario)
    assert result["available"] == 1
    assert result["choices"] == [0]
    assert result["operations"] > 0
    assert result["uploads"] == result["downloads"] == result["launches"] == result["allocations"] == 0


def test_native_plan_publishes_preexisting_device_current_values(model):
    result = model("existing_device_native")
    assert result["available"] == 1
    assert result["choices"] == [0]
    assert result["uploads"] == result["allocations"] == result["launches"] == 0
    assert result["download_bytes"] == result["peak"] == 4096
    assert result["downloads"] == 1
    assert result["waits"] == 2


def test_planning_and_query_construction_cost_can_prevent_offload(model):
    ordinary, costly = model("basic"), model("query_cost")
    assert ordinary["choices"] == [1]
    assert costly["choices"] == [0]
    assert costly["operations"] == ordinary["operations"] + 10**12
    assert costly["seconds"] > costly["native_seconds"]


def test_empty_resource_avoids_overflow_of_unused_physical_axes(model):
    result = model("empty")
    assert result["available"] == 1
    assert result["choices"] == []
    assert result["allocations"] == result["peak"] == 0


def test_zero_fitted_transfer_latency_is_valid_without_artificial_epsilon(model):
    result = model("zero_transfer_latency")
    assert result["available"] == 1
    assert result["choices"] == [1]
    assert result["upload_bytes"] == result["download_bytes"] == 4096
    assert result["seconds"] < model("basic")["seconds"]


@pytest.mark.parametrize(("scenario", "reason"), [
    ("unknown_work", "unknown_worker_work"),
    ("overflowed_work", "unknown_or_overflowed_work"),
    ("missing_costs", "missing_or_incompatible_calibration"),
    ("invalid_costs", "invalid_calibration_cost"),
    ("negative_h2d_latency", "invalid_calibration_cost"),
    ("negative_d2h_latency", "invalid_calibration_cost"),
    ("uninitialized", "uninitialized_read"),
    ("record_budget", "planning_record_budget_exceeded"),
    ("worker_budget", "planning_worker_budget_exceeded"),
    ("overflow", "arithmetic_overflow"),
])
def test_unknown_and_overflowed_estimates_remain_native(model, scenario, reason):
    result = model(scenario)
    assert result["available"] == 0
    assert not any(result["choices"])
    assert result["reason"] == reason
