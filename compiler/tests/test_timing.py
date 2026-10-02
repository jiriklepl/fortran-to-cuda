"""Check phase attribution and profiling concurrency with a deterministic CUDA clock."""

from __future__ import annotations

import re
from importlib.resources import files

import pytest

from compiler.tests.test_memory_runtime import _run, _tool

FAKE_TIMING = r"""
#pragma once
#define __CUDACC__
#include <atomic>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <utility>
#include <vector>
namespace fake {
inline std::atomic<long long> clock{0};
inline std::atomic<int> events{0}, syncs{0}, event_syncs{0};
inline std::vector<std::string> operations;
inline void advance(int duration) { clock.fetch_add(duration); }
}
struct Event { long long time = 0; };
using cudaEvent_t = Event*;
inline int cudaEventCreate(Event** event) { ++fake::events; *event = new Event; return 0; }
inline int cudaEventDestroy(Event* event) { delete event; return 0; }
inline int cudaDeviceSynchronize() {
    ++fake::syncs;
    fake::advance(1000);
    fake::operations.push_back("device_sync");
    return 0;
}
inline int cudaEventRecord(Event* event, int) {
    event->time = fake::clock.load();
    fake::operations.push_back("record");
    return 0;
}
inline int cudaEventSynchronize(Event*) {
    ++fake::event_syncs;
    fake::advance(2000);
    fake::operations.push_back("event_sync");
    return 0;
}
inline int cudaEventElapsedTime(float* elapsed, Event* begin, Event* end) {
    *elapsed = static_cast<float>(end->time - begin->time);
    return 0;
}
#define CUCH(call) do { assert((call) == 0); } while (false)
#include "timing.hpp"
namespace timing = generated_kernels::timing;
inline void phases() {
    timing::measure_alloc([] { fake::advance(2); });
    timing::measure_h2d(80, [] { fake::advance(5); });
    fake::advance(100); // Host execution must not enter kernel timing.
    timing::measure_kernel_executions([] { fake::advance(7); });
    timing::measure_kernel_executions([] { fake::advance(11); });
    timing::measure_d2h(80, [] { fake::advance(13); });
    timing::measure_free([] { fake::advance(17); });
}
"""

TIMING_MAIN = r"""
#include "fake_timing.hpp"
#include <chrono>
#include <future>
extern void other_entry();
extern const void* other_profiling_state();
int main(int argc, char** argv) {
    assert(argc == 2);
    const std::string mode = argv[1];
    assert(other_profiling_state() == &timing::profiling);
    if (mode == "disabled") {
        // Taking the profiling lock must not block an unprofiled public call.
        std::unique_lock<std::recursive_mutex> lock(timing::profiling.mutex);
        std::promise<void> finished;
        auto result = finished.get_future();
        std::thread worker([&] {
            timing::ProfiledCallGuard call;
            assert(!call.enabled());
            timing::record_run();
            phases();
            finished.set_value();
        });
        assert(result.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
        lock.unlock();
        worker.join();
        timing::print_timing_summary();
        assert(fake::events == 0 && fake::syncs == 0 && fake::event_syncs == 0);
        assert(timing::profiling.calls == 0 && timing::profiling.kernel_ms.empty());
        return 0;
    }
    timing::reset_timing_vectors();
    if (mode == "host") {
        timing::ProfiledCallGuard call;
        timing::record_run();
        fake::advance(100);
        timing::measure_d2h(80, [] { fake::advance(13); });
    } else if (mode == "empty") {
        timing::ProfiledCallGuard call;
        timing::record_run();
    } else if (mode == "phases") {
        timing::ProfiledCallGuard call;
        timing::record_run();
        phases();
        assert(fake::events == 12 && fake::syncs == 7 && fake::event_syncs == 6);
        assert(fake::operations[1] == "device_sync");
        assert(fake::operations[2] == "record");
        assert(fake::operations[3] == "record");
        assert(fake::operations[4] == "event_sync");
    } else if (mode == "reset") {
        other_entry();
        timing::reset_timing_vectors();
        timing::record_run(); // New empty batch discards every old phase and byte sample.
    } else if (mode == "concurrent") {
        std::atomic<int> ready{0}, active{0};
        std::atomic<bool> go{false};
        std::vector<std::thread> workers;
        for (int t = 0; t < 8; ++t) {
            workers.emplace_back([&] {
                ++ready;
                while (!go.load()) std::this_thread::yield();
                for (int iteration = 0; iteration < 25; ++iteration) {
                    timing::ProfiledCallGuard call;
                    assert(call.enabled());
                    assert(active.fetch_add(1) == 0);
                    other_entry(); // Nested public guard and helpers share a recursive lock.
                    std::this_thread::yield();
                    assert(active.fetch_sub(1) == 1);
                }
            });
        }
        while (ready.load() != 8) std::this_thread::yield();
        go = true;
        for (auto& worker : workers) worker.join();
        assert(fake::events == 2400 && fake::event_syncs == 1200);
    } else {
        assert(false);
    }
    timing::print_timing_summary();
    const int events = fake::events.load(), syncs = fake::syncs.load();
    other_entry(); // finish disables the next execution and a repeated finish is a no-op.
    timing::print_timing_summary();
    assert(fake::events == events && fake::syncs == syncs);
}
"""

OTHER_ENTRY = r"""
#include "fake_timing.hpp"
void other_entry() {
    timing::ProfiledCallGuard call;
    timing::record_run();
    phases();
}
const void* other_profiling_state() { return &timing::profiling; }
"""


@pytest.fixture(scope="module")
def timing_runtime(tmp_path_factory):
    directory = tmp_path_factory.mktemp("timing_runtime")
    (directory / "timing.hpp").write_text(files("compiler.runtime").joinpath("timing.hpp").read_text())
    (directory / "fake_timing.hpp").write_text(FAKE_TIMING)
    (directory / "main.cpp").write_text(TIMING_MAIN)
    (directory / "other.cpp").write_text(OTHER_ENTRY)
    _run([_tool("g++"), "-std=c++17", "-pthread", "main.cpp", "other.cpp", "-o", "run"], directory)
    return directory


def _summary(output):
    return {name: float(value) for name, value in re.findall(r"^(\w+):\s+([\d.]+)", output, re.MULTILINE)}


@pytest.mark.native
def test_disabled_profiling_takes_no_locks_events_or_barriers(timing_runtime):
    assert _run(["./run", "disabled"], timing_runtime).stdout == ""


@pytest.mark.native
@pytest.mark.parametrize("mode", ["host", "empty", "reset"])
def test_run_count_does_not_depend_on_kernel_launches(timing_runtime, mode):
    summary = _summary(_run(["./run", mode], timing_runtime).stdout)
    assert summary["calls"] == 1
    assert summary["kernel_launches"] == 0
    assert summary["kernel_total_ms"] == 0
    assert summary["d2h_total_ms"] == (13 if mode == "host" else 0)
    assert summary["h2d_total_ms"] == summary["malloc_total_ms"] == summary["free_total_ms"] == 0


@pytest.mark.native
@pytest.mark.parametrize(("mode", "calls"), [("phases", 1), ("concurrent", 200)])
def test_kernel_samples_exclude_host_work_transfers_and_synchronization(timing_runtime, mode, calls):
    summary = _summary(_run(["./run", mode], timing_runtime).stdout)
    assert summary == {
        "calls": calls,
        "kernel_launches": calls * 2,
        "malloc_total_ms": calls * 2,
        "h2d_total_ms": calls * 5,
        "kernel_total_ms": calls * 18,
        "d2h_total_ms": calls * 13,
        "free_total_ms": calls * 17,
    }
