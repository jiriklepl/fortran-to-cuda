"""Measure generated execution paths with an instrumented CUDA clock."""

from __future__ import annotations

import pytest

from compiler.tests import test_call_ownership as ownership
from compiler.tests.test_call_ownership import compile_simulated, counted_runtime
from compiler.tests.test_timing import _summary


def _clock_runtime():
    runtime = counted_runtime()
    replacements = {
        "inline int cudaMalloc": "inline std::atomic<int> fort_test_clock{0};\ninline int cudaMalloc",
        "++fort_test_allocations;": "++fort_test_allocations; fort_test_clock += 2;",
        "++fort_test_frees;": "++fort_test_frees; fort_test_clock += 3;",
        "std::memcpy(to, from, n);": "fort_test_clock += 5; std::memcpy(to, from, n);",
        "++fort_test_syncs;": "++fort_test_syncs; fort_test_clock += 1000;",
        "cudaGetLastError() { return 0; }": "cudaGetLastError() { fort_test_clock += 100; return 0; }",
        "cudaEventRecord(cudaEvent_t, int) { return 0; }": "cudaEventRecord(cudaEvent_t event, int) { *event = fort_test_clock.load(); return 0; }",
        "cudaEventSynchronize(cudaEvent_t) { return 0; }": "cudaEventSynchronize(cudaEvent_t) { fort_test_clock += 1000; return 0; }",
        "cudaEventElapsedTime(float* value, cudaEvent_t, cudaEvent_t) { *value = 0; return 0; }": "cudaEventElapsedTime(float* value, cudaEvent_t start, cudaEvent_t stop) { "
        "*value = *stop - *start; return 0; }",
        "    blockDim.x = threads;": "    fort_test_clock += 7;\n    blockDim.x = threads;",
    }
    for before, after in replacements.items():
        assert before in runtime
        runtime = runtime.replace(before, after)
    return runtime


@pytest.fixture(autouse=True)
def instrument_clock(monkeypatch):
    runtime = _clock_runtime()
    monkeypatch.setattr(ownership, "counted_runtime", lambda: runtime)


def _source(body, *, logical=False):
    return f"""! kernels
module timing_case
contains
! kernel
subroutine entry(a,n{",enabled" if logical else ""})
integer,intent(inout)::a(:)
integer,intent(in)::n
{"logical,intent(in)::enabled" if logical else ""}
integer::i
{body}
end subroutine
end module
"""


@pytest.mark.native
@pytest.mark.parametrize("host_only", [False, True], ids=["empty", "host"])
def test_generated_non_kernel_execution_counts_run_without_kernel_time(tmp_path, host_only):
    source = _source("a(1)=a(1)+10" if host_only else "do i=1,n\na(i)=a(i)+10\nenddo")
    driver = """#include <cassert>
int main() {
    int a[] = {1, 2, 3, 4};
    generated_kernels::cpp_start_hot();
    generated_kernels::cpp_entry(a, 4, 0);
    generated_kernels::cpp_finish_hot();
    assert(a[0] == EXPECTED_FIRST && a[1] == 2 && a[2] == 3 && a[3] == 4);
    assert(fort_test_allocations == 1 && fort_test_frees == 1);
    assert(fort_test_uploads == 1 && fort_test_downloads == 1);
}
""".replace("EXPECTED_FIRST", "11" if host_only else "1")
    _, run = compile_simulated(tmp_path, source, driver)
    assert _summary(run.stdout) == {
        "calls": 1,
        "kernel_launches": 0,
        "malloc_total_ms": 2,
        "h2d_total_ms": 5,
        "kernel_total_ms": 0,
        "d2h_total_ms": 5,
        "free_total_ms": 3,
    }


@pytest.mark.native
def test_generated_kernels_exclude_intervening_host_coherence_transfers(tmp_path):
    source = _source("do i=1,n\na(i)=a(i)+1\nenddo\na(1)=a(1)+10\ndo i=1,n\na(i)=a(i)*2\nenddo")
    driver = """#include <cassert>
int main() {
    int a[] = {1, 2, 3, 4};
    generated_kernels::cpp_start_hot();
    generated_kernels::cpp_entry(a, 4, 4);
    generated_kernels::cpp_finish_hot();
    assert(a[0] == 24 && a[1] == 6 && a[2] == 8 && a[3] == 10);
    assert(fort_test_uploads == 2 && fort_test_downloads == 2);
}
"""
    _, run = compile_simulated(tmp_path, source, driver)
    assert _summary(run.stdout) == {
        "calls": 1,
        "kernel_launches": 2,
        "malloc_total_ms": 2,
        "h2d_total_ms": 10,
        "kernel_total_ms": 14,
        "d2h_total_ms": 10,
        "free_total_ms": 3,
    }


@pytest.mark.native
def test_generated_branch_counts_only_executed_nonempty_launches(tmp_path):
    source = _source("if(enabled)then\ndo i=1,n\na(i)=a(i)+10\nenddo\nelse\na(1)=a(1)+100\nendif", logical=True)
    driver = """#include <cassert>
int main() {
    int a[] = {1, 2, 3, 4};
    generated_kernels::cpp_start_hot();
    generated_kernels::cpp_entry(a, 4, 4, true);
    assert(a[0] == 11 && a[1] == 12 && a[2] == 13 && a[3] == 14);
    generated_kernels::cpp_entry(a, 4, 4, false);
    assert(a[0] == 111 && a[1] == 12 && a[2] == 13 && a[3] == 14);
    generated_kernels::cpp_entry(a, 4, 0, true);
    assert(a[0] == 111 && a[1] == 12 && a[2] == 13 && a[3] == 14);
    generated_kernels::cpp_finish_hot();
    assert(fort_test_uploads == 3 && fort_test_downloads == 3);
}
"""
    _, run = compile_simulated(tmp_path, source, driver)
    assert _summary(run.stdout) == {
        "calls": 3,
        "kernel_launches": 1,
        "malloc_total_ms": 6,
        "h2d_total_ms": 15,
        "kernel_total_ms": 7,
        "d2h_total_ms": 15,
        "free_total_ms": 9,
    }


@pytest.mark.native
def test_generated_concurrent_profiled_calls_have_exact_phase_totals(tmp_path):
    source = _source("do i=1,n\na(i)=a(i)+10\nenddo")
    driver = """#include <cassert>
#include <thread>
#include <vector>
int main() {
    constexpr int threads = 8, repeats = 20;
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::vector<std::thread> workers;
    generated_kernels::cpp_start_hot();
    for (int t = 0; t < threads; ++t) workers.emplace_back([&, t] {
        ++ready;
        while (!go.load()) std::this_thread::yield();
        for (int repeat = 0; repeat < repeats; ++repeat) {
            int a[] = {t, repeat, 7, -3};
            generated_kernels::cpp_entry(a, 4, 4);
            assert(a[0] == t + 10 && a[1] == repeat + 10 && a[2] == 17 && a[3] == 7);
        }
    });
    while (ready.load() != threads) std::this_thread::yield();
    go = true;
    for (auto& worker : workers) worker.join();
    generated_kernels::cpp_finish_hot();
    assert(fort_test_allocations == threads * repeats && fort_test_frees == threads * repeats);
    assert(fort_test_uploads == threads * repeats && fort_test_downloads == threads * repeats);
    assert(fort_test_events == threads * repeats * 10);
}
"""
    _, run = compile_simulated(tmp_path, source, driver)
    calls = 8 * 20
    assert _summary(run.stdout) == {
        "calls": calls,
        "kernel_launches": calls,
        "malloc_total_ms": calls * 2,
        "h2d_total_ms": calls * 5,
        "kernel_total_ms": calls * 7,
        "d2h_total_ms": calls * 5,
        "free_total_ms": calls * 3,
    }
