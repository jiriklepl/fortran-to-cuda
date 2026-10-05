"""Generated ordinary calls reuse device backing while retaining fresh call state."""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from compiler.driver.options import CompilerOptions
from compiler.driver.pipeline import prepare_function
from compiler.emission import generate_sources, read_common_header
from compiler.frontend import lower_file
from compiler.tests.test_device_pool_runtime import FAKE_CUDA_POOL
from compiler.tests.test_sessions import _simulate_cuda
from compiler.tests.test_timing import _summary

KERNEL_EXECUTION = r"""
#define __global__
#include <thread>
namespace fake {
inline std::atomic<int> generated_launches{0}, first_wave_target{0}, first_wave_arrived{0};
inline thread_local bool first_launch = true;
}
struct TestDimension { unsigned x = 0; };
inline thread_local TestDimension blockIdx, threadIdx, blockDim, gridDim;
template <typename Function, typename... Args>
void fort_test_launch(Function function, unsigned blocks, unsigned threads, Args... args) {
    ++fake::generated_launches;
    fake::clock += 7;
    if (fake::first_launch && fake::first_wave_target.load()) {
        fake::first_launch = false;
        ++fake::first_wave_arrived;
        while (fake::first_wave_arrived.load() < fake::first_wave_target.load()) std::this_thread::yield();
    }
    blockDim.x = threads;
    gridDim.x = blocks;
    for (blockIdx.x = 0; blockIdx.x < blocks; ++blockIdx.x)
        for (threadIdx.x = 0; threadIdx.x < threads; ++threadIdx.x)
            function(args...);
}
"""


def _source(body, *, intent="inout"):
    return f"""! kernels
module pooled_case
contains
! kernel
subroutine entry(a,n,delta)
integer,intent({intent})::a(:)
integer,intent(in)::n,delta
integer::i
{body}
end subroutine
end module
"""


def _compile_and_run(
    tmp_path, source, driver, *, native_pool=True, sanitize=False, pool_bytes="67108864", fallback="error"
):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("Native capability unavailable: g++")
    env = {**os.environ, "FORT_CUDA_POOL_BYTES": pool_bytes, "ASAN_OPTIONS": "detect_leaks=0"}
    flags = ["-fsanitize=address", "-g"] if sanitize else []
    if sanitize:
        (tmp_path / "probe.cpp").write_text("int main() {}\n")
        probe = subprocess.run(
            [compiler, *flags, "probe.cpp", "-o", "probe"], cwd=tmp_path, capture_output=True, text=True, check=False
        )
        if probe.returncode:
            pytest.skip("AddressSanitizer is unavailable: " + probe.stderr)
        probe = subprocess.run(["./probe"], cwd=tmp_path, env=env, capture_output=True, text=True, check=False)
        if probe.returncode:
            pytest.skip("AddressSanitizer execution is unavailable: " + probe.stderr)
    path = tmp_path / "source.f90"
    path.write_text(source)
    function, plan = prepare_function(lower_file(path, "entry"), options=CompilerOptions(fallback=fallback))
    generated = generate_sources(function, plan)
    (tmp_path / "common_functions.cuh").write_text(read_common_header())
    (tmp_path / "cuda_runtime.h").write_text(FAKE_CUDA_POOL + KERNEL_EXECUTION)
    driver = driver.replace("POOL_SUPPORTED", "1" if native_pool else "0")
    (tmp_path / "test.cpp").write_text(_simulate_cuda(generated.cuda) + "\n" + driver)
    build = subprocess.run(
        [compiler, "-std=c++17", "-O1", "-pthread", *flags, "test.cpp", "-o", "run"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    run = subprocess.run(["./run"], cwd=tmp_path, env=env, capture_output=True, text=True, check=False, timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr
    return run


@pytest.mark.native
@pytest.mark.parametrize("native_pool", [True, False], ids=["native-pool", "fresh-fallback"])
def test_reused_backing_has_fresh_inputs_scalars_shapes_and_independent_array_slots(tmp_path, native_pool):
    source = """! kernels
module pooled_case
contains
! kernel
subroutine entry(a,b,out,n,delta)
integer,intent(inout)::a(:),b(:)
integer,intent(out)::out(:)
integer,intent(in)::n,delta
integer::i
do i=1,n
 a(i)=a(i)+delta
 b(i)=b(i)*2
 out(i)=a(i)+b(i)
enddo
end subroutine
end module
"""
    driver = r"""#include <cassert>
#include <vector>
int main() {
    fake::pools_supported = POOL_SUPPORTED;
    int calls = 0;
    for (int extent : {4, 9, 2}) {
        int warm_allocations = -1;
        for (int repetition = 0; repetition < 2; ++repetition) {
            const int delta = ++calls;
            // New caller storage on every invocation must replace cached values.
            std::vector<int> a(extent, 10 * calls), b(extent, 100 + calls), out(extent, -999);
            generated_kernels::cpp_entry(a.data(), extent, b.data(), extent, out.data(), extent, extent, delta);
            for (int i = 0; i < extent; ++i) {
                assert(a[i] == 11 * calls);
                assert(b[i] == 2 * (100 + calls));
                assert(out[i] == a[i] + b[i]);
            }
            if (!repetition) warm_allocations = fake::backing_allocations.load();
            else if (POOL_SUPPORTED) assert(fake::backing_allocations == warm_allocations);
            else assert(fake::backing_allocations == calls * 3);
        }
    }
    assert(fake::backing_allocations >= 3); // Simultaneous slots have exclusive storage.
    assert(fake::backing_allocations == (POOL_SUPPORTED ? 9 : calls * 3));
    assert(fake::uploads == calls * 2 && fake::downloads == calls * 3);
    assert(fake::generated_launches == calls && fake::events == 0);
    assert(fake::syncs == calls * 2 && fake::frees == calls * 3);
    generated_kernels::cpp_entry_trim_cache();
    assert(fake::live.empty());
    assert(fake::pool_destroys == (POOL_SUPPORTED ? 1 : 0));
}
"""
    _compile_and_run(tmp_path, source, driver, native_pool=native_pool)


@pytest.mark.native
@pytest.mark.parametrize("native_pool", [True, False], ids=["native-pool", "fresh-fallback"])
@pytest.mark.parametrize("output_only", [False, True], ids=["inout", "output-only"])
def test_empty_arrays_and_zero_trip_calls_preserve_transfer_intents(tmp_path, native_pool, output_only):
    source = _source(
        "do i=1,n\na(i)=" + ("delta" if output_only else "a(i)+delta") + "\nenddo",
        intent="out" if output_only else "inout",
    )
    driver = r"""#include <cassert>
int main() {
    fake::pools_supported = POOL_SUPPORTED;
    generated_kernels::cpp_entry(nullptr, 0, 0, 7);
    assert(fake::backing_allocations == 0 && fake::uploads == 0 && fake::downloads == 0);
    assert(fake::generated_launches == 0);
    for (int repetition = 0; repetition < 2; ++repetition) {
        int a[] = {10, 20, 30, 40};
        generated_kernels::cpp_entry(a, 4, 4, repetition + 3);
        for (int i = 0; i < 4; ++i) assert(a[i] == EXPECTED_VALUE);
    }
    assert(fake::backing_allocations == (POOL_SUPPORTED ? 1 : 2));
    assert(fake::uploads == EXPECTED_UPLOADS && fake::downloads == 2);
    assert(fake::generated_launches == 2);
    ZERO_TRIP_CALLS
    generated_kernels::cpp_entry(nullptr, 0, 0, 9);
    assert(fake::events == 0);
    generated_kernels::cpp_entry_trim_cache();
    assert(fake::live.empty());
    assert(fake::pool_destroys == (POOL_SUPPORTED ? 1 : 0));
}
"""
    driver = driver.replace("EXPECTED_VALUE", "repetition + 3" if output_only else "10 * (i + 1) + repetition + 3")
    driver = driver.replace("EXPECTED_UPLOADS", "0" if output_only else "2")
    driver = driver.replace(
        "ZERO_TRIP_CALLS",
        ""
        if output_only
        else r"""
    for (int repetition = 0; repetition < 2; ++repetition) {
        int a[] = {repetition, 20, -30, 40};
        generated_kernels::cpp_entry(a, 4, 0, 999);
        assert(a[0] == repetition && a[1] == 20 && a[2] == -30 && a[3] == 40);
    }
    assert(fake::backing_allocations == (POOL_SUPPORTED ? 1 : 4));
    assert(fake::uploads == 4 && fake::downloads == 4 && fake::generated_launches == 2);
""",
    )
    _compile_and_run(tmp_path, source, driver, native_pool=native_pool)


@pytest.mark.native
@pytest.mark.parametrize("native_pool", [True, False], ids=["native-pool", "fresh-fallback"])
def test_reused_calls_preserve_host_partial_writes_fallback_branches_and_device_produced_bounds(tmp_path, native_pool):
    source = """! kernels
module pooled_case
contains
! kernel
subroutine entry(a,bounds,n,enabled,delta)
integer,intent(inout)::a(:),bounds(:)
integer,intent(in)::n,delta
logical,intent(in)::enabled
integer::i
do i=1,n
 a(i)=a(i)+delta
enddo
if(enabled)then
 a(2)=a(2)+100
else
 do i=2,n
  a(i)=a(i)+a(i-1)
 enddo
endif
do i=1,1
 bounds(i)=n
enddo
do i=1,bounds(1)
 a(i)=a(i)*2
enddo
end subroutine
end module
"""
    driver = r"""#include <cassert>
#include <vector>
int main() {
    fake::pools_supported = POOL_SUPPORTED;
    int previous_allocations = -1;
    constexpr int calls = 6;
    for (int repetition = 0; repetition < calls; ++repetition) {
        const int n = repetition < 4 ? 4 : 7;
        const int delta = repetition + 1;
        const bool enabled = repetition % 2 == 0;
        std::vector<int> a(n), expected(n);
        for (int i = 0; i < n; ++i) expected[i] = (a[i] = repetition * 10 + i + 1) + delta;
        if (enabled) expected[1] += 100;
        else for (int i = 1; i < n; ++i) expected[i] += expected[i - 1];
        for (int& value : expected) value *= 2;
        int bounds[] = {-500};
        generated_kernels::cpp_entry(a.data(), n, bounds, 1, n, enabled, delta);
        assert(a == expected && bounds[0] == n);
        if (repetition == 0 || repetition == 4) previous_allocations = fake::backing_allocations.load();
        else if (POOL_SUPPORTED) assert(fake::backing_allocations == previous_allocations);
        else assert(fake::backing_allocations == 2 * (repetition + 1));
    }
    assert(fake::uploads == calls * 3 && fake::downloads == calls * 4);
    assert(fake::generated_launches == calls * 3 && fake::events == 0);
    generated_kernels::cpp_entry_trim_cache();
    assert(fake::live.empty());
    assert(fake::pool_destroys == (POOL_SUPPORTED ? 1 : 0));
}
"""
    _compile_and_run(tmp_path, source, driver, native_pool=native_pool, fallback="host")


@pytest.mark.native
@pytest.mark.parametrize("native_pool", [True, False], ids=["native-pool", "fresh-fallback"])
def test_explicit_sessions_remain_dedicated_and_survive_ordinary_call_cache_trimming(tmp_path, native_pool):
    source = _source("do i=1,n\na(i)=a(i)+delta\nenddo")
    driver = r"""#include <cassert>
#include <vector>
int main() {
    fake::pools_supported = POOL_SUPPORTED;
    int ordinary[] = {1, 2, 3, 4};
    generated_kernels::cpp_entry(ordinary, 4, 4, 1);
    const int legacy_allocations = fake::allocations.load();
    const int pooled_allocations = fake::pool_allocations.load();
    std::int64_t token;
    {
        std::vector<int> temporary{10, 20, 30, 40};
        token = generated_kernels::cpp_entry_create(temporary.data(), 4);
    }
    assert(fake::allocations == legacy_allocations + 1);
    assert(fake::pool_allocations == pooled_allocations);
    const int backing = fake::backing_allocations.load();
    generated_kernels::cpp_entry_run(token, 4, 3);
    generated_kernels::cpp_entry(ordinary, 4, 4, 7);
    assert(fake::backing_allocations == backing + (POOL_SUPPORTED ? 0 : 1));
    generated_kernels::cpp_entry_trim_cache();
    generated_kernels::cpp_entry_run(token, 4, 5);
    int result[4];
    generated_kernels::cpp_entry_update_host_0(token, result, 4);
    for (int i = 0; i < 4; ++i) {
        assert(result[i] == 10 * (i + 1) + 8);
        assert(ordinary[i] == i + 9);
    }
    generated_kernels::cpp_entry_destroy(token);
    assert(fake::uploads == 3 && fake::downloads == 3);
    assert(fake::generated_launches == 4 && fake::events == 0);
    assert(fake::live.empty());
    assert(fake::pool_destroys == (POOL_SUPPORTED ? 1 : 0));
}
"""
    _compile_and_run(tmp_path, source, driver, native_pool=native_pool)


@pytest.mark.native
@pytest.mark.parametrize("native_pool", [True, False], ids=["native-pool", "fresh-fallback"])
@pytest.mark.parametrize("sanitize", [False, True], ids=["ordinary", "asan"])
def test_concurrent_reused_calls_have_exclusive_leases_on_each_device(tmp_path, native_pool, sanitize):
    source = _source("do i=1,n\na(i)=a(i)+delta\nenddo")
    driver = r"""#include <cassert>
#include <thread>
#include <vector>
int main() {
    fake::pools_supported = POOL_SUPPORTED;
    constexpr int threads = 6, repeats = 40;
    fake::first_wave_target = threads;
    std::vector<std::thread> workers;
    for (int thread = 0; thread < threads; ++thread) workers.emplace_back([thread] {
        cudaSetDevice(thread % 2);
        for (int repetition = 0; repetition < repeats; ++repetition) {
            int a[] = {thread, repetition, 7, -3};
            const int delta = repetition % 5 + 1;
            generated_kernels::cpp_entry(a, 4, 4, delta);
            assert(a[0] == thread + delta && a[1] == repetition + delta);
            assert(a[2] == 7 + delta && a[3] == -3 + delta);
            assert(fake::current_device == thread % 2);
        }
    });
    for (auto& worker : workers) worker.join();
    assert(fake::backing_allocations == (POOL_SUPPORTED ? threads : threads * repeats));
    assert(fake::uploads == threads * repeats && fake::downloads == threads * repeats);
    assert(fake::generated_launches == threads * repeats && fake::events == 0);
    for (int device : {0, 1}) {
        cudaSetDevice(device);
        generated_kernels::cpp_entry_trim_cache();
    }
    assert(fake::live.empty());
    assert(fake::pool_destroys == (POOL_SUPPORTED ? 2 : 0));
}
"""
    _compile_and_run(tmp_path, source, driver, native_pool=native_pool, sanitize=sanitize)


@pytest.mark.native
@pytest.mark.parametrize("native_pool", [True, False], ids=["native-pool", "fresh-fallback"])
def test_profiled_concurrent_hits_count_calls_kernels_and_transfers_without_new_backing(tmp_path, native_pool):
    source = _source("do i=1,n\na(i)=a(i)+delta\nenddo")
    driver = r"""#include <cassert>
#include <thread>
#include <vector>
int main() {
    fake::pools_supported = POOL_SUPPORTED;
    int warm[] = {1, 2, 3, 4};
    generated_kernels::cpp_entry(warm, 4, 4, 1);
    assert(fake::backing_allocations == 1 && fake::events == 0);
    constexpr int threads = 4, repeats = 7;
    generated_kernels::cpp_start_hot();
    std::vector<std::thread> workers;
    for (int thread = 0; thread < threads; ++thread) workers.emplace_back([thread] {
        for (int repetition = 0; repetition < repeats; ++repetition) {
            int a[] = {thread, repetition, 7, -3};
            generated_kernels::cpp_entry(a, 4, 4, 2);
            assert(a[0] == thread + 2 && a[1] == repetition + 2 && a[2] == 9 && a[3] == -1);
        }
    });
    for (auto& worker : workers) worker.join();
    generated_kernels::cpp_finish_hot();
    assert(fake::backing_allocations == (POOL_SUPPORTED ? 1 : threads * repeats + 1));
    assert(fake::events > 0);
    const int events = fake::events.load();
    generated_kernels::cpp_entry(warm, 4, 4, 1);
    generated_kernels::cpp_entry_trim_cache();
    assert(fake::events == events);
    assert(fake::uploads == threads * repeats + 2 && fake::downloads == threads * repeats + 2);
    assert(fake::live.empty());
    assert(fake::pool_destroys == (POOL_SUPPORTED ? 1 : 0));
}
"""
    result = _compile_and_run(tmp_path, source, driver, native_pool=native_pool)
    calls = 4 * 7
    assert _summary(result.stdout) == {
        "calls": calls,
        "kernel_launches": calls,
        "malloc_total_ms": 0 if native_pool else calls * 2,
        "h2d_total_ms": calls * 5,
        "kernel_total_ms": calls * 7,
        "d2h_total_ms": calls * 5,
        "free_total_ms": 0 if native_pool else calls * 3,
    }


@pytest.mark.native
@pytest.mark.parametrize("native_pool", [True, False], ids=["native-pool", "fresh-fallback"])
def test_zero_pool_budget_retains_ordinary_fresh_allocation_behavior(tmp_path, native_pool):
    source = _source("do i=1,n\na(i)=a(i)+delta\nenddo")
    driver = r"""#include <cassert>
int main() {
    fake::pools_supported = POOL_SUPPORTED;
    for (int repetition = 0; repetition < 3; ++repetition) {
        int a[] = {repetition, 2, 3, 4};
        generated_kernels::cpp_entry(a, 4, 4, 5);
        assert(a[0] == repetition + 5 && a[1] == 7 && a[2] == 8 && a[3] == 9);
        assert(fake::backing_allocations == repetition + 1 && fake::frees == repetition + 1);
    }
    assert(fake::pool_creates == 0 && fake::uploads == 3 && fake::downloads == 3 && fake::events == 0);
    generated_kernels::cpp_entry_trim_cache();
    assert(fake::frees == 3);
}
"""
    _compile_and_run(tmp_path, source, driver, native_pool=native_pool, pool_bytes="0")


@pytest.mark.native
@pytest.mark.parametrize(
    "unsupported_api", ["capability_error", "create_error"], ids=["capability-query", "pool-create"]
)
def test_unsupported_pool_probe_does_not_poison_generated_kernel_error_checks(tmp_path, unsupported_api):
    source = _source("do i=1,n\na(i)=a(i)+delta\nenddo")
    driver = r"""#include <cassert>
int main() {
    fake::UNSUPPORTED_API = cudaErrorNotSupported;
    for (int repetition = 0; repetition < 2; ++repetition) {
        int a[] = {repetition, 2, 3, 4};
        generated_kernels::cpp_entry(a, 4, 4, 5);
        assert(a[0] == repetition + 5 && a[1] == 7 && a[2] == 8 && a[3] == 9);
        assert(cudaGetLastError() == cudaSuccess);
        assert(fake::backing_allocations == repetition + 1 && fake::frees == repetition + 1);
    }
    assert(fake::capability_queries == 1 && fake::pool_creates == 0);
    assert(fake::generated_launches == 2 && fake::uploads == 2 && fake::downloads == 2);
    assert(fake::events == 0 && fake::live.empty());
    generated_kernels::cpp_entry_trim_cache();
    assert(fake::pool_destroys == 0);
}
"""
    _compile_and_run(tmp_path, source, driver.replace("UNSUPPORTED_API", unsupported_api))
