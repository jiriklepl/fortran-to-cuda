"""Compile both default-stream modes and measure native backing reuse when available."""

from __future__ import annotations

import json
import os
import re

import pytest

from compiler.driver.pipeline import prepare_function
from compiler.emission import generate_sources, read_common_header
from compiler.frontend import lower_file
from compiler.tests.test_memory_runtime import _run, _tool

SOURCE = """! kernels
module pool_case
integer,parameter::knd=kind(1d0)
contains
! kernel
subroutine entry(a,b,n,delta)
real(knd),intent(inout)::a(:)
real(knd),intent(in)::b(:)
integer,intent(in)::n,delta
integer::i
do i=1,n
a(i)=a(i)+b(i)*delta
enddo
end subroutine
end module
"""

DRIVER = r"""
#include "generated_code.cu"
#include <cassert>
#include <chrono>
#include <iomanip>
#include <string>

using Clock = std::chrono::steady_clock;
using namespace generated_kernels;

#if CUDART_VERSION >= 11030
cudaMemPool_t existing_pool() {
    auto& registry = storage::allocation_detail::pool_registry();
    std::lock_guard<std::mutex> guard(registry.mutex);
    auto found = registry.devices.find(0);
    return found == registry.devices.end() ? nullptr : found->second->pool;
}

std::uint64_t attribute(cudaMemPool_t pool, cudaMemPoolAttr name) {
    std::uint64_t value = 0;
    CUCH(cudaMemPoolGetAttribute(pool, name, &value));
    return value;
}
#endif

int main() {
    int count = 0;
    cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess || count == 0) {
        std::fprintf(stderr, "CUDA device unavailable: %s; count=%d\n", cudaGetErrorString(status), count);
        return 77;
    }
    status = cudaSetDevice(0);
    if (status == cudaSuccess) status = cudaFree(nullptr);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA context unavailable: %s\n", cudaGetErrorString(status));
        return 77;
    }
#if CUDART_VERSION < 11030
    std::fprintf(stderr, "Native pool-statistics test requires CUDA 11.3 or newer\n");
    return 77;
#else
    int supported = 0;
    CUCH(cudaDeviceGetAttribute(&supported, cudaDevAttrMemoryPoolsSupported, 0));
    if (!supported) {
        std::fprintf(stderr, "CUDA device has no native memory pool support\n");
        return 77;
    }
    const bool pooled = std::string(std::getenv("FORT_CUDA_POOL_BYTES")) != "0";
    cudaMemPool_t application_pool = nullptr;
    CUCH(cudaDeviceGetMemPool(&application_pool, 0));
    constexpr int n = 16384, repetitions = 8;
    std::vector<double> a(n), b(n);
    auto call = [&](int iteration) {
        for (int i = 0; i < n; ++i) {
            a[i] = i - iteration;
            b[i] = (i % 7) * 0.25;
        }
        const auto begin = Clock::now();
        cpp_entry(a.data(), n, b.data(), n, n, iteration + 1);
        const auto end = Clock::now();
        for (int i = 0; i < n; ++i)
            assert(a[i] == i - iteration + b[i] * (iteration + 1));
        return std::chrono::duration<double, std::milli>(end - begin).count();
    };

    const double cold = call(0);
    cudaMemPool_t private_pool = existing_pool();
    std::uint64_t reserved_before = 0;
    if (pooled) {
        assert(private_pool && private_pool != application_pool);
        reserved_before = attribute(private_pool, cudaMemPoolAttrReservedMemCurrent);
        assert(reserved_before > 0);
        assert(attribute(private_pool, cudaMemPoolAttrUsedMemCurrent) == 0);
    } else {
        assert(!private_pool);
    }
    double warm = 0;
    for (int i = 1; i <= repetitions; ++i) warm += call(i);
    const auto reserved_after = pooled ? attribute(private_pool, cudaMemPoolAttrReservedMemCurrent) : 0;
    if (pooled) {
        assert(attribute(private_pool, cudaMemPoolAttrUsedMemCurrent) == 0);
        assert(reserved_after == reserved_before);
    }
    cudaMemPool_t still_application_pool = nullptr;
    CUCH(cudaDeviceGetMemPool(&still_application_pool, 0));
    assert(still_application_pool == application_pool);
    cpp_entry_trim_cache();
    assert(!existing_pool());
    // The documented reset boundary first releases the compiler's private pool.
    CUCH(cudaDeviceReset());
    CUCH(cudaSetDevice(0));
    call(repetitions + 1);
    if (pooled) assert(existing_pool());
    cpp_entry_trim_cache();
    assert(!existing_pool());
    std::cout << std::setprecision(12)
              << "{\"pooled\":" << (pooled ? "true" : "false")
              << ",\"cold_ms\":" << cold << ",\"warm_mean_ms\":" << warm / repetitions
              << ",\"reserved_before\":" << reserved_before
              << ",\"reserved_after\":" << reserved_after << "}\n";
#endif
}
"""


@pytest.fixture(scope="module", params=["legacy", "per-thread"])
def native_pool_binary(request, tmp_path_factory):
    directory = tmp_path_factory.mktemp("native_pools_" + request.param)
    source = directory / "source.f90"
    source.write_text(SOURCE)
    function, plan = prepare_function(lower_file(source, "entry"))
    generated = generate_sources(function, plan)
    (directory / "generated_code.cu").write_text(generated.cuda)
    (directory / "common_functions.cuh").write_text(read_common_header())
    (directory / "driver.cu").write_text(DRIVER)
    executable = directory / "run"
    _run(
        [_tool("nvcc"), "-std=c++17", "--default-stream", request.param, "driver.cu", "-o", str(executable)],
        directory,
    )
    return executable


@pytest.mark.cuda
def test_pool_runtime_compiles_in_both_default_stream_modes(native_pool_binary):
    assert native_pool_binary.stat().st_size > 0


@pytest.mark.cuda
@pytest.mark.parametrize("budget", ["0", "268435456"], ids=["dedicated", "pooled"])
def test_native_backing_reuse_and_reset(native_pool_binary, budget):
    result = _run(
        [str(native_pool_binary)],
        native_pool_binary.parent,
        check=False,
        env={**os.environ, "FORT_CUDA_POOL_BYTES": budget, "FORT_RUNTIME_TRACE": "1"},
    )
    if result.returncode == 77:
        pytest.skip(result.stderr.strip())
    assert result.returncode == 0, result.stdout + result.stderr
    measurements = json.loads(result.stdout)
    assert measurements["pooled"] == (budget != "0")
    # One cold run, eight warm runs, and one run after reset; every run transfers.
    operations = re.findall(r"^FORT_RUNTIME (\w+)", result.stderr, re.MULTILINE)
    assert operations.count("upload") == 20
    assert operations.count("download") == 10
    assert operations.count("kernel") == 10
    (native_pool_binary.parent / f"measurements-{budget}.json").write_text(result.stdout)
