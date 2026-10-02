"""Ordinary CUDA calls own independent state, even with concurrent callers."""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from compiler.driver.pipeline import prepare_function
from compiler.emission import generate_sources, read_common_header
from compiler.frontend import lower_file
from compiler.tests.test_language_cuda import CUDA_RUNTIME
from compiler.tests.test_sessions import _simulate_cuda


def counted_runtime():
    runtime = CUDA_RUNTIME.replace("#include <cstdlib>", "#include <cstdlib>\n#include <atomic>")
    runtime = runtime.replace("inline Dimension blockIdx", "inline thread_local Dimension blockIdx")
    runtime = runtime.replace(
        "inline int cudaMalloc",
        "inline std::atomic<int> fort_test_allocations{0}, fort_test_frees{0}, fort_test_uploads{0}, "
        "fort_test_downloads{0}, fort_test_syncs{0}, fort_test_events{0};\ninline int cudaMalloc",
    )
    runtime = runtime.replace("*p = std::malloc(n);", "++fort_test_allocations; *p = std::malloc(n);")
    runtime = runtime.replace("std::free(p);", "++fort_test_frees; std::free(p);")
    runtime = runtime.replace("std::size_t n, int)", "std::size_t n, int direction)")
    runtime = runtime.replace(
        "std::memcpy(to, from, n);",
        "++(direction == cudaMemcpyHostToDevice ? fort_test_uploads : fort_test_downloads); std::memcpy(to, from, n);",
    )
    runtime = runtime.replace(
        "cudaDeviceSynchronize() { return 0; }", "cudaDeviceSynchronize() { ++fort_test_syncs; return 0; }"
    )
    runtime = runtime.replace("*event = new float(0);", "++fort_test_events; *event = new float(0);")
    return runtime


def compile_simulated(tmp_path, source, driver, *, sanitize=False):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("Native capability unavailable: g++")
    env = {**os.environ, "ASAN_OPTIONS": "detect_leaks=0"}
    if sanitize:
        (tmp_path / "probe.cpp").write_text("int main() {}\n")
        probe = subprocess.run(
            [compiler, "-fsanitize=address", "probe.cpp", "-o", "probe"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode:
            pytest.skip("AddressSanitizer is unavailable: " + probe.stderr)
        probe = subprocess.run(["./probe"], cwd=tmp_path, env=env, capture_output=True, text=True, check=False)
        if probe.returncode:
            pytest.skip("AddressSanitizer execution is unavailable: " + probe.stderr)
    path = tmp_path / "source.f90"
    path.write_text(source)
    function, plan = prepare_function(lower_file(path, "entry"))
    generated = generate_sources(function, plan)
    (tmp_path / "common_functions.cuh").write_text(read_common_header())
    (tmp_path / "cuda_runtime.h").write_text(counted_runtime())
    (tmp_path / "test.cpp").write_text(_simulate_cuda(generated.cuda) + "\n" + driver)
    flags = ["-fsanitize=address", "-g"] if sanitize else []
    build = subprocess.run(
        [compiler, "-std=c++17", "-O1", "-pthread", *flags, "test.cpp", "-o", "run"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert build.returncode == 0, build.stderr
    run = subprocess.run(["./run"], cwd=tmp_path, env=env, capture_output=True, text=True, check=False, timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr
    return generated.cuda, run


@pytest.mark.native
@pytest.mark.parametrize("parallel", [False, True], ids=["host", "kernel"])
@pytest.mark.parametrize("sanitize", [False, True], ids=["ordinary", "asan"])
def test_concurrent_ordinary_calls_have_independent_storage(tmp_path, parallel, sanitize):
    body = "do i=1,n\na(i)=a(i)+delta\nenddo" if parallel else "a(1)=a(1)+delta"
    source = f"""! kernels
module call_case
contains
! kernel
subroutine entry(a,n,delta)
integer,intent(inout)::a(:)
integer,intent(in)::n,delta
integer::i
{body}
end subroutine
end module
"""
    driver = """#include <cassert>
#include <thread>
#include <vector>
int main() {
    constexpr int threads = 8, repeats = 400;
    std::vector<std::thread> workers;
    for (int t = 0; t < threads; ++t) workers.emplace_back([t]() {
        for (int repeat = 0; repeat < repeats; ++repeat) {
            int a[] = {t, repeat, 7, -3};
            const int delta = repeat % 5 + 1;
            generated_kernels::cpp_entry(a, 4, 4, delta);
            assert(a[0] == t + delta);
            assert(a[1] == repeat + EXPECTED_DELTA);
            assert(a[2] == 7 + EXPECTED_DELTA);
            assert(a[3] == -3 + EXPECTED_DELTA);
        }
    });
    for (auto& worker : workers) worker.join();
    assert(fort_test_allocations == threads * repeats);
    assert(fort_test_frees == threads * repeats);
    assert(fort_test_uploads == threads * repeats);
    assert(fort_test_downloads == threads * repeats);
    assert(fort_test_events == 0);
    assert(fort_test_syncs == 2 * threads * repeats);
}
""".replace("EXPECTED_DELTA", "delta" if parallel else "0")
    generated, _ = compile_simulated(tmp_path, source, driver, sanitize=sanitize)
    ordinary = generated.split('extern "C" void cpp_entry(')[1]
    assert "fort_internal_token" not in ordinary
    assert "_registry" not in ordinary


@pytest.mark.native
def test_scalar_only_ordinary_call_completes_without_session_tokens(tmp_path):
    source = """! kernels
module call_case
contains
! kernel
subroutine entry(n)
integer,intent(in)::n
end subroutine
end module
"""
    driver = """#include <cassert>
int main() {
    generated_kernels::cpp_entry(0);
    assert(fort_test_allocations == 0);
    assert(fort_test_frees == 0);
    assert(fort_test_syncs == 2);
    assert(fort_test_events == 0);
}
"""
    compile_simulated(tmp_path, source, driver)
