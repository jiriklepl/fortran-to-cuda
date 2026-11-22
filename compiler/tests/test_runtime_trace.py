"""Concurrent runtime and placement diagnostics retain whole line records."""

import os
import shutil
import subprocess
from collections import Counter

import pytest

from compiler.emission.common.resources import read_common_header


@pytest.mark.native
def test_concurrent_storage_and_placement_records(tmp_path):
    cxx = shutil.which("g++")
    if not cxx:
        pytest.skip("C++ compiler required")
    (tmp_path / "common.hpp").write_text(read_common_header())
    (tmp_path / "trace.cpp").write_text(r'''
#define FORT_OFFLOAD_ENABLED
#include "common.hpp"
#include <atomic>
#include <thread>
int main() {
    using namespace generated_kernels;
    std::atomic<bool> ready{false};
    std::vector<std::thread> threads;
    for (int thread = 0; thread < 8; ++thread) {
        threads.emplace_back([&, thread] {
            while (!ready.load()) std::this_thread::yield();
            for (int i = 0; i < 2000; ++i) {
                if (thread % 2) {
                    storage::trace("pool_alloc", 34881600);
                    storage::trace("release");
                } else {
                    offload::decision_trace_single("sample", "native", 0, 3);
                }
            }
        });
    }
    ready.store(true);
    for (auto &thread : threads) thread.join();
}
''')
    binary = tmp_path / "trace"
    build = subprocess.run([cxx, "-std=c++17", "-O2", "-pthread", "trace.cpp", "-o", str(binary)],
                           cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert build.returncode == 0, build.stdout + build.stderr
    expected = Counter({
        "FORT_RUNTIME pool_alloc bytes=34881600": 8000,
        "FORT_RUNTIME release": 8000,
        "FORT_OFFLOAD entry=sample mode=native gpu_units=0 cpu_units=3 unit_kind=regions": 8000,
    })
    for enabled in ("1", "0"):
        result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30,
                                env={**os.environ, "FORT_RUNTIME_TRACE": enabled, "FORT_OFFLOAD_TRACE": enabled})
        assert result.returncode == 0, result.stderr
        assert not result.stdout
        assert Counter(result.stderr.splitlines()) == (expected if enabled == "1" else Counter())
