"""Check memory plans, owned buffers, coherence, and opt-in profiling."""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from compiler.emission.common.resources import read_common_header


def _tool(name):
    result = shutil.which(name)
    if not result:
        pytest.skip(f"Native capability unavailable: {name}")
    return result


def _run(arguments, directory, *, check=True, env=None):
    result = subprocess.run(arguments, cwd=directory, text=True, capture_output=True, timeout=60, env=env)
    if check:
        assert result.returncode == 0, result.stdout + result.stderr
    return result


FAKE_CUDA = r"""
#define __CUDACC__
#define __host__
#define __device__
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <string>
using cudaError_t = int;
struct Event {};
using cudaEvent_t = Event*;
constexpr int cudaSuccess = 0, cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2;
constexpr int cudaHostRegisterPortable = 1;
static int allocations = 0, uploads = 0, downloads = 0, frees = 0, syncs = 0, events = 0;
static int registrations = 0, unregistrations = 0;
const char* cudaGetErrorString(int) { return "fake CUDA failure"; }
int cudaMalloc(void** pointer, std::size_t bytes) { ++allocations; *pointer = std::malloc(bytes); return 0; }
int cudaFree(void* pointer) { ++frees; std::free(pointer); return 0; }
int cudaMemcpy(void* destination, const void* source, std::size_t bytes, int direction) {
    if (direction == cudaMemcpyHostToDevice) ++uploads; else ++downloads;
    std::memcpy(destination, source, bytes); return 0;
}
int cudaHostRegister(void*, std::size_t, unsigned int) { ++registrations; return 0; }
int cudaHostUnregister(void*) { ++unregistrations; return 0; }
int cudaDeviceSynchronize() { ++syncs; return 0; }
int cudaEventCreate(Event** event) { ++events; *event = new Event; return 0; }
int cudaEventDestroy(Event* event) { delete event; return 0; }
int cudaEventRecord(Event*, int) { return 0; }
int cudaEventSynchronize(Event*) { return 0; }
int cudaEventElapsedTime(float* time, Event*, Event*) { *time = 1.0f; return 0; }
#include "common_functions.cuh"
using namespace generated_kernels;
struct State { storage::Buffer<double> data; State() : data({4}) {} };
int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    storage::Registry<State> registry;
    auto first = registry.create();
    auto second = registry.create();
    auto& a = registry.get(first).data;
    auto& b = registry.get(second).data;
    double input[] = {1, 2, 3, 4}, output[4] = {};
    a.update_device(input, {4});
    b.update_device(input, {4});
    input[0] = 99; // No caller address is retained.
    a.device_data()[0] += 10;
    a.device_written();
    a.device_data()[0] += 10;
    a.device_written();
    assert(allocations == 2 && uploads == 2 && downloads == 0);
    assert(events == 0 && syncs == 0); // Default execution has no profiling barriers.
    b.update_device(input, {4});
    assert(a.device_data()[0] == 21); // An unrelated explicit update preserves device-newer a.
    a.host_data()[1] = 17; // Partial host write must preserve device-produced a[0].
    a.host_written();
    assert(downloads == 1);
    assert(a.device_data()[0] == 21 && a.device_data()[1] == 17);
    assert(uploads == 4);
    a.device_data()[2] = 30;
    a.device_written();
    a.update_host(output, {4});
    assert(output[0] == 21 && output[1] == 17 && output[2] == 30 && output[3] == 4);
    assert(downloads == 2);
    if (mode == "shape") a.update_device(input, {2, 2});
    if (mode == "overflow") { storage::Buffer<double> huge({std::numeric_limits<std::size_t>::max(), 2}); }
    if (mode == "bytes") { storage::Buffer<double> huge({std::numeric_limits<std::size_t>::max()}); }
    timing::reset_timing_vectors();
    assert(syncs == 1);
    timing::measure_kernel_executions([]() {});
    assert(events == 2);
    timing::print_timing_summary();
    const int previous_events = events, previous_syncs = syncs;
    timing::measure_kernel_executions([]() {});
    assert(events == previous_events && syncs == previous_syncs);
    registry.destroy(first);
    if (mode == "stale") registry.get(first);
    registry.destroy(0);
    assert(registry.create() > second); // Tokens are never reused.
    registry.destroy(second);
    { storage::Buffer<double> empty({std::numeric_limits<std::size_t>::max(), 0});
      empty.update_device(nullptr, {std::numeric_limits<std::size_t>::max(), 0});
      empty.update_host(nullptr, {std::numeric_limits<std::size_t>::max(), 0}); }
#ifdef USE_PINNED_MEMORY
    assert(registrations == uploads + downloads);
    assert(unregistrations == registrations);
#else
    assert(registrations == 0);
#endif
    assert(frees == 2);
}
"""


@pytest.fixture(scope="module", params=[False, True], ids=["pageable", "pinned"])
def fake_runtime(tmp_path_factory, request):
    directory = tmp_path_factory.mktemp("fake_runtime")
    (directory / "common_functions.cuh").write_text(read_common_header())
    (directory / "runtime.cpp").write_text(FAKE_CUDA)
    flags = ["-DUSE_PINNED_MEMORY"] if request.param else []
    _run([_tool("g++"), "-std=c++17", *flags, "runtime.cpp", "-o", "run"], directory)
    return directory


@pytest.mark.native
def test_runtime_coherence_and_opt_in_profiling(fake_runtime):
    result = _run(["./run"], fake_runtime, env={**os.environ, "FORT_RUNTIME_TRACE": "1"})
    assert "calls:              1" in result.stdout
    assert result.stderr.count("FORT_RUNTIME upload") == 4
    assert result.stderr.count("FORT_RUNTIME download") == 2


@pytest.mark.native
@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("shape", "shape does not match"),
        ("stale", "stale workspace"),
        ("overflow", "extent product overflow"),
        ("bytes", "byte size overflow"),
    ],
)
def test_runtime_diagnoses_invalid_lifecycle_and_shapes(fake_runtime, mode, message):
    result = _run(["./run", mode], fake_runtime, check=False)
    assert result.returncode != 0
    assert message in result.stderr


def test_memory_lifecycle_preserves_parameters_after_skipped_intents():
    from compiler.ir import ExecutionPlan, ScalarType, Symbol
    from compiler.memory import plan_memory

    output = Symbol(0, "output", ScalarType.REAL, rank=1, intent="out", parameter=True)
    source = Symbol(1, "source", ScalarType.REAL, rank=1, intent="in", parameter=True)
    updated = Symbol(2, "updated", ScalarType.REAL, rank=1, intent="inout", parameter=True)
    for symbols in ((output, source, updated), (source, output, updated)):
        memory = plan_memory(ExecutionPlan(()), symbols)
        assert next(op.symbols for op in memory.create if op.kind == "device") == (source, updated)
        assert set(next(op.symbols for op in memory.retrieve if op.kind == "host")) == {output, updated}


def test_memory_plan_preserves_predicate_and_launch_bound_reads(tmp_path):
    from compiler.analysis import build_execution_plan
    from compiler.ir import ConditionalRegion, ParallelRegion
    from compiler.memory import format_memory, plan_memory
    from compiler.tests.test_language import lower

    function = lower(
        tmp_path,
        "if(a(1)>0)then\ndo i=1,limit(1)\na(i)=a(i)+1\nenddo\nelse\na(2)=0\nendif",
        "integer,intent(in)::limit(:)",
        "a,n,limit",
    )
    memory = plan_memory(build_execution_plan(function), function.parameters)
    predicate, branch = memory.run
    assert predicate.kind == "host"
    assert predicate.symbols == (function.parameters[0],)
    assert branch.kind == "branch"
    assert isinstance(branch.step, ConditionalRegion)
    bound_read = branch.then_ops[0]
    assert bound_read.kind == "host"
    assert bound_read.symbols == (function.parameters[2],)
    assert any(isinstance(operation.step, ParallelRegion) for operation in branch.then_ops)
    assert tuple(operation.kind for operation in branch.else_ops) == ("host", "execute", "host_write")
    assert branch.else_ops[0].symbols == predicate.symbols
    assert memory.retrieve[0].kind == "sync"
    assert tuple(operation.kind for operation in memory.destroy) == ("sync", "release")
    report = format_memory(memory)
    assert "branch:" in report
    assert "then:" in report
    assert "else:" in report


@pytest.mark.native
def test_cpu_storage_owns_buffers_and_keeps_independent_updates(tmp_path):
    (tmp_path / "common_functions.cuh").write_text(read_common_header())
    (tmp_path / "runtime.cpp").write_text(r"""
#include <cassert>
#include "common_functions.cuh"
using namespace generated_kernels;
struct State { storage::Buffer<int> data; State() : data({3}) {} };
int main() {
    storage::Registry<State> registry;
    const auto first = registry.create(), second = registry.create();
    const auto alias = first;
    int input[] = {1, 2, 3}, output[3] = {};
    auto& a = registry.get(first).data;
    auto& b = registry.get(second).data;
    a.update_device(input, {3});
    b.update_device(input, {3});
    input[0] = 99;
    registry.get(alias).data.host_data()[1] = 20;
    a.host_written();
    a.device_data()[2] = 30;
    a.device_written();
    b.update_device(input, {3});
    a.update_host(output, {3});
    assert(output[0] == 1 && output[1] == 20 && output[2] == 30);
    b.update_host(output, {3});
    assert(output[0] == 99 && output[1] == 2 && output[2] == 3);
    registry.destroy(alias);
    registry.destroy(second);
    registry.destroy(0);
    storage::Buffer<int> empty({std::numeric_limits<std::size_t>::max(), 0});
    empty.update_device(nullptr, {std::numeric_limits<std::size_t>::max(), 0});
    empty.update_host(nullptr, {std::numeric_limits<std::size_t>::max(), 0});
}
""")
    _run([_tool("g++"), "-std=c++17", "runtime.cpp", "-o", "run"], tmp_path)
    _run(["./run"], tmp_path)
