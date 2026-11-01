"""Finite opt-in batching variants use real host preflight and no GPU execution."""
import ctypes as c
import shutil

import pytest

from compiler.emission.cuda.tile_task_provider import EXPERIMENTAL_BATCH_TARGET_BYTES, emit_tile_task_provider
from compiler.frontend.source_device_tasks import source_device_task
from compiler.ir import CompilationError
from compiler.tests.test_tile_private_team import team_fixture
from compiler.tests.test_tile_task_provider import RUNTIME, TEMPLATES, Report, checked


def issued(tmp_path):
    _, analysis, routine, nodes = team_fixture(tmp_path)
    return source_device_task(analysis, routine, nodes)


@pytest.mark.parametrize("target", [True, False, 4.0, "4194304", 0, -1, 1, 8 * 1024 * 1024, 128 * 1024 * 1024])
def test_only_exact_finite_experiment_targets_are_accepted(tmp_path, target):
    proof = issued(tmp_path)
    with pytest.raises(CompilationError, match="exactly 4, 16 or 64 MiB"):
        emit_tile_task_provider(proof.team, access_envelopes=True, batch_target_bytes=target)


def test_experiment_target_cannot_enter_unqualified_or_priced_protocol(tmp_path):
    proof = issued(tmp_path)
    with pytest.raises(CompilationError, match="requires unpriced access envelopes"):
        emit_tile_task_provider(proof.team, batch_target_bytes=4194304)
    for extra in ({"pricing": object()}, {"placement": object()}):
        with pytest.raises(CompilationError, match="management costs are unpriced"):
            emit_tile_task_provider(proof.team, access_envelopes=True, batch_target_bytes=4194304, **extra)


@pytest.mark.parametrize("mapping", ["cooperative", "thread"])
def test_target_namespace_is_unique_and_changes_only_batch_plan_and_symbols(tmp_path, mapping):
    proof = issued(tmp_path)
    default = emit_tile_task_provider(proof.team, readable_scalars=proof, access_envelopes=True, mapping=mapping)
    none = emit_tile_task_provider(proof.team, readable_scalars=proof, access_envelopes=True, mapping=mapping,
                                   batch_target_bytes=None)
    assert default == none
    assert default.public()["version"] == 7
    assert default.public()["batching"]["version"] == 1
    names = set()
    for target in EXPERIMENTAL_BATCH_TARGET_BYTES:
        variant = emit_tile_task_provider(proof.team, readable_scalars=proof, access_envelopes=True, mapping=mapping,
                                         batch_target_bytes=target)
        public = variant.public()
        assert public["version"] == 8
        assert public["batching"] == {"version": 2, "target_bytes": target, "minimum_tiles": 1,
                                      "planned_launches": "sum ceil(unit_tiles/batch_tiles)"}
        assert f"_b{target // 1048576}m_boxes" in variant.entry
        assert len(variant.entry) <= 63
        assert len(variant.module) <= 63
        names.update((variant.entry, variant.module))
        # All lowered arithmetic, boxes, failures and ordered submissions remain exact.
        restored = variant.cuda_source.replace(variant.entry, default.entry).replace(
            f"(1,{target}/max_stride)", "(1,4194304/max_stride)")
        assert restored == default.cuda_source
        assert variant.fortran_source.replace(variant.entry, default.entry).replace(
            variant.module, default.module) == default.fortran_source
        assert public["parameters"] == default.public()["parameters"]
        assert public["access_envelopes"] == default.public()["access_envelopes"]
    assert len(names) == 6


@pytest.fixture(scope="module")
def compiled_targets(tmp_path_factory):
    nvcc, host, fc = shutil.which("nvcc"), shutil.which("g++-14"), shutil.which("gfortran-15")
    if not nvcc or not host or not fc:
        pytest.skip("CUDA and GNU compilation toolchains required; no device used")
    directory = tmp_path_factory.mktemp("finite-batch-targets")
    proof = issued(directory)
    providers = [emit_tile_task_provider(proof.team, readable_scalars=proof, access_envelopes=True,
                                       batch_target_bytes=target) for target in EXPERIMENTAL_BATCH_TARGET_BYTES]
    sources = []
    for provider in providers:
        path = directory / (provider.entry + ".cu")
        path.write_text(provider.cuda_source)
        sources.append(str(path))
        (directory / (provider.module + ".f90")).write_text(provider.fortran_source)
    (directory / "runtime.cpp").write_text(r'''
#include "scoped_runtime.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
static int mode=0,calls[5]={};static std::size_t requested=0;static std::uint64_t expected=0;
extern "C" void reset(int value){mode=value;for(auto&x:calls)x=0;requested=0;expected=0;}
extern "C" int counter(int i){return calls[i];}
extern "C" std::size_t scratch(){return requested;}
extern "C" std::uint64_t launches(){return expected;}
extern "C" int fort_scope_serial_caller(){return 1;}
extern "C" int fort_scope_numerical_environment_supported(){return 1;}
extern "C" int fort_scope_layout_get(fort_scope_t,fort_buffer_t,fort_scope_layout* p){
 ++calls[0];static std::size_t extent=870;static std::int64_t lower=-3;
 extent=mode==2?2:870;*p={1,FORT_SCOPE_REAL64,8,nullptr,&extent,&lower,1};return 0;
}
extern "C" int fort_scope_wait(fort_scope_t){++calls[1];return 0;}
extern "C" int fort_scope_scratch_acquire_v1(fort_scope_t,std::size_t n,fort_scope_scratch_lease_v1* p){
 ++calls[2];requested=n;if(mode==1)return FORT_SCOPE_RESOURCE;
 *p={1,0,1,reinterpret_cast<void*>(4096),n,n};return 0;
}
extern "C" int fort_scope_scratch_release_v1(fort_scope_t,std::uint64_t){++calls[4];return 0;}
extern "C" int fort_scope_source_execute_v1(fort_scope_t,const fort_scope_external_call_v1* call,
 std::uint64_t n,fort_scope_source_worker_v1,void*,fort_scope_external_report_v1* report){
 ++calls[3];expected=n;assert(call->binding_count==2&&call->workspace_bytes==0&&call->workspace_lease==0);
 for(const auto* p=call->bindings;p!=call->bindings+2;++p){assert(p->access.flags==0&&p->access.overwrite_count==0);}
 *report={1,FORT_SCOPE_RESOURCE,0,0};return FORT_SCOPE_RESOURCE;
}
''')
    checked([nvcc, "-std=c++17", "-ccbin", host, "-arch=sm_86", "--fmad=false", "--ftz=false",
             "-shared", "-Xcompiler=-fPIC", "-I", str(RUNTIME), "-I", str(TEMPLATES), *sources,
             str(directory / "runtime.cpp"), "-o", str(directory / "providers.so")], directory, "cuda-three-target-link")
    checked([fc, "-cpp", "-ffree-line-length-none", "-J", str(directory), "-c", str(RUNTIME / "scoped_memory.f90"),
             "-o", str(directory / "memory.o")], directory, "fortran-runtime")
    for provider in providers:
        checked([fc, "-I", str(directory), "-J", str(directory), "-c", str(directory / (provider.module + ".f90")),
                 "-o", str(directory / (provider.module + ".o"))], directory, provider.module)
    lib = c.CDLL(str(directory / "providers.so"))
    lib.scratch.restype = lib.launches.restype = c.c_uint64
    return providers, lib


def invoke(compiled, index, *, mode=0, n=864, upper=9718):
    providers, lib = compiled
    provider = providers[index]
    lib.reset(mode)
    values = []
    for p in provider.parameters:
        value = ({"n": n, "t": 3}[p.binding.name] if p.role == "scalar" else
                 -3 if p.role == "external_lower" else -1 if p.role == "scratch_lower" else upper)
        values.append(c.c_int(value))
    function = getattr(lib, provider.entry)
    function.argtypes = [c.c_int64, c.c_int64, c.c_int64,
                        *[c.POINTER(c.c_int) for _ in provider.parameters], c.POINTER(Report)]
    function.restype = c.c_int
    report = Report()
    status = function(1, 11, 12, *[c.byref(v) for v in values], c.byref(report))
    return status, report, [lib.counter(i) for i in range(5)], lib.scratch(), lib.launches()


@pytest.mark.parametrize("index", range(3))
def test_compiled_real_planner_preserves_tail_and_minimum_one_tile(compiled_targets, index):
    target = EXPERIMENTAL_BATCH_TARGET_BYTES[index]
    for upper in (9718, target // 8 + 1):
        status, report, calls, scratch, launches = invoke(compiled_targets, index, upper=upper)
        assert status == report.status == 4
        assert report.started == report.committed == 0
        assert calls == [2, 1, 1, 1, 1]
        stride = ((upper + 2) * 8 + 7) & ~7
        batch = min(288, max(1, target // stride))
        assert scratch == stride * batch
        assert launches == (288 + batch - 1) // batch
        if upper == 9718:
            assert launches == (6, 2, 1)[index]
        else:
            assert batch == 1
            assert scratch > target


@pytest.mark.parametrize("index", range(3))
def test_compiled_zero_and_bad_layout_never_acquire_payload_or_scratch(compiled_targets, index):
    status, report, calls, scratch, launches = invoke(compiled_targets, index, n=0)
    assert status == report.status == 0
    assert calls == [0, 0, 0, 0, 0]
    assert scratch == launches == 0
    status, report, calls, scratch, launches = invoke(compiled_targets, index, mode=2)
    assert status == report.status == 1
    assert report.started == report.committed == 0
    assert calls == [2, 0, 0, 0, 0]
    assert scratch == launches == 0


@pytest.mark.parametrize("index", range(3))
def test_compiled_budget_rejection_precedes_source_transaction(compiled_targets, index):
    status, report, calls, _scratch, launches = invoke(compiled_targets, index, mode=1)
    assert status == report.status == 4
    assert report.started == report.committed == 0
    assert calls == [2, 1, 1, 0, 0]
    assert launches == 0
