"""Compile the real line provider against a nonexecuting runtime boundary."""

import ctypes as c
import shutil

import pytest

from compiler.emission.cuda.tile_task_provider import emit_tile_task_provider
from compiler.frontend.source_device_tasks import source_device_task
from compiler.tests.test_line_snapshot import source
from compiler.tests.test_line_source_tasks import source_builder
from compiler.tests.test_tile_task_provider import RUNTIME, TEMPLATES, Report, checked


@pytest.fixture(scope="module", params=[1, 2, 3])
def compiled(request, tmp_path_factory):
    nvcc, host = shutil.which("nvcc"), shutil.which("g++-14")
    if not nvcc or not host:
        pytest.skip("CUDA compilation toolkit required; no GPU executes")
    directory = tmp_path_factory.mktemp(f"line-provider-{request.param}")
    axis = request.param
    builder, nodes = source_builder(directory, source(axis))
    issued = source_device_task(builder.analysis, builder.entry, nodes, builder=builder)
    provider = emit_tile_task_provider(issued.team, mapping="thread", access_envelopes=True, readable_scalars=issued)
    (directory / "provider.cu").write_text(provider.cuda_source)
    (directory / "runtime.cpp").write_text(
        r"""
#include "scoped_runtime.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
static int mode=0,calls[5]={};
static std::size_t extent[3]={13,11,9};
static std::int64_t lower[3]={-2,-2,-2};
extern "C" void reset(int value){mode=value;for(auto&v:calls)v=0;extent[0]=13;extent[1]=11;extent[2]=9;}
extern "C" int counter(int i){return calls[i];}
extern "C" int fort_scope_serial_caller(){return 1;}
extern "C" int fort_scope_numerical_environment_supported(){return 1;}
extern "C" int fort_scope_layout_get(fort_scope_t,fort_buffer_t field,fort_scope_layout* out){
 ++calls[0];if(mode==1)extent[AXIS]=1;
 *out={3,field==11?FORT_SCOPE_REAL64:FORT_SCOPE_INTEGER32,field==11?8u:4u,nullptr,extent,lower,1};
 if(mode==2&&field==12)out->type=FORT_SCOPE_REAL64;
 return 0;
}
extern "C" int fort_scope_wait(fort_scope_t){++calls[1];return 0;}
extern "C" int fort_scope_scratch_acquire_v1(fort_scope_t,std::size_t n,fort_scope_scratch_lease_v1* lease){
 ++calls[2];if(mode==3)return FORT_SCOPE_RESOURCE;
 *lease={1,0,1,reinterpret_cast<void*>(4096),n,n};return 0;
}
extern "C" int fort_scope_scratch_release_v1(fort_scope_t,std::uint64_t){++calls[4];return mode==7?FORT_SCOPE_EXECUTION:0;}
extern "C" int fort_scope_source_execute_v1(fort_scope_t,const fort_scope_external_call_v1* call,
 std::uint64_t expected,fort_scope_source_worker_v1,void*,fort_scope_external_report_v1* report){
 ++calls[3];assert(expected==1&&call->binding_count==2);
 const auto&a=call->bindings[0].access;const auto&m=call->bindings[1].access;
 assert(a.flags==0&&m.flags==0&&a.read_count==1&&a.write_count==1&&m.read_count==2&&m.write_count==0);
 assert(a.overwrite_count==0&&m.overwrite_count==0);
 for(int d=0;d<3;++d){
  assert(a.reads[0].lower[d]==std::size_t(d==AXIS?0:3));
  assert(a.reads[0].upper[d]==(d==AXIS?extent[d]:extent[d]-3));
  assert(a.writes[0].lower[d]==3&&a.writes[0].upper[d]==extent[d]-3);
 }
 const int status=mode==4?FORT_SCOPE_UNINITIALIZED:mode==5?FORT_SCOPE_ALIAS:mode==6?FORT_SCOPE_EXECUTION:FORT_SCOPE_RESOURCE;
 *report={1,status,mode==6?1u:0u,0};return status;
}
""".replace("AXIS", str(axis - 1))
    )
    checked(
        [
            nvcc,
            "-std=c++17",
            "-ccbin",
            host,
            "-arch=sm_86",
            "--fmad=false",
            "--ftz=false",
            "-shared",
            "-Xcompiler=-fPIC",
            "-I",
            str(RUNTIME),
            "-I",
            str(TEMPLATES),
            str(directory / "provider.cu"),
            str(directory / "runtime.cpp"),
            "-o",
            str(directory / "provider.so"),
        ],
        directory,
        "compile",
    )
    library = c.CDLL(str(directory / "provider.so"))
    function = getattr(library, provider.entry)
    function.argtypes = [
        c.c_int64,
        c.c_int64,
        c.c_int64,
        *[c.POINTER(c.c_int) for _ in provider.parameters],
        c.POINTER(Report),
    ]
    function.restype = c.c_int
    return axis, provider, library, function


def invoke(compiled, mode=0, overrides=None):
    axis, provider, library, function = compiled
    values = {("scalar", name, None): value for name, value in zip(("nx", "ny", "nz"), (7, 5, 3), strict=True)}
    for field in ("a", "mask"):
        for dim in (1, 2, 3):
            values["external_lower", field, dim] = -2
    values["scratch_lower", "tmp", 1] = -2
    values["scratch_upper", "tmp", 1] = (7, 5, 3)[axis - 1] + 3
    values.update({} if overrides is None else overrides)
    arguments = [
        None
        if values.get((p.role, p.binding.name, p.axis)) is None
        else c.c_int(values[p.role, p.binding.name, p.axis])
        for p in provider.parameters
    ]
    library.reset(mode)
    report = Report()
    status = function(1, 11, 12, *[None if x is None else c.byref(x) for x in arguments], c.byref(report))
    return status, report, [library.counter(i) for i in range(5)]


def test_exact_source_producer_reads_and_partial_writes_reach_existing_transaction(compiled):
    status, report, calls = invoke(compiled)
    assert status == report.status == 4
    assert report.started == report.committed == 0
    assert calls == [2, 1, 1, 1, 1]


@pytest.mark.parametrize("mode", [1, 2])
def test_bad_layout_declines_before_runtime_payload_or_scratch(compiled, mode):
    status, report, calls = invoke(compiled, mode)
    assert status == report.status == 1
    assert report.started == report.committed == 0
    assert calls == [2, 0, 0, 0, 0]


@pytest.mark.parametrize(("key", "value"), [("lower", -1), ("upper", 0)])
def test_nonconforming_original_scratch_rejects_before_layout_reads(compiled, key, value):
    status, report, calls = invoke(compiled, overrides={("scratch_" + key, "tmp", 1): value})
    assert status == report.status == 1
    assert report.started == report.committed == 0
    # A positive but short scratch needs the source descriptor to detect shape mismatch.
    assert calls == ([0, 0, 0, 0, 0] if key == "lower" else [2, 0, 0, 0, 0])


def test_outer_zero_trip_never_reads_later_axis_or_descriptor_values(compiled):
    axis, provider, _library, _function = compiled
    transverse = {1: ("nz", "ny"), 2: ("nz", "nx"), 3: ("ny", "nx")}[axis]
    missing = {(p.role, p.binding.name, p.axis): None for p in provider.parameters}
    missing["scalar", transverse[0], None] = 0
    status, report, calls = invoke(compiled, overrides=missing)
    assert status == report.status == 0
    assert report.started == report.committed == 0
    assert calls == [0, 0, 0, 0, 0]


@pytest.mark.parametrize(
    ("mode", "status", "started", "calls"),
    [
        (3, 4, 0, [2, 1, 1, 0, 0]),
        (4, 7, 0, [2, 1, 1, 1, 1]),
        (5, 3, 0, [2, 1, 1, 1, 1]),
        (6, 6, 1, [2, 1, 1, 1, 1]),
        (7, 6, 1, [2, 1, 1, 1, 1]),
    ],
)
def test_runtime_admission_and_uncertain_failure_reports_are_preserved(compiled, mode, status, started, calls):
    actual, report, actual_calls = invoke(compiled, mode)
    assert actual == report.status == status
    assert report.started == started
    assert report.committed == 0
    assert actual_calls == calls
