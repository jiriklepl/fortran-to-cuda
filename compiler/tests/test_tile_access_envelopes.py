"""Typed original-index boxes, compiled with a nonexecuting runtime double."""

import ctypes as c
import json
import shutil

import pytest

from compiler.emission.cuda.tile_task_provider import emit_tile_task_provider
from compiler.frontend.source_device_tasks import source_device_task
from compiler.ir import CompilationError
from compiler.tests.test_tile_private import SOURCE, SOURCE_3D
from compiler.tests.test_tile_private_team import team_fixture
from compiler.tests.test_tile_task_provider import RUNTIME, TEMPLATES, Report, checked


@pytest.fixture(scope="module", params=[1, 3])
def metadata_provider(request, tmp_path_factory):
    nvcc, host = shutil.which("nvcc"), shutil.which("g++-14")
    if not nvcc or not host:
        pytest.skip("CUDA compiler and GNU host compiler required; no GPU used")
    rank = request.param
    directory = tmp_path_factory.mktemp(f"tile-access-rank{rank}")
    source = SOURCE if rank == 1 else SOURCE_3D
    source = source.replace("a(:)", "a(-3:)").replace("b(:)", "b(-3:)")
    source = source.replace("a(:,:,:)", "a(-3:,-2:,-1:)").replace("b(:,:,:)", "b(-3:,-2:,-1:)")
    _, analysis, routine, nodes = team_fixture(directory, source)
    issued = source_device_task(analysis, routine, nodes)
    provider = emit_tile_task_provider(issued.team, readable_scalars=issued, access_envelopes=True)
    (directory / "provider.cu").write_text(provider.cuda_source)
    (directory / "provider.json").write_text(json.dumps(provider.public(), indent=2))
    (directory / "runtime-double.cpp").write_text(r'''
#include "scoped_runtime.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
static int mode=0,calls[5]={};
static std::size_t extents[3]={21,13,9};
static std::int64_t lower[3]={-3,-2,-1};
extern "C" void reset(int value){mode=value;for(auto&x:calls)x=0;}
extern "C" int counter(int index){return calls[index];}
extern "C" int fort_scope_serial_caller(){return 1;}
extern "C" int fort_scope_numerical_environment_supported(){return 1;}
extern "C" int fort_scope_layout_get(fort_scope_t,fort_buffer_t,fort_scope_layout* layout){
 ++calls[0];extents[0]=mode==1?2:21;
 *layout={RANK,FORT_SCOPE_REAL64,8,nullptr,extents,lower,1};
 if(mode==2)layout->rank=2;
 return 0;
}
extern "C" int fort_scope_wait(fort_scope_t){++calls[1];return 0;}
extern "C" int fort_scope_scratch_acquire_v1(fort_scope_t,std::size_t n,fort_scope_scratch_lease_v1* lease){
 ++calls[2];*lease={1,0,1,reinterpret_cast<void*>(4096),n,n};return 0;
}
extern "C" int fort_scope_scratch_release_v1(fort_scope_t,std::uint64_t){++calls[4];return 0;}
extern "C" int fort_scope_source_execute_v1(fort_scope_t,const fort_scope_external_call_v1* call,
 std::uint64_t,fort_scope_source_worker_v1,void*,fort_scope_external_report_v1* report){
 ++calls[3];assert(call->binding_count==2);
 for(int i=0;i<2;++i){
  const auto& access=call->bindings[i].access;
  assert(access.flags==0&&access.read_count==1&&access.write_count==std::size_t(i));
  assert(access.overwrite_count==0&&access.overwrites==nullptr);
  const auto& box=access.reads[0];
  assert(box.lower[0]==std::size_t(i==0?3:4));
  assert(box.upper[0]==std::size_t(i==0?19:17));
  if(RANK==3){assert(box.lower[1]==3&&box.upper[1]==10);assert(box.lower[2]==2&&box.upper[2]==7);}
  if(i){for(int axis=0;axis<RANK;++axis){assert(access.writes[0].lower[axis]==box.lower[axis]);assert(access.writes[0].upper[axis]==box.upper[axis]);}}
 }
 *report={1,FORT_SCOPE_RESOURCE,0,0};return FORT_SCOPE_RESOURCE;
}
'''.replace("RANK", str(rank)))
    checked([nvcc, "-std=c++17", "-ccbin", host, "-arch=sm_86", "--fmad=false", "--ftz=false",
             "-shared", "-Xcompiler=-fPIC", "-I", str(RUNTIME), "-I", str(TEMPLATES),
             str(directory / "provider.cu"), str(directory / "runtime-double.cpp"),
             "-o", str(directory / "provider.so")], directory, "compile")
    library = c.CDLL(str(directory / "provider.so"))
    function = getattr(library, provider.entry)
    function.argtypes = [c.c_int64, c.c_int64, c.c_int64,
                         *[c.POINTER(c.c_int) for _ in provider.parameters], c.POINTER(Report)]
    function.restype = c.c_int
    return provider, library, function


def invoke(compiled, mode=0, zero=False):
    provider, library, function = compiled
    library.reset(mode)
    values = {"n": 0 if zero else 13, "t": 3, "nx": 0 if zero else 13,
              "ny": 7, "nz": 5, "tx": 3, "ty": 4, "tz": 2}
    widths = (3, 4, 2)
    lower = (-3, -2, -1)
    arguments = []
    for parameter in provider.parameters:
        if parameter.role == "scalar":
            value = values[parameter.binding.name]
        elif zero:
            arguments.append(None)
            continue
        elif parameter.role == "external_lower":
            value = lower[parameter.axis-1]
        else:
            value = -1 if parameter.role == "scratch_lower" else widths[parameter.axis-1]+2
        arguments.append(c.c_int(value))
    report = Report()
    status = function(1, 11, 12, *[None if value is None else c.byref(value) for value in arguments], c.byref(report))
    return status, report, [library.counter(i) for i in range(5)]


def test_exact_conservative_original_boxes_have_no_overwrite_or_initialization_credit(metadata_provider):
    status, report, calls = invoke(metadata_provider)
    assert status == report.status == 4
    assert report.started == report.committed == 0
    assert calls == [2, 1, 1, 1, 1]
    contract = metadata_provider[0].public()
    assert contract["version"] == 7
    assert contract["access_envelopes"]["overwrites_authorized"] is False
    assert contract["access_envelopes"]["initialization_authorized"] is False


@pytest.mark.parametrize("mode", [1, 2])
def test_bad_physical_envelope_or_rank_rejects_before_wait_scratch_or_payload(metadata_provider, mode):
    status, report, calls = invoke(metadata_provider, mode)
    assert status == report.status == 1
    assert report.started == report.committed == 0
    assert calls[0] == (2 if mode == 1 else 1)
    assert calls[1:] == [0, 0, 0, 0]


def test_zero_domain_does_not_read_missing_descriptors_or_initialize_cuda(metadata_provider):
    status, report, calls = invoke(metadata_provider, zero=True)
    assert status == report.status == 0
    assert report.started == report.committed == 0
    assert calls == [0, 0, 0, 0, 0]


def test_envelopes_do_not_inherit_unqualified_whole_root_management_costs(tmp_path):
    _, analysis, routine, nodes = team_fixture(tmp_path)
    issued = source_device_task(analysis, routine, nodes)
    for value in (1, None, "yes"):
        with pytest.raises(CompilationError, match="explicit checked emission"):
            emit_tile_task_provider(issued.team, access_envelopes=value)
    with pytest.raises(CompilationError, match="management costs are unpriced"):
        emit_tile_task_provider(issued.team, access_envelopes=True, pricing=object())
