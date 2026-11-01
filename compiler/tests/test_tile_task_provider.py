"""Compile the real provider and exercise metadata prework without a device.

The host runtime double never accepts device work. It records the submitted
typed call, or invokes the real callback only with intentionally invalid views,
which must reject before launch. Numerical CUDA execution has a separate gate.
"""

import ctypes as c
import shutil
import subprocess
from copy import copy
from dataclasses import replace
from pathlib import Path

import pytest

from compiler.emission.cuda.tile_task_provider import BATCH_TARGET_BYTES, emit_tile_task_provider
from compiler.frontend.source_device_tasks import source_device_task
from compiler.frontend.tile_private import prove_tile_private_team
from compiler.ir import CompilationError
from compiler.tests.test_tile_private import SOURCE, SOURCE_3D
from compiler.tests.test_tile_private_team import team_fixture

RUNTIME = Path(__file__).resolve().parents[1] / "runtime"
TEMPLATES = Path(__file__).resolve().parents[1] / "emission/common/templates"


class Report(c.Structure):
    _fields_ = [("version", c.c_uint32), ("status", c.c_int32), ("started", c.c_uint32), ("committed", c.c_uint32)]


def test_provider_rejects_substitute_team_and_scalar_readability_authority(tmp_path):
    _, analysis, routine, nodes = team_fixture(tmp_path)
    team = prove_tile_private_team(analysis, routine, nodes)
    issued = source_device_task(analysis, routine, nodes)
    for forged in (copy(team), replace(team)):
        with pytest.raises(CompilationError):
            emit_tile_task_provider(forged)
    for forged in (True, issued.public(), replace(issued)):
        with pytest.raises(CompilationError):
            emit_tile_task_provider(team, readable_scalars=forged)
    result = emit_tile_task_provider(team, readable_scalars=issued)
    assert result.public()["scalar_readability_identity"] == issued.identity
    assert result.public()["unit_proof_identities"] == [unit.identity for unit in team.units]


def test_conditional_nonconstant_scalar_is_not_made_readable_by_reference(tmp_path):
    source = SOURCE.replace("integer::bi,i,li", "integer::bi,i,li\nreal(8)::scale")
    source = source.replace("  b(i)=b(i)+(tmp(li+1)-tmp(li-1))", "  if(mod(i,2)==0)then\n   b(i)=scale*(tmp(li+1)-tmp(li-1))\n  endif")
    _, analysis, routine, nodes = team_fixture(tmp_path, source)
    team = prove_tile_private_team(analysis, routine, nodes)
    with pytest.raises(CompilationError, match="conditional scalar"):
        emit_tile_task_provider(team)
    with pytest.raises(CompilationError, match="whole-defined capture authority"):
        source_device_task(analysis, routine, nodes)


def checked(argv, directory, name):
    completed = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    (directory / (name + ".stdout")).write_text(completed.stdout)
    (directory / (name + ".stderr")).write_text(completed.stderr)
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.fixture(scope="module", params=["thread", "cooperative"])
def compiled_provider(request, tmp_path_factory):
    nvcc, host = shutil.which("nvcc"), shutil.which("g++-14") or shutil.which("g++")
    fc = shutil.which("gfortran")
    if not nvcc or not host or not fc:
        pytest.skip("CUDA compilation toolkit and gfortran are required; no device used")
    directory = tmp_path_factory.mktemp("tile-provider-host-" + request.param)
    _, analysis, routine, nodes = team_fixture(directory)
    provider = emit_tile_task_provider(prove_tile_private_team(analysis, routine, nodes), mapping=request.param)
    (directory / "provider.cu").write_text(provider.cuda_source)
    (directory / "provider.f90").write_text(provider.fortran_source)
    (directory / "host.cpp").write_text(r'''
#include "scoped_runtime.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
static int mode=0,calls[4]={};
static std::uint64_t expected_count=0;
extern "C" void reset(int value){mode=value;for(auto& x:calls)x=0;expected_count=0;}
extern "C" int counter(int index){return calls[index];}
extern "C" std::uint64_t expected(){return expected_count;}
extern "C" int fort_scope_serial_caller(){return 1;}
extern "C" int fort_scope_numerical_environment_supported(){return 1;}
extern "C" int fort_scope_wait(fort_scope_t){++calls[0];return 0;}
extern "C" int fort_scope_scratch_acquire_v1(fort_scope_t,std::size_t n,fort_scope_scratch_lease_v1* lease){
 ++calls[1];if(mode==3)return FORT_SCOPE_RESOURCE;
 *lease={1,0,1,reinterpret_cast<void*>(4096),n,n};return 0;
}
extern "C" int fort_scope_scratch_release_v1(fort_scope_t,std::uint64_t){++calls[3];return 0;}
extern "C" int fort_scope_source_execute_v1(fort_scope_t,const fort_scope_external_call_v1* call,
 std::uint64_t count,fort_scope_source_worker_v1 worker,void* user,fort_scope_external_report_v1* report){
 ++calls[2];expected_count=count;assert(call->binding_count==2);
 assert(call->bindings[0].access.flags==FORT_SCOPE_READ_ALL);
 assert(call->bindings[1].access.flags==(FORT_SCOPE_READ_ALL|FORT_SCOPE_WRITE_ALL));
 assert(call->workspace_lease==0&&call->workspace_bytes==0);
 int status=FORT_SCOPE_RESOURCE,started=0;std::uint64_t accepted=0;
 if(mode==1){
  std::size_t extent=2,stride=8;std::int64_t lower=-3;
  fort_scope_external_view_v1 views[2]={};
  for(auto& view:views){view.extents=&extent;view.strides=&stride;view.lower_bounds=&lower;view.device=reinterpret_cast<void*>(4096);}
  status=worker(user,views,2,nullptr,&started,&accepted);
  assert(status==FORT_SCOPE_ARGUMENT&&started==0&&accepted==0);
 }
 *report={1,status,0,0};return status;
}
''')
    checked([nvcc, "-std=c++17", "-ccbin", host, "-arch=sm_86", "--fmad=false", "--ftz=false",
             "-shared", "-Xcompiler=-fPIC", "-I", str(RUNTIME), "-I", str(TEMPLATES),
             str(directory / "provider.cu"), str(directory / "host.cpp"), "-o", str(directory / "provider.so")], directory, "cuda-host-link")
    checked([fc, "-cpp", "-ffree-line-length-none", "-J", str(directory), "-c", str(RUNTIME / "scoped_memory.f90"),
             "-o", str(directory / "memory.o")], directory, "fortran-runtime")
    checked([fc, "-I", str(directory), "-J", str(directory), "-c", str(directory / "provider.f90"),
             "-o", str(directory / "provider.o")], directory, "fortran-interface")
    lib = c.CDLL(str(directory / "provider.so"))
    function = getattr(lib, provider.entry)
    function.argtypes = [c.c_int64, c.c_int64, c.c_int64, *[c.POINTER(c.c_int) for _ in provider.parameters], c.POINTER(Report)]
    function.restype = c.c_int
    lib.expected.restype = c.c_uint64
    return provider, lib, function


def invoke(compiled, values, mode=0):
    provider, lib, function = compiled
    lib.reset(mode)
    supplied = [None if values.get((parameter.role, parameter.binding.name, parameter.axis)) is None else
                c.c_int(values[parameter.role, parameter.binding.name, parameter.axis]) for parameter in provider.parameters]
    report = Report()
    status = function(1, 11, 12, *[None if value is None else c.byref(value) for value in supplied], c.byref(report))
    return status, report, [lib.counter(i) for i in range(4)], lib.expected()


def inputs(n=13, width=3):
    return {("scalar", "n", None): n, ("scalar", "t", None): width,
            ("external_lower", "a", 1): -3, ("external_lower", "b", 1): -3,
            ("scratch_lower", "tmp", 1): -1, ("scratch_upper", "tmp", 1): width + 2}


def test_zero_domain_does_not_touch_descriptor_arguments_or_runtime(compiled_provider):
    status, report, calls, expected = invoke(compiled_provider, {("scalar", "n", None): 0, ("scalar", "t", None): 3})
    assert (status, report.version, report.status, report.started, report.committed) == (0, 1, 0, 0, 0)
    assert calls == [0, 0, 0, 0]
    assert expected == 0


@pytest.mark.parametrize(("n", "width"), [(1, 3), (7, 3), (13, 4), (19, 7)])
def test_exact_original_tile_launch_count_and_typed_access_use_one_existing_transaction(compiled_provider, n, width):
    status, report, calls, expected = invoke(compiled_provider, inputs(n, width))
    assert status == report.status == 4
    assert (report.started, report.committed) == (0, 0)
    assert calls == [1, 1, 1, 1]
    tiles = (n - 1) // width + 1
    stride = ((width + 4) * 8 + 7) & ~7
    batch = min(tiles, max(1, BATCH_TARGET_BYTES // stride))
    assert expected == tiles // batch + bool(tiles % batch)


@pytest.mark.parametrize(("key", "value"), [(('scratch_lower', 'tmp', 1), 1), (('scratch_upper', 'tmp', 1), 4),
                                       (('scalar', 't', None), 0), (('external_lower', 'a', 1), None)])
def test_bad_original_descriptors_decline_before_runtime_or_submission(compiled_provider, key, value):
    values = inputs()
    values[key] = value
    status, report, calls, expected = invoke(compiled_provider, values)
    assert status in {1, 4}
    assert report.status == status
    assert (report.started, report.committed) == (0, 0)
    assert calls == [0, 0, 0, 0]
    assert expected == 0


def test_metadata_index_envelope_rejects_short_field_before_first_launch(compiled_provider):
    status, report, calls, expected = invoke(compiled_provider, inputs(), mode=1)
    assert status == report.status == 1
    assert (report.started, report.committed) == (0, 0)
    assert calls == [1, 1, 1, 1]
    assert expected == 1


def test_scratch_resource_failure_does_not_enter_transaction_or_release_missing_lease(compiled_provider):
    status, report, calls, expected = invoke(compiled_provider, inputs(), mode=3)
    assert status == report.status == 4
    assert (report.started, report.committed) == (0, 0)
    assert calls == [1, 1, 0, 0]
    assert expected == 0


@pytest.mark.parametrize("upper", [196613, 1048581])
def test_scratch_target_handles_odd_batch_tail_and_single_tile_larger_than_target(compiled_provider, upper):
    values = inputs(19)
    values["scratch_upper", "tmp", 1] = upper
    status, report, calls, expected = invoke(compiled_provider, values)
    assert status == report.status == 4
    assert calls == [1, 1, 1, 1]
    stride = (((upper + 2) * 8) + 7) & ~7
    batch = min(7, max(1, BATCH_TARGET_BYTES // stride))
    assert expected == 7 // batch + bool(7 % batch)
    if stride > BATCH_TARGET_BYTES:
        assert batch == 1
    else:
        assert 7 % batch


@pytest.fixture(scope="module")
def compiled_volume(compiled_provider, tmp_path_factory):
    mapping = compiled_provider[0].public()["mapping"]
    directory = tmp_path_factory.mktemp("tile-provider-volume-" + mapping)
    _, analysis, routine, nodes = team_fixture(directory, SOURCE_3D)
    provider = emit_tile_task_provider(prove_tile_private_team(analysis, routine, nodes), mapping=mapping)
    (directory / "provider.cu").write_text(provider.cuda_source)
    original_host = Path(compiled_provider[1]._name).parent / "host.cpp"
    checked([shutil.which("nvcc"), "-std=c++17", "-ccbin", shutil.which("g++-14") or shutil.which("g++"),
             "-arch=sm_86", "--fmad=false", "--ftz=false", "-shared", "-Xcompiler=-fPIC", "-I", str(RUNTIME),
             "-I", str(TEMPLATES), str(directory / "provider.cu"), str(original_host), "-o", str(directory / "provider.so")],
            directory, "cuda-volume-host-link")
    lib = c.CDLL(str(directory / "provider.so"))
    function = getattr(lib, provider.entry)
    function.argtypes = [c.c_int64, c.c_int64, c.c_int64, *[c.POINTER(c.c_int) for _ in provider.parameters], c.POINTER(Report)]
    function.restype = c.c_int
    lib.expected.restype = c.c_uint64
    return provider, lib, function


@pytest.mark.parametrize("empty_axis", ["z", "y", "x"])
def test_nested_zero_domain_never_reads_later_axis_values_or_descriptors(compiled_volume, empty_axis):
    values = {}
    for axis in "zyx":
        values["scalar", "n" + axis, None] = 0 if axis == empty_axis else 1
        values["scalar", "t" + axis, None] = 3
        if axis == empty_axis:
            break
    status, report, calls, expected = invoke(compiled_volume, values)
    assert (status, report.started, report.committed) == (0, 0, 0)
    assert calls == [0, 0, 0, 0]
    assert expected == 0


@pytest.mark.parametrize("overflow", ["tiles", "scratch"])
def test_three_dimensional_size_products_reject_overflow_before_runtime(compiled_volume, overflow):
    values = {}
    for axis in "xyz":
        values["scalar", "n" + axis, None] = 2147483647 if overflow == "tiles" else 1
        values["scalar", "t" + axis, None] = 1 if overflow == "tiles" else 3
    for axis in range(1, 4):
        values["scratch_lower", "tmp", axis] = -1
        values["scratch_upper", "tmp", axis] = 2147483647
    status, report, calls, expected = invoke(compiled_volume, values)
    assert status == report.status == 4
    assert (report.started, report.committed) == (0, 0)
    assert calls == [0, 0, 0, 0]
    assert expected == 0
