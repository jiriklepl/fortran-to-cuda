"""Compile normal kernels and inspect atomic prework without a CUDA device."""
import ctypes as c
import shutil
from pathlib import Path

import pytest

from compiler.driver.options import CompilerOptions
from compiler.emission.cuda.atomic_team_provider import emit_atomic_team_provider
from compiler.frontend.atomic_worksharing_team import atomic_worksharing_alternatives, prove_atomic_worksharing_team
from compiler.frontend.numerical_effects import prove_numerical_effects
from compiler.frontend.numerical_preparation import NumericalPreparationProvider
from compiler.frontend.reached_read_requirements import prove_reached_read_requirements
from compiler.offload.config import OffloadConfig
from compiler.tests.test_atomic_worksharing_team import ENTRY, fixture
from compiler.tests.test_tile_task_provider import Report, checked

RUNTIME = Path(__file__).resolve().parents[1] / "runtime"
TEMPLATES = Path(__file__).resolve().parents[1] / "emission/common/templates"


def provider(directory):
    _, analysis, routine, nodes = fixture(directory)
    team = prove_atomic_worksharing_team(analysis, routine, nodes)
    alternative = atomic_worksharing_alternatives(team)[0]
    policy = NumericalPreparationProvider(analysis, options=CompilerOptions(opt_level=0),
        config=OffloadConfig(policy="sections", host_threads=4), environment_root=ENTRY)
    prepared = policy.prepare_atomic_alternative(alternative)
    effects = prove_numerical_effects(analysis, ENTRY, nodes, prepared)
    reads = prove_reached_read_requirements(analysis, ENTRY, nodes, prepared)
    return emit_atomic_team_provider(alternative, prepared, effects, reads)


@pytest.fixture(scope="module")
def compiled(tmp_path_factory):
    nvcc, host = shutil.which("nvcc"), shutil.which("g++-14") or shutil.which("g++")
    fc = shutil.which("gfortran")
    if not nvcc or not host or not fc:
        pytest.skip("CUDA compilation toolkit and gfortran required; no device used")
    out = tmp_path_factory.mktemp("atomic-provider-host")
    emitted = provider(out)
    (out / "provider.cu").write_text(emitted.cuda_source)
    (out / "provider.f90").write_text(emitted.fortran_source)
    (out / "host.cpp").write_text(r'''
#include "scoped_runtime.h"
#include <cassert>
#include <cstdint>
static int mode=0,calls=0;
static std::size_t lengths[3]={10,10,10};
static std::int64_t lower=1;
extern "C" void reset(int value){mode=value;calls=0;lengths[0]=lengths[1]=lengths[2]=10;if(mode==1)lengths[1]=7;}
extern "C" int counter(){return calls;}
extern "C" int fort_scope_serial_caller(){return 1;}
extern "C" int fort_scope_numerical_environment_supported(){return 1;}
extern "C" int fort_scope_layout_get(fort_scope_t,fort_buffer_t buffer,fort_scope_layout* layout){
 if(buffer<11||buffer>13)return FORT_SCOPE_ARGUMENT;
 *layout={1,FORT_SCOPE_REAL64,8,nullptr,&lengths[buffer-11],&lower,1};return 0;
}
extern "C" int fort_scope_source_execute_v1(fort_scope_t,const fort_scope_external_call_v1* call,
 std::uint64_t count,fort_scope_source_worker_v1 worker,void* user,fort_scope_external_report_v1* report){
 ++calls;assert(count==2&&call->binding_count==3);
 assert(!call->workspace_bytes&&!call->workspace_lease);
 for(std::size_t i=0;i<3;++i){
  assert(call->bindings[i].access.flags==0);
  assert(call->bindings[i].access.read_count>0);
  if(i<2){assert(call->bindings[i].access.write_count==1);assert(call->bindings[i].access.overwrite_count==1);}
 }
 if(mode==2){int started=-1;std::uint64_t accepted=99;fort_scope_external_view_v1 invalid[3]={};
  int result=worker(user,invalid,3,nullptr,&started,&accepted);
  assert(result==FORT_SCOPE_ARGUMENT&&started==0&&accepted==0);
 }
 *report={1,FORT_SCOPE_RESOURCE,0,0};return FORT_SCOPE_RESOURCE;
}
''')
    checked([nvcc, "-std=c++17", "-ccbin", host, "-arch=sm_86", "--fmad=false", "--ftz=false",
             "-shared", "-Xcompiler=-fPIC", "-I", str(RUNTIME), "-I", str(TEMPLATES),
             str(out / "provider.cu"), str(out / "host.cpp"), "-o", str(out / "provider.so")], out, "nvcc")
    checked([fc, "-cpp", "-ffree-line-length-none", "-J", str(out), "-c", str(RUNTIME / "scoped_memory.f90"),
             "-o", str(out / "memory.o")], out, "memory-interface")
    checked([fc, "-I", str(out), "-J", str(out), "-c", str(out / "provider.f90"),
             "-o", str(out / "provider.o")], out, "provider-interface")
    lib = c.CDLL(str(out / "provider.so"))
    entry = getattr(lib, emitted.entry)
    entry.argtypes = [c.c_int64, *[c.c_int64 for _ in emitted.fields],
                     *[c.POINTER(c.c_int) for _ in emitted.parameters], c.POINTER(Report)]
    entry.restype = c.c_int
    return emitted, lib, entry


def invoke(compiled, *, n=8, mode=0, alias=False):
    emitted, lib, entry = compiled
    lib.reset(mode)
    arguments = [c.c_int(n if p.axis is None else 0 if p.binding.name == "p" else 1) for p in emitted.parameters]
    report = Report()
    status = entry(1, 11, 11 if alias else 12, 13, *[c.byref(value) for value in arguments], c.byref(report))
    return status, report, lib.counter()


@pytest.mark.parametrize("mode", [0, 2])
def test_exact_all_unit_access_union_precedes_one_transaction(compiled, mode):
    status, report, calls = invoke(compiled, mode=mode)
    assert status == report.status == 4  # Host double safely rejects before work.
    assert (report.started, report.committed, calls) == (0, 0, 1)


def test_invalid_later_unit_prevents_every_submission(compiled):
    status, report, calls = invoke(compiled, mode=1)
    assert status == report.status == 5
    assert (report.started, report.committed, calls) == (0, 0, 0)


def test_empty_path_and_alias_reject_before_transaction(compiled):
    status, report, calls = invoke(compiled, n=0)
    assert (status, report.started, report.committed, calls) == (0, 0, 0, 0)
    status, report, calls = invoke(compiled, alias=True)
    assert status == report.status == 1
    assert (report.started, report.committed, calls) == (0, 0, 0)
