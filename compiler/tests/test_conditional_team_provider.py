"""Compiled metadata/legality guard precedes read credit and transaction."""
import ctypes as c
import shutil
from pathlib import Path

import pytest

from compiler.emission.cuda.atomic_team_provider import emit_atomic_team_provider
from compiler.frontend.numerical_effects import prove_numerical_effects
from compiler.frontend.reached_read_requirements import prove_reached_read_requirements
from compiler.tests.test_atomic_team_provider import RUNTIME, TEMPLATES
from compiler.tests.test_conditional_numerical import prepared
from compiler.tests.test_tile_task_provider import Report, checked


@pytest.fixture(scope='module')
def compiled(tmp_path_factory):
    nvcc, host = shutil.which('nvcc'), shutil.which('g++-14') or shutil.which('g++')
    if not nvcc or not host:
        pytest.skip('CUDA toolkit required for host-only conditional metadata compilation')
    out = tmp_path_factory.mktemp('conditional-provider-host')
    analysis, routine, nodes, team, policy, context, alternative, result = prepared(out)
    effects = prove_numerical_effects(analysis, routine.qualified, nodes, result)
    reads = prove_reached_read_requirements(analysis, routine.qualified, nodes, result)
    emitted = emit_atomic_team_provider(alternative, result, effects, reads)
    (out/'provider.cu').write_text(emitted.cuda_source)
    (out/'provider.json').write_text(__import__('json').dumps(emitted.public(), indent=2))
    (out/'host.cpp').write_text(r'''
#include "scoped_runtime.h"
#include <cassert>
#include <cstdint>
static int admissions=0,calls=0,mode=0;
static std::size_t length=14;
static std::int64_t lower=-2;
extern "C" void reset(int value){admissions=calls=0;mode=value;length=mode==1?5:14;}
extern "C" int admission_count(){return admissions;}
extern "C" int transaction_count(){return calls;}
extern "C" int fort_scope_serial_caller(){return 1;}
extern "C" int fort_scope_numerical_environment_supported(){return 1;}
extern "C" int fort_scope_layout_get(fort_scope_t,fort_buffer_t buffer,fort_scope_layout* layout){
 if(buffer!=11)return FORT_SCOPE_ARGUMENT;
 *layout={1,FORT_SCOPE_REAL64,8,nullptr,&length,&lower,1};return 0;
}
extern "C" int fort_scope_admit_source_reads_v1(fort_scope_t,fort_buffer_t buffer,
 const fort_scope_section* sections,std::size_t count){
 assert(buffer==11&&count==2&&sections);++admissions;
 return mode==2?FORT_SCOPE_RESOURCE:FORT_SCOPE_OK;
}
extern "C" int fort_scope_source_execute_v1(fort_scope_t,const fort_scope_external_call_v1* call,
 std::uint64_t count,fort_scope_source_worker_v1 worker,void* state,fort_scope_external_report_v1* report){
 assert(admissions==1&&count==2&&call->binding_count==1);++calls;
 assert(call->bindings[0].access.read_count==2&&call->bindings[0].access.write_count==2);
 if(mode==3){fort_scope_external_view_v1 views[1]={};int started=-1;std::uint64_t accepted=99;
   int status=worker(state,views,1,nullptr,&started,&accepted);
   assert(status==FORT_SCOPE_STALE&&started==0&&accepted==0);
 }
 *report={1,FORT_SCOPE_RESOURCE,0,0};return FORT_SCOPE_RESOURCE;
}
''')
    checked([nvcc,'-std=c++17','-ccbin',host,'-arch=sm_86','--fmad=false','--ftz=false',
             '-shared','-Xcompiler=-fPIC','-I',str(RUNTIME),'-I',str(TEMPLATES),
             str(out/'provider.cu'),str(out/'host.cpp'),'-o',str(out/'provider.so')],out,'nvcc')
    lib=c.CDLL(str(out/'provider.so'));entry=getattr(lib,emitted.entry)
    entry.argtypes=[c.c_int64,c.c_int64,*[c.POINTER(c.c_int) for _ in emitted.parameters],c.POINTER(Report)]
    entry.restype=c.c_int
    return emitted,lib,entry


def invoke(compiled,n,mode=0):
    emitted,lib,entry=compiled;lib.reset(mode)
    values=[c.c_int(-2 if parameter.axis is not None else n) for parameter in emitted.parameters]
    report=Report();status=entry(1,11,*[c.byref(value) for value in values],c.byref(report))
    return status,report,lib.admission_count(),lib.transaction_count()


@pytest.mark.parametrize('n',[0,3,8])
def test_exact_legal_domains_allow_read_admission_then_safe_transaction_decline(compiled,n):
    status,report,admissions,calls=invoke(compiled,n)
    assert (status,report.status,report.started,report.committed,admissions,calls)==(4,4,0,0,1,1)


@pytest.mark.parametrize('n',[1,2])
def test_shifted_preceding_write_domains_decline_before_any_read_credit(compiled,n):
    status,report,admissions,calls=invoke(compiled,n)
    assert (status,report.status,report.started,report.committed,admissions,calls)==(5,5,0,0,0,0)


@pytest.mark.parametrize('n',[-2**31,2**31-1,2**31-2])
def test_original_intermediate_and_terminal_overflow_decline_before_read_credit(compiled,n):
    status,report,admissions,calls=invoke(compiled,n)
    assert status==report.status==1
    assert (report.started,report.committed,admissions,calls)==(0,0,0,0)


def test_invalid_descriptor_and_admission_failure_prevent_submission(compiled):
    status,report,admissions,calls=invoke(compiled,8,mode=1)
    assert status==report.status==1
    assert (report.started,report.committed,admissions,calls)==(0,0,0,0)
    status,report,admissions,calls=invoke(compiled,8,mode=2)
    assert status==report.status==4
    assert (report.started,report.committed,admissions,calls)==(0,0,1,0)


def test_callback_rechecks_exact_generation_before_any_attempt(compiled):
    status,report,admissions,calls=invoke(compiled,8,mode=3)
    assert (status,report.started,report.committed,admissions,calls)==(4,0,0,1,1)
