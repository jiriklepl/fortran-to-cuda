"""Only exact current metadata can reuse a candidate; simulation is never cached."""
# ruff: noqa: F811 - imported pytest fixtures.
import ctypes as c
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from compiler.tests.test_scoped_continuation_runtime import Report
from compiler.tests.test_scoped_planning_runtime import Binding, Decision, costs, planning, select  # noqa: F401
from compiler.tests.test_scoped_runtime import TOKEN, Scope, access, runtime  # noqa: F401

RUNTIME = Path(__file__).resolve().parents[1] / 'runtime'
DRIVER = r'''
#include "scoped_memo.hpp"
#include <cassert>
#include <iostream>
#include <random>
using namespace fort_scoped::planning;
namespace memo=fort_scoped::memo;
fort_scope_plan_costs costs(){
 fort_scope_plan_costs c{};c.version=FORT_SCOPE_PLANNING_ABI_VERSION;c.valid=1;c.max_allocation_bytes=1<<20;
 c.cpu_flops=1e5;c.cpu_bandwidth=1e8;c.gpu_flops=1e8;c.gpu_bandwidth=1e10;
 c.h2d_bandwidth=c.d2h_bandwidth=1e9;
 c.create_seconds=c.register_seconds=c.host_access_seconds=c.device_access_seconds=1e-8;
 c.gpu_setup_seconds=c.cold_driver_startup_seconds=c.allocation_seconds=c.release_seconds=1e-6;
 c.wait_seconds=c.launch_enqueue_seconds=1e-6;c.planning_operation_seconds=1e-10;
 return c;
}
Inputs input(){
 Inputs in;in.graph_mode=true;in.definitions_validated=true;in.continuation=true;
 in.driver_initialized=true;in.query_construction_operations=3;
 for(size_t i=0;i<3;++i){Resource r;r.handle=10+i;r.element_bytes=8;r.bytes=128;r.extents={16};
  r.initialized=r.host_current={Box{{0},{16}}};in.resources.push_back(r);}
 for(size_t i=0;i<2;++i){Operation op;op.kind=FORT_SCOPE_PLAN_WORKER;op.unit=100+i;
  op.flops=1e5;op.memory_bytes=256;op.gpu_available=true;
  Region full=in.resources[i].initialized;op.bindings={{10+i,{full,full,full}}};in.operations.push_back(op);}
 return in;
}
int main(int argc,char**argv){
 assert(argc==2);std::string mode=argv[1];auto in=input();auto c=costs();
 const std::vector<uint64_t> env={0,4,1,1};auto key=memo::make_key(in,c,env);assert(key);
 if(mode=="normalize"){
  auto changed=in;for(auto &r:changed.resources)r.handle+=1000;
  for(auto &op:changed.operations)for(auto &b:op.bindings)b.buffer+=1000;
  auto second=memo::make_key(changed,c,env);assert(second && key->words==second->words && key->hash==second->hash);
  std::reverse(changed.resources.begin(),changed.resources.end());
  assert(memo::make_key(changed,c,env)->words!=key->words);
  auto unused=in.resources.back();unused.handle=99;unused.bytes=256;unused.extents={32};
  in.resources.push_back(unused);auto a=memo::make_key(in,c,env);
  std::swap(in.resources[2],in.resources[3]);auto b=memo::make_key(in,c,env);assert(a && b && a->words==b->words);
  in.asynchronous_release=false;auto sync=memo::make_key(in,c,env);
  std::swap(in.resources[0],in.resources[2]);auto reordered=memo::make_key(in,c,env);
  assert(sync && reordered && sync->words!=reordered->words);
 }else if(mode=="mutations"){
  for(int test=0;test<22;++test){auto x=in;auto p=c;auto e=env;
   switch(test){
    case 0:x.device_budget=128;break;case 1:x.driver_initialized=false;break;
    case 2:x.continuation=false;break;case 3:x.charge_create=false;break;
    case 4:x.registrations_incurred=0;break;case 5:x.query_construction_operations++;break;
    case 6:x.asynchronous_release=false;break;case 7:x.graph_minimum_relative_saving=.1;break;
    case 8:x.operations[0].unit++;break;case 9:x.operations[0].flops++;break;
    case 10:x.operations[0].memory_bytes++;break;case 11:x.operations[0].gpu_available=false;break;
    case 12:x.operations[0].cpu_numerical_seconds=1;break;case 13:x.operations[0].gpu_numerical_seconds=1;break;
    case 14:x.operations[0].bindings[0].effects.reads={Box{{0},{8}}};break;
    case 15:x.operations[0].bindings[0].effects.overwrites.clear();break;
    case 16:x.resources[2].initialized={Box{{0},{8}}};break;
    case 17:x.resources[2].initialized=x.resources[2].host_current={Box{{0},{8}}};break;
    case 18:x.operations[1].bindings[0].buffer=10;break;
    case 19:p.cpu_flops++;break;case 20:p.planning_operation_seconds*=2;break;
    case 21:e[1]++;break;
   }
   auto changed=memo::make_key(x,p,e);assert(changed && changed->words!=key->words);
  }
 }else if(mode=="ineligible"){
  for(int test=0;test<10;++test){auto x=in;
   switch(test){case 0:x.graph_mode=false;break;case 1:x.definitions_validated=false;break;
    case 2:x.device_ready=true;break;case 3:x.pending=true;break;case 4:x.resources[0].allocated=true;break;
    case 5:x.resources[0].device_current=x.resources[0].initialized;break;
    case 6:x.resources[0].pending_host_reads=x.resources[0].initialized;break;
    case 7:x.resources[0].last_device_use=1;break;
    case 8:x.operations[0].memory_bytes=std::numeric_limits<double>::quiet_NaN();break;
    case 9:x.operations[0].flops=std::numeric_limits<double>::infinity();break;}
   assert(!memo::make_key(x,c,env));
  }
 }else if(mode=="optional_models"){
  auto absolute=in;absolute.operations[0].compute_costs=fort_scope_compute_costs_v1{2,7,.1,.2,.05};
  auto backend=memo::make_key(absolute,c,env);assert(backend && backend->words!=key->words);
  for(int test=0;test<5;++test){auto x=absolute;auto &v=*x.operations[0].compute_costs;
   switch(test){case 0:v.version=1;break;case 1:v.known_backends=3;break;
    case 2:v.native_fortran_seconds+=1;break;case 3:v.generated_cpu_seconds+=1;break;
    case 4:v.gpu_seconds+=1;break;}
   auto changed=memo::make_key(x,c,env);assert(changed && changed->words!=backend->words);
  }
  auto collective=in;collective.continuation=false;fort_scope_team_costs team{};
  team.version=FORT_SCOPE_TEAM_ABI_VERSION;team.valid=1;team.cpu_threads=4;team.expected_omp_level=1;
  team.protocol_id=FORT_SCOPE_TEAM_PROTOCOL_ID;team.native_cpu_flops=1e5;team.native_cpu_bandwidth=1e8;
  collective.team_costs=team;auto teamkey=memo::make_key(collective,c,env);assert(teamkey);
  collective.team_costs->cpu_threads=8;
  assert(memo::make_key(collective,c,env)->words!=teamkey->words);
  collective.team_costs->cpu_threads=4;collective.team_costs->native_call_seconds=1;
  assert(memo::make_key(collective,c,env)->words!=teamkey->words);
  auto pinned=in;fort_scope_batch_costs transfer{};
  transfer.version=FORT_SCOPE_BATCH_ABI_VERSION;transfer.valid=1;transfer.max_slot_bytes=16*1024*1024;
  for(auto &x:transfer.staging_cold_seconds)x=1e-6;
  for(auto &x:transfer.staging_reuse_seconds)x=1e-8;
  transfer.event_record_seconds=transfer.event_wait_seconds=transfer.ready_event_seconds=1e-8;
  transfer.preparation_operation_seconds=1e-8;
  transfer.pack_bytes_per_second=transfer.unpack_bytes_per_second=1e9;
  transfer.pinned_h2d_bandwidth=transfer.pinned_d2h_bandwidth=1e9;
  pinned.transfer_mode=FORT_SCOPE_TRANSFERS_PINNED;pinned.transfer_costs=transfer;
  auto pin=memo::make_key(pinned,c,env);assert(pin && pin->words!=key->words);
  pinned.transfer_costs->staging_reuse_seconds[3]*=2;
  assert(memo::make_key(pinned,c,env)->words!=pin->words);
  pinned.transfer_costs->pack_row_seconds=1;
  assert(memo::make_key(pinned,c,env)->words!=pin->words);
 }else if(mode=="collision_lru"){
  memo::Cache cache;auto collision=*key;collision.words.push_back(99); // Deliberate same hash.
  cache.insert(*key,{true,false});assert(!cache.find(collision));
  auto found=cache.find(*key);assert(found && (*found)[0] && !(*found)[1]);
  for(size_t i=1;i<memo::entry_limit;++i){auto k=*key;k.integer(i);cache.insert(k,{false,false});}
  assert(cache.find(*key));auto extra=*key;extra.integer(999);cache.insert(extra,{true,true});
  assert(cache.size()==memo::entry_limit && cache.bytes()<=memo::byte_limit && cache.find(*key));
  auto evicted=*key;evicted.integer(1);assert(!cache.find(evicted));
 }else if(mode=="budget"){
  memo::Cache cache;auto oversized=*key;oversized.words.resize(memo::byte_limit/sizeof(uint64_t));
  cache.insert(std::move(oversized),{true,false});assert(!cache.size() && !cache.bytes());
  auto normal=select(in,c);assert(normal.decision.available && normal.decision.gpu_units);
 }else if(mode=="cold_debt"){
  in.driver_initialized=false;c.cold_driver_startup_seconds=20;
  auto first=select(in,c);auto reused=select(in,c,&first.gpu_workers);
  assert(first.decision.gpu_units && reused.decision.gpu_units);
  assert(first.cold_driver_startup_debt_seconds==20 && reused.cold_driver_startup_debt_seconds==20);
  assert(reused.decision.estimated_seconds>reused.decision.native_seconds);
  assert(reused.graph_ranking_seconds<reused.graph_counterfactual_seconds);
 }else if(mode=="reprice"){
  std::mt19937 rng(192571);uint64_t before=0,after=0;
  for(int test=0;test<160;++test){auto x=in;x.operations.clear();
   for(size_t i=0,n=1+rng()%6;i<n;++i){Operation op;op.kind=FORT_SCOPE_PLAN_WORKER;op.unit=100+i;
    op.flops=(rng()%3)?1e5:1;op.memory_bytes=256;op.gpu_available=(rng()%4)!=0;
    auto &r=x.resources[rng()%2];Region whole=r.initialized;
    op.bindings={{r.handle,{whole,whole,whole}}};x.operations.push_back(op);}
   auto full=select(x,c);assert(full.decision.available);
   auto reused=select(x,c,&full.gpu_workers);assert(reused.decision.available && reused.gpu_workers==full.gpu_workers);
   assert(reused.decision.upload_bytes==full.decision.upload_bytes && reused.decision.download_bytes==full.decision.download_bytes);
   assert(reused.decision.launches==full.decision.launches && reused.decision.allocations==full.decision.allocations);
   assert(reused.report.terminal.download_bytes==full.report.terminal.download_bytes);
   const double a=full.report.execution_seconds-full.decision.simulated_operations*c.planning_operation_seconds;
   const double b=reused.report.execution_seconds-reused.decision.simulated_operations*c.planning_operation_seconds;
   assert(std::abs(a-b)<1e-12);before+=full.decision.simulated_operations;after+=reused.decision.simulated_operations;
  }
  assert(after<before);std::cout<<before<<" "<<after<<"\n";
 }else if(mode=="recheck"){
  auto full=select(in,c);assert(full.decision.available && full.decision.gpu_units);
  in.resources[0].initialized.clear();in.definitions_validated=false;
  assert(validate_definitions(in).status==FORT_SCOPE_UNINITIALIZED);
  assert(!memo::make_key(in,c,env));
  auto invalid=select(in,c,&full.gpu_workers);assert(!invalid.decision.available);
 }else return 2;
}
'''


@pytest.fixture(scope='module')
def memo_binary(tmp_path_factory):
    path = tmp_path_factory.mktemp('placement-memo')
    source = path / 'memo.cpp'
    source.write_text(DRIVER)
    compiler = shutil.which('g++-14') or shutil.which('g++')
    if compiler is None:
        pytest.skip('C++ compiler required')
    result = subprocess.run([compiler, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                             '-I', str(RUNTIME), str(source), '-o', str(path / 'memo')],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    return path / 'memo'


@pytest.mark.parametrize('mode', ['normalize', 'mutations', 'ineligible', 'optional_models', 'collision_lru', 'budget',
                                 'cold_debt', 'reprice', 'recheck'])
def test_exact_metadata_candidate_memo(memo_binary, mode):
    result = subprocess.run([str(memo_binary), mode], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture
def memo_runtime(planning):
    for name, signature in {
        'task_graph_enable_v1': [TOKEN], 'plan_reset_mode': [TOKEN, c.c_uint32],
        'plan_abandon_remaining_v1': [TOKEN], 'plan_report_v2': [TOKEN, c.POINTER(Report)],
    }.items():
        getattr(planning, 'fort_scope_' + name).argtypes = signature
    return planning


def query(lib, unit, *, initialized=True, compatible=1, memory_bytes=128):
    scope = Scope(lib)
    values = (c.c_double * 16)(*range(16))
    status, handle = scope.register(values, [16], identity=unit, initialized=initialized)
    scope.check(status)
    scope.check(lib.fort_scope_task_graph_enable_v1(scope.handle))
    scope.check(lib.fort_scope_plan_reset_mode(scope.handle, 1))
    bindings = (Binding * 1)(Binding(handle, access(flags=1)))
    scope.check(lib.fort_scope_plan_add(scope.handle, 1, unit, bindings, 1, 1e9, memory_bytes, 1))
    decision = select(scope, costs(), compatible=compatible)
    report = Report()
    scope.check(lib.fort_scope_plan_report_v2(scope.handle, c.byref(report)))
    assert report.available == report.owner_available == 0
    assert scope.stats().uploads == scope.stats().downloads == scope.stats().allocations == 0
    scope.check(lib.fort_scope_plan_abandon_remaining_v1(scope.handle))
    scope.close()
    assert list(values) == list(range(16))
    return decision


def decision_rows(capfd):
    result = []
    for line in capfd.readouterr().err.splitlines():
        if '"event":"decision"' in line:
            result.append(json.loads(line[line.index('{'):]))
    return result


def test_fresh_context_hit_reprices_without_old_search_or_hardware_authority(memo_runtime, monkeypatch, capfd):
    monkeypatch.setenv('FORT_RUNTIME_TRACE', '1')
    first = query(memo_runtime, 91001)
    second = query(memo_runtime, 91001)
    rows = decision_rows(capfd)
    assert len(rows) == 2
    assert not rows[0]['placement_memo_hit']
    assert rows[1]['placement_memo_hit']
    assert all(row['unpriced_memo_overhead'] and row['placement_memo_key_seconds'] >= 0 for row in rows)
    assert first.available == second.available == 1
    assert second.simulated_operations < first.simulated_operations
    assert first.gpu_units == second.gpu_units == 1
    incompatible = query(memo_runtime, 91001, compatible=0)
    assert not incompatible.available
    assert incompatible.gpu_units == 0


def test_memo_never_substitutes_for_current_definition_proof(memo_runtime, monkeypatch, capfd):
    monkeypatch.setenv('FORT_RUNTIME_TRACE', '1')
    query(memo_runtime, 91002)
    decision_rows(capfd)
    scope = Scope(memo_runtime)
    values = (c.c_double * 16)(*range(16))
    status, handle = scope.register(values, [16], identity=91002, initialized=False)
    scope.check(status)
    scope.check(memo_runtime.fort_scope_task_graph_enable_v1(scope.handle))
    scope.check(memo_runtime.fort_scope_plan_reset_mode(scope.handle, 1))
    bindings = (Binding * 1)(Binding(handle, access(flags=1)))
    scope.check(memo_runtime.fort_scope_plan_add(scope.handle, 1, 91002, bindings, 1, 1e9, 128, 1))
    out = Decision()
    assert memo_runtime.fort_scope_plan_select(scope.handle, c.byref(costs()), 1, c.byref(out)) == 7
    assert not decision_rows(capfd)
    assert scope.stats().uploads == scope.stats().downloads == scope.stats().allocations == 0
    scope.close()
    assert list(values) == list(range(16))


def test_unknown_memory_never_enters_metadata_memo(memo_runtime, monkeypatch, capfd):
    monkeypatch.setenv('FORT_RUNTIME_TRACE', '1')
    for _ in range(2):
        decision = query(memo_runtime, 91003, memory_bytes=float('nan'))
        assert not decision.available
        assert decision.gpu_units == 0
    rows = decision_rows(capfd)
    assert len(rows) == 2
    assert not any(row['placement_memo_hit'] for row in rows)


def test_prepared_access_rejects_before_memo_and_preserves_plan_for_retry(memo_runtime, monkeypatch, capfd):
    monkeypatch.setenv('FORT_RUNTIME_TRACE', '1')
    query(memo_runtime, 91005)
    decision_rows(capfd)
    scope = Scope(memo_runtime)
    values = (c.c_double * 16)(*range(16))
    status, handle = scope.register(values, [16], identity=91005)
    scope.check(status)
    scope.check(memo_runtime.fort_scope_task_graph_enable_v1(scope.handle))
    scope.check(memo_runtime.fort_scope_plan_reset_mode(scope.handle, 1))
    bindings = (Binding * 1)(Binding(handle, access(flags=1)))
    scope.check(memo_runtime.fort_scope_plan_add(scope.handle, 1, 91005, bindings, 1, 1e9, 128, 1))
    scope.check(memo_runtime.fort_scope_plan_validate(scope.handle))
    # Access preparation invalidates even a previously built query snapshot.
    scope.cpu_begin(handle, access(flags=1))
    out = Decision()
    assert memo_runtime.fort_scope_plan_select(scope.handle, c.byref(costs()), 1, c.byref(out)) == 8
    assert b'planning during a prepared buffer access' in memo_runtime.fort_scope_error()
    assert not decision_rows(capfd)
    scope.cpu_end(handle)
    assert select(scope, costs()).gpu_units == 1
    rows = decision_rows(capfd)
    assert len(rows) == 1
    assert rows[0]['placement_memo_hit']
    assert scope.stats().uploads == scope.stats().downloads == scope.stats().allocations == 0
    scope.check(memo_runtime.fort_scope_plan_abandon_remaining_v1(scope.handle))
    scope.close()
    assert list(values) == list(range(16))


@pytest.mark.skipif(not hasattr(os, 'sched_getaffinity'), reason='Linux affinity metadata required')
def test_current_affinity_and_placement_environment_are_part_of_key(memo_runtime, monkeypatch, capfd):
    monkeypatch.setenv('FORT_RUNTIME_TRACE', '1')
    original = os.sched_getaffinity(0)
    if len(original) < 2:
        pytest.skip('two permitted CPUs required')
    query(memo_runtime, 91004)
    try:
        os.sched_setaffinity(0, {min(original)})
        query(memo_runtime, 91004)
    finally:
        os.sched_setaffinity(0, original)
    query(memo_runtime, 91004)
    rows = decision_rows(capfd)
    assert [row['placement_memo_hit'] for row in rows] == [False, False, True]
    with monkeypatch.context() as changed:
        changed.setenv('OMP_PLACES', '{0}')
        query(memo_runtime, 91004)
    query(memo_runtime, 91004)
    rows = decision_rows(capfd)
    assert [row['placement_memo_hit'] for row in rows] == [False, True]


CUDA_DRIVER = r'''
#include "scoped_runtime.h"
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
void check(int status) {
 if(status){std::fprintf(stderr,"status=%d %s\n",status,fort_scope_error());std::abort();}
}
__global__ void transform(const double *input,double *output,double add,double factor) {
 const int i=threadIdx.x;if(i<8)output[i]=(input[i]+add)*factor;
}
fort_scope_plan_costs model() {
 fort_scope_plan_costs c{};c.version=FORT_SCOPE_PLANNING_ABI_VERSION;c.valid=1;c.max_allocation_bytes=1u<<26;
 c.cpu_flops=c.cpu_bandwidth=1e8;
 c.gpu_flops=c.gpu_bandwidth=c.h2d_bandwidth=c.d2h_bandwidth=1e12;
 c.create_seconds=c.register_seconds=c.host_access_seconds=c.device_access_seconds=1e-9;
 c.gpu_setup_seconds=c.allocation_seconds=c.release_seconds=1e-9;
 c.wait_seconds=c.launch_enqueue_seconds=c.planning_operation_seconds=1e-9;
 c.cold_driver_startup_seconds=20;return c;
}
void execute(int round,int compatible=1,bool expensive_gpu=false,uint64_t offset=0,size_t budget=0,
             bool undefined=false) {
 fort_scope_t context=0;check(fort_scope_create(0,&context));
 check(fort_scope_task_graph_enable_v1(context));
 if(budget)check(fort_scope_set_device_budget(context,budget));
 double a[8],b[8]{},out[8]{};for(int i=0;i<8;++i)a[i]=100*round+i;
 size_t extent[1]={8};int64_t lower[1]={-7};fort_buffer_t handles[3]{};
 double *host[3]={a,b,out};
 for(int i=0;i<3;++i){fort_scope_layout layout{1,FORT_SCOPE_REAL64,8,host[i],extent,lower,1};
  check(fort_scope_register(context,10+i,1,&layout,i==0&&!undefined,&handles[i]));}
 fort_scope_access read{},write{};read.flags=FORT_SCOPE_READ_ALL;
 write.flags=FORT_SCOPE_WRITE_ALL|FORT_SCOPE_OVERWRITE_ALL;
 fort_scope_plan_binding first[2]={{handles[0],read},{handles[1],write}};
 fort_scope_plan_binding second[2]={{handles[1],read},{handles[2],write}};
 check(fort_scope_plan_reset_mode(context,FORT_SCOPE_PLAN_CONTINUE));
 check(fort_scope_plan_add(context,FORT_SCOPE_PLAN_WORKER,51001+offset,first,2,1e9,128,1));
 check(fort_scope_plan_add(context,FORT_SCOPE_PLAN_WORKER,51002+offset,second,2,1e9,128,1));
 if(undefined){
  assert(fort_scope_plan_validate(context)==FORT_SCOPE_UNINITIALIZED);
  auto c=model();fort_scope_plan_decision rejected{};
  assert(fort_scope_plan_select(context,&c,1,&rejected)==FORT_SCOPE_UNINITIALIZED);
  fort_scope_stats s{};check(fort_scope_stats_get(context,&s));assert(!s.allocations && !s.launches);
  check(fort_scope_close(context));return;
 }
 check(fort_scope_plan_validate(context));auto c=model();if(expensive_gpu)c.gpu_flops=1;
 fort_scope_plan_decision selected{};check(fort_scope_plan_select(context,&c,compatible,&selected));
 assert(selected.gpu_units==((compatible&&!expensive_gpu)?2u:0u));
 if(!compatible)assert(!selected.available);
 fort_scope_plan_report report{};check(fort_scope_plan_report_v2(context,&report));
 assert(!report.available && !report.owner_available);
 fort_scope_stats s{};check(fort_scope_stats_get(context,&s));assert(!s.allocations && !s.uploads && !s.launches);
 double add=2+round,factor=3;const bool gpu=selected.gpu_units;
 for(int step=0;step<2;++step){
  auto *bindings=step?second:first;int chosen=-1;
  check(fort_scope_plan_next(context,51001+offset+step,bindings,2,&chosen));assert(bool(chosen)==gpu);
  if(gpu){
   fort_buffer_t requested[2]={handles[step],handles[step+1]};void *pointers[2]{};
   check(fort_scope_device_reserve_many_v1(context,requested,2,pointers));
   auto *input=static_cast<double*>(pointers[0]);auto *output=static_cast<double*>(pointers[1]);
   double extra=step?0:add,scale=step?factor:1;void *args[4]={&input,&output,&extra,&scale};
   size_t sizes[4]={sizeof(input),sizeof(output),sizeof(extra),sizeof(scale)};
   fort_scope_graph_task_v1 task{{1,0,reinterpret_cast<const void*>(transform),{1,1,1},{32,1,1},0,args,4},bindings,2};
   check(fort_scope_graph_enqueue_v1(context,51001+offset+step,&task,sizes));
  }else{
   check(fort_scope_access_begin_v1(context,bindings,2,0,nullptr));
   for(int i=0;i<8;++i)host[step+1][i]=(host[step][i]+(step?0:add))*(step?factor:1);
   check(fort_scope_host_end(context,handles[step]));check(fort_scope_host_end(context,handles[step+1]));
  }
 }
 if(gpu){
  check(fort_scope_observe_v1(context,&handles[2],1));
  for(int i=0;i<8;++i)assert(out[i]==(a[i]+add)*factor);
  check(fort_scope_host_begin(context,handles[1],&read));
  for(int i=0;i<8;++i)assert(b[i]==a[i]+add);
  check(fort_scope_host_end(context,handles[1]));
  check(fort_scope_stats_get(context,&s));const auto downloads=s.download_bytes;
  // CPU -> GPU read retained the original host mirror.
  check(fort_scope_host_begin(context,handles[0],&read));check(fort_scope_host_end(context,handles[0]));
  check(fort_scope_stats_get(context,&s));assert(s.download_bytes==downloads);
  // GPU -> CPU read retained B's device mirror for another GPU consumer.
  fort_buffer_t requested[2]={handles[1],handles[2]};void *pointers[2]{};
  check(fort_scope_device_reserve_many_v1(context,requested,2,pointers));
  auto *input=static_cast<double*>(pointers[0]);auto *output=static_cast<double*>(pointers[1]);
  double extra=0,scale=factor+1;void *args[4]={&input,&output,&extra,&scale};
  size_t sizes[4]={sizeof(input),sizeof(output),sizeof(extra),sizeof(scale)};
  fort_scope_graph_task_v1 task{{1,0,reinterpret_cast<const void*>(transform),{1,1,1},{32,1,1},0,args,4},second,2};
  check(fort_scope_graph_enqueue_v1(context,51002+offset,&task,sizes));
  check(fort_scope_observe_v1(context,&handles[2],1));
  for(int i=0;i<8;++i)assert(out[i]==(a[i]+add)*scale);
  check(fort_scope_stats_get(context,&s));assert(s.upload_bytes==64 && s.download_bytes==192 && s.launches==3);
 }else{
  for(int i=0;i<8;++i)assert(out[i]==(a[i]+add)*factor);
  check(fort_scope_stats_get(context,&s));assert(!s.allocations && !s.uploads && !s.downloads && !s.launches);
 }
 check(fort_scope_close(context));
}
int main(){
 int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices)return 77;
 execute(0);execute(1);execute(2); // Cold miss, warm miss, exact warm hit.
 execute(3,1,false,10); // Changed compiled identity must miss.
 execute(4,1,false,0,128); // Changed budget is repriced and must miss.
 execute(5,0); // Cached candidate cannot authorize incompatible hardware.
 execute(6,1,true); // Changed current numeric costs select native.
 execute(7,1,false,0,0,true); // A current undefined read fails before memo.
 std::puts("actual CUDA memo and mirrors verified");
}
'''


@pytest.fixture(scope='module')
def cuda_memo_binary(tmp_path_factory):
    prepared = os.environ.get('FORT_TEST_MEMO_PREBUILT')
    if prepared:
        return Path(prepared)
    nvcc = shutil.which('nvcc')
    host = shutil.which('g++-14') or shutil.which('g++')
    if nvcc is None or host is None:
        pytest.skip('CUDA toolkit required')
    directory = tmp_path_factory.mktemp('placement-memo-cuda')
    source = directory / 'client.cu'
    source.write_text(CUDA_DRIVER)
    objects = []
    for name, path in [('runtime', RUNTIME / 'scoped_runtime.cu'), ('client', source)]:
        target = directory / (name + '.o')
        result = subprocess.run([nvcc, '-std=c++17', '-ccbin', host, '-arch=sm_86', '-I', str(RUNTIME),
                                 '-c', str(path), '-o', str(target)], capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
        objects.append(str(target))
    target = directory / 'run'
    result = subprocess.run([nvcc, '-ccbin', host, *objects, '-o', str(target)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return target


@pytest.mark.cuda
def test_actual_cuda_memo_reprices_current_data_and_preserves_read_mirrors(cuda_memo_binary):
    result = subprocess.run([str(cuda_memo_binary)], capture_output=True, text=True,
                            env={**os.environ, 'FORT_RUNTIME_TRACE': '1'}, timeout=30)
    if result.returncode == 77:
        pytest.skip('CUDA device unavailable')
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [json.loads(line[line.index('{'):]) for line in result.stderr.splitlines() if '"event":"decision"' in line]
    decisions = [row for row in rows if row.get('event') == 'decision']
    assert len(decisions) == 7
    assert [row['placement_memo_hit'] for row in decisions] == [False, False, True, False, False, False, False]
    assert all(row['unpriced_memo_overhead'] for i, row in enumerate(decisions) if i != 5)
    assert 'actual CUDA memo and mirrors verified' in result.stdout
