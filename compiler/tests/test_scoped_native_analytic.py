"""Stationary native candidates avoid physical snapshots without cost authority."""
from __future__ import annotations

import ctypes as c
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from compiler.tests.test_scoped_placement_memo import memo_runtime  # noqa: F401
from compiler.tests.test_scoped_planning_runtime import Binding, Decision, costs, planning, select  # noqa: F401
from compiler.tests.test_scoped_runtime import Scope, access, runtime  # noqa: F401

RUNTIME = Path(__file__).resolve().parents[1] / 'runtime'
DRIVER = r'''
#include "scoped_planning.hpp"
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
size_t allocations=0, fail_allocation=0;
__attribute__((noinline)) void *operator new(size_t bytes){++allocations;if(fail_allocation==allocations){fail_allocation=0;throw std::bad_alloc{};}void *p=std::malloc(bytes ? bytes : 1);if(!p)throw std::bad_alloc{};return p;}
__attribute__((noinline)) void operator delete(void *p) noexcept{std::free(p);}
__attribute__((noinline)) void operator delete(void *p,size_t) noexcept{std::free(p);}
void *operator new[](size_t bytes){return ::operator new(bytes);}
void operator delete[](void *p) noexcept{::operator delete(p);}
void operator delete[](void *p,size_t) noexcept{::operator delete(p);}
using namespace fort_scoped::planning;
fort_scope_plan_costs costs(){fort_scope_plan_costs c{};c.version=FORT_SCOPE_PLANNING_ABI_VERSION;c.valid=1;c.max_allocation_bytes=1<<20;
 c.cpu_flops=1e6;c.cpu_bandwidth=2e6;c.gpu_flops=1e10;c.gpu_bandwidth=2e10;c.h2d_bandwidth=c.d2h_bandwidth=1e11;
 c.h2d_latency=c.d2h_latency=0;
 c.create_seconds=2e-7;c.register_seconds=3e-7;c.host_access_seconds=4e-7;c.device_access_seconds=5e-7;
 c.gpu_setup_seconds=c.allocation_seconds=c.release_seconds=c.wait_seconds=c.launch_enqueue_seconds=1e-8;
 c.cold_driver_startup_seconds=20;c.planning_operation_seconds=6e-9;return c;}
Resource resource(fort_buffer_t h,bool blank=false){Resource b;b.handle=h;b.element_bytes=8;b.bytes=2048;b.extents={16,16};
 if(!blank)b.initialized=b.host_current={Box{{2,3},{14,13}}};
 return b;}
Operation worker(uint64_t unit,fort_buffer_t h,Effects e,bool gpu=false){Operation op;op.kind=FORT_SCOPE_PLAN_WORKER;
 op.unit=unit;op.bindings={{h,e}};op.flops=10000;op.memory_bytes=4096;op.gpu_available=gpu;return op;}
Inputs basic(){Inputs in;in.graph_mode=true;in.continuation=true;in.resources={resource(1)};
 auto full=in.resources[0].initialized;in.operations={worker(1,1,{full,full,{}})};return in;}
void close(double a,double b){assert(std::abs(a-b)<=1e-12*std::max(1.,std::max(std::abs(a),std::abs(b))));}
void exact(){std::mt19937 rng(91573);const auto c=costs();
 for(int trial=0;trial<240;++trial){Inputs in;in.graph_mode=true;in.continuation=trial%2;in.driver_initialized=trial%3;
  in.charge_create=trial%5;in.registrations_incurred=trial%4;in.query_construction_operations=trial%11;
  const size_t n=1+rng()%6;
  for(size_t k=0;k<n;++k)in.resources.push_back(resource(10+k,k%2));
  // First complete overwrite defines each initially blank interior exactly.
  for(size_t k=0;k<n;++k)if(k%2){Region box={Box{{2,3},{14,13}}};in.operations.push_back(worker(100+k,10+k,{{},box,box}));}
  for(size_t k=0;k<25;++k){size_t h=10+rng()%n;const Region small={Box{{3,4},{5+rng()%8,6+rng()%6}}};
   auto op=worker(200+k,h,{small, k%3?small:Region{}, k%3 && k%4==0?small:Region{}});
   if(k%5==0){op.kind=FORT_SCOPE_PLAN_NATIVE;op.unit=0;op.flops=0;op.memory_bytes=0;}
   if(k%7==0){Operation marker;marker.kind=FORT_SCOPE_PLAN_TEAM_ENTRY;in.operations.push_back(marker);}
   in.operations.push_back(op);
  }
  assert(validate_definitions(in).status==FORT_SCOPE_OK);in.definitions_validated=trial%2;
  const size_t workers=std::count_if(in.operations.begin(),in.operations.end(),[](const auto &o){return o.kind==FORT_SCOPE_PLAN_WORKER;});
  const std::vector<bool> cpu(workers,false);uint64_t work=0;
  auto physical=detail::simulate(in,c,cpu,work);auto result=select(in,c);
  assert(result.native_analytic_shortcut && !result.native_startup_shortcut);
  assert(result.decision.available && !result.report.available && !result.report.owner_available);
  assert(result.gpu_workers==cpu && result.decision.cpu_units==workers && !result.decision.gpu_units);
  assert(!result.decision.uploads && !result.decision.downloads && !result.decision.waits && !result.decision.allocations);
  const double planning=result.decision.simulated_operations*c.planning_operation_seconds;
  close(result.report.execution_seconds-planning,physical.execution_time);
  close(result.report.ranking_seconds-planning,physical.time);
  close(result.report.native_execution_seconds,result.report.execution_seconds);
  close(result.report.terminal.seconds,0);close(result.report.entry_terminal.seconds,0);
  if(in.continuation)close(result.decision.native_seconds,result.report.native_execution_seconds);
  assert(result.native_common_compute_excluded && result.unpriced_team_overhead);
 }
}
void guards(){auto c=costs();auto in=basic();
 auto normal=[&](Inputs q){auto r=select(q,c);assert(!r.native_analytic_shortcut);assert(!r.unpriced_native_analytic_overhead || !r.report.available);return r;};
 in.graph_mode=false;assert(normal(in).decision.available);in=basic();
 in.device_ready=true;assert(normal(in).decision.available);in=basic();
 in.pending=true;assert(normal(in).decision.available);in=basic();
 in.resources[0].allocated=true;assert(normal(in).decision.available);in=basic();
 in.resources[0].allocated=true;in.resources[0].device_current=in.resources[0].initialized;
 assert(normal(in).decision.available);in=basic();
 in.resources[0].pending_host_reads=in.resources[0].initialized;assert(normal(in).decision.waits==1);in=basic();
 // A semantic kill or physical footprint expansion must use normal simulation.
 in.operations.insert(in.operations.begin(),Operation{FORT_SCOPE_PLAN_FORGET,0,{{1,{}}}});
 in.operations[1].bindings[0].effects.reads.clear();in.operations[1].bindings[0].effects.overwrites=in.resources[0].initialized;
 assert(normal(in).decision.available);in=basic();
 in.operations.insert(in.operations.begin(),Operation{FORT_SCOPE_PLAN_DISCARD,0,{{1,{{},in.resources[0].initialized,{}}}}});
 in.operations[1].bindings[0].effects.reads.clear();in.operations[1].bindings[0].effects.overwrites=in.resources[0].initialized;
 assert(normal(in).decision.available);in=basic();
 auto extra=Box{{0,0},{1,1}};in.operations[0].bindings[0].effects={{},{extra},{extra}};
 assert(normal(in).decision.available);in=basic();
 in.resources[0].host_current={Box{{0,0},{16,16}}};assert(normal(in).decision.available);in=basic();
 in.resources[0].initialized.push_back(extra);in.resources[0].host_current=in.resources[0].initialized;
 assert(normal(in).decision.available);in=basic();
 in.resources[0].initialized.clear();in.resources[0].host_current.clear();
 in.operations[0].bindings[0].effects={{},{Box{{2,3},{14,13}},extra},{Box{{2,3},{14,13}},extra}};
 assert(normal(in).decision.available);in=basic();
 fort_scope_compute_costs_v1 backend{};backend.version=1;backend.known_backends=FORT_SCOPE_COMPUTE_ALL;
 backend.native_fortran_seconds=.2;backend.generated_cpu_seconds=.7;backend.gpu_seconds=.01;
 in.operations[0].compute_costs=backend;assert(normal(in).decision.available);in=basic();
 fort_scope_team_costs team{};team.version=FORT_SCOPE_TEAM_ABI_VERSION;team.valid=1;team.cpu_threads=4;
 team.expected_omp_level=1;team.protocol_id=FORT_SCOPE_TEAM_PROTOCOL_ID;team.native_cpu_flops=1e5;team.native_cpu_bandwidth=2e5;
 team.entry_seconds=.01;team.native_call_seconds=.02;team.cpu_worker_seconds=.03;team.native_worker_seconds=.04;
 in.team_costs=team;in.continuation=false;assert(normal(in).decision.available);in=basic();
 in.operations[0].bindings[0].buffer=999;assert(!normal(in).decision.available);in=basic();
 in.operations[0].bindings[0].effects.overwrites={Box{{0,0},{16,16}}};assert(!normal(in).decision.available);in=basic();
 in.resources[0].initialized.clear();in.resources[0].host_current.clear();assert(!normal(in).decision.available);in=basic();
 in.operations[0].memory_bytes=std::numeric_limits<double>::quiet_NaN();assert(!normal(in).decision.available);in=basic();
 c.valid=0;assert(!normal(in).decision.available);c=costs();in=basic();
 // Missing/current GPU candidates retain the ordinary cold/warm ranking path.
 in.operations[0].gpu_available=true;auto first=select(in,c);assert(first.decision.gpu_units==1 && !first.native_analytic_shortcut);
 assert(first.cold_driver_startup_debt_seconds==20);
 std::vector<bool> gpu{true},cpu{false};auto chosen_gpu=select(in,c,&gpu);assert(chosen_gpu.decision.gpu_units==1 && !chosen_gpu.native_analytic_shortcut);
 auto chosen_cpu=select(in,c,&cpu);assert(chosen_cpu.native_analytic_shortcut && !chosen_cpu.decision.gpu_units);
 // A declined analytic guard followed by the existing startup bound retains
 // both kinds of actual logical input scans in its work counter.
 auto edge=basic();edge.operations[0].gpu_available=true;edge.resources[0].initialized.push_back(extra);
 edge.resources[0].host_current=edge.resources[0].initialized;auto costly_setup=c;costly_setup.gpu_setup_seconds=.1;
 auto startup=select(edge,costly_setup,&cpu);
 assert(startup.native_startup_shortcut && !startup.native_analytic_shortcut);
 assert(startup.unpriced_native_analytic_overhead && !startup.report.available);
 assert(startup.decision.simulated_operations==5);
 auto count=chosen_cpu.decision.simulated_operations;
 assert(count==in.operations.size()+in.resources.size()+1+1+in.query_construction_operations);
}
void heap(){Inputs in;in.graph_mode=true;in.continuation=true;in.definitions_validated=true;
 for(size_t k=1;k<=16;++k)in.resources.push_back(resource(k));
 for(size_t k=0;k<48;++k){auto op=worker(1+k,1,in.resources[0].initialized.empty()?Effects{}:Effects{in.resources[0].initialized,in.resources[0].initialized,{}});
  for(size_t h=2;h<=16;++h)op.bindings.push_back({h,{in.resources[h-1].initialized,in.resources[h-1].initialized,{}}});
  in.operations.push_back(op);}
 auto c=costs();std::vector<bool> cpu(48,false);uint64_t work=0;
 allocations=0;detail::validate(in,c);const auto validation_allocations=allocations;(void)detail::entry_terminal(in,c,work);(void)detail::simulate(in,c,cpu,work);const auto old=allocations;
 allocations=0;const auto result=select(in,c);const auto now=allocations;
 assert(result.native_analytic_shortcut && now*2<old);
 allocations=0;fail_allocation=validation_allocations+2;const auto fallback=select(in,c);
 assert(!fail_allocation && fallback.decision.available && !fallback.native_analytic_shortcut);
 assert(fallback.unpriced_native_analytic_overhead && !fallback.report.available);
 std::cout<<"metadata allocations baseline="<<old<<" analytic="<<now<<"\n";
}
int main(int argc,char**argv){assert(argc==2);std::string mode=argv[1];
 if(mode=="exact")exact();else if(mode=="guards")guards();else if(mode=="heap")heap();else return 2;
 std::cout<<mode<<" native analytic verified\n";}
'''


@pytest.fixture(scope='module')
def analytic_binary(tmp_path_factory):
    compiler = shutil.which('g++-14') or shutil.which('g++')
    if compiler is None:
        pytest.skip('C++ compiler required')
    directory = tmp_path_factory.mktemp('native-analytic')
    source = directory / 'native.cpp'
    source.write_text(DRIVER)
    target = directory / 'run'
    result = subprocess.run([compiler, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                             '-I', str(RUNTIME), str(source), '-o', str(target)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    return target


@pytest.mark.parametrize('mode', ['exact', 'guards', 'heap'])
def test_stationary_native_matches_physical_model(analytic_binary, mode):
    result = subprocess.run([str(analytic_binary), mode], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def evidence(capfd):
    return [json.loads(line[line.index('{'):]) for line in capfd.readouterr().err.splitlines()
            if '"event":"decision"' in line]


def native_query(lib, *, malformed=False, initialized=True, gpu_available=False, unit=91100):
    scope = Scope(lib)
    values = (c.c_double * 16)(*range(16))
    status, handle = scope.register(values, [16], identity=unit, initialized=initialized)
    scope.check(status)
    scope.check(lib.fort_scope_task_graph_enable_v1(scope.handle))
    scope.check(lib.fort_scope_plan_reset_mode(scope.handle, 1))
    effects = access(read=[([0], [16])], write=[([0], [1])])
    bindings = (Binding * 1)(Binding(handle, effects))
    scope.check(lib.fort_scope_plan_add(scope.handle, 1, unit, bindings, 1,
                                      1e9, float('nan') if malformed else 128, int(gpu_available)))
    return scope, values, handle, bindings


def test_native_only_bypasses_memo_but_installs_exact_choices(memo_runtime, monkeypatch, capfd):  # noqa: F811
    monkeypatch.setenv('FORT_RUNTIME_TRACE', '1')
    for _ in range(2):
        scope, values, handle, bindings = native_query(memo_runtime)
        decision = select(scope, costs())
        assert decision.available == 1
        assert decision.cpu_units == 1
        assert decision.gpu_units == 0
        chosen = c.c_int(99)
        scope.check(memo_runtime.fort_scope_plan_next(scope.handle, 91100, bindings, 1, c.byref(chosen)))
        assert chosen.value == 0
        scope.cpu_begin(handle, bindings[0].access)
        values[0] = 42
        scope.cpu_end(handle)
        scope.close()
        assert list(values) == [42] + list(range(1, 16))
    rows = evidence(capfd)
    assert len(rows) == 2
    assert all(row['aggregate_only'] and row['evidence'] == 'native_stationary_single_candidate' for row in rows)
    assert all(row['unpriced_native_analytic_overhead'] for row in rows)
    assert not any(row['unpriced_memo_overhead'] or row['placement_memo_hit'] or row['placement_memo_key_seconds'] for row in rows)
    assert all(row['continuation']['available'] == 0 and row['continuation']['owner_available'] == 0 for row in rows)


def test_native_only_bypass_preserves_current_definition_and_cost_guards(memo_runtime, monkeypatch, capfd):  # noqa: F811
    monkeypatch.setenv('FORT_RUNTIME_TRACE', '1')
    scope, values, _, _ = native_query(memo_runtime, malformed=True)
    decision = select(scope, costs())
    assert not decision.available
    assert not decision.gpu_units
    scope.check(memo_runtime.fort_scope_plan_abandon_remaining_v1(scope.handle))
    scope.close()
    rows = evidence(capfd)
    assert len(rows) == 1
    assert not rows[0]['unpriced_memo_overhead']
    assert not rows[0]['unpriced_native_analytic_overhead']
    scope, values, _, _ = native_query(memo_runtime, initialized=False)
    decision = Decision()
    assert memo_runtime.fort_scope_plan_select(scope.handle, c.byref(costs()), 1, c.byref(decision)) == 7
    assert not evidence(capfd)
    scope.close()
    assert list(values) == list(range(16))


def test_cached_all_cpu_candidate_keeps_order_and_uses_current_accounting(memo_runtime, monkeypatch, capfd):  # noqa: F811
    monkeypatch.setenv('FORT_RUNTIME_TRACE', '1')
    calibration = costs()
    calibration.gpu_flops = 1
    for _ in range(2):
        scope, values, handle, bindings = native_query(memo_runtime, gpu_available=True, unit=91101)
        decision = select(scope, calibration)
        assert decision.available
        assert decision.cpu_units == 1
        assert decision.gpu_units == 0
        chosen = c.c_int(99)
        assert memo_runtime.fort_scope_plan_next(scope.handle, 91102, bindings, 1, c.byref(chosen)) == 9
        scope.check(memo_runtime.fort_scope_plan_next(scope.handle, 91101, bindings, 1, c.byref(chosen)))
        assert chosen.value == 0
        scope.cpu_begin(handle, bindings[0].access)
        values[0] += 100
        scope.cpu_end(handle)
        assert scope.stats().uploads == scope.stats().downloads == scope.stats().allocations == 0
        scope.close()
        assert list(values) == [100] + list(range(1, 16))
    rows = evidence(capfd)
    assert len(rows) == 2
    assert not rows[0]['placement_memo_hit']
    assert rows[1]['placement_memo_hit']
    assert not rows[0]['unpriced_native_analytic_overhead']
    assert rows[1]['unpriced_native_analytic_overhead']
    assert rows[1]['aggregate_only']
    assert all(row['unpriced_memo_overhead'] and row['continuation']['available'] == 0 for row in rows)
