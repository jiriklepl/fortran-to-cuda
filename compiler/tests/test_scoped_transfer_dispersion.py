"""Measured transfer-fit scenarios cannot promote a nominally losing schedule."""
# ruff: noqa: F811 - imported pytest fixtures.
from __future__ import annotations

import ctypes as c
import json
import mmap
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from compiler.emission.cuda.batch import base_cost_fields
from compiler.offload.profile import ProfileError, validate_profile
from compiler.tests.test_scoped_planning_entries import calibration
from compiler.tests.test_scoped_planning_runtime import Costs, Decision, costs, planning, record, select  # noqa: F401
from compiler.tests.test_scoped_runtime import TOKEN, Scope, access, runtime  # noqa: F401

RUNTIME = Path(__file__).resolve().parents[1] / 'runtime'
DRIVER = r'''
#include "scoped_planning.hpp"
#include "scoped_memo.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
using namespace fort_scoped::planning;
fort_scope_plan_costs costs() {
 fort_scope_plan_costs c{};c.version=FORT_SCOPE_PLANNING_ABI_VERSION;c.valid=1;c.max_allocation_bytes=1<<20;
 c.cpu_flops=1;c.cpu_bandwidth=1e12;c.gpu_flops=100;c.gpu_bandwidth=1e12;
 c.h2d_bandwidth=c.d2h_bandwidth=1e12;c.h2d_latency=c.d2h_latency=.4;
 c.create_seconds=c.register_seconds=c.host_access_seconds=c.device_access_seconds=1e-10;
 c.gpu_setup_seconds=c.cold_driver_startup_seconds=c.allocation_seconds=c.release_seconds=1e-10;
 c.wait_seconds=c.launch_enqueue_seconds=c.planning_operation_seconds=1e-10;return c;
}
Inputs inputs(bool device=false, double work=1) {
 Inputs in;in.graph_mode=in.continuation=true;in.driver_initialized=true;
 Resource b;b.handle=1;b.element_bytes=8;b.bytes=64;b.extents={8};
 Region full={Box{{0},{8}}};b.initialized=full;
 if(device){b.device_current=full;b.allocated=true;in.device_ready=true;}else b.host_current=full;
 in.resources={b};Effects effects=device?Effects{full,{},{}}:Effects{full,full,{}};
 in.operations={{FORT_SCOPE_PLAN_WORKER,1,{{1,effects}},work,0,true}};return in;
}
void close(double a,double b){assert(std::abs(a-b)<1e-10*std::max(1.,std::max(std::abs(a),std::abs(b))));}
int main() {
 auto c=costs();auto in=inputs();auto nominal=select(in,c);
 assert(nominal.decision.available && nominal.decision.gpu_units==1);
 c.h2d_fit_relative_error=c.d2h_fit_relative_error=.25;
 auto guarded=select(in,c);assert(guarded.decision.available && !guarded.decision.gpu_units);
 assert(guarded.reason=="native_graph_transfer_fit_dispersion_not_met");
 auto other_request=in;other_request.direct_transfer_fit_applicable=false;
 assert(select(other_request,c).decision.gpu_units==1);
 auto strong=select(inputs(false,10),c);assert(strong.decision.gpu_units==1);
 assert(strong.graph_transfer_dispersion_seconds>0);
 assert(strong.graph_ranking_seconds+strong.graph_transfer_dispersion_seconds <=
        strong.graph_counterfactual_seconds-strong.graph_required_saving_seconds);
 // The same final download belongs to both schedules and cancels. Read-only
 // access does not invalidate either mirror or pay a second upload.
 auto resident=inputs(true);std::vector<bool> cpu{false},gpu{true};uint64_t work=0;
 auto a=detail::simulate(resident,c,gpu,work),b=detail::simulate(resident,c,cpu,work);
 assert(a.terminal.downloads==1 && b.count.downloads==1 && b.terminal.downloads==0);
 close(a.d2h_seconds,b.d2h_seconds);close(detail::transfer_dispersion_debit(a,b,c),0);
 assert(detail::transfer_dispersion_debit(a,b,c,false)>0);
 assert(select(resident,c).decision.gpu_units==1);
 // Algebra against all four endpoints of the two independent direction
 // coefficients. A negative transfer delta loses part of its saving.
 std::mt19937 random(315091);
 for(int n=0;n<400;++n){
  detail::State x,y;x.h2d_seconds=random()%1000/100.;x.d2h_seconds=random()%1000/100.;
  y.h2d_seconds=random()%1000/100.;y.d2h_seconds=random()%1000/100.;
  c.h2d_fit_relative_error=random()%90/100.;c.d2h_fit_relative_error=random()%90/100.;
  double worst=-1e100;const double dh=x.h2d_seconds-y.h2d_seconds,dd=x.d2h_seconds-y.d2h_seconds;
  for(int h:{-1,1})for(int d:{-1,1})worst=std::max(worst,
   dh/(1+h*c.h2d_fit_relative_error)+dd/(1+d*c.d2h_fit_relative_error));
  const double debit=detail::transfer_dispersion_debit(x,y,c);assert(debit>=0);close(dh+dd+debit,worst);
 }
 in.definitions_validated=true;
 const std::vector<uint64_t> environment{1};const auto original=fort_scoped::memo::make_key(in,c,environment);
 assert(original);
 for(int k=0;k<3;++k){auto input=in;auto changed=c;
  if(k==0)changed.h2d_fit_relative_error+=.001;
  if(k==1)changed.d2h_fit_relative_error+=.001;
  if(k==2)input.direct_transfer_fit_applicable=false;
  const auto key=fort_scoped::memo::make_key(input,changed,environment);
  assert(key && key->words!=original->words);
 }
 for(double bad:{-1.,1.,2.,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}){
  c=costs();c.h2d_fit_relative_error=bad;auto invalid=select(in,c);
  assert(!invalid.decision.available && invalid.reason=="invalid_transfer_fit_dispersion");
 }
 std::cout<<"{\"nominal_gpu\":1,\"guarded_gpu\":0,\"strong_gpu\":1,\"corner_trials\":400}";
}
'''


def test_transfer_scenario_gates_and_correlated_coefficient_algebra(tmp_path):
    cc = shutil.which('g++-14') or shutil.which('g++')
    if not cc:
        pytest.skip('C++ compiler unavailable')
    source, binary = tmp_path / 'driver.cpp', tmp_path / 'run'
    source.write_text(DRIVER)
    result = subprocess.run([cc, '-std=c++17', '-O2', '-I', str(RUNTIME), str(source), '-o', str(binary)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)['corner_trials'] == 400


@pytest.mark.parametrize('requested_mode', [0, 2, 3, 4])
def test_preview_reprices_changed_dispersion_without_array_or_gpu_activity(planning, requested_mode):
    planning.fort_scope_task_graph_enable_v1.argtypes = [TOKEN]
    planning.fort_scope_set_transfers.argtypes = [TOKEN, c.c_uint32]
    scope = Scope(planning)
    scope.check(planning.fort_scope_set_transfers(scope.handle, requested_mode))
    values = (c.c_double * 8)(*range(8))
    status, handle = scope.register(values, [8])
    scope.check(status)
    scope.check(planning.fort_scope_task_graph_enable_v1(scope.handle))
    scope.check(planning.fort_scope_plan_reset(scope.handle))
    bindings = record(scope, 117, handle, access(flags=3), flops=1)
    model = costs()
    model.cpu_flops, model.gpu_flops = 1, 100
    model.h2d_latency = model.d2h_latency = .4
    model.cold_driver_startup_seconds = 1e-9
    assert select(scope, model, -1).gpu_units == 1
    model.h2d_fit_relative_error = model.d2h_fit_relative_error = .25
    assert select(scope, model).gpu_units == (0 if requested_mode == 0 else 1)
    assert scope.stats().uploads == scope.stats().downloads == scope.stats().launches == 0
    chosen = c.c_int()
    scope.check(planning.fort_scope_plan_next(scope.handle, 117, bindings, 1, c.byref(chosen)))
    assert chosen.value == (0 if requested_mode == 0 else 1)
    scope.close()
    assert list(values) == list(range(8))


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux guard-page diagnostic')
@pytest.mark.parametrize('version', [0, 1, 77])
def test_old_short_cost_header_is_not_read_beyond_its_version(planning, version):
    scope = Scope(planning)
    scope.check(planning.fort_scope_plan_reset(scope.handle))
    memory = mmap.mmap(-1, 2 * mmap.PAGESIZE)
    address = c.addressof(c.c_char.from_buffer(memory))
    old = (c.c_uint32 * 2).from_address(address + mmap.PAGESIZE - 8)
    old[0], old[1] = version, 1
    libc = c.CDLL(None, use_errno=True)
    libc.mprotect.argtypes = [c.c_void_p, c.c_size_t, c.c_int]
    assert libc.mprotect(address + mmap.PAGESIZE, mmap.PAGESIZE, 0) == 0
    try:
        decision = Decision()
        status = planning.fort_scope_plan_select(scope.handle, c.cast(old, c.POINTER(Costs)), 1, c.byref(decision))
        assert status == (0 if version == 0 else 1)
        if version == 0:
            assert not decision.available
    finally:
        assert libc.mprotect(address + mmap.PAGESIZE, mmap.PAGESIZE, mmap.PROT_READ | mmap.PROT_WRITE) == 0
        memory.close()
    scope.close()


@pytest.mark.parametrize('bad', [-1, float('nan'), float('inf')])
def test_profile_rejects_invalid_observed_residual(bad):
    value = calibration()
    value['rates']['h2d_pageable']['fit_max_relative_error'] = bad
    with pytest.raises(ProfileError, match='fit_max_relative_error'):
        validate_profile(value)


def test_shared_emitter_cost_fields_preserve_observations():
    value = calibration()
    value['rates']['h2d_pageable']['fit_max_relative_error'] = .05
    value['rates']['d2h_pageable']['fit_max_relative_error'] = .25
    fields = base_cost_fields(value, value['scoped']['costs'])
    assert fields['h2d_fit_relative_error'] == .05
    assert fields['d2h_fit_relative_error'] == .25
    assert fields['h2d_bandwidth'] == value['rates']['h2d_pageable']['bandwidth_bytes_per_second']
