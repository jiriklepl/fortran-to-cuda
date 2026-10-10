"""Context scratch public ABI: lifetime, budget, coherence, and failure safety."""

from __future__ import annotations

import ctypes as c
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from compiler.tests.test_scoped_planning_runtime import Binding, Costs, Decision, costs, record, select
from compiler.tests.test_scoped_runtime import TOKEN, Access, Layout, Scope, Stats, access

RUNTIME = Path(__file__).resolve().parents[1] / "runtime"


class Lease(c.Structure):
    _fields_ = [
        ("version", c.c_uint32), ("reserved", c.c_uint32), ("token", TOKEN),
        ("device", c.c_void_p), ("bytes", c.c_size_t), ("capacity", c.c_size_t),
    ]


class ScratchStats(c.Structure):
    _fields_ = [("version", c.c_uint32), ("active", c.c_uint32)] + [
        (name, TOKEN) for name in (
            "acquisitions", "allocations", "reuses", "releases", "grows", "capacity_bytes",
            "active_bytes", "peak_scratch_bytes", "peak_total_device_bytes",
        )
    ]


@pytest.fixture(scope="module")
def scratch_runtime(tmp_path_factory):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("g++ unavailable")
    directory = tmp_path_factory.mktemp("scoped_scratch")
    target = directory / "runtime.so"
    result = subprocess.run([
        compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-fPIC", "-shared", "-pthread",
        "-DFORT_SCOPE_CPU_TEST", "-DFORT_SCOPE_TEST_FAULTS", "-x", "c++",
        str(RUNTIME / "scoped_runtime.cu"), "-o", str(target),
    ], capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    lib = c.CDLL(str(target))
    lib.fort_scope_error.restype = c.c_char_p
    signatures = {
        "create": [c.c_int, c.POINTER(TOKEN)],
        "set_device_budget": [TOKEN, c.c_size_t],
        "register": [TOKEN, TOKEN, TOKEN, c.POINTER(Layout), c.c_int, c.POINTER(TOKEN)],
        "device_begin": [TOKEN, TOKEN, c.POINTER(Access), c.POINTER(c.c_void_p)],
        "device_end": [TOKEN, TOKEN], "cancel_access": [TOKEN, TOKEN],
        "unregister": [TOKEN, TOKEN], "close": [TOKEN], "abandon": [TOKEN], "wait": [TOKEN],
        "note_launch": [TOKEN], "execution_error": [TOKEN, c.c_char_p],
        "stats_get": [TOKEN, c.POINTER(Stats)],
        "scratch_acquire_v1": [TOKEN, c.c_size_t, c.POINTER(Lease)],
        "scratch_release_v1": [TOKEN, TOKEN],
        "scratch_stats_get_v1": [TOKEN, c.POINTER(ScratchStats)],
        "plan_reset": [TOKEN],
        "plan_reset_mode": [TOKEN, c.c_uint32],
        "plan_add": [TOKEN, c.c_uint32, TOKEN, c.POINTER(Binding), c.c_size_t,
                     c.c_double, c.c_double, c.c_int],
        "plan_select": [TOKEN, c.POINTER(Costs), c.c_int, c.POINTER(Decision)],
    }
    for name, signature in signatures.items():
        getattr(lib, "fort_scope_" + name).argtypes = signature
    return lib


def acquire(scope, size):
    lease = Lease()
    scope.check(scope.lib.fort_scope_scratch_acquire_v1(scope.handle, size, c.byref(lease)))
    assert lease.version == 1
    assert lease.reserved == 0
    assert lease.token != 0
    assert lease.bytes == size
    assert lease.capacity >= size
    return lease


def release(scope, lease):
    scope.check(scope.lib.fort_scope_scratch_release_v1(scope.handle, lease.token))


def scratch_stats(scope):
    out = ScratchStats()
    scope.check(scope.lib.fort_scope_scratch_stats_get_v1(scope.handle, c.byref(out)))
    assert out.version == 1
    return out


@pytest.mark.native
def test_empty_lease_does_not_initialize_or_allocate(scratch_runtime, monkeypatch, capfd):
    scope = Scope(scratch_runtime)
    monkeypatch.setenv("FORT_RUNTIME_TRACE", "1")
    monkeypatch.setenv("FORT_SCOPE_TEST_FAIL_ALLOC", "1")
    lease = acquire(scope, 0)
    assert lease.device is None
    assert lease.capacity == 0
    # Budget remains configurable, proving no device initialization occurred.
    scope.check(scope.lib.fort_scope_set_device_budget(scope.handle, 0))
    stats = scratch_stats(scope)
    assert (stats.active, stats.acquisitions, stats.allocations, stats.capacity_bytes) == (1, 1, 0, 0)
    release(scope, lease)
    scope.close()
    stderr = capfd.readouterr().err
    assert "initialize" not in stderr
    assert "scratch_allocate" not in stderr


@pytest.mark.native
@pytest.mark.parametrize(("budget", "requested_bytes"), [(0, 1), (64, c.c_size_t(-1).value)])
def test_invalid_capacity_fails_before_device_initialization(
    scratch_runtime, monkeypatch, capfd, budget, requested_bytes,
):
    scope = Scope(scratch_runtime)
    monkeypatch.setenv("FORT_RUNTIME_TRACE", "1")
    scope.check(scratch_runtime.fort_scope_set_device_budget(scope.handle, budget))
    out = Lease()
    assert scratch_runtime.fort_scope_scratch_acquire_v1(scope.handle, requested_bytes, c.byref(out)) == 4
    assert bytes(out) == bytes(Lease())
    scope.check(scratch_runtime.fort_scope_set_device_budget(scope.handle, 0))
    scope.close()
    assert "initialize" not in capfd.readouterr().err


@pytest.mark.native
def test_one_active_lease_stale_foreign_and_missing_outputs(scratch_runtime):
    first, second = Scope(scratch_runtime), Scope(scratch_runtime)
    assert scratch_runtime.fort_scope_scratch_acquire_v1(first.handle, 0, None) == 1
    assert scratch_runtime.fort_scope_scratch_stats_get_v1(first.handle, None) == 1
    lease = acquire(first, 0)
    other = acquire(second, 0)
    rejected = Lease(9, 9, 99, 17, 23, 23)
    assert scratch_runtime.fort_scope_scratch_acquire_v1(first.handle, 0, c.byref(rejected)) == 8
    assert bytes(rejected) == bytes(Lease())
    assert scratch_runtime.fort_scope_scratch_release_v1(second.handle, lease.token) == 2
    assert scratch_runtime.fort_scope_scratch_release_v1(first.handle, 0) == 2
    release(first, lease)
    assert scratch_runtime.fort_scope_scratch_release_v1(first.handle, lease.token) == 2
    renewed = acquire(first, 0)
    assert len({lease.token, other.token, renewed.token}) == 3
    assert scratch_runtime.fort_scope_scratch_release_v1(first.handle, lease.token) == 2
    release(first, renewed)
    release(second, other)
    first.close()
    second.close()
    assert scratch_runtime.fort_scope_scratch_release_v1(first.handle, renewed.token) == 2
    assert scratch_runtime.fort_scope_scratch_stats_get_v1(first.handle, c.byref(ScratchStats())) == 2


@pytest.mark.native
def test_reuse_retains_capacity_and_does_not_wait_or_publish(scratch_runtime):
    scope = Scope(scratch_runtime)
    lease = acquire(scope, 64)
    data = (c.c_uint64 * 8).from_address(lease.device)
    data[:] = list(range(8))
    scope.check(scratch_runtime.fort_scope_note_launch(scope.handle))
    release(scope, lease)
    before = scope.stats()
    smaller = acquire(scope, 24)
    assert smaller.device == lease.device
    assert smaller.capacity == 64
    assert list((c.c_uint64 * 8).from_address(smaller.device)) == list(range(8))
    release(scope, smaller)
    empty = acquire(scope, 0)
    assert empty.device is None
    assert empty.capacity == 64
    release(scope, empty)
    assert scope.stats().waits == before.waits == 0
    stats = scratch_stats(scope)
    assert (stats.acquisitions, stats.allocations, stats.reuses, stats.releases, stats.grows) == (3, 1, 1, 3, 0)
    assert stats.capacity_bytes == stats.peak_scratch_bytes == stats.peak_total_device_bytes == 64
    assert not stats.active
    assert stats.active_bytes == 0
    legacy = scope.stats()
    assert legacy.allocations == legacy.allocated_bytes == legacy.peak_device_bytes == 0
    assert legacy.uploads == legacy.downloads == 0
    scope.close()


@pytest.mark.native
def test_growth_completes_pending_work_and_retires_old_capacity(scratch_runtime):
    scope = Scope(scratch_runtime)
    scope.check(scratch_runtime.fort_scope_set_device_budget(scope.handle, 96))
    initial = acquire(scope, 64)
    scope.check(scratch_runtime.fort_scope_note_launch(scope.handle))
    release(scope, initial)
    grown = acquire(scope, 96)
    assert grown.capacity == 96
    stats = scratch_stats(scope)
    assert (stats.allocations, stats.grows, stats.capacity_bytes, stats.peak_total_device_bytes) == (2, 1, 96, 96)
    assert scope.stats().waits == 1
    release(scope, grown)
    scope.close()


@pytest.mark.native
@pytest.mark.parametrize("scratch_first", [False, True])
def test_cached_scratch_and_fields_share_device_budget(scratch_runtime, scratch_first):
    scope = Scope(scratch_runtime)
    scope.check(scratch_runtime.fort_scope_set_device_budget(scope.handle, 160))
    values = (c.c_double * 16)(*range(16))
    status, handle = scope.register(values, [16])
    scope.check(status)
    if scratch_first:
        lease = acquire(scope, 64)
        release(scope, lease)
        pointer = c.c_void_p()
        assert scratch_runtime.fort_scope_device_begin(scope.handle, handle, c.byref(access(flags=1)),
                                                        c.byref(pointer)) == 4
        assert scope.stats().allocations == 0
        assert scratch_stats(scope).capacity_bytes == 64
    else:
        scope.gpu(handle, access(flags=1))
        scope.gpu_end(handle)
        rejected = Lease()
        assert scratch_runtime.fort_scope_scratch_acquire_v1(scope.handle, 64, c.byref(rejected)) == 4
        assert bytes(rejected) == bytes(Lease())
        assert scratch_stats(scope).capacity_bytes == 0
        lease = acquire(scope, 32)
        release(scope, lease)
        assert scratch_stats(scope).peak_total_device_bytes == 160
        assert scope.stats().peak_device_bytes == 128
    scope.close()
    assert list(values) == list(range(16))


@pytest.mark.native
def test_field_release_preserves_scratch_cache_and_recovers_budget(scratch_runtime):
    scope = Scope(scratch_runtime)
    scope.check(scratch_runtime.fort_scope_set_device_budget(scope.handle, 160))
    values = (c.c_double * 16)(*range(16))
    status, handle = scope.register(values, [16])
    scope.check(status)
    scope.gpu(handle, access(flags=1))
    scope.gpu_end(handle)
    first = acquire(scope, 32)
    release(scope, first)
    scope.check(scratch_runtime.fort_scope_unregister(scope.handle, handle))
    grown = acquire(scope, 160)
    release(scope, grown)
    assert scope.stats().allocated_bytes == 0
    assert scratch_stats(scope).peak_total_device_bytes == 160
    scope.close()


@pytest.mark.native
def test_cached_capacity_invalidates_preview_and_constrains_planning(scratch_runtime):
    scope = Scope(scratch_runtime)
    scope.check(scratch_runtime.fort_scope_set_device_budget(scope.handle, 224))
    values = (c.c_double * 16)(*range(16))
    status, handle = scope.register(values, [16])
    scope.check(status)
    scope.check(scratch_runtime.fort_scope_plan_reset(scope.handle))
    record(scope, 31, handle, access(flags=3))
    assert select(scope, costs(), -1).gpu_units == 1
    lease = acquire(scope, 128)
    release(scope, lease)
    # Idle capacity still leaves only 96 bytes for this 128-byte field.
    decision = select(scope, costs(), -1)
    assert decision.available
    assert decision.gpu_units == 0
    assert scope.stats().allocations == scope.stats().uploads == 0
    scope.close()


@pytest.mark.native
def test_continuation_counts_cached_scratch_and_existing_fields(scratch_runtime):
    scope = Scope(scratch_runtime)
    scope.check(scratch_runtime.fort_scope_set_device_budget(scope.handle, 256))
    first = (c.c_double * 16)(*range(16))
    second = (c.c_double * 16)(*range(16))
    status, first_handle = scope.register(first, [16])
    scope.check(status)
    device = scope.gpu(first_handle, access(flags=6))
    for i in range(16):
        device[i] = 10+i
    scope.gpu_end(first_handle)
    lease = acquire(scope, 128)
    release(scope, lease)
    status, second_handle = scope.register(second, [16], identity=2)
    scope.check(status)
    # Reached queries require completion of the previous numerical segment;
    # this wait does not publish its device-current field or retire scratch.
    scope.check(scratch_runtime.fort_scope_wait(scope.handle))
    scope.check(scratch_runtime.fort_scope_plan_reset_mode(scope.handle, 1))
    record(scope, 32, second_handle, access(flags=3))
    decision = select(scope, costs(), -1)
    assert decision.available
    assert decision.gpu_units == 0
    assert scope.stats().allocated_bytes == 128
    assert scope.stats().downloads == 0
    assert scratch_stats(scope).peak_total_device_bytes == 256
    scope.close()
    assert list(first) == list(range(10, 26))
    assert list(second) == list(range(16))


@pytest.mark.native
def test_active_close_fails_before_field_publication(scratch_runtime):
    scope = Scope(scratch_runtime)
    values = (c.c_double * 4)(*range(4))
    status, handle = scope.register(values, [4])
    scope.check(status)
    device = scope.gpu(handle, access(flags=6))
    for i in range(4):
        device[i] = 10+i
    scope.gpu_end(handle)
    lease = acquire(scope, 32)
    assert scratch_runtime.fort_scope_close(scope.handle) == 8
    assert list(values) == list(range(4))
    assert scope.stats().downloads == 0
    release(scope, lease)
    scope.close()
    assert list(values) == [10, 11, 12, 13]


@pytest.mark.native
@pytest.mark.parametrize("growing", [False, True])
def test_allocation_failure_acquires_no_lease_and_allows_native_continuation(scratch_runtime, monkeypatch, growing):
    scope = Scope(scratch_runtime)
    if growing:
        lease = acquire(scope, 32)
        release(scope, lease)
    monkeypatch.setenv("FORT_SCOPE_TEST_FAIL_SCRATCH_ALLOC", "1")
    failed = Lease(1, 0, 9, 1, 64, 64)
    assert scratch_runtime.fort_scope_scratch_acquire_v1(scope.handle, 64, c.byref(failed)) == 4
    assert bytes(failed) == bytes(Lease())
    stats = scratch_stats(scope)
    assert stats.active == stats.capacity_bytes == 0
    monkeypatch.delenv("FORT_SCOPE_TEST_FAIL_SCRATCH_ALLOC")
    recovered = acquire(scope, 64)
    release(scope, recovered)
    scope.close()


@pytest.mark.native
def test_budget_failure_keeps_previous_idle_arena(scratch_runtime):
    scope = Scope(scratch_runtime)
    scope.check(scratch_runtime.fort_scope_set_device_budget(scope.handle, 64))
    initial = acquire(scope, 64)
    release(scope, initial)
    out = Lease()
    assert scratch_runtime.fort_scope_scratch_acquire_v1(scope.handle, 65, c.byref(out)) == 4
    assert bytes(out) == bytes(Lease())
    renewed = acquire(scope, 64)
    assert renewed.device == initial.device
    release(scope, renewed)
    scope.close()


@pytest.mark.native
def test_failed_growth_wait_poison_prohibits_replay(scratch_runtime, monkeypatch):
    scope = Scope(scratch_runtime)
    lease = acquire(scope, 32)
    scope.check(scratch_runtime.fort_scope_note_launch(scope.handle))
    release(scope, lease)
    monkeypatch.setenv("FORT_SCOPE_TEST_FAIL_WAIT", "1")
    out = Lease()
    assert scratch_runtime.fort_scope_scratch_acquire_v1(scope.handle, 64, c.byref(out)) == 6
    assert bytes(out) == bytes(Lease())
    assert scratch_runtime.fort_scope_close(scope.handle) == 6
    assert scratch_runtime.fort_scope_scratch_acquire_v1(scope.handle, 0, c.byref(out)) == 6
    scope.check(scratch_runtime.fort_scope_abandon(scope.handle))
    assert scratch_runtime.fort_scope_scratch_release_v1(scope.handle, lease.token) == 2


@pytest.mark.native
@pytest.mark.parametrize("active", [False, True])
def test_abandon_retires_scratch_without_publishing_failed_fields(scratch_runtime, active):
    scope = Scope(scratch_runtime)
    values = (c.c_double * 4)(*range(4))
    status, handle = scope.register(values, [4])
    scope.check(status)
    device = scope.gpu(handle, access(flags=6))
    for i in range(4):
        device[i] = 10+i
    scope.gpu_end(handle)
    lease = acquire(scope, 32)
    if not active:
        release(scope, lease)
    assert scratch_runtime.fort_scope_abandon(scope.handle) == 8
    assert scratch_runtime.fort_scope_execution_error(scope.handle, b"injected numerical failure") == 6
    assert scratch_runtime.fort_scope_scratch_release_v1(scope.handle, lease.token) == 6
    scope.check(scratch_runtime.fort_scope_abandon(scope.handle))
    assert list(values) == list(range(4))


@pytest.mark.native
def test_fortran_scratch_public_interface(scratch_runtime, tmp_path):
    compiler = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not compiler:
        pytest.skip("gfortran unavailable")
    source = tmp_path / "caller.f90"
    source.write_text('''program caller
  use iso_c_binding
  use fort_scoped_memory
  implicit none
  integer(c_int64_t) :: context, token
  type(fort_scope_scratch_lease_v1) :: lease
  type(fort_scope_scratch_stats_v1) :: stats
  type(fort_scope_stats) :: legacy
  integer(c_int64_t), pointer :: values(:)
  call check(fort_scope_create(0_c_int,context))
  call check(fort_scope_set_device_budget(context,64_c_size_t))
  call check(fort_scope_scratch_acquire_v1(context,64_c_size_t,lease))
  if(lease%version/=FORT_SCOPE_SCRATCH_ABI_VERSION .or. lease%reserved/=0) error stop 'version'
  if(lease%bytes/=64 .or. lease%capacity/=64 .or. .not.c_associated(lease%device)) error stop 'lease'
  token=lease%token
  call c_f_pointer(lease%device,values,[8])
  values=17
  call check(fort_scope_scratch_release_v1(context,token))
  call check(fort_scope_scratch_acquire_v1(context,8_c_size_t,lease))
  if(lease%token==token .or. lease%capacity/=64) error stop 'reuse'
  call c_f_pointer(lease%device,values,[8])
  if(any(values/=17)) error stop 'storage'
  call check(fort_scope_scratch_stats_get_v1(context,stats))
  if(stats%version/=1 .or. stats%active/=1 .or. stats%active_bytes/=8) error stop 'stats ABI'
  if(stats%allocations/=1 .or. stats%reuses/=1 .or. stats%peak_total_device_bytes/=64) error stop 'counts'
  call check(fort_scope_stats_get(context,legacy))
  if(legacy%allocations/=0 .or. legacy%allocated_bytes/=0) error stop 'legacy stats'
  call check(fort_scope_scratch_release_v1(context,lease%token))
  call check(fort_scope_close(context))
contains
  subroutine check(status)
    integer(c_int),intent(in)::status
    if(status/=FORT_SCOPE_OK) error stop 'runtime status'
  end subroutine
end program
''')
    result = subprocess.run([
        compiler, "-std=f2018", "-fcheck=all", str(RUNTIME / "scoped_memory.f90"), str(source),
        scratch_runtime._name, "-Wl,-rpath," + str(Path(scratch_runtime._name).parent), "-o", "caller",
    ], cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(tmp_path / "caller")], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


CUDA_CALLER = r'''#include "scoped_runtime.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
void check(int status) {
    if(status) { std::fprintf(stderr,"scope status %d: %s\n",status,fort_scope_error()); std::exit(2); }
}
void expect(bool value,const char*message) {
    if(!value) { std::fprintf(stderr,"%s\n",message); std::exit(3); }
}
__global__ void snapshot(double*temporary,size_t n,int origin) {
    const size_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) temporary[i]=double(int(i)+origin);
}
__global__ void consume(const double*temporary,double*field,size_t n) {
    const size_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) field[i+2]=2*temporary[i]+3;
}
int main(int argc,char**argv) {
    expect(argc==2,"missing mode");
    const char*mode=argv[1];
    const char*ordinal=std::getenv("FORT_TEST_CUDA_DEVICE");
    const int device=ordinal ? std::atoi(ordinal) : 0;
    fort_scope_t context=0;
    check(fort_scope_create(device,&context));
    fort_scope_scratch_lease_v1 lease{};
    fort_scope_stats legacy{};
    fort_scope_scratch_stats_v1 stats{};
    if(!std::strcmp(mode,"empty") || !std::strcmp(mode,"zero_budget")) {
        check(fort_scope_set_device_budget(context,0));
        if(!std::strcmp(mode,"zero_budget"))
            expect(fort_scope_scratch_acquire_v1(context,1,&lease)==FORT_SCOPE_RESOURCE,"zero budget");
        check(fort_scope_scratch_acquire_v1(context,0,&lease));
        expect(lease.token && !lease.device && !lease.bytes && !lease.capacity,"empty lease");
        check(fort_scope_set_device_budget(context,0));
        check(fort_scope_scratch_release_v1(context,lease.token));
        check(fort_scope_stats_get(context,&legacy));
        check(fort_scope_scratch_stats_get_v1(context,&stats));
        expect(!legacy.allocations && !legacy.launches && !legacy.waits && !stats.allocations,"empty activity");
        check(fort_scope_close(context));
        std::puts("SCRATCH_EMPTY_OK"); return 0;
    }
    if(!std::strcmp(mode,"fail_before")) {
        expect(fort_scope_scratch_acquire_v1(context,64,&lease)==FORT_SCOPE_RESOURCE,"resource failure");
        expect(!lease.token && !lease.device,"failed lease");
        check(fort_scope_set_device_budget(context,0));
        check(fort_scope_scratch_acquire_v1(context,0,&lease));
        check(fort_scope_scratch_release_v1(context,lease.token));
        check(fort_scope_close(context));
        std::puts("SCRATCH_RESOURCE_OK"); return 0;
    }
    int count=0;
    if(cudaGetDeviceCount(&count)!=cudaSuccess || device<0 || device>=count) return 77;
    check(fort_scope_scratch_acquire_v1(context,64,&lease));
    if(!std::strcmp(mode,"poison_active") || !std::strcmp(mode,"poison_idle")) {
        if(!std::strcmp(mode,"poison_idle")) check(fort_scope_scratch_release_v1(context,lease.token));
        expect(fort_scope_execution_error(context,"injected numerical failure")==FORT_SCOPE_EXECUTION,"poison");
        expect(fort_scope_close(context)==FORT_SCOPE_EXECUTION,"unsafe close");
        check(fort_scope_abandon(context));
        expect(fort_scope_scratch_release_v1(context,lease.token)==FORT_SCOPE_STALE,"retired lease");
        std::puts("SCRATCH_ABANDON_OK"); return 0;
    }
    check(fort_scope_scratch_release_v1(context,lease.token));
    check(fort_scope_close(context));
    // New owner with a finite inclusive budget; preceding branch only checks
    // that actual device allocation/retirement works before numerical use.
    check(fort_scope_create(device,&context));
    constexpr size_t n=1003, scratch_bytes=n*sizeof(double), field_bytes=(n+4)*sizeof(double);
    check(fort_scope_set_device_budget(context,field_bytes+scratch_bytes+16));
    std::vector<double> host(n+4,-91);
    size_t extent=n+4,lo=2,hi=n+2;
    int64_t lower=-13;
    fort_scope_layout layout{1,FORT_SCOPE_REAL64,sizeof(double),host.data(),&extent,&lower,1};
    fort_buffer_t buffer=0;
    check(fort_scope_register(context,1,1,&layout,1,&buffer));
    fort_scope_section interior{&lo,&hi};
    fort_scope_access writes{};
    writes.write_count=writes.overwrite_count=1;
    writes.writes=writes.overwrites=&interior;
    check(fort_scope_scratch_acquire_v1(context,scratch_bytes,&lease));
    int previous=0; void*stream=nullptr;
    check(fort_scope_gpu_enter(context,&previous,&stream));
    snapshot<<<(n+127)/128,128,0,static_cast<cudaStream_t>(stream)>>>(static_cast<double*>(lease.device),n,123);
    expect(cudaGetLastError()==cudaSuccess,"snapshot launch");
    check(fort_scope_note_launch(context));
    check(fort_scope_gpu_leave(context,previous));
    void*original=lease.device; const auto old_token=lease.token;
    check(fort_scope_scratch_release_v1(context,lease.token));
    check(fort_scope_scratch_acquire_v1(context,scratch_bytes,&lease));
    expect(lease.device==original && lease.token!=old_token,"arena reuse");
    void*field=nullptr;
    check(fort_scope_device_begin(context,buffer,&writes,&field));
    check(fort_scope_gpu_enter(context,&previous,&stream));
    // A new lease does not retain numerical definitions. Overwrite scratch
    // before consumption; this producer and consumer share one active lease.
    snapshot<<<(n+127)/128,128,0,static_cast<cudaStream_t>(stream)>>>(static_cast<double*>(lease.device),n,-11);
    expect(cudaGetLastError()==cudaSuccess,"reused snapshot launch");
    check(fort_scope_note_launch(context));
    consume<<<(n+127)/128,128,0,static_cast<cudaStream_t>(stream)>>>(static_cast<double*>(lease.device),
                                                                  static_cast<double*>(field),n);
    expect(cudaGetLastError()==cudaSuccess,"consume launch");
    check(fort_scope_note_launch(context));
    check(fort_scope_gpu_leave(context,previous));
    check(fort_scope_device_end(context,buffer));
    check(fort_scope_scratch_release_v1(context,lease.token));
    check(fort_scope_stats_get(context,&legacy));
    expect(!legacy.upload_bytes && !legacy.download_bytes && !legacy.waits,"premature synchronization/publication");
    check(fort_scope_scratch_acquire_v1(context,scratch_bytes+16,&lease));
    check(fort_scope_scratch_stats_get_v1(context,&stats));
    expect(stats.allocations==2 && stats.reuses==1 && stats.grows==1,"growth statistics");
    expect(stats.peak_total_device_bytes==field_bytes+scratch_bytes+16,"inclusive peak");
    expect(fort_scope_close(context)==FORT_SCOPE_STATE,"active close");
    for(double value:host) expect(value==-91,"active close published");
    check(fort_scope_scratch_release_v1(context,lease.token));
    fort_scope_access reads{}; reads.flags=FORT_SCOPE_READ_ALL;
    check(fort_scope_host_begin(context,buffer,&reads));
    check(fort_scope_host_end(context,buffer));
    for(size_t i=0;i<n;++i) expect(host[i+2]==2*double(int(i)-11)+3,"logical coordinate output");
    expect(host[0]==-91 && host[1]==-91 && host[n+2]==-91 && host[n+3]==-91,"halo changed");
    check(fort_scope_stats_get(context,&legacy));
    expect(legacy.allocations==1 && legacy.peak_device_bytes==field_bytes,"legacy field accounting");
    expect(!legacy.upload_bytes && legacy.download_bytes==scratch_bytes && legacy.launches==3,"transfer accounting");
    check(fort_scope_close(context));
    std::puts("SCRATCH_STREAM_OK");
}
'''


@pytest.fixture(scope="module")
def scratch_cuda(tmp_path_factory):
    nvcc = shutil.which("nvcc")
    host = shutil.which("g++-14") or shutil.which("g++")
    if not nvcc or not host:
        pytest.skip("CUDA/C++ toolchains unavailable")
    directory = tmp_path_factory.mktemp("scoped_scratch_cuda")
    source, binary = directory / "caller.cu", directory / "caller"
    source.write_text(CUDA_CALLER)
    # Native code for visible hardware; no architecture or host-target library
    # directory is embedded in this acceptance fixture.
    result = subprocess.run([
        nvcc, "-std=c++17", "-O2", "-arch=native", "-ccbin", host,
        "-DFORT_SCOPE_TEST_FAULTS", "-I", str(RUNTIME),
        str(RUNTIME / "scoped_runtime.cu"), str(source), "-o", str(binary),
    ], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    return binary


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["empty", "zero_budget", "fail_before", "stream", "poison_active", "poison_idle"])
def test_cuda_context_scratch(scratch_cuda, mode):
    environment = os.environ.copy()
    environment.pop("FORT_SCOPE_TEST_FAIL_SCRATCH_ALLOC", None)
    if mode in {"empty", "zero_budget", "fail_before"}:
        environment["CUDA_VISIBLE_DEVICES"] = ""
    if mode == "fail_before":
        environment["FORT_SCOPE_TEST_FAIL_SCRATCH_ALLOC"] = "1"
    result = subprocess.run([str(scratch_cuda), mode], env=environment, capture_output=True, text=True, timeout=30)
    if result.returncode == 77:
        pytest.skip("selected CUDA device unavailable")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SCRATCH_" in result.stdout
    assert "_OK" in result.stdout
