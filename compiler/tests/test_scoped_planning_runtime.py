"""Public planning queries agree with the real coherence reference backend."""
from __future__ import annotations

import ctypes as c

import pytest

from compiler.tests.test_scoped_runtime import TOKEN, Access, Scope, access, runtime  # noqa: F401


class Binding(c.Structure):
    _fields_ = [("buffer", TOKEN), ("access", Access)]


class Costs(c.Structure):
    _fields_ = [("version", c.c_uint32), ("valid", c.c_uint32), ("max_allocation_bytes", c.c_size_t)] + [
        (name, c.c_double) for name in (
            "cpu_flops", "cpu_bandwidth", "gpu_flops", "gpu_bandwidth",
            "h2d_latency", "h2d_bandwidth", "d2h_latency", "d2h_bandwidth",
            "create_seconds", "register_seconds", "host_access_seconds", "device_access_seconds",
            "gpu_setup_seconds", "cold_driver_startup_seconds", "allocation_seconds", "release_seconds",
            "wait_seconds", "launch_enqueue_seconds", "planning_operation_seconds")]


class Decision(c.Structure):
    _fields_ = [(name, c.c_uint32) for name in ("available", "gpu_units", "cpu_units", "candidates")] + [
        (name, c.c_uint64) for name in (
            "simulated_operations", "upload_bytes", "download_bytes", "uploads", "downloads",
            "launches", "waits", "allocations", "peak_device_bytes")] + [
        (name, c.c_double) for name in ("estimated_seconds", "native_seconds")]


def costs():
    result = Costs(1, 1, 1 << 26)
    for name, _ in Costs._fields_[3:]:
        setattr(result, name, 1e-9)
    result.cpu_flops = result.cpu_bandwidth = 1e8
    result.gpu_flops = result.gpu_bandwidth = result.h2d_bandwidth = result.d2h_bandwidth = 1e12
    return result


@pytest.fixture
def planning(runtime):  # noqa: F811 - pytest discovers the imported fixture.
    signatures = {
        "plan_reset": [TOKEN],
        "plan_host_current": [TOKEN, TOKEN],
        "plan_add": [TOKEN, c.c_uint32, TOKEN, c.POINTER(Binding), c.c_size_t, c.c_double, c.c_double, c.c_int],
        "plan_select": [TOKEN, c.POINTER(Costs), c.c_int, c.POINTER(Decision)],
        "plan_next": [TOKEN, TOKEN, c.POINTER(Binding), c.c_size_t, c.POINTER(c.c_int)],
        "gpu_enter": [TOKEN, c.POINTER(c.c_int), c.POINTER(c.c_void_p)],
        "note_launch": [TOKEN],
    }
    for name, signature in signatures.items():
        getattr(runtime, "fort_scope_" + name).argtypes = signature
    return runtime


def record(scope, unit, handle, effects, *, flops=1e8):
    bindings = (Binding * 1)(Binding(handle, effects))
    scope.check(scope.lib.fort_scope_plan_add(scope.handle, 1, unit, bindings, 1, flops, 256, 1))
    return bindings


def select(scope, calibration, compatible=1):
    decision = Decision()
    scope.check(scope.lib.fort_scope_plan_select(scope.handle, c.byref(calibration), compatible, c.byref(decision)))
    return decision


def test_query_records_without_execution_and_cached_preview_checks_costs(planning):
    scope = Scope(planning)
    values = (c.c_double * 32)(*range(32))
    status, handle = scope.register(values, [32])
    scope.check(status)
    scope.check(planning.fort_scope_plan_reset(scope.handle))
    record(scope, 41, handle, access(flags=3))
    calibration = costs()
    preview = select(scope, calibration, -1)
    assert preview.available
    assert preview.gpu_units == 1
    assert scope.stats().allocations == scope.stats().uploads == scope.stats().launches == 0
    # Cost changes cannot reuse a previously favourable preview.
    calibration.gpu_flops = 1
    decision = select(scope, calibration)
    assert decision.gpu_units == 0
    scope.close()
    assert list(values) == list(range(32))


def test_native_thin_worker_and_missing_profile_have_zero_transfers(planning):
    for calibration, flops in [(costs(), 1), (Costs(), 1e8)]:
        calibration.h2d_latency = calibration.d2h_latency = 1e-3
        scope = Scope(planning)
        values = (c.c_double * 32)(*range(32))
        status, handle = scope.register(values, [32])
        scope.check(status)
        scope.check(planning.fort_scope_plan_reset(scope.handle))
        bindings = record(scope, 42, handle, access(flags=3), flops=flops)
        decision = select(scope, calibration)
        assert decision.gpu_units == 0
        chosen = c.c_int(99)
        scope.check(planning.fort_scope_plan_next(scope.handle, 42, bindings, 1, c.byref(chosen)))
        assert chosen.value == 0
        assert scope.stats().upload_bytes == scope.stats().download_bytes == scope.stats().allocations == 0
        scope.close()


def test_selected_partial_effects_predict_actual_transfers_and_preserve_host_writes(planning):
    scope = Scope(planning)
    values = (c.c_double * 32)(*range(32))
    status, handle = scope.register(values, [32])
    scope.check(status)
    gpu_effect = access(read=[([2], [30])], write=[([2], [30])])
    native_effect = access(write=[([0], [2]), ([30], [32])], overwrite=[([0], [2]), ([30], [32])])
    scope.check(planning.fort_scope_plan_reset(scope.handle))
    bindings = record(scope, 43, handle, gpu_effect)
    fixed = (Binding * 1)(Binding(handle, native_effect))
    scope.check(planning.fort_scope_plan_add(scope.handle, 0, 0, fixed, 1, 0, 0, 0))
    calibration = costs()
    preview = select(scope, calibration, -1)
    decision = select(scope, calibration)
    assert decision.gpu_units == 1
    assert decision.simulated_operations == preview.simulated_operations
    chosen = c.c_int()
    scope.check(planning.fort_scope_plan_next(scope.handle, 43, bindings, 1, c.byref(chosen)))
    assert chosen.value == 1
    previous, stream = c.c_int(), c.c_void_p()
    scope.check(planning.fort_scope_gpu_enter(scope.handle, c.byref(previous), c.byref(stream)))
    pointer = c.c_void_p()
    scope.check(planning.fort_scope_device_begin(scope.handle, handle, c.byref(gpu_effect), c.byref(pointer)))
    device = (c.c_double * 32).from_address(pointer.value)
    for i in range(2, 30):
        device[i] *= 2
    scope.check(planning.fort_scope_note_launch(scope.handle))
    scope.check(planning.fort_scope_device_end(scope.handle, handle))
    scope.cpu_begin(handle, native_effect)
    values[0] = values[1] = values[30] = values[31] = -7
    scope.cpu_end(handle)
    # Explicit publication is the same final host-visible boundary as close.
    scope.cpu_begin(handle, access(flags=1))
    scope.cpu_end(handle)
    stats = scope.stats()
    for field in ("upload_bytes", "download_bytes", "uploads", "downloads", "launches", "waits", "allocations", "peak_device_bytes"):
        assert getattr(stats, field) == getattr(decision, field), field
    scope.close()
    assert list(values) == [-7, -7, *[2*i for i in range(2, 30)], -7, -7]


def test_installed_worker_effects_and_sequence_cannot_change(planning):
    scope = Scope(planning)
    values = (c.c_double * 32)(*range(32))
    status, handle = scope.register(values, [32])
    scope.check(status)
    scope.check(planning.fort_scope_plan_reset(scope.handle))
    bindings = record(scope, 44, handle, access(flags=3))
    assert select(scope, costs()).gpu_units == 1
    chosen = c.c_int()
    assert planning.fort_scope_plan_next(scope.handle, 45, bindings, 1, c.byref(chosen)) == 8
    changed = (Binding * 1)(Binding(handle, access(flags=1)))
    assert planning.fort_scope_plan_next(scope.handle, 44, changed, 1, c.byref(chosen)) == 8
    assert planning.fort_scope_close(scope.handle) == 8
    scope.check(planning.fort_scope_plan_next(scope.handle, 44, bindings, 1, c.byref(chosen)))
    scope.close()
    assert list(values) == list(range(32))


def test_host_metadata_check_does_not_refresh_stale_or_undefined_values(planning):
    scope = Scope(planning)
    values = (c.c_double * 32)(*range(32))
    status, handle = scope.register(values, [32])
    scope.check(status)
    scope.check(planning.fort_scope_plan_host_current(scope.handle, handle))
    scope.gpu(handle, access(flags=6))
    scope.gpu_end(handle)
    before = scope.stats()
    assert planning.fort_scope_plan_host_current(scope.handle, handle) == 5
    assert scope.stats().downloads == before.downloads
    scope.close()
