"""Public planning queries agree with the real coherence reference backend."""
from __future__ import annotations

import ctypes as c
import json

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
        "plan_validate": [TOKEN],
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


def add_operation(scope, kind, unit, bindings, *, flops=float("nan"), memory_bytes=float("inf")):
    packed = (Binding * len(bindings))(*(Binding(handle, effect) for handle, effect in bindings))
    scope.check(scope.lib.fort_scope_plan_add(scope.handle, kind, unit, packed, len(packed),
                                            flops, memory_bytes, kind == 1))


def test_definition_validation_needs_no_costs_and_never_commits_simulated_writes(planning):
    scope = Scope(planning)
    first = (c.c_double * 8)(*range(8))
    output = (c.c_double * 8)(*[-99] * 8)
    status, a = scope.register(first, [8], lower=[-2])
    scope.check(status)
    status, b = scope.register(output, [8], identity=2, initialized=False, lower=[-5])
    scope.check(status)
    interior = [([2], [6])]
    scope.check(planning.fort_scope_plan_reset(scope.handle))
    add_operation(scope, 1, 71, [(a, access(read=interior)),
                                (b, access(write=interior, overwrite=interior))])
    add_operation(scope, 0, 0, [(b, access(write=interior, overwrite=interior))], flops=-1)
    add_operation(scope, 1, 72, [(b, access(read=interior, write=interior))], flops=0, memory_bytes=0)
    before = bytes(scope.stats())
    for _ in range(2):
        scope.check(planning.fort_scope_plan_validate(scope.handle))
        assert bytes(scope.stats()) == before
        assert list(first) == list(range(8))
        assert list(output) == [-99] * 8
    # A simulated producer did not define the live output, even after success.
    assert planning.fort_scope_host_begin(scope.handle, b, c.byref(access(read=interior))) == 7
    assert bytes(scope.stats()) == before
    # Missing calibration affects selection independently of definition safety.
    assert select(scope, Costs()).available == 0
    assert bytes(scope.stats()) == before
    scope.close()


@pytest.mark.parametrize(("defined", "reads", "overwrite", "expected"), [
    ([], False, False, 7),              # Unknown/conditional write holes need preservation.
    ([], True, True, 7),               # RMW cannot consume its own later overwrite.
    ([], False, True, 0),              # Guaranteed overwrite needs no previous values.
    ([([0], [2]), ([6], [8])], False, False, 0),
    ([([0], [2])], False, False, 7),    # Opposite undefined face remains a hole.
])
def test_definition_validation_preserves_conditional_holes_and_rmw(planning, defined, reads, overwrite, expected):
    scope = Scope(planning)
    values = (c.c_double * 8)(*range(8))
    status, handle = scope.register(values, [8], defined=defined)
    scope.check(status)
    full, interior = [([0], [8])], [([2], [6])]
    effect = access(read=full if reads else (), write=full,
                    overwrite=full if overwrite else interior)
    scope.check(planning.fort_scope_plan_reset(scope.handle))
    add_operation(scope, 1, 73, [(handle, effect)])
    before = bytes(scope.stats())
    assert planning.fort_scope_plan_validate(scope.handle) == expected
    if expected:
        message = planning.fort_scope_error().decode()
        assert "uninitialized_read" in message
        assert "operation=0 unit=73 resource=1 generation=1" in message
    assert bytes(scope.stats()) == before
    assert list(values) == list(range(8))
    scope.close()


def test_definition_validation_forget_is_ordered_and_isolated(planning):
    scope = Scope(planning)
    values = (c.c_double * 8)(*range(8))
    other = (c.c_double * 8)(*range(20, 28))
    status, a = scope.register(values, [8])
    scope.check(status)
    status, b = scope.register(other, [8], identity=2)
    scope.check(status)
    interior = [([2], [6])]
    overwrite = access(write=interior, overwrite=interior)
    before = bytes(scope.stats())
    for nested_forget, expected in [(False, 0), (True, 7)]:
        scope.check(planning.fort_scope_plan_reset(scope.handle))
        add_operation(scope, 2, 0, [(a, access())])
        add_operation(scope, 1, 74, [(a, overwrite)])
        if nested_forget:
            add_operation(scope, 2, 0, [(a, access())])
        add_operation(scope, 0, 0, [(a, access(read=interior)), (b, access(flags=1))])
        assert planning.fort_scope_plan_validate(scope.handle) == expected
        assert bytes(scope.stats()) == before
        # Even a failed nested simulation did not forget the live input.
        scope.cpu_begin(a, access(flags=1))
        scope.cpu_end(a)
        assert bytes(scope.stats()) == before
    scope.close()


def test_definition_validation_checks_all_bindings_before_any_commit(planning):
    scope = Scope(planning)
    values = [(c.c_double * 8)(*[-99] * 8) for _ in range(2)]
    handles = []
    for identity, payload in enumerate(values, 1):
        status, handle = scope.register(payload, [8], identity=identity, initialized=False)
        scope.check(status)
        handles.append(handle)
    scope.check(planning.fort_scope_plan_reset(scope.handle))
    add_operation(scope, 1, 75, [(handles[0], access(flags=6)), (handles[1], access(flags=1))])
    before = bytes(scope.stats())
    assert planning.fort_scope_plan_validate(scope.handle) == 7
    assert "operation=0 unit=75 resource=2" in planning.fort_scope_error().decode()
    for handle in handles:
        assert planning.fort_scope_host_begin(scope.handle, handle, c.byref(access(flags=1))) == 7
    assert bytes(scope.stats()) == before
    assert all(list(payload) == [-99] * 8 for payload in values)
    scope.close()


def test_definition_validation_does_not_refresh_an_existing_device_copy(planning):
    scope = Scope(planning)
    values = (c.c_double * 8)(*range(8))
    status, handle = scope.register(values, [8])
    scope.check(status)
    device = scope.gpu(handle, access(flags=6))
    for i in range(8):
        device[i] = i + 100
    scope.gpu_end(handle)
    scope.check(planning.fort_scope_wait(scope.handle))
    scope.check(planning.fort_scope_plan_reset(scope.handle))
    add_operation(scope, 0, 0, [(handle, access(flags=1))])
    before = bytes(scope.stats())
    assert planning.fort_scope_plan_host_current(scope.handle, handle) == 5
    scope.check(planning.fort_scope_plan_validate(scope.handle))
    assert planning.fort_scope_plan_host_current(scope.handle, handle) == 5
    assert bytes(scope.stats()) == before
    assert list(values) == list(range(8))
    scope.close()
    assert list(values) == [i + 100 for i in range(8)]


def test_definition_validation_requeries_effects_generations_and_shapes(planning):
    scope = Scope(planning)
    values = (c.c_double * 8)(*range(8))
    status, handle = scope.register(values, [8], generation=1)
    scope.check(status)
    assert planning.fort_scope_plan_validate(scope.handle) == 8
    scope.check(planning.fort_scope_plan_reset(scope.handle))
    record(scope, 76, handle, access(flags=3))
    scope.check(planning.fort_scope_plan_validate(scope.handle))
    assert select(scope, costs(), -1).gpu_units == 1
    # Adding FORGET invalidates an old preview; validation and selection see it.
    add_operation(scope, 2, 0, [(handle, access())])
    record(scope, 77, handle, access(flags=1))
    assert planning.fort_scope_plan_validate(scope.handle) == 7
    assert select(scope, costs(), -1).available == 0
    scope.check(planning.fort_scope_unregister(scope.handle, handle))
    replacement = (c.c_double * 5)(*range(5))
    status, new_handle = scope.register(replacement, [5], generation=2, lower=[-3])
    scope.check(status)
    # Old recorded handles cannot silently acquire the new generation/layout.
    assert planning.fort_scope_plan_validate(scope.handle) == 5
    assert "unknown_resource_handle" in planning.fort_scope_error().decode()
    scope.check(planning.fort_scope_plan_reset(scope.handle))
    bindings = record(scope, 78, new_handle, access(read=[([1], [4])], write=[([1], [4])]))
    scope.check(planning.fort_scope_plan_validate(scope.handle))
    decision = select(scope, costs())
    assert decision.available
    assert decision.gpu_units == 1
    assert decision.upload_bytes == decision.download_bytes == 3 * 8
    assert planning.fort_scope_plan_validate(scope.handle) == 8
    chosen = c.c_int()
    scope.check(planning.fort_scope_plan_next(scope.handle, 78, bindings, 1, c.byref(chosen)))
    assert chosen.value == 1
    assert bytes(scope.stats()) == bytes(type(scope.stats())())
    scope.close()


def test_definition_validation_empty_capture_and_worker_budget(planning):
    scope = Scope(planning)
    status, handle = scope.register(None, [0, c.c_size_t(-1).value], initialized=False)
    scope.check(status)
    scope.check(planning.fort_scope_plan_reset(scope.handle))
    add_operation(scope, 1, 79, [(handle, access(flags=3))])
    scope.check(planning.fort_scope_plan_validate(scope.handle))
    assert bytes(scope.stats()) == bytes(type(scope.stats())())
    for unit in range(80, 144):
        add_operation(scope, 1, unit, [(handle, access())])
    assert planning.fort_scope_plan_validate(scope.handle) == 5
    assert "planning_worker_budget_exceeded" in planning.fort_scope_error().decode()
    assert bytes(scope.stats()) == bytes(type(scope.stats())())
    scope.close()


@pytest.mark.parametrize("enabled", [None, "0", "1"])
def test_definition_validation_optional_evidence_survives_metadata_cleanup(planning, monkeypatch, capfd, enabled):
    if enabled is None:
        monkeypatch.delenv("FORT_RUNTIME_TRACE", raising=False)
    else:
        monkeypatch.setenv("FORT_RUNTIME_TRACE", enabled)
    scope = Scope(planning)
    values = (c.c_double * 8)(*[-99] * 8)
    status, handle = scope.register(values, [8], identity=123, generation=17, initialized=False)
    scope.check(status)
    # Even query phase failures are observable without inventing an operation.
    assert planning.fort_scope_plan_validate(scope.handle) == 8
    scope.check(planning.fort_scope_plan_reset(scope.handle))
    add_operation(scope, 1, 80, [(handle, access(flags=1))])
    assert planning.fort_scope_plan_validate(scope.handle) == 7
    scope.check(planning.fort_scope_plan_reset(scope.handle))
    add_operation(scope, 1, 81, [(handle, access(flags=6))])
    scope.check(planning.fort_scope_plan_validate(scope.handle))
    assert bytes(scope.stats()) == bytes(type(scope.stats())())
    scope.close()
    assert list(values) == [-99] * 8
    rows = [json.loads(line.removeprefix("FORT_SCOPED evidence "))
            for line in capfd.readouterr().err.splitlines() if line.startswith("FORT_SCOPED evidence ")]
    if enabled != "1":
        assert rows == []
        return
    assert [row["status"] for row in rows] == [8, 7, 0]
    assert all(row["event"] == "definition_validation" for row in rows)
    assert all(row["schema_version"] == 1 for row in rows)
    assert all(row["context"] == scope.handle.value for row in rows)
    assert all(row["mutates_live_state"] is False for row in rows)
    assert all(row["evidence"] == "ordered_definition_preflight" for row in rows)
    failure = rows[1]
    assert failure["reason"] == "uninitialized_read"
    assert failure["operation"] == 0
    assert failure["unit"] == 80
    assert failure["resource"] == handle.value
    assert failure["identity"] == 123
    assert failure["generation"] == 17
    assert rows[2]["reason"] == "definition_plan_valid"
