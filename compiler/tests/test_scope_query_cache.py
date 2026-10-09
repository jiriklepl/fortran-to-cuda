"""Context-local planning reuse tracks query, definition and coherence changes."""
# ruff: noqa: F811 - pytest discovers imported fixtures under these names.
from __future__ import annotations

import ctypes as c
import json

import pytest

from compiler.tests.test_scoped_planning_runtime import (
    costs,
    planning,  # noqa: F401
    record,
    select,
)
from compiler.tests.test_scoped_runtime import (
    TOKEN,
    Layout,
    Scope,
    access,
    runtime,  # noqa: F401
)


def diagnostics(capfd):
    lines = capfd.readouterr().err.splitlines()
    evidence = [json.loads(line.removeprefix("FORT_SCOPED evidence ")) for line in lines
                if line.startswith("FORT_SCOPED evidence ")]
    timings = [dict(field.split("=", 1) for field in line.split()[2:]) for line in lines
               if line.startswith("FORT_SCOPED planning_timing ")]
    return evidence, timings


def query(planning, *, initialized=True, device=0):
    scope = Scope(planning)
    if device:
        scope.close()
        scope.check(planning.fort_scope_create(device, c.byref(scope.handle)))
    values = (c.c_double * 8)(*range(8))
    status, handle = scope.register(values, [8], lower=[-9], initialized=initialized)
    scope.check(status)
    scope.check(planning.fort_scope_plan_reset(scope.handle))
    record(scope, 101, handle, access(flags=1))
    return scope, values, handle


def validate(scope):
    scope.check(scope.lib.fort_scope_plan_validate(scope.handle))


def test_successful_proof_and_preview_survive_read_only_probes(planning, monkeypatch, capfd):
    monkeypatch.setenv("FORT_RUNTIME_TRACE", "1")
    scope, values, handle = query(planning)
    validate(scope)
    first_stats = bytes(scope.stats())
    layout, device = Layout(), c.c_int()
    scope.check(planning.fort_scope_layout_get(scope.handle, handle, c.byref(layout)))
    scope.check(planning.fort_scope_device_get(scope.handle, c.byref(device)))
    scope.check(planning.fort_scope_plan_host_current(scope.handle, handle))
    validate(scope)
    preview = select(scope, costs(), -1)
    assert bytes(scope.stats()) == first_stats
    validate(scope)
    decision = select(scope, costs(), -1)
    assert bytes(decision) == bytes(preview)
    select(scope, costs())
    rows, timings = diagnostics(capfd)
    proofs = [row for row in rows if row["event"] == "definition_validation"]
    assert [row["cache_hit"] for row in proofs] == [False, True, True]
    assert len({(row["query_generation"], row["state_generation"]) for row in proofs}) == 1
    assert all(row["seconds"] >= 0 and not row["mutates_live_state"] for row in proofs)
    assert [row["cache_hit"] for row in timings if row["phase"] == "selection"] == ["0", "1", "1"]
    # One reset, one add and one immutable snapshot construction. Repeated
    # validation and selection do not reconstruct copied resource metadata.
    assert sum(row["phase"] == "query_construction" for row in timings) == 3
    assert {row["phase"] for row in timings} == {"query_construction", "validation", "selection"}
    assert all(float(row["seconds"]) >= 0 for row in timings)
    assert list(values) == list(range(8))
    # Metadata-only selection did not execute an installed GPU schedule.
    if decision.gpu_units:
        scope.check(planning.fort_scope_plan_next(scope.handle, 101,
                    record_bindings(handle, access(flags=1)), 1, c.byref(c.c_int())))
    scope.close()


def record_bindings(handle, effects):
    from compiler.tests.test_scoped_planning_runtime import Binding
    return (Binding * 1)(Binding(handle, effects))


@pytest.mark.parametrize("mutation", [
    "reset", "add", "duplicate_register", "new_register", "failed_register",
    "host_write", "device_write", "cancel_access", "budget", "wait", "gpu_enter_leave",
])
def test_successful_proof_rechecks_after_query_or_runtime_mutation(planning, monkeypatch, capfd, mutation):
    monkeypatch.setenv("FORT_RUNTIME_TRACE", "1")
    planning.fort_scope_gpu_leave.argtypes = [TOKEN, c.c_int]
    scope, values, handle = query(planning)
    validate(scope)
    validate(scope)
    if mutation == "reset":
        scope.check(planning.fort_scope_plan_reset(scope.handle))
        record(scope, 101, handle, access(flags=1))
    elif mutation == "add":
        record(scope, 102, handle, access(flags=1))
    elif mutation in {"duplicate_register", "failed_register"}:
        status, returned = scope.register(values, [8], generation=2 if mutation == "failed_register" else 1,
                                          lower=[-9])
        assert status == (2 if mutation == "failed_register" else 0)
        if not status:
            assert returned.value == handle.value
    elif mutation == "new_register":
        extra = (c.c_double * 3)(7, 8, 9)
        status, added = scope.register(extra, [3], identity=2)
        scope.check(status)
        scope.check(planning.fort_scope_unregister(scope.handle, added))
    elif mutation == "host_write":
        scope.cpu_begin(handle, access(flags=6))
        values[3] = 99
        scope.cpu_end(handle)
    elif mutation == "device_write":
        pointer = scope.gpu(handle, access(flags=3))
        pointer[3] = 99
        scope.gpu_end(handle)
        assert planning.fort_scope_plan_validate(scope.handle) == 8
        scope.check(planning.fort_scope_wait(scope.handle))
    elif mutation == "cancel_access":
        scope.cpu_begin(handle, access(flags=1))
        assert planning.fort_scope_plan_validate(scope.handle) == 8
        scope.check(planning.fort_scope_cancel_access(scope.handle, handle))
    elif mutation == "budget":
        scope.check(planning.fort_scope_set_device_budget(scope.handle, 64))
    elif mutation == "wait":
        scope.check(planning.fort_scope_wait(scope.handle))
    elif mutation == "gpu_enter_leave":
        previous, stream = c.c_int(), c.c_void_p()
        scope.check(planning.fort_scope_gpu_enter(scope.handle, c.byref(previous), c.byref(stream)))
        scope.check(planning.fort_scope_gpu_leave(scope.handle, previous))
    validate(scope)
    validate(scope)
    rows, _ = diagnostics(capfd)
    proofs = [row for row in rows if row["event"] == "definition_validation"]
    successful = [row for row in proofs if row["status"] == 0]
    assert [row["cache_hit"] for row in successful] == [False, True, False, True]
    old = successful[0]
    new = successful[2]
    assert (old["query_generation"], old["state_generation"]) != (new["query_generation"], new["state_generation"])
    assert all(not row["cache_hit"] for row in proofs if row["status"] != 0)
    scope.close()
    if mutation in {"host_write", "device_write"}:
        assert values[3] == 99


def test_definition_loss_and_failed_validations_never_reuse_a_success(planning, monkeypatch, capfd):
    monkeypatch.setenv("FORT_RUNTIME_TRACE", "1")
    scope, values, handle = query(planning)
    validate(scope)
    scope.check(planning.fort_scope_forget_definition(scope.handle, handle))
    for _ in range(2):
        assert planning.fort_scope_plan_validate(scope.handle) == 7
        assert "uninitialized_read" in planning.fort_scope_error().decode()
    assert planning.fort_scope_host_begin(scope.handle, handle, c.byref(access(flags=1))) == 7
    scope.cpu_begin(handle, access(flags=6))
    for index in range(8):
        values[index] = 2 * index
    scope.cpu_end(handle)
    validate(scope)
    validate(scope)
    rows, _ = diagnostics(capfd)
    proofs = [row for row in rows if row["event"] == "definition_validation"]
    assert [row["status"] for row in proofs] == [0, 7, 7, 0, 0]
    assert [row["cache_hit"] for row in proofs] == [False, False, False, False, True]
    assert scope.stats().allocations == scope.stats().uploads == scope.stats().downloads == 0
    scope.close()
    assert list(values) == [2 * index for index in range(8)]


def test_unregistration_rejects_old_query_and_reallocation_needs_new_bindings(planning, monkeypatch, capfd):
    monkeypatch.setenv("FORT_RUNTIME_TRACE", "1")
    scope, _, handle = query(planning)
    validate(scope)
    scope.check(planning.fort_scope_unregister(scope.handle, handle))
    assert planning.fort_scope_plan_validate(scope.handle) == 5
    replacement = (c.c_double * 5)(1, 2, 3, 4, 5)
    status, new_handle = scope.register(replacement, [5], generation=2, lower=[-20])
    scope.check(status)
    assert planning.fort_scope_plan_validate(scope.handle) == 5
    scope.check(planning.fort_scope_plan_reset(scope.handle))
    record(scope, 103, new_handle, access(flags=1))
    validate(scope)
    validate(scope)
    rows, _ = diagnostics(capfd)
    proofs = [row for row in rows if row["event"] == "definition_validation"]
    assert [row["status"] for row in proofs] == [0, 5, 5, 0, 0]
    assert [row["cache_hit"] for row in proofs] == [False, False, False, False, True]
    scope.close()


def test_preview_rechecks_coherence_and_driver_changes_from_other_context(planning, monkeypatch, capfd):
    monkeypatch.setenv("FORT_RUNTIME_TRACE", "1")
    scope, _, handle = query(planning, device=37)
    calibration = costs()
    calibration.cold_driver_startup_seconds = 10
    assert not select(scope, calibration, -1).gpu_units
    assert not select(scope, calibration, -1).gpu_units
    warmer = Scope(planning)
    warmer.close()
    warmer.check(planning.fort_scope_create(37, c.byref(warmer.handle)))
    previous, stream = c.c_int(), c.c_void_p()
    warmer.check(planning.fort_scope_gpu_enter(warmer.handle, c.byref(previous), c.byref(stream)))
    assert select(scope, calibration, -1).gpu_units == 1
    assert select(scope, calibration, -1).gpu_units == 1
    # A coherence access also invalidates an otherwise identical preview.
    scope.cpu_begin(handle, access(flags=1))
    scope.cpu_end(handle)
    assert select(scope, calibration, -1).gpu_units == 1
    _, timings = diagnostics(capfd)
    assert [row["cache_hit"] for row in timings if row["phase"] == "selection"] == ["0", "1", "0", "1", "0"]
    warmer.close()
    scope.close()


def test_successful_definition_proof_never_defines_simulated_outputs(planning, monkeypatch, capfd):
    monkeypatch.setenv("FORT_RUNTIME_TRACE", "1")
    scope, values, handle = query(planning, initialized=False)
    scope.check(planning.fort_scope_plan_reset(scope.handle))
    record(scope, 104, handle, access(flags=6))
    record(scope, 105, handle, access(flags=1))
    validate(scope)
    validate(scope)
    assert planning.fort_scope_host_begin(scope.handle, handle, c.byref(access(flags=1))) == 7
    validate(scope)
    rows, _ = diagnostics(capfd)
    proofs = [row for row in rows if row["event"] == "definition_validation"]
    assert [row["cache_hit"] for row in proofs] == [False, True, False]
    assert bytes(scope.stats()) == bytes(type(scope.stats())())
    scope.close()
    assert list(values) == list(range(8))


def test_poisoned_execution_cannot_restore_cached_proof_or_replay(planning, monkeypatch):
    scope, values, handle = query(planning)
    validate(scope)
    pointer = scope.gpu(handle, access(flags=3))
    pointer[0] = 999
    scope.gpu_end(handle)
    monkeypatch.setenv("FORT_SCOPE_TEST_FAIL_WAIT", "1")
    assert planning.fort_scope_wait(scope.handle) == 6
    assert planning.fort_scope_plan_validate(scope.handle) == 6
    assert planning.fort_scope_plan_reset(scope.handle) == 6
    assert planning.fort_scope_host_begin(scope.handle, handle, c.byref(access(flags=1))) == 6
    assert planning.fort_scope_close(scope.handle) == 6
    assert list(values) == list(range(8))
    scope.check(planning.fort_scope_abandon(scope.handle))
