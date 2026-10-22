"""Original native execution explicitly retires only an unstarted CPU choice."""
# ruff: noqa: F811 - pytest discovers imported fixtures under these names.
from __future__ import annotations

import ctypes as c

import pytest

from compiler.tests.test_scoped_access_batch import pack
from compiler.tests.test_scoped_continuation_runtime import Report, consume
from compiler.tests.test_scoped_graph_queue import QueueStats, enqueue, reserve, stats
from compiler.tests.test_scoped_graph_replay import Task
from compiler.tests.test_scoped_planning_runtime import (
    Binding,
    costs,
    planning,  # noqa: F401 - imported pytest fixture
    record,
    select,
)
from compiler.tests.test_scoped_runtime import SIZE, TOKEN, Scope, access, runtime  # noqa: F401


@pytest.fixture
def handoff(planning):
    signatures = {
        "plan_reset_mode": [TOKEN, c.c_uint32],
        "plan_report_v2": [TOKEN, c.POINTER(Report)],
        "task_graph_enable_v1": [TOKEN],
        "device_reserve_v1": [TOKEN, TOKEN, c.POINTER(c.c_void_p)],
        "graph_enqueue_v1": [TOKEN, TOKEN, c.POINTER(Task), c.POINTER(SIZE)],
        "graph_queue_cancel_v1": [TOKEN],
        "graph_queue_stats_get_v1": [TOKEN, c.POINTER(QueueStats)],
    }
    for name, signature in signatures.items():
        getattr(planning, "fort_scope_" + name).argtypes = signature
    return planning


def cancel(scope):
    # Resolve lazily so a baseline run first demonstrates the unfinished-plan
    # failure, before discovering the new explicit handoff API is absent.
    function = scope.lib.fort_scope_plan_cancel_native_v1
    function.argtypes = [TOKEN]
    return function(scope.handle)


def begin(scope):
    scope.check(scope.lib.fort_scope_task_graph_enable_v1(scope.handle))
    scope.check(scope.lib.fort_scope_plan_reset_mode(scope.handle, 1))


def choose_cpu(scope, compatible=1):
    calibration = costs()
    calibration.gpu_flops = calibration.gpu_bandwidth = 1
    scope.check(scope.lib.fort_scope_plan_validate(scope.handle))
    decision = select(scope, calibration, compatible)
    assert decision.gpu_units == 0
    assert decision.cpu_units > 0
    return decision


@pytest.mark.parametrize("compatible", [0, 1])
def test_cpu_choice_requires_explicit_handoff_before_native_fallback(handoff, compatible):
    scope = Scope(handoff)
    values = (c.c_double * 8)(*range(8))
    status, handle = scope.register(values, [8])
    scope.check(status)
    begin(scope)
    record(scope, 11, handle, access(flags=3))
    choose_cpu(scope, compatible)
    before = bytes(scope.stats())
    # These guards exposed the original source fallback failure. Neither
    # ordinary reset nor close silently discards an installed CPU schedule.
    assert handoff.fort_scope_plan_reset_mode(scope.handle, 1) == 8
    assert handoff.fort_scope_close(scope.handle) == 8
    scope.check(cancel(scope))
    assert bytes(scope.stats()) == before
    assert list(values) == list(range(8))
    assert handoff.fort_scope_plan_report_v2(scope.handle, c.byref(Report())) == 8
    scope.check(handoff.fort_scope_plan_reset_mode(scope.handle, 1))
    # The original body now uses its independently checked effects exactly
    # once. Cancellation never executes the predicted companion CPU worker.
    scope.cpu_begin(handle, access(flags=3))
    for index in range(8):
        values[index] += 10
    scope.cpu_end(handle)
    scope.close()
    assert list(values) == [index + 10 for index in range(8)]


def test_cancellation_does_not_commit_predicted_intermediate_definitions(handoff):
    scope = Scope(handoff)
    values = (c.c_double * 8)(*range(8))
    status, handle = scope.register(values, [8], initialized=False)
    scope.check(status)
    begin(scope)
    record(scope, 21, handle, access(flags=6))
    record(scope, 22, handle, access(flags=1))
    choose_cpu(scope)
    scope.check(cancel(scope))
    assert scope.stats().allocations == scope.stats().uploads == scope.stats().downloads == 0
    # Only the query's private simulation defined this intermediate.
    assert handoff.fort_scope_host_begin(scope.handle, handle, c.byref(access(flags=1))) == 7
    scope.check(handoff.fort_scope_plan_reset_mode(scope.handle, 1))
    record(scope, 23, handle, access(flags=1))
    assert handoff.fort_scope_plan_validate(scope.handle) == 7
    scope.check(handoff.fort_scope_plan_reset_mode(scope.handle, 1))
    scope.cpu_begin(handle, access(flags=6))
    for index in range(8):
        values[index] = 100 + index
    scope.cpu_end(handle)
    bindings = record(scope, 24, handle, access(flags=1))
    choose_cpu(scope)
    consume(scope, 24, bindings, 0)
    scope.cpu_begin(handle, access(flags=1))
    scope.cpu_end(handle)
    scope.check(handoff.fort_scope_wait(scope.handle))
    report = Report()
    scope.check(handoff.fort_scope_plan_report_v2(scope.handle, c.byref(report)))
    assert report.owner_available == 0
    scope.close()
    assert list(values) == [100 + index for index in range(8)]


def test_native_handoff_preserves_dirty_device_version_and_read_mirror(handoff):
    scope = Scope(handoff)
    values = (c.c_double * 8)(*range(8))
    status, handle = scope.register(values, [8])
    scope.check(status)
    device = scope.gpu(handle, access(flags=3))
    for index in range(8):
        device[index] += 100
    scope.gpu_end(handle)
    begin(scope)
    record(scope, 31, handle, access(flags=1))
    choose_cpu(scope)
    before = bytes(scope.stats())
    scope.check(cancel(scope))
    assert bytes(scope.stats()) == before
    assert list(values) == list(range(8))
    scope.cpu_begin(handle, access(flags=1))
    assert list(values) == [index + 100 for index in range(8)]
    scope.cpu_end(handle)
    assert scope.stats().download_bytes == 64
    uploads = scope.stats().upload_bytes
    device = scope.gpu(handle, access(flags=1))
    assert list(device[:8]) == list(values)
    scope.gpu_end(handle)
    assert scope.stats().upload_bytes == uploads
    scope.close()


@pytest.mark.parametrize("started", ["prepared", "completed_native", "consumed_cpu", "gpu_choice", "accepted_queue"])
def test_native_handoff_rejects_prepared_started_and_gpu_schedules(handoff, started):
    scope = Scope(handoff)
    values = (c.c_double * 8)(*range(8))
    status, handle = scope.register(values, [8])
    scope.check(status)
    if started == "accepted_queue":
        _, queued = reserve(scope, 2)
    begin(scope)
    if started in {"prepared", "completed_native"}:
        prefix = (Binding * 1)(Binding(handle, access(flags=1)))
        # This is an original native operation, not a numerical worker.
        scope.check(handoff.fort_scope_plan_add(scope.handle, 0, 40, prefix, 1, 0, 0, 0))
    bindings = record(scope, 41, handle, access(flags=1))
    if started == "gpu_choice":
        decision = select(scope, costs())
        assert decision.gpu_units == 1
    else:
        choose_cpu(scope)
    if started in {"prepared", "completed_native"}:
        # A NATIVE prefix may finish before the first numerical worker is
        # consumed. A zero cursor alone cannot prove execution is unstarted.
        scope.cpu_begin(handle, access(flags=1))
        if started == "completed_native":
            scope.cpu_end(handle)
    elif started == "consumed_cpu":
        consume(scope, 41, bindings, 0)
    elif started == "accepted_queue":
        scope.check(enqueue(scope, 42, pack((queued, access(flags=6)))))
    before = bytes(scope.stats())
    assert cancel(scope) == 8
    assert bytes(scope.stats()) == before
    if started == "prepared":
        scope.cpu_end(handle)
        assert cancel(scope) == 8
    if started != "consumed_cpu":
        consume(scope, 41, bindings, int(started == "gpu_choice"))
    if started == "accepted_queue":
        assert stats(scope).pending_tasks == 1
        scope.check(handoff.fort_scope_graph_queue_cancel_v1(scope.handle))
    scope.close()


def test_native_handoff_preserves_earlier_accepted_queue_and_virtual_definitions(handoff):
    scope = Scope(handoff)
    _, queued = reserve(scope, 1)
    _, native = reserve(scope, 2, initialized=True)
    scope.check(enqueue(scope, 51, pack((queued, access(flags=6)))))
    begin(scope)
    record(scope, 52, native, access(flags=1))
    choose_cpu(scope)
    before = bytes(stats(scope))
    scope.check(cancel(scope))
    assert bytes(stats(scope)) == before
    assert stats(scope).pending_tasks == 1
    scope.check(handoff.fort_scope_plan_reset_mode(scope.handle, 1))
    record(scope, 53, queued, access(flags=1))
    scope.check(handoff.fort_scope_plan_validate(scope.handle))
    # Explicitly discard the never-submitted fixture task so the CPU backend
    # can retire storage; handoff itself must leave that ownership untouched.
    scope.check(handoff.fort_scope_graph_queue_cancel_v1(scope.handle))
    assert handoff.fort_scope_host_begin(scope.handle, queued, c.byref(access(flags=1))) == 7
    scope.close()


def test_native_handoff_requires_finalized_continuation_choice(handoff):
    scope = Scope(handoff)
    assert cancel(scope) == 8
    values = (c.c_double * 8)(*range(8))
    status, handle = scope.register(values, [8])
    scope.check(status)
    begin(scope)
    record(scope, 61, handle, access(flags=1))
    select(scope, costs(), -1)
    assert cancel(scope) == 8
    scope.check(handoff.fort_scope_plan_reset(scope.handle))
    record(scope, 62, handle, access(flags=1))
    choose_cpu(scope)
    assert cancel(scope) == 8  # COMPLETE native fallback never installed a plan.
    scope.close()
