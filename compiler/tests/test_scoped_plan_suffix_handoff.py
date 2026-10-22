"""A proved source suffix can fall back without replaying its GPU prefix."""
# ruff: noqa: F811 - pytest discovers imported fixtures under these names.
from __future__ import annotations

import ctypes as c

import pytest

from compiler.tests.test_scoped_access_batch import pack
from compiler.tests.test_scoped_continuation_runtime import Report, consume, gpu
from compiler.tests.test_scoped_graph_queue import enqueue, reserve, stats
from compiler.tests.test_scoped_native_plan_handoff import begin, handoff  # noqa: F401
from compiler.tests.test_scoped_planning_runtime import Binding, costs, planning, record, select  # noqa: F401
from compiler.tests.test_scoped_runtime import TOKEN, Scope, access, runtime  # noqa: F401
from compiler.tests.test_scoped_scratch import Lease


def abandon(scope):
    function = scope.lib.fort_scope_plan_abandon_remaining_v1
    function.argtypes = [TOKEN]
    return function(scope.handle)


def install_gpu(scope):
    scope.check(scope.lib.fort_scope_plan_validate(scope.handle))
    decision = select(scope, costs())
    assert decision.gpu_units > 0


def test_pending_gpu_prefix_survives_suffix_native_fallback_and_read_mirror(handoff):
    scope = Scope(handoff)
    values = (c.c_double * 8)(*range(8))
    status, handle = scope.register(values, [8])
    scope.check(status)
    begin(scope)
    prefix = record(scope, 11, handle, access(flags=3))
    record(scope, 12, handle, access(flags=3))
    install_gpu(scope)
    consume(scope, 11, prefix, 1)
    gpu(scope, handle, access(flags=3), lambda value, _: value + 100)
    before = bytes(scope.stats())
    assert handoff.fort_scope_plan_reset_mode(scope.handle, 1) == 8
    assert handoff.fort_scope_close(scope.handle) == 8
    scope.check(abandon(scope))
    assert bytes(scope.stats()) == before
    assert list(values) == list(range(8))
    assert handoff.fort_scope_plan_report_v2(scope.handle, c.byref(Report())) == 8
    # The original suffix reads the accepted prefix once. Its read-only mirror
    # leaves the GPU version usable by a later task without another upload.
    scope.cpu_begin(handle, access(flags=1))
    assert list(values) == [index + 100 for index in range(8)]
    scope.cpu_end(handle)
    assert scope.stats().download_bytes == 64
    uploads = scope.stats().upload_bytes
    pointer = scope.gpu(handle, access(flags=1))
    assert list(pointer[:8]) == list(values)
    scope.gpu_end(handle)
    assert scope.stats().upload_bytes == uploads
    scope.close()


def test_abandoned_gpu_suffix_never_defines_its_predicted_output(handoff):
    scope = Scope(handoff)
    values = (c.c_double * 8)(*range(8))
    output = (c.c_double * 8)(*([-1] * 8))
    status, handle = scope.register(values, [8])
    scope.check(status)
    status, other = scope.register(output, [8], identity=2, initialized=False)
    scope.check(status)
    begin(scope)
    prefix = record(scope, 21, handle, access(flags=3))
    record(scope, 22, other, access(flags=6))
    record(scope, 23, other, access(flags=1))
    install_gpu(scope)
    consume(scope, 21, prefix, 1)
    gpu(scope, handle, access(flags=3), lambda value, _: value + 10)
    scope.check(abandon(scope))
    assert handoff.fort_scope_host_begin(scope.handle, other, c.byref(access(flags=1))) == 7
    begin(scope)
    record(scope, 24, other, access(flags=1))
    assert handoff.fort_scope_plan_validate(scope.handle) == 7
    # Drop the failed recording, then execute the original suffix once.
    begin(scope)
    scope.cpu_begin(other, access(flags=6))
    for index in range(8):
        output[index] = index + 20
    scope.cpu_end(other)
    scope.close()
    assert list(values) == [index + 10 for index in range(8)]
    assert list(output) == [index + 20 for index in range(8)]


def test_accepted_queue_and_virtual_prefix_definitions_survive_suffix_handoff(handoff):
    scope = Scope(handoff)
    _, prefix_handle = reserve(scope, 1)
    _, suffix_handle = reserve(scope, 2)
    begin(scope)
    prefix = record(scope, 31, prefix_handle, access(flags=6))
    record(scope, 32, suffix_handle, access(flags=6))
    install_gpu(scope)
    consume(scope, 31, prefix, 1)
    scope.check(enqueue(scope, 31, pack((prefix_handle, access(flags=6)))))
    before = bytes(stats(scope)), bytes(scope.stats())
    scope.check(abandon(scope))
    assert (bytes(stats(scope)), bytes(scope.stats())) == before
    assert stats(scope).pending_tasks == 1
    # Query validation sees only the accepted producer's virtual definition.
    begin(scope)
    record(scope, 33, prefix_handle, access(flags=1))
    scope.check(handoff.fort_scope_plan_validate(scope.handle))
    begin(scope)
    record(scope, 34, suffix_handle, access(flags=1))
    assert handoff.fort_scope_plan_validate(scope.handle) == 7
    begin(scope)
    # A CPU-test backend cannot launch the fixture's opaque CUDA task. Explicit
    # test cleanup cancels it; suffix retirement itself preserved it bytewise.
    scope.check(handoff.fort_scope_graph_queue_cancel_v1(scope.handle))
    scope.close()


def test_prepared_suffix_access_rejects_abandon_without_discarding_schedule(handoff):
    scope = Scope(handoff)
    values = (c.c_double * 8)(*range(8))
    status, handle = scope.register(values, [8])
    scope.check(status)
    begin(scope)
    prefix = record(scope, 41, handle, access(flags=3))
    suffix = record(scope, 42, handle, access(flags=1))
    install_gpu(scope)
    consume(scope, 41, prefix, 1)
    gpu(scope, handle, access(flags=3), lambda value, _: value + 1)
    scope.cpu_begin(handle, access(flags=1))
    before = bytes(scope.stats())
    assert abandon(scope) == 8
    assert bytes(scope.stats()) == before
    assert handoff.fort_scope_plan_reset_mode(scope.handle, 1) == 8
    scope.cpu_end(handle)
    consume(scope, 42, suffix, 1)
    scope.close()
    assert list(values) == [index + 1 for index in range(8)]


def test_active_scratch_lease_prevents_suffix_handoff(handoff):
    handoff.fort_scope_scratch_acquire_v1.argtypes = [TOKEN, c.c_size_t, c.POINTER(Lease)]
    handoff.fort_scope_scratch_release_v1.argtypes = [TOKEN, TOKEN]
    scope = Scope(handoff)
    values = (c.c_double * 8)(*range(8))
    status, handle = scope.register(values, [8])
    scope.check(status)
    begin(scope)
    record(scope, 45, handle, access(flags=1))
    install_gpu(scope)
    lease = Lease()
    scope.check(handoff.fort_scope_scratch_acquire_v1(scope.handle, 0, c.byref(lease)))
    before = bytes(scope.stats())
    assert abandon(scope) == 8
    assert bytes(scope.stats()) == before
    assert handoff.fort_scope_plan_reset_mode(scope.handle, 1) == 8
    scope.check(handoff.fort_scope_scratch_release_v1(scope.handle, lease.token))
    scope.check(abandon(scope))
    scope.close()


@pytest.mark.parametrize("mismatch", ["unit", "effects"])
def test_prework_plan_mismatch_leaves_prefix_owned_for_original_suffix_once(handoff, mismatch):
    scope = Scope(handoff)
    values = (c.c_double * 8)(*range(8))
    status, handle = scope.register(values, [8])
    scope.check(status)
    begin(scope)
    prefix = record(scope, 46, handle, access(flags=3))
    suffix = record(scope, 47, handle, access(flags=3))
    install_gpu(scope)
    consume(scope, 46, prefix, 1)
    gpu(scope, handle, access(flags=3), lambda value, _: value + 100)
    changed = (Binding * 1)(Binding(handle, access(flags=1))) if mismatch == "effects" else suffix
    chosen = c.c_int(-1)
    before = bytes(scope.stats())
    status = handoff.fort_scope_plan_next(scope.handle, 48 if mismatch == "unit" else 47,
                                         changed, len(changed), c.byref(chosen))
    assert status == 9
    assert chosen.value == 0
    assert bytes(scope.stats()) == before
    assert handoff.fort_scope_plan_reset_mode(scope.handle, 1) == 8
    scope.check(abandon(scope))
    scope.cpu_begin(handle, access(flags=3))
    for index in range(8):
        values[index] += 3
    scope.cpu_end(handle)
    scope.close()
    assert list(values) == [index + 103 for index in range(8)]


@pytest.mark.parametrize("stage", ["empty", "recording", "preview", "complete"])
def test_suffix_handoff_requires_installed_continuation(handoff, stage):
    scope = Scope(handoff)
    if stage != "empty":
        values = (c.c_double * 8)(*range(8))
        status, handle = scope.register(values, [8])
        scope.check(status)
        begin(scope)
        if stage == "complete":
            scope.check(handoff.fort_scope_plan_reset(scope.handle))
        bindings = record(scope, 51, handle, access(flags=1))
        if stage == "preview":
            select(scope, costs(), -1)
        elif stage == "complete":
            install_gpu(scope)
    assert abandon(scope) == 8
    if stage == "complete":
        consume(scope, 51, bindings, 1)
    scope.close()
