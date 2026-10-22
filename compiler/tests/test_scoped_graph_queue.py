"""Owned deferred tasks keep virtual definitions separate from executed state."""
# ruff: noqa: F811 - imported pytest fixture.
from __future__ import annotations

import ctypes as c

import pytest

from compiler.tests.test_scoped_access_batch import Binding, pack
from compiler.tests.test_scoped_graph_replay import Kernel, Task, graphs, kernel  # noqa: F401
from compiler.tests.test_scoped_runtime import SIZE, TOKEN, Scope, access, runtime  # noqa: F401


class QueueStats(c.Structure):
    _fields_ = [(name, c.c_uint32) for name in ("version", "queue_enabled", "graphs_enabled", "reserved")] + [
        (name, TOKEN) for name in ("accepted_tasks", "submissions", "conflict_flushes", "explicit_flushes",
                                 "pending_tasks", "peak_pending_tasks", "copied_argument_bytes",
                                 "direct_submissions", "direct_kernels", "graph_resource_fallbacks")]


@pytest.fixture
def queue(graphs):
    graphs.fort_scope_graph_enqueue_v1.argtypes = [TOKEN, TOKEN, c.POINTER(Task), c.POINTER(SIZE)]
    graphs.fort_scope_graph_flush_v1.argtypes = graphs.fort_scope_graph_queue_cancel_v1.argtypes = [TOKEN]
    graphs.fort_scope_graph_queue_stats_get_v1.argtypes = [TOKEN, c.POINTER(QueueStats)]
    graphs.fort_scope_plan_reset.argtypes = graphs.fort_scope_plan_validate.argtypes = [TOKEN]
    graphs.fort_scope_plan_add.argtypes = [TOKEN, c.c_uint32, TOKEN, c.POINTER(Binding), SIZE,
                                         c.c_double, c.c_double, c.c_int]
    return graphs


def stats(scope):
    out = QueueStats()
    scope.check(scope.lib.fort_scope_graph_queue_stats_get_v1(scope.handle, c.byref(out)))
    assert out.version == 1
    return out


def reserve(scope, identity, *, initialized=False):
    values = (c.c_double * 8)(*range(8))
    status, handle = scope.register(values, [8], identity=identity, initialized=initialized)
    scope.check(status)
    pointer = c.c_void_p()
    scope.check(scope.lib.fort_scope_device_reserve_v1(scope.handle, handle, c.byref(pointer)))
    return values, handle


def enqueue(scope, key, bindings, *, scalar=7):
    value = c.c_int(scalar)
    arguments = (c.c_void_p * 1)(c.addressof(value))
    descriptor = kernel()
    descriptor.arguments, descriptor.argument_count = arguments, 1
    task = Task(descriptor, bindings, len(bindings))
    sizes = (SIZE * 1)(c.sizeof(value))
    return scope.lib.fort_scope_graph_enqueue_v1(scope.handle, key, c.byref(task), sizes)


@pytest.mark.native
def test_virtual_definitions_span_entries_and_native_independent_work(queue):
    scope = Scope(queue)
    _, a = reserve(scope, 1, initialized=True)
    _, b = reserve(scope, 2)
    _, out = reserve(scope, 3)
    host, x = reserve(scope, 4, initialized=True)
    scope.check(enqueue(scope, 101, pack((a, access(flags=1)), (b, access(flags=6)))))
    scope.cpu_begin(x, access(flags=3))
    host[0] = 999
    scope.cpu_end(x)
    assert stats(scope).pending_tasks == 1
    scope.check(enqueue(scope, 102, pack((b, access(flags=1)), (out, access(flags=6)))))
    scope.check(queue.fort_scope_plan_reset(scope.handle))
    reads = pack((out, access(flags=1)))
    scope.check(queue.fort_scope_plan_add(scope.handle, 0, 0, reads, 1, 0, 0, 0))
    scope.check(queue.fort_scope_plan_validate(scope.handle))
    assert scope.stats().launches == scope.stats().uploads == scope.stats().downloads == 0
    assert stats(scope).accepted_tasks == stats(scope).pending_tasks == 2
    # The reference backend cannot execute CUDA. Failed drain must retain all
    # accepted work and leave physical/semantic execution state uncommitted.
    assert queue.fort_scope_host_begin(scope.handle, out, c.byref(access(flags=1))) == 4
    assert stats(scope).pending_tasks == 2
    scope.check(queue.fort_scope_graph_queue_cancel_v1(scope.handle))
    assert queue.fort_scope_host_begin(scope.handle, out, c.byref(access(flags=1))) == 7
    scope.close()


@pytest.mark.native
def test_invalid_enqueue_preserves_previously_accepted_tasks(queue):
    scope = Scope(queue)
    _, a = reserve(scope, 1)
    _, b = reserve(scope, 2)
    scope.check(enqueue(scope, 1, pack((a, access(flags=6)))))
    before = bytes(stats(scope))
    assert enqueue(scope, 2, pack((b, access(flags=1)))) == 7
    assert bytes(stats(scope)) == before
    assert enqueue(scope, 3, pack((TOKEN(2**63), access(flags=1)))) == 2
    assert bytes(stats(scope)) == before
    # Failed validation cannot leave its private candidate definitions cached.
    scope.check(enqueue(scope, 4, pack((a, access(flags=1)), (b, access(flags=6)))))
    assert stats(scope).pending_tasks == 2
    scope.check(queue.fort_scope_graph_queue_cancel_v1(scope.handle))
    assert queue.fort_scope_host_begin(scope.handle, a, c.byref(access(flags=1))) == 7
    scope.close()


@pytest.mark.native
def test_deferred_chain_exceeds_old_source_windows_without_executing(queue):
    scope = Scope(queue)
    host, handle = reserve(scope, 1)
    for index in range(100):
        scope.check(enqueue(scope, index + 1, pack((handle, access(flags=6 if index == 0 else 3))), scalar=index))
    result = stats(scope)
    assert result.accepted_tasks == result.pending_tasks == result.peak_pending_tasks == 100
    assert result.copied_argument_bytes == 100 * c.sizeof(c.c_int)
    assert scope.stats().launches == 0
    scope.check(queue.fort_scope_graph_queue_cancel_v1(scope.handle))
    scope.close()
    assert list(host) == list(range(8))


@pytest.mark.native
def test_context_captures_queue_policies_once(queue, monkeypatch):
    monkeypatch.setenv("FORT_SCOPE_GRAPH_QUEUE", "0")
    monkeypatch.setenv("FORT_SCOPE_CUDA_GRAPHS", "0")
    scope = Scope(queue)
    monkeypatch.setenv("FORT_SCOPE_GRAPH_QUEUE", "1")
    monkeypatch.setenv("FORT_SCOPE_CUDA_GRAPHS", "1")
    result = stats(scope)
    assert result.queue_enabled == result.graphs_enabled == 0
    scope.close()


@pytest.mark.native
def test_enqueue_requires_owned_sizes_and_reserved_storage(queue):
    scope = Scope(queue)
    _, handle = reserve(scope, 1)
    bindings = pack((handle, access(flags=6)))
    value = c.c_int(7)
    arguments = (c.c_void_p * 1)(c.addressof(value))
    descriptor = Kernel(1, 0, 1, (c.c_uint32 * 3)(1, 1, 1), (c.c_uint32 * 3)(32, 1, 1), 0, arguments, 1)
    task = Task(descriptor, bindings, 1)
    assert queue.fort_scope_graph_enqueue_v1(scope.handle, 7, c.byref(task), None) == 1
    assert queue.fort_scope_graph_enqueue_v1(scope.handle, 7, c.byref(task), (SIZE * 1)(0)) == 1
    assert stats(scope).pending_tasks == 0
    scope.close()


@pytest.mark.native
def test_graph_runtime_accepts_query_beyond_legacy_record_and_worker_caps(queue):
    scope = Scope(queue)
    queue.fort_scope_task_graph_enable_v1.argtypes = [TOKEN]
    scope.check(queue.fort_scope_task_graph_enable_v1(scope.handle))
    values = (c.c_double * 8)(*range(8))
    status, handle = scope.register(values, [8])
    scope.check(status)
    scope.check(queue.fort_scope_plan_reset(scope.handle))
    bindings = pack((handle, access(flags=3)))
    for index in range(320):
        scope.check(queue.fort_scope_plan_add(scope.handle, 1, index + 1, bindings, 1, 100, 128, 1))
        if index % 3 == 0:
            scope.check(queue.fort_scope_plan_add(scope.handle, 0, 0, None, 0, 0, 0, 0))
    scope.check(queue.fort_scope_plan_validate(scope.handle))
    assert scope.stats().launches == scope.stats().allocations == 0
    scope.close()
