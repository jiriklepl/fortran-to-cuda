"""Host tasks wait for conflicting DMA uses rather than unrelated device work."""
# ruff: noqa: F811 - imported pytest fixture.
from __future__ import annotations

import ctypes as c

import pytest

from compiler.tests.test_scoped_access_batch import Binding, pack
from compiler.tests.test_scoped_runtime import SIZE, TOKEN, Scope, access, runtime  # noqa: F401


class CompletionStats(c.Structure):
    _fields_ = [("version", c.c_uint32), ("reserved", c.c_uint32)] + [(name, TOKEN) for name in (
        "event_records", "event_queries", "event_waits", "stream_waits", "created_events", "reused_events",
        "independent_host_tasks")]


@pytest.fixture
def completions(runtime):
    runtime.fort_scope_completion_stats_get_v1.argtypes = [TOKEN, c.POINTER(CompletionStats)]
    runtime.fort_scope_access_begin_v1.argtypes = [TOKEN, c.POINTER(Binding), SIZE, c.c_int, c.POINTER(c.c_void_p)]
    return runtime


def stats(scope):
    out = CompletionStats()
    scope.check(scope.lib.fort_scope_completion_stats_get_v1(scope.handle, c.byref(out)))
    assert out.version == 1
    return out


def register(scope, identity):
    host = (c.c_double * 8)(*range(8))
    status, handle = scope.register(host, [8], identity=identity)
    scope.check(status)
    return host, handle


@pytest.mark.native
def test_host_task_on_disjoint_resource_does_not_wait(completions):
    scope = Scope(completions)
    _, gpu_buffer = register(scope, 1)
    host, cpu_buffer = register(scope, 2)
    device = scope.gpu(gpu_buffer, access(flags=6))
    device[0] = 100
    scope.gpu_end(gpu_buffer)
    before = scope.stats().waits
    scope.cpu_begin(cpu_buffer, access(flags=3))
    host[0] = 999
    scope.cpu_end(cpu_buffer)
    assert scope.stats().waits == before
    assert stats(scope).independent_host_tasks == 1
    scope.close()


@pytest.mark.native
def test_pending_upload_allows_host_reads_and_disjoint_writes_but_fences_overlap(completions):
    scope = Scope(completions)
    host, handle = register(scope, 1)
    middle = [([2], [6])]
    scope.gpu(handle, access(read=middle))
    scope.gpu_end(handle)
    before = scope.stats().waits
    scope.cpu_begin(handle, access(flags=1))
    assert list(host) == list(range(8))
    scope.cpu_end(handle)
    first = [([0], [2])]
    scope.cpu_begin(handle, access(write=first, overwrite=first))
    host[0], host[1] = -1, -2
    scope.cpu_end(handle)
    assert scope.stats().waits == before
    scope.cpu_begin(handle, access(write=middle, overwrite=middle))
    assert scope.stats().waits == before + 1
    scope.check(completions.fort_scope_cancel_access(scope.handle, handle))
    # Cancellation neither commits a numerical write nor discards a resident
    # mirror; the next identical device read needs no second upload.
    before_uploads = scope.stats().uploads
    device = scope.gpu(handle, access(read=middle))
    assert [device[i] for i in range(2, 6)] == [2, 3, 4, 5]
    scope.gpu_end(handle)
    assert scope.stats().uploads == before_uploads
    assert stats(scope).event_waits == 1
    scope.close()


@pytest.mark.native
def test_one_upload_batch_shares_completion_for_all_resources(completions):
    scope = Scope(completions)
    arrays = [register(scope, identity) for identity in range(1, 6)]
    bindings = pack(*[(handle, access(flags=1)) for _, handle in arrays])
    pointers = (c.c_void_p * len(arrays))()
    scope.check(completions.fort_scope_access_begin_v1(scope.handle, bindings, len(bindings), 1, pointers))
    for _, handle in arrays:
        scope.gpu_end(handle)
    assert stats(scope).event_records == 1
    writes = pack(*[(handle, access(flags=6)) for _, handle in arrays])
    before = scope.stats().waits
    scope.check(completions.fort_scope_access_begin_v1(scope.handle, writes, len(writes), 0, None))
    assert scope.stats().waits == before + 1
    assert stats(scope).event_waits == 1
    for _, handle in arrays:
        scope.check(completions.fort_scope_cancel_access(scope.handle, handle))
    scope.close()


@pytest.mark.native
def test_host_only_tasks_need_no_completion_tokens(completions):
    scope = Scope(completions)
    host, handle = register(scope, 1)
    for _ in range(5):
        scope.cpu_begin(handle, access(flags=3))
        host[0] += 1
        scope.cpu_end(handle)
    result = stats(scope)
    assert result.event_records == result.created_events == result.event_waits == result.stream_waits == 0
    assert result.independent_host_tasks == 5
    scope.close()


@pytest.mark.native
@pytest.mark.parametrize("destination", ["host", "device"])
def test_readonly_fragmentation_reconciliation_preserves_original_mirror(completions, destination):
    scope = Scope(completions)
    host = (c.c_double * 128)(*range(128))
    status, handle = scope.register(host, [128])
    scope.check(status)
    points = [([2 * i], [2 * i + 1]) for i in range(32)]
    if destination == "host":
        for lo, hi in points:
            device = scope.gpu(handle, access(write=[(lo, hi)], overwrite=[(lo, hi)]))
            device[lo[0]] = 1000 + lo[0]
            scope.gpu_end(handle)
        scope.cpu_begin(handle, access(flags=1))
        scope.cpu_end(handle)
        before = scope.stats().upload_bytes
        device = scope.gpu(handle, access(read=points))
        assert [device[lo[0]] for lo, _ in points] == [1000 + lo[0] for lo, _ in points]
        scope.gpu_end(handle)
        assert scope.stats().upload_bytes == before
    else:
        device = scope.gpu(handle, access(flags=6))
        for i in range(128):
            device[i] = 1000 + i
        scope.gpu_end(handle)
        for lo, hi in points:
            scope.cpu_begin(handle, access(write=[(lo, hi)], overwrite=[(lo, hi)]))
            host[lo[0]] = 2000 + lo[0]
            scope.cpu_end(handle)
        scope.gpu(handle, access(flags=1))
        scope.gpu_end(handle)
        before = scope.stats().download_bytes
        scope.cpu_begin(handle, access(read=points))
        assert [host[lo[0]] for lo, _ in points] == [2000 + lo[0] for lo, _ in points]
        scope.cpu_end(handle)
        assert scope.stats().download_bytes == before
    assert scope.stats().reconciliations >= 1
    scope.close()
