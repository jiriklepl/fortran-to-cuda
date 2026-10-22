"""Whole-task reservations retain mirrors and evict only safe, cold storage."""
# ruff: noqa: F811 - pytest discovers the imported module-scoped fixture.
from __future__ import annotations

import ctypes as c

import pytest

from compiler.tests.test_scoped_runtime import SIZE, TOKEN, Scope, Section, access, runtime  # noqa: F401


class Residency(c.Structure):
    _fields_ = [(name, c.c_uint32) for name in ("version", "reserved")] + [
        (name, TOKEN) for name in ("reserve_batches", "requested_resources", "reuse_hits", "evictions",
                                 "evicted_bytes", "dirty_evictions", "dirty_download_bytes", "pressure_flushes")]


@pytest.fixture
def residency(runtime):
    runtime.fort_scope_device_reserve_many_v1.argtypes = [TOKEN, c.POINTER(TOKEN), SIZE, c.POINTER(c.c_void_p)]
    runtime.fort_scope_residency_stats_get_v1.argtypes = [TOKEN, c.POINTER(Residency)]
    runtime.fort_scope_buffer_valid_v1.argtypes = [TOKEN, TOKEN, c.POINTER(c.c_int)]
    runtime.fort_scope_admit_source_reads_v1.argtypes = [TOKEN, TOKEN, c.POINTER(Section), SIZE]
    return runtime


def reserve(scope, *handles, outputs=None):
    packed = (TOKEN * len(handles))(*(handle.value for handle in handles))
    outputs = outputs if outputs is not None else (c.c_void_p * len(handles))()
    status = scope.lib.fort_scope_device_reserve_many_v1(scope.handle, packed, len(handles), outputs)
    return status, outputs


def statistics(scope):
    stats = Residency()
    scope.check(scope.lib.fort_scope_residency_stats_get_v1(scope.handle, c.byref(stats)))
    assert stats.version == 1
    return stats


def register(scope, identity, *, initialized=True, defined=None):
    host = (c.c_double * 8)(*(identity * 100 + i for i in range(8)))
    status, handle = scope.register(host, [8], identity=identity, initialized=initialized, defined=defined)
    scope.check(status)
    return host, handle


@pytest.mark.native
def test_lru_dirty_eviction_preserves_values_and_reloads_exactly(residency):
    scope = Scope(residency)
    scope.check(residency.fort_scope_set_device_budget(scope.handle, 128))
    host_a, a = register(scope, 1)
    _, b = register(scope, 2)
    _, other = register(scope, 3)
    device = scope.gpu(a, access(flags=3))
    for i in range(8):
        device[i] += 1000
    scope.gpu_end(a)
    scope.gpu(b, access(flags=1))
    scope.gpu_end(b)
    before = scope.stats()
    scope.check(reserve(scope, other)[0])
    assert list(host_a) == [1100 + i for i in range(8)]
    assert scope.stats().download_bytes - before.download_bytes == 64
    observed = statistics(scope)
    assert (observed.evictions, observed.evicted_bytes, observed.dirty_evictions, observed.dirty_download_bytes) == (1, 64, 1, 64)
    device = scope.gpu(a, access(flags=1))
    assert list(device[:8]) == list(host_a)
    scope.gpu_end(a)
    assert scope.stats().allocated_bytes == scope.stats().peak_device_bytes == 128
    scope.close()


@pytest.mark.native
def test_read_mirror_eviction_preserves_host_without_download(residency):
    scope = Scope(residency)
    scope.check(residency.fort_scope_set_device_budget(scope.handle, 64))
    host, a = register(scope, 1)
    _, b = register(scope, 2)
    scope.gpu(a, access(flags=1))
    scope.gpu_end(a)
    before = scope.stats()
    scope.check(reserve(scope, b)[0])
    scope.cpu_begin(a, access(flags=1))
    assert list(host) == [100 + i for i in range(8)]
    scope.cpu_end(a)
    assert scope.stats().download_bytes == before.download_bytes == 0
    assert (statistics(scope).evictions, statistics(scope).dirty_evictions) == (1, 0)
    scope.close()


@pytest.mark.native
def test_impossible_task_is_rejected_before_dirty_eviction_or_output_mutation(residency):
    scope = Scope(residency)
    scope.check(residency.fort_scope_set_device_budget(scope.handle, 64))
    host, a = register(scope, 1)
    _, b = register(scope, 2)
    device = scope.gpu(a, access(flags=6))
    for i in range(8):
        device[i] = 999 + i
    scope.gpu_end(a)
    before, metrics = scope.stats(), statistics(scope)
    outputs = (c.c_void_p * 2)(0x1234, 0x5678)
    assert reserve(scope, a, b, outputs=outputs)[0] == 4
    assert list(outputs) == [0x1234, 0x5678]
    assert bytes(scope.stats()) == bytes(before)
    assert bytes(statistics(scope)) == bytes(metrics)
    assert host[0] == 100
    device = scope.gpu(a, access(flags=1))
    assert device[0] == 999
    scope.gpu_end(a)
    scope.close()
    assert host[0] == 999


@pytest.mark.native
def test_whole_task_pins_duplicates_and_updates_lru_on_reuse(residency):
    scope = Scope(residency)
    scope.check(residency.fort_scope_set_device_budget(scope.handle, 128))
    arrays = [register(scope, identity) for identity in range(1, 4)]
    a, b, other = [handle for _, handle in arrays]
    scope.check(reserve(scope, a, b)[0])
    status, duplicate = reserve(scope, a, a)
    scope.check(status)
    assert duplicate[0] == duplicate[1]
    scope.check(reserve(scope, a, other)[0])
    allocations = scope.stats().allocations
    scope.check(reserve(scope, a)[0])
    assert scope.stats().allocations == allocations  # touched A survived
    scope.check(reserve(scope, b)[0])
    assert scope.stats().allocations == allocations + 1  # cold B was evicted
    scope.close()


@pytest.mark.native
def test_eviction_publishes_exact_defined_partial_values_and_preserves_holes(residency):
    scope = Scope(residency)
    scope.check(residency.fort_scope_set_device_budget(scope.handle, 64))
    host, a = register(scope, 1, defined=[([2], [6])])
    _, b = register(scope, 2)
    effect = access(read=[([2], [6])], write=[([2], [6])], overwrite=[([3], [4])])
    device = scope.gpu(a, effect)
    for i in range(2, 6):
        device[i] = 1000 + i
    scope.gpu_end(a)
    scope.check(reserve(scope, b)[0])
    assert list(host) == [100, 101, 1002, 1003, 1004, 1005, 106, 107]
    assert scope.stats().upload_bytes == scope.stats().download_bytes == 32
    assert residency.fort_scope_host_begin(scope.handle, a, c.byref(access(flags=1))) == 7
    scope.cpu_begin(a, access(read=[([2], [6])]))
    scope.cpu_end(a)
    scope.close()


@pytest.mark.native
def test_prepared_access_cannot_be_evicted_or_reserved(residency):
    scope = Scope(residency)
    scope.check(residency.fort_scope_set_device_budget(scope.handle, 64))
    host, a = register(scope, 1)
    _, b = register(scope, 2)
    device = scope.gpu(a, access(flags=3))
    assert reserve(scope, a)[0] == 8
    assert reserve(scope, b)[0] == 4
    assert not statistics(scope).evictions
    device[0] = 987
    scope.gpu_end(a)
    scope.check(reserve(scope, b)[0])
    scope.cpu_begin(a, access(flags=1))
    assert host[0] == 987
    scope.cpu_end(a)
    scope.close()


@pytest.mark.native
def test_borrowed_handle_membership_distinguishes_context_epochs_without_cuda(residency):
    first, second = Scope(residency), Scope(residency)
    _, old = register(first, 1)
    _, current = register(second, 1)
    present = c.c_int(-1)
    first.check(residency.fort_scope_buffer_valid_v1(first.handle, old, c.byref(present)))
    assert present.value == 1
    second.check(residency.fort_scope_buffer_valid_v1(second.handle, old, c.byref(present)))
    assert present.value == 0
    first.close()
    second.check(residency.fort_scope_buffer_valid_v1(second.handle, old, c.byref(present)))
    assert present.value == 0
    second.check(residency.fort_scope_buffer_valid_v1(second.handle, current, c.byref(present)))
    assert present.value == 1
    assert not second.stats().allocations
    assert residency.fort_scope_buffer_valid_v1(first.handle, old, c.byref(present)) == 2
    second.close()


@pytest.mark.native
def test_eviction_preserves_physically_dirty_values_after_semantic_definition_kill(residency):
    scope = Scope(residency)
    scope.check(residency.fort_scope_set_device_budget(scope.handle, 64))
    host, a = register(scope, 1)
    _, b = register(scope, 2)
    device = scope.gpu(a, access(flags=6))
    for i in range(8):
        device[i] = 5000 + i
    scope.gpu_end(a)
    scope.check(residency.fort_scope_forget_definition(scope.handle, a))
    assert not scope.stats().download_bytes
    scope.check(reserve(scope, b)[0])
    assert list(host) == [5000 + i for i in range(8)]
    assert scope.stats().download_bytes == 64
    assert residency.fort_scope_host_begin(scope.handle, a, c.byref(access(flags=1))) == 7
    reads = access(read=[([0], [8])])
    scope.check(residency.fort_scope_admit_source_reads_v1(scope.handle, a, reads.reads, reads.read_count))
    scope.cpu_begin(a, access(flags=1))
    assert list(host) == [5000 + i for i in range(8)]
    scope.cpu_end(a)
    assert scope.stats().download_bytes == 64
    scope.close()
