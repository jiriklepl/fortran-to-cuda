"""Reached source-read preconditions add only untracked host values."""
# ruff: noqa: F811 - pytest discovers the imported module-scoped fixture.
from __future__ import annotations

import ctypes as c

import pytest

from compiler.tests.test_scoped_planning_runtime import costs, planning, record, select  # noqa: F401
from compiler.tests.test_scoped_runtime import SIZE, TOKEN, Scope, Section, access, runtime  # noqa: F401


def admit(scope, buffer, boxes):
    operation = scope.lib.fort_scope_admit_source_reads_v1
    operation.argtypes = [TOKEN, TOKEN, c.POINTER(Section), SIZE]
    reads = access(read=boxes)
    return operation(scope.handle, buffer, reads.reads, reads.read_count)


def counters(scope):
    state = scope.stats()
    return tuple(getattr(state, name) for name, _type in state._fields_)


@pytest.mark.native
def test_exact_source_read_admission_preserves_dirty_device_and_unread_halos(runtime):
    scope = Scope(runtime)
    host = (c.c_double * 8)(*range(8))
    status, buffer = scope.register(host, [8], lower=[-3], defined=[([2], [4])])
    scope.check(status)
    effect = access(read=[([2], [4])], write=[([2], [4])])
    device = scope.gpu(buffer, effect)
    device[2], device[3] = 102, 103
    scope.gpu_end(buffer)
    before = counters(scope)
    scope.check(admit(scope, buffer, [([1], [6])]))
    scope.check(admit(scope, buffer, [([1], [6])]))
    assert counters(scope) == before
    # A repeated claim must not mark stale host values current over GPU writes.
    effect = access(read=[([1], [6])])
    device = scope.gpu(buffer, effect)
    assert list(device[1:6]) == [1, 102, 103, 4, 5]
    scope.gpu_end(buffer)
    assert scope.stats().upload_bytes == 5 * 8
    unread = access(read=[([0], [1])])
    assert runtime.fort_scope_host_begin(scope.handle, buffer, c.byref(unread)) == 7
    scope.close()
    assert list(host) == [0, 1, 102, 103, 4, 5, 6, 7]


@pytest.mark.native
def test_source_read_admission_has_no_implicit_storage_or_prepared_access_authority(runtime):
    scope = Scope(runtime)
    host = (c.c_double * 8)(*range(8))
    status, buffer = scope.register(host, [8], initialized=False)
    scope.check(status)
    assert admit(scope, buffer, [([0], [9])]) == 1
    read = access(read=[([0], [1])])
    assert runtime.fort_scope_host_begin(scope.handle, buffer, c.byref(read)) == 7
    scope.check(admit(scope, buffer, [([0], [1])]))
    scope.cpu_begin(buffer, read)
    assert admit(scope, buffer, [([1], [2])]) != 0
    assert b"prepared" in runtime.fort_scope_error()
    scope.cpu_end(buffer)
    unread = access(read=[([1], [2])])
    assert runtime.fort_scope_host_begin(scope.handle, buffer, c.byref(unread)) == 7
    scope.close()
    assert admit(scope, buffer, [([0], [1])]) == 2


@pytest.mark.native
def test_fragmented_source_read_admission_commits_neither_coverage_set(runtime):
    scope = Scope(runtime)
    host = (c.c_double * 100)(*range(100))
    initial = [([3 * i], [3 * i + 1]) for i in range(32)]
    status, buffer = scope.register(host, [100], defined=initial)
    scope.check(status)
    before = counters(scope)
    assert admit(scope, buffer, [([96], [97])]) == 5
    assert counters(scope) == before
    unread = access(read=[([96], [97])])
    assert runtime.fort_scope_host_begin(scope.handle, buffer, c.byref(unread)) == 7
    scope.cpu_begin(buffer, access(read=initial))
    scope.cpu_end(buffer)
    scope.close()
    assert list(host) == list(range(100))


@pytest.mark.native
def test_reached_claim_invalidates_failed_query_and_prices_only_exact_transfers(planning):
    scope = Scope(planning)
    host = (c.c_double * 8)(*range(8))
    status, buffer = scope.register(host, [8], initialized=False)
    scope.check(status)
    scope.check(planning.fort_scope_plan_reset(scope.handle))
    effect = access(read=[([2], [6])], write=[([2], [6])])
    record(scope, 701, buffer, effect)
    assert planning.fort_scope_plan_validate(scope.handle) == 7
    scope.check(admit(scope, buffer, [([2], [6])]))
    scope.check(planning.fort_scope_plan_validate(scope.handle))
    preview = select(scope, costs(), -1)
    assert preview.available
    assert preview.gpu_units == 1
    assert (preview.upload_bytes, preview.download_bytes) == (32, 32)
    assert not any(counters(scope))
    device = scope.gpu(buffer, effect)
    for i in range(2, 6):
        device[i] += 100
    scope.gpu_end(buffer)
    scope.cpu_begin(buffer, access(read=[([2], [6])]))
    scope.cpu_end(buffer)
    observed = scope.stats()
    assert (observed.upload_bytes, observed.download_bytes) == (preview.upload_bytes, preview.download_bytes)
    scope.close()
    assert list(host) == [0, 1, 102, 103, 104, 105, 6, 7]


@pytest.mark.native
def test_empty_original_domain_admits_no_storage(runtime):
    scope = Scope(runtime)
    status, buffer = scope.register(None, [0], initialized=False)
    scope.check(status)
    scope.check(admit(scope, buffer, []))
    scope.check(admit(scope, buffer, [([0], [0])]))
    assert not any(counters(scope))
    scope.close()
