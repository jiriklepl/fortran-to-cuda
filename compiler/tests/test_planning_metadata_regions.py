"""Metadata snapshots inspect exact current regions without executing work."""
# ruff: noqa: F811 - imported pytest fixtures.
import ctypes as c

import pytest

from compiler.tests.test_scoped_access_batch import pack
from compiler.tests.test_scoped_graph_queue import enqueue, queue, reserve, stats  # noqa: F401
from compiler.tests.test_scoped_graph_replay import graphs  # noqa: F401
from compiler.tests.test_scoped_runtime import SIZE, TOKEN, Scope, Section, access, runtime  # noqa: F401

pytestmark = pytest.mark.native


def probe(scope, handle, boxes):
    function = scope.lib.fort_scope_plan_host_sections_current_v1
    function.argtypes = [TOKEN, TOKEN, c.POINTER(Section), SIZE]
    reads = access(read=boxes)
    return function(scope.handle, handle, reads.reads, reads.read_count)


def whole(scope, handle):
    function = scope.lib.fort_scope_plan_host_current
    function.argtypes = [TOKEN, TOKEN]
    return function(scope.handle, handle)


def test_exact_known_element_does_not_require_or_define_other_metadata(runtime):
    scope = Scope(runtime)
    values = (c.c_double * 8)(*range(8))
    status, handle = scope.register(values, [8], lower=[-7], defined=[([3], [4])])
    scope.check(status)
    before = bytes(scope.stats())
    scope.check(probe(scope, handle, [([3], [4])]))
    scope.check(probe(scope, handle, []))
    assert whole(scope, handle) == 5
    assert probe(scope, handle, [([2], [4])]) == 5
    assert probe(scope, handle, [([0], [1]), ([3], [4])]) == 5
    assert runtime.fort_scope_host_begin(scope.handle, handle, c.byref(access(read=[([0], [1])]))) == 7
    assert bytes(scope.stats()) == before
    assert list(values) == list(range(8))
    scope.close()


def test_partial_device_write_invalidates_only_overlapping_metadata(runtime):
    scope = Scope(runtime)
    values = (c.c_double * 8)(*range(8))
    status, handle = scope.register(values, [8])
    scope.check(status)
    effects = access(write=[([3], [4])], overwrite=[([3], [4])])
    device = scope.gpu(handle, effects)
    device[3] = 99
    scope.gpu_end(handle)
    before = bytes(scope.stats())
    scope.check(probe(scope, handle, [([0], [3]), ([4], [8])]))
    assert probe(scope, handle, [([3], [4])]) == whole(scope, handle) == 5
    assert bytes(scope.stats()) == before
    assert values[3] == 3
    scope.cpu_begin(handle, access(read=[([3], [4])]))
    scope.cpu_end(handle)
    scope.check(probe(scope, handle, [([3], [4])]))
    assert values[3] == 99
    before = bytes(scope.stats())
    scope.gpu(handle, access(read=[([3], [4])]))
    scope.gpu_end(handle)
    assert scope.stats().uploads == 0
    assert scope.stats().downloads == 1
    scope.close()


def test_probe_rejects_bad_coordinates_prepared_access_and_stale_handles(runtime):
    scope = Scope(runtime)
    values = (c.c_double * 8)(*range(8))
    status, handle = scope.register(values, [8])
    scope.check(status)
    before = bytes(scope.stats())
    assert probe(scope, handle, [([0], [9])]) == 1
    assert probe(scope, handle, [([4], [3])]) == 1
    assert probe(scope, TOKEN(2**63), []) == 2
    assert bytes(scope.stats()) == before
    scope.cpu_begin(handle, access(read=[([3], [4])]))
    assert probe(scope, handle, [([3], [4])]) == 8
    scope.cpu_end(handle)
    scope.close()
    assert probe(scope, handle, [([3], [4])]) == 2


@pytest.mark.parametrize("write", [False, True])
def test_queued_metadata_dependencies_reject_only_writes_without_draining(queue, write):
    scope = Scope(queue)
    host, handle = reserve(scope, 1, initialized=True)
    effects = (access(write=[([3], [4])], overwrite=[([3], [4])]) if write
               else access(read=[([3], [4])]))
    scope.check(enqueue(scope, 51, pack((handle, effects))))
    before, queued = bytes(scope.stats()), bytes(stats(scope))
    scope.check(probe(scope, handle, [([0], [3]), ([4], [8])]))
    assert probe(scope, handle, [([3], [4])]) == (5 if write else 0)
    assert whole(scope, handle) == (5 if write else 0)
    assert bytes(scope.stats()) == before
    assert bytes(stats(scope)) == queued
    assert list(host) == list(range(8))
    scope.check(queue.fort_scope_graph_queue_cancel_v1(scope.handle))
    scope.check(whole(scope, handle))
    scope.close()
