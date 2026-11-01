"""Source-kernel accounting shares exact external transaction coherence."""
from __future__ import annotations

import ctypes as c

import pytest

from compiler.tests.test_scoped_domains import Call, Domain, Report, View, binding
from compiler.tests.test_scoped_domains import domain_runtime as _domain_runtime
from compiler.tests.test_scoped_runtime import SIZE, TOKEN, access

domain_runtime = _domain_runtime
pytestmark = pytest.mark.native
SourceWorker = c.CFUNCTYPE(c.c_int, c.c_void_p, c.POINTER(View), SIZE, c.c_void_p,
                           c.POINTER(c.c_int), c.POINTER(c.c_uint64))


def execute(scope, bindings, expected, worker):
    from compiler.tests.test_scoped_domains import Binding
    packed = (Binding * len(bindings))(*bindings)
    call, report = Call(1, 0, packed, len(bindings), 0, 0), Report()
    callback = SourceWorker(worker)
    function = scope.lib.fort_scope_source_execute_v1
    function.argtypes = [TOKEN, c.POINTER(Call), c.c_uint64, SourceWorker, c.c_void_p, c.POINTER(Report)]
    function.restype = c.c_int
    status = function(scope.handle, c.byref(call), expected, callback, None, c.byref(report))
    return status, report


def test_source_task_commits_same_region_engine_and_records_only_accepted_kernels(domain_runtime, capfd, monkeypatch):
    monkeypatch.setenv("FORT_RUNTIME_TRACE", "1")
    domain = Domain(domain_runtime)
    scope = domain.context()
    a, b = (c.c_double * 4)(1, 2, 3, 4), (c.c_double * 4)(10, 20, 30, 40)
    status, ah = scope.register(a, [4], lower=[-2])
    scope.check(status)
    status, bh = scope.register(b, [4], identity=2, lower=[-2])
    scope.check(status)

    def worker(_user, views, count, stream, started, accepted):
        assert count == 2
        assert stream is None  # CPU reference backend, identical typed views.
        av, bv = [c.cast(views[i].device, c.POINTER(c.c_double)) for i in range(2)]
        started[0] = 1
        for index in range(4):
            bv[index] += av[index]
        accepted[0] += 1
        for index in range(4):
            bv[index] *= 2
        accepted[0] += 1
        return 0

    status, report = execute(scope, [binding(ah, access(flags=1)), binding(bh, access(flags=3))], 2, worker)
    scope.check(status)
    assert (report.started, report.committed) == (1, 1)
    assert list(a) == [1, 2, 3, 4]
    assert list(b) == [10, 20, 30, 40]
    stats = scope.stats()
    assert (stats.launches, stats.upload_bytes, stats.download_bytes) == (2, 64, 0)
    # A host read of the immutable input uses its still-current host replica.
    scope.cpu_begin(ah, access(flags=1))
    scope.cpu_end(ah)
    assert scope.stats().download_bytes == 0
    scope.cpu_begin(bh, access(read=[([1], [2])]))
    scope.cpu_end(bh)
    assert b[1] == 44
    assert scope.stats().download_bytes == 8
    scope.close()
    domain.close()
    assert list(b) == [22, 44, 66, 88]
    launches = [line for line in capfd.readouterr().err.splitlines() if line.startswith("FORT_SCOPED launch ")]
    assert len(launches) == 2


@pytest.mark.parametrize("mode", ["rejected", "zero_expected", "empty"])
def test_source_prework_rejection_or_empty_task_does_not_publish_or_invent_launches(domain_runtime, mode):
    domain = Domain(domain_runtime)
    scope = domain.context()
    host = (c.c_double * 2)(7, 9)
    status, handle = scope.register(host, [2])
    scope.check(status)
    called = []

    def worker(_user, _views, _count, _stream, started, accepted):
        called.append(True)
        started[0], accepted[0] = 0, 0
        return 4

    bindings = [] if mode == "empty" else [binding(handle, access(flags=3))]
    expected = 1 if mode == "rejected" else 0
    status, report = execute(scope, bindings, expected, worker)
    assert status == {"rejected": 4, "zero_expected": 1, "empty": 0}[mode]
    assert called == ([True] if mode == "rejected" else [])
    assert (report.started, report.committed) == (0, 0)
    assert scope.stats().launches == 0
    scope.close()
    domain.close()
    assert list(host) == [7, 9]


@pytest.mark.parametrize("mode", ["prefix_failure", "under_count", "over_count", "contradiction", "commit_failure"])
def test_source_accepted_prefix_or_count_mismatch_poison_without_publication_or_replay(
        domain_runtime, capfd, monkeypatch, mode):
    monkeypatch.setenv("FORT_RUNTIME_TRACE", "1")
    if mode == "commit_failure":
        monkeypatch.setenv("FORT_SCOPE_TEST_FAIL_EXTERNAL_COMMIT", "1")
    domain = Domain(domain_runtime)
    scope = domain.context()
    host = (c.c_double * 2)(7, 9)
    status, handle = scope.register(host, [2])
    scope.check(status)
    calls = []

    def worker(_user, views, _count, _stream, started, accepted):
        calls.append(True)
        started[0] = 0 if mode == "contradiction" else 1
        c.cast(views[0].device, c.POINTER(c.c_double))[0] = 1000
        accepted[0] = 3 if mode == "over_count" else 1
        return 4 if mode in {"prefix_failure", "contradiction"} else 0

    expected = 1 if mode == "commit_failure" else 2
    status, report = execute(scope, [binding(handle, access(flags=3))], expected, worker)
    assert status == report.status == 6
    assert (report.started, report.committed) == (1, 0)
    assert domain_runtime.fort_scope_wait(scope.handle) == 6
    status, _report = execute(scope, [binding(handle, access(flags=3))], expected, worker)
    assert status == 6
    assert len(calls) == 1
    scope.check(domain_runtime.fort_scope_abandon(scope.handle))
    domain.check(domain_runtime.fort_scope_domain_abandon_v1(domain.handle))
    assert list(host) == [7, 9]
    launches = [line for line in capfd.readouterr().err.splitlines() if line.startswith("FORT_SCOPED launch ")]
    assert len(launches) == (0 if mode == "over_count" else 1)
