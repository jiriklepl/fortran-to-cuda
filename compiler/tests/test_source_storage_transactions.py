"""Original device representations share coherence without defining holes."""
import ctypes as c

import pytest

from compiler.tests.test_original_host_access import begin, strict_read
from compiler.tests.test_scoped_domains import Binding, Call, Domain, Report, binding, layout
from compiler.tests.test_scoped_domains import domain_runtime as _domain_runtime
from compiler.tests.test_scoped_domains import execute as external_execute
from compiler.tests.test_scoped_planning_runtime import Binding as PlanBinding
from compiler.tests.test_scoped_runtime import SIZE, TOKEN, access
from compiler.tests.test_scoped_source_tasks import SourceWorker
from compiler.tests.test_scoped_source_tasks import execute as strict_execute

pytestmark = pytest.mark.native


@pytest.fixture(scope="module")
def domain_runtime(tmp_path_factory):
    library = _domain_runtime.__wrapped__(tmp_path_factory)
    library.fort_scope_original_host_begin_v1.argtypes = [TOKEN, c.POINTER(PlanBinding), SIZE]
    library.fort_scope_forget_definition.argtypes = [TOKEN, TOKEN]
    library.fort_scope_unregister.argtypes = [TOKEN, TOKEN]
    library.fort_scope_observe_v1.argtypes = [TOKEN, c.POINTER(TOKEN), SIZE]
    library.fort_scope_observe_all_v1.argtypes = [TOKEN]
    return library


def execute(scope, bindings, worker, *, expected=1):
    packed = (Binding * len(bindings))(*bindings)
    call, report = Call(1, 0, packed, len(bindings), 0, 0), Report()
    callback = SourceWorker(worker)
    function = scope.lib.fort_scope_source_execute_v2
    function.argtypes = [TOKEN, c.POINTER(Call), c.c_uint64, SourceWorker, c.c_void_p, c.POINTER(Report)]
    function.restype = c.c_int
    status = function(scope.handle, c.byref(call), expected, callback, None, c.byref(report))
    return status, report


def context(library, n=8):
    domain = Domain(library)
    scope = domain.context()
    values = (c.c_double * n)(*range(n))
    status, handle = scope.register(values, [n], lower=[-3], initialized=False)
    scope.check(status)
    return domain, scope, values, handle


def worker(operation):
    def submit(_user, views, count, _stream, started, accepted):
        assert count == 1
        started[0] = 1
        operation(c.cast(views[0].device, c.POINTER(c.c_double)))
        accepted[0] = 1
        return 0
    return submit


def assert_undefined(scope, handle):
    assert strict_read(scope, handle, access(flags=1)) == 7
    called = []
    def strict_worker(*_):
        called.append(True)
        return 0
    assert strict_execute(scope, [binding(handle, access(flags=1))], 1, strict_worker)[0] == 7
    assert external_execute(scope, [binding(handle, access(flags=1))], strict_worker)[0] == 7
    assert not called


@pytest.mark.parametrize("write_host", [False, True])
def test_cpu_gpu_cpu_gpu_preserves_replicas_and_definition_holes(domain_runtime, write_host):
    domain, scope, values, handle = context(domain_runtime)
    seen = []
    readonly = binding(handle, access(flags=1))
    for repeat in range(3):
        scope.check(execute(scope, [readonly], worker(lambda v: seen.append([v[i] for i in range(8)])))[0])
        assert_undefined(scope, handle)
        begin(scope, handle, access(read=[([2], [3])], write=[([2], [3])] if write_host else ()))
        assert values[2] == 2 + (repeat if write_host else 0)
        if write_host:
            values[2] += 1
        scope.cpu_end(handle)
    assert [v[2] for v in seen] == ([2, 3, 4] if write_host else [2, 2, 2])
    stats = scope.stats()
    assert stats.upload_bytes == 64 + (16 if write_host else 0)
    assert stats.download_bytes == 0
    assert stats.launches == 3
    scope.close()
    domain.close()


@pytest.mark.parametrize("forget", [False, True])
def test_newer_device_bytes_survive_definition_kill_and_partial_may_writes(domain_runtime, forget):
    domain, scope, values, handle = context(domain_runtime)
    known = [([2], [4])]
    gpu = scope.gpu(handle, access(write=known, overwrite=known))
    gpu[2], gpu[3] = 20, 30
    scope.gpu_end(handle)
    if forget:
        scope.check(scope.lib.fort_scope_forget_definition(scope.handle, handle))
    def modify(v):
        assert [v[i] for i in range(8)] == [0, 1, 20, 30, 4, 5, 6, 7]
        v[3] += 100
    scope.check(execute(scope, [binding(handle, access(flags=1, write=[([1], [7])]))], worker(modify))[0])
    assert scope.stats().upload_bytes == 48
    assert list(values) == list(range(8))
    assert_undefined(scope, handle)
    if forget:
        assert strict_read(scope, handle, access(read=known)) == 7
    begin(scope, handle, access(flags=1))
    scope.cpu_end(handle)
    assert list(values) == [0, 1, 20, 130, 4, 5, 6, 7]
    assert scope.stats().download_bytes == 48
    before = scope.stats().upload_bytes
    scope.check(execute(scope, [binding(handle, access(flags=1))], worker(lambda v: None))[0])
    assert scope.stats().upload_bytes == before
    scope.close()
    domain.close()


def test_only_exact_overwrites_gain_definitions_without_upload(domain_runtime):
    domain, scope, values, handle = context(domain_runtime)
    cells = [([2], [4])]
    def replace(v):
        v[2], v[3] = 20, 30
    scope.check(execute(scope, [binding(handle, access(write=cells, overwrite=cells))], worker(replace))[0])
    assert scope.stats().upload_bytes == 0
    assert_undefined(scope, handle)
    scope.cpu_begin(handle, access(read=cells))
    scope.cpu_end(handle)
    assert list(values) == [0, 1, 20, 30, 4, 5, 6, 7]
    assert scope.stats().download_bytes == 16
    scope.close()
    domain.close()


@pytest.mark.parametrize("all_resources", [False, True])
@pytest.mark.parametrize("source_storage", [False, True])
def test_entry_observation_publishes_physical_versions_without_definition_authority(
        domain_runtime, all_resources, source_storage):
    domain, scope, values, handle = context(domain_runtime)
    cells = [([2], [4])]
    if source_storage:
        def modify(v):
            v[3] = 123
        scope.check(execute(scope, [binding(handle, access(read=cells, write=cells))], worker(modify))[0])
    else:
        device = scope.gpu(handle, access(write=cells, overwrite=cells))
        device[2], device[3] = 2, 123
        scope.gpu_end(handle)
        scope.check(scope.lib.fort_scope_forget_definition(scope.handle, handle))
    uploads = scope.stats().upload_bytes
    for _ in range(2):
        if all_resources:
            scope.check(scope.lib.fort_scope_observe_all_v1(scope.handle))
        else:
            scope.check(scope.lib.fort_scope_observe_v1(scope.handle, c.byref(handle), 1))
        assert list(values) == [0, 1, 2, 123, 4, 5, 6, 7]
        assert scope.stats().download_bytes == 16
        assert_undefined(scope, handle)
    scope.check(execute(scope, [binding(handle, access(read=cells))], worker(lambda v: None))[0])
    assert scope.stats().upload_bytes == uploads
    scope.close()
    domain.close()


@pytest.mark.parametrize("finish", ["close", "evict", "generation", "unregister"])
def test_unknown_modified_representations_are_published_at_lifetime_boundaries(domain_runtime, finish):
    domain, scope, values, handle = context(domain_runtime)
    if finish == "evict":
        scope.check(scope.lib.fort_scope_set_device_budget(scope.handle, 64))
    def modify(v):
        v[3] = 123
    scope.check(execute(scope, [binding(handle, access(flags=3))], worker(modify))[0])
    assert_undefined(scope, handle)
    if finish == "evict":
        other = (c.c_double * 8)(*range(8))
        status, second = scope.register(other, [8], identity=2)
        scope.check(status)
        scope.gpu(second, access(flags=1))
        scope.gpu_end(second)
    elif finish == "generation":
        assert scope.register(values, [8], generation=2, initialized=False)[0] == 2
        scope.check(scope.lib.fort_scope_unregister(scope.handle, handle))
        status, second = scope.register(values, [8], generation=2, initialized=False)
        scope.check(status)
        assert second.value != handle.value
        assert strict_read(scope, second, access(flags=1)) == 7
    elif finish == "unregister":
        scope.check(scope.lib.fort_scope_unregister(scope.handle, handle))
    scope.close()
    domain.close()
    assert list(values) == [0, 1, 2, 123, 4, 5, 6, 7]


def test_descriptor_exchange_keeps_unknown_physical_versions_and_borrowed_lifetime(domain_runtime):
    domain, scope, values, first = context(domain_runtime)
    other = (c.c_double * 4)(10, 20, 30, 40)
    status, second = scope.register(other, [4], identity=2, generation=3, lower=[-7], initialized=False)
    scope.check(status)
    def modify(v):
        v[2] += 100
    for handle in (first, second):
        scope.check(execute(scope, [binding(handle, access(flags=3))], worker(modify))[0])
    scope.close()
    assert (values[2], other[2]) == (2, 30)
    continued = domain.context()
    before = bytes(continued.stats())
    after_first, after_second = layout(other, [4], 3, [-7]), layout(values, [8], 1, [-3])
    new_first, new_second = TOKEN(), TOKEN()
    continued.check(domain_runtime.fort_scope_identity_swap_v1(
        continued.handle, first, second, c.byref(after_first), c.byref(after_second),
        c.byref(new_first), c.byref(new_second)))
    assert (new_first.value, new_second.value) == (second.value, first.value)
    assert bytes(continued.stats()) == before
    seen = []
    for handle in (new_first, new_second):
        continued.check(execute(continued, [binding(handle, access(flags=1))], worker(lambda v: seen.append(v[2])))[0])
        assert_undefined(continued, handle)
    assert seen == [130, 102]
    assert (continued.stats().upload_bytes, continued.stats().download_bytes) == (96, 0)
    continued.close()
    domain.close()
    assert (values[2], other[2]) == (102, 130)


def test_rank_two_overlapping_boxes_preserve_exact_physical_offsets(domain_runtime):
    domain = Domain(domain_runtime)
    scope = domain.context()
    values = (c.c_double * 20)(*range(20))
    status, handle = scope.register(values, [4, 5], lower=[-2, 3], initialized=False)
    scope.check(status)
    selected = [([1, 1], [3, 4])]
    def modify(v):
        for j in range(1, 4):
            for i in range(1, 3):
                v[i+4*j] += 100
    scope.check(execute(scope, [Binding(handle, 2, 2, 8, access(read=selected, write=selected))], worker(modify))[0])
    assert scope.stats().upload_bytes == 48
    begin(scope, handle, access(read=[([2, 2], [4, 5])]))
    scope.cpu_end(handle)
    assert scope.stats().download_bytes == 16
    assert (values[10], values[14]) == (110, 114)
    assert (values[9], values[13]) == (9, 13)
    scope.close()
    domain.close()
    expected = [i + (100 if i%4 in (1, 2) and i//4 in (1, 2, 3) else 0) for i in range(20)]
    assert list(values) == expected


@pytest.mark.parametrize("mode", ["empty", "rejected", "bad_binding", "alias", "bad_overwrite"])
def test_prework_failure_cannot_commit_or_evaluate_a_payload(domain_runtime, mode):
    domain, scope, values, handle = context(domain_runtime)
    calls = []
    def reject(_user, _views, _count, _stream, started, accepted):
        calls.append(True)
        started[0], accepted[0] = 0, 0
        return 4
    effects = access(flags=3)
    bindings = [binding(handle, effects)]
    if mode == "empty":
        bindings = []
    elif mode == "bad_binding":
        bindings.append(binding(TOKEN(999999), effects))
    elif mode == "alias":
        bindings.append(binding(handle, effects))
    elif mode == "bad_overwrite":
        bindings = [binding(handle, access(overwrite=[([0], [1])]))]
    status, report = execute(scope, bindings, reject, expected=0 if mode == "empty" else 1)
    assert status == {"empty": 0, "rejected": 4, "alias": 3, "bad_binding": 2, "bad_overwrite": 1}[mode]
    assert calls == ([True] if mode == "rejected" else [])
    assert (report.started, report.committed) == (0, 0)
    assert_undefined(scope, handle)
    assert scope.stats().launches == scope.stats().download_bytes == 0
    assert scope.stats().upload_bytes == (64 if mode == "rejected" else 0)
    scope.close()
    domain.close()
    assert list(values) == list(range(8))


@pytest.mark.parametrize("mode", ["prefix", "undercount", "overcount", "commit"])
def test_post_submission_failure_poison_is_identical_to_strict_transactions(domain_runtime, monkeypatch, mode):
    domain, scope, values, handle = context(domain_runtime)
    if mode == "commit":
        monkeypatch.setenv("FORT_SCOPE_TEST_FAIL_EXTERNAL_COMMIT", "1")
    calls = []
    def fail(_user, views, _count, _stream, started, accepted):
        calls.append(True)
        started[0] = 1
        c.cast(views[0].device, c.POINTER(c.c_double))[3] = 999
        accepted[0] = 2 if mode == "overcount" else 1
        return 4 if mode == "prefix" else 0
    status, report = execute(scope, [binding(handle, access(flags=3))], fail, expected=2 if mode == "undercount" else 1)
    assert status == report.status == 6
    assert (report.started, report.committed) == (1, 0)
    assert scope.lib.fort_scope_wait(scope.handle) == 6
    assert execute(scope, [binding(handle, access(flags=3))], fail)[0] == 6
    assert calls == [True]
    scope.check(scope.lib.fort_scope_abandon(scope.handle))
    domain.check(scope.lib.fort_scope_domain_abandon_v1(domain.handle))
    assert list(values) == list(range(8))
