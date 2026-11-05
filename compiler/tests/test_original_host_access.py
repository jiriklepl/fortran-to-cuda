"""Original CPU may-accesses preserve tracked values without defining holes."""
import ctypes as c

import pytest

from compiler.tests.test_scoped_planning_runtime import Binding, costs, planning, record, select  # noqa: F401
from compiler.tests.test_scoped_runtime import SIZE, TOKEN, Scope, Section, access
from compiler.tests.test_scoped_runtime import runtime as shared_runtime  # noqa: F401


@pytest.fixture(scope="module")
def runtime(shared_runtime):  # noqa: F811 - imported pytest fixture
    shared_runtime.fort_scope_original_host_begin_v1.argtypes = [TOKEN, c.POINTER(Binding), SIZE]
    shared_runtime.fort_scope_admit_source_reads_v1.argtypes = [TOKEN, TOKEN, c.POINTER(Section), SIZE]
    return shared_runtime


def begin(scope, buffer, effects):
    scope.check(scope.lib.fort_scope_original_host_begin_v1(scope.handle, c.byref(Binding(buffer, effects)), 1))


def strict_read(scope, buffer, effects):
    pointer = c.c_void_p()
    return scope.lib.fort_scope_device_begin(scope.handle, buffer, c.byref(effects), c.byref(pointer))


@pytest.mark.native
def test_unknown_original_host_reads_do_not_authorize_numerical_uploads(runtime):
    scope = Scope(runtime)
    values = (c.c_double * 8)(*range(8))
    status, buffer = scope.register(values, [8], initialized=False)
    scope.check(status)
    begin(scope, buffer, access(flags=1))
    assert values[3] == 3
    scope.cpu_end(buffer)
    assert strict_read(scope, buffer, access(flags=1)) == 7
    stats = scope.stats()
    assert stats.uploads == stats.downloads == stats.allocations == 0
    scope.close()


@pytest.mark.native
@pytest.mark.parametrize("forget", [False, True])
def test_original_host_reads_publish_only_tracked_cells_and_keep_device_replica(runtime, forget):
    scope = Scope(runtime)
    host = (c.c_double * 8)(*([-99] * 8))
    status, buffer = scope.register(host, [8], initialized=False)
    scope.check(status)
    cells = [([2], [5])]
    device = scope.gpu(buffer, access(write=cells, overwrite=cells))
    for i in range(2, 5):
        device[i] = i * 7
    scope.gpu_end(buffer)
    if forget:
        scope.check(runtime.fort_scope_forget_definition(scope.handle, buffer))
    begin(scope, buffer, access(flags=1))
    assert list(host) == [-99, -99, 14, 21, 28, -99, -99, -99]
    scope.cpu_end(buffer)
    assert strict_read(scope, buffer, access(flags=1)) == 7
    if forget:
        assert strict_read(scope, buffer, access(read=cells)) == 7
        reads = access(read=cells)
        scope.check(runtime.fort_scope_admit_source_reads_v1(scope.handle, buffer, reads.reads, reads.read_count))
    again = scope.gpu(buffer, access(read=cells))
    assert [again[i] for i in range(2, 5)] == [14, 21, 28]
    scope.gpu_end(buffer)
    stats = scope.stats()
    assert stats.upload_bytes == 0
    assert stats.download_bytes == 24
    scope.close()


@pytest.mark.native
def test_possible_writes_preserve_unknown_holes_and_invalidate_device_values(runtime):
    scope = Scope(runtime)
    host = (c.c_double * 8)(*([-99] * 8))
    status, buffer = scope.register(host, [8], initialized=False)
    scope.check(status)
    known = [([2], [5])]
    device = scope.gpu(buffer, access(write=known, overwrite=known))
    for i in range(2, 5):
        device[i] = i
    scope.gpu_end(buffer)
    begin(scope, buffer, access(flags=2))
    host[3] += 10
    host[0] = 17
    scope.cpu_end(buffer)
    assert list(host) == [17, -99, 2, 13, 4, -99, -99, -99]
    assert strict_read(scope, buffer, access(read=[([0], [1])])) == 7
    again = scope.gpu(buffer, access(read=known))
    assert [again[i] for i in range(2, 5)] == [2, 13, 4]
    scope.gpu_end(buffer)
    stats = scope.stats()
    assert stats.upload_bytes == stats.download_bytes == 24
    scope.close()


@pytest.mark.native
def test_only_proved_overwrites_define_previously_unknown_values(runtime):
    scope = Scope(runtime)
    host = (c.c_double * 8)(*([-99] * 8))
    status, buffer = scope.register(host, [8], initialized=False)
    scope.check(status)
    cells = [([1], [3])]
    begin(scope, buffer, access(flags=2, overwrite=cells))
    host[1], host[2] = 7, 9
    scope.cpu_end(buffer)
    assert strict_read(scope, buffer, access(flags=1)) == 7
    device = scope.gpu(buffer, access(read=cells))
    assert [device[1], device[2]] == [7, 9]
    scope.gpu_end(buffer)
    stats = scope.stats()
    assert stats.upload_bytes == 16
    assert stats.download_bytes == 0
    scope.close()


@pytest.mark.native
def test_later_invalid_binding_cannot_publish_or_prepare_an_earlier_resource(runtime):
    scope = Scope(runtime)
    host = (c.c_double * 8)(*([-99] * 8))
    status, buffer = scope.register(host, [8], initialized=False)
    scope.check(status)
    cells = [([2], [4])]
    device = scope.gpu(buffer, access(write=cells, overwrite=cells))
    device[2], device[3] = 7, 9
    scope.gpu_end(buffer)
    bindings = (Binding * 2)(Binding(buffer, access(flags=1)), Binding(999999, access(flags=1)))
    assert runtime.fort_scope_original_host_begin_v1(scope.handle, bindings, 2) != 0
    assert list(host) == [-99] * 8
    assert scope.stats().download_bytes == 0
    begin(scope, buffer, access(flags=1))
    scope.cpu_end(buffer)
    assert list(host) == [-99, -99, 7, 9, -99, -99, -99, -99]
    scope.close()


@pytest.mark.native
@pytest.mark.parametrize("write", [False, True])
def test_original_host_plan_matches_actual_copies_without_initializing_holes(planning, write):  # noqa: F811
    scope = Scope(planning)
    host = (c.c_double * 8)(*([-99] * 8))
    status, buffer = scope.register(host, [8], initialized=False)
    scope.check(status)
    cells = [([2], [5])]
    device = scope.gpu(buffer, access(write=cells, overwrite=cells))
    for i in range(2, 5):
        device[i] = i * 7
    scope.gpu_end(buffer)
    before = scope.stats()
    scope.check(planning.fort_scope_wait(scope.handle))
    scope.check(planning.fort_scope_plan_reset(scope.handle))
    effects = access(flags=3 if write else 1)
    binding = Binding(buffer, effects)
    scope.check(planning.fort_scope_plan_add(scope.handle, 6, 0, c.byref(binding), 1, 0, 0, 0))
    record(scope, 42, buffer, access(read=cells))
    scope.check(planning.fort_scope_plan_validate(scope.handle))
    prediction = select(scope, costs(), -1)
    assert prediction.available
    assert prediction.gpu_units == 1
    assert prediction.download_bytes == 24
    assert prediction.upload_bytes == (24 if write else 0)
    assert scope.stats().download_bytes == before.download_bytes
    begin(scope, buffer, effects)
    if write:
        host[3] += 10
    scope.cpu_end(buffer)
    device = scope.gpu(buffer, access(read=cells))
    assert [device[i] for i in range(2, 5)] == [14, 31 if write else 21, 28]
    scope.gpu_end(buffer)
    after = scope.stats()
    for field in ("uploads", "downloads", "upload_bytes", "download_bytes"):
        assert getattr(after, field) - getattr(before, field) == getattr(prediction, field)
    assert strict_read(scope, buffer, access(flags=1)) == 7
    scope.close()


@pytest.mark.native
@pytest.mark.parametrize(("kind", "overwrite", "whole", "status"), [(0, False, False, 7), (6, False, False, 7),
                                                        (6, True, False, 0), (6, True, True, 7)])
def test_definition_preflight_keeps_strict_access_and_only_credits_must_overwrites(
        planning, kind, overwrite, whole, status):  # noqa: F811
    scope = Scope(planning)
    host = (c.c_double * 8)(*([-99] * 8))
    registered, buffer = scope.register(host, [8], initialized=False)
    scope.check(registered)
    cells = [([2], [5])]
    effects = access(flags=2, overwrite=cells if overwrite else ())
    binding = Binding(buffer, effects)
    scope.check(planning.fort_scope_plan_reset(scope.handle))
    scope.check(planning.fort_scope_plan_add(scope.handle, kind, 0, c.byref(binding), 1, 0, 0, 0))
    record(scope, 42, buffer, access(flags=1) if whole else access(read=cells))
    assert planning.fort_scope_plan_validate(scope.handle) == status
    assert scope.stats().allocations == scope.stats().uploads == scope.stats().downloads == 0
    scope.close()


@pytest.mark.native
def test_original_host_cannot_be_planned_as_device_work(planning):  # noqa: F811
    scope = Scope(planning)
    scope.check(planning.fort_scope_plan_reset(scope.handle))
    assert planning.fort_scope_plan_add(scope.handle, 6, 42, None, 0, 1e8, 0, 1) != 0
    scope.close()


@pytest.mark.native
def test_cancelled_original_host_write_cannot_publish_a_definition(runtime):
    scope = Scope(runtime)
    host = (c.c_double * 8)(*([-99] * 8))
    registered, buffer = scope.register(host, [8], initialized=False)
    scope.check(registered)
    cells = [([2], [5])]
    begin(scope, buffer, access(flags=2, overwrite=cells))
    scope.check(runtime.fort_scope_cancel_access(scope.handle, buffer))
    assert strict_read(scope, buffer, access(read=cells)) == 7
    begin(scope, buffer, access(flags=1))
    scope.cpu_end(buffer)
    assert list(host) == [-99] * 8
    assert scope.stats().uploads == scope.stats().downloads == 0
    scope.close()
