"""Original CPU may-accesses preserve tracked values without defining holes."""
import ctypes as c

import pytest

from compiler.tests.test_scoped_runtime import SIZE, TOKEN, Binding, Scope, Section, access
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
