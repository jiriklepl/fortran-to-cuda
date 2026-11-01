"""Existing runtime primitives for a provider-only retained lifetime.

Host-runtime control only: this grants no compiler lifetime or GPU qualification.
"""
import ctypes as c

import pytest

from compiler.tests.test_scoped_domains import Domain, binding, execute
from compiler.tests.test_scoped_domains import domain_runtime as _domain_runtime
from compiler.tests.test_scoped_runtime import TOKEN, access

domain_runtime = _domain_runtime
pytestmark = pytest.mark.native


@pytest.mark.parametrize('replace_storage', [False, True])
def test_publication_retires_every_field_but_retains_explicit_workspace(domain_runtime, replace_storage):
    lib = domain_runtime
    lib.fort_scope_unregister.argtypes = [TOKEN, TOKEN]
    lib.fort_scope_unregister.restype = c.c_int
    domain = Domain(lib)
    first = domain.context()
    first.check(lib.fort_scope_set_device_budget(first.handle, 128))
    lease = TOKEN()
    first.check(lib.fort_scope_domain_workspace_reserve_v1(domain.handle, 64, c.byref(lease)))
    values = (c.c_double * 4)(10, 20, 30, 40)
    output = (c.c_double * 4)(-1, -2, -3, -4)
    original_values, original_output = values, output
    calls = []
    retired = []
    interior = [([1], [3])]

    def provider(_opaque, views, count, _stream, started):
        assert count == 2
        incoming = c.cast(views[0].device, c.POINTER(c.c_double))
        outgoing = c.cast(views[1].device, c.POINTER(c.c_double))
        calls.append((views[0].generation, incoming[1], incoming[2]))
        started[0] = 1
        for i in (1, 2):
            outgoing[i] = 2 * incoming[i]
        return 0

    for generation in (1, 2):
        scope = first if generation == 1 else domain.context()
        if generation == 2:
            pointer = c.c_void_p()
            for handle in retired:
                assert lib.fort_scope_device_begin(scope.handle, handle, c.byref(access(flags=1)), c.byref(pointer)) == 2
        status, source = scope.register(values, [4], identity=1, generation=generation, lower=[-2])
        scope.check(status)
        status, target = scope.register(output, [4], identity=2, generation=generation, initialized=False, lower=[-2])
        scope.check(status)
        before = list(output)
        status, report = execute(scope, [binding(source, access(read=interior)),
            binding(target, access(write=interior, overwrite=interior))], provider, lease=lease.value, workspace=64)
        scope.check(status)
        assert (report.version, report.started, report.committed) == (1, 1, 1)
        assert list(output) == before, 'a commit alone must not publish output'
        for handle in (source, target):
            scope.check(lib.fort_scope_unregister(scope.handle, handle))
        assert list(output) == [before[0], 2 * values[1], 2 * values[2], before[3]]
        assert scope.stats().allocated_bytes == 0
        retired.extend((source, target))
        scope.close()
        assert lib.fort_scope_domain_close_v1(domain.handle) == 8, 'workspace owner must retire its provider first'
        domain.check(lib.fort_scope_domain_workspace_resize_v1(domain.handle, lease, 64))
        if generation == 1:
            if replace_storage:
                values = (c.c_double * 4)(100, 200, 300, 400)
                output = (c.c_double * 4)(-100, -200, -300, -400)
            else:
                values[1], values[2] = 200, 300
                output[0], output[3] = -100, -400
    assert calls == [(1, 20, 30), (2, 200, 300)]
    assert list(output) == [-100, 400, 600, -400]
    if replace_storage:
        assert list(original_values) == [10, 20, 30, 40]
        assert list(original_output) == [-1, 40, 60, -4]
    stats = domain.stats()
    assert stats.upload_bytes == stats.download_bytes == 32
    assert stats.allocations == 4
    assert stats.allocated_bytes == 0
    domain.check(lib.fort_scope_domain_workspace_release_v1(domain.handle, lease))
    domain.close()
