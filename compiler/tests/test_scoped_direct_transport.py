"""DIRECT copies exact sparse sections without transient transport storage."""
# ruff: noqa: F811 - imported pytest fixtures.
import ctypes as c

import pytest

from compiler.tests.test_scoped_runtime import TOKEN, Scope, access, runtime  # noqa: F401
from compiler.tests.test_scoped_scratch import ScratchStats


@pytest.mark.native
@pytest.mark.parametrize('exact_budget', [False, True])
def test_sparse_faces_preserve_host_interior_and_readonly_mirrors(runtime, exact_budget):
    x, y, z = 134, 70, 70
    count = x * y * z
    scope = Scope(runtime)
    if exact_budget:
        scope.check(runtime.fort_scope_set_device_budget(scope.handle, count * 8))
    host = (c.c_double * count)(*([-7.0] * count))
    status, buffer = scope.register(host, [x, y, z], lower=[-10, 2, -100])
    scope.check(status)
    device = scope.gpu(buffer, access(flags=6))
    for i in range(count):
        device[i] = 1000000.0 + i
    scope.gpu_end(buffer)
    interior = [([3, 3, 2], [131, 67, 68])]
    scope.cpu_begin(buffer, access(write=interior, overwrite=interior))
    for k in range(2, 68):
        for j in range(3, 67):
            for i in range(3, 131):
                host[i + x * (j + y * k)] = -42.0
    scope.cpu_end(buffer)
    scope.cpu_begin(buffer, access(flags=1))
    scope.cpu_end(buffer)
    halo_bytes = (count - 128 * 64 * 66) * 8
    downloaded = scope.stats()
    assert (downloaded.downloads, downloaded.download_bytes) == (6, halo_bytes)
    for k in range(z):
        for j in range(y):
            for i in range(x):
                at = i + x * (j + y * k)
                interior_point = 3 <= i < 131 and 3 <= j < 67 and 2 <= k < 68
                assert host[at] == (-42.0 if interior_point else 1000000.0 + at)
    scope.gpu(buffer, access(flags=1))
    scope.gpu_end(buffer)
    mirrored = scope.stats()
    assert mirrored.uploads == 1
    assert mirrored.upload_bytes == 128 * 64 * 66 * 8
    for _ in range(20):
        scope.cpu_begin(buffer, access(flags=1))
        scope.cpu_end(buffer)
        scope.gpu(buffer, access(flags=1))
        scope.gpu_end(buffer)
    final = scope.stats()
    assert (final.uploads, final.downloads, final.allocations) == (mirrored.uploads, mirrored.downloads, 1)
    runtime.fort_scope_scratch_stats_get_v1.argtypes = [TOKEN, c.POINTER(ScratchStats)]
    scratch = ScratchStats()
    scope.check(runtime.fort_scope_scratch_stats_get_v1(scope.handle, c.byref(scratch)))
    assert scratch.capacity_bytes == scratch.allocations == scratch.reuses == 0
    scope.close()
