"""Original CPU teams use the existing exact host replica transaction."""

import ctypes as c

import pytest

from compiler.tests.test_scoped_domains import Binding, Call, Domain, binding
from compiler.tests.test_scoped_domains import domain_runtime as _domain_runtime
from compiler.tests.test_scoped_runtime import TOKEN, access

domain_runtime = _domain_runtime
pytestmark = pytest.mark.native


def prepare(scope, bindings):
    packed = (Binding * len(bindings))(*bindings)
    call, token = Call(1, 0, packed, len(bindings), 0, 0), TOKEN()
    function = scope.lib.fort_scope_source_host_prepare_v1
    function.argtypes = [TOKEN, c.POINTER(Call), c.POINTER(TOKEN)]
    function.restype = c.c_int
    for suffix in ("start", "complete", "cancel"):
        f = getattr(scope.lib, "fort_scope_source_host_" + suffix + "_v1")
        f.argtypes, f.restype = [TOKEN, TOKEN], c.c_int
    return function(scope.handle, c.byref(call), c.byref(token)), token


def test_readonly_cpu_work_retains_live_domain_and_device_replica(domain_runtime):
    domain = Domain(domain_runtime)
    scope = domain.context()
    host = (c.c_double * 8)(*range(8))
    status, handle = scope.register(host, [8], lower=[-3])
    scope.check(status)
    device = scope.gpu(handle, access(flags=3))
    for i in range(8):
        device[i] += 10
    scope.gpu_end(handle)
    scope.check(domain_runtime.fort_scope_wait(scope.handle))
    status, token = prepare(scope, [binding(handle, access(flags=1))])
    scope.check(status)
    assert list(host) == list(range(10, 18))
    assert scope.stats().download_bytes == 64
    scope.check(domain_runtime.fort_scope_source_host_start_v1(scope.handle, token))
    original_observation = sum(host)
    scope.check(domain_runtime.fort_scope_source_host_complete_v1(scope.handle, token))
    assert original_observation == sum(range(10, 18))
    before = scope.stats()
    again = scope.gpu(handle, access(flags=1))
    assert [again[i] for i in range(8)] == list(range(10, 18))
    scope.gpu_end(handle)
    assert scope.stats().upload_bytes == before.upload_bytes
    assert scope.stats().allocations == before.allocations
    scope.close()
    domain.close()


def test_partial_cpu_write_invalidates_only_declared_device_cells(domain_runtime):
    domain = Domain(domain_runtime)
    scope = domain.context()
    host = (c.c_double * 8)(*range(8))
    status, handle = scope.register(host, [8])
    scope.check(status)
    device = scope.gpu(handle, access(flags=3))
    for i in range(8):
        device[i] += 10
    scope.gpu_end(handle)
    box = [([2], [4])]
    status, token = prepare(scope, [binding(handle, access(read=box, write=box))])
    scope.check(status)
    assert scope.stats().download_bytes == 16
    scope.check(domain_runtime.fort_scope_source_host_start_v1(scope.handle, token))
    for i in range(2, 4):
        host[i] *= 3
    scope.check(domain_runtime.fort_scope_source_host_complete_v1(scope.handle, token))
    before = scope.stats().upload_bytes
    device = scope.gpu(handle, access(flags=1))
    assert [device[i] for i in range(8)] == [10, 11, 36, 39, 14, 15, 16, 17]
    scope.gpu_end(handle)
    assert scope.stats().upload_bytes == before + 16
    scope.close()
    domain.close()


@pytest.mark.parametrize("fault", ["undefined", "type", "duplicate"])
def test_all_preflight_rejections_precede_publication_or_prepared_effects(domain_runtime, fault):
    domain = Domain(domain_runtime)
    scope = domain.context()
    a, b = (c.c_double * 4)(1, 2, 3, 4), (c.c_double * 4)(5, 6, 7, 8)
    status, ha = scope.register(a, [4])
    scope.check(status)
    status, hb = scope.register(b, [4], identity=2, initialized=fault != "undefined")
    scope.check(status)
    pointer = scope.gpu(ha, access(flags=3))
    pointer[0] = 100
    scope.gpu_end(ha)
    bindings = [binding(ha, access(flags=1)), binding(hb, access(flags=1))]
    if fault == "type":
        bindings[-1].element_bytes = 4
    elif fault == "duplicate":
        bindings[-1] = bindings[0]
    status, token = prepare(scope, bindings)
    assert status == {"undefined": 7, "type": 1, "duplicate": 3}[fault]
    assert token.value == 0
    assert scope.stats().download_bytes == 0
    assert a[0] == 1
    # No earlier resource was left prepared by the rejected transaction.
    scope.cpu_begin(ha, access(flags=1))
    scope.cpu_end(ha)
    assert a[0] == 100
    scope.close()
    domain.close()


def test_cancel_before_start_and_context_owned_tokens(domain_runtime):
    first, second = Domain(domain_runtime), Domain(domain_runtime)
    a, b = first.context(), second.context()
    host = (c.c_double * 2)(4, 7)
    status, handle = a.register(host, [2])
    a.check(status)
    status, token = prepare(a, [binding(handle, access(flags=3))])
    a.check(status)
    assert domain_runtime.fort_scope_source_host_start_v1(b.handle, token) == 2
    a.check(domain_runtime.fort_scope_source_host_cancel_v1(a.handle, token))
    assert domain_runtime.fort_scope_source_host_start_v1(a.handle, token) == 2
    a.close()
    b.close()
    first.close()
    second.close()
    assert list(host) == [4, 7]


@pytest.mark.parametrize("misuse", ["cancel", "repeat_start", "mutation", "wrong_complete"])
def test_every_failure_after_original_cpu_start_is_fatal_and_never_replayed(domain_runtime, misuse):
    domain = Domain(domain_runtime)
    scope = domain.context()
    host = (c.c_double * 2)(4, 7)
    status, handle = scope.register(host, [2])
    scope.check(status)
    status, token = prepare(scope, [binding(handle, access(flags=3))])
    scope.check(status)
    scope.check(domain_runtime.fort_scope_source_host_start_v1(scope.handle, token))
    host[0] += 1  # The original CPU span executes exactly once.
    if misuse == "cancel":
        status = domain_runtime.fort_scope_source_host_cancel_v1(scope.handle, token)
    elif misuse == "repeat_start":
        status = domain_runtime.fort_scope_source_host_start_v1(scope.handle, token)
    elif misuse == "mutation":
        status = domain_runtime.fort_scope_host_end(scope.handle, handle)
    else:
        status = domain_runtime.fort_scope_source_host_complete_v1(scope.handle, token.value + 1)
    assert status == 6
    assert domain_runtime.fort_scope_source_host_complete_v1(scope.handle, token) == 6
    assert domain_runtime.fort_scope_close(scope.handle) == 6
    scope.check(domain_runtime.fort_scope_abandon(scope.handle))
    domain.check(domain_runtime.fort_scope_domain_abandon_v1(domain.handle))
    assert list(host) == [5, 7]
