"""Transport calibration follows storage through borrowed contexts and retirement."""
# ruff: noqa: F811 - imported maintained pytest fixture.
import ctypes as c

import pytest

from compiler.tests.test_scoped_compact_runtime import CopyCosts, compact_events, prepare, synthetic_copy_costs
from compiler.tests.test_scoped_domains import Domain, domain_runtime  # noqa: F401
from compiler.tests.test_scoped_planning_runtime import Costs, costs
from compiler.tests.test_scoped_runtime import TOKEN, access


def configure(scope):
    lib = scope.lib
    lib.fort_scope_set_transfers.argtypes = [TOKEN, c.c_uint32]
    lib.fort_scope_set_copy_costs_v3.argtypes = [TOKEN, c.POINTER(CopyCosts), c.POINTER(Costs), c.c_int]
    scope.check(lib.fort_scope_set_transfers(scope.handle, 3))
    copy, base = synthetic_copy_costs(), costs()
    scope.check(lib.fort_scope_set_copy_costs_v3(scope.handle, c.byref(copy), c.byref(base), 1))


def expected():
    return [1000 + i if i < 8 or i >= 56 else -42 for i in range(64)]


@pytest.mark.parametrize("publication", ["retirement", "host_read", "unregister"])
def test_copy_model_survives_detach_and_publishes_exact_regions(domain_runtime, monkeypatch, capfd, publication):
    monkeypatch.setenv("FORT_RUNTIME_TRACE", "1")
    domain = Domain(domain_runtime)
    first = domain.context()
    configure(first)
    values, handle = prepare(first)
    first.close()
    assert domain.stats().downloads == 0
    assert list(values) == [-77 if i < 8 or i >= 56 else -42 for i in range(64)]
    second = domain.context()  # Deliberately no second configuration call.
    if publication == "host_read":
        second.cpu_begin(handle, access(flags=1))
        second.cpu_end(handle)
        assert (second.stats().downloads, second.stats().download_bytes) == (1, 128)
        before = bytes(second.stats())
        for _ in range(3):
            second.gpu(handle, access(read=[([0], [8]), ([56], [64])]))
            second.gpu_end(handle)
            second.cpu_begin(handle, access(flags=1))
            second.cpu_end(handle)
        assert bytes(second.stats()) == before  # Read replicas survive both directions.
    elif publication == "unregister":
        domain_runtime.fort_scope_unregister.argtypes = [TOKEN, TOKEN]
        second.check(domain_runtime.fort_scope_unregister(second.handle, handle))
    second.close()
    domain.close()
    assert list(values) == expected()
    event, = compact_events(capfd)
    assert (event["bytes"], event["physical_operations"], event["dma_calls"]) == (128, 2, 1)
    assert event["domain_role"] == ("storage" if publication == "retirement" else "execution")


@pytest.mark.parametrize("replacement", ["incompatible", "missing", "invalid", "wrong_abi", "invalid_base"])
def test_later_child_clears_stale_transport_model(domain_runtime, monkeypatch, capfd, replacement):
    monkeypatch.setenv("FORT_RUNTIME_TRACE", "1")
    domain = Domain(domain_runtime)
    first = domain.context()
    configure(first)
    values, _ = prepare(first)
    first.close()
    second = domain.context()
    copy, base = synthetic_copy_costs(), costs()
    if replacement == "invalid":
        copy.valid = 0
    elif replacement == "wrong_abi":
        copy.version = 2
    elif replacement == "invalid_base":
        base.version = 0
    status = domain_runtime.fort_scope_set_copy_costs_v3(
        second.handle, None if replacement == "missing" else c.byref(copy), c.byref(base),
        int(replacement != "incompatible"))
    assert status == (1 if replacement == "wrong_abi" else 0)
    if replacement == "wrong_abi":
        assert b"expected version3" in domain_runtime.fort_scope_error()
    second.close()
    domain.close()
    assert list(values) == expected()
    assert not compact_events(capfd)


def test_transport_model_is_not_shared_between_domains(domain_runtime, monkeypatch, capfd):
    monkeypatch.setenv("FORT_RUNTIME_TRACE", "1")
    domains = [Domain(domain_runtime), Domain(domain_runtime)]
    allocations = []
    for index, domain in enumerate(domains):
        scope = domain.context()
        if index == 0:
            configure(scope)
        else:
            scope.check(domain_runtime.fort_scope_set_transfers(scope.handle, 3))
        values, _ = prepare(scope)
        allocations.append(values)
        scope.close()
    domains[1].close()
    assert list(allocations[1]) == expected()
    assert not compact_events(capfd)
    domains[0].close()
    assert list(allocations[0]) == expected()
    event, = compact_events(capfd)
    assert event["domain"] == domains[0].handle.value


def test_retirement_copy_failure_poisoning_preserves_no_replay_rule(domain_runtime, monkeypatch):
    domain = Domain(domain_runtime)
    scope = domain.context()
    configure(scope)
    values, _ = prepare(scope)
    scope.close()
    before = list(values)
    monkeypatch.setenv("FORT_SCOPE_TEST_FAIL_COMPACT_SUBMIT", "1")
    assert domain_runtime.fort_scope_domain_close_v1(domain.handle) == 6
    monkeypatch.delenv("FORT_SCOPE_TEST_FAIL_COMPACT_SUBMIT")
    assert domain_runtime.fort_scope_domain_close_v1(domain.handle) == 6
    assert list(values) == before
    domain.check(domain_runtime.fort_scope_domain_abandon_v1(domain.handle))
