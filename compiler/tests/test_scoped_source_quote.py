"""Live source quotes reuse coherence without executing or installing work."""

from __future__ import annotations

import ctypes as c
import math

import pytest

from compiler.tests.test_scoped_compute_costs import Compute
from compiler.tests.test_scoped_domains import Binding, Call, Domain, binding
from compiler.tests.test_scoped_domains import domain_runtime as _domain_runtime
from compiler.tests.test_scoped_planning_runtime import Costs, Decision, costs
from compiler.tests.test_scoped_runtime import SIZE, TOKEN, access
from compiler.tests.test_scoped_scratch import Lease

domain_runtime = _domain_runtime
pytestmark = pytest.mark.native


class Protocol(c.Structure):
    _fields_ = [(name, c.c_uint32) for name in ("version", "valid", "precision_bits", "reserved")] + [
        (name, TOKEN) for name in ("protocol_id", "max_bindings", "max_launches")
    ] + [("max_scratch_bytes", SIZE)] + [(name, c.c_double) for name in (
        "source_submit_seconds", "scratch_cold_acquire_seconds", "scratch_reuse_acquire_seconds",
        "scratch_idle_grow_seconds", "scratch_release_seconds")]


class Work(c.Structure):
    _fields_ = [(name, c.c_uint32) for name in ("version", "precision_bits", "flags", "reserved")] + [
        (name, TOKEN) for name in ("expected_launches", "real_work_proxy", "memory_bytes")
    ] + [("scratch_bytes", SIZE), ("minimum_relative_saving", c.c_double), ("compute", Compute)]


class Terminal(c.Structure):
    _fields_ = [("seconds", c.c_double)] + [
        (name, TOKEN) for name in ("download_bytes", "downloads", "waits", "releases")]


class Quote(c.Structure):
    _fields_ = [(name, c.c_uint32) for name in (
        "version", "available", "gpu_preferred", "reason", "native_endpoint", "gpu_endpoint")
    ] + [(name, c.c_double) for name in (
        "native_seconds", "gpu_seconds", "gpu_execution_seconds", "transfer_dispersion_debit_seconds")
    ] + [("gpu", Decision), ("native_terminal", Terminal), ("gpu_terminal", Terminal)] + [
        (name, TOKEN) for name in ("scratch_capacity_before", "scratch_capacity_after", "peak_total_device_bytes")]


def protocol():
    return Protocol(1, 1, 64, 0, 0x4654534F55520001, 8, 256, 1 << 26,
                    .001, .002, .003, .004, .005)


def work():
    return Work(1, 64, 2, 0, 7, 1000, 128, 0, .05, Compute(2, 1 | 4, 1., 0., .01))


@pytest.fixture
def quote_api(domain_runtime):
    domain_runtime.fort_scope_source_quote_v1.argtypes = [TOKEN, c.POINTER(Call), c.POINTER(Work),
        c.POINTER(Costs), c.POINTER(Protocol), c.c_int, c.POINTER(Quote)]
    domain_runtime.fort_scope_source_quote_v1.restype = c.c_int
    domain_runtime.fort_scope_scratch_acquire_v1.argtypes = [TOKEN, SIZE, c.POINTER(Lease)]
    domain_runtime.fort_scope_scratch_release_v1.argtypes = [TOKEN, TOKEN]
    domain_runtime.fort_scope_task_graph_enable_v1.argtypes = [TOKEN]
    domain_runtime.fort_scope_device_reserve_many_v1.argtypes = [TOKEN, c.POINTER(TOKEN), SIZE,
                                                              c.POINTER(c.c_void_p)]
    return domain_runtime


def quote(scope, bindings, *, request=None, rates=None, base=None, compatible=1):
    packed = (Binding * len(bindings))(*bindings)
    call = Call(1, 0, packed, len(bindings), 0, 0)
    result = Quote()
    status = scope.lib.fort_scope_source_quote_v1(scope.handle, c.byref(call),
        c.byref(request or work()), c.byref(base or costs()), c.byref(rates or protocol()),
        compatible, c.byref(result))
    scope.check(status)
    return result


def registered(domain, *, count=8, initialized=True):
    scope = domain.context()
    host = (c.c_double * count)(*range(count))
    status, handle = scope.register(host, [count], initialized=initialized)
    scope.check(status)
    return scope, host, handle


def test_fresh_quote_separates_local_and_terminal_and_never_initializes(quote_api):
    domain = Domain(quote_api)
    scope, host, handle = registered(domain)
    before = bytes(scope.stats())
    result = quote(scope, [binding(handle, access(flags=3))])
    assert result.available
    assert result.gpu_preferred
    assert (result.native_endpoint, result.gpu_endpoint) == (2, 2)
    assert result.native_seconds == 1.
    assert (result.gpu.launches, result.gpu.upload_bytes, result.gpu.download_bytes) == (7, 64, 0)
    assert (result.gpu_terminal.download_bytes, result.gpu_terminal.releases) == (64, 1)
    assert result.gpu_seconds == pytest.approx(result.gpu_execution_seconds + result.gpu_terminal.seconds)
    assert bytes(scope.stats()) == before
    assert list(host) == list(range(8))
    # Budget can still be changed: not even lazy device initialization occurred.
    scope.check(quote_api.fort_scope_set_device_budget(scope.handle, 64))
    scope.close()
    domain.close()


def test_submit_replaces_old_enqueue_and_field_preparation_exactly_once(quote_api):
    domain = Domain(quote_api)
    scope, _host, handle = registered(domain)
    bindings = [binding(handle, access(flags=1))]
    before = quote(scope, bindings)
    old = costs()
    old.device_access_seconds, old.launch_enqueue_seconds = 100., 1000.
    after = quote(scope, bindings, base=old)
    assert after.gpu_seconds == before.gpu_seconds
    rates = protocol()
    rates.source_submit_seconds += .25
    assert quote(scope, bindings, rates=rates).gpu_seconds - before.gpu_seconds == pytest.approx(.25)
    scope.close()
    domain.close()


def test_live_dirty_replica_and_partial_host_mirror_price_exact_publication(quote_api):
    domain = Domain(quote_api)
    scope, host, handle = registered(domain)
    data = scope.gpu(handle, access(flags=3))
    for i in range(8):
        data[i] += 100
    scope.gpu_end(handle)
    scope.cpu_begin(handle, access(read=[([2], [3])]))
    scope.cpu_end(handle)
    before = bytes(scope.stats())
    result = quote(scope, [binding(handle, access(flags=1))])
    assert result.available
    assert result.gpu.upload_bytes == result.gpu.download_bytes == 0
    assert result.native_terminal.download_bytes == result.gpu_terminal.download_bytes == 56
    assert result.native_seconds == pytest.approx(1. + result.native_terminal.seconds)
    assert list(host) == [0, 1, 102, 3, 4, 5, 6, 7]
    assert bytes(scope.stats()) == before
    scope.close()
    domain.close()
    assert list(host) == [i + 100 for i in range(8)]


def test_clean_replica_evicted_before_older_dirty_replica_in_quote_and_execution(quote_api):
    domain = Domain(quote_api)
    scope, dirty_host, dirty = registered(domain)
    scope.check(quote_api.fort_scope_task_graph_enable_v1(scope.handle))
    scope.check(quote_api.fort_scope_set_device_budget(scope.handle, 128))
    clean_host, new_host = (c.c_double * 8)(*range(8)), (c.c_double * 8)(*range(8))
    status, clean = scope.register(clean_host, [8], identity=2)
    scope.check(status)
    status, new = scope.register(new_host, [8], identity=3)
    scope.check(status)
    data = scope.gpu(dirty, access(flags=3))
    data[0] = 100
    scope.gpu_end(dirty)
    scope.gpu(clean, access(flags=1))
    scope.gpu_end(clean)
    result = quote(scope, [binding(new, access(flags=1))])
    assert result.available
    assert (result.gpu.upload_bytes, result.gpu.download_bytes) == (64, 0)
    assert result.gpu_terminal.download_bytes == 64
    requested, pointers = (TOKEN * 1)(new.value), (c.c_void_p * 1)()
    scope.check(quote_api.fort_scope_device_reserve_many_v1(scope.handle, requested, 1, pointers))
    assert scope.stats().download_bytes == 0
    assert dirty_host[0] == 0
    # The dirty allocation survived pressure, and needs no reupload.
    before = scope.stats().upload_bytes
    assert scope.gpu(dirty, access(flags=1))[0] == 100
    scope.gpu_end(dirty)
    assert scope.stats().upload_bytes == before
    required = [binding(clean, access(flags=1)), binding(new, access(flags=1))]
    dirty_eviction = quote(scope, required)
    assert dirty_eviction.available
    assert dirty_eviction.gpu.download_bytes == 64
    assert dirty_eviction.gpu_terminal.download_bytes == 0
    requested, pointers = (TOKEN * 2)(clean.value, new.value), (c.c_void_p * 2)()
    scope.check(quote_api.fort_scope_device_reserve_many_v1(scope.handle, requested, 2, pointers))
    assert scope.stats().download_bytes == 64
    assert dirty_host[0] == 100
    scope.close()
    domain.close()
    assert dirty_host[0] == 100


def test_scratch_acquire_reuse_growth_and_terminal_budget_are_distinct(quote_api):
    domain = Domain(quote_api)
    scope, _host, handle = registered(domain)
    request = work()
    request.flags |= 1
    request.scratch_bytes = 64
    result = quote(scope, [binding(handle, access(flags=1))], request=request)
    assert result.available
    assert (result.scratch_capacity_before, result.scratch_capacity_after) == (0, 64)
    assert (result.peak_total_device_bytes, result.gpu_terminal.releases) == (128, 2)
    lease = Lease()
    scope.check(quote_api.fort_scope_scratch_acquire_v1(scope.handle, 64, c.byref(lease)))
    assert quote(scope, [binding(handle, access(flags=1))], request=request).reason == 4
    scope.check(quote_api.fort_scope_scratch_release_v1(scope.handle, lease.token))
    reused = quote(scope, [binding(handle, access(flags=1))], request=request)
    assert reused.available
    assert (reused.scratch_capacity_before, reused.scratch_capacity_after) == (64, 64)
    request.scratch_bytes = 128
    grown = quote(scope, [binding(handle, access(flags=1))], request=request)
    assert grown.available
    assert grown.gpu_execution_seconds - reused.gpu_execution_seconds == pytest.approx(.001)
    assert grown.native_terminal.releases == 1
    scope.close()
    domain.close()


@pytest.mark.parametrize("case", ["missing_native", "missing_gpu", "launch_included", "no_endpoint",
    "bad_protocol", "nonfinite", "launch_range", "byte_range", "incompatible", "undefined"])
def test_unknown_or_incompatible_evidence_declines_without_changing_state(quote_api, case):
    domain = Domain(quote_api)
    scope, _host, handle = registered(domain, initialized=case != "undefined")
    request, rates = work(), protocol()
    if case == "missing_native":
        request.compute.known_backends = 4
    elif case == "missing_gpu":
        request.compute.known_backends = 1
    elif case == "launch_included":
        request.compute.known_backends |= 8
    elif case == "no_endpoint":
        request.flags = 0
    elif case == "bad_protocol":
        rates.protocol_id += 1
    elif case == "nonfinite":
        rates.source_submit_seconds = math.nan
    elif case == "launch_range":
        request.expected_launches = 257
    elif case == "byte_range":
        request.flags |= 1
        request.scratch_bytes = rates.max_scratch_bytes + 1
    before = bytes(scope.stats())
    result = quote(scope, [binding(handle, access(flags=1))], request=request, rates=rates,
                   compatible=case != "incompatible")
    assert not result.available
    assert not result.gpu_preferred
    assert bytes(scope.stats()) == before
    scope.close()
    domain.close()


def test_publication_cost_covers_pending_upload_readers_without_flushing_query(quote_api):
    domain = Domain(quote_api)
    scope, _host, handle = registered(domain)
    scope.gpu(handle, access(flags=1))
    scope.gpu_end(handle)
    before = bytes(scope.stats())
    result = quote(scope, [binding(handle, access(flags=1))])
    assert result.available
    assert result.native_terminal.waits == result.gpu_terminal.waits == 1
    assert result.native_terminal.download_bytes == 0
    assert result.gpu.upload_bytes == 0
    assert bytes(scope.stats()) == before
    scope.close()
    domain.close()


def test_pending_scratch_growth_declines_but_idle_reuse_and_completed_growth_work(quote_api):
    domain = Domain(quote_api)
    scope, _host, handle = registered(domain)
    lease = Lease()
    scope.check(quote_api.fort_scope_scratch_acquire_v1(scope.handle, 64, c.byref(lease)))
    scope.check(quote_api.fort_scope_scratch_release_v1(scope.handle, lease.token))
    scope.check(quote_api.fort_scope_note_launch(scope.handle))
    request = work()
    request.flags |= 1
    request.scratch_bytes = 64
    assert quote(scope, [binding(handle, access(flags=1))], request=request).available
    request.scratch_bytes = 128
    assert quote(scope, [binding(handle, access(flags=1))], request=request).reason == 4
    request.flags |= 4
    fenced = quote(scope, [binding(handle, access(flags=1))], request=request)
    assert fenced.available
    assert fenced.gpu.waits == 1
    assert fenced.gpu_terminal.waits == 1
    scope.check(quote_api.fort_scope_wait(scope.handle))
    completed = quote(scope, [binding(handle, access(flags=1))], request=request)
    assert completed.available
    assert completed.gpu.waits == 0
    scope.close()
    domain.close()


def test_scratch_cannot_evict_fields_before_its_actual_acquire(quote_api):
    domain = Domain(quote_api)
    scope, _host, handle = registered(domain)
    scope.check(quote_api.fort_scope_set_device_budget(scope.handle, 128))
    scope.gpu(handle, access(flags=1))
    scope.gpu_end(handle)
    request = work()
    request.flags |= 1
    request.scratch_bytes = 128
    assert quote(scope, [binding(handle, access(flags=1))], request=request).reason == 5
    before = bytes(scope.stats())
    lease = Lease()
    assert quote_api.fort_scope_scratch_acquire_v1(scope.handle, 128, c.byref(lease)) == 4
    assert bytes(scope.stats()) == before
    scope.close()
    domain.close()


def test_opaque_workspace_cleanup_remains_unavailable_and_balanced(quote_api):
    domain = Domain(quote_api)
    scope, _host, handle = registered(domain)
    lease = TOKEN()
    domain.check(quote_api.fort_scope_domain_workspace_reserve_v1(domain.handle, 128, c.byref(lease)))
    assert quote(scope, [binding(handle, access(flags=1))]).reason == 4
    domain.check(quote_api.fort_scope_domain_workspace_release_v1(domain.handle, lease))
    assert quote(scope, [binding(handle, access(flags=1))]).available
    scope.close()
    domain.close()


def test_query_metadata_and_canonical_handles_are_not_installed_or_replaced(quote_api):
    from compiler.tests.test_scoped_planning_runtime import Binding as PlanBinding

    domain = Domain(quote_api)
    scope, _host, handle = registered(domain)
    scope.check(quote_api.fort_scope_plan_reset(scope.handle))
    bindings = (PlanBinding * 1)(PlanBinding(handle, access(flags=1)))
    scope.check(quote_api.fort_scope_plan_add(scope.handle, 1, 111, bindings, 1, 1e8, 64., 1))
    scope.check(quote_api.fort_scope_plan_validate(scope.handle))
    assert quote(scope, [binding(handle, access(flags=1))]).available
    decision = Decision()
    scope.check(quote_api.fort_scope_plan_select(scope.handle, c.byref(costs()), 1, c.byref(decision)))
    assert decision.available
    # The original metadata query survived the unrelated source quote.
    quote_api.fort_scope_plan_next.argtypes = [TOKEN, TOKEN, c.POINTER(PlanBinding), SIZE, c.POINTER(c.c_int)]
    chosen = c.c_int()
    scope.check(quote_api.fort_scope_plan_next(scope.handle, 111, bindings, 1, c.byref(chosen)))
    assert chosen.value == 1
    scope.close()
    domain.close()


def test_new_helper_is_materialized_and_hashes_into_runtime_identity():
    import hashlib

    from compiler.emission.common.resources import read_scoped_runtime

    sources, manifest = read_scoped_runtime()
    name = "scoped_source_quote.hpp"
    assert name in manifest["headers"]
    assert '#include "scoped_source_quote.hpp"' in sources["scoped_runtime.cu"]
    hashes = {path: hashlib.sha256(source.encode()).hexdigest() for path, source in sources.items()}
    def identity(values):
        return hashlib.sha256("\n".join(
            f"{path}:{digest}" for path, digest in sorted(values.items())).encode()).hexdigest()
    assert manifest["runtime_id"] == identity(hashes)
    hashes[name] = hashlib.sha256((sources[name] + "\n").encode()).hexdigest()
    assert manifest["runtime_id"] != identity(hashes)
