"""Native-live quotes price exact host accesses without domain retirement."""

import ctypes as c
import math

import pytest

from compiler.tests.test_scoped_domains import Binding, Call, Domain, binding
from compiler.tests.test_scoped_domains import domain_runtime as _domain_runtime
from compiler.tests.test_scoped_planning_runtime import costs
from compiler.tests.test_scoped_runtime import TOKEN, access
from compiler.tests.test_scoped_source_quote import Costs, Protocol, Quote, Work, protocol, registered, work
from compiler.tests.test_scoped_source_quote import quote_api as _quote_api

quote_api, domain_runtime = _quote_api, _domain_runtime
pytestmark = pytest.mark.native


class NativeCosts(c.Structure):
    _fields_ = [("version", c.c_uint32), ("valid", c.c_uint32), ("protocol_id", TOKEN),
                ("max_bindings", TOKEN), ("prepare_start_complete_seconds", c.c_double)]


def native_costs():
    return NativeCosts(1, 1, 0x465453484F530001, 8, .025)


def live(scope, bindings, *, native=None, request=None, base=None):
    packed = (Binding * len(bindings))(*bindings)
    call, result = Call(1, 0, packed, len(bindings), 0, 0), Quote()
    function = scope.lib.fort_scope_source_quote_v2
    function.argtypes = [TOKEN, c.POINTER(Call), c.POINTER(Work), c.POINTER(Costs), c.POINTER(Protocol),
                         c.POINTER(NativeCosts), c.c_int, c.POINTER(Quote)]
    function.restype = c.c_int
    if request is None:
        request = work()
        request.version, request.flags = 2, 8
    scope.check(function(scope.handle, c.byref(call), c.byref(request), c.byref(base or costs()),
                         c.byref(protocol()), c.byref(native or native_costs()), 1, c.byref(result)))
    return result


def test_live_endpoint_does_not_publish_or_price_unrelated_dirty_resources(quote_api):
    domain = Domain(quote_api)
    scope, host, handle = registered(domain)
    dirty = (c.c_double * 8)(*range(8))
    status, other = scope.register(dirty, [8], identity=2)
    scope.check(status)
    device = scope.gpu(other, access(flags=3))
    device[0] += 100
    scope.gpu_end(other)
    before = bytes(scope.stats())
    result = live(scope, [binding(handle, access(flags=1))])
    assert result.available
    assert (result.version, result.native_endpoint, result.gpu_endpoint) == (2, 8, 8)
    assert result.native_seconds == pytest.approx(1.025)
    assert bytes(result.native_terminal) == bytes(type(result.native_terminal)())
    assert bytes(result.gpu_terminal) == bytes(type(result.gpu_terminal)())
    assert result.gpu_seconds == result.gpu_execution_seconds
    assert result.gpu.download_bytes == 0
    assert bytes(scope.stats()) == before
    assert dirty[0] == host[0] == 0
    scope.close()
    domain.close()
    assert dirty[0] == 100


def test_native_transaction_measurement_replaces_per_binding_host_charge(quote_api):
    domain = Domain(quote_api)
    scope, _host, handle = registered(domain)
    bindings = [binding(handle, access(flags=1))]
    before = live(scope, bindings)
    base = costs()
    base.host_access_seconds = 100
    after = live(scope, bindings, base=base)
    assert after.native_seconds == before.native_seconds
    native = native_costs()
    native.prepare_start_complete_seconds += .25
    changed = live(scope, bindings, native=native)
    assert changed.native_seconds - before.native_seconds == pytest.approx(.25)
    assert changed.gpu_seconds == before.gpu_seconds
    scope.close()
    domain.close()


def test_native_partial_mirror_acquisition_is_local_and_quote_stays_pure(quote_api):
    domain = Domain(quote_api)
    scope, host, handle = registered(domain)
    device = scope.gpu(handle, access(flags=3))
    for i in range(8):
        device[i] += 10
    scope.gpu_end(handle)
    scope.cpu_begin(handle, access(read=[([2], [3])]))
    scope.cpu_end(handle)
    before = bytes(scope.stats())
    result = live(scope, [binding(handle, access(read=[([1], [4])]))])
    assert result.available
    assert result.native_seconds > 1.025
    assert result.native_terminal.download_bytes == 0
    assert result.gpu.upload_bytes == result.gpu.download_bytes == 0
    assert bytes(scope.stats()) == before
    assert list(host) == [0, 1, 12, 3, 4, 5, 6, 7]
    scope.close()
    domain.close()


@pytest.mark.parametrize("fault", ["version", "protocol", "nonfinite", "missing", "range", "endpoint", "work_version"])
def test_missing_or_incompatible_native_live_costs_decline_without_invented_zero(quote_api, fault):
    domain = Domain(quote_api)
    scope, _host, handle = registered(domain)
    native, request = native_costs(), work()
    request.version, request.flags = 2, 8
    if fault == "version":
        native.version = 2
    elif fault == "protocol":
        native.protocol_id += 1
    elif fault == "nonfinite":
        native.prepare_start_complete_seconds = math.nan
    elif fault == "missing":
        native.prepare_start_complete_seconds = 0
    elif fault == "range":
        native.max_bindings = 0
    elif fault == "endpoint":
        request.flags = 2
    else:
        request.version = 1
    before = bytes(scope.stats())
    result = live(scope, [binding(handle, access(flags=1))], native=native, request=request)
    assert not result.available
    assert not result.gpu_preferred
    assert bytes(scope.stats()) == before
    scope.close()
    domain.close()


def test_live_keeps_existing_provider_workspace_without_pricing_its_destruction(quote_api):
    domain = Domain(quote_api)
    scope, _host, handle = registered(domain)
    lease = TOKEN()
    domain.check(quote_api.fort_scope_domain_workspace_reserve_v1(domain.handle, 64, c.byref(lease)))
    result = live(scope, [binding(handle, access(flags=1))])
    assert result.available
    assert result.gpu_terminal.releases == result.native_terminal.releases == 0
    assert result.peak_total_device_bytes == 128
    domain.check(quote_api.fort_scope_domain_workspace_release_v1(domain.handle, lease))
    scope.close()
    domain.close()
