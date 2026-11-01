"""Caller-owned physical residency and checked foreign GPU transactions."""

from __future__ import annotations

import ctypes as c
import json
import shutil
import subprocess

import pytest

from compiler.tests.test_scoped_planning_runtime import Binding as PlanBinding
from compiler.tests.test_scoped_planning_runtime import Costs, Decision, costs
from compiler.tests.test_scoped_runtime import RUNTIME, SIZE, TOKEN, Access, Layout, Scope, Stats, access

pytestmark = pytest.mark.native


class Binding(c.Structure):
    _fields_ = [("buffer", TOKEN), ("type", c.c_uint32), ("rank", c.c_uint32),
                ("element_bytes", SIZE), ("access", Access)]


class View(c.Structure):
    _fields_ = [("buffer", TOKEN), ("generation", TOKEN), ("type", c.c_uint32),
                ("rank", c.c_uint32), ("element_bytes", SIZE), ("bytes", SIZE),
                ("extents", c.POINTER(SIZE)), ("strides", c.POINTER(SIZE)),
                ("lower_bounds", c.POINTER(c.c_int64)), ("device", c.c_void_p)]


class Call(c.Structure):
    _fields_ = [("version", c.c_uint32), ("reserved", c.c_uint32),
                ("bindings", c.POINTER(Binding)), ("binding_count", SIZE),
                ("workspace_lease", TOKEN), ("workspace_bytes", SIZE)]


class Report(c.Structure):
    _fields_ = [("version", c.c_uint32), ("status", c.c_int32),
                ("started", c.c_uint32), ("committed", c.c_uint32)]


class Hint(c.Structure):
    _fields_ = [("version", c.c_uint32), ("eligible", c.c_uint32), ("word_count", SIZE),
                ("gpu_generation", TOKEN), ("error_generation", TOKEN), ("words", TOKEN * 512)]


Worker = c.CFUNCTYPE(c.c_int, c.c_void_p, c.POINTER(View), SIZE, c.c_void_p, c.POINTER(c.c_int))


@pytest.fixture(scope="module")
def domain_runtime(tmp_path_factory):
    directory = tmp_path_factory.mktemp("domain_runtime")
    compiler = shutil.which("g++-14") or shutil.which("g++")
    if not compiler:
        pytest.skip("g++ unavailable")
    target = directory / "runtime.so"
    command = [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-fPIC", "-shared", "-pthread", "-fopenmp",
               "-DFORT_SCOPE_CPU_TEST", "-x", "c++", str(RUNTIME / "scoped_runtime.cu"), "-o", str(target)]
    built = subprocess.run(command, capture_output=True, text=True, timeout=90)
    assert built.returncode == 0, built.stderr
    lib = c.CDLL(str(target))
    lib.fort_scope_error.restype = c.c_char_p
    signatures = {
        "create": [c.c_int, c.POINTER(TOKEN)],
        "domain_create_v1": [c.c_int, c.POINTER(TOKEN)],
        "create_in_domain_v1": [TOKEN, c.POINTER(TOKEN)],
        "domain_get_v1": [TOKEN, c.POINTER(TOKEN)],
        "domain_device_get_v1": [TOKEN, c.POINTER(c.c_int)],
        "domain_stats_get_v1": [TOKEN, c.POINTER(Stats)],
        "domain_close_v1": [TOKEN], "domain_abandon_v1": [TOKEN],
        "domain_workspace_reserve_v1": [TOKEN, SIZE, c.POINTER(TOKEN)],
        "domain_workspace_resize_v1": [TOKEN, TOKEN, SIZE],
        "domain_workspace_release_v1": [TOKEN, TOKEN],
        "register": [TOKEN, TOKEN, TOKEN, c.POINTER(Layout), c.c_int, c.POINTER(TOKEN)],
        "identity_swap_v1": [TOKEN, TOKEN, TOKEN, c.POINTER(Layout), c.POINTER(Layout),
                             c.POINTER(TOKEN), c.POINTER(TOKEN)],
        "layout_get": [TOKEN, TOKEN, c.POINTER(Layout)],
        "plan_reset": [TOKEN],
        "plan_validate": [TOKEN],
        "plan_add": [TOKEN, c.c_uint32, TOKEN, c.POINTER(PlanBinding), SIZE, c.c_double, c.c_double, c.c_int],
        "plan_select": [TOKEN, c.POINTER(Costs), c.c_int, c.POINTER(Decision)],
        "set_device_budget": [TOKEN, SIZE],
        "device_begin": [TOKEN, TOKEN, c.POINTER(Access), c.POINTER(c.c_void_p)],
        "host_begin": [TOKEN, TOKEN, c.POINTER(Access)],
        "device_end": [TOKEN, TOKEN], "host_end": [TOKEN, TOKEN], "cancel_access": [TOKEN, TOKEN],
        "retire_v1": [TOKEN, TOKEN], "close": [TOKEN], "abandon": [TOKEN], "wait": [TOKEN],
        "stats_get": [TOKEN, c.POINTER(Stats)],
        "external_execute_v1": [TOKEN, c.POINTER(Call), Worker, c.c_void_p, c.POINTER(Report)],
        "native_hint_begin_v1": [c.POINTER(TOKEN), SIZE, c.POINTER(Hint), c.POINTER(c.c_int)],
        "native_hint_finish_v1": [c.POINTER(Hint)],
    }
    for name, signature in signatures.items():
        function = getattr(lib, "fort_scope_" + name)
        function.argtypes, function.restype = signature, c.c_int
    lib.fort_scope_trace_set_v1.argtypes = [TOKEN, c.c_uint32, *([c.c_char_p] * 6), c.c_void_p]
    lib.fort_scope_trace_set_v1.restype = None
    return lib


class Domain:
    def __init__(self, library, *, device=0):
        self.lib, self.handle = library, TOKEN()
        self.check(library.fort_scope_domain_create_v1(device, c.byref(self.handle)))

    def check(self, status):
        assert status == 0, self.lib.fort_scope_error().decode()

    def context(self):
        result = Scope.__new__(Scope)
        result.lib, result.handle, result.references = self.lib, TOKEN(), []
        self.check(self.lib.fort_scope_create_in_domain_v1(self.handle, c.byref(result.handle)))
        return result

    def stats(self):
        result = Stats()
        self.check(self.lib.fort_scope_domain_stats_get_v1(self.handle, c.byref(result)))
        return result

    def close(self):
        self.check(self.lib.fort_scope_domain_close_v1(self.handle))


def execute(scope, bindings, worker, *, lease=0, workspace=0):
    packed = (Binding * len(bindings))(*bindings)
    descriptor = Call(1, 0, packed, len(bindings), lease, workspace)
    callback, report = Worker(worker), Report()
    status = scope.lib.fort_scope_external_execute_v1(scope.handle, c.byref(descriptor), callback, None,
                                                     c.byref(report))
    return status, report


def binding(handle, effects):
    return Binding(handle, 2, 1, 8, effects)


def layout(values, shape, generation, lower):
    extents, bounds = (SIZE * len(shape))(*shape), (c.c_int64 * len(shape))(*lower)
    result = Layout(len(shape), 2, 8, c.cast(values, c.c_void_p), extents, bounds, generation)
    result.references = [values, extents, bounds]
    return result


def test_detach_retains_exact_values_readonly_mirrors_and_canonical_generation(domain_runtime):
    domain = Domain(domain_runtime)
    host = (c.c_double * 8)(*range(8))
    first = domain.context()
    status, handle = first.register(host, [8], lower=[-2])
    first.check(status)
    interior = [([1], [7])]
    device = first.gpu(handle, access(read=interior, write=interior))
    for i in range(1, 7):
        device[i] += 100
    first.gpu_end(handle)
    first.close()
    assert list(host) == list(range(8)), "detach must not publish dirty values"
    assert domain.stats().download_bytes == 0

    second = domain.context()
    status, same = second.register(host, [8], lower=[-2], initialized=False)
    second.check(status)
    assert same.value == handle.value
    second.cpu_begin(same, access(read=[([3], [4])]))
    assert host[3] == 103
    assert host[2] == 2
    second.cpu_end(same)
    before = second.stats()
    device = second.gpu(same, access(read=interior))
    assert [device[i] for i in range(1, 7)] == [i + 100 for i in range(1, 7)]
    second.gpu_end(same)
    assert second.stats().upload_bytes == before.upload_bytes == 48
    assert second.stats().download_bytes == 8
    second.close()
    domain.close()
    assert list(host) == [0, *[i + 100 for i in range(1, 7)], 7]
    assert domain_runtime.fort_scope_wait(first.handle) == 2
    assert domain_runtime.fort_scope_create_in_domain_v1(domain.handle, c.byref(TOKEN())) == 2


def test_one_active_context_and_prepared_access_block_detach(domain_runtime):
    domain = Domain(domain_runtime)
    scope = domain.context()
    assert domain_runtime.fort_scope_create_in_domain_v1(domain.handle, c.byref(TOKEN())) == 8
    assert domain_runtime.fort_scope_domain_close_v1(domain.handle) == 8
    found = TOKEN()
    scope.check(domain_runtime.fort_scope_domain_get_v1(scope.handle, c.byref(found)))
    assert found.value == domain.handle.value
    host = (c.c_double * 2)(1, 2)
    status, handle = scope.register(host, [2])
    scope.check(status)
    scope.gpu(handle, access(flags=1))
    assert domain_runtime.fort_scope_close(scope.handle) == 8
    scope.gpu_end(handle)
    scope.close()
    domain.close()
    private = Scope(domain_runtime)
    private.check(domain_runtime.fort_scope_domain_get_v1(private.handle, c.byref(found)))
    assert found.value == 0
    private.close()


def test_domain_ordinal_query_is_metadata_only_and_stale_after_retirement(domain_runtime):
    domain = Domain(domain_runtime, device=3)
    ordinal = c.c_int(-1)
    domain.check(domain_runtime.fort_scope_domain_device_get_v1(domain.handle, c.byref(ordinal)))
    assert ordinal.value == 3
    assert bytes(domain.stats()) == bytes(Stats())
    domain.close()
    assert domain_runtime.fort_scope_domain_device_get_v1(domain.handle, c.byref(ordinal)) == 2


def test_storage_retirement_and_aba_never_reuse_handles_or_definitions(domain_runtime):
    domain = Domain(domain_runtime)
    host = (c.c_double * 4)(1, 2, 3, 4)
    scope = domain.context()
    status, old = scope.register(host, [4])
    scope.check(status)
    scope.check(domain_runtime.fort_scope_retire_v1(scope.handle, old))
    status, new = scope.register(host, [4], generation=2, initialized=False)
    scope.check(status)
    assert new.value != old.value
    pointer = c.c_void_p()
    assert domain_runtime.fort_scope_device_begin(scope.handle, old, c.byref(access(flags=1)), c.byref(pointer)) == 2
    assert domain_runtime.fort_scope_device_begin(scope.handle, new, c.byref(access(flags=1)), c.byref(pointer)) == 7
    scope.close()
    reopened = domain.context()
    assert reopened.register(host, [4], generation=1)[0] == 2
    status, same = reopened.register(host, [4], generation=2, initialized=True)
    reopened.check(status)
    assert same.value == new.value
    assert domain_runtime.fort_scope_device_begin(reopened.handle, same, c.byref(access(flags=1)), c.byref(pointer)) == 7
    reopened.close()
    domain.close()


def test_workspace_claims_share_exact_payload_budget_and_survive_detach(domain_runtime):
    domain = Domain(domain_runtime)
    scope = domain.context()
    scope.check(domain_runtime.fort_scope_set_device_budget(scope.handle, 96))
    lease = TOKEN()
    scope.check(domain_runtime.fort_scope_domain_workspace_reserve_v1(domain.handle, 64, c.byref(lease)))
    host = (c.c_double * 8)(*range(8))
    status, handle = scope.register(host, [8])
    scope.check(status)
    pointer = c.c_void_p()
    assert domain_runtime.fort_scope_device_begin(scope.handle, handle, c.byref(access(flags=1)), c.byref(pointer)) == 4
    assert scope.stats().allocations == 0
    scope.check(domain_runtime.fort_scope_domain_workspace_resize_v1(domain.handle, lease, 32))
    scope.gpu(handle, access(flags=1))
    scope.gpu_end(handle)
    assert domain_runtime.fort_scope_domain_workspace_resize_v1(domain.handle, lease, 33) == 4
    scope.close()
    assert domain_runtime.fort_scope_domain_close_v1(domain.handle) == 8
    other = Domain(domain_runtime)
    assert domain_runtime.fort_scope_domain_workspace_release_v1(other.handle, lease) == 2
    other.close()
    domain.check(domain_runtime.fort_scope_domain_workspace_release_v1(domain.handle, lease))
    assert domain_runtime.fort_scope_domain_workspace_release_v1(domain.handle, lease) == 2
    domain.close()


def test_publish_retire_storage_cut_then_descriptor_swap_rebinds_current_allocation_generation(domain_runtime):
    left, right = (c.c_double * 2)(1, 2), (c.c_double * 2)(3, 4)
    first_domain = Domain(domain_runtime)
    before_cut = first_domain.context()
    old_handles = []
    for identity, host, increment in [(1, left, 10), (2, right, 20)]:
        status, handle = before_cut.register(host, [2], identity=identity)
        before_cut.check(status)
        old_handles.append(handle)
        device = before_cut.gpu(handle, access(flags=3))
        for i in range(2):
            device[i] += increment
        before_cut.gpu_end(handle)
    before_cut.close()
    assert list(left) == [1, 2]
    assert list(right) == [3, 4]
    first_domain.close()  # Mandatory publication/storage frontier before swap.
    assert list(left) == [11, 12]
    assert list(right) == [23, 24]

    left, right = right, left
    second_domain = Domain(domain_runtime)
    after_cut = second_domain.context()
    current_handles = []
    for identity, host, expected in [(1, left, [23, 24]), (2, right, [11, 12])]:
        status, handle = after_cut.register(host, [2], identity=identity, generation=2)
        after_cut.check(status)
        current_handles.append(handle)
        device = after_cut.gpu(handle, access(flags=1))
        assert [device[i] for i in range(2)] == expected
        after_cut.gpu_end(handle)
    pointer = c.c_void_p()
    for handle in old_handles:
        assert domain_runtime.fort_scope_device_begin(after_cut.handle, handle, c.byref(access(flags=1)),
                                                     c.byref(pointer)) == 2
    assert set(item.value for item in old_handles).isdisjoint(item.value for item in current_handles)
    assert after_cut.register(right, [2], identity=1, generation=2)[0] == 1
    assert after_cut.register(left, [2], identity=1, generation=1)[0] == 2
    after_cut.close()
    continued = second_domain.context()
    for identity, host, original in [(1, left, current_handles[0]), (2, right, current_handles[1])]:
        status, handle = continued.register(host, [2], identity=identity, generation=2, initialized=False)
        continued.check(status)
        assert handle.value == original.value
    assert domain_runtime.fort_scope_host_begin(after_cut.handle, current_handles[0], c.byref(access(flags=1))) == 2
    continued.close()
    second_domain.close()


def test_contexts_never_adopt_foreign_domain_or_private_buffer_handles(domain_runtime):
    first, second = Domain(domain_runtime), Domain(domain_runtime)
    owner, foreign = first.context(), second.context()
    private = Scope(domain_runtime)
    host = (c.c_double * 2)(7, 9)
    status, owned = owner.register(host, [2])
    owner.check(status)
    # Equal host address/identity/generation does not grant access across
    # domains. There is deliberately no global pointer-to-replica cache.
    for scope in (foreign, private):
        status, handle = scope.register(host, [2])
        scope.check(status)
        assert handle.value != owned.value
        assert domain_runtime.fort_scope_host_begin(scope.handle, owned, c.byref(access(flags=1))) == 2
        assert domain_runtime.fort_scope_host_begin(owner.handle, handle, c.byref(access(flags=1))) == 2
    owner.close()
    foreign.close()
    private.close()
    first.close()
    second.close()


def test_checked_descriptor_exchange_preserves_physical_generation_dirty_values_and_handles(domain_runtime, monkeypatch, capfd):
    monkeypatch.setenv("FORT_RUNTIME_TRACE", "1")
    domain = Domain(domain_runtime)
    scope = domain.context()
    a, b = (c.c_double * 4)(1, 2, 3, 4), (c.c_double * 6)(11, 12, 13, 14, 15, 16)
    physical = []
    for identity, host, shape, lower, generation in [(1, a, [2, 2], [-2, 3], 11),
                                                    (2, b, [3, 2], [-7, 9], 27)]:
        status, handle = scope.register(host, shape, identity=identity, lower=lower, generation=generation)
        scope.check(status)
        physical.append(handle)
        device = scope.gpu(handle, access(flags=3))
        for i in range(len(host)):
            device[i] += 100 * identity
        scope.gpu_end(handle)
    scope.close()
    continued = domain.context()
    before = bytes(continued.stats())
    after_a, after_b = layout(b, [3, 2], 27, [-7, 9]), layout(a, [2, 2], 11, [-2, 3])
    current_a, current_b = TOKEN(), TOKEN()
    continued.check(domain_runtime.fort_scope_identity_swap_v1(
        continued.handle, *physical, c.byref(after_a), c.byref(after_b), c.byref(current_a), c.byref(current_b)))
    assert (current_a.value, current_b.value) == (physical[1].value, physical[0].value)
    assert bytes(continued.stats()) == before, "descriptor exchange must not copy, allocate, wait or replay"
    for identity, host, shape, lower, generation, expected_handle in [
            (1, b, [3, 2], [-7, 9], 27, current_a), (2, a, [2, 2], [-2, 3], 11, current_b)]:
        status, handle = continued.register(host, shape, identity=identity, lower=lower, generation=generation)
        continued.check(status)
        assert handle.value == expected_handle.value
        descriptor = Layout()
        continued.check(domain_runtime.fort_scope_layout_get(continued.handle, handle, c.byref(descriptor)))
        assert descriptor.generation == generation
        device = continued.gpu(handle, access(flags=1))
        expected = [11 + i + 200 for i in range(6)] if identity == 1 else [1 + i + 100 for i in range(4)]
        assert [device[i] for i in range(len(host))] == expected
        continued.gpu_end(handle)
    assert continued.stats().upload_bytes == 80
    assert continued.stats().download_bytes == 0
    continued.close()
    assert list(a) == [1, 2, 3, 4]
    assert list(b) == [11, 12, 13, 14, 15, 16]
    domain.close()
    assert list(a) == [101, 102, 103, 104]
    assert list(b) == [211, 212, 213, 214, 215, 216]
    exchanges = [line for line in capfd.readouterr().err.splitlines() if line.startswith("FORT_SCOPED identity_swap ")]
    assert len(exchanges) == 1
    fields = dict(item.split("=", 1) for item in exchanges[0].split()[2:])
    assert fields["swap_version"] == "1"
    assert (fields["first_identity"], fields["second_identity"]) == ("1", "2")
    assert int(fields["first_before_handle"]) == int(fields["second_after_handle"]) == physical[0].value
    assert int(fields["second_before_handle"]) == int(fields["first_after_handle"]) == physical[1].value
    assert (fields["first_before_generation"], fields["second_after_generation"]) == ("11", "11")
    assert (fields["second_before_generation"], fields["first_after_generation"]) == ("27", "27")


def test_descriptor_exchange_never_seeds_definitions_from_the_new_logical_name(domain_runtime):
    domain = Domain(domain_runtime)
    scope = domain.context()
    a, b = (c.c_double * 2)(-9, -8), (c.c_double * 2)(3, 5)
    status, old_a = scope.register(a, [2], initialized=False, generation=2)
    scope.check(status)
    status, old_b = scope.register(b, [2], identity=2, generation=7)
    scope.check(status)
    after_a, after_b = layout(b, [2], 7, [0]), layout(a, [2], 2, [0])
    new_a, new_b = TOKEN(), TOKEN()
    scope.check(domain_runtime.fort_scope_identity_swap_v1(scope.handle, old_a, old_b, c.byref(after_a),
                                                         c.byref(after_b), c.byref(new_a), c.byref(new_b)))
    assert scope.register(a, [2], identity=2, generation=2, initialized=True)[1].value == new_b.value
    pointer = c.c_void_p()
    assert domain_runtime.fort_scope_device_begin(scope.handle, new_b, c.byref(access(flags=1)), c.byref(pointer)) == 7
    device = scope.gpu(new_a, access(flags=1))
    assert [device[0], device[1]] == [3, 5]
    scope.gpu_end(new_a)
    scope.close()
    domain.close()
    assert list(a) == [-9, -8]


def test_validated_uninstalled_query_is_abandoned_after_actual_host_commit_and_descriptor_swap(domain_runtime):
    domain = Domain(domain_runtime)
    scope = domain.context()
    a, b = (c.c_double * 2)(1, 2), (c.c_double * 2)(3, 4)
    status, old_a = scope.register(a, [2], generation=2)
    scope.check(status)
    status, old_b = scope.register(b, [2], identity=2, generation=7)
    scope.check(status)
    effects = access(flags=3)
    native = (PlanBinding * 1)(PlanBinding(old_a, effects))
    scope.check(domain_runtime.fort_scope_plan_reset(scope.handle))
    scope.check(domain_runtime.fort_scope_plan_add(scope.handle, 0, 0, native, 1, 0, 16, 0))
    scope.check(domain_runtime.fort_scope_plan_validate(scope.handle))
    # Forced sections/native fallback has performed the original operation,
    # even though its validation-only query was never installed as a schedule.
    scope.cpu_begin(old_a, effects)
    a[0], a[1] = 11, 12
    scope.cpu_end(old_a)
    after_a, after_b = layout(b, [2], 7, [0]), layout(a, [2], 2, [0])
    new_a, new_b = TOKEN(), TOKEN()
    scope.check(domain_runtime.fort_scope_identity_swap_v1(scope.handle, old_a, old_b, c.byref(after_a),
                                                         c.byref(after_b), c.byref(new_a), c.byref(new_b)))
    assert (new_a.value, new_b.value) == (old_b.value, old_a.value)
    assert domain_runtime.fort_scope_plan_validate(scope.handle) == 8
    device = scope.gpu(new_b, access(flags=1))
    assert [device[0], device[1]] == [11, 12]
    scope.gpu_end(new_b)
    scope.close()
    domain.close()
    assert list(a) == [11, 12]
    assert list(b) == [3, 4]


@pytest.mark.parametrize("invalid", ["host", "generation", "extent", "lower", "same_handle", "prepared", "installed_query"])
def test_postmutation_descriptor_exchange_rejects_fatally_without_replay_or_publication(domain_runtime, invalid):
    domain = Domain(domain_runtime)
    scope = domain.context()
    a, b = (c.c_double * 2)(1, 2), (c.c_double * 2)(3, 4)
    status, old_a = scope.register(a, [2], generation=2)
    scope.check(status)
    status, old_b = scope.register(b, [2], identity=2, generation=7)
    scope.check(status)
    device = scope.gpu(old_a, access(flags=3))
    device[0] = 1000
    scope.gpu_end(old_a)
    after_a, after_b = layout(b, [2], 7, [0]), layout(a, [2], 2, [0])
    if invalid == "host":
        after_a.host = c.cast(a, c.c_void_p)
    elif invalid == "generation":
        after_a.generation = 2
    elif invalid == "extent":
        after_a = layout(b, [1], 7, [0])
    elif invalid == "lower":
        after_a = layout(b, [2], 7, [-1])
    elif invalid == "same_handle":
        old_b = old_a
    elif invalid == "prepared":
        scope.gpu(old_b, access(flags=1))
    elif invalid == "installed_query":
        scope.check(domain_runtime.fort_scope_wait(scope.handle))
        scope.check(domain_runtime.fort_scope_plan_reset(scope.handle))
        worker = (PlanBinding * 1)(PlanBinding(old_b, access(flags=1)))
        scope.check(domain_runtime.fort_scope_plan_add(scope.handle, 1, 42, worker, 1, 1e8, 16, 1))
        scope.check(domain_runtime.fort_scope_plan_validate(scope.handle))
        decision = Decision()
        scope.check(domain_runtime.fort_scope_plan_select(scope.handle, c.byref(costs()), 1, c.byref(decision)))
        assert decision.available == 1
        assert decision.gpu_units == 1
    current_a, current_b = TOKEN(111), TOKEN(222)
    assert domain_runtime.fort_scope_identity_swap_v1(scope.handle, old_a, old_b, c.byref(after_a),
                                                     c.byref(after_b), c.byref(current_a), c.byref(current_b)) == 6
    assert (current_a.value, current_b.value) == (111, 222)
    assert domain_runtime.fort_scope_wait(scope.handle) == 6
    assert domain_runtime.fort_scope_close(scope.handle) == 6
    scope.check(domain_runtime.fort_scope_abandon(scope.handle))
    domain.check(domain_runtime.fort_scope_domain_abandon_v1(domain.handle))
    assert list(a) == [1, 2]
    assert list(b) == [3, 4]


def test_external_typed_partial_effects_and_workspace_are_checked(domain_runtime):
    domain = Domain(domain_runtime)
    scope = domain.context()
    source, output = (c.c_double * 6)(*range(6)), (c.c_double * 6)(*([-99] * 6))
    status, a = scope.register(source, [6], lower=[-3], generation=11)
    scope.check(status)
    status, b = scope.register(output, [6], identity=2, initialized=False)
    scope.check(status)
    box = [([1], [5])]
    bindings = [binding(a, access(read=box)), binding(b, access(write=box, overwrite=box))]
    observations = []

    def worker(_user, views, count, stream, started):
        observations.append((count, stream, views[0].generation, views[0].extents[0], views[0].strides[0],
                             views[0].lower_bounds[0], views[1].bytes))
        started[0] = 1
        av, bv = [c.cast(views[i].device, c.POINTER(c.c_double)) for i in range(2)]
        for i in range(1, 5):
            bv[i] = 3 * av[i]
        return 0

    lease = TOKEN()
    scope.check(domain_runtime.fort_scope_domain_workspace_reserve_v1(domain.handle, 24, c.byref(lease)))
    status, report = execute(scope, bindings, worker, lease=lease.value, workspace=25)
    assert status == 4
    assert observations == []
    status, report = execute(scope, bindings, worker, lease=lease.value, workspace=24)
    scope.check(status)
    assert (report.version, report.status, report.started, report.committed) == (1, 0, 1, 1)
    assert observations == [(2, None, 11, 6, 8, -3, 48)]
    assert list(output) == [-99] * 6
    scope.cpu_begin(b, access(read=box))
    scope.cpu_end(b)
    assert list(output) == [-99, 3, 6, 9, 12, -99]
    assert (scope.stats().upload_bytes, scope.stats().download_bytes) == (32, 32)
    assert scope.stats().launches == 0, "opaque provider kernel counts must not be fabricated"
    scope.close()
    domain.check(domain_runtime.fort_scope_domain_workspace_release_v1(domain.handle, lease))
    domain.close()


@pytest.mark.parametrize("invalid", ["type", "rank", "duplicate", "undefined"])
def test_external_preflight_rejects_before_callback_and_preserves_prepared_access(domain_runtime, invalid):
    domain = Domain(domain_runtime)
    scope = domain.context()
    host = (c.c_double * 4)(1, 2, 3, 4)
    status, handle = scope.register(host, [4], initialized=invalid != "undefined")
    scope.check(status)
    effects = access(flags=1)
    item = binding(handle, effects)
    bindings, calls = [item], []
    expected = 1
    if invalid == "type":
        item.type = 1
    elif invalid == "rank":
        item.rank = 2
    elif invalid == "duplicate":
        bindings.append(item)
        expected = 3
    else:
        expected = 7

    def worker(_user, _views, _count, _stream, started):
        calls.append(True)
        started[0] = 1
        return 0

    status, report = execute(scope, bindings, worker)
    assert status == expected
    assert (report.started, report.committed) == (0, 0)
    assert calls == []
    assert scope.stats().allocations == 0
    if invalid != "undefined":
        scope.gpu(handle, effects)
        status, report = execute(scope, [binding(handle, effects)], worker)
        assert status == 8
        assert domain_runtime.fort_scope_close(scope.handle) == 8
        scope.gpu_end(handle)
    scope.close()
    domain.close()


def test_external_checked_rejection_grants_no_output_definitions_and_allows_host_recovery(domain_runtime):
    domain = Domain(domain_runtime)
    scope = domain.context()
    host = (c.c_double * 4)(*([-8] * 4))
    status, handle = scope.register(host, [4], initialized=False)
    scope.check(status)

    def reject(_user, _views, _count, _stream, started):
        started[0] = 0
        return 4

    status, report = execute(scope, [binding(handle, access(flags=6))], reject)
    assert status == report.status == 4
    assert (report.started, report.committed) == (0, 0)
    assert domain_runtime.fort_scope_host_begin(scope.handle, handle, c.byref(access(flags=1))) == 7
    scope.cpu_begin(handle, access(flags=6))
    for i in range(4):
        host[i] = 2 * i
    scope.cpu_end(handle)
    scope.close()
    domain.close()
    assert list(host) == [0, 2, 4, 6]


@pytest.mark.parametrize("mode", ["submitted_failure", "missing_marker", "success_without_submission",
                                 "execution_failure", "commit_failure"])
def test_external_uncertain_or_started_failure_poison_domain_without_publication_or_replay(domain_runtime, monkeypatch, mode):
    domain = Domain(domain_runtime)
    scope = domain.context()
    host = (c.c_double * 4)(1, 2, 3, 4)
    status, handle = scope.register(host, [4])
    scope.check(status)
    calls = []

    def fail(_user, views, _count, _stream, started):
        calls.append(True)
        if mode in {"submitted_failure", "commit_failure"}:
            started[0] = 1
            c.cast(views[0].device, c.POINTER(c.c_double))[0] = 1000
            return 4 if mode == "submitted_failure" else 0
        if mode == "missing_marker":
            return 0
        started[0] = 0
        return 0 if mode == "success_without_submission" else 6

    if mode == "commit_failure":
        monkeypatch.setenv("FORT_SCOPE_TEST_FAIL_EXTERNAL_COMMIT", "1")
    status, report = execute(scope, [binding(handle, access(flags=3))], fail)
    assert status == report.status == 6
    assert report.committed == 0
    assert len(calls) == 1
    assert domain_runtime.fort_scope_wait(scope.handle) == 6
    assert domain_runtime.fort_scope_domain_close_v1(domain.handle) == 6
    assert domain_runtime.fort_scope_create_in_domain_v1(domain.handle, c.byref(TOKEN())) == 6
    status, _report = execute(scope, [binding(handle, access(flags=3))], fail)
    assert status == 6
    assert len(calls) == 1
    found = TOKEN()
    scope.check(domain_runtime.fort_scope_domain_get_v1(scope.handle, c.byref(found)))
    assert found.value == domain.handle.value
    assert domain_runtime.fort_scope_domain_abandon_v1(domain.handle) == 8
    scope.check(domain_runtime.fort_scope_abandon(scope.handle))
    domain.check(domain_runtime.fort_scope_domain_abandon_v1(domain.handle))
    assert list(host) == [1, 2, 3, 4]


def test_failed_domain_abandonment_requires_provider_workspace_retirement(domain_runtime):
    domain = Domain(domain_runtime)
    lease = TOKEN()
    domain.check(domain_runtime.fort_scope_domain_workspace_reserve_v1(domain.handle, 16, c.byref(lease)))
    scope = domain.context()
    host = (c.c_double * 2)(1, 2)
    status, handle = scope.register(host, [2])
    scope.check(status)

    def fail(_user, _views, _count, _stream, started):
        started[0] = 1
        return 6

    status, _report = execute(scope, [binding(handle, access(flags=3))], fail,
                              lease=lease.value, workspace=16)
    assert status == 6
    scope.check(domain_runtime.fort_scope_abandon(scope.handle))
    assert domain_runtime.fort_scope_domain_abandon_v1(domain.handle) == 8
    # Release is permitted after failure, once the provider has completed its
    # own destruction. Runtime cleanup cannot pretend it owns provider bytes.
    domain.check(domain_runtime.fort_scope_domain_workspace_release_v1(domain.handle, lease))
    domain.check(domain_runtime.fort_scope_domain_abandon_v1(domain.handle))
    assert list(host) == [1, 2]


def test_empty_domain_and_external_request_never_touch_storage(domain_runtime):
    domain = Domain(domain_runtime)
    scope = domain.context()
    calls = []

    def worker(_user, _views, _count, _stream, _started):
        calls.append(True)
        return 0

    status, report = execute(scope, [], worker)
    scope.check(status)
    assert (report.status, report.started, report.committed) == (0, 0, 0)
    assert calls == []
    assert bytes(scope.stats()) == bytes(Stats())
    scope.close()
    assert bytes(domain.stats()) == bytes(Stats())
    domain.close()


def test_domain_lifecycle_and_physical_totals_keep_explicit_source_attribution(domain_runtime, monkeypatch, capfd):
    monkeypatch.setenv("FORT_RUNTIME_TRACE", "1")
    domain = Domain(domain_runtime)
    scope = domain.context()
    identity = b"a" * 64
    domain_runtime.fort_scope_trace_set_v1(scope.handle, 63, *([identity] * 6), None)
    host = (c.c_double * 2)(1, 2)
    status, handle = scope.register(host, [2])
    scope.check(status)
    device = scope.gpu(handle, access(flags=3))
    device[1] += 10
    scope.gpu_end(handle)
    scope.close()
    assert list(host) == [1, 2]
    domain.close()
    assert list(host) == [1, 12]
    lines = capfd.readouterr().err.splitlines()
    lifecycle = [json.loads(line.removeprefix("FORT_SCOPED domain "))
                 for line in lines if line.startswith("FORT_SCOPED domain ")]
    assert [row["event"] for row in lifecycle] == ["domain_create", "domain_attach", "domain_detach", "domain_retire"]
    assert all(row["schema_version"] == 1 and row["domain"] == domain.handle.value for row in lifecycle)
    assert [row["context"] for row in lifecycle] == [0, scope.handle.value, scope.handle.value, 0]
    assert lifecycle[-1]["physical_context"] == domain.handle.value
    assert lifecycle[-1]["last_context"] == scope.handle.value
    assert lifecycle[-1]["owner"] == identity.decode()
    assert lifecycle[-1]["boundary"] == identity.decode()
    evidence = [json.loads(line.removeprefix("FORT_SCOPED evidence "))
                for line in lines if line.startswith("FORT_SCOPED evidence ")]
    assert all(row["domain_version"] == 1 and row["domain"] == domain.handle.value for row in evidence)
    completion = [row for row in evidence if row["event"] == "completion_statistics"]
    assert len(completion) == 1
    assert completion[0]["domain_role"] == "storage"
    assert completion[0]["context"] == domain.handle.value
    assert all(row["domain_role"] == "execution" for row in evidence
               if row["event"] in {"graph_statistics", "command_queue_statistics"})
    raw = [line for line in lines if line.startswith("FORT_SCOPED ")
           and not line.startswith(("FORT_SCOPED domain ", "FORT_SCOPED evidence "))]
    assert all("domain_version=1" in line and f"domain={domain.handle.value} " in line for line in raw)
    closes = [line for line in raw if line.startswith("FORT_SCOPED close ")]
    assert len(closes) == 1
    assert f"context={domain.handle.value} " in closes[0]
    assert "domain_role=storage" in closes[0]
    assert "owner=" + identity.decode() in closes[0]


def test_detached_domain_vetoes_contextless_native_hint_lookup_and_learning(domain_runtime):
    words = (TOKEN * 1)(0xD04A19)

    def begin():
        observation, hit = Hint(), c.c_int(-1)
        assert domain_runtime.fort_scope_native_hint_begin_v1(words, 1, c.byref(observation), c.byref(hit)) == 0
        return observation, hit.value

    observation, hit = begin()
    assert (observation.eligible, hit) == (1, 0)
    domain = Domain(domain_runtime)
    scope = domain.context()
    scope.close()
    # No live execution context remains, but the domain can still own dirty
    # storage: neither lookup bypass nor a learned observation is permitted.
    live, hit = begin()
    assert (live.eligible, hit) == (0, 0)
    assert domain_runtime.fort_scope_native_hint_finish_v1(c.byref(observation)) == 0
    domain.close()
    observation, hit = begin()
    assert (observation.eligible, hit) == (1, 0)
    assert domain_runtime.fort_scope_native_hint_finish_v1(c.byref(observation)) == 0
    observation, hit = begin()
    assert (observation.eligible, hit) == (0, 1)
    domain = Domain(domain_runtime)
    observation, hit = begin()
    assert (observation.eligible, hit) == (0, 0)
    domain.close()


def test_cpp_callback_exception_is_contained_and_cannot_publish_or_replay(domain_runtime, tmp_path):
    source = tmp_path / "throws.cpp"
    source.write_text(r'''
#include "scoped_runtime.h"
#include <cassert>
#include <stdexcept>
int worker(void *,const fort_scope_external_view_v1 *,size_t,void *,int *) {
    throw std::runtime_error("provider exception");
}
int main() {
    fort_scope_domain_t domain=0; fort_scope_t context=0; fort_buffer_t buffer=0;
    assert(!fort_scope_domain_create_v1(0,&domain));
    assert(!fort_scope_create_in_domain_v1(domain,&context));
    double host[2]={3,5}; size_t extents[1]={2}; int64_t lower[1]={-1};
    fort_scope_layout layout{1,FORT_SCOPE_REAL64,8,host,extents,lower,1};
    assert(!fort_scope_register(context,1,1,&layout,1,&buffer));
    fort_scope_access access{}; access.flags=FORT_SCOPE_READ_ALL|FORT_SCOPE_WRITE_ALL;
    fort_scope_external_binding_v1 binding{buffer,FORT_SCOPE_REAL64,1,8,access};
    fort_scope_external_call_v1 call{1,0,&binding,1,0,0}; fort_scope_external_report_v1 report{};
    assert(fort_scope_external_execute_v1(context,&call,worker,nullptr,&report)==FORT_SCOPE_EXECUTION);
    assert(report.started==1 && !report.committed);
    assert(fort_scope_close(context)==FORT_SCOPE_EXECUTION);
    assert(!fort_scope_abandon(context)); assert(!fort_scope_domain_abandon_v1(domain));
    assert(host[0]==3 && host[1]==5);
}
''')
    target = tmp_path / "throws"
    compiler = shutil.which("g++-14") or shutil.which("g++")
    command = [compiler, "-std=c++17", "-I", str(RUNTIME), str(source), domain_runtime._name,
               "-o", str(target)]
    # The already-built runtime is passed as an absolute DT_NEEDED path, so
    # this client does not compile another implementation or select by PATH.
    built = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert built.returncode == 0, built.stderr
    run = subprocess.run([str(target)], capture_output=True, text=True, timeout=10)
    assert run.returncode == 0, run.stdout + run.stderr
