"""Task acquisition validates every effect before publishing or preparing any resource."""

from __future__ import annotations

import ctypes as c
import shutil
import subprocess

import pytest

from compiler.tests.test_scoped_runtime import RUNTIME, SIZE, TOKEN, Access, Layout, Scope, Stats, access
from compiler.tests.test_scoped_views import View, view


class Binding(c.Structure):
    _fields_ = [("buffer", TOKEN), ("access", Access)]


@pytest.fixture(scope="module")
def acquisition(tmp_path_factory):
    directory = tmp_path_factory.mktemp("task_acquisition")
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("g++ unavailable")
    glue = directory / "entry.cpp"
    glue.write_text('''#include "view_entry.hpp"
extern "C" int entry_read(fort_scope_t context, const fort_buffer_t *handles, size_t count) {
    fort_scoped::AccessBatch<128> task(context);
    fort_scope_access reads{}; reads.flags=FORT_SCOPE_READ_ALL;
    for(size_t i=0;i<count;++i) { const int status=task.add(handles[i],reads); if(status) return status; }
    const int status=task.begin(false); if(status) return status;
    task.executing();
    return task.finish();
}
extern "C" int entry_rebegin(fort_scope_t context, fort_buffer_t handle) {
    fort_scoped::AccessBatch<1> task(context);
    fort_scope_access reads{}; reads.flags=FORT_SCOPE_READ_ALL;
    int status=task.add(handle,reads); if(status) return status;
    status=task.begin(false); if(status) return status;
    if(task.begin(false)!=FORT_SCOPE_STATE) return -1;
    if(task.add(handle,reads)!=FORT_SCOPE_STATE) return -2;
    task.executing();
    return task.finish();
}
extern "C" int entry_cancel(fort_scope_t context, fort_buffer_t handle) {
    fort_scoped::AccessBatch<1> task(context);
    fort_scope_access write{}; write.flags=FORT_SCOPE_WRITE_ALL|FORT_SCOPE_OVERWRITE_ALL;
    int status=task.add(handle,write); if(status) return status;
    status=task.begin(false); if(status) return status;
    task.cancel();
    // Reuse after cancellation must not retain the previous prepared flag.
    status=task.begin(false); if(status) return status;
    return FORT_SCOPE_OK; // Destructor cancels this second acquisition.
}
extern "C" int entry_export(fort_scope_t context, fort_buffer_t handle, fort_scope_plan_binding *out,
                             size_t capacity,size_t *count) {
    fort_scoped::AccessBatch<2> task(context);
    fort_scope_access reads{}; reads.flags=FORT_SCOPE_READ_ALL;
    int status=task.add(handle,reads); if(status) return status;
    status=task.add(handle,reads); if(status) return status;
    return task.export_bindings(out,capacity,count);
}
extern "C" int entry_view_export(fort_scope_t context,const fort_scope_view_v1 *first,
                                  const fort_scope_view_v1 *second,size_t *coordinates) {
    fort_scoped::ViewAccessBatch<2> task(context);
    fort_scope_access reads{}; reads.flags=FORT_SCOPE_READ_ALL;
    int status=task.add(first,reads); if(status) return status;
    status=task.add(second,reads); if(status) return status;
    fort_scope_plan_binding bindings[2]{}; size_t count=0;
    status=task.export_bindings(bindings,2,&count); if(status) return status;
    if(count!=1 || bindings[0].access.read_count!=2) return -1;
    for(size_t i=0;i<2;++i) {
        coordinates[2*i]=bindings[0].access.reads[i].lower[0];
        coordinates[2*i+1]=bindings[0].access.reads[i].upper[0];
    }
    return task.add(first,reads)==FORT_SCOPE_STATE ? FORT_SCOPE_OK : -2;
}
''')
    target = directory / "runtime.so"
    result = subprocess.run([
        compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-fPIC", "-shared", "-pthread",
        "-DFORT_SCOPE_CPU_TEST", "-DFORT_SCOPE_TEST_FAULTS", "-I", str(RUNTIME),
        "-x", "c++", str(RUNTIME / "scoped_runtime.cu"), str(glue), "-o", str(target),
    ], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    lib = c.CDLL(str(target))
    lib.fort_scope_error.restype = c.c_char_p
    signatures = {
        "create": [c.c_int, c.POINTER(TOKEN)],
        "register": [TOKEN, TOKEN, TOKEN, c.POINTER(Layout), c.c_int, c.POINTER(TOKEN)],
        "set_device_budget": [TOKEN, SIZE],
        "device_begin": [TOKEN, TOKEN, c.POINTER(Access), c.POINTER(c.c_void_p)],
        "host_begin": [TOKEN, TOKEN, c.POINTER(Access)],
        "device_end": [TOKEN, TOKEN], "host_end": [TOKEN, TOKEN],
        "cancel_access": [TOKEN, TOKEN], "unregister": [TOKEN, TOKEN],
        "close": [TOKEN], "abandon": [TOKEN], "wait": [TOKEN],
        "stats_get": [TOKEN, c.POINTER(Stats)],
        "access_begin_v1": [TOKEN, c.POINTER(Binding), SIZE, c.c_int, c.POINTER(c.c_void_p)],
    }
    for name, signature in signatures.items():
        getattr(lib, "fort_scope_" + name).argtypes = signature
    lib.entry_read.argtypes = [TOKEN, c.POINTER(TOKEN), SIZE]
    lib.entry_rebegin.argtypes = lib.entry_cancel.argtypes = [TOKEN, TOKEN]
    lib.entry_export.argtypes = [TOKEN, TOKEN, c.POINTER(Binding), SIZE, c.POINTER(SIZE)]
    lib.entry_view_export.argtypes = [TOKEN, c.POINTER(View), c.POINTER(View), c.POINTER(SIZE)]
    return lib


def pack(*bindings):
    return (Binding * len(bindings))(*(Binding(handle, effects) for handle, effects in bindings))


def begin(scope, bindings, *, device=False, outputs=None):
    return scope.lib.fort_scope_access_begin_v1(scope.handle, bindings, len(bindings), device, outputs)


def register(scope, identity, *, initialized=True, count=8):
    values = (c.c_double * count)(*(identity * 10 + i for i in range(count)))
    status, handle = scope.register(values, [count], identity=identity, initialized=initialized)
    scope.check(status)
    return values, handle


@pytest.mark.native
def test_entry_downloads_all_resources_before_one_host_wait(acquisition):
    scope = Scope(acquisition)
    arrays = [register(scope, identity, initialized=False) for identity in range(1, 6)]
    bindings = pack(*[(handle, access(flags=6)) for _, handle in arrays])
    pointers = (c.c_void_p * len(arrays))()
    scope.check(begin(scope, bindings, device=True, outputs=pointers))
    for identity, ((_, handle), pointer) in enumerate(zip(arrays, pointers, strict=True), 1):
        device = c.cast(pointer, c.POINTER(c.c_double))
        for i in range(8):
            device[i] = identity * 100 + i
        scope.gpu_end(handle)
    before = scope.stats()
    handles = (TOKEN * len(arrays))(*(handle.value for _, handle in arrays))
    scope.check(acquisition.entry_read(scope.handle, handles, len(handles)))
    after = scope.stats()
    assert after.downloads - before.downloads == len(arrays)
    assert after.waits - before.waits == 1
    assert [list(values) for values, _ in arrays] == [[identity * 100 + i for i in range(8)]
                                                     for identity in range(1, 6)]
    scope.close()


@pytest.mark.native
@pytest.mark.parametrize("device", [False, True])
@pytest.mark.parametrize("failure", ["stale", "effects", "undefined", "duplicate"])
def test_invalid_late_binding_changes_no_resource(acquisition, device, failure):
    scope = Scope(acquisition)
    values, first = register(scope, 1)
    _, second = register(scope, 2, initialized=False)
    pointer = scope.gpu(first, access(flags=3))
    pointer[0] = 987
    scope.gpu_end(first)
    first_read = access(flags=1)
    if failure == "stale":
        final = (TOKEN(2**63), access(flags=1))
        expected = 2
    elif failure == "effects":
        final = (second, access(write=[([0], [9])]))
        expected = 1
    elif failure == "undefined":
        final = (second, access(flags=1))
        expected = 7
    else:
        final = (first, access(flags=1))
        expected = 3
    bindings = pack((first, first_read), final)
    outputs = (c.c_void_p * 2)(123, 456)
    before = bytes(scope.stats())
    assert begin(scope, bindings, device=device, outputs=outputs) == expected
    assert bytes(scope.stats()) == before
    assert list(outputs) == [123, 456]
    assert values[0] == 10  # The first resource was not prematurely published.
    scope.cpu_begin(first, first_read)  # No prepared access survives the error.
    assert values[0] == 987
    scope.cpu_end(first)
    assert acquisition.fort_scope_host_begin(scope.handle, second, c.byref(access(flags=1))) == 7
    scope.close()


@pytest.mark.native
def test_device_allocation_failure_retains_no_prepared_or_written_state(acquisition, monkeypatch):
    scope = Scope(acquisition)
    _, first = register(scope, 1, initialized=False)
    _, second = register(scope, 2, initialized=False)
    bindings = pack((first, access(flags=6)), (second, access(flags=6)))
    outputs = (c.c_void_p * 2)(123, 456)
    monkeypatch.setenv("FORT_SCOPE_TEST_FAIL_ALLOC_AFTER", "1")
    assert begin(scope, bindings, device=True, outputs=outputs) == 4
    assert list(outputs) == [123, 456]
    assert scope.stats().uploads == 0
    assert scope.stats().allocations == 1
    for handle in (first, second):
        assert acquisition.fort_scope_host_begin(scope.handle, handle, c.byref(access(flags=1))) == 7
        scope.cpu_begin(handle, access(flags=6))
        scope.check(acquisition.fort_scope_cancel_access(scope.handle, handle))
    scope.close()


@pytest.mark.native
def test_aggregate_budget_failure_allocates_nothing(acquisition):
    scope = Scope(acquisition)
    _, first = register(scope, 1)
    _, second = register(scope, 2)
    scope.check(acquisition.fort_scope_set_device_budget(scope.handle, 64))
    outputs = (c.c_void_p * 2)(123, 456)
    assert begin(scope, pack((first, access(flags=1)), (second, access(flags=1))),
                 device=True, outputs=outputs) == 4
    assert list(outputs) == [123, 456]
    assert scope.stats().allocations == scope.stats().upload_bytes == 0
    scope.close()


@pytest.mark.native
def test_cancellation_preserves_partial_definedness_and_maywrite_values(acquisition):
    scope = Scope(acquisition)
    values, first = register(scope, 1, initialized=False)
    _, second = register(scope, 2, initialized=False)
    middle = [([2], [6])]
    pointer = scope.gpu(first, access(write=middle, overwrite=middle))
    for i in range(2, 6):
        pointer[i] = 1000 + i
    scope.gpu_end(first)
    # May-write without overwrite must bring existing values to the host.
    bindings = pack((first, access(write=middle)), (second, access(flags=6)))
    scope.check(begin(scope, bindings))
    assert list(values)[2:6] == [1002, 1003, 1004, 1005]
    for handle in (first, second):
        scope.check(acquisition.fort_scope_cancel_access(scope.handle, handle))
    assert acquisition.fort_scope_host_begin(scope.handle, first, c.byref(access(flags=1))) == 7
    assert acquisition.fort_scope_host_begin(scope.handle, second, c.byref(access(flags=1))) == 7
    # Cancelled writes leave the GPU's latest values valid.
    pointer = scope.gpu(first, access(read=middle))
    assert [pointer[i] for i in range(2, 6)] == [1002, 1003, 1004, 1005]
    scope.gpu_end(first)
    scope.check(acquisition.entry_cancel(scope.handle, second))
    assert acquisition.fort_scope_host_begin(scope.handle, second, c.byref(access(flags=1))) == 7
    scope.close()


@pytest.mark.native
def test_empty_task_does_not_wait_and_host_overwrite_fences_pending_upload(acquisition, monkeypatch):
    scope = Scope(acquisition)
    values, handle = register(scope, 1)
    pointer = scope.gpu(handle, access(flags=1))
    scope.gpu_end(handle)
    before = scope.stats().waits
    monkeypatch.setenv("FORT_SCOPE_TEST_FAIL_WAIT", "1")
    scope.check(acquisition.fort_scope_access_begin_v1(scope.handle, None, 0, 0, None))
    assert scope.stats().waits == before
    monkeypatch.delenv("FORT_SCOPE_TEST_FAIL_WAIT")
    scope.check(begin(scope, pack((handle, access(flags=6)))))
    assert scope.stats().waits == before + 1
    values[0] = 555
    scope.cpu_end(handle)
    pointer = scope.gpu(handle, access(flags=1))
    assert pointer[0] == 555
    scope.gpu_end(handle)
    scope.close()


@pytest.mark.native
def test_wait_failure_poisons_task_without_enabling_replay(acquisition, monkeypatch):
    scope = Scope(acquisition)
    arrays = [register(scope, identity) for identity in (1, 2)]
    for _, handle in arrays:
        pointer = scope.gpu(handle, access(flags=3))
        pointer[0] += 100
        scope.gpu_end(handle)
    monkeypatch.setenv("FORT_SCOPE_TEST_FAIL_WAIT", "1")
    bindings = pack(*[(handle, access(flags=1)) for _, handle in arrays])
    assert begin(scope, bindings) == 6
    assert begin(scope, bindings) == 6
    assert acquisition.fort_scope_close(scope.handle) == 6
    scope.check(acquisition.fort_scope_abandon(scope.handle))


@pytest.mark.native
def test_entry_read_aliases_and_repeated_begin_keep_valid_acquisition(acquisition):
    scope = Scope(acquisition)
    _, handle = register(scope, 1)
    aliases = (TOKEN * 3)(handle.value, handle.value, handle.value)
    scope.check(acquisition.entry_read(scope.handle, aliases, 3))
    scope.check(acquisition.entry_rebegin(scope.handle, handle))
    assert scope.stats().allocations == scope.stats().waits == 0
    scope.close()


@pytest.mark.native
def test_validation_failure_preserves_an_existing_foreign_acquisition(acquisition):
    scope = Scope(acquisition)
    _, first = register(scope, 1)
    _, second = register(scope, 2)
    scope.cpu_begin(first, access(flags=1))
    assert begin(scope, pack((second, access(flags=1)), (first, access(flags=1)))) == 8
    # The transaction owns neither acquisition: the existing one remains
    # valid, and the earlier staged second resource was never installed.
    scope.cpu_end(first)
    scope.cpu_begin(second, access(flags=1))
    scope.cpu_end(second)
    scope.close()


@pytest.mark.native
def test_retired_generation_is_rejected_before_any_transfer(acquisition):
    scope = Scope(acquisition)
    _, first = register(scope, 1)
    old_values, old = register(scope, 2)
    scope.check(acquisition.fort_scope_unregister(scope.handle, old))
    status, current = scope.register(old_values, [8], identity=2, generation=2)
    scope.check(status)
    before = bytes(scope.stats())
    outputs = (c.c_void_p * 2)()
    assert begin(scope, pack((first, access(flags=1)), (old, access(flags=1))),
                 device=True, outputs=outputs) == 2
    assert bytes(scope.stats()) == before
    scope.check(begin(scope, pack((first, access(flags=1)), (current, access(flags=1))),
                      device=True, outputs=outputs))
    scope.gpu_end(first)
    scope.gpu_end(current)
    scope.close()


@pytest.mark.native
def test_zero_size_resources_and_empty_tasks_do_not_allocate_or_wait(acquisition, monkeypatch):
    scope = Scope(acquisition)
    status, handle = scope.register(None, [0], initialized=False)
    scope.check(status)
    monkeypatch.setenv("FORT_SCOPE_TEST_FAIL_ALLOC", "1")
    scope.check(acquisition.fort_scope_access_begin_v1(scope.handle, None, 0, 1, None))
    outputs = (c.c_void_p * 1)(123)
    scope.check(begin(scope, pack((handle, access(flags=7))), device=True, outputs=outputs))
    assert outputs[0] is None
    scope.gpu_end(handle)
    scope.check(begin(scope, pack((handle, access(flags=1)))))
    scope.cpu_end(handle)
    assert bytes(scope.stats()) == bytes(Stats())
    scope.close()


@pytest.mark.native
def test_exported_bindings_are_canonical_and_view_sections_remain_exact(acquisition):
    scope = Scope(acquisition)
    _, handle = register(scope, 1)
    bindings = (Binding * 2)()
    count = SIZE(99)
    assert acquisition.entry_export(scope.handle, handle, bindings, 0, c.byref(count)) == 1
    assert count.value == 99
    scope.check(acquisition.entry_export(scope.handle, handle, bindings, 2, c.byref(count)))
    assert count.value == 1
    assert bindings[0].buffer == handle.value
    assert bindings[0].access.flags == 1
    first, second = view(handle, [0], [2]), view(handle, [4], [2])
    coordinates = (SIZE * 4)()
    scope.check(acquisition.entry_view_export(scope.handle, c.byref(first), c.byref(second), coordinates))
    assert list(coordinates) == [0, 2, 4, 6]
    assert scope.stats().allocations == scope.stats().waits == 0
    scope.close()
