"""Diagnostic provenance is separate from numerical and coherence authority."""
# ruff: noqa: F811 - pytest discovers imported fixtures under these names.

import ctypes as c
import re

import pytest

from compiler.emission.common.resources import read_scoped_runtime
from compiler.scopes.provenance import Provenance
from compiler.tests.test_scoped_planning_runtime import costs, planning, select  # noqa: F401
from compiler.tests.test_scoped_planning_runtime import record as record_plan
from compiler.tests.test_scoped_runtime import TOKEN, Scope, access, runtime  # noqa: F401


class State(c.Structure):
    _fields_ = [("version", c.c_uint32), ("reserved", c.c_uint32),
                *[(name, c.c_char * 65) for name in
                  ("owner", "procedure", "segment", "operation", "implementation", "boundary")]]


def api(library):
    library.fort_scope_trace_set_v1.argtypes = [TOKEN, c.c_uint32, *[c.c_char_p]*6, c.POINTER(State)]
    library.fort_scope_trace_set_v1.restype = None
    library.fort_scope_trace_restore_v1.argtypes = [TOKEN, c.POINTER(State)]
    library.fort_scope_trace_restore_v1.restype = None
    return library


def events(text):
    return [(match[1], dict(re.findall(r"(\w+)=([^\s]+)", match[2]))) for line in text.splitlines()
            if (match := re.match(r"FORT_SCOPED (\w+) (.*)", line)) and match[1] != "evidence"]


def set_position(lib, context, mask=63, *, previous=None, **fields):
    values = [fields.get(name, b"") for name in
              ("owner", "procedure", "segment", "operation", "implementation", "boundary")]
    lib.fort_scope_trace_set_v1(context, mask, *values, c.byref(previous) if previous is not None else None)


@pytest.mark.native
def test_disabled_and_malformed_diagnostics_do_not_touch_numerical_error_or_lookup(runtime, monkeypatch, capfd):
    lib = api(runtime)
    monkeypatch.delenv("FORT_RUNTIME_TRACE", raising=False)
    scope = Scope(lib)
    assert lib.fort_scope_layout_get(scope.handle, 999999, None) != 0
    original_error = lib.fort_scope_error()
    state = State(version=123)
    set_position(lib, 999999, previous=state, owner=b"a"*64)
    assert state.version == 0
    lib.fort_scope_trace_restore_v1(999999, c.byref(state))
    assert lib.fort_scope_error() == original_error
    assert not capfd.readouterr().err
    monkeypatch.setenv("FORT_RUNTIME_TRACE", "1")
    set_position(lib, 999999, previous=state, owner=b"a"*64)
    assert state.version == 0
    assert lib.fort_scope_error() == original_error
    # Invalid masks never authorize or mutate provenance.
    set_position(lib, scope.handle, 64, previous=state, owner=b"a"*64)
    assert state.version == 0
    set_position(lib, scope.handle, owner=b"a"*64, procedure=b"bad\nfield=alias", segment=b"z"*64,
                 implementation=b"b"*65)
    scope.close()
    records = events(capfd.readouterr().err)
    assert records[-1][0] == "close"
    assert records[-1][1]["owner"] == "a"*64
    assert records[-1][1]["procedure"] == "unknown"
    assert records[-1][1]["segment"] == "unknown"
    assert records[-1][1]["implementation"] == "unknown"


@pytest.mark.native
def test_actual_events_have_unique_handles_and_restore_borrowed_positions(runtime, monkeypatch, capfd):
    lib = api(runtime)
    monkeypatch.setenv("FORT_RUNTIME_TRACE", "1")
    handles, buffers = [], []
    for _invocation in range(2):
        scope = Scope(lib)
        handles.append(scope.handle.value)
        host = (c.c_double*4)(1, 2, 3, 4)
        status, buffer = scope.register(host, (4,), identity=1)
        scope.check(status)
        buffers.append(buffer.value)
        set_position(lib, scope.handle, owner=b"a"*64, procedure=b"b"*64, segment=b"c"*64,
                     operation=b"d"*64)
        saved = State()
        set_position(lib, scope.handle, 2+8+16, previous=saved, procedure=b"e"*64,
                     operation=b"f"*64, implementation=b"1"*64)
        pointer = scope.gpu(buffer, access(flags=1+2))
        pointer[0] = 9
        scope.check(lib.fort_scope_note_launch(scope.handle))
        scope.gpu_end(buffer)
        lib.fort_scope_trace_restore_v1(scope.handle, c.byref(saved))
        scope.cpu_begin(buffer, access(read=[((0,), (1,))]))
        scope.cpu_end(buffer)
        assert host[0] == 9
        set_position(lib, scope.handle, 32, boundary=b"2"*64)
        before = scope.stats()
        set_position(lib, scope.handle, 0, previous=State())
        after = scope.stats()
        assert bytes(before) == bytes(after)
        scope.close()
        # A child closing ownership makes later restoration harmless.
        lib.fort_scope_trace_restore_v1(scope.handle, c.byref(saved))
    assert len(set(handles)) == len(set(buffers)) == 2
    rows = events(capfd.readouterr().err)
    for operation, record in rows:
        assert record["provenance_version"] == "1"
        assert int(record["context"]) in handles
        assert record["bytes"].isdigit()
        if operation in {"upload", "download", "device_commit", "host_commit", "release"}:
            assert record["buffer"] == "1"
            assert int(record["buffer_handle"]) in buffers
    launches = [record for operation, record in rows if operation == "launch"]
    assert len(launches) == 2
    assert all(record["owner"] == "a"*64 and record["segment"] == "c"*64 and
               record["procedure"] == "e"*64 and record["implementation"] == "1"*64 for record in launches)
    downloads = [record for operation, record in rows if operation == "download"]
    assert all(record["procedure"] == "b"*64 and record["operation"] == "d"*64 and
               record["implementation"] == "unknown" for record in downloads)
    closes = [record for operation, record in rows if operation == "close"]
    assert all(record["boundary"] == "2"*64 for record in closes)


@pytest.mark.native
def test_restore_explicit_unknown_state_and_masked_clear(runtime, monkeypatch, capfd):
    lib = api(runtime)
    monkeypatch.setenv("FORT_RUNTIME_TRACE", "1")
    scope = Scope(lib)
    original = State()
    set_position(lib, scope.handle, previous=original, owner=b"a"*64, segment=b"b"*64)
    set_position(lib, scope.handle, 1, owner=b"")
    set_position(lib, scope.handle, 0, previous=State())
    lib.fort_scope_trace_restore_v1(scope.handle, c.byref(original))
    scope.close()
    rows = events(capfd.readouterr().err)
    assert rows[-1][1]["owner"] == rows[-1][1]["segment"] == "unknown"


@pytest.mark.native
def test_diagnostics_preserve_definition_validation_and_preview_caches(planning, monkeypatch, capfd):
    lib = api(planning)
    monkeypatch.setenv("FORT_RUNTIME_TRACE", "1")
    scope = Scope(lib)
    values = (c.c_double * 32)(*range(32))
    status, handle = scope.register(values, [32])
    scope.check(status)
    scope.check(lib.fort_scope_plan_reset(scope.handle))
    record_plan(scope, 71, handle, access(flags=3))
    scope.check(lib.fort_scope_plan_validate(scope.handle))
    before = select(scope, costs(), -1)
    first = events(capfd.readouterr().err)
    first_timing = [row for operation, row in first if operation == "planning_timing"][-1]
    previous = State()
    set_position(lib, scope.handle, previous=previous, owner=b"a"*64)
    lib.fort_scope_trace_restore_v1(scope.handle, c.byref(previous))
    scope.check(lib.fort_scope_plan_validate(scope.handle))
    after = select(scope, costs(), -1)
    second = events(capfd.readouterr().err)
    second_timing = [row for operation, row in second if operation == "planning_timing"][-1]
    assert second_timing["cache_hit"] == "1"
    for generation in ("query_generation", "state_generation"):
        assert first_timing[generation] == second_timing[generation]
    assert bytes(before) == bytes(after)
    assert scope.stats().uploads == scope.stats().downloads == scope.stats().launches == 0
    scope.close()


def test_public_identity_mapping_is_full_hash_and_runtime_identity_changes():
    provenance = Provenance()
    identity = provenance.add("owner", procedure="renamed::worker", source="input.f90", summary_identity="abc")
    assert re.fullmatch("[0-9a-f]{64}", identity)
    assert identity == provenance.add("owner", summary_identity="abc", source="input.f90", procedure="renamed::worker")
    assert provenance.public()["records"][0]["id"] == identity
    _, public = read_scoped_runtime()
    assert public["runtime_provenance"]["schema_version"] == 1
    assert public["runtime_provenance"]["set"] == "fort_scope_trace_set_v1"


def test_batch_callback_does_not_reenter_diagnostic_context_apis(tmp_path):
    from compiler.tests.test_scoped_batch_sources import numerical
    emitted = numerical(tmp_path)
    batch = emitted.report["batch_execution"]
    assert batch["enabled"]
    worker = emitted.cuda.split('extern "C" int '+batch["window_entry"], 1)[1].split("static int", 1)[0]
    assert "TracePosition" not in worker
    assert "fort_scope_trace_" not in worker
    assert emitted.report["runtime_provenance"]["batch_worker_attribution"].startswith("unavailable")


def test_collective_worker_tags_are_inside_existing_master_coordinators(tmp_path):
    from compiler.tests.test_scoped_team_sources import emit, team_text
    emitted = emit(tmp_path)
    team = team_text(emitted)
    assert "TracePosition" in team
    for occurrence in re.finditer("TracePosition", team):
        prefix = team[:occurrence.start()]
        # No thread-local guard may restore shared attribution while the master
        # is still preparing copies or launching a kernel.
        assert prefix.rfind("#pragma omp master") > prefix.rfind("#pragma omp barrier")


def test_borrowed_call_iso_imports_do_not_shadow_original_actuals(tmp_path):
    from compiler.tests.test_lexical_source_owner import MODULE_SOURCE, emit
    source = MODULE_SOURCE.replace("step(a,b,out,n,escape)", "step(a,b,out,n,escape,c_null_char)").replace(
        "integer,intent(in)::n\nlogical,intent(in)::escape\ninteger::i",
        "integer,intent(in)::n,c_null_char\nlogical,intent(in)::escape\ninteger::i").replace(
        "call adjust(b,n,escape)", "call adjust(b,c_null_char,escape)")
    _path, outputs, report = emit(tmp_path, source)
    assert report["scope_count"] == 1
    text = "\n".join(outputs.values())
    assert re.search(r"fort_trace_i32_[0-9a-f]{12} => c_int32_t", text)
    assert re.search(r"fort_trace_nul_[0-9a-f]{12} => c_null_char", text)
    assert "adjust(b,c_null_char,escape)" in text.replace(" ", "").lower()


def test_borrowed_state_does_not_shadow_an_imported_original_actual(tmp_path):
    from compiler.tests.test_lexical_source_owner import MODULE_SOURCE, emit
    source = ("module original_data\ninteger::fort_parent_trace=7\nend module\n" + MODULE_SOURCE.replace(
        "module local_owner\n", "module local_owner\nuse original_data,only:fort_parent_trace\n").replace(
        "call adjust(b,n,escape)", "call adjust(b,fort_parent_trace,escape)"))
    _path, outputs, report = emit(tmp_path, source)
    assert report["scope_count"] == 1
    text = "\n".join(outputs.values())
    assert "adjust(b,fort_parent_trace,escape)" in text.replace(" ", "").lower()
    assert re.search(r"target :: fort_trace_saved_[0-9a-f]{12}", text)
    assert "target :: fort_parent_trace" not in text


def test_diagnostic_helper_collision_declines_before_original_call_emission(monkeypatch):
    from types import SimpleNamespace

    from compiler.ir import CompilationError
    from compiler.scopes.provenance import call_frame
    identity = "a" * 64
    analysis = SimpleNamespace(_binding=lambda _scope, name: name == "fort_trace_saved_"+identity[:12],
                               _candidates=lambda _scope, _name: [])
    builder = SimpleNamespace(entry=SimpleNamespace(scope=object()), analysis=analysis)
    with pytest.raises(CompilationError, match="diagnostic helper name"):
        call_frame(builder, "fort_context", identity, ["call original()"])
