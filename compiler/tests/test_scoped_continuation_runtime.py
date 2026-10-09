"""Public continuation queries retain ownership across GPU and native segments."""
# ruff: noqa: F811 - pytest discovers imported fixtures.
from __future__ import annotations

import ctypes as c
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from compiler.tests.test_scoped_planning_runtime import (
    Binding,  # noqa: F401
    Costs,
    Decision,
    costs,
    planning,  # noqa: F401 - imported pytest fixture.
    record,
    select,
)
from compiler.tests.test_scoped_runtime import TOKEN, Scope, access, runtime  # noqa: F401


class Terminal(c.Structure):
    _fields_ = [("seconds", c.c_double)] + [(name, TOKEN) for name in (
        "download_bytes", "downloads", "waits", "releases")]


class Report(c.Structure):
    _fields_ = [(name, c.c_uint32) for name in ("version", "endpoint_mode", "available", "owner_available")] + [
        ("execution_seconds", c.c_double), ("native_execution_seconds", c.c_double),
        ("entry_terminal", Terminal), ("terminal", Terminal), ("native_terminal", Terminal),
        ("ranking_seconds", c.c_double), ("native_ranking_seconds", c.c_double), ("owner_segments", TOKEN),
        ("owner_execution_seconds", c.c_double), ("owner_terminal_seconds", c.c_double),
        ("owner_complete_seconds", c.c_double), ("native_common_compute_excluded", c.c_uint32)]


@pytest.fixture
def continuation(planning):
    planning.fort_scope_plan_reset_mode.argtypes = [TOKEN, c.c_uint32]
    planning.fort_scope_plan_report_v2.argtypes = [TOKEN, c.POINTER(Report)]
    return planning


def report(scope):
    result = Report()
    scope.check(scope.lib.fort_scope_plan_report_v2(scope.handle, c.byref(result)))
    assert result.version == 2
    assert result.endpoint_mode == 1
    return result


def begin(scope):
    scope.check(scope.lib.fort_scope_plan_reset_mode(scope.handle, 1))


def choose(scope, calibration=None, compatible=1):
    scope.check(scope.lib.fort_scope_plan_validate(scope.handle))
    return select(scope, costs() if calibration is None else calibration, compatible)


def consume(scope, unit, bindings, expected):
    chosen = c.c_int(-1)
    scope.check(scope.lib.fort_scope_plan_next(scope.handle, unit, bindings, len(bindings), c.byref(chosen)))
    assert chosen.value == expected


def gpu(scope, handle, effects, update):
    pointer = scope.gpu(handle, effects)
    for index in range(8):
        pointer[index] = update(pointer[index], index)
    scope.check(scope.lib.fort_scope_note_launch(scope.handle))
    scope.gpu_end(handle)


def native_record(scope, unit, handle, effects, *, flops=1):
    bindings = (Binding * 1)(Binding(handle, effects))
    scope.check(scope.lib.fort_scope_plan_add(scope.handle, 1, unit, bindings, 1, flops, 8, 0))
    return bindings


def test_three_segments_keep_device_copy_through_native_mirror(continuation, monkeypatch, capfd):
    monkeypatch.setenv("FORT_RUNTIME_TRACE", "1")
    scope = Scope(continuation)
    values = (c.c_double * 8)(*range(8))
    status, handle = scope.register(values, [8], lower=[-7])
    scope.check(status)
    rw = access(flags=3)
    executions = []
    begin(scope)
    bindings = record(scope, 11, handle, rw)
    assert choose(scope).gpu_units == 1
    first = report(scope)
    assert first.terminal.download_bytes == 64
    assert first.terminal.releases == 1
    assert first.owner_segments == first.owner_available == 0
    assert scope.stats().downloads == scope.stats().allocations == 0
    consume(scope, 11, bindings, 1)
    gpu(scope, handle, rw, lambda value, _: 2 * value)
    scope.check(continuation.fort_scope_wait(scope.handle))
    first = report(scope)
    assert first.owner_segments == first.owner_available == 1
    executions.append(first.execution_seconds)

    begin(scope)
    mirror = access(read=[([1], [3])])
    bindings = native_record(scope, 12, handle, mirror)
    decision = choose(scope)
    assert decision.gpu_units == 0
    assert decision.cpu_units == 1
    second = report(scope)
    assert second.entry_terminal.download_bytes == 64
    assert decision.download_bytes == 16
    assert second.terminal.download_bytes == 48
    consume(scope, 12, bindings, 0)
    scope.cpu_begin(handle, mirror)
    assert values[1] + values[2] == 6
    scope.cpu_end(handle)
    scope.check(continuation.fort_scope_wait(scope.handle))
    second = report(scope)
    assert second.owner_segments == 2
    assert second.owner_available
    executions.append(second.execution_seconds)

    begin(scope)
    bindings = record(scope, 13, handle, rw)
    decision = choose(scope)
    assert decision.gpu_units == 1
    assert decision.upload_bytes == 0
    consume(scope, 13, bindings, 1)
    gpu(scope, handle, rw, lambda value, _: value + 5)
    scope.check(continuation.fort_scope_wait(scope.handle))
    final = report(scope)
    executions.append(final.execution_seconds)
    assert final.owner_segments == 3
    assert final.owner_available
    assert final.owner_execution_seconds == pytest.approx(sum(executions))
    assert final.owner_complete_seconds == pytest.approx(sum(executions) + final.terminal.seconds)
    assert final.owner_terminal_seconds == final.terminal.seconds
    assert scope.stats().upload_bytes == 64
    assert scope.stats().download_bytes == 16
    rows = [json.loads(line.removeprefix("FORT_SCOPED evidence ")) for line in capfd.readouterr().err.splitlines()
            if line.startswith("FORT_SCOPED evidence ")]
    decisions = [row for row in rows if row["event"] == "decision"]
    assert len(decisions) == 3
    assert all(row["continuation"]["terminal_hypothetical"] for row in decisions)
    assert not any(row["event"] == "export" for row in rows)
    copies = [row for row in rows if row["event"] == "copy"]
    assert sum(row["bytes"] for row in copies if row["direction"] == "d2h") == 16
    scope.close()
    assert list(values) == [2 * index + 5 for index in range(8)]


@pytest.mark.parametrize("incompatible", [False, True])
def test_dirty_input_missing_or_incompatible_estimate_still_installs_cpu(continuation, incompatible):
    scope = Scope(continuation)
    values = (c.c_double * 8)(*range(8))
    status, handle = scope.register(values, [8])
    scope.check(status)
    gpu(scope, handle, access(flags=6), lambda _, index: index + 100)
    scope.check(continuation.fort_scope_wait(scope.handle))
    begin(scope)
    bindings = record(scope, 21, handle, access(flags=1))
    decision = choose(scope, costs() if incompatible else Costs(), 0 if incompatible else 1)
    assert not decision.available
    assert not decision.gpu_units
    assert decision.cpu_units == 1
    assert not report(scope).available
    assert continuation.fort_scope_close(scope.handle) == 8
    consume(scope, 21, bindings, 0)
    scope.cpu_begin(handle, access(flags=1))
    assert list(values) == [index + 100 for index in range(8)]
    scope.cpu_end(handle)
    scope.check(continuation.fort_scope_wait(scope.handle))
    assert scope.stats().download_bytes == 64
    assert report(scope).owner_segments == 1
    assert not report(scope).owner_available
    scope.close()


def test_lifecycle_charges_once_and_new_registrations_once(continuation):
    scope = Scope(continuation)
    calibration = costs()
    calibration.create_seconds = .031
    calibration.register_seconds = .017
    values = (c.c_double * 8)(*range(8))
    status, handle = scope.register(values, [8])
    scope.check(status)
    estimates = []
    for step in range(4):
        if step == 2:
            extra = (c.c_double * 2)(9, 10)
            status, _ = scope.register(extra, [2], identity=2)
            scope.check(status)
        begin(scope)
        bindings = native_record(scope, step + 30, handle, access(flags=1), flops=1000)
        choose(scope, calibration)
        estimates.append(report(scope).execution_seconds)
        consume(scope, step + 30, bindings, 0)
        scope.cpu_begin(handle, access(flags=1))
        scope.cpu_end(handle)
        scope.check(continuation.fort_scope_wait(scope.handle))
    assert estimates[0]-estimates[1] == pytest.approx(.031 + .017)
    assert estimates[2]-estimates[3] == pytest.approx(.017)
    assert report(scope).owner_complete_seconds == pytest.approx(sum(estimates))
    scope.close()


def test_previews_and_partially_consumed_schedule_do_not_commit_owner_costs(continuation):
    scope = Scope(continuation)
    values = (c.c_double * 8)(*range(8))
    status, handle = scope.register(values, [8])
    scope.check(status)
    begin(scope)
    first = record(scope, 41, handle, access(flags=3))
    second = record(scope, 42, handle, access(flags=3))
    choose(scope, compatible=-1)
    choose(scope, compatible=-1)
    assert continuation.fort_scope_plan_report_v2(scope.handle, c.byref(Report())) == 8
    choose(scope)
    assert report(scope).owner_segments == 0
    consume(scope, 41, first, 1)
    gpu(scope, handle, access(flags=3), lambda value, _: value + 1)
    scope.check(continuation.fort_scope_wait(scope.handle))
    assert report(scope).owner_segments == 0
    assert not report(scope).owner_available
    consume(scope, 42, second, 1)
    gpu(scope, handle, access(flags=3), lambda value, _: value + 2)
    scope.check(continuation.fort_scope_wait(scope.handle))
    final = report(scope)
    assert final.owner_segments == 1
    assert final.owner_available
    assert final.owner_execution_seconds == final.execution_seconds
    scope.check(continuation.fort_scope_wait(scope.handle))
    assert report(scope).owner_execution_seconds == final.execution_seconds
    scope.close()
    assert list(values) == [index + 3 for index in range(8)]


def test_failed_nested_definition_query_falls_back_only_current_segment(continuation):
    scope = Scope(continuation)
    values = (c.c_double * 8)(*range(8))
    status, handle = scope.register(values, [8])
    scope.check(status)
    begin(scope)
    bindings = record(scope, 51, handle, access(flags=3))
    choose(scope)
    consume(scope, 51, bindings, 1)
    gpu(scope, handle, access(flags=3), lambda value, _: value + 10)
    scope.check(continuation.fort_scope_wait(scope.handle))
    begin(scope)
    forgotten = (Binding * 1)(Binding(handle, access()))
    scope.check(continuation.fort_scope_plan_add(scope.handle, 2, 0, forgotten, 1, 0, 0, 0))
    record(scope, 52, handle, access(flags=1))
    assert continuation.fort_scope_plan_validate(scope.handle) == 7
    # Only discard the failed query. Native hooks publish earlier GPU results.
    begin(scope)
    scope.cpu_begin(handle, access(flags=3))
    for index in range(8):
        values[index] += 1
    scope.cpu_end(handle)
    scope.check(continuation.fort_scope_wait(scope.handle))
    begin(scope)
    bindings = native_record(scope, 53, handle, access(flags=1))
    choose(scope)
    consume(scope, 53, bindings, 0)
    scope.cpu_begin(handle, access(flags=1))
    scope.cpu_end(handle)
    scope.check(continuation.fort_scope_wait(scope.handle))
    assert not report(scope).owner_available
    scope.close()
    assert list(values) == [index + 11 for index in range(8)]


def test_unvalidated_undefined_query_cannot_install_cpu_with_missing_profile(continuation):
    scope = Scope(continuation)
    values = (c.c_double * 8)(*range(8))
    status, handle = scope.register(values, [8], initialized=False)
    scope.check(status)
    begin(scope)
    record(scope, 61, handle, access(flags=1))
    assert continuation.fort_scope_plan_select(scope.handle, c.byref(Costs()), 1, c.byref(Decision())) == 7
    assert scope.stats().allocations == scope.stats().uploads == 0
    scope.close()


def test_failure_after_gpu_work_prohibits_continuation_replay(continuation, monkeypatch):
    scope = Scope(continuation)
    values = (c.c_double * 8)(*range(8))
    status, handle = scope.register(values, [8])
    scope.check(status)
    begin(scope)
    bindings = record(scope, 71, handle, access(flags=3))
    choose(scope)
    consume(scope, 71, bindings, 1)
    gpu(scope, handle, access(flags=3), lambda value, _: value + 77)
    monkeypatch.setenv("FORT_SCOPE_TEST_FAIL_WAIT", "1")
    assert continuation.fort_scope_wait(scope.handle) == 6
    assert continuation.fort_scope_plan_reset_mode(scope.handle, 1) == 6
    assert continuation.fort_scope_host_begin(scope.handle, handle, c.byref(access(flags=1))) == 6
    assert continuation.fort_scope_close(scope.handle) == 6
    assert list(values) == list(range(8))
    scope.check(continuation.fort_scope_abandon(scope.handle))


def test_prework_resource_fallback_preserves_results_and_invalidates_owner_estimate(continuation, monkeypatch):
    scope = Scope(continuation)
    values = (c.c_double * 8)(*range(8))
    status, handle = scope.register(values, [8])
    scope.check(status)
    begin(scope)
    bindings = record(scope, 81, handle, access(flags=3))
    choose(scope)
    consume(scope, 81, bindings, 1)
    monkeypatch.setenv("FORT_SCOPE_TEST_FAIL_ALLOC", "1")
    pointer = c.c_void_p()
    assert continuation.fort_scope_device_begin(scope.handle, handle, c.byref(access(flags=3)), c.byref(pointer)) == 4
    monkeypatch.delenv("FORT_SCOPE_TEST_FAIL_ALLOC")
    scope.cpu_begin(handle, access(flags=3))
    for index in range(8):
        values[index] += 19
    scope.cpu_end(handle)
    scope.check(continuation.fort_scope_wait(scope.handle))
    assert not report(scope).owner_available
    assert scope.stats().launches == 0
    scope.close()
    assert list(values) == [index + 19 for index in range(8)]


def test_fragmented_definition_query_has_no_live_effects(continuation):
    scope = Scope(continuation)
    values = (c.c_double * 64)(*range(64))
    even = [([index], [index + 1]) for index in range(0, 64, 2)]
    odd = [([index], [index + 1]) for index in range(1, 64, 2)]
    status, handle = scope.register(values, [64], defined=even)
    scope.check(status)
    begin(scope)
    record(scope, 91, handle, access(write=odd, overwrite=odd))
    before = bytes(scope.stats())
    assert continuation.fort_scope_plan_validate(scope.handle) == 5
    assert "fragmentation" in continuation.fort_scope_error().decode()
    assert bytes(scope.stats()) == before
    assert list(values) == list(range(64))
    scope.close()


def test_native_work_after_committed_segment_invalidates_owner_estimate(continuation):
    scope = Scope(continuation)
    values = (c.c_double * 8)(*range(8))
    status, handle = scope.register(values, [8])
    scope.check(status)
    begin(scope)
    bindings = record(scope, 101, handle, access(flags=3))
    choose(scope)
    consume(scope, 101, bindings, 1)
    gpu(scope, handle, access(flags=3), lambda value, _: value + 40)
    scope.check(continuation.fort_scope_wait(scope.handle))
    assert report(scope).owner_available
    boundary = access(read=[([0], [1])], write=[([0], [1])])
    scope.cpu_begin(handle, boundary)
    values[0] += 1
    scope.cpu_end(handle)
    scope.check(continuation.fort_scope_wait(scope.handle))
    assert not report(scope).owner_available
    assert scope.stats().download_bytes == 8
    scope.close()
    assert list(values) == [41, *[index + 40 for index in range(1, 8)]]


@pytest.mark.parametrize("device_prefix", [False, True])
def test_unplanned_prefix_is_not_reported_as_complete_owner(continuation, device_prefix):
    scope = Scope(continuation)
    values = (c.c_double * 8)(*range(8))
    status, handle = scope.register(values, [8])
    scope.check(status)
    rw = access(flags=3)
    if device_prefix:
        gpu(scope, handle, rw, lambda value, _: value + 50)
    else:
        scope.cpu_begin(handle, rw)
        for index in range(8):
            values[index] += 50
        scope.cpu_end(handle)
    scope.check(continuation.fort_scope_wait(scope.handle))
    begin(scope)
    bindings = native_record(scope, 111, handle, access(flags=1))
    choose(scope)
    consume(scope, 111, bindings, 0)
    scope.cpu_begin(handle, access(flags=1))
    assert list(values) == [index + 50 for index in range(8)]
    scope.cpu_end(handle)
    scope.check(continuation.fort_scope_wait(scope.handle))
    final = report(scope)
    assert final.available
    assert final.owner_segments == 1
    assert not final.owner_available
    assert scope.stats().download_bytes == (64 if device_prefix else 0)
    scope.close()


def test_switching_from_continuation_to_legacy_cannot_omit_work(continuation):
    scope = Scope(continuation)
    values = (c.c_double * 8)(*range(8))
    status, handle = scope.register(values, [8])
    scope.check(status)
    rw = access(flags=3)
    begin(scope)
    bindings = record(scope, 121, handle, rw)
    choose(scope)
    consume(scope, 121, bindings, 1)
    gpu(scope, handle, rw, lambda value, _: value + 60)
    scope.check(continuation.fort_scope_wait(scope.handle))
    assert report(scope).owner_available

    scope.check(continuation.fort_scope_plan_reset_mode(scope.handle, 0))
    bindings = native_record(scope, 122, handle, access(flags=1))
    assert not choose(scope).gpu_units
    scope.cpu_begin(handle, access(flags=1))
    assert list(values) == [index + 60 for index in range(8)]
    scope.cpu_end(handle)
    scope.check(continuation.fort_scope_wait(scope.handle))
    legacy = Report()
    scope.check(continuation.fort_scope_plan_report_v2(scope.handle, c.byref(legacy)))
    assert legacy.endpoint_mode == 0
    assert legacy.owner_segments == 1
    assert not legacy.owner_available

    begin(scope)
    bindings = native_record(scope, 123, handle, access(flags=1))
    choose(scope)
    consume(scope, 123, bindings, 0)
    scope.cpu_begin(handle, access(flags=1))
    scope.cpu_end(handle)
    scope.check(continuation.fort_scope_wait(scope.handle))
    assert report(scope).owner_segments == 2
    assert not report(scope).owner_available
    scope.close()
    assert list(values) == [index + 60 for index in range(8)]


def test_fortran_report_layout_matches_public_runtime(tmp_path):
    compiler = shutil.which("gfortran")
    if not compiler:
        pytest.skip("gfortran unavailable")
    source = tmp_path / "report.f90"
    source.write_text("""program check_report
use iso_c_binding
use fort_scoped_memory
implicit none
type(fort_scope_plan_report) :: report
print *, c_sizeof(report), FORT_SCOPE_PLAN_COMPLETE, FORT_SCOPE_PLAN_CONTINUE, FORT_SCOPE_PLANNING_REPORT_VERSION
end program
""")
    runtime_source = Path(__file__).resolve().parents[1] / "runtime" / "scoped_memory.f90"
    executable = tmp_path / "check_report"
    built = subprocess.run([compiler, "-std=f2008", "-Wall", "-Wextra", "-Werror", str(runtime_source),
                            str(source), "-o", str(executable)], cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert built.returncode == 0, built.stderr
    checked = subprocess.run([str(executable)], capture_output=True, text=True, timeout=10)
    assert checked.returncode == 0, checked.stderr
    assert [int(value) for value in checked.stdout.split()] == [c.sizeof(Report), 0, 1, 2]
