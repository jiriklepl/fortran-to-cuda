"""Synchronous pinned controls use exact physical sections and conservative costs."""
# ruff: noqa: F811 - imported pytest fixtures.
from __future__ import annotations

import ctypes as c
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from compiler.tests.test_scoped_continuation_runtime import begin, consume, report
from compiler.tests.test_scoped_planning_runtime import costs, planning, record, select  # noqa: F401
from compiler.tests.test_scoped_runtime import TOKEN, Scope, access, runtime  # noqa: F401


class TransferStats(c.Structure):
    _fields_ = [(name, c.c_uint32) for name in (
        "version", "requested_mode", "effective_mode", "fallback_reason")] + [(name, TOKEN) for name in (
        "fallbacks", "pinned_uploads", "pinned_downloads", "pinned_upload_bytes", "pinned_download_bytes",
        "packed_bytes", "unpacked_bytes", "tiles", "events", "event_waits", "staging_allocations", "staging_reuses",
        "slot_capacity", "staging_device_bytes", "process_reserved_bytes", "process_peak_bytes")] + [
        (name, c.c_double) for name in ("packing_seconds", "unpacking_seconds", "event_wait_seconds")]


@pytest.fixture
def pinned(planning):
    planning.fort_scope_set_transfers.argtypes = [TOKEN, c.c_uint32]
    planning.fort_scope_transfer_stats_get_v1.argtypes = [TOKEN, c.POINTER(TransferStats)]
    planning.fort_scope_plan_reset_mode.argtypes = [TOKEN, c.c_uint32]
    from compiler.tests.test_scoped_continuation_runtime import Report
    planning.fort_scope_plan_report_v2.argtypes = [TOKEN, c.POINTER(Report)]
    return planning


def stats(scope):
    result = TransferStats()
    scope.check(scope.lib.fort_scope_transfer_stats_get_v1(scope.handle, c.byref(result)))
    assert result.version == 1
    return result


def configured(library, mode=1):
    scope = Scope(library)
    scope.check(library.fort_scope_set_transfers(scope.handle, mode))
    return scope


@pytest.mark.parametrize(("mode", "reason"), [(0, 0), (1, 0), (2, 2), (3, 1)])
def test_configuration_and_readonly_statistics_do_not_initialize_cuda(pinned, mode, reason):
    scope = configured(pinned, mode)
    before = bytes(scope.stats())
    first = stats(scope)
    assert first.requested_mode == mode
    assert first.effective_mode == (1 if mode == 1 else 0)
    assert first.fallback_reason == reason
    assert first.fallbacks == bool(reason)
    assert bytes(scope.stats()) == before
    assert pinned.fort_scope_set_transfers(scope.handle, 4) == 1
    assert stats(scope).requested_mode == mode
    scope.close()


def test_public_manual_planner_never_uses_direct_calibration_for_pinned(pinned):
    scope = configured(pinned)
    values = (c.c_double * 8)(*range(8))
    status, handle = scope.register(values, [8])
    scope.check(status)
    assert pinned.fort_scope_set_transfers(scope.handle, 0) == 8
    begin(scope)
    bindings = record(scope, 301, handle, access(flags=3))
    scope.check(pinned.fort_scope_plan_validate(scope.handle))
    decision = select(scope, costs())
    assert not decision.available
    assert decision.gpu_units == 0
    assert decision.cpu_units == 1
    assert not report(scope).available
    consume(scope, 301, bindings, 0)
    scope.cpu_begin(handle, access(flags=3))
    for index in range(8):
        values[index] += 4
    scope.cpu_end(handle)
    scope.check(pinned.fort_scope_wait(scope.handle))
    assert scope.stats().allocations == stats(scope).pinned_uploads == 0
    scope.close()
    assert list(values) == [index + 4 for index in range(8)]


def test_opposite_rectangular_faces_preserve_host_halos_and_full_pitches(pinned):
    scope = configured(pinned)
    shape = [5, 7, 3]
    values = (c.c_double * 105)(*range(105))
    expected = (c.c_double * 105)(*range(105))
    status, handle = scope.register(values, shape, lower=[-2, -4, -1])
    scope.check(status)
    faces = [([0, 0, 0], [1, 7, 3]), ([4, 0, 0], [5, 7, 3])]
    effects = access(write=faces, overwrite=faces)
    pointer = scope.gpu(handle, effects)
    for z in range(3):
        for y in range(7):
            for x in (0, 4):
                index = x + 5*y + 35*z
                pointer[index] = expected[index] = (x-2) + 10*(y-4) + 100*(z-1)
    scope.check(pinned.fort_scope_note_launch(scope.handle))
    scope.gpu_end(handle)
    scope.check(pinned.fort_scope_wait(scope.handle))
    waits = scope.stats().waits
    scope.cpu_begin(handle, access(read=faces))
    scope.cpu_end(handle)
    assert scope.stats().waits == waits  # Completed staging work does not set context.pending.
    detail = stats(scope)
    assert detail.pinned_upload_bytes == 0
    assert detail.pinned_download_bytes == detail.unpacked_bytes == 2*7*3*8
    assert detail.events == detail.event_waits == detail.tiles == 2
    assert detail.staging_device_bytes == 0
    assert scope.stats().allocations == 1
    scope.close()
    assert bytes(values) == bytes(expected)


def test_large_row_is_tiled_without_changing_negative_logical_bounds(pinned):
    scope = configured(pinned)
    width = (16*1024*1024)//8 + 7
    values = (c.c_double * (2*width))()
    expected = (c.c_double * (2*width))()
    for index in (0, width-1, width, 2*width-1):
        values[index] = expected[index] = index-7
    status, handle = scope.register(values, [width, 2], lower=[-7, -3])
    scope.check(status)
    row = [([0, 0], [width, 1])]
    rw = access(read=row, write=row)
    pointer = scope.gpu(handle, rw)
    for index in (0, width//2, width-1):
        pointer[index] += 17
        expected[index] += 17
    scope.check(pinned.fort_scope_note_launch(scope.handle))
    scope.gpu_end(handle)
    scope.check(pinned.fort_scope_wait(scope.handle))
    scope.cpu_begin(handle, access(read=row))
    scope.cpu_end(handle)
    detail = stats(scope)
    assert detail.slot_capacity == 16*1024*1024
    assert detail.pinned_uploads == detail.pinned_downloads == 2
    assert detail.pinned_upload_bytes == detail.pinned_download_bytes == width*8
    assert detail.packed_bytes == detail.unpacked_bytes == width*8
    assert detail.events == detail.event_waits == 4
    scope.close()
    assert bytes(values) == bytes(expected)


def test_staging_resource_failure_can_change_copy_implementation_without_replay(pinned, monkeypatch):
    scope = configured(pinned)
    values = (c.c_double * 8)(*range(8))
    status, handle = scope.register(values, [8])
    scope.check(status)
    pointer = scope.gpu(handle, access(flags=3))
    for index in range(8):
        pointer[index] += 9
    scope.check(pinned.fort_scope_note_launch(scope.handle))
    scope.gpu_end(handle)
    scope.check(pinned.fort_scope_wait(scope.handle))
    monkeypatch.setenv("FORT_SCOPE_TEST_FAIL_STAGING_ALLOC", "1")
    scope.cpu_begin(handle, access(flags=1))
    scope.cpu_end(handle)
    detail = stats(scope)
    assert detail.fallback_reason == 4
    assert detail.fallbacks == 1
    assert detail.effective_mode == 0
    assert scope.stats().launches == 1
    assert scope.stats().allocations == 1
    assert scope.stats().download_bytes == 64
    scope.close()
    assert list(values) == [index + 9 for index in range(8)]


def test_completion_failure_before_valid_event_never_replays(pinned, monkeypatch):
    scope = configured(pinned)
    values = (c.c_double * 8)(*range(8))
    status, handle = scope.register(values, [8])
    scope.check(status)
    monkeypatch.setenv("FORT_SCOPE_TEST_FAIL_STAGING_EVENT", "1")
    output = c.c_void_p()
    assert pinned.fort_scope_device_begin(scope.handle, handle, c.byref(access(flags=3)), c.byref(output)) == 6
    assert pinned.fort_scope_host_begin(scope.handle, handle, c.byref(access(flags=3))) == 6
    assert pinned.fort_scope_close(scope.handle) == 6
    scope.check(pinned.fort_scope_abandon(scope.handle))
    assert list(values) == list(range(8))


def test_final_statistics_include_close_publication(pinned, monkeypatch, capfd):
    monkeypatch.setenv("FORT_RUNTIME_TRACE", "1")
    scope = configured(pinned)
    values = (c.c_double * 8)(*range(8))
    status, handle = scope.register(values, [8])
    scope.check(status)
    pointer = scope.gpu(handle, access(flags=3))
    for index in range(8):
        pointer[index] += 3
    scope.check(pinned.fort_scope_note_launch(scope.handle))
    scope.gpu_end(handle)
    assert stats(scope).pinned_download_bytes == 0
    scope.close()
    rows = [json.loads(line.removeprefix("FORT_SCOPED evidence ")) for line in capfd.readouterr().err.splitlines()
            if line.startswith("FORT_SCOPED evidence ")]
    final = [row for row in rows if row["event"] == "transfer_statistics"]
    assert len(final) == 1
    assert final[0]["complete"]
    assert final[0]["stats_version"] == 1
    assert final[0]["pinned_upload_bytes"] == final[0]["pinned_download_bytes"] == 64
    assert list(values) == [index + 3 for index in range(8)]


def test_fortran_transfer_statistics_layout_matches_c(tmp_path):
    compiler = shutil.which("gfortran")
    if not compiler:
        pytest.skip("gfortran unavailable")
    source = tmp_path / "stats.f90"
    source.write_text("""program check_stats
use iso_c_binding
use fort_scoped_memory
implicit none
type(fort_scope_transfer_stats) :: stats
print *, c_sizeof(stats), FORT_SCOPE_TRANSFER_ABI_VERSION, FORT_SCOPE_TRANSFERS_PINNED
end program
""")
    runtime_source = Path(__file__).resolve().parents[1] / "runtime" / "scoped_memory.f90"
    binary = tmp_path / "check_stats"
    built = subprocess.run([compiler, "-std=f2008", "-Wall", "-Wextra", "-Werror", str(runtime_source),
                            str(source), "-o", str(binary)], cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert built.returncode == 0, built.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert [int(value) for value in result.stdout.split()] == [c.sizeof(TransferStats), 1, 1]
