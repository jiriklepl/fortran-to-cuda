"""AUTO physical batching preserves the existing exact replica transitions."""
# ruff: noqa: F811 - imported maintained pytest fixtures.
import ctypes as c
import json

import pytest

from compiler.tests.test_scoped_pinned_runtime import pinned, stats  # noqa: F401
from compiler.tests.test_scoped_planning_runtime import Costs, costs, planning  # noqa: F401
from compiler.tests.test_scoped_runtime import TOKEN, Scope, access, runtime  # noqa: F401


class Rate(c.Structure):
    _fields_ = [(name, c.c_double) for name in (
        "batch_seconds", "submission_seconds", "byte_seconds", "row_seconds", "descriptor_seconds",
        "median_relative_error", "p90_relative_error", "maximum_relative_error")] + [
        (name, c.c_size_t) for name in (
            "minimum_payload_bytes", "maximum_payload_bytes", "minimum_rows", "maximum_rows",
            "minimum_row_bytes", "maximum_row_bytes", "minimum_pitch_bytes", "maximum_pitch_bytes",
            "maximum_span_bytes", "minimum_operations", "maximum_operations", "minimum_descriptor_work",
            "maximum_descriptor_work", "minimum_kernel_chunks", "maximum_kernel_chunks")
    ] + [("word_bytes_mask", c.c_uint32), ("layout_combinations_mask", c.c_uint32)]


class CopyCosts(c.Structure):
    _fields_ = [("version", c.c_uint32), ("valid", c.c_uint32), ("minimum_relative_saving", c.c_double),
                ("direct", Rate * 2), ("compact_cold", Rate * 2), ("compact_reuse", Rate * 2)]


def synthetic_copy_costs():
    """Complete-path diagnostic selector coefficients; never a hardware profile."""
    result = CopyCosts(2, 1, .5)
    for direction in range(2):
        for path in ("direct", "compact_cold", "compact_reuse"):
            rate = getattr(result, path)[direction]
            rate.submission_seconds = 1 if path == "direct" else 0
            rate.batch_seconds = 0 if path == "direct" else 1e-6
            rate.byte_seconds = 1e-9
            rate.minimum_payload_bytes = rate.minimum_rows = rate.minimum_row_bytes = 1
            rate.maximum_payload_bytes = rate.maximum_rows = rate.maximum_row_bytes = 16 << 20
            rate.minimum_pitch_bytes, rate.maximum_pitch_bytes = 1, 16 << 20
            rate.maximum_span_bytes = 1 << 30
            rate.minimum_operations, rate.maximum_operations = (1 if path == "direct" else 2), 256
            rate.minimum_descriptor_work, rate.maximum_descriptor_work = 1, 1 << 30
            rate.minimum_kernel_chunks, rate.maximum_kernel_chunks = 1, 6
            rate.word_bytes_mask, rate.layout_combinations_mask = 7, 254
    return result


def configured(library, *, mode=3, budget=None, calibrated=True):
    scope = Scope(library)
    library.fort_scope_set_copy_costs_v2.argtypes = [TOKEN, c.POINTER(CopyCosts), c.POINTER(Costs), c.c_int]
    scope.check(library.fort_scope_set_transfers(scope.handle, mode))
    if budget is not None:
        scope.check(library.fort_scope_set_device_budget(scope.handle, budget))
    if calibrated:
        base, copy = costs(), synthetic_copy_costs()
        scope.check(library.fort_scope_set_copy_costs_v2(scope.handle, c.byref(copy), c.byref(base), 1))
    return scope


def prepare(scope):
    values = (c.c_double * 64)(*([-77] * 64))
    status, handle = scope.register(values, [64], lower=[-17])
    scope.check(status)
    pointer = scope.gpu(handle, access(flags=6))
    for i in range(64):
        pointer[i] = 1000 + i
    scope.gpu_end(handle)
    middle = [([8], [56])]
    scope.cpu_begin(handle, access(write=middle, overwrite=middle))
    for i in range(8, 56):
        values[i] = -42
    scope.cpu_end(handle)
    return values, handle


def compact_events(capfd):
    return [json.loads(line.split("evidence ", 1)[1]) for line in capfd.readouterr().err.splitlines()
            if '"event":"compact_transfer"' in line]


@pytest.mark.parametrize("mode,calibrated,budget,expected", [(3, True, None, 1), (0, True, None, 2),
                                                          (3, False, None, 2), (3, True, 512, 2)])
def test_exact_replica_updates_and_pure_direct_missing_cost_budget_controls(pinned, monkeypatch, capfd,
                                                                          mode, calibrated, budget, expected):
    monkeypatch.setenv("FORT_RUNTIME_TRACE", "1")
    scope = configured(pinned, mode=mode, budget=budget, calibrated=calibrated)
    values, handle = prepare(scope)
    scope.cpu_begin(handle, access(flags=1)); scope.cpu_end(handle)
    assert list(values) == [1000 + i if i < 8 or i >= 56 else -42 for i in range(64)]
    assert (scope.stats().downloads, scope.stats().download_bytes) == (expected, 128)
    events = compact_events(capfd)
    assert len(events) == (expected == 1)
    if events:
        event, = events
        assert (event["logical_rectangles"], event["physical_operations"], event["dma_calls"],
                event["copy_kernel_launches"], event["word_bytes"]) == (2, 2, 1, 1, 8)
        assert event["workspace_bytes"] == 512 * 1024
    scope.gpu(handle, access(flags=1)); scope.gpu_end(handle)
    before = bytes(scope.stats())
    for _ in range(3):
        scope.cpu_begin(handle, access(flags=1)); scope.cpu_end(handle)
        scope.gpu(handle, access(flags=1)); scope.gpu_end(handle)
    assert bytes(scope.stats()) == before
    edges = [([0], [3]), ([61], [64])]
    scope.cpu_begin(handle, access(write=edges, overwrite=edges))
    for i in [0, 1, 2, 61, 62, 63]:
        values[i] = -i
    scope.cpu_end(handle)
    uploads = scope.stats().uploads
    pointer = scope.gpu(handle, access(flags=1)); scope.gpu_end(handle)
    assert scope.stats().uploads - uploads == expected
    assert [pointer[i] for i in range(64)] == list(values)
    scope.close()


def test_pre_submission_allocation_miss_uses_direct_once(pinned, monkeypatch):
    scope = configured(pinned)
    values, handle = prepare(scope)
    monkeypatch.setenv("FORT_SCOPE_TEST_FAIL_COMPACT_ALLOC", "1")
    scope.cpu_begin(handle, access(flags=1)); scope.cpu_end(handle)
    assert scope.stats().downloads == 2 and stats(scope).fallback_reason == 4
    assert list(values) == [1000 + i if i < 8 or i >= 56 else -42 for i in range(64)]
    scope.close()


@pytest.mark.parametrize("stage", ["SUBMIT", "RECORD", "WAIT"])
def test_submitted_copy_error_poisons_without_source_replay(pinned, monkeypatch, stage):
    scope = configured(pinned)
    _, handle = prepare(scope)
    monkeypatch.setenv("FORT_SCOPE_TEST_FAIL_COMPACT_" + stage, "1")
    assert pinned.fort_scope_host_begin(scope.handle, handle, c.byref(access(flags=1))) == 6
    monkeypatch.delenv("FORT_SCOPE_TEST_FAIL_COMPACT_" + stage)
    assert pinned.fort_scope_host_begin(scope.handle, handle, c.byref(access(flags=1))) == 6
    assert pinned.fort_scope_close(scope.handle) == 6
    assert pinned.fort_scope_abandon(scope.handle) == 0


def test_uncovered_physical_rate_executes_exact_direct_without_new_definition_credit(pinned, monkeypatch, capfd):
    monkeypatch.setenv("FORT_RUNTIME_TRACE", "1")
    scope = configured(pinned)
    base, copy = costs(), synthetic_copy_costs()
    # The complete128-byte D2H edge batch lies below this valid observed
    # direct-path range. The planner must not price it, while execution
    # still performs each exact original copy safely.
    copy.direct[1].minimum_payload_bytes = 256
    scope.check(pinned.fort_scope_set_copy_costs_v2(scope.handle, c.byref(copy), c.byref(base), 1))
    values, handle = prepare(scope)
    scope.cpu_begin(handle, access(flags=1)); scope.cpu_end(handle)
    assert list(values) == [1000 + i if i < 8 or i >= 56 else -42 for i in range(64)]
    assert (scope.stats().downloads, scope.stats().download_bytes) == (2, 128)
    assert not compact_events(capfd)
    scope.gpu(handle, access(flags=1)); scope.gpu_end(handle)
    before = bytes(scope.stats())
    scope.cpu_begin(handle, access(flags=1)); scope.cpu_end(handle)
    scope.gpu(handle, access(flags=1)); scope.gpu_end(handle)
    assert bytes(scope.stats()) == before
    scope.close()


def test_compact_pool_race_invalidates_only_cost_authority(tmp_path):
    """A changed physical lease must not leave a stale accepted owner price."""
    from pathlib import Path
    import shutil
    import subprocess

    compiler = shutil.which("g++")
    assert compiler is not None
    root = Path(__file__).resolve().parents[1] / "runtime"
    source = tmp_path / "estimate.cpp"
    source.write_text(r'''#define FORT_SCOPE_CPU_TEST
#include "scoped_runtime.cu"
#include <cassert>
int main(){
  Context c(0);
  auto reset=[&]{c.owner_estimate_available=true;c.last_report=fort_scope_plan_report{};
    c.last_report->available=c.last_report->owner_available=1;};
  auto invalid=[&]{assert(!c.owner_estimate_available && !c.last_report->available &&
    !c.last_report->owner_available && !c.poisoned && c.stats.uploads==0 && c.stats.downloads==0);};
  reset();compact_lease_observation(c,262144,true,262144,true);
  assert(c.owner_estimate_available && c.last_report->available && c.last_report->owner_available);
  reset();compact_lease_observation(c,262144,true,262144,false);invalid();
  reset();compact_lease_observation(c,262144,false,1048576,false);invalid();
  reset();compact_fallback(c,FORT_SCOPE_TRANSFER_ALLOCATION);invalid();
  assert(c.transfer_stats.fallback_reason==FORT_SCOPE_TRANSFER_ALLOCATION);
  reset();compact_fallback(c,FORT_SCOPE_TRANSFER_BUDGET);invalid();
  assert(c.transfer_stats.fallback_reason==FORT_SCOPE_TRANSFER_BUDGET && c.transfer_stats.fallbacks==2);
  reset();compact_estimate_unavailable(c);invalid();
}
''')
    binary = tmp_path / "estimate"
    built = subprocess.run([compiler, "-O0", "-std=c++17", "-pthread", "-I", str(root),
                            str(source), "-o", str(binary)], capture_output=True, text=True, timeout=120)
    assert built.returncode == 0, built.stdout + built.stderr
    ran = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
    assert ran.returncode == 0, ran.stdout + ran.stderr
