"""Optional public JSONL evidence follows the actual physical copy schedule."""
# ruff: noqa: F811 - pytest discovers imported fixtures under these names.
from __future__ import annotations

import ctypes as c
import json
from concurrent.futures import ThreadPoolExecutor

from compiler.tests.test_scoped_planning_runtime import (
    Binding,
    Costs,
    costs,
    planning,  # noqa: F401 - imported pytest fixture
    record,
    select,
)
from compiler.tests.test_scoped_runtime import Scope, access, runtime  # noqa: F401 - imported pytest fixture


def evidence(capfd):
    return [json.loads(line.removeprefix("FORT_SCOPED evidence "))
            for line in capfd.readouterr().err.splitlines() if line.startswith("FORT_SCOPED evidence ")]


def native(scope, handle, effects):
    bindings = (Binding * 1)(Binding(handle, effects))
    scope.check(scope.lib.fort_scope_plan_add(scope.handle, 0, 0, bindings, 1, 0, 0, 0))


def gpu(scope, unit, bindings, effects, update):
    chosen = c.c_int()
    scope.check(scope.lib.fort_scope_plan_next(scope.handle, unit, bindings, 1, c.byref(chosen)))
    assert chosen.value == 1
    pointer = c.c_void_p()
    scope.check(scope.lib.fort_scope_device_begin(scope.handle, bindings[0].buffer, c.byref(effects), c.byref(pointer)))
    device = (c.c_double * 32).from_address(pointer.value)
    for index in range(2, 30):
        device[index] = update(device[index])
    scope.check(scope.lib.fort_scope_note_launch(scope.handle))
    scope.check(scope.lib.fort_scope_device_end(scope.handle, bindings[0].buffer))


def test_final_schedule_evidence_explains_mirrors_faces_and_exports(planning, monkeypatch, capfd):
    monkeypatch.setenv("FORT_RUNTIME_TRACE", "1")
    scope = Scope(planning)
    values = (c.c_double * 32)(*range(32))
    status, handle = scope.register(values, [32], identity=17, generation=9, lower=[-5])
    scope.check(status)
    interior = access(read=[([2], [30])], write=[([2], [30])])
    faces = access(write=[([0], [2]), ([30], [32])], overwrite=[([0], [2]), ([30], [32])])
    mirror = access(read=[([2], [30])])
    scope.check(planning.fort_scope_plan_reset(scope.handle))
    first = record(scope, 91, handle, interior)
    native(scope, handle, faces)
    native(scope, handle, mirror)
    second = record(scope, 92, handle, interior)
    decision = select(scope, costs())
    rows = evidence(capfd)
    assert decision.gpu_units == 2
    assert {row["context"] for row in rows} == {scope.handle.value}
    assert all(row["schema_version"] == 1 for row in rows)
    assert rows[0]["event"] == "decision"
    assert rows[0]["evidence"] == "modeled_final_schedule"
    assert rows[-1]["event"] == "end"
    assert rows[-1]["complete"]
    assert not rows[-1]["truncated"]
    snapshot, = [row for row in rows if row["event"] == "snapshot"]
    assert (snapshot["identity"], snapshot["generation"]) == (17, 9)
    assert snapshot["initialized"] == snapshot["host_current"] == [{"lower": [0], "upper": [32]}]
    assert snapshot["device_current"] == []
    requirements = [row for row in rows if row["event"] == "requirement"]
    assert requirements[0]["preservation_reads"] == [{"lower": [2], "upper": [30]}]
    assert requirements[1]["writes"] == [{"lower": [0], "upper": [2]}, {"lower": [30], "upper": [32]}]
    assert requirements[1]["preservation_reads"] == []
    assert requirements[-1]["host_current"] == [
        {"lower": [0], "upper": [2]}, {"lower": [30], "upper": [32]}, {"lower": [2], "upper": [30]}]
    assert requirements[-1]["device_current"] == [{"lower": [2], "upper": [30]}]
    copies = [row for row in rows if row["event"] == "copy"]
    assert [(row["direction"], row["phase"], row["unit"]) for row in copies] == [
        ("h2d", "read", 91), ("d2h", "read", 0), ("d2h", "scope_close", 0)]
    assert all(row["rectangle"] == {"lower": [2], "upper": [30]} for row in copies)
    assert sum(row["bytes"] for row in copies if row["direction"] == "h2d") == decision.upload_bytes == 224
    assert sum(row["bytes"] for row in copies if row["direction"] == "d2h") == decision.download_bytes == 448
    assert sum(row["copy_calls"] for row in copies) == decision.uploads + decision.downloads == 3
    exported, = [row for row in rows if row["event"] == "export"]
    assert exported["initialized"] == [{"lower": [0], "upper": [32]}]
    gates = [row for row in rows if row["event"] == "gate"]
    assert [(row["first_worker"], row["last_worker_exclusive"]) for row in gates] == [(0, 1), (1, 2)]
    assert all(row["accepted"] and row["required_saving_seconds"] > 0 for row in gates)
    assert scope.stats().uploads == scope.stats().downloads == scope.stats().launches == 0
    monkeypatch.delenv("FORT_RUNTIME_TRACE")
    gpu(scope, 91, first, interior, lambda value: value * 2)
    scope.cpu_begin(handle, faces)
    values[0] = values[1] = values[30] = values[31] = -7
    scope.cpu_end(handle)
    scope.cpu_begin(handle, mirror)
    assert values[5] == 10
    scope.cpu_end(handle)
    gpu(scope, 92, second, interior, lambda value: value + 3)
    scope.cpu_begin(handle, access(flags=1))
    scope.cpu_end(handle)
    stats = scope.stats()
    for field in ("upload_bytes", "download_bytes", "uploads", "downloads", "launches", "waits", "peak_device_bytes"):
        assert getattr(stats, field) == getattr(decision, field), field
    scope.close()
    assert list(values) == [-7, -7, *[2 * index + 3 for index in range(2, 30)], -7, -7]


def test_trace_off_emits_no_planning_json_or_execution(planning, monkeypatch, capfd):
    monkeypatch.delenv("FORT_RUNTIME_TRACE", raising=False)
    scope = Scope(planning)
    values = (c.c_double * 32)(*range(32))
    status, handle = scope.register(values, [32])
    scope.check(status)
    scope.check(planning.fort_scope_plan_reset(scope.handle))
    record(scope, 93, handle, access(flags=3), flops=1)
    calibration = costs()
    calibration.h2d_latency = calibration.d2h_latency = 1e-3
    decision = select(scope, calibration)
    assert decision.available
    assert not decision.gpu_units
    assert evidence(capfd) == []
    assert scope.stats().allocations == scope.stats().uploads == scope.stats().downloads == 0
    scope.close()


def test_unavailable_evidence_explains_native_without_inventing_copies(planning, monkeypatch, capfd):
    monkeypatch.setenv("FORT_RUNTIME_TRACE", "1")
    scope = Scope(planning)
    values = (c.c_double * 32)(*range(32))
    status, handle = scope.register(values, [32])
    scope.check(status)
    scope.check(planning.fort_scope_plan_reset(scope.handle))
    record(scope, 94, handle, access(flags=3))
    decision = select(scope, Costs())
    rows = evidence(capfd)
    assert not decision.available
    assert not decision.gpu_units
    assert rows[0]["reason"] == "missing_or_incompatible_calibration"
    assert [row["event"] for row in rows] == ["decision", "snapshot", "end"]
    assert scope.stats().allocations == scope.stats().uploads == scope.stats().downloads == 0
    scope.close()


def test_preview_produces_no_duplicate_final_evidence(planning, monkeypatch, capfd):
    monkeypatch.setenv("FORT_RUNTIME_TRACE", "1")
    scope = Scope(planning)
    values = (c.c_double * 32)(*range(32))
    status, handle = scope.register(values, [32])
    scope.check(status)
    scope.check(planning.fort_scope_plan_reset(scope.handle))
    record(scope, 95, handle, access(flags=3), flops=1)
    calibration = costs()
    calibration.h2d_latency = calibration.d2h_latency = 1e-3
    preview = select(scope, calibration, -1)
    assert preview.available
    assert evidence(capfd) == []
    select(scope, calibration)
    rows = evidence(capfd)
    assert sum(row["event"] == "decision" for row in rows) == 1
    assert not any(row["event"] in {"copy", "gate"} for row in rows)
    scope.close()


def test_evidence_output_is_bounded_and_reports_truncation(planning, monkeypatch, capfd):
    monkeypatch.setenv("FORT_RUNTIME_TRACE", "1")
    scope = Scope(planning)
    bindings = (Binding * 32)()
    for index in range(32):
        values = (c.c_double * 1)(index)
        status, handle = scope.register(values, [1], identity=index + 1)
        scope.check(status)
        bindings[index] = Binding(handle, access(flags=1))
    scope.check(planning.fort_scope_plan_reset(scope.handle))
    for _ in range(256):
        scope.check(planning.fort_scope_plan_add(scope.handle, 0, 0, bindings, 32, 0, 0, 0))
    decision = select(scope, costs())
    assert decision.available
    assert not decision.gpu_units
    rows = evidence(capfd)
    assert len(rows) == 4098  # Decision + capped detailed rows + end marker.
    assert rows[-1]["event"] == "end"
    assert rows[-1]["rows"] == 4096
    assert rows[-1]["truncated"]
    assert not rows[-1]["complete"]
    assert scope.stats().allocations == scope.stats().uploads == scope.stats().downloads == 0
    scope.close()


def test_separate_context_evidence_lines_remain_valid_json(planning, monkeypatch, capfd):
    monkeypatch.setenv("FORT_RUNTIME_TRACE", "1")

    def run(identity):
        scope = Scope(planning)
        values = (c.c_double * 32)(*range(32))
        status, handle = scope.register(values, [32], identity=identity)
        scope.check(status)
        for index in range(4):
            scope.check(planning.fort_scope_plan_reset(scope.handle))
            record(scope, 100 + index, handle, access(flags=3), flops=1)
            calibration = costs()
            calibration.h2d_latency = calibration.d2h_latency = 1e-3
            assert not select(scope, calibration).gpu_units
        scope.close()
        return scope.handle.value

    with ThreadPoolExecutor(max_workers=2) as workers:
        handles = set(workers.map(run, [71, 72]))
    rows = evidence(capfd)
    assert {row["context"] for row in rows} == handles
    assert sum(row["event"] == "decision" for row in rows) == 8
    assert sum(row["event"] == "end" and row["complete"] for row in rows) == 8
