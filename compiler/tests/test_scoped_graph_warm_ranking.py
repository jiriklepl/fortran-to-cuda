"""Graph placement reports startup debt without pretending it executed CUDA."""
# ruff: noqa: F811 - pytest discovers imported fixtures under these names.
from __future__ import annotations

import ctypes as c
import json

import pytest

from compiler.tests.test_scoped_planning_runtime import (
    costs,
    planning,  # noqa: F401 - imported pytest fixture
    record,
    select,
)
from compiler.tests.test_scoped_runtime import TOKEN, Scope, access, runtime  # noqa: F401


@pytest.mark.parametrize("unpriced_team", [False, True])
def test_graph_preview_preserves_warm_ranking_and_honest_cold_evidence(planning, monkeypatch, capfd, unpriced_team):
    monkeypatch.setenv("FORT_RUNTIME_TRACE", "1")
    planning.fort_scope_task_graph_enable_v1.argtypes = [TOKEN]
    planning.fort_scope_plan_team_entry_v1.argtypes = [TOKEN]
    scope = Scope(planning)
    values = (c.c_double * 32)(*range(32))
    status, handle = scope.register(values, [32])
    scope.check(status)
    scope.check(planning.fort_scope_task_graph_enable_v1(scope.handle))
    scope.check(planning.fort_scope_plan_reset(scope.handle))
    if unpriced_team:
        scope.check(planning.fort_scope_plan_team_entry_v1(scope.handle))
    bindings = record(scope, 971, handle, access(flags=3))
    calibration = costs()
    calibration.cold_driver_startup_seconds = 2
    preview = select(scope, calibration, -1)
    assert preview.available == preview.gpu_units == 1
    assert "FORT_SCOPED evidence " not in capfd.readouterr().err
    decision = select(scope, calibration)
    rows = [json.loads(line.removeprefix("FORT_SCOPED evidence "))
            for line in capfd.readouterr().err.splitlines() if line.startswith("FORT_SCOPED evidence ")]
    reported, = [row for row in rows if row["event"] == "decision"]
    assert decision.available == decision.gpu_units == 1
    assert reported["ranking_policy"] == "warm_graph_heuristic"
    assert reported["cold_driver_startup_debt_seconds"] == 2
    assert reported["startup_debt_is_prediction"] is True
    assert reported["unpriced_team_overhead"] is unpriced_team
    rank = reported["terminal_debt_adjusted_ranking_seconds"]
    assert reported["estimated_seconds"] == pytest.approx(rank + 2)
    assert rank < reported["native_seconds"] < reported["estimated_seconds"]
    gate, = [row for row in rows if row["event"] == "gate"]
    assert gate["phase"] == "task_graph"
    assert gate["accepted"] is True
    assert gate["estimated_seconds"] == reported["warm_gate_seconds"]
    assert gate["counterfactual_seconds"] == reported["warm_gate_counterfactual_seconds"]
    assert gate["required_saving_seconds"] == reported["warm_gate_required_saving_seconds"]
    assert scope.stats().allocations == scope.stats().uploads == scope.stats().downloads == scope.stats().launches == 0
    chosen = c.c_int()
    scope.check(planning.fort_scope_plan_next(scope.handle, 971, bindings, 1, c.byref(chosen)))
    assert chosen.value == 1
    scope.close()
    assert list(values) == list(range(32))


def test_dirty_continuation_separates_warm_gate_from_terminal_debt_adjusted_ranking(planning, monkeypatch, capfd):
    monkeypatch.setenv("FORT_RUNTIME_TRACE", "1")
    planning.fort_scope_task_graph_enable_v1.argtypes = [TOKEN]
    planning.fort_scope_plan_reset_mode.argtypes = [TOKEN, c.c_uint32]
    scope = Scope(planning)
    values = (c.c_double * 32)(*range(32))
    status, handle = scope.register(values, [32])
    scope.check(status)
    scope.check(planning.fort_scope_task_graph_enable_v1(scope.handle))
    device = scope.gpu(handle, access(flags=3))
    for index in range(32):
        device[index] += 5
    scope.check(planning.fort_scope_note_launch(scope.handle))
    scope.gpu_end(handle)
    scope.check(planning.fort_scope_plan_reset_mode(scope.handle, 1))
    bindings = record(scope, 972, handle, access(flags=3))
    calibration = costs()
    calibration.d2h_latency = .5
    select(scope, calibration)
    rows = [json.loads(line.removeprefix("FORT_SCOPED evidence "))
            for line in capfd.readouterr().err.splitlines() if line.startswith("FORT_SCOPED evidence ")]
    reported, = [row for row in rows if row["event"] == "decision"]
    gate, = [row for row in rows if row["event"] == "gate"]
    continuation = reported["continuation"]
    entry_debt = continuation["entry_terminal"]["seconds"]
    assert entry_debt > .5
    assert reported["cold_driver_startup_debt_seconds"] == 0
    assert reported["warm_gate_seconds"] == gate["estimated_seconds"]
    assert reported["warm_gate_counterfactual_seconds"] == gate["counterfactual_seconds"]
    assert reported["warm_gate_required_saving_seconds"] == gate["required_saving_seconds"]
    assert reported["terminal_debt_adjusted_ranking_seconds"] == pytest.approx(gate["estimated_seconds"] - entry_debt)
    assert reported["estimated_seconds"] == continuation["execution_seconds"]
    assert gate["estimated_seconds"] == pytest.approx(reported["estimated_seconds"] + continuation["terminal"]["seconds"])
    assert gate["counterfactual_seconds"] == pytest.approx(reported["native_seconds"] + continuation["native_terminal"]["seconds"])
    assert gate["accepted"] is True
    assert scope.stats().downloads == 0
    chosen = c.c_int()
    scope.check(planning.fort_scope_plan_next(scope.handle, 972, bindings, 1, c.byref(chosen)))
    assert chosen.value == 1
    scope.close()
    assert list(values) == [index + 5 for index in range(32)]
