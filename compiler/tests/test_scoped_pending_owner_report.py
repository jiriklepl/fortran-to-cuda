"""A pending segment cannot disappear inside an available owner prediction."""
# ruff: noqa: F811 - imported fixtures are discovered by pytest.
import ctypes as c

import pytest

from compiler.tests.test_scoped_continuation_runtime import (
    begin,
    choose,
    consume,
    continuation,  # noqa: F401
    gpu,
    report,
)
from compiler.tests.test_scoped_planning_runtime import planning, record  # noqa: F401
from compiler.tests.test_scoped_runtime import TOKEN, Scope, access, runtime  # noqa: F401


@pytest.mark.parametrize("pending", [True, False])
def test_owner_prediction_requires_every_reached_segment(continuation, pending):
    continuation.fort_scope_task_graph_enable_v1.argtypes = [TOKEN]
    scope = Scope(continuation)
    values = (c.c_double * 8)(*range(8))
    status, handle = scope.register(values, [8])
    scope.check(status)
    scope.check(continuation.fort_scope_task_graph_enable_v1(scope.handle))
    predictions = []
    selected_reports = []
    for unit in (1, 2):
        begin(scope)
        bindings = record(scope, unit, handle, access(flags=3), flops=unit * 1e8)
        assert choose(scope).gpu_units == 1
        selected_reports.append(report(scope))
        predictions.append(selected_reports[-1].execution_seconds)
        consume(scope, unit, bindings, 1)
        gpu(
            scope,
            handle,
            access(flags=3),
            lambda value, _, increment=unit: value + increment,
        )
        if not pending:
            scope.check(continuation.fort_scope_wait(scope.handle))
    scope.check(continuation.fort_scope_wait(scope.handle))
    final = report(scope)
    # The first fresh query includes unpriced memo-key work. The second query
    # has device state and does not use that memo; its available local report
    # cannot restore completeness of the whole reached owner's prediction.
    assert not selected_reports[0].available
    assert selected_reports[1].available
    assert final.available
    assert final.execution_seconds == pytest.approx(predictions[-1])
    assert not final.owner_available
    if not pending:
        assert final.owner_segments == 2
        priced = sum(item.execution_seconds for item in selected_reports if item.available)
        assert final.owner_execution_seconds == pytest.approx(priced)
        assert final.owner_execution_seconds < sum(predictions)
    scope.close()
    assert list(values) == [index + 3 for index in range(8)]
