"""Independent numerical calibration changes costs without pretending to be FLOPs."""
import ctypes as c
import math

import pytest

from compiler.tests.test_scoped_planning_runtime import (  # noqa: F401 - pytest fixtures
    TOKEN,
    Binding,
    Scope,
    costs,
    planning,
    runtime,
    select,
)


@pytest.mark.parametrize("cpu,gpu,expected", [(10., .01, 1), (.01, 10., 0)])
def test_versioned_numerical_costs_change_placement(planning, cpu, gpu, expected):
    call = planning.fort_scope_plan_add_costs_v2
    call.argtypes = [TOKEN, c.c_uint32, TOKEN, c.POINTER(Binding), c.c_size_t,
                    c.c_double, c.c_double, c.c_int, c.c_double, c.c_double]
    scope = Scope(planning)
    scope.check(planning.fort_scope_plan_reset(scope.handle))
    scope.check(call(scope.handle, 1, 31, None, 0, 1., 0., 1, cpu, gpu))
    result = select(scope, costs())
    assert result.available and result.gpu_units == expected
    assert result.native_seconds >= cpu
    assert scope.stats().allocations == scope.stats().uploads == 0
    chosen = c.c_int(-1)
    scope.check(planning.fort_scope_plan_next(scope.handle, 31, None, 0, c.byref(chosen)))
    assert chosen.value == expected
    scope.close()


@pytest.mark.parametrize("bad", [-1., math.inf, math.nan])
def test_invalid_numerical_costs_fail_before_work(planning, bad):
    call = planning.fort_scope_plan_add_costs_v2
    call.argtypes = [TOKEN, c.c_uint32, TOKEN, c.POINTER(Binding), c.c_size_t,
                    c.c_double, c.c_double, c.c_int, c.c_double, c.c_double]
    scope = Scope(planning)
    scope.check(planning.fort_scope_plan_reset(scope.handle))
    assert call(scope.handle, 1, 31, None, 0, 1., 0., 1, bad, .01) != 0
    assert scope.stats().allocations == scope.stats().launches == 0
    scope.close()
