"""Independent numerical calibration changes costs without pretending to be FLOPs."""
import ctypes as c
import math

import pytest

from compiler.tests.test_scoped_planning_runtime import (
    Binding, Scope, TOKEN, costs, planning, runtime, select,
)
from compiler.analysis import build_execution_plan
from compiler.emission import generate_sources
from compiler.emission.common.resources import read_scoped_runtime
from compiler.frontend import lower_file
from compiler.offload.config import OffloadConfig
from compiler.offload.numerical_calibration import MIX_FAMILIES, profile_from_measurements
from compiler.tests.test_numerical_calibration import observations
from compiler.tests.test_offload_profile import scoped_profile


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


def emitted_numerical_case(tmp_path, evidence):
    """Exercise the public emitter with complete independent profile evidence."""
    source = tmp_path / "generic_math.f90"
    source.write_text("""module generic_math
contains
subroutine evaluate(a,n)
real(8),intent(inout)::a(:)
integer,intent(in)::n
integer::i
do i=1,n
a(i)=sqrt(a(i)*a(i)+1.0_8)+acos(0.5_8)+cos(a(i))
enddo
end subroutine
end module
""")
    function = lower_file(source, "evaluate")
    calibration = scoped_profile()
    calibration["scoped"]["runtime_id"] = read_scoped_runtime()[1]["runtime_id"]
    calibration["toolchain"].update(host_cxx_version="GCC 14.4.0", nvcc_version="NVCC V13.4.92")
    if evidence != "missing":
        records = observations()
        if evidence == "rejected":
            # The exact-family fits still pass, but the independently unfit
            # mixed fixtures reject applying primitive costs to other kernels.
            for record in records:
                if record.get("family") in MIX_FAMILIES:
                    record["seconds"] = [2*value for value in record["seconds"]]
        calibration = profile_from_measurements(calibration, records, calibration={})
    generated = generate_sources(function, build_execution_plan(function),
        offload_config=OffloadConfig(policy="auto", profile=calibration), memory_model="scoped")
    return generated


def test_public_scoped_emission_supplies_independently_validated_intrinsic_costs(tmp_path, planning):
    generated = emitted_numerical_case(tmp_path, "accepted")
    public = generated.scoped
    assert public["automatic_estimate_available"] and public["planning"]["available"]
    unit, = public["planning"]["units"]
    assert unit["intrinsic_work_per_iteration"] == {"acos": 1, "cos": 1, "sqrt": 1}
    assert unit["work_per_iteration"] > 0
    cpu = unit["cpu_numerical_seconds_per_iteration"]
    gpu = unit["gpu_numerical_seconds_per_iteration"]
    assert math.isfinite(cpu) and cpu > 0
    assert math.isfinite(gpu) and 0 < gpu < cpu
    ordinary, = generated.offload["analysis"]["units"]
    assert ordinary["cpu_numerical_seconds_per_iteration"] == cpu
    assert ordinary["gpu_numerical_seconds_per_iteration"] == gpu

    # Consume the emitted public unit's additional seconds through the versioned
    # planner interface. This requires no CUDA numerical work or device setup.
    add = planning.fort_scope_plan_add_costs_v2
    add.argtypes = [TOKEN, c.c_uint32, TOKEN, c.POINTER(Binding), c.c_size_t,
                   c.c_double, c.c_double, c.c_int, c.c_double, c.c_double]
    scope = Scope(planning)
    scope.check(planning.fort_scope_plan_reset(scope.handle))
    iterations = 10_000_000
    scope.check(add(scope.handle, 1, unit["id"], None, 0,
                   unit["work_per_iteration"]*iterations, 0, 1, cpu*iterations, gpu*iterations))
    decision = select(scope, costs())
    assert decision.available and decision.gpu_units == 1
    assert math.isfinite(decision.estimated_seconds)
    assert decision.native_seconds >= cpu*iterations
    assert scope.stats().allocations == scope.stats().uploads == scope.stats().launches == 0
    chosen = c.c_int(-1)
    scope.check(planning.fort_scope_plan_next(scope.handle, unit["id"], None, 0, c.byref(chosen)))
    assert chosen.value == 1
    scope.close()


@pytest.mark.parametrize("evidence,reason", [("missing", "numerical calibration"),
                                             ("rejected", "mixed holdout")])
def test_public_scoped_missing_or_rejected_intrinsic_costs_keep_auto_native(tmp_path, planning, evidence, reason):
    generated = emitted_numerical_case(tmp_path, evidence)
    public = generated.scoped
    assert not public["automatic_estimate_available"]
    assert not public["planning"]["available"]
    assert reason in public["automatic_reason"]
    unit, = public["planning"]["units"]
    assert unit["work_per_iteration"] is None
    assert unit["cpu_numerical_seconds_per_iteration"] == 0
    assert unit["gpu_numerical_seconds_per_iteration"] == 0
    assert not generated.offload["estimate_available"]

    add = planning.fort_scope_plan_add_costs_v2
    add.argtypes = [TOKEN, c.c_uint32, TOKEN, c.POINTER(Binding), c.c_size_t,
                   c.c_double, c.c_double, c.c_int, c.c_double, c.c_double]
    scope = Scope(planning)
    scope.check(planning.fort_scope_plan_reset(scope.handle))
    scope.check(add(scope.handle, 1, unit["id"], None, 0, 0, 0, 0, 0, 0))
    decision = select(scope, costs())
    assert not decision.available and decision.gpu_units == 0
    chosen = c.c_int(-1)
    scope.check(planning.fort_scope_plan_next(scope.handle, unit["id"], None, 0, c.byref(chosen)))
    assert chosen.value == 0
    assert scope.stats().allocations == scope.stats().uploads == scope.stats().launches == 0
    scope.close()
