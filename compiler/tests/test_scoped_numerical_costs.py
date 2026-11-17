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
