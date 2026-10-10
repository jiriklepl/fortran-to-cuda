"""Dependency diagnostics preserve existing numerical and cost contracts."""

from compiler.analysis import build_execution_plan
from compiler.driver.options import CompilerOptions
from compiler.emission.common.resources import read_scoped_runtime
from compiler.emission.cuda.scoped import generate_scoped
from compiler.frontend import lower_source
from compiler.offload import analyze_offload
from compiler.offload.config import OffloadConfig
from compiler.tests.test_source_compute_costs import profile_v2


def prepare(body):
    source = f"""module independent_cost_features
contains
subroutine evaluate(a,b,n)
real(8),intent(inout)::a(:)
real(8),intent(in)::b(:)
integer,intent(in)::n
integer::i
real(8)::x,y
do i=1,n
{body}
enddo
end subroutine
end module
"""
    function = lower_source(source, "evaluate", source_name="independent_cost_features.f90")
    plan = build_execution_plan(function, options=CompilerOptions(opt_level=0))
    analysis = analyze_offload(function, plan)
    assert analysis.available, analysis.reason
    return function, plan, analysis


def test_public_analysis_and_scoped_entry_share_dependency_identity():
    function, plan, analysis = prepare("x=b(i)*2d0\ny=x+3d0\na(i)=sqrt(y)")
    unit, = analysis.units
    features = unit.compute_dependencies
    assert features.available, features.reason
    public = analysis.to_dict()["units"][0]["compute_dependencies"]
    assert public == features.to_dict()
    emitted = generate_scoped(function, plan, OffloadConfig("sections"), "common_functions.cuh")
    generated, = emitted.report["planning"]["units"]
    assert generated["compute_dependencies"] == public
    assert unit.compute_arithmetic_operations_per_iteration == 2
    assert dict(unit.intrinsic_work_per_iteration) == {"sqrt": 1}


def test_dependency_diagnostics_do_not_authorize_missing_numerical_costs():
    function, plan, analysis = prepare("a(i)=sqrt(b(i)*b(i)+1d0)")
    unit, = analysis.units
    assert unit.compute_dependencies.available
    emitted = generate_scoped(function, plan,
        OffloadConfig("auto", native_participation="serial"), "common_functions.cuh")
    assert not emitted.report["automatic_estimate_available"]
    assert emitted.report["planning"]["units"][0]["compute_model"] is None


def test_unavailable_dependency_shape_preserves_validated_v2_contract():
    function, plan, analysis = prepare("a(i)=b(i)\na(i)=a(i)+1d0")
    unit, = analysis.units
    # Ordered memory SSA is deliberately outside the first dependency collector.
    assert not unit.compute_dependencies.available
    assert unit.compute_arithmetic_operations_per_iteration == 1
    emitted = generate_scoped(function, plan,
        OffloadConfig("auto", profile_v2(), native_participation="serial"), "common_functions.cuh",
        runtime_id=read_scoped_runtime()[1]["runtime_id"])
    assert emitted.report["automatic_estimate_available"]
    generated, = emitted.report["planning"]["units"]
    assert generated["compute_model"] is not None
    assert not generated["compute_dependencies"]["available"]
