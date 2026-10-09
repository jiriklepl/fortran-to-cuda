"""Source-backed array operations keep bounds and RHS snapshot semantics."""

from fparser.two import Fortran2003 as F
import pytest

from compiler.driver.options import CompilerOptions
from compiler.driver.pipeline import prepare_function
from compiler.frontend import lower_source
from compiler.frontend.source_effects import SourceEffects, _children, _kind
from compiler.ir import CompilationError
from compiler.offload.analysis import analyze_offload
from compiler.scopes.array_operations import extract_array_operation


SOURCE = """module renamed_fields
implicit none
contains
subroutine advance(a,b,lo,hi,scale)
real(8),intent(inout)::a(-3:,-2:)
real(8),intent(in)::b(-7:,-6:)
integer,intent(in)::lo,hi
real(8),intent(in)::scale
STATEMENT
end subroutine
end module
"""


def extracted(tmp_path, statement, source=SOURCE):
    path = tmp_path / "source.f90"
    path.write_text(source.replace("STATEMENT", statement))
    analysis = SourceEffects([path])
    routine = analysis.routines["renamed_fields::advance"]
    node = next(node for node in _children(routine.execution) if _kind(node) == "Assignment_Stmt")
    region = extract_array_operation(analysis, routine, node)
    function, plan = prepare_function(lower_source(region.source, region.entry, source_name="array-operation.f90"),
                                      options=CompilerOptions())
    return analysis, routine, node, region, function, plan


@pytest.mark.parametrize("statement", [
    "a=scale", "a=a*scale", "a(:,-2)=b(:,-6)*scale", "a(lo:hi,-2)=b(lo-4:hi-4,-6)",
    "a(lo:hi:2,-2)=b(lo-4:hi-4:2,-6)", "a(hi:lo:-2,-2)=b(hi-4:lo-4:-2,-6)",
    "a(:,-2)=real(lbound(a,1),8)+scale",
    "a=scale/real(size(a),8)",
])
def test_conformant_operations_are_proved_without_rebasing_logical_indices(tmp_path, statement):
    _, _, node, region, function, plan = extracted(tmp_path, statement)
    assert region.nodes == (node,)
    assert region.private_scalars == ()
    assert "fort_array_lower_" in region.source
    assert region.runtime_guards
    assert len(plan.regions) == 1
    public = analyze_offload(function, plan)
    assert public.available, public.reason


@pytest.mark.parametrize("statement", [
    "a(lo:hi,-2)=a(lo-1:hi-1,-2)",
    "a(lo:hi,-2)=a(hi:lo:-1,-2)",
])
def test_rhs_snapshots_needing_cross_iteration_order_are_not_parallelized(tmp_path, statement):
    with pytest.raises(CompilationError, match="loop-carried conflict"):
        extracted(tmp_path, statement)


def test_runtime_guards_include_ordered_intermediate_and_conformance_checks(tmp_path):
    _, _, _, region, _, _ = extracted(tmp_path, "a(lo:hi*2,-2)=b(lo-4:hi*2-4,-6)")
    guards = region.runtime_guards
    multiplication = next(i for i, guard in enumerate(guards) if "*2_c_int64_t" in guard)
    hi_check = next(i for i, guard in enumerate(guards) if "int(hi,kind=c_int64_t)" in guard)
    assert hi_check < multiplication
    assert any(" == " in guard and "max(" in guard for guard in guards)


def test_source_authority_cannot_be_forged_with_same_statement(tmp_path):
    analysis, routine, node, _, _, _ = extracted(tmp_path, "a=scale")
    with pytest.raises(CompilationError, match="source authority"):
        extract_array_operation(analysis, routine, F.Assignment_Stmt(str(node)))


@pytest.mark.parametrize("statement,reason", [
    ("a(lo:hi:lo,-2)=scale", "constant|INTEGER|parameter"),
    ("a=unknown(b)", "unresolved"),
    ("a(:,lo:hi)=b(:,-6)", "rank"),
])
def test_uncertain_operations_keep_native_execution(tmp_path, statement, reason):
    with pytest.raises(CompilationError, match=reason):
        extracted(tmp_path, statement)


def test_new_array_math_retains_native_ieee_environment_guard(tmp_path):
    _, _, _, region, _, _ = extracted(tmp_path, "a=a*scale")
    assert region.requires_numerical_environment
    assert region.public()["numerical_environment"]["rounding"] == "round to nearest"
    with pytest.raises(CompilationError, match="native ordering contract"):
        extracted(tmp_path, "a=max(a,b,scale)")
    observer = SOURCE.replace("real(8),intent(inout)",
                              "use,intrinsic::ieee_exceptions,only:ieee_get_flag,ieee_invalid\nreal(8),intent(inout)").replace(
                                  "STATEMENT", "STATEMENT\ncall ieee_get_flag(ieee_invalid,seen)")
    with pytest.raises(CompilationError, match="source-observable"):
        extracted(tmp_path, "a=a*scale", observer)
