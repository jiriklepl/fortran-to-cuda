"""Independent compute counts match real-operation calibration conventions."""

import pytest

from compiler.analysis import build_execution_plan
from compiler.driver.options import CompilerOptions
from compiler.frontend import lower_file
from compiler.ir import (
    ArrayAccess,
    Assignment,
    Binary,
    Block,
    IntrinsicCall,
    Literal,
    Reference,
    ScalarType,
    SourceLocation,
    Symbol,
)
from compiler.offload import analyze_offload
from compiler.offload.analysis import _compute_arithmetic_operations


def analyze(tmp_path, body, *, kind=8, declarations=""):
    source = tmp_path / "ordinary_work.f90"
    source.write_text(f"""module ordinary_work
contains
subroutine evaluate(a,b,n)
real({kind}),intent(out)::a(:)
real({kind}),intent(in)::b(:)
integer,intent(in)::n
integer::i,k
real({kind})::x,x1,x2,x3,x4
{declarations}
do i=1,n
{body}
enddo
end subroutine
end module
""")
    function = lower_file(source, "evaluate")
    plan = build_execution_plan(function, options=CompilerOptions(opt_level=0))
    result = analyze_offload(function, plan)
    assert result.available, result.reason
    unit, = result.units
    return unit, result.to_dict()["units"][0]


@pytest.mark.parametrize("kind", [4, 8])
def test_real_operations_exclude_assignment_and_array_indices(tmp_path, kind):
    unit, public = analyze(tmp_path, "a(i)=2*b(n-i+1)+1", kind=kind)
    assert unit.compute_arithmetic_operations_per_iteration == 2
    assert unit.arithmetic_work_per_iteration == unit.work_per_iteration == 5
    assert public["compute_arithmetic_operations_per_iteration"] == 2
    assert public["compute_arithmetic_estimate_reason"] is None


def test_copy_has_known_zero_real_arithmetic_and_preserves_legacy_work(tmp_path):
    unit, _ = analyze(tmp_path, "a(i)=b(i)")
    assert unit.compute_arithmetic_operations_per_iteration == 0
    assert unit.arithmetic_work_per_iteration == unit.work_per_iteration == 1
    assert unit.compute_runtime_divisions_per_iteration == 0


@pytest.mark.parametrize("kind", [4, 8])
def test_runtime_real_division_has_separate_primitive_count(tmp_path, kind):
    unit, public = analyze(tmp_path, f"a(i)=(b(i)/real(i,{kind}))/(-b(i))", kind=kind)
    assert unit.compute_arithmetic_operations_per_iteration == 3
    assert unit.compute_runtime_divisions_per_iteration == 2
    assert public["compute_runtime_divisions_per_iteration"] == 2
    # Legacy intrinsic/work contracts do not gain a synthetic Fortran intrinsic.
    assert unit.intrinsic_work_per_iteration == ()
    assert unit.work_per_iteration is not None


def test_constant_denominator_and_integer_division_need_no_runtime_divide_family(tmp_path):
    unit, _ = analyze(tmp_path, "k=i/2\na(i)=b(i)/(1.0_8+0.125_8)+real(k,8)")
    assert unit.compute_arithmetic_operations_per_iteration == 2
    assert unit.compute_runtime_divisions_per_iteration == 0


def test_runtime_division_counts_stop_at_address_expressions(tmp_path):
    unit, _ = analyze(tmp_path, "a(i)=b(int(real(i,8)/2.0_8))")
    assert unit.compute_arithmetic_operations_per_iteration == 0
    assert unit.compute_runtime_divisions_per_iteration == 0


def test_integer_work_and_casts_do_not_become_floating_point_operations(tmp_path):
    unit, _ = analyze(tmp_path, "k=i+1\na(i)=real(k,8)+b(i-1)*0.5_8+real(int(b(i)),8)")
    assert unit.compute_arithmetic_operations_per_iteration == 3
    assert unit.arithmetic_work_per_iteration > 3


@pytest.mark.parametrize("kind", [4, 8])
def test_numerical_primitive_surrounding_arithmetic_matches_fixture(tmp_path, kind):
    unit, _ = analyze(tmp_path,
        f"a(i)=0.25_{kind}+0.125_{kind}*sqrt(1.0_{kind}+0.25_{kind}*b(i)*b(i))", kind=kind)
    assert unit.compute_arithmetic_operations_per_iteration == 5
    assert dict(unit.intrinsic_work_per_iteration) == {"sqrt": 1}
    assert unit.arithmetic_work_per_iteration == 6
    assert unit.work_per_iteration is None


def test_unrolled_arithmetic_recipe_has_exactly_518_real_operations(tmp_path):
    initial = "x=b(i)\nx1=x\nx2=x*0.5_8\nx3=x*0.25_8\nx4=x*0.125_8\n"
    repeated = "\n".join(f"x{k}=x{k}*1.00000{k}_8+0.00000{k}_8" for k in range(1, 5)) + "\n"
    unit, _ = analyze(tmp_path, initial + repeated * 64 + "a(i)=x1+x2+x3+x4")
    assert unit.compute_arithmetic_operations_per_iteration == 518
    assert unit.arithmetic_work_per_iteration > 518


@pytest.mark.parametrize("kind", [4, 8])
def test_guarded_math_holdout_real_operators_have_matching_counts(tmp_path, kind):
    unit, _ = analyze(tmp_path, f"""x=-abs(b(i))
x=max(-0.9_{kind},min(-0.1_{kind},x))
a(i)=(x-0.1_{kind})/(-1.125_{kind})""", kind=kind)
    assert unit.compute_arithmetic_operations_per_iteration == 6
    assert not unit.work_is_upper_bound


def test_many_argument_minmax_counts_each_comparison_select(tmp_path):
    unit, _ = analyze(tmp_path, "a(i)=max(b(i),1.0_8,2.0_8,min(b(i),3.0_8,4.0_8))")
    assert unit.compute_arithmetic_operations_per_iteration == 5


def test_literal_only_real_expressions_do_not_become_runtime_work(tmp_path):
    unit, _ = analyze(tmp_path, "a(i)=b(i)+(2.0_8*3.0_8)-(1.0_8/4.0_8)")
    assert unit.compute_arithmetic_operations_per_iteration == 2
    assert unit.arithmetic_work_per_iteration == 5


@pytest.mark.parametrize(("width", "expected"), [(2, 37), (4, 199)])
def test_scalarized_private_matrix_recipe_excludes_constant_coordinate_work(tmp_path, width, expected):
    declarations = f"integer::j,l\nreal(8)::p({width},{width}),q({width},{width}),r({width},{width}),s({width},{width}),total"
    unit, _ = analyze(tmp_path, f"""x=b(i)
do j=1,{width}
do k=1,{width}
p(k,j)=x+real(k+j,8)*0.01_8
q(k,j)=0.02_8*x+merge(1.0_8,0.0_8,k==j)
enddo
enddo
do j=1,{width}
do k=1,{width}
total=0.0_8
do l=1,{width}
total=total+p(k,l)*q(l,j)
enddo
r(k,j)=total
s(k,j)=r(k,j)+p(k,j)
enddo
enddo
total=s(1,1)
do k=2,{width}
total=total+s(k,k)
enddo
x=0.125_8+0.01_8*total
a(i)=x+0.001_8*x""", declarations=declarations)
    assert unit.compute_arithmetic_operations_per_iteration == expected
    assert unit.workload_class == "fixed_private_array_v2"
    assert unit.arithmetic_work_per_iteration > expected


@pytest.mark.parametrize("expression", ["sign(b(i),1.0_8)", "mod(b(i),1.0_8)",
                                         "merge(b(i),1.0_8,b(i)>0)"])
def test_unpriced_real_operations_leave_new_estimate_unknown_without_changing_legality(tmp_path, expression):
    unit, public = analyze(tmp_path, "a(i)=" + expression)
    assert unit.compute_arithmetic_operations_per_iteration is None
    assert "unsupported real intrinsic" in unit.compute_arithmetic_estimate_reason
    assert public["compute_arithmetic_estimate_reason"] == unit.compute_arithmetic_estimate_reason
    assert unit.work_per_iteration is not None


def test_conditional_body_does_not_sum_alternative_paths_as_exact_work(tmp_path):
    unit, _ = analyze(tmp_path, "if(b(i)>0)then\na(i)=b(i)*2\nelse\na(i)=b(i)+1\nendif")
    assert unit.compute_arithmetic_operations_per_iteration is None
    assert unit.compute_runtime_divisions_per_iteration is None
    assert unit.compute_arithmetic_estimate_reason == "conditional arithmetic work is unknown"
    assert unit.arithmetic_work_per_iteration is not None
    assert unit.work_is_upper_bound


def test_retained_loop_has_unknown_real_operation_count(tmp_path):
    unit, _ = analyze(tmp_path, "x=b(i)\ndo k=1,i\nx=x*0.5_8\nenddo\na(i)=x")
    assert unit.compute_arithmetic_operations_per_iteration is None
    assert unit.compute_arithmetic_estimate_reason == "retained-loop arithmetic work is unknown"
    assert unit.arithmetic_work_per_iteration is None
    assert unit.work_is_upper_bound


def test_address_expression_tree_is_not_counted_even_when_it_contains_real_conversions():
    source = Symbol(1, "source", ScalarType.REAL, rank=1)
    target = Symbol(2, "target", ScalarType.REAL)
    # The compute cost walk must stop at array reads. Their index computation
    # has a separate address/protocol contract, rather than ordinary FLOPs.
    index = IntrinsicCall("int", (Binary("+", Literal("1.0", ScalarType.REAL),
                                         Literal("2.0", ScalarType.REAL)),), ScalarType.INTEGER)
    body = Block((Assignment(Reference(target), ArrayAccess(source, (index,)),
                             SourceLocation("generic_address.f90")),))
    assert _compute_arithmetic_operations(body) == (0, None)
