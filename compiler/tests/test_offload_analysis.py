"""Physical footprints of plan regions never weaken compiler proofs."""

import re
from itertools import product
from types import SimpleNamespace

import pytest

from compiler.analysis import build_execution_plan
from compiler.driver.options import CompilerOptions
from compiler.frontend import lower_file
from compiler.ir import Binary, CompilationError, IntrinsicCall, Literal, Reference, Size, Unary
from compiler.offload.analysis import Unit, _unit_footprints


def analyze(tmp_path, declarations, body):
    source = tmp_path / "physical.f90"
    # Footprint fixtures need only the scalars their body actually consumes.
    scalars = [name for name in ("n", "m") if re.search(r"\b" + name + r"\b", body, re.I)]
    parameters = ",".join(["a", "b", *scalars])
    scalar_declarations = "integer, intent(in) :: " + ",".join(scalars) + "\n" if scalars else ""
    source.write_text(
        "module physical\ncontains\nsubroutine work(" + parameters + ")\n"
        + scalar_declarations + declarations + "\n" + body + "\nend subroutine\nend module\n"
    )
    function = lower_file(source, "work")
    plan = build_execution_plan(function, options=CompilerOptions(opt_level=0))
    parameters = frozenset(function.parameters)
    return SimpleNamespace(units=tuple(_unit_footprints(Unit(index, region, ()), parameters)
                                       for index, region in enumerate(plan.regions)))


def value(expression, scalars, dimensions):
    if isinstance(expression, Literal):
        return int(expression.value)
    if isinstance(expression, Reference):
        return scalars[expression.symbol.name.lower()]
    if isinstance(expression, Size):
        return dimensions[expression.symbol.name.lower()][expression.dimension - 1]
    if isinstance(expression, Unary):
        result = value(expression.operand, scalars, dimensions)
        return -result if expression.operator == "-" else result
    if isinstance(expression, IntrinsicCall):
        values = [value(argument, scalars, dimensions) for argument in expression.arguments]
        return {"min": min, "max": max}[expression.name](*values)
    assert isinstance(expression, Binary)
    left = value(expression.left, scalars, dimensions)
    right = value(expression.right, scalars, dimensions)
    return {"+": lambda: left + right, "-": lambda: left - right, "*": lambda: left * right}[expression.operator]()


def points(box, scalars, dimensions):
    return set(
        product(
            *(
                range(value(lo, scalars, dimensions), value(hi, scalars, dimensions) + 1)
                for lo, hi in zip(box.lower, box.upper, strict=True)
            )
        )
    )


def footprint(analysis, name, unit=0):
    return next(value for value in analysis.units[unit].footprints if value.symbol.name.lower() == name)


RANK1 = "real, intent(inout) :: a(:)\nreal, intent(in) :: b(:)\ninteger :: i,j,t"
RANK2 = "real, intent(inout) :: a(:,:)\nreal, intent(in) :: b(:,:)\ninteger :: i,j,k,t"


@pytest.mark.parametrize(("body", "incoming"), [
    ("a(i)=b(i)\na(i)=a(i)*2", False),
    ("a(i)=a(i)+b(i)\na(i)=a(i)*2", True),
    ("if (b(i)>0) then\na(i)=b(i)\nendif\na(i)=a(i)*2", True),
    ("a(i)=b(i)\nt=i\na(t)=a(t)*2", False),
])
def test_same_iteration_definition_does_not_require_an_incoming_replica(tmp_path, body, incoming):
    analysis = analyze(tmp_path, RANK1, "do i=1,n\n" + body + "\nend do")
    result = footprint(analysis, "a")
    assert bool(result.reads or result.full_read) is incoming
    assert result.writes




def test_physical_affine_signed_permutation_matches_enumerated_accesses(tmp_path):
    analysis = analyze(tmp_path, RANK2, "do j=1,m\ndo i=2,n\na(n-i+2,j+1)=b(j,i)\nend do\nend do")
    scalars, dimensions = {"n": 5, "m": 3}, {"a": (6, 5), "b": (5, 6)}
    destination = footprint(analysis, "a")
    source = footprint(analysis, "b")
    assert destination.exact
    assert source.exact
    assert points(destination.writes[0], scalars, dimensions) == {
        (7 - i, j + 1) for j in range(1, 4) for i in range(2, 6)
    }
    assert points(source.reads[0], scalars, dimensions) == {(j, i) for j in range(1, 4) for i in range(2, 6)}


def test_opposite_planes_remain_finite_union_not_whole_volume(tmp_path):
    declarations = "real, intent(inout) :: a(:,:,:)\nreal, intent(in) :: b(:,:)\ninteger :: i,j"
    analysis = analyze(tmp_path, declarations, "do j=1,m\ndo i=1,n\na(i,j,1)=b(i,j)\na(i,j,n)=b(i,j)\nend do\nend do")
    result = footprint(analysis, "a")
    assert len(result.writes) == 2
    scalars, dimensions = {"n": 5, "m": 3}, {"a": (5, 3, 5), "b": (5, 3)}
    transferred = set().union(*(points(box, scalars, dimensions) for box in result.writes))
    assert len(transferred) == 30
    assert {point[2] for point in transferred} == {1, 5}
    assert len(footprint(analysis, "b").reads) == 1


def test_reverse_unit_stride_preserves_physical_bounds(tmp_path):
    analysis = analyze(tmp_path, RANK1, "do i=n,2,-1\na(i)=b(i-1)\nend do")
    assert points(footprint(analysis, "a").writes[0], {"n": 6}, {}) == {(i,) for i in range(2, 7)}
    assert points(footprint(analysis, "b").reads[0], {"n": 6}, {}) == {(i,) for i in range(1, 6)}


def test_nonunit_stride_is_a_full_array_footprint_not_a_legality_rejection(tmp_path):
    analysis = analyze(tmp_path, RANK1, "do i=1,n,2\na(i)=b(i)\nend do")
    assert footprint(analysis, "a").full_write
    assert footprint(analysis, "b").full_read


def test_squared_indices_never_reuse_injective_dependence_surrogates(tmp_path):
    analysis = analyze(tmp_path, RANK1, "do i=1,n\na(i*i)=b(i*i)\nend do")
    # A nonconservative legality proof is not an exact physical address map.
    assert not analysis.units[0].region.report.conservative
    assert footprint(analysis, "a").full_write
    assert footprint(analysis, "b").full_read


def test_private_affine_index_substitution_uses_current_definition(tmp_path):
    analysis = analyze(tmp_path, RANK1, "do i=1,n\nt=i+1\na(i)=b(t)\nt=i+2\na(i)=a(i)+b(t)\nend do")
    source = footprint(analysis, "b")
    assert not source.full_read
    assert len(source.reads) == 2
    assert {value(box.axes[0].offset, {}, {}) for box in source.reads} == {1, 2}


def test_private_affine_index_is_resolved(tmp_path):
    analysis = analyze(tmp_path, RANK1, "do i=1,n\nt=i+1\na(t)=b(i)\nend do")
    result = footprint(analysis, "a")
    assert not result.full_write
    assert points(result.writes[0], {"n": 4}, {}) == {(2,), (3,), (4,), (5,)}






def test_conditional_writes_preserve_holes_and_mark_upper_bounds(tmp_path):
    body = "do i=1,n\nif (b(i)>0) then\na(i)=b(i)\nend if\nend do"
    analysis = analyze(tmp_path, RANK1, body)
    destination = footprint(analysis, "a")
    assert not destination.exact
    assert destination.writes







def test_zero_divisor_is_rejected(tmp_path):
    with pytest.raises(CompilationError, match="division by zero"):
        analyze(tmp_path, RANK1, "do i=1,n/0\na(i)=b(i)\nend do")


def test_descriptor_minimum_bound_limits_the_box(tmp_path):
    analysis = analyze(tmp_path, RANK1, "do i=1,min(n,size(a,1))\na(i)=b(i)\nend do")
    assert points(footprint(analysis, "a").writes[0], {"n": 8}, {"a": (4,)}) == {(1,), (2,), (3,), (4,)}


def test_unused_loop_axis_still_guards_a_read_box(tmp_path):
    declarations = "real, intent(inout) :: a(:,:)\nreal, intent(in) :: b(:)\ninteger :: i,j"
    analysis = analyze(tmp_path, declarations, "do j=1,m\ndo i=1,n\na(i,j)=b(i)\nend do\nend do")
    source = footprint(analysis, "b")
    assert source.reads[0].active_units == (0,)
    assert source.read_units == (0,)
    # m=0 is an empty unit even though the projected read box has n elements.
    assert len(points(source.reads[0], {"n": 5, "m": 0}, {})) == 5


def test_imperfect_retained_loop_uses_full_footprints(tmp_path):
    body = "do i=1,n\nt=1\ndo j=1,m\na(i,j)=b(i,j)+t\nend do\nend do"
    analysis = analyze(tmp_path, RANK2, body)
    assert footprint(analysis, "a").full_write


def test_repeated_iterator_in_array_axes_is_not_a_rectangle(tmp_path):
    analysis = analyze(tmp_path, RANK2, "do i=1,n\na(i,i)=b(i,i)\nend do")
    assert footprint(analysis, "a").full_write
    assert footprint(analysis, "b").full_read
