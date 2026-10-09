"""Physical footprints and optional partitions never weaken compiler proofs."""

import json
import re
from itertools import product

import pytest

from compiler.analysis import build_execution_plan
from compiler.driver.options import CompilerOptions
from compiler.frontend import lower_file
from compiler.ir import Binary, CompilationError, IntrinsicCall, Literal, Reference, Size, Unary
from compiler.offload import analyze_offload


def analyze(tmp_path, declarations, body):
    source = tmp_path / "physical.f90"
    # Footprint fixtures need only the scalars their body actually consumes;
    # unused public scalar inputs intentionally select native at the value ABI.
    scalars = [name for name in ("n", "m") if re.search(r"\b" + name + r"\b", body, re.I)]
    parameters = ",".join(["a", "b", *scalars])
    scalar_declarations = "integer, intent(in) :: " + ",".join(scalars) + "\n" if scalars else ""
    source.write_text(
        "module physical\ncontains\nsubroutine work(" + parameters + ")\n"
        + scalar_declarations + declarations + "\n" + body + "\nend subroutine\nend module\n"
    )
    function = lower_file(source, "work")
    plan = build_execution_plan(function, options=CompilerOptions(opt_level=0))
    return analyze_offload(function, plan)


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


def test_scalar_math_has_explicit_counts_and_cannot_use_ordinary_fma_rate(tmp_path):
    analysis = analyze(tmp_path, RANK1, "do i=1,n\na(i)=sqrt(abs(b(i)))+cos(acos(b(i)))\nend do")
    unit = analysis.units[0]
    assert dict(unit.intrinsic_work_per_iteration) == {"sqrt": 1, "cos": 1, "acos": 1}
    assert unit.work_per_iteration is None
    assert unit.arithmetic_work_per_iteration > 0
    assert "numerical calibration" in unit.work_estimate_reason
    record = analysis.to_dict()["units"][0]
    assert record["intrinsic_work_per_iteration"] == {"acos": 1, "cos": 1, "sqrt": 1}
    assert record["work_per_iteration"] is None


def test_ordinary_arithmetic_preserves_existing_public_cost_report(tmp_path):
    analysis = analyze(tmp_path, RANK1, "do i=1,n\na(i)=2*b(i)+1\nend do")
    assert analysis.units[0].work_per_iteration is not None
    assert "intrinsic_work_per_iteration" not in analysis.to_dict()["units"][0]


def test_physical_affine_signed_permutation_matches_enumerated_accesses(tmp_path):
    analysis = analyze(tmp_path, RANK2, "do j=1,m\ndo i=2,n\na(n-i+2,j+1)=b(j,i)\nend do\nend do")
    assert analysis.available
    scalars, dimensions = {"n": 5, "m": 3}, {"a": (6, 5), "b": (5, 6)}
    destination = footprint(analysis, "a")
    source = footprint(analysis, "b")
    assert destination.exact
    assert source.exact
    assert points(destination.writes[0], scalars, dimensions) == {
        (7 - i, j + 1) for j in range(1, 4) for i in range(2, 6)
    }
    assert points(source.reads[0], scalars, dimensions) == {(j, i) for j in range(1, 4) for i in range(2, 6)}
    assert not destination.uploads  # Every element in each downloaded box is overwritten.
    assert analysis.chunk.axis == 0
    assert dict((array.symbol.name.lower(), array.dimension) for array in analysis.chunk.arrays) == {"a": 1, "b": 0}


def test_opposite_planes_remain_finite_union_not_whole_volume(tmp_path):
    declarations = "real, intent(inout) :: a(:,:,:)\nreal, intent(in) :: b(:,:)\ninteger :: i,j"
    analysis = analyze(tmp_path, declarations, "do j=1,m\ndo i=1,n\na(i,j,1)=b(i,j)\na(i,j,n)=b(i,j)\nend do\nend do")
    result = footprint(analysis, "a")
    assert len(result.writes) == 2
    scalars, dimensions = {"n": 5, "m": 3}, {"a": (5, 3, 5), "b": (5, 3)}
    transferred = set().union(*(points(box, scalars, dimensions) for box in result.downloads))
    assert len(transferred) == 30
    assert {point[2] for point in transferred} == {1, 5}
    assert len(footprint(analysis, "b").reads) == 1
    report = analysis.to_dict()
    volume = next(item for item in report["units"][0]["footprints"] if item["symbol"] == "a")["transfer_volume"]
    assert volume["upload"]["basis"] == "physical_rectangles"
    assert volume["upload"]["element_bytes"] == 4
    assert not volume["upload"]["rectangles"]
    assert len(volume["download"]["rectangles"]) == 2


def test_reverse_unit_stride_preserves_physical_bounds(tmp_path):
    analysis = analyze(tmp_path, RANK1, "do i=n,2,-1\na(i)=b(i-1)\nend do")
    assert points(footprint(analysis, "a").writes[0], {"n": 6}, {}) == {(i,) for i in range(2, 7)}
    assert points(footprint(analysis, "b").reads[0], {"n": 6}, {}) == {(i,) for i in range(1, 6)}
    assert analysis.chunk is not None


def test_nonunit_stride_is_full_array_fallback_not_legality_rejection(tmp_path):
    analysis = analyze(tmp_path, RANK1, "do i=1,n,2\na(i)=b(i)\nend do")
    assert analysis.available
    assert footprint(analysis, "a").full_write
    assert footprint(analysis, "a").full_upload
    assert footprint(analysis, "b").full_read
    assert analysis.chunk is None
    volume = analysis.to_dict()["units"][0]["footprints"][0]["transfer_volume"]["upload"]
    assert volume["basis"] == "whole_array"
    assert "size(" in volume["byte_count_upper_bound"]


def test_squared_indices_never_reuse_injective_dependence_surrogates(tmp_path):
    analysis = analyze(tmp_path, RANK1, "do i=1,n\na(i*i)=b(i*i)\nend do")
    assert analysis.available
    # A nonconservative legality proof is not an exact physical address map.
    assert not analysis.units[0].region.report.conservative
    assert footprint(analysis, "a").full_write
    assert footprint(analysis, "b").full_read
    assert analysis.chunk is None


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


def test_readonly_halos_allow_chunking_and_report_offset_extrema(tmp_path):
    analysis = analyze(tmp_path, RANK1, "do i=2,n-1\na(i)=b(i-1)+b(i+1)\nend do")
    assert analysis.chunk.axis == 0
    source = next(array for array in analysis.chunk.arrays if array.symbol.name.lower() == "b")
    assert value(source.read_lower_offset, {}, {}) == -1
    assert value(source.read_upper_offset, {}, {}) == 1
    assert source.write_lower_offset is None


@pytest.mark.parametrize("kind", ["RAW", "WAR", "WAW"])
def test_ordered_cross_chunk_dependencies_are_rejected(tmp_path, kind):
    declarations = "real, intent(inout) :: a(:),b(:)\ninteger :: i,j"
    statements = {
        "RAW": ("a(i)=b(i)", "b(j)=a(j-1)"),
        "WAR": ("b(i)=a(i-1)", "a(j)=b(j)"),
        "WAW": ("a(i)=b(i)", "a(j-1)=b(j)"),
    }
    first, second = statements[kind]
    analysis = analyze(tmp_path, declarations, f"do i=2,n\n{first}\nend do\ndo j=2,n\n{second}\nend do")
    assert analysis.available
    assert len(analysis.units) == 2
    assert analysis.chunk is None
    assert "cross-chunk" in analysis.chunk_reason


def test_equal_pointwise_units_share_chunk_domain_and_upload_boxes(tmp_path):
    declarations = "real, intent(inout) :: a(:),b(:)\ninteger :: i,j"
    analysis = analyze(tmp_path, declarations, "do i=1,n\na(i)=b(i)\nend do\ndo j=1,n\nb(j)=a(j)+1\nend do")
    assert analysis.chunk is not None
    combined = next(interval for interval in analysis.intervals if interval.stop - interval.start == 2)
    for item in combined.footprints:
        assert len(item.uploads) == 1
        assert item.uploads[0].active_units == item.read_units


def test_alternative_axis_can_avoid_neighbor_dependence(tmp_path):
    declarations = "real, intent(inout) :: a(:,:),b(:,:)\ninteger :: i,j,k,t"
    body = "do j=2,m\ndo i=1,n\na(i,j)=b(i,j)\nend do\nend do\n"
    body += "do k=2,m\ndo t=1,n\nb(t,k)=a(t,k-1)\nend do\nend do"
    analysis = analyze(tmp_path, declarations, body)
    assert analysis.chunk.axis == 1


def test_conditional_writes_preserve_holes_and_mark_upper_bounds(tmp_path):
    body = "do i=1,n\nif (b(i)>0) then\na(i)=b(i)\nend if\nend do"
    analysis = analyze(tmp_path, RANK1, body)
    destination = footprint(analysis, "a")
    assert not destination.exact
    assert destination.uploads == destination.writes
    assert analysis.units[0].work_is_upper_bound
    assert analysis.chunk is not None


def test_interval_search_is_bounded_but_always_includes_whole_entry(tmp_path):
    body = "\n".join(f"do i=1,n\na(i)=b(i)+{index}\nend do" for index in range(6))
    analysis = analyze(tmp_path, RANK1, body)
    assert len(analysis.units) == 6
    assert len(analysis.intervals) == 19
    assert (0, 6) in {(interval.start, interval.stop) for interval in analysis.intervals}
    assert all(
        interval.stop - interval.start <= 4 or (interval.start, interval.stop) == (0, 6)
        for interval in analysis.intervals
    )
    assert json.loads(json.dumps(analysis.to_dict()))["available"]


@pytest.mark.parametrize("prefix", ["t=n", "if (n>0) then"])
def test_host_steps_are_not_silently_dropped(tmp_path, prefix):
    body = prefix + "\ndo i=1,n\na(i)=b(i)\nend do"
    if prefix.startswith("if"):
        body += "\nend if"
    analysis = analyze(tmp_path, RANK1, body)
    assert not analysis.available
    assert analysis.units == ()


def test_array_valued_bound_is_not_eagerly_queried(tmp_path):
    declarations = "real, intent(inout) :: a(:)\ninteger, intent(in) :: b(:)\ninteger :: i"
    analysis = analyze(tmp_path, declarations, "do i=1,b(1)\na(i)=1\nend do")
    assert not analysis.available
    assert "bounds" in analysis.reason


def test_positive_constant_divisor_bound_has_a_total_cost_query(tmp_path):
    analysis = analyze(tmp_path, RANK1, "do i=1,n/2\na(i)=b(i)\nend do")
    assert analysis.available, analysis.reason


@pytest.mark.parametrize("divisor", ["n", "-1"])
def test_unproved_divisor_bound_is_conservatively_unavailable(tmp_path, divisor):
    analysis = analyze(tmp_path, RANK1, f"do i=1,n/({divisor})\na(i)=b(i)\nend do")
    assert not analysis.available


def test_zero_divisor_is_rejected_before_cost_analysis(tmp_path):
    with pytest.raises(CompilationError, match="division by zero"):
        analyze(tmp_path, RANK1, "do i=1,n/0\na(i)=b(i)\nend do")


def test_descriptor_minimum_bound_is_queryable(tmp_path):
    analysis = analyze(tmp_path, RANK1, "do i=1,min(n,size(a,1))\na(i)=b(i)\nend do")
    assert analysis.available
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
    assert analysis.available
    assert footprint(analysis, "a").full_write
    assert analysis.units[0].work_per_iteration is None
    assert analysis.chunk is None


def test_repeated_iterator_in_array_axes_is_not_a_rectangle(tmp_path):
    analysis = analyze(tmp_path, RANK2, "do i=1,n\na(i,i)=b(i,i)\nend do")
    assert analysis.available
    assert footprint(analysis, "a").full_write
    assert footprint(analysis, "b").full_read


def test_chunk_rejects_different_domains(tmp_path):
    body = "do i=1,n\na(i)=b(i)\nend do\ndo j=2,n\na(j)=b(j)\nend do"
    analysis = analyze(tmp_path, RANK1, body)
    assert analysis.available
    assert analysis.chunk is None
    assert "equal mapped domains" in analysis.chunk_reason
