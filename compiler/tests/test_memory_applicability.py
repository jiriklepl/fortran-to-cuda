"""Cost classes use original physical indices and preserve forced legality."""
from dataclasses import FrozenInstanceError, replace

import pytest

from compiler.analysis import build_execution_plan
from compiler.driver.options import CompilerOptions
from compiler.frontend import lower_file
from compiler.ir import (
    ArrayAccess,
    Assignment,
    Binary,
    Block,
    If,
    Loop,
    ScalarType,
    Symbol,
)
from compiler.offload.memory_applicability import (
    MAX_DEPTH,
    MAX_STATEMENTS,
    POINTWISE_THREE_ARRAY,
    memory_access_requirement,
)
from compiler.tests.test_compute_dependencies import (
    INTEGER,
    ITERATOR,
    LOCATION,
    REAL,
    A,
    B,
    K,
    N,
    X,
    Y,
    access,
    assign,
    literal,
    ref,
    region,
)

C = Symbol(101, "second_input", REAL, rank=1)


def pointwise(value=None, **kwargs):
    return region(assign(access(B), value or Binary("+", access(A), access(C))),
                  captures=(A, B, C, N), **kwargs)


def test_original_three_array_fixture_class_is_supported_and_detached():
    original = pointwise(Binary("+", access(A), Binary("*", literal(.25), access(C))))
    before = repr(original)
    result = memory_access_requirement(original)
    assert result.available, result.reason
    assert result.class_id == POINTWISE_THREE_ARRAY
    assert repr(original) == before
    assert [a.symbol for a in result.accesses] == [A, C, B]
    assert not result.to_dict()["runtime_alias_proven"]
    assert not result.to_dict()["gpu_legality_proven"]
    assert result.to_dict()["traffic_equals_unique_working_set_required"]
    detached = result.to_dict()
    detached["physical_accesses"][0]["resource"] = "changed"
    assert result.to_dict()["physical_accesses"][0]["resource"] == A.name
    with pytest.raises(FrozenInstanceError):
        result.available = False


def test_ordered_integer_definitions_preserve_actual_translations():
    original = region(
        assign(K, Binary("-", ref(ITERATOR), literal(2, INTEGER))),
        assign(X, access(A, ref(K))),
        assign(K, Binary("+", ref(ITERATOR), literal(5, INTEGER))),
        assign(Y, access(C, ref(K))),
        assign(access(B), Binary("+", ref(X), ref(Y))),
        captures=(A, B, C, N),
    )
    # Transformed dependence-report strings cannot supply physical coordinates.
    original = replace(original, report=replace(original.report, reads="surrogate affine i", writes="surrogate i"))
    result = memory_access_requirement(original)
    assert result.available, result.reason
    assert [item.offsets[0].constant for item in result.accesses] == [-2, 5, 0]


@pytest.mark.parametrize(("kind", "reason"), [
    ("zero", "zero-fill"), ("copy", "zero-fill"), ("rmw", "read/modify/write"),
    ("accumulate", "read/modify/write"), ("stencil", "stencil"),
    ("repeated", "repeated"), ("multiple_writes", "write-only"),
])
def test_missing_access_contracts_do_not_become_generic_bandwidth_estimates(kind, reason):
    if kind == "zero":
        original = pointwise(literal(0))
    elif kind == "copy":
        original = pointwise(access(A))
    elif kind == "rmw":
        original = pointwise(Binary("*", access(B), literal(2)))
    elif kind == "accumulate":
        original = pointwise(Binary("+", access(B), access(A)))
    elif kind == "stencil":
        original = pointwise(Binary("+", Binary("+", access(A),
            access(A, Binary("+", ref(ITERATOR), literal(1, INTEGER)))), access(C)))
    elif kind == "repeated":
        original = pointwise(Binary("+", Binary("+", access(A), access(A)), access(C)))
    else:
        original = pointwise()
        original = replace(original, body=Block((*original.body.statements, *original.body.statements)))
    before = repr(original)
    result = memory_access_requirement(original)
    assert not result.available
    assert reason in result.reason
    assert repr(original) == before
    assert result.to_dict()["missing_contracts"]["stencil_v1"]


def test_conditional_and_retained_loop_costs_are_not_assumed_pointwise():
    original = pointwise()
    for body in (Block((If(literal(1, INTEGER), original.body, Block(()), LOCATION),)),
                 Block((Loop(K, literal(1, INTEGER), literal(2, INTEGER), original.body, LOCATION),))):
        result = memory_access_requirement(replace(original, body=body))
        assert not result.available
        assert "conditional or retained-loop" in result.reason


def test_non_affine_physical_access_does_not_borrow_surrogate_dependence_indices():
    original = pointwise(Binary("+", access(A, Binary("*", ref(ITERATOR), ref(ITERATOR))), access(C)))
    original = replace(original, report=replace(original.report, reads="proved surrogate linear map"))
    result = memory_access_requirement(original)
    assert not result.available
    assert "physical subscript" in result.reason


def test_mixed_precision_and_nonprivate_scalar_writes_remain_unpriced():
    second = replace(C, dtype=ScalarType.REAL32)
    original = pointwise(Binary("+", access(A), access(second)))
    assert "mixed memory precision" in memory_access_requirement(original).reason
    original = pointwise()
    shared = Symbol(102, "shared_scalar", REAL)
    original = replace(original, body=Block((*original.body.statements, assign(shared, literal(0)))))
    assert "nonprivate scalar" in memory_access_requirement(original).reason


def multidimensional():
    outer = Symbol(103, "j", INTEGER)
    arrays = tuple(replace(s, rank=2) for s in (A, C, B))
    indices = (ref(ITERATOR), ref(outer))
    assignment = Assignment(ArrayAccess(arrays[2], indices),
                            Binary("+", ArrayAccess(arrays[0], indices), ArrayAccess(arrays[1], indices)), LOCATION)
    body = Block((assignment,))
    loops = (Loop(outer, literal(-2, INTEGER), ref(N), body, LOCATION),
             Loop(ITERATOR, literal(-5, INTEGER), ref(N), body, LOCATION))
    return replace(pointwise(), loops=loops, body=body, captured_symbols=(*arrays, N))


def test_rectangular_nd_access_requires_runtime_pitch_and_alias_proofs():
    result = memory_access_requirement(multidimensional())
    assert result.available, result.reason
    assert all(access.loop_axes == (1, 0) for access in result.accesses)
    assert "full descriptor extent" in " ".join(result.runtime_requirements)
    assert "canonical allocations" in " ".join(result.runtime_requirements)


def test_first_fortran_axis_must_match_original_inner_loop():
    original = multidimensional()
    result = memory_access_requirement(replace(original, loops=tuple(reversed(original.loops))))
    assert not result.available
    assert "innermost source loop" in result.reason


def test_triangular_domains_and_nonunit_steps_do_not_match_calibration():
    original = multidimensional()
    outer, inner = original.loops
    triangular = replace(original, loops=(outer, replace(inner, upper=ref(outer.iterator))))
    assert "rectangular" in memory_access_requirement(triangular).reason
    for stride in (-1, 2, ref(N)):
        result = memory_access_requirement(replace(original, loops=(outer, replace(inner, step=stride))))
        assert not result.available
        assert "unit source strides" in result.reason


def test_uninitialized_and_redefined_integer_aliases_do_not_reuse_stale_maps():
    original = region(assign(K, ref(ITERATOR)), assign(K, literal(.25)),
                      assign(access(B), Binary("+", access(A, ref(K)), access(C))), captures=(A, B, C, N))
    assert "not proved affine" in memory_access_requirement(original).reason


def test_budget_failures_are_cost_unavailability_only():
    original = pointwise()
    result = memory_access_requirement(replace(original, body=Block(original.body.statements * (MAX_STATEMENTS + 1))))
    assert not result.available
    value = access(A)
    for _ in range(MAX_DEPTH + 1):
        value = Binary("+", value, literal(1))
    result = memory_access_requirement(pointwise(Binary("+", value, access(C))))
    assert not result.available
    assert "budget" in result.reason


def test_renamed_native_three_array_loop_is_recognized(tmp_path):
    source = tmp_path / "renamed.f90"
    source.write_text("""module independent_example
contains
subroutine update(first,second,result,n)
integer,intent(in)::n
real(8),intent(in)::first(:),second(:)
real(8),intent(out)::result(:)
integer::i
do i=1,n
result(i)=first(i)+.25_8*second(i)
enddo
end subroutine
end module
""")
    function = lower_file(source, "update")
    plan = build_execution_plan(function, options=CompilerOptions(opt_level=0))
    selected, = plan.regions
    result = memory_access_requirement(selected)
    assert result.available, result.reason
    assert result.class_id == POINTWISE_THREE_ARRAY


@pytest.mark.parametrize("precision", [32, 64])
def test_predeclared_dependency_fixtures_have_the_actual_three_array_access_contract(precision):
    from compiler.offload.cpu_dependency_workloads import dependency_recipes

    for recipe in dependency_recipes(precision):
        result = memory_access_requirement(recipe.region)
        assert result.available, (recipe.name, result.reason)
        assert result.class_id == POINTWISE_THREE_ARRAY
