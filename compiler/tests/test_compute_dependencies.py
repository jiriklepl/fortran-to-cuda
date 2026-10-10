"""Cost features preserve source data flow without changing numerical proofs."""

import json
import math
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
    IntrinsicCall,
    Literal,
    Loop,
    PrivateArrayOrigin,
    Reference,
    ScalarType,
    Size,
    SourceLocation,
    Symbol,
    Unary,
)
from compiler.ir.plan import ParallelRegion, RegionReport
from compiler.offload.analysis import _compute_operation_counts
from compiler.offload.compute_dependencies import (
    MAX_DEFINITIONS,
    MAX_EDGES,
    MAX_EXPRESSION_DEPTH,
    MAX_OPERATIONS,
    MAX_VISITS,
    ComputeDependencies,
    ComputeOperation,
    analyze_compute_dependencies,
)

REAL = ScalarType.REAL
INTEGER = ScalarType.INTEGER
LOCATION = SourceLocation("generic.f90", 1)
ITERATOR = Symbol(0, "i", INTEGER)
N = Symbol(1, "n", INTEGER, parameter=True)
A = Symbol(2, "input", REAL, rank=1)
B = Symbol(3, "output", REAL, rank=1)
X = Symbol(4, "x", REAL)
Y = Symbol(5, "y", REAL)
Z = Symbol(6, "z", REAL)
K = Symbol(7, "k", INTEGER)
SCALE = Symbol(8, "scale", REAL, parameter=True)


def literal(value, dtype=REAL):
    return Literal(str(value), dtype)


def ref(symbol):
    return Reference(symbol)


def access(symbol, index=None):
    return ArrayAccess(symbol, (ref(ITERATOR) if index is None else index,))


def assign(target, value):
    return Assignment(target if isinstance(target, (ArrayAccess, Reference)) else ref(target), value, LOCATION)


def region(*statements, private=(X, Y, Z, K), captures=(A, B, SCALE, N)):
    body = Block(tuple(statements))
    loop = Loop(ITERATOR, literal(1, INTEGER), ref(N), body, LOCATION)
    return ParallelRegion(
        0,
        (loop,),
        tuple(s for s in statements if isinstance(s, Assignment)),
        private,
        captures,
        RegionReport("", "", "", "", "", "", ""),
        body,
    )


def analyze(*statements, **kwargs):
    result = analyze_compute_dependencies(region(*statements, **kwargs))
    assert result.available, result.reason
    return result


def test_equal_operation_counts_have_different_dependency_paths():
    independent = region(
        assign(X, Binary("*", access(A), literal(2))),
        assign(Y, Binary("*", access(A), literal(3))),
        assign(Z, Binary("*", access(A), literal(4))),
        assign(access(B), Binary("+", Binary("+", ref(X), ref(Y)), ref(Z))),
    )
    dependent = region(
        assign(X, Binary("*", access(A), literal(2))),
        assign(X, Binary("*", ref(X), literal(3))),
        assign(X, Binary("*", ref(X), literal(4))),
        assign(X, Binary("+", ref(X), literal(5))),
        assign(access(B), Binary("+", ref(X), literal(6))),
    )
    assert _compute_operation_counts(independent.body) == _compute_operation_counts(dependent.body) == (5, 0, None)
    first = analyze_compute_dependencies(independent).to_dict()
    second = analyze_compute_dependencies(dependent).to_dict()
    assert first["operation_count"] == second["operation_count"] == 5
    assert (first["unweighted_span"], second["unweighted_span"]) == (3, 5)
    assert (first["maximum_same_depth_operations"], second["maximum_same_depth_operations"]) == (3, 1)
    assert not first["native_simd_proven"]


def test_latest_definitions_and_repeated_operand_positions_are_retained():
    result = analyze(
        assign(X, Binary("*", access(A), literal(2))),
        assign(Y, ref(X)),
        assign(X, Binary("+", ref(X), literal(1))),
        assign(access(B), Binary("-", ref(X), ref(Y))),
    )
    assert [op.operands for op in result.operations] == [((), ()), ((0,), ()), ((1,), (0,))]
    repeated = analyze(assign(X, Binary("*", access(A), literal(2))), assign(access(B), Binary("+", ref(X), ref(X))))
    assert repeated.operations[1].operands == ((0,), (0,))
    assert repeated.operations[1].predecessors == (0, 0)
    assert repeated.edge_count == 2
    assert repeated.outputs == ((1,),)
    assert repeated.weighted_span({"multiply": 2, "add": 5}) == 7


def test_constant_invariant_and_varying_are_separate_without_folding_claim():
    result = analyze(
        assign(X, Binary("+", literal(1), literal(2))),
        assign(Y, Binary("*", ref(X), ref(SCALE))),
        assign(access(B), Binary("+", ref(Y), access(A))),
    )
    assert [(op.constant, op.invariant) for op in result.operations] == [(True, True), (False, True), (False, False)]
    assert result.weighted_span({"add": 1, "multiply": 2}) == 4
    assert result.to_dict()["constant_operations"] == 1
    assert result.to_dict()["invariant_operations"] == 2
    assert not result.to_dict()["native_folding_proven"]


def test_division_is_a_full_node_with_denominator_provenance():
    result = analyze(
        assign(X, literal(2)),
        assign(Y, Binary("/", access(A), ref(X))),
        assign(access(B), Binary("/", ref(Y), ref(SCALE))),
    )
    assert [op.family for op in result.operations] == ["divide_constant", "divide_dynamic"]
    assert result.weighted_span({"divide_constant": 2, "divide_dynamic": 5}) == 7
    assert result.operations[0].literal_operands == (None, ("real", "2"))
    assert result.operations[1].literal_operands == (None, None)


def test_constant_expression_divisor_is_not_guessed_as_a_literal():
    result = analyze(assign(X, Binary("+", literal(1), literal(2))), assign(access(B), Binary("/", access(A), ref(X))))
    assert result.operations[-1].family == "divide_constant"
    assert result.operations[-1].literal_operands == (None, None)


def test_math_intrinsics_and_typed_cast_dependencies():
    single = Symbol(10, "single", ScalarType.REAL32)
    result = analyze(
        assign(
            single, IntrinsicCall("sqrt", (IntrinsicCall("real", (access(A),), ScalarType.REAL32),), ScalarType.REAL32)
        ),
        assign(X, IntrinsicCall("dble", (ref(single),), REAL)),
        assign(access(B), IntrinsicCall("cos", (IntrinsicCall("acos", (ref(X),), REAL),), REAL)),
        private=(single, X),
    )
    assert [(op.family, op.dtype) for op in result.operations] == [
        ("sqrt", ScalarType.REAL32),
        ("acos", REAL),
        ("cos", REAL),
    ]
    assert result.operations[1].predecessors == (0,)
    assert result.weighted_span({"sqrt": 2, "acos": 4, "cos": 3}) == 9
    assert result.floating_dtypes == (REAL, ScalarType.REAL32)


def test_zero_cost_casts_retain_every_floating_precision():
    narrowed = IntrinsicCall("real", (access(A),), ScalarType.REAL32)
    widened = IntrinsicCall("dble", (narrowed,), REAL)
    result = analyze(assign(access(B), Binary("*", widened, literal(2))))
    assert [(op.family, op.dtype) for op in result.operations] == [("multiply", REAL)]
    assert result.floating_dtypes == (REAL, ScalarType.REAL32)
    public = result.to_dict()
    assert public["floating_dtypes"] == ["real", "real32"]
    public["floating_dtypes"].clear()
    assert result.floating_dtypes == (REAL, ScalarType.REAL32)
    without_rounding = analyze(assign(access(B), Binary("*", access(A), literal(2))))
    assert without_rounding.floating_dtypes == (REAL,)
    assert result.identity != without_rounding.identity


def test_assignment_target_precision_is_retained_without_numeric_operation():
    single = Symbol(10, "single", ScalarType.REAL32)
    result = analyze(assign(single, access(A)), private=(single,))
    assert result.operations == ()
    assert result.floating_dtypes == (REAL, ScalarType.REAL32)
    single_output = replace(B, dtype=ScalarType.REAL32)
    array_result = analyze(assign(access(single_output), access(A)))
    assert array_result.floating_dtypes == (REAL, ScalarType.REAL32)


@pytest.mark.parametrize("dtype", [REAL, ScalarType.REAL32])
def test_single_precision_classification_includes_leaves_and_invariants(dtype):
    source, output, scale = (replace(symbol, dtype=dtype) for symbol in (A, B, SCALE))
    result = analyze(assign(access(output), Binary("*", access(source), ref(scale))))
    assert result.floating_dtypes == (dtype,)
    assert result.to_dict()["floating_dtypes"] == [dtype.value]


def test_integer_addresses_are_excluded_and_unknown_types_are_not_published():
    index = Binary("+", ref(ITERATOR), literal(1, INTEGER))
    result = analyze(assign(access(B, index), access(A, index)))
    assert result.operations == ()
    assert result.floating_dtypes == (REAL,)
    empty = analyze(assign(K, Binary("+", ref(ITERATOR), literal(1, INTEGER))))
    assert empty.floating_dtypes == ()
    assert empty.to_dict()["floating_dtypes"] == []
    unavailable = analyze_compute_dependencies(
        region(assign(X, access(A)), If(literal(1), Block(()), None, LOCATION))
    )
    assert not unavailable.available
    assert unavailable.floating_dtypes == ()
    assert unavailable.to_dict()["floating_dtypes"] is None


def test_integer_division_chain_reaching_real_output_is_explicitly_unpriced():
    j = Symbol(11, "integer_intermediate", INTEGER)
    result = analyze(
        assign(K, Binary("/", ref(ITERATOR), literal(2, INTEGER))),
        assign(j, Binary("+", ref(K), literal(1, INTEGER))),
        assign(X, IntrinsicCall("real", (ref(j),), REAL)),
        assign(access(B), Binary("+", ref(X), access(A))),
        private=(K, j, X),
    )
    assert [op.family for op in result.operations] == ["add"]
    assert dict(result.unpriced_numerical_operations) == {
        "integer:/": 1, "integer:+": 1, "conversion:integer->real": 1,
    }
    assert result.to_dict()["unpriced_numerical_operations"] == dict(result.unpriced_numerical_operations)


def test_integer_private_address_only_chain_does_not_claim_unpriced_numerical_work():
    j = Symbol(11, "address_intermediate", INTEGER)
    result = analyze(
        assign(K, Binary("+", ref(ITERATOR), literal(1, INTEGER))),
        assign(j, Binary("-", ref(K), literal(1, INTEGER))),
        assign(access(B, ref(j)), access(A, ref(j))),
        private=(K, j),
    )
    assert result.operations == ()
    assert result.unpriced_numerical_operations == ()


def test_address_use_cannot_hide_integer_work_also_used_as_data():
    result = analyze(
        assign(K, Binary("+", ref(ITERATOR), literal(1, INTEGER))),
        assign(access(B), Binary("+", access(A, ref(K)), ref(K))),
    )
    assert dict(result.unpriced_numerical_operations) == {
        "integer:+": 1, "implicit_real_integer_conversion": 1,
    }
    direct_index_data = analyze(assign(access(B), ref(ITERATOR)))
    assert direct_index_data.unpriced_numerical_operations == (("conversion:integer->real", 1),)


def test_non_address_logical_and_comparison_work_remains_unpriced():
    flag = Symbol(12, "flag", ScalarType.LOGICAL)
    result = analyze(
        assign(flag, Binary(">", access(A), literal(0))),
        assign(flag, Unary(".not.", ref(flag))),
        assign(access(B), access(A)),
        private=(flag,),
    )
    assert result.operations == ()
    assert dict(result.unpriced_numerical_operations) == {"comparison:>": 1, "logical:not": 1}


def test_unpriced_operations_are_bounded_independently_from_real_nodes():
    statements = (
        assign(K, Binary("+", ref(ITERATOR), literal(1, INTEGER))),
        assign(K, Binary("+", ref(K), literal(1, INTEGER))),
    )
    result = analyze_compute_dependencies(region(*statements), max_nodes=1)
    assert not result.available
    assert "unpriced numerical operation budget" in result.reason


def test_variadic_intrinsic_preserves_one_occurrence_and_pair_work():
    result = analyze(
        assign(X, Binary("*", access(A), literal(2))),
        assign(access(B), IntrinsicCall("max", (literal(0), ref(X), ref(X)), REAL)),
    )
    operation = result.operations[-1]
    assert operation.operands == ((), (0,), (0,))
    assert operation.work_units == 2
    assert result.weighted_span({"multiply": 2, "max": 3}) == 8


def test_first_read_then_same_cell_write_is_admitted():
    result = analyze(assign(access(A), Binary("+", access(A), literal(1))))
    assert result.to_dict()["operation_count"] == 1


def test_same_ast_index_after_private_reassignment_is_not_same_address():
    result = analyze_compute_dependencies(
        region(
            assign(K, ref(ITERATOR)),
            assign(X, access(A, ref(K))),
            assign(K, Binary("+", ref(K), literal(1, INTEGER))),
            assign(access(A, ref(K)), ref(X)),
        )
    )
    assert not result.available
    assert result.reason == "dependency mutable array uses different index definitions"
    assert result.to_dict()["operation_count"] is None


def test_private_unchanged_index_and_descriptor_index_are_supported():
    result = analyze(
        assign(K, Binary("-", ref(ITERATOR), literal(1, INTEGER))),
        assign(X, access(A, ref(K))),
        assign(access(A, ref(K)), Binary("+", ref(X), literal(1))),
        assign(access(B), IntrinsicCall("real", (Size(A, 1),), REAL)),
    )
    assert result.to_dict()["operation_count"] == 1


@pytest.mark.parametrize(
    "statements",
    [
        (assign(access(A), literal(1)), assign(access(B), access(A))),
        (assign(X, access(A)), assign(access(A, Binary("+", ref(ITERATOR), literal(1, INTEGER))), ref(X))),
    ],
)
def test_uncertain_array_memory_definitions_decline_cost_features(statements):
    result = analyze_compute_dependencies(region(*statements))
    assert not result.available
    assert "array" in result.reason
    assert result.operations == result.outputs == ()
    assert result.identity is None


def test_read_only_multiple_indices_are_admitted():
    result = analyze(
        assign(access(B), Binary("+", access(A), access(A, Binary("+", ref(ITERATOR), literal(1, INTEGER)))))
    )
    assert result.available


def test_indirect_and_real_derived_indices_decline():
    indices = Symbol(20, "indices", INTEGER, rank=1)
    for index in (access(indices), IntrinsicCall("int", (access(A),), INTEGER)):
        result = analyze_compute_dependencies(region(assign(access(B), access(A, index))))
        assert not result.available
        assert "coordinate" in result.reason or "conversion" in result.reason
    result = analyze_compute_dependencies(region(assign(K, access(indices)), assign(access(B), access(A, ref(K)))))
    assert not result.available
    assert "coordinates" in result.reason


@pytest.mark.parametrize(
    "statement",
    [
        If(Literal(".true.", ScalarType.LOGICAL), Block(()), Block(()), LOCATION),
        Loop(K, literal(1, INTEGER), literal(2, INTEGER), Block(()), LOCATION),
    ],
)
def test_control_flow_declines_cost_only(statement):
    original = region(statement)
    result = analyze_compute_dependencies(original)
    assert not result.available
    assert original.body.statements == (statement,)


def test_unpriced_intrinsic_and_undefined_private_are_unknown():
    for value in (IntrinsicCall("mod", (access(A), literal(2)), REAL), ref(X)):
        result = analyze_compute_dependencies(region(assign(access(B), value)))
        assert not result.available


def test_fixed_private_elements_have_independent_latest_definitions():
    first = Symbol(30, "element_0", REAL, private_array_origin=PrivateArrayOrigin(0, ((-1, 0),), 0))
    second = Symbol(31, "element_1", REAL, private_array_origin=PrivateArrayOrigin(0, ((-1, 0),), 1))
    result = analyze(
        assign(first, Binary("*", access(A), literal(2))),
        assign(second, Binary("*", access(A), literal(3))),
        assign(first, Binary("+", ref(first), literal(1))),
        assign(access(B), Binary("+", ref(first), ref(second))),
        private=(first, second),
    )
    assert result.operations[-1].operands == ((2,), (1,))


def test_lowered_private_helpers_preserve_counts_and_original_ir(tmp_path):
    source = tmp_path / "renamed.f90"
    source.write_text("""module renamed
contains
subroutine calculate(a,b,n)
integer,intent(in)::n
real(8),intent(in)::a(:)
real(8),intent(out)::b(:)
integer::i
do i=1,n
b(i)=measure(a(i))+measure(a(i)+1.0_8)
enddo
contains
pure function measure(x) result(y)
real(8),intent(in)::x
real(8)::y,small(-1:0)
small(-1)=x*x
small(0)=x+1.0_8
y=dot_product(small,small)
end function
end subroutine
end module
""")
    function = lower_file(source, "calculate")
    plan = build_execution_plan(function, options=CompilerOptions(opt_level=0))
    original = repr(function), repr(plan)
    (selected,) = plan.regions
    before = _compute_operation_counts(selected.body)
    result = analyze_compute_dependencies(selected)
    assert result.available, result.reason
    assert result.to_dict()["operation_count"] == before[0] == 14
    assert _compute_operation_counts(selected.body) == before
    assert (repr(function), repr(plan)) == original


def test_identity_ignores_symbol_names_but_retains_constants():
    original = analyze(assign(X, Binary("*", access(A), literal(2))), assign(access(B), ref(X)))
    renamed_a, renamed_b, renamed_x = (
        replace(A, name="other_input"),
        replace(B, name="other_output"),
        replace(X, name="other"),
    )
    renamed = analyze(
        assign(renamed_x, Binary("*", access(renamed_a), literal(2))),
        assign(access(renamed_b), ref(renamed_x)),
        private=(renamed_x,),
    )
    changed = analyze(assign(X, Binary("*", access(A), literal(3))), assign(access(B), ref(X)))
    assert original.identity == renamed.identity != changed.identity


def test_graph_and_public_records_are_immutable_and_detached():
    result = analyze(assign(access(B), Binary("+", access(A), literal(2))))
    with pytest.raises(FrozenInstanceError):
        result.operations[0].family = "multiply"
    public = json.loads(json.dumps(result.to_dict(include_graph=True)))
    public["operations"][0]["operands"].append([99])
    assert result.operations[0].operands == ((), ())
    assert "operations" not in result.to_dict()


@pytest.mark.parametrize("weights", [{}, {"add": True}, {"add": -1}, {"add": math.inf}, {"add": math.nan}])
def test_weighted_span_rejects_missing_or_unsafe_weights(weights):
    result = analyze(assign(access(B), Binary("+", access(A), literal(2))))
    with pytest.raises(ValueError, match="weight"):
        result.weighted_span(weights)


def test_weighted_span_rejects_overflow_and_unavailable_graph():
    result = analyze(assign(X, Binary("+", access(A), literal(2))), assign(access(B), Binary("+", ref(X), literal(2))))
    with pytest.raises(ValueError, match="overflows"):
        result.weighted_span({"add": 1e308})
    with pytest.raises(ValueError, match="unavailable"):
        ComputeDependencies(False, "unknown").weighted_span({})
    malformed = ComputeDependencies(True, operations=(ComputeOperation("add", REAL, ((0,),), False, False),))
    with pytest.raises(ValueError, match="topological"):
        malformed.weighted_span({"add": 1})


@pytest.mark.parametrize("budget", [0, -1, True, 1.5, MAX_OPERATIONS + 1])
def test_operation_budget_is_checked(budget):
    with pytest.raises(ValueError, match="max_nodes"):
        analyze_compute_dependencies(region(), max_nodes=budget)


def test_operation_budget_failure_retains_no_partial_graph():
    original = region(assign(X, Binary("+", access(A), literal(1))), assign(access(B), Binary("*", ref(X), literal(2))))
    result = analyze_compute_dependencies(original, max_nodes=1)
    assert not result.available
    assert result.reason == "dependency operation budget exceeded"
    assert result.operations == ()


def test_expression_depth_has_a_fixed_guard():
    expression = access(A)
    for _ in range(MAX_EXPRESSION_DEPTH + 1):
        expression = Unary("+", expression)
    result = analyze_compute_dependencies(region(assign(access(B), expression)))
    assert not result.available
    assert result.reason == "dependency expression depth budget exceeded"


@pytest.mark.parametrize(
    ("name", "limit", "reason"),
    [
        ("MAX_EDGES", 1, "edge"),
        ("MAX_VISITS", 1, "visit"),
        ("MAX_DEFINITIONS", 1, "definition"),
    ],
)
def test_independent_resource_guards(monkeypatch, name, limit, reason):
    # Shrink the public fixed budgets so each guard is tested without large
    # allocations, deep fixture parsing, or heavy compilation.
    monkeypatch.setattr("compiler.offload.compute_dependencies." + name, limit)
    result = analyze_compute_dependencies(
        region(assign(X, Binary("+", access(A), literal(1))), assign(access(B), Binary("*", ref(X), ref(X))))
    )
    assert not result.available
    assert reason in result.reason


def test_public_work_bounds_are_separate_from_planning_budget():
    assert MAX_OPERATIONS == 4096
    assert MAX_EDGES == 16384
    assert MAX_VISITS == 32768
    assert MAX_DEFINITIONS == 8192
    assert MAX_EXPRESSION_DEPTH == 128
    assert not analyze_compute_dependencies(Block(())).available
    empty = analyze()
    assert empty.to_dict()["operation_count"] == 0
    assert empty.weighted_span({}) == 0
