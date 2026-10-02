"""Bound source-width arithmetic before selecting wide subscript evaluation."""

from dataclasses import FrozenInstanceError, replace
from itertools import product

import pytest

from compiler.addressing import plan_addressing
from compiler.addressing.ranges import FULL_INTEGER_RANGE, iterator_range, prove_range
from compiler.driver.options import CompilerOptions
from compiler.ir import (
    ArrayAccess,
    Assignment,
    Binary,
    Block,
    ConditionalRegion,
    ExecutionPlan,
    HostBlock,
    If,
    IntegerRange,
    IntrinsicCall,
    Literal,
    Loop,
    ParallelRegion,
    Reference,
    RegionReport,
    RegionSchedule,
    ScalarType,
    SequentialRegion,
    Size,
    SourceLocation,
    Symbol,
    Unary,
)
from compiler.ir.integers import INTEGER_MAX, INTEGER_MIN

INTEGER = ScalarType.INTEGER
ITERATOR = Symbol(0, "i", INTEGER)
INNER = Symbol(1, "j", INTEGER)
N = Symbol(2, "n", INTEGER, intent="in", parameter=True)
TEMP = Symbol(3, "temporary", INTEGER)
A = Symbol(4, "a", INTEGER, rank=1, intent="inout", parameter=True)
INDICES = Symbol(5, "indices", INTEGER, rank=1, intent="in", parameter=True)
LOCATION = SourceLocation("addressing.f90", 12)
POINT = Reference(ITERATOR)
OTHER = Reference(INNER)


def integer(value):
    return Literal(str(value), INTEGER)


def intrinsic(name, *arguments):
    return IntrinsicCall(name, arguments, INTEGER)


def region(*, body=None, lower=None, upper=None, step=1):
    lower = integer(2) if lower is None else lower
    upper = Binary("+", Reference(N), integer(1)) if upper is None else upper
    body = body or Block((Assignment(ArrayAccess(A, (POINT,)), integer(1), LOCATION),))
    loop = Loop(ITERATOR, lower, upper, body, LOCATION, step)
    return ParallelRegion(
        0,
        (loop,),
        (),
        (),
        (A, N),
        RegionReport("", "", "", "", "", "", ""),
        body,
        schedule=RegionSchedule((0,), (4,)),
    )


def planned(item, **options):
    return plan_addressing(ExecutionPlan((item,)), options=CompilerOptions(**options)).regions[0]


@pytest.mark.parametrize(
    ("expression", "bounds", "expected"),
    [
        (POINT, IntegerRange(2, 9), IntegerRange(2, 9)),
        (Binary("-", POINT, integer(1)), IntegerRange(2, INTEGER_MAX), IntegerRange(1, INTEGER_MAX - 1)),
        (Unary("-", POINT), IntegerRange(-9, -2), IntegerRange(2, 9)),
        (Binary("*", POINT, integer(-2)), IntegerRange(-9, 2), IntegerRange(-4, 18)),
        (Binary("/", POINT, integer(3)), IntegerRange(-8, -2), IntegerRange(-2, 0)),
        (Binary("/", POINT, integer(-3)), IntegerRange(-8, 7), IntegerRange(-2, 2)),
        (Binary("/", POINT, integer(INTEGER_MIN)), FULL_INTEGER_RANGE, IntegerRange(0, 1)),
        (intrinsic("abs", POINT), IntegerRange(-9, 2), IntegerRange(0, 9)),
        (intrinsic("abs", POINT), IntegerRange(-9, -2), IntegerRange(2, 9)),
        (intrinsic("min", POINT, integer(7)), FULL_INTEGER_RANGE, IntegerRange(INTEGER_MIN, 7)),
        (intrinsic("max", POINT, integer(2)), FULL_INTEGER_RANGE, IntegerRange(2, INTEGER_MAX)),
        (Binary("+", POINT, Binary("*", Size(A, 1), integer(0))), IntegerRange(2, 9), IntegerRange(2, 9)),
    ],
)
def test_integer_ranges_preserve_signed_arithmetic(expression, bounds, expected):
    assert prove_range(expression, {ITERATOR: bounds}).interval == expected


@pytest.mark.parametrize(
    ("expression", "bounds", "reason"),
    [
        (Binary("+", POINT, integer(1)), FULL_INTEGER_RANGE, "addition"),
        (Binary("-", POINT, integer(1)), FULL_INTEGER_RANGE, "subtraction"),
        (Binary("*", POINT, integer(2)), FULL_INTEGER_RANGE, "multiplication"),
        (Unary("-", POINT), FULL_INTEGER_RANGE, "unary minus"),
        (intrinsic("abs", POINT), FULL_INTEGER_RANGE, "ABS"),
        (Binary("/", POINT, integer(-1)), FULL_INTEGER_RANGE, "INTEGER_MIN / -1"),
        (Binary("/", integer(1), POINT), IntegerRange(-1, 1), "zero"),
        (Binary("/", POINT, Reference(N)), IntegerRange(2, 9), "zero"),
        (Binary("+", Binary("-", POINT, integer(1)), integer(1)), FULL_INTEGER_RANGE, "subtraction"),
        (intrinsic("min", Binary("+", POINT, integer(1)), integer(10)), FULL_INTEGER_RANGE, "addition"),
        (Binary("+", POINT, Size(A, 1)), IntegerRange(2, 9), "addition"),
        (Binary("+", POINT, ArrayAccess(INDICES, (integer(1),))), IntegerRange(2, 9), "addition"),
    ],
)
def test_unsafe_intermediates_cannot_be_hidden_by_parent_expressions(expression, bounds, reason):
    proof = prove_range(expression, {ITERATOR: bounds})
    assert proof.interval is None
    assert reason in proof.reason


def _evaluate(expression, values):
    """Independent exact evaluator; validate every source arithmetic intermediate."""
    if isinstance(expression, Literal):
        result = int(expression.value)
    elif isinstance(expression, Reference):
        result = values[expression.symbol]
    elif isinstance(expression, Unary):
        result = -_evaluate(expression.operand, values)
    elif isinstance(expression, IntrinsicCall):
        arguments = [_evaluate(argument, values) for argument in expression.arguments]
        result = abs(arguments[0]) if expression.name == "abs" else {"min": min, "max": max}[expression.name](arguments)
    else:
        left = _evaluate(expression.left, values)
        right = _evaluate(expression.right, values)
        if expression.operator == "+":
            result = left + right
        elif expression.operator == "-":
            result = left - right
        elif expression.operator == "*":
            result = left * right
        else:
            if right == 0:
                raise ArithmeticError("zero divisor")
            quotient, _ = divmod(abs(left), abs(right))
            result = quotient if (left >= 0) == (right >= 0) else -quotient
    if not INTEGER_MIN <= result <= INTEGER_MAX:
        raise ArithmeticError("integer overflow")
    return result


@pytest.mark.parametrize(
    "expression",
    [
        *(Binary(operator, POINT, OTHER) for operator in ("+", "-", "*", "/")),
        Unary("-", POINT),
        intrinsic("abs", POINT),
        intrinsic("min", POINT, OTHER, integer(0)),
        intrinsic("max", POINT, OTHER, integer(0)),
        Binary("+", Binary("*", POINT, OTHER), integer(1)),
        Binary("-", Binary("+", POINT, integer(INTEGER_MAX)), integer(INTEGER_MAX)),
        Binary("/", POINT, intrinsic("max", integer(1), OTHER)),
    ],
)
def test_range_proofs_are_sound_against_enumerated_source_arithmetic(expression):
    intervals = [
        IntegerRange(-3, -1),
        IntegerRange(-2, 2),
        IntegerRange(0, 3),
        IntegerRange(1, 3),
        IntegerRange(INTEGER_MIN, INTEGER_MIN + 2),
        IntegerRange(INTEGER_MAX - 2, INTEGER_MAX),
    ]
    accepted = 0
    for left, right in product(intervals, repeat=2):
        proof = prove_range(expression, {ITERATOR: left, INNER: right})
        if proof.interval is None:
            continue
        accepted += 1
        for x, y in product(range(left.lower, left.upper + 1), range(right.lower, right.upper + 1)):
            result = _evaluate(expression, {ITERATOR: x, INNER: y})
            assert proof.interval.lower <= result <= proof.interval.upper
    assert accepted


@pytest.mark.parametrize(
    ("lower", "upper", "step", "expected"),
    [
        (2, 10, 3, IntegerRange(2, 8)),
        (10, 2, -3, IntegerRange(4, 10)),
        (0, INTEGER_MAX, 2, IntegerRange(0, INTEGER_MAX - 1)),
        (INTEGER_MAX, 0, INTEGER_MIN, IntegerRange(INTEGER_MAX, INTEGER_MAX)),
        (INTEGER_MIN, INTEGER_MIN + 4, 3, IntegerRange(INTEGER_MIN, INTEGER_MIN + 3)),
        (2, 10, Reference(N), IntegerRange(2, 10)),
        (10, 2, Reference(N), IntegerRange(2, 10)),
        (2, 1, 1, FULL_INTEGER_RANGE),
        (1, 2, -1, FULL_INTEGER_RANGE),
    ],
)
def test_iterator_ranges_cover_signed_runtime_and_boundary_strides(lower, upper, step, expected):
    item = region(lower=integer(lower), upper=integer(upper), step=step)
    assert iterator_range(item.loops[0]) == expected


def test_bounded_runtime_headers_provide_only_proven_snapshot_ranges():
    item = region(upper=intrinsic("min", Reference(N), integer(INTEGER_MAX - 1)))
    assert iterator_range(item.loops[0]) == IntegerRange(2, INTEGER_MAX - 1)
    # n+1 need not be representable for arbitrary n; its snapshot has only type facts.
    assert iterator_range(region().loops[0]) == IntegerRange(2, INTEGER_MAX)
    item = region(lower=intrinsic("max", Reference(N), integer(3)), upper=integer(1), step=-1)
    assert iterator_range(item.loops[0]) == IntegerRange(1, INTEGER_MAX)


def test_iterator_range_soundness_for_small_counted_domains():
    for lower, upper, step in product(range(-4, 5), range(-4, 5), (-3, -2, -1, 1, 2, 3)):
        loop = region(lower=integer(lower), upper=integer(upper), step=step).loops[0]
        interval = iterator_range(loop)
        values = range(lower, upper + (1 if step > 0 else -1), step)
        assert all(interval.lower <= value <= interval.upper for value in values)
        if values:
            assert interval == IntegerRange(min(values), max(values))


def test_planner_mixes_source_and_wide_subscripts_without_changing_ir():
    minus = Binary("-", POINT, integer(1))
    plus = Binary("+", POINT, integer(1))
    body = Block(
        (
            Assignment(
                ArrayAccess(A, (POINT,)), Binary("+", ArrayAccess(A, (minus,)), ArrayAccess(A, (plus,))), LOCATION
            ),
        )
    )
    original = region(body=body)
    result = planned(original)
    decisions = {decision.expression: decision for decision in result.addressing.decisions}
    assert decisions[POINT].mode == decisions[minus].mode == "wide"
    assert decisions[plus].mode == "source"
    assert "addition" in decisions[plus].reason
    assert result.addressing.wide_iterators == (ITERATOR,)
    assert result.body is original.body
    assert result.loops is original.loops
    assert result.schedule is original.schedule
    assert original.addressing is None
    with pytest.raises(FrozenInstanceError):
        result.addressing.wide_iterators = ()


def test_reports_count_repeated_occurrences_with_source_symbol_names():
    body = Block((Assignment(ArrayAccess(A, (POINT,)), ArrayAccess(A, (POINT,)), LOCATION),))
    result = plan_addressing(ExecutionPlan((region(body=body),)), options=CompilerOptions())
    assert result.regions[0].addressing.decisions[0].occurrences == 2
    assert "region 0: auto indexing, 2/2 subscripts widened" in result.reports
    assert any("i: wide" in line and "2 occurrence(s)" in line for line in result.reports)
    assert all("fort_v" not in line for line in result.reports)


@pytest.mark.parametrize("options", [{"indexing": "source"}, {"opt_level": 0}])
def test_source_mode_keeps_decisions_native(options):
    result = planned(region(), **options)
    assert not result.addressing.wide_iterators
    assert all(decision.mode == "source" for decision in result.addressing.decisions)
    assert all(decision.reason == "source indexing requested" for decision in result.addressing.decisions)
    assert planned(region(), opt_level=0, indexing="auto").addressing.wide_iterators == (ITERATOR,)


def test_nested_subscripts_are_decided_separately_from_integer_loads():
    outer = ArrayAccess(INDICES, (POINT,))
    body = Block((Assignment(ArrayAccess(A, (outer,)), integer(1), LOCATION),))
    result = planned(region(body=body))
    decisions = {decision.expression: decision for decision in result.addressing.decisions}
    assert decisions[outer].mode == "source"
    assert decisions[outer].interval == FULL_INTEGER_RANGE
    assert decisions[POINT].mode == "wide"
    assert result.addressing.wide_iterators == (ITERATOR,)


def test_bounds_and_sequential_headers_are_excluded_even_for_equal_subscripts():
    point_access = ArrayAccess(A, (POINT,))
    body = Block(
        (
            Loop(
                INNER,
                point_access,
                point_access,
                Block((Assignment(point_access, integer(1), LOCATION),)),
                LOCATION,
                point_access,
            ),
        )
    )
    item = region(body=body, lower=ArrayAccess(INDICES, (integer(1),)), upper=integer(4))
    result = planned(item)
    assert len(result.addressing.decisions) == 1
    assert result.addressing.decisions[0].expression == POINT
    assert result.addressing.decisions[0].occurrences == 1
    assert item.body is result.body


def test_body_predicates_count_accesses_without_refining_branch_ranges_or_assignments():
    plus = Binary("+", POINT, integer(1))
    body = Block(
        (
            Assignment(Reference(TEMP), POINT, LOCATION),
            If(
                Binary("<", ArrayAccess(A, (POINT,)), integer(0)),
                Block((Assignment(ArrayAccess(A, (plus,)), integer(1), LOCATION),)),
                Block((Assignment(ArrayAccess(A, (Reference(TEMP),)), integer(1), LOCATION),)),
                LOCATION,
            ),
        )
    )
    decisions = {decision.expression: decision for decision in planned(region(body=body)).addressing.decisions}
    assert decisions[POINT].mode == "wide"
    assert decisions[plus].mode == "source"
    assert decisions[Reference(TEMP)].mode == "source"
    assert decisions[Reference(TEMP)].interval == FULL_INTEGER_RANGE


def test_host_and_fallback_steps_remain_unchanged_while_branch_regions_are_planned():
    assignment = Assignment(ArrayAccess(A, (integer(1),)), integer(1), LOCATION)
    host = HostBlock((assignment,))
    sequential = SequentialRegion(1, Block((assignment,)), "carried dependence")
    branch = ConditionalRegion(
        Binary(">", Reference(N), integer(0)),
        ExecutionPlan((host, region())),
        ExecutionPlan((sequential, replace(region(), id=2))),
        LOCATION,
    )
    result = plan_addressing(ExecutionPlan((branch,)), options=CompilerOptions())
    assert result.steps[0].then_plan.steps[0] is host
    assert result.steps[0].else_plan.steps[0] is sequential
    assert all(item.addressing.wide_iterators == (ITERATOR,) for item in result.regions)


def test_empty_outer_domain_does_not_evaluate_or_rewrite_inner_header_accesses():
    invalid_at_runtime = Binary("/", integer(1), ArrayAccess(INDICES, (integer(100),)))
    inner = Loop(INNER, invalid_at_runtime, integer(4), Block(()), LOCATION, Reference(N))
    outer = region(lower=integer(2), upper=integer(1))
    item = replace(outer, loops=(*outer.loops, inner))
    result = planned(item)
    assert result.loops is item.loops
    assert dict(result.addressing.iterator_ranges)[ITERATOR] == FULL_INTEGER_RANGE
    assert result.addressing.decisions[0].expression == POINT
