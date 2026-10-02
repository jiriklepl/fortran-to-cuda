"""Conservative default-INTEGER ranges without parser or dependence-engine facts.

An expression is safe to widen only when every arithmetic intermediate fits the
source kind. Loads and SIZE remain int-valued leaves: widening their consumers
never changes the evaluation of their subscripts or extent conversions.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from compiler.ir import (
    ArrayAccess,
    Binary,
    Expr,
    IntegerRange,
    IntrinsicCall,
    Literal,
    Loop,
    Reference,
    ScalarType,
    Size,
    Symbol,
    Unary,
)
from compiler.ir.integers import INTEGER_MAX, INTEGER_MIN, integer_literal

FULL_INTEGER_RANGE = IntegerRange(INTEGER_MIN, INTEGER_MAX)


@dataclass(frozen=True)
class RangeProof:
    interval: IntegerRange | None
    reason: str


def _checked(lower: int, upper: int, operation: str) -> RangeProof:
    if lower < INTEGER_MIN or upper > INTEGER_MAX:
        return RangeProof(None, f"{operation} may overflow default INTEGER")
    return RangeProof(IntegerRange(lower, upper), "all integer intermediates fit default INTEGER")


def _quotient(left: int, right: int) -> int:
    value = abs(left) // abs(right)
    return -value if (left < 0) != (right < 0) else value


def prove_range(expression: Expr, iterator_ranges: Mapping[Symbol, IntegerRange]) -> RangeProof:
    """Bound this arithmetic tree, without propagating assignments or predicates."""
    if isinstance(expression, Literal):
        if expression.dtype is not ScalarType.INTEGER:
            return RangeProof(None, "non-integer expression")
        value = integer_literal(expression.value)
        return _checked(value, value, "literal")
    if isinstance(expression, Reference):
        if expression.symbol.dtype is not ScalarType.INTEGER or expression.symbol.rank:
            return RangeProof(None, "non-integer expression")
        interval = iterator_ranges.get(expression.symbol, FULL_INTEGER_RANGE)
        return _checked(interval.lower, interval.upper, "reference")
    if isinstance(expression, Size):
        return RangeProof(FULL_INTEGER_RANGE, "SIZE retains its source INTEGER conversion")
    if isinstance(expression, ArrayAccess):
        if expression.symbol.dtype is not ScalarType.INTEGER:
            return RangeProof(None, "non-integer expression")
        return RangeProof(FULL_INTEGER_RANGE, "array load retains its source INTEGER value")
    if isinstance(expression, Unary):
        if expression.operator not in {"+", "-"}:
            return RangeProof(None, "unsupported integer unary operator")
        operand = prove_range(expression.operand, iterator_ranges)
        if operand.interval is None:
            return operand
        lower, upper = operand.interval.lower, operand.interval.upper
        return _checked(-upper, -lower, "unary minus") if expression.operator == "-" else operand
    if isinstance(expression, Binary):
        if expression.operator not in {"+", "-", "*", "/"}:
            return RangeProof(None, "unsupported integer binary operator")
        left = prove_range(expression.left, iterator_ranges)
        right = prove_range(expression.right, iterator_ranges)
        if left.interval is None:
            return left
        if right.interval is None:
            return right
        a, b = left.interval, right.interval
        if expression.operator == "+":
            return _checked(a.lower + b.lower, a.upper + b.upper, "addition")
        if expression.operator == "-":
            return _checked(a.lower - b.upper, a.upper - b.lower, "subtraction")
        if expression.operator == "*":
            products = [x * y for x in (a.lower, a.upper) for y in (b.lower, b.upper)]
            return _checked(min(products), max(products), "multiplication")
        if b.lower <= 0 <= b.upper:
            return RangeProof(None, "division denominator may be zero")
        if a.lower == INTEGER_MIN and b.lower <= -1 <= b.upper:
            return RangeProof(None, "division may overflow at INTEGER_MIN / -1")
        quotients = [_quotient(x, y) for x in (a.lower, a.upper) for y in (b.lower, b.upper)]
        return _checked(min(quotients), max(quotients), "division")
    if isinstance(expression, IntrinsicCall):
        if expression.dtype is not ScalarType.INTEGER:
            return RangeProof(None, "non-integer intrinsic")
        arguments = [prove_range(argument, iterator_ranges) for argument in expression.arguments]
        for argument in arguments:
            if argument.interval is None:
                return argument
        intervals = [argument.interval for argument in arguments]
        name = expression.name.lower()
        if name == "abs" and len(intervals) == 1:
            value = intervals[0]
            lower = 0 if value.lower <= 0 <= value.upper else min(abs(value.lower), abs(value.upper))
            return _checked(lower, max(abs(value.lower), abs(value.upper)), "ABS")
        if name in {"min", "max"} and len(intervals) >= 2:
            select = min if name == "min" else max
            return _checked(
                select(value.lower for value in intervals), select(value.upper for value in intervals), name.upper()
            )
        return RangeProof(None, "unsupported integer intrinsic")
    return RangeProof(None, "unsupported integer expression")


def iterator_range(loop: Loop) -> IntegerRange:
    """Bound executed mapped coordinates; preserve all header evaluation in IR.

    Each header denotes a source-width snapshot. If its arithmetic cannot be
    proved safe, no tighter facts than its resulting INTEGER type are assumed.
    Empty domains deliberately receive no refinement.
    """
    lower = prove_range(loop.lower, {}).interval or FULL_INTEGER_RANGE
    upper = prove_range(loop.upper, {}).interval or FULL_INTEGER_RANGE
    step = Literal(str(loop.step), ScalarType.INTEGER) if isinstance(loop.step, int) else loop.step
    stride = prove_range(step, {}).interval or FULL_INTEGER_RANGE
    if lower.lower == lower.upper and upper.lower == upper.upper and stride.lower == stride.upper:
        start, stop, increment = lower.lower, upper.lower, stride.lower
        if increment == 0 or (increment > 0 and start > stop) or (increment < 0 and start < stop):
            return FULL_INTEGER_RANGE
        count = (stop - start) // increment
        last = start + count * increment
        return IntegerRange(min(start, last), max(start, last))
    if stride.lower > 0:
        low, high = lower.lower, upper.upper
    elif stride.upper < 0:
        low, high = upper.lower, lower.upper
    else:
        low, high = min(lower.lower, upper.lower), max(lower.upper, upper.upper)
    return IntegerRange(low, high) if low <= high else FULL_INTEGER_RANGE
