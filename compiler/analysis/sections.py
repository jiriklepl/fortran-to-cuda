"""Whether two array sections with symbolic bounds must share an element."""
import islpy as isl

from compiler.ir import Binary, Literal, Reference, ScalarType, Size, Unary


class _NotAffine(Exception):
    pass


def _affine(expression):
    """An integer expression as ``{name: coefficient}``; the constant has the name None."""
    if isinstance(expression, int):
        return {None: expression}
    if isinstance(expression, Literal) and expression.dtype is ScalarType.INTEGER:
        return {None: int(expression.value)}
    if isinstance(expression, Reference) and expression.symbol.dtype is ScalarType.INTEGER and not expression.symbol.rank:
        return {f"s{expression.symbol.id}": 1}
    if isinstance(expression, Size):
        return {f"d{expression.symbol.id}_{expression.dimension}": 1}
    if isinstance(expression, Unary) and expression.operator in {"+", "-"}:
        sign = -1 if expression.operator == "-" else 1
        return {name: sign * value for name, value in _affine(expression.operand).items()}
    if isinstance(expression, Binary) and expression.operator in {"+", "-", "*"}:
        left, right = _affine(expression.left), _affine(expression.right)
        if expression.operator == "*":
            if set(left) <= {None}:
                return {name: left.get(None, 0) * value for name, value in right.items()}
            if set(right) <= {None}:
                return {name: right.get(None, 0) * value for name, value in left.items()}
            raise _NotAffine
        sign = -1 if expression.operator == "-" else 1
        return {name: left.get(name, 0) + sign * right.get(name, 0) for name in {*left, *right}}
    raise _NotAffine


def _difference(upper, lower):
    high, low = _affine(upper), _affine(lower)
    return {name: high.get(name, 0) - low.get(name, 0) for name in {*high, *low}}


def _nonnegative(terms):
    return " + ".join(f"{value}" if name is None else f"{value}*{name}" for name, value in terms.items()) + " >= 0"


def must_overlap(first, second, nonempty=()):
    """Whether the boxes ``first`` and ``second`` of one array share an element whatever their bounds are.

    ``nonempty`` lists ``(lower, upper)`` ranges known to hold a value, as the
    loops of a nest that runs do; they are what relates the symbols in the
    bounds to each other. The answer is True as well when it cannot be
    decided because a bound is not an affine integer expression.
    """
    try:
        facts = [_difference(upper, lower) for lower, upper in nonempty]
        wanted = []
        for dimension in range(len(first.lower)):
            wanted.append(_difference(second.upper[dimension], first.lower[dimension]))
            wanted.append(_difference(first.upper[dimension], second.lower[dimension]))
    except _NotAffine:
        return True
    names = sorted({name for terms in (*facts, *wanted) for name in terms if name is not None})
    prefix = "[" + ", ".join(names) + "] -> { : "
    known = isl.Set(prefix + " and ".join(_nonnegative(terms) for terms in facts or [{None: 0}]) + " }")
    overlap = isl.Set(prefix + " and ".join(_nonnegative(terms) for terms in wanted) + " }")
    return known.is_subset(overlap)
