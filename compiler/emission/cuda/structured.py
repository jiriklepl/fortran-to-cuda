"""Ordered scalar preparation and guarded kernels with shared section buffers."""


from compiler.emission.common.abi import dimension_name
from compiler.emission.common.c_family import render_expression
from compiler.ir import (
    ArrayAccess,
    Binary,
    IntrinsicCall,
    Literal,
    Reference,
    Size,
    Unary,
)


def _query_expression(expression, scalar_indices):
    """Render source-width checked operations; invalid preparation stops reads."""

    def q(value):
        return _query_expression(value, scalar_indices)

    if isinstance(expression, Literal):
        result = render_expression(expression)
    elif isinstance(expression, Reference):
        name = expression.symbol.cpp_name
        if expression.symbol in scalar_indices:
            result = f"(coverage[{scalar_indices[expression.symbol]}] = true, {name})"
        else:
            result = name
    elif isinstance(expression, Size):
        result = f"offload::query_size({dimension_name(expression.symbol, expression.dimension)}, d.valid)"
    elif isinstance(expression, ArrayAccess):
        dims = ", ".join(dimension_name(expression.symbol, a + 1) for a in range(expression.symbol.rank))
        indices = ", ".join(q(value) for value in expression.indices)
        result = f"offload::query_load({expression.symbol.cpp_name}, {{{dims}}}, {{{indices}}}, d.valid)"
    elif isinstance(expression, Unary):
        operand = q(expression.operand)
        result = (
            f"!({operand})"
            if expression.operator == ".not."
            else f"offload::query_integer({expression.operator}static_cast<long long>({operand}), d.valid)"
        )
    elif isinstance(expression, Binary):
        # Locals force checked evaluation in source order, including both
        # operands of LOGICAL expressions (no reliance on short circuiting).
        left, right = q(expression.left), q(expression.right)
        operators = {
            ".and.": "&&",
            ".or.": "||",
            ".eqv.": "==",
            ".neqv.": "!=",
            "/=": "!=",
            ".eq.": "==",
            ".ne.": "!=",
            ".lt.": "<",
            ".le.": "<=",
            ".gt.": ">",
            ".ge.": ">=",
        }
        operator = operators.get(expression.operator, expression.operator)
        if operator == "/":
            result = "offload::query_divide(left, right, d.valid)"
        elif operator in {"+", "-", "*"}:
            result = f"offload::query_integer(static_cast<long long>(left) {operator} right, d.valid)"
        else:
            result = f"left {operator} right"
        return (
            "([&]() -> int { if (!d.valid) return 0; const int left = "
            + left
            + "; if (!d.valid) return 0; const int right = "
            + right
            + "; if (!d.valid) return 0; return "
            + result
            + "; }())"
        )
    elif isinstance(expression, IntrinsicCall):
        result = f"std::{expression.name.lower()}({{{', '.join(q(a) for a in expression.arguments)}}})"
    else:
        raise TypeError(expression)
    return f"([&]() -> int {{ if (!d.valid) return 0; return {result}; }}())"
