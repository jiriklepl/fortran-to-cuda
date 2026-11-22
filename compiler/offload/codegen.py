"""Checked wide-arithmetic query expressions for generated entries."""

from compiler.emission.common.abi import dimension_name
from compiler.ir import Binary, IntrinsicCall, Literal, Reference, Size, Unary
from compiler.ir.integers import integer_literal


def query_expression(expression):
    """Wide, total arithmetic for analyzer-approved physical/bound expressions."""
    if isinstance(expression, Literal):
        return f"({integer_literal(expression.value)}.0L)"
    if isinstance(expression, Reference):
        return f"static_cast<long double>({expression.symbol.cpp_name})"
    if isinstance(expression, Size):
        return f"static_cast<long double>({dimension_name(expression.symbol, expression.dimension)})"
    if isinstance(expression, Unary):
        return f"({expression.operator}{query_expression(expression.operand)})"
    if isinstance(expression, Binary):
        value = f"({query_expression(expression.left)} {expression.operator} {query_expression(expression.right)})"
        return f"std::trunc({value})" if expression.operator == "/" else value
    if isinstance(expression, IntrinsicCall) and expression.name.lower() in {"min", "max"}:
        return f"std::{expression.name.lower()}({{{', '.join(query_expression(v) for v in expression.arguments)}}})"
    raise ValueError(f"Unsafe cost expression: {expression!r}")
