"""Render scalar types, expressions, and statements shared by C++ and CUDA."""

from __future__ import annotations

from typing import TYPE_CHECKING

from compiler.emission.common.abi import dimension_name
from compiler.ir import (
    ArrayAccess,
    Assignment,
    Binary,
    CompilationError,
    Expr,
    IntrinsicCall,
    Literal,
    Reference,
    ScalarType,
    Size,
    Symbol,
    Unary,
)
from compiler.ir.integers import INTEGER_MAX, INTEGER_MIN, integer_literal
from compiler.ir.intrinsics import REAL_MATH

if TYPE_CHECKING:
    from compiler.ir import RegionAddressing


def cpp_type(symbol: Symbol) -> str:
    return {
        ScalarType.INTEGER: "int",
        ScalarType.REAL: "double",
        ScalarType.REAL32: "float",
        ScalarType.LOGICAL: "bool",
    }[symbol.dtype]


def wide_iterator_name(symbol: Symbol) -> str:
    return f"fort_internal_wide{symbol.id}"


def _wide_expression(expression: Expr, addressing: RegionAddressing) -> str:
    """Render a proved subscript in one signed wide type, preserving its tree."""
    if isinstance(expression, Literal) and expression.dtype is ScalarType.INTEGER:
        value = integer_literal(expression.value)
        # Signed literals can occur in manually constructed IR. Parentheses
        # prevent a surrounding unary minus from becoming C++ decrement.
        return f"({value}LL)" if value < 0 else f"{value}LL"
    if isinstance(expression, Reference) and expression.symbol in addressing.wide_iterators:
        return wide_iterator_name(expression.symbol)
    if isinstance(expression, (Reference, Size, ArrayAccess)):
        # SIZE conversions and integer loads retain their source value type.
        # Nested array subscripts can independently consume their own decisions.
        return f"static_cast<long long>({render_expression(expression, addressing=addressing)})"
    if isinstance(expression, Unary) and expression.operator in {"+", "-"}:
        return f"({expression.operator}{_wide_expression(expression.operand, addressing)})"
    if isinstance(expression, Binary) and expression.operator in {"+", "-", "*", "/"}:
        left = _wide_expression(expression.left, addressing)
        right = _wide_expression(expression.right, addressing)
        return f"({left} {expression.operator} {right})"
    if isinstance(expression, IntrinsicCall) and expression.dtype is ScalarType.INTEGER:
        names = {"min": "minimum", "max": "maximum", "abs": "absolute"}
        name = names.get(expression.name.lower())
        if name is not None:
            arguments = ", ".join(_wide_expression(argument, addressing) for argument in expression.arguments)
            return f"::generated_kernels::numeric::{name}({arguments})"
    raise TypeError(f"Unsupported proved-wide subscript: {type(expression).__name__}")


def _render_subscript(expression: Expr, addressing: RegionAddressing | None) -> str:
    if addressing is not None and any(
        decision.expression == expression and decision.mode == "wide" for decision in addressing.decisions
    ):
        return _wide_expression(expression, addressing)
    return render_expression(expression, addressing=addressing)


def render_expression(expression: Expr, *, addressing: RegionAddressing | None = None) -> str:
    if isinstance(expression, Literal):
        if expression.dtype is ScalarType.LOGICAL:
            return "true" if expression.value.lower() in {".true.", "true"} else "false"
        if expression.dtype is ScalarType.INTEGER:
            # Fortran integer literals are decimal even with leading zeros.
            value = integer_literal(expression.value)
            return f"(-{INTEGER_MAX} - 1)" if value == INTEGER_MIN else str(value)
        # The frontend normalizes literals; accepting Fortran exponent spelling
        # here also makes standalone construction of the public IR convenient.
        value = expression.value.split("_", 1)[0].replace("d", "e").replace("D", "e")
        return value + "f" if expression.dtype is ScalarType.REAL32 else value
    if isinstance(expression, Reference):
        return expression.symbol.cpp_name
    if isinstance(expression, Size):
        return f"static_cast<int>({dimension_name(expression.symbol, expression.dimension)})"
    if isinstance(expression, ArrayAccess):
        values = [_render_subscript(index, addressing) for index in expression.indices]
        values.extend(dimension_name(expression.symbol, dim) for dim in range(1, expression.symbol.rank + 1))
        return f"{expression.symbol.cpp_name}[F_IDX({', '.join(values)})]"
    if isinstance(expression, IntrinsicCall):
        intrinsic = expression.name.lower()
        arguments = [render_expression(arg, addressing=addressing) for arg in expression.arguments]
        if intrinsic in {"real", "int", "dble"}:
            target = {ScalarType.REAL32: "float", ScalarType.REAL: "double", ScalarType.INTEGER: "int"}[
                expression.dtype
            ]
            return f"static_cast<{target}>({arguments[0]})"
        if intrinsic in {"min", "max", "mod", "modulo", "sign", "dim", "merge", "nint", "floor", "ceiling"}:
            name = {"min": "minimum", "max": "maximum"}.get(intrinsic, intrinsic)
            if intrinsic in {"nint", "floor", "ceiling"}:
                arguments = arguments[:1]
            return f"::generated_kernels::numeric::{name}({', '.join(arguments)})"
        if intrinsic == "abs":
            name = "abs" if expression.dtype is ScalarType.INTEGER else "fabs"
        elif intrinsic in REAL_MATH or intrinsic == "atan2":
            name = intrinsic
        else:
            raise CompilationError(f"unsupported emission intrinsic: {intrinsic}")
        if expression.dtype is ScalarType.REAL32:
            name += "f"
        return f"::{name}({', '.join(arguments)})"
    if isinstance(expression, Unary):
        operator = "!" if expression.operator == ".not." else expression.operator
        return f"({operator}{render_expression(expression.operand, addressing=addressing)})"
    if isinstance(expression, Binary):
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
        if operator not in {"+", "-", "*", "/", "<", ">", "<=", ">=", "==", "!=", "&&", "||"}:
            raise CompilationError(f"unsupported emission operator: {expression.operator}")
        return f"({render_expression(expression.left, addressing=addressing)} {operator} {render_expression(expression.right, addressing=addressing)})"
    raise CompilationError(f"unsupported IR expression: {type(expression).__name__}")


def render_assignment(assignment: Assignment, *, addressing: RegionAddressing | None = None) -> str:
    return (
        f"{render_expression(assignment.target, addressing=addressing)} = "
        f"{render_expression(assignment.value, addressing=addressing)};"
    )


def indent(lines: list[str], depth: int = 1) -> list[str]:
    return ["    " * depth + line if line else "" for line in lines]
