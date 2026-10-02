"""Render scalar types, expressions, and statements shared by C++ and CUDA."""

from __future__ import annotations

from compiler.emission.common.abi import dimension_name
from compiler.ir import (
    ArrayAccess,
    Assignment,
    Binary,
    CompilationError,
    Expr,
    Literal,
    Reference,
    ScalarType,
    Size,
    Symbol,
    Unary,
)


def cpp_type(symbol: Symbol) -> str:
    return {ScalarType.INTEGER: "int", ScalarType.REAL: "double", ScalarType.REAL32: "float"}[symbol.dtype]


def render_expression(expression: Expr) -> str:
    if isinstance(expression, Literal):
        if expression.dtype is ScalarType.INTEGER:
            # Fortran integer literals are decimal even with leading zeros.
            return str(int(expression.value))
        # The frontend normalizes literals; accepting Fortran exponent spelling
        # here also makes standalone construction of the public IR convenient.
        value = expression.value.split("_", 1)[0].replace("d", "e").replace("D", "e")
        return value + "f" if expression.dtype is ScalarType.REAL32 else value
    if isinstance(expression, Reference):
        return expression.symbol.cpp_name
    if isinstance(expression, Size):
        return f"static_cast<int>({dimension_name(expression.symbol, expression.dimension)})"
    if isinstance(expression, ArrayAccess):
        values = [render_expression(index) for index in expression.indices]
        values.extend(dimension_name(expression.symbol, dim) for dim in range(1, expression.symbol.rank + 1))
        return f"{expression.symbol.cpp_name}[F_IDX({', '.join(values)})]"
    if isinstance(expression, Unary):
        return f"({expression.operator}{render_expression(expression.operand)})"
    if isinstance(expression, Binary):
        if expression.operator not in {"+", "-", "*", "/"}:
            raise CompilationError(f"unsupported emission operator: {expression.operator}")
        return f"({render_expression(expression.left)} {expression.operator} {render_expression(expression.right)})"
    raise CompilationError(f"unsupported IR expression: {type(expression).__name__}")


def render_assignment(assignment: Assignment) -> str:
    return f"{render_expression(assignment.target)} = {render_expression(assignment.value)};"


def indent(lines: list[str], depth: int = 1) -> list[str]:
    return ["    " * depth + line if line else "" for line in lines]
