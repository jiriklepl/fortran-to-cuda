"""Immutable computation IR shared by lowering, analysis, and code generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ScalarType(Enum):
    INTEGER = "integer"
    REAL = "real"
    REAL32 = "real32"
    LOGICAL = "logical"


@dataclass(frozen=True)
class SourceLocation:
    path: str
    line: int = 1
    call_stack: tuple[str, ...] = ()

    def __str__(self) -> str:
        origin = f"{self.path}:{self.line}"
        return origin + (f" (inlined through {' -> '.join(self.call_stack)})" if self.call_stack else "")


class CompilationError(Exception):
    def __init__(self, message: str, location: SourceLocation | None = None):
        self.message = message
        self.location = location
        super().__init__(f"{location}: {message}" if location else message)


@dataclass(frozen=True)
class PrivateArrayOrigin:
    """Cost provenance for one scalarized element of original fixed storage.

    ``group_id`` is local to the lowered function, including its inlined helper
    activations. Bounds describe the original declared storage, before a helper's
    dummy association rebases coordinates. This metadata proves no numerical
    legality, native participation, or calibrated workload applicability.
    """

    group_id: int
    bounds: tuple[tuple[int, int], ...]
    element_offset: int
    schema_version: int = 1

    @property
    def extents(self) -> tuple[int, ...]:
        return tuple(max(0, upper - lower + 1) for lower, upper in self.bounds)


@dataclass(frozen=True)
class Symbol:
    id: int
    name: str
    dtype: ScalarType
    rank: int = 0
    intent: str | None = None
    parameter: bool = False
    # Cost metadata must not change numerical symbol identity or legality.
    private_array_origin: PrivateArrayOrigin | None = field(default=None, compare=False)

    @property
    def cpp_name(self) -> str:
        # Leave room for extent suffixes in the Fortran C interface (63
        # characters maximum). The identity prefix prevents truncation collisions.
        prefix = f"fort_v{self.id}_"
        return prefix + self.name.lower()[: 57 - len(prefix)]


@dataclass(frozen=True)
class Literal:
    value: str
    dtype: ScalarType


@dataclass(frozen=True)
class Reference:
    symbol: Symbol


@dataclass(frozen=True)
class ArrayAccess:
    symbol: Symbol
    indices: tuple[Expr, ...]


@dataclass(frozen=True)
class Unary:
    operator: str
    operand: Expr


@dataclass(frozen=True)
class Binary:
    operator: str
    left: Expr
    right: Expr


@dataclass(frozen=True)
class Size:
    symbol: Symbol
    dimension: int


@dataclass(frozen=True)
class IntrinsicCall:
    name: str
    arguments: tuple[Expr, ...]
    dtype: ScalarType


Expr = Literal | Reference | ArrayAccess | Unary | Binary | Size | IntrinsicCall


@dataclass(frozen=True)
class Assignment:
    target: Reference | ArrayAccess
    value: Expr
    location: SourceLocation


@dataclass(frozen=True)
class Block:
    statements: tuple[Statement, ...]


@dataclass(frozen=True)
class Loop:
    iterator: Symbol
    lower: Expr
    upper: Expr
    body: Block
    location: SourceLocation
    step: Expr | int = 1


@dataclass(frozen=True)
class If:
    condition: Expr
    then_body: Block
    else_body: Block
    location: SourceLocation


Statement = Assignment | Loop | If


@dataclass(frozen=True)
class FunctionIR:
    name: str
    module: str
    parameters: tuple[Symbol, ...]
    symbols: tuple[Symbol, ...]
    body: Block
    source: str


def walk_expr(expression: Expr):
    """Yield expression nodes in source order, including the root."""
    yield expression
    if isinstance(expression, Binary):
        yield from walk_expr(expression.left)
        yield from walk_expr(expression.right)
    elif isinstance(expression, Unary):
        yield from walk_expr(expression.operand)
    elif isinstance(expression, IntrinsicCall):
        for argument in expression.arguments:
            yield from walk_expr(argument)
    elif isinstance(expression, ArrayAccess):
        for index in expression.indices:
            yield from walk_expr(index)


def referenced_symbols(expression: Expr) -> frozenset[Symbol]:
    return frozenset(node.symbol for node in walk_expr(expression) if isinstance(node, (Reference, ArrayAccess, Size)))


def statement_reads(statement: Statement) -> frozenset[Symbol]:
    if isinstance(statement, Assignment):
        result = referenced_symbols(statement.value)
        if isinstance(statement.target, ArrayAccess):
            for index in statement.target.indices:
                result |= referenced_symbols(index)
        return result
    if isinstance(statement, If):
        return (
            referenced_symbols(statement.condition)
            | block_reads(statement.then_body)
            | block_reads(statement.else_body)
        )
    step_reads = referenced_symbols(statement.step) if not isinstance(statement.step, int) else frozenset()
    return (
        referenced_symbols(statement.lower)
        | referenced_symbols(statement.upper)
        | step_reads
        | block_reads(statement.body)
    )


def block_reads(block: Block) -> frozenset[Symbol]:
    result: frozenset[Symbol] = frozenset()
    for statement in block.statements:
        result |= statement_reads(statement)
    return result


def block_writes(block: Block) -> frozenset[Symbol]:
    result: frozenset[Symbol] = frozenset()
    for statement in block.statements:
        if isinstance(statement, Assignment):
            result |= {statement.target.symbol}
        elif isinstance(statement, If):
            result |= block_writes(statement.then_body) | block_writes(statement.else_body)
        else:
            result |= {statement.iterator} | block_writes(statement.body)
    return result


def format_ir(function: FunctionIR) -> str:
    def expression(expr: Expr) -> str:
        if isinstance(expr, Literal):
            return expr.value
        if isinstance(expr, Reference):
            return expr.symbol.cpp_name
        if isinstance(expr, ArrayAccess):
            return f"{expr.symbol.cpp_name}({', '.join(expression(i) for i in expr.indices)})"
        if isinstance(expr, Size):
            return f"size({expr.symbol.cpp_name}, {expr.dimension})"
        if isinstance(expr, IntrinsicCall):
            return f"{expr.name}({', '.join(expression(arg) for arg in expr.arguments)})"
        if isinstance(expr, Unary):
            return f"({expr.operator}{expression(expr.operand)})"
        return f"({expression(expr.left)} {expr.operator} {expression(expr.right)})"

    def block(value: Block, depth: int):
        for statement in value.statements:
            prefix = "  " * depth
            if isinstance(statement, Assignment):
                yield f"{prefix}{expression(statement.target)} = {expression(statement.value)}"
            elif isinstance(statement, If):
                yield f"{prefix}if ({expression(statement.condition)}) then"
                yield from block(statement.then_body, depth + 1)
                if statement.else_body.statements:
                    yield prefix + "else"
                    yield from block(statement.else_body, depth + 1)
                yield prefix + "end if"
            else:
                step = str(statement.step) if isinstance(statement.step, int) else expression(statement.step)
                yield f"{prefix}do {statement.iterator.cpp_name} = {expression(statement.lower)}, {expression(statement.upper)}, {step}"
                yield from block(statement.body, depth + 1)
                yield prefix + "end do"

    return "\n".join([f"function {function.module}::{function.name}", *block(function.body, 1)])
