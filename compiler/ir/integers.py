"""Checked constants for the compiler's default INTEGER (signed 32-bit) ABI.

This models compile-time expressions only. Unknown runtime values stay unknown;
no arithmetic is reassociated, widened, or replaced with wrapping arithmetic.
"""

from .nodes import (
    ArrayAccess,
    Binary,
    CompilationError,
    Expr,
    IntrinsicCall,
    Literal,
    Reference,
    ScalarType,
    Size,
    SourceLocation,
    Unary,
)

INTEGER_MIN = -(1 << 31)
INTEGER_MAX = (1 << 31) - 1


def _checked(value: int, location: SourceLocation | None, *, literal: bool = False) -> int:
    if not INTEGER_MIN <= value <= INTEGER_MAX:
        reason = "literal is out of range" if literal else "constant expression overflows"
        raise CompilationError(f"default INTEGER {reason} [{INTEGER_MIN}, {INTEGER_MAX}]", location)
    return value


def integer_literal(value: str, location: SourceLocation | None = None, *, source_token: bool = False) -> int:
    """Validate decimal literals; source tokens are unsigned, before unary signs."""
    try:
        result = int(value, 10)
    except ValueError as error:
        raise CompilationError("invalid default INTEGER literal", location) from error
    if source_token and result < 0:
        raise CompilationError("default INTEGER source literal must be unsigned", location)
    return _checked(result, location, literal=True)


def constant_integer(expression: Expr, location: SourceLocation | None = None) -> int | None:
    """Evaluate integer constants and check every subtree, even beneath unknowns.

    The internal type flag distinguishes an unknown integer from a real operand:
    ``n / 0`` is invalid integer division, while ``real_value / 0`` is not an
    integer operation. All children are visited before a parent is considered.
    """

    def visit(node) -> tuple[bool, int | None]:
        if isinstance(node, Literal):
            integer = node.dtype is ScalarType.INTEGER
            return integer, integer_literal(node.value, location) if integer else None
        if isinstance(node, Reference):
            return node.symbol.dtype is ScalarType.INTEGER, None
        if isinstance(node, Size):
            return True, None
        if isinstance(node, ArrayAccess):
            for index in node.indices:
                visit(index)
            return node.symbol.dtype is ScalarType.INTEGER, None
        if isinstance(node, Unary):
            integer, value = visit(node.operand)
            if not integer or node.operator not in {"+", "-"}:
                return False, None
            if value is None:
                return True, None
            return True, _checked(-value if node.operator == "-" else value, location)
        if isinstance(node, Binary):
            left_integer, left = visit(node.left)
            right_integer, right = visit(node.right)
            if not (left_integer and right_integer) or node.operator not in {"+", "-", "*", "/"}:
                return False, None
            if node.operator == "/" and right == 0:
                raise CompilationError("default INTEGER division by zero", location)
            if left is None or right is None:
                return True, None
            if node.operator == "+":
                value = left + right
            elif node.operator == "-":
                value = left - right
            elif node.operator == "*":
                value = left * right
            else:
                # Python // rounds down; Fortran INTEGER division truncates to zero.
                value = abs(left) // abs(right)
                if (left < 0) != (right < 0):
                    value = -value
            return True, _checked(value, location)
        if isinstance(node, IntrinsicCall):
            arguments = [visit(argument)[1] for argument in node.arguments]
            integer = node.dtype is ScalarType.INTEGER
            name = node.name.lower()
            if integer and name in {"mod", "modulo"} and len(arguments) == 2 and arguments[1] == 0:
                raise CompilationError("default INTEGER division by zero", location)
            if not integer or any(value is None for value in arguments):
                return integer, None
            if name == "abs" and len(arguments) == 1:
                value = abs(arguments[0])
            elif name in {"min", "max"} and len(arguments) >= 2:
                value = (min if name == "min" else max)(arguments)
            elif name in {"mod", "modulo"} and len(arguments) == 2:
                a, p = arguments
                value = a % p if name == "modulo" else (abs(a) % abs(p)) * (-1 if a < 0 else 1)
            elif name == "sign" and len(arguments) == 2:
                value = abs(arguments[0]) * (-1 if arguments[1] < 0 else 1)
            elif name == "dim" and len(arguments) == 2:
                value = max(arguments[0] - arguments[1], 0)
            elif name == "int" and len(arguments) in {1, 2}:
                value = arguments[0]
            else:
                return integer, None
            return True, _checked(value, location)
        return False, None

    return visit(expression)[1]
