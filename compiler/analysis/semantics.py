"""Source validity and definite definitions, run before choosing execution policy."""

from compiler.ir import (
    ArrayAccess,
    Assignment,
    Binary,
    Block,
    CompilationError,
    If,
    IntrinsicCall,
    Literal,
    Loop,
    Reference,
    ScalarType,
    Size,
    Unary,
    block_writes,
    referenced_symbols,
    statement_reads,
)
from compiler.ir.integers import constant_integer
from compiler.ir.intrinsics import intrinsic_type

from .effects import header_reads, loop_step


def expression_type(expression, location):
    dtype = _expression_type(expression, location)
    constant_integer(expression, location)
    return dtype


def _expression_type(expression, location):
    if isinstance(expression, Literal):
        return expression.dtype
    if isinstance(expression, (Reference, ArrayAccess)):
        if isinstance(expression, Reference) and expression.symbol.rank:
            raise CompilationError("whole-array expressions are unsupported", location)
        if isinstance(expression, ArrayAccess):
            if len(expression.indices) != expression.symbol.rank:
                raise CompilationError("array access rank mismatch", location)
            if any(_expression_type(index, location) is not ScalarType.INTEGER for index in expression.indices):
                raise CompilationError("array subscripts must be INTEGER expressions", location)
        return expression.symbol.dtype
    if isinstance(expression, Size):
        if not 1 <= expression.dimension <= expression.symbol.rank:
            raise CompilationError("SIZE dimension is outside array rank", location)
        return ScalarType.INTEGER
    if isinstance(expression, IntrinsicCall):
        dtype = intrinsic_type(
            expression.name, tuple(_expression_type(arg, location) for arg in expression.arguments), location
        )
        if expression.dtype is not dtype:
            raise CompilationError("intrinsic result type does not match its signature", location)
        return dtype
    if isinstance(expression, Unary):
        if expression.operator not in {"+", "-", ".not."}:
            raise CompilationError("unsupported unary operator " + expression.operator, location)
        dtype = _expression_type(expression.operand, location)
        if (expression.operator == ".not.") != (dtype is ScalarType.LOGICAL):
            raise CompilationError("invalid operand type for unary operator", location)
        return dtype
    if isinstance(expression, Binary):
        if expression.operator not in {
            "+",
            "-",
            "*",
            "/",
            "==",
            "/=",
            "<",
            "<=",
            ">",
            ">=",
            ".eq.",
            ".ne.",
            ".lt.",
            ".le.",
            ".gt.",
            ".ge.",
            ".and.",
            ".or.",
            ".eqv.",
            ".neqv.",
        }:
            raise CompilationError("unsupported binary operator " + expression.operator, location)
        left, right = _expression_type(expression.left, location), _expression_type(expression.right, location)
        logical = expression.operator in {".and.", ".or.", ".eqv.", ".neqv."}
        if any((dtype is ScalarType.LOGICAL) != logical for dtype in (left, right)):
            raise CompilationError("invalid operand types for operator " + expression.operator, location)
        if expression.operator not in {"+", "-", "*", "/"}:
            return ScalarType.LOGICAL
        return (
            ScalarType.REAL
            if ScalarType.REAL in {left, right}
            else ScalarType.REAL32
            if ScalarType.REAL32 in {left, right}
            else ScalarType.INTEGER
        )
    raise CompilationError("unsupported expression", location)


def validate_block(block: Block, defined: set, active=frozenset()):
    defined = set(defined)
    loop_written = set()

    def require(symbols, location):
        for symbol in sorted(symbols, key=lambda s: s.id):
            if not symbol.rank and symbol not in defined:
                detail = "read before its per-iteration definition" if active else "read before definition"
                if symbol in loop_written:
                    detail += "; loop-written scalar is live after the region but its loop may be empty"
                raise CompilationError(f"Scalar '{symbol.name}' is {detail}", location)

    for statement in block.statements:
        if isinstance(statement, Assignment):
            require(statement_reads(statement), statement.location)
            dtype = expression_type(statement.value, statement.location)
            target_type = expression_type(statement.target, statement.location)
            if (dtype is ScalarType.LOGICAL) != (target_type is ScalarType.LOGICAL):
                raise CompilationError("assignment requires compatible logical or numeric types", statement.location)
            symbol = statement.target.symbol
            if symbol in active:
                raise CompilationError("cannot modify active loop iterator", statement.location)
            if symbol.intent == "in":
                raise CompilationError("cannot write INTENT(IN) variable " + symbol.name, statement.location)
            if isinstance(statement.target, Reference):
                if symbol.parameter:
                    raise CompilationError("Assignments to scalar dummy arguments are unsupported", statement.location)
                defined.add(symbol)
        elif isinstance(statement, If):
            require(referenced_symbols(statement.condition), statement.location)
            if expression_type(statement.condition, statement.location) is not ScalarType.LOGICAL:
                raise CompilationError("IF condition must be LOGICAL", statement.location)
            then_defined = validate_block(statement.then_body, defined, active)
            else_defined = validate_block(statement.else_body, defined, active)
            defined.update(then_defined & else_defined)
        elif isinstance(statement, Loop):
            iterator = statement.iterator
            if iterator.rank or iterator.dtype is not ScalarType.INTEGER:
                raise CompilationError("loop iterators must be INTEGER scalars", statement.location)
            if iterator.parameter or iterator.intent == "in":
                raise CompilationError("loop iterators must be writable local INTEGER scalars", statement.location)
            if iterator in active:
                raise CompilationError("nested loops cannot reuse an active iterator", statement.location)
            require(header_reads(statement), statement.location)
            if any(
                expression_type(expr, statement.location) is not ScalarType.INTEGER
                for expr in (statement.lower, statement.upper, loop_step(statement))
            ):
                raise CompilationError("loop bounds and strides must be INTEGER expressions", statement.location)
            stride = constant_integer(loop_step(statement))
            if stride == 0:
                raise CompilationError("DO stride cannot be zero", statement.location)
            loop_written.update(block_writes(statement.body))
            body_defined = validate_block(statement.body, defined | {statement.iterator}, active | {statement.iterator})
            defined.add(statement.iterator)
            lower, upper = constant_integer(statement.lower), constant_integer(statement.upper)
            if (
                lower is not None
                and upper is not None
                and stride is not None
                and ((stride > 0 and lower <= upper) or (stride < 0 and lower >= upper))
            ):
                defined.update(body_defined)
        else:
            raise CompilationError("unsupported IR statement")
    return defined


def validate_function(function):
    for symbol in function.symbols:
        if symbol.rank and symbol.dtype is ScalarType.LOGICAL:
            raise CompilationError("LOGICAL arrays are unsupported")
    return validate_block(function.body, {symbol for symbol in function.parameters if not symbol.rank})
