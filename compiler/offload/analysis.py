"""Physical footprints of the regions of an already-proved plan.

This module never uses dependence-report coordinates as memory addresses: the
dependence proof may replace nonlinear subscripts with injective surrogates.
Unknown physical accesses select whole-array transfers, not a legality failure.

Boxes have inclusive *logical* indices. Their ``active_units`` identify the
nonempty loop domains which enable a transfer. Emitters must check those domains
before evaluating a box, preserve elements in every downloaded write box, and
check arithmetic and actual array bounds before using a section. Conditional
stores give conservative boxes: uploading the write boxes preserves holes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from compiler.analysis.dependence import Affine, affine_expression
from compiler.ir import (
    ArrayAccess,
    Assignment,
    Binary,
    Block,
    CompilationError,
    Expr,
    If,
    IntrinsicCall,
    Literal,
    Loop,
    ParallelRegion,
    Reference,
    ScalarType,
    Size,
    SourceLocation,
    Symbol,
    Unary,
    block_writes,
    referenced_symbols,
    walk_expr,
)
from compiler.ir.integers import constant_integer


@dataclass(frozen=True)
class AxisMapping:
    """One physical subscript: coefficient * source_iterator[axis] + offset.

    ``axis=None`` means an invariant coordinate (coefficient is then zero).
    Axes and array dimensions are zero based, unlike Fortran SIZE dimensions.
    """

    axis: int | None
    coefficient: int
    offset: Expr


@dataclass(frozen=True)
class Box:
    lower: tuple[Expr, ...]
    upper: tuple[Expr, ...]
    axes: tuple[AxisMapping, ...]
    active_units: tuple[int, ...] = ()


@dataclass(frozen=True)
class Footprint:
    symbol: Symbol
    reads: tuple[Box, ...] = ()
    writes: tuple[Box, ...] = ()
    full_read: bool = False
    full_write: bool = False
    exact: bool = True
    reason: str | None = None
    read_units: tuple[int, ...] = ()
    write_units: tuple[int, ...] = ()


@dataclass(frozen=True)
class Unit:
    index: int
    region: ParallelRegion
    footprints: tuple[Footprint, ...]


def _integer(value: int) -> Literal:
    return Literal(str(value), ScalarType.INTEGER)


def _constant(expression: Expr) -> int | None:
    try:
        return constant_integer(expression)
    except CompilationError:
        return None


def _add(left: Expr, right: Expr) -> Expr:
    a, b = _constant(left), _constant(right)
    if a is not None and b is not None:
        return _integer(a + b)
    if a == 0:
        return right
    if b == 0:
        return left
    return Binary("+", left, right)


def _scale(expression: Expr, value: int) -> Expr:
    constant = _constant(expression)
    if constant is not None:
        return _integer(constant * value)
    if value == 0:
        return _integer(0)
    if value == 1:
        return expression
    if value == -1:
        return Unary("-", expression)
    return Binary("*", _integer(value), expression)


def _safe_query(expression: Expr, parameters: frozenset[Symbol]) -> bool:
    """Total arithmetic after descriptor guards; emitter checks numeric range.

    Division requires a positive INTEGER constant denominator. Array accesses, local scalar
    state, and model inquiries with unevaluated operands are not hoisted.
    """
    if isinstance(expression, Literal):
        return expression.dtype is ScalarType.INTEGER and _constant(expression) is not None
    if isinstance(expression, Reference):
        symbol = expression.symbol
        return symbol in parameters and not symbol.rank and symbol.dtype is ScalarType.INTEGER
    if isinstance(expression, Size):
        return expression.symbol in parameters and 1 <= expression.dimension <= expression.symbol.rank
    if isinstance(expression, Unary):
        return expression.operator in {"+", "-"} and _safe_query(expression.operand, parameters)
    if isinstance(expression, Binary):
        if expression.operator == "/":
            denominator = _constant(expression.right)
            return (denominator is not None and denominator > 0
                    and _safe_query(expression.left, parameters))
        return expression.operator in {"+", "-", "*"} and all(
            _safe_query(value, parameters) for value in (expression.left, expression.right)
        )
    return (
        isinstance(expression, IntrinsicCall)
        and expression.name.lower() in {"min", "max"}
        and expression.dtype is ScalarType.INTEGER
        and all(_safe_query(value, parameters) for value in expression.arguments)
    )


def _canonical(expression: Expr, parameters: frozenset[Symbol]) -> Expr:
    """Reuse compiler affine arithmetic to compare invariant offsets reliably."""
    environment = {
        symbol: Affine(terms=((f"p{symbol.id}", 1),))
        for symbol in parameters
        if not symbol.rank and symbol.dtype is ScalarType.INTEGER
    }
    atoms: dict[str, Expr] = {f"p{symbol.id}": Reference(symbol) for symbol in environment}
    for symbol in parameters:
        for dimension in range(1, symbol.rank + 1):
            atoms[f"d{symbol.id}_{dimension}"] = Size(symbol, dimension)
    try:
        value = affine_expression(expression, environment, {}, SourceLocation("<offload>"))
    except CompilationError:
        return expression
    result: Expr = _integer(value.constant)
    for name, coefficient in value.terms:
        result = _add(result, _scale(atoms[name], coefficient))
    return result


def _resolve(expression: Expr, definitions: dict[Symbol, Expr]) -> Expr:
    if isinstance(expression, Reference):
        return definitions.get(expression.symbol, expression)
    if isinstance(expression, Unary):
        return replace(expression, operand=_resolve(expression.operand, definitions))
    if isinstance(expression, Binary):
        return replace(
            expression, left=_resolve(expression.left, definitions), right=_resolve(expression.right, definitions)
        )
    if isinstance(expression, (IntrinsicCall, ArrayAccess)):
        name = "arguments" if isinstance(expression, IntrinsicCall) else "indices"
        return replace(expression, **{name: tuple(_resolve(value, definitions) for value in getattr(expression, name))})
    return expression


def _index_terms(expression, iterators, parameters):
    """Extract physical iterator coefficients, retaining invariant offsets."""
    if _safe_query(expression, parameters):
        return {}, _canonical(expression, parameters)
    if isinstance(expression, Reference) and expression.symbol in iterators:
        return {iterators[expression.symbol]: 1}, _integer(0)
    if isinstance(expression, Unary) and expression.operator in {"+", "-"}:
        value = _index_terms(expression.operand, iterators, parameters)
        if value is not None:
            coefficients, offset = value
            sign = -1 if expression.operator == "-" else 1
            return {axis: sign * coefficient for axis, coefficient in coefficients.items()}, _scale(offset, sign)
    if isinstance(expression, Binary) and expression.operator in {"+", "-", "*"}:
        left = _index_terms(expression.left, iterators, parameters)
        right = _index_terms(expression.right, iterators, parameters)
        if left is None or right is None:
            return None
        if expression.operator == "*":
            for scalar, value in ((expression.left, right), (expression.right, left)):
                factor = _constant(scalar)
                if factor is not None:
                    coefficients, offset = value
                    return {axis: factor * value for axis, value in coefficients.items() if factor * value}, _scale(
                        offset, factor
                    )
            return None
        sign = -1 if expression.operator == "-" else 1
        coefficients = dict(left[0])
        for axis, value in right[0].items():
            coefficients[axis] = coefficients.get(axis, 0) + sign * value
        return {axis: value for axis, value in coefficients.items() if value}, _canonical(
            _add(left[1], _scale(right[1], sign)), parameters
        )
    return None


def _access_box(access, unit, parameters):
    loops = unit.region.loops
    iterators = {loop.iterator: axis for axis, loop in enumerate(loops)}
    mappings, lower, upper, used_axes = [], [], [], set()
    for expression in access.indices:
        value = _index_terms(expression, iterators, parameters)
        if value is None or len(value[0]) > 1:
            return None
        coefficients, offset = value
        if not coefficients:
            mapping = AxisMapping(None, 0, offset)
            lo = hi = offset
        else:
            ((axis, coefficient),) = coefficients.items()
            if abs(coefficient) != 1 or axis in used_axes:
                return None
            used_axes.add(axis)
            loop = loops[axis]
            stride = loop.step if isinstance(loop.step, int) else _constant(loop.step)
            if stride not in {-1, 1}:
                return None
            first, last = (loop.lower, loop.upper) if stride == 1 else (loop.upper, loop.lower)
            lo, hi = (first, last) if coefficient == 1 else (last, first)
            lo = _canonical(_add(_scale(lo, coefficient), offset), parameters)
            hi = _canonical(_add(_scale(hi, coefficient), offset), parameters)
            mapping = AxisMapping(axis, coefficient, offset)
        mappings.append(mapping)
        lower.append(lo)
        upper.append(hi)
    return Box(tuple(lower), tuple(upper), tuple(mappings), (unit.index,))


def _merge_boxes(boxes):
    result: dict[tuple, Box] = {}
    for box in boxes:
        key = (box.lower, box.upper, box.axes)
        previous = result.get(key)
        active = set(box.active_units) | (set(previous.active_units) if previous is not None else set())
        result[key] = replace(box, active_units=tuple(sorted(active)))
    return tuple(result.values())


def _merge_footprints(footprints):
    by_symbol: dict[Symbol, list[Footprint]] = {}
    for footprint in footprints:
        by_symbol.setdefault(footprint.symbol, []).append(footprint)
    result = []
    for symbol, values in sorted(by_symbol.items(), key=lambda value: value[0].id):
        reasons = dict.fromkeys(value.reason for value in values if value.reason)
        result.append(
            Footprint(
                symbol,
                _merge_boxes(box for value in values for box in value.reads),
                _merge_boxes(box for value in values for box in value.writes),
                any(value.full_read for value in values),
                any(value.full_write for value in values),
                all(value.exact for value in values),
                "; ".join(reasons) or None,
                tuple(sorted({index for value in values for index in value.read_units})),
                tuple(sorted({index for value in values for index in value.write_units})),
            )
        )
    return tuple(result)


def _unit_footprints(unit, parameters):
    accesses: list[Footprint] = []
    # A same-point read after an unconditional assignment in this iteration
    # consumes the value just produced by the kernel, not an incoming replica.
    # Keep exact expressions, not bounding boxes: reversed/shifted accesses can
    # cover the same rectangle while reading different iteration values.
    defined_points = []

    def record(access, kind, definitions, conditional, unknown):
        access = _resolve(access, definitions)
        box = None if unknown else _access_box(access, unit, parameters)
        reason = "retained-loop footprint" if unknown else "nonrectangular or non-affine physical subscript"
        is_read = kind == "read"
        if is_read and not conditional and not unknown and box and access in defined_points:
            return
        accesses.append(
            Footprint(
                symbol=access.symbol,
                reads=(box,) if is_read and box else (),
                writes=(box,) if not is_read and box else (),
                full_read=is_read and box is None,
                full_write=not is_read and box is None,
                exact=box is not None and not conditional,
                reason=reason if box is None else "conditional access upper bound" if conditional else None,
                read_units=(unit.index,) if is_read else (),
                write_units=(unit.index,) if not is_read else (),
            )
        )
        if not is_read and not conditional and not unknown and box and access not in defined_points:
            defined_points.append(access)

    def reads(expression, definitions, conditional, unknown):
        for node in walk_expr(expression):
            if isinstance(node, ArrayAccess):
                record(node, "read", definitions, conditional, unknown)

    def visit(block, definitions, conditional=False, unknown=False):
        definitions = dict(definitions)
        for statement in block.statements:
            if isinstance(statement, Assignment):
                reads(statement.value, definitions, conditional, unknown)
                if isinstance(statement.target, ArrayAccess):
                    for expression in statement.target.indices:
                        reads(expression, definitions, conditional, unknown)
                    record(statement.target, "write", definitions, conditional, unknown)
                else:
                    symbol = statement.target.symbol
                    # Retire only points whose resolved coordinates depend on
                    # the changed scalar. Unrelated numerical temporaries do
                    # not erase an earlier same-iteration array definition.
                    defined_points[:] = [point for point in defined_points
                                         if symbol not in referenced_symbols(point)]
                    value = _resolve(statement.value, definitions)
                    # Only scalar integer expressions can affect physical maps.
                    if (
                        symbol.dtype is ScalarType.INTEGER
                        and _index_terms(
                            value, {loop.iterator: axis for axis, loop in enumerate(unit.region.loops)}, parameters
                        )
                        is not None
                    ):
                        definitions[symbol] = value
                    else:
                        definitions.pop(symbol, None)
            elif isinstance(statement, If):
                defined_points.clear()
                reads(statement.condition, definitions, conditional, unknown)
                visit(statement.then_body, definitions, True, unknown)
                visit(statement.else_body, definitions, True, unknown)
                for symbol in block_writes(statement.then_body) | block_writes(statement.else_body):
                    definitions.pop(symbol, None)
                defined_points.clear()
            else:
                defined_points.clear()
                reads(statement.lower, definitions, conditional, True)
                reads(statement.upper, definitions, conditional, True)
                if not isinstance(statement.step, int):
                    reads(statement.step, definitions, conditional, True)
                visit(statement.body, definitions, conditional, True)
                for symbol in block_writes(Block((statement,))):
                    definitions.pop(symbol, None)
                defined_points.clear()
        return definitions

    visit(unit.region.body, {})
    return replace(unit, footprints=_merge_footprints(accesses))


def _protected_scalar_inputs(region, parameters):
    """Find inputs the numerical value ABI would read before native control.

    Positive queries imply an active mapped domain, so its headers and straight
    body statements must already require their inputs. IF branches and retained
    loop bodies may never execute; their exclusive inputs must remain behind the
    original native guards. This intentionally avoids reasoning about predicates
    or proving that a retained loop executes at least once.
    """
    unconditional, protected = set(), set()

    def reads(expression, guarded):
        target = protected if guarded else unconditional
        target.update(
            node.symbol for node in walk_expr(expression)
            if isinstance(node, Reference) and node.symbol in parameters and not node.symbol.rank
        )

    def header(loop, guarded):
        reads(loop.lower, guarded)
        reads(loop.upper, guarded)
        if not isinstance(loop.step, int):
            reads(loop.step, guarded)

    def visit(block, guarded=False):
        for statement in block.statements:
            if isinstance(statement, Assignment):
                reads(statement.value, guarded)
                if isinstance(statement.target, ArrayAccess):
                    for index in statement.target.indices:
                        reads(index, guarded)
            elif isinstance(statement, If):
                reads(statement.condition, guarded)
                visit(statement.then_body, True)
                visit(statement.else_body, True)
            elif isinstance(statement, Loop):
                header(statement, guarded)
                visit(statement.body, True)

    for loop in region.loops:
        header(loop, False)
    visit(region.body)
    return {symbol for symbol in protected - unconditional if not symbol.by_value}, protected | unconditional
