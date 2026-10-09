"""Physical transfer footprints and safe partitions of already-proved plans.

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
    ExecutionPlan,
    Expr,
    FunctionIR,
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

    @property
    def uploads(self) -> tuple[Box, ...]:
        """Reads plus values needed to preserve conditional or unknown writes."""
        preserve = self.writes if not self.exact or self.full_write else ()
        return _merge_boxes((*self.reads, *preserve))

    @property
    def downloads(self) -> tuple[Box, ...]:
        return self.writes

    @property
    def full_upload(self) -> bool:
        return self.full_read or self.full_write


@dataclass(frozen=True)
class Unit:
    index: int
    region: ParallelRegion
    footprints: tuple[Footprint, ...]
    work_per_iteration: int | None
    work_is_upper_bound: bool = False


@dataclass(frozen=True)
class Interval:
    """A contiguous [start, stop) group retaining source kernel order."""

    start: int
    stop: int
    footprints: tuple[Footprint, ...]

    @property
    def unit_indices(self) -> tuple[int, ...]:
        return tuple(range(self.start, self.stop))


@dataclass(frozen=True)
class ChunkArray:
    symbol: Symbol
    dimension: int
    coefficient: int
    read_lower_offset: Expr | None
    read_upper_offset: Expr | None
    write_lower_offset: Expr | None
    write_upper_offset: Expr | None


@dataclass(frozen=True)
class ChunkPlan:
    axis: int
    arrays: tuple[ChunkArray, ...]
    domains: tuple[Loop, ...]

    @property
    def array_axes(self) -> tuple[tuple[Symbol, int], ...]:
        return tuple((array.symbol, array.dimension) for array in self.arrays)


@dataclass(frozen=True)
class ScopeSlabArray:
    """Full-layout mapping, or a fixed exact union for immutable input."""

    symbol: Symbol
    dimension: int | None
    coefficient: int = 0
    offset: Expr | None = None

    @property
    def immutable_prefix(self) -> bool:
        return self.dimension is None


@dataclass(frozen=True)
class ScopeSlabPlan:
    axis: int
    domains: tuple[Loop, ...]
    arrays: tuple[ScopeSlabArray, ...]

    def to_dict(self) -> dict:
        return {"axis": self.axis, "partition": "source-loop ordinals",
                "arrays": [{"symbol": array.symbol.name, "dimension": array.dimension,
                            "coefficient": array.coefficient,
                            "offset": _expression_text(array.offset) if array.offset is not None else None,
                            "immutable_prefix": array.immutable_prefix} for array in self.arrays],
                "coordinates": "original logical coordinates and full-array pitches",
                "independence": "complete subchain RAW/WAR/WAW proof"}


@dataclass(frozen=True)
class OffloadAnalysis:
    available: bool
    reason: str | None
    units: tuple[Unit, ...] = ()
    intervals: tuple[Interval, ...] = ()
    chunk: ChunkPlan | None = None
    chunk_reason: str | None = None

    def to_dict(self) -> dict:
        """Small explanatory JSON contract, not serialized compiler IR."""

        def box_record(box):
            return {
                "lower": [_expression_text(value) for value in box.lower],
                "upper": [_expression_text(value) for value in box.upper],
                "active_units": list(box.active_units),
            }

        def footprint_record(footprint):
            element_bytes = {
                ScalarType.REAL: 8, ScalarType.REAL32: 4,
                ScalarType.INTEGER: 4, ScalarType.LOGICAL: 1,
            }[footprint.symbol.dtype]

            def transfer_record(*, upload):
                full = footprint.full_upload if upload else footprint.full_write
                boxes = footprint.uploads if upload else footprint.downloads
                if full:
                    elements = " * ".join(f"size({footprint.symbol.name},{axis})"
                                          for axis in range(1, footprint.symbol.rank + 1))
                else:
                    elements = " + ".join(
                        "(" + " * ".join(
                            f"max(0,({_expression_text(hi)})-({_expression_text(lo)})+1)"
                            for lo, hi in zip(box.lower, box.upper, strict=True)
                        ) + ")" for box in boxes
                    ) or "0"
                return {
                    "available": True,
                    "resolved_at": "runtime_after_domain_and_allocation_guards",
                    "basis": "whole_array" if full else "physical_rectangles",
                    "element_bytes": element_bytes,
                    "byte_count_upper_bound": f"{element_bytes} * ({elements})",
                    "rectangles": [] if full else [box_record(box) for box in boxes],
                    "active_units": sorted(set(footprint.read_units + footprint.write_units if full and upload
                                               else footprint.write_units if full
                                               else (unit for box in boxes for unit in box.active_units))),
                }

            return {
                "symbol": footprint.symbol.name,
                "reads": [box_record(box) for box in footprint.reads],
                "writes": [box_record(box) for box in footprint.writes],
                "full_read": footprint.full_read,
                "full_write": footprint.full_write,
                "exact": footprint.exact,
                "reason": footprint.reason,
                "read_units": list(footprint.read_units),
                "write_units": list(footprint.write_units),
                "transfer_volume": {"upload": transfer_record(upload=True),
                                    "download": transfer_record(upload=False)},
            }

        return {
            "available": self.available,
            "reason": self.reason,
            "units": [
                {
                    "index": unit.index,
                    "region": unit.region.id,
                    "work_per_iteration": unit.work_per_iteration,
                    "work_is_upper_bound": unit.work_is_upper_bound,
                    "footprints": [footprint_record(value) for value in unit.footprints],
                }
                for unit in self.units
            ],
            "intervals": [{"start": value.start, "stop": value.stop,
                           "launches_max": value.stop - value.start,
                           "transfers": [footprint_record(item) for item in value.footprints]}
                          for value in self.intervals],
            "chunk_axis": None if self.chunk is None else self.chunk.axis,
            "chunk_reason": self.chunk_reason,
        }


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

    Nonconstant division is deliberately excluded. Array accesses, local scalar
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
            return _constant(expression) is not None
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
    work = 0
    upper_bound = False
    retained_loop = False

    def record(access, kind, definitions, conditional, unknown):
        access = _resolve(access, definitions)
        box = None if unknown else _access_box(access, unit, parameters)
        reason = "retained-loop footprint" if unknown else "nonrectangular or non-affine physical subscript"
        is_read = kind == "read"
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

    def reads(expression, definitions, conditional, unknown):
        nonlocal work
        for node in walk_expr(expression):
            if isinstance(node, ArrayAccess):
                record(node, "read", definitions, conditional, unknown)
            elif isinstance(node, (Binary, Unary, IntrinsicCall)):
                work += 1

    def visit(block, definitions, conditional=False, unknown=False):
        nonlocal work, upper_bound, retained_loop
        definitions = dict(definitions)
        for statement in block.statements:
            if isinstance(statement, Assignment):
                work += 1
                reads(statement.value, definitions, conditional, unknown)
                if isinstance(statement.target, ArrayAccess):
                    for expression in statement.target.indices:
                        reads(expression, definitions, conditional, unknown)
                    record(statement.target, "write", definitions, conditional, unknown)
                else:
                    symbol = statement.target.symbol
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
                upper_bound = True
                reads(statement.condition, definitions, conditional, unknown)
                visit(statement.then_body, definitions, True, unknown)
                visit(statement.else_body, definitions, True, unknown)
                for symbol in block_writes(statement.then_body) | block_writes(statement.else_body):
                    definitions.pop(symbol, None)
            else:
                retained_loop = True
                reads(statement.lower, definitions, conditional, True)
                reads(statement.upper, definitions, conditional, True)
                if not isinstance(statement.step, int):
                    reads(statement.step, definitions, conditional, True)
                visit(statement.body, definitions, conditional, True)
                for symbol in block_writes(Block((statement,))):
                    definitions.pop(symbol, None)
        return definitions

    visit(unit.region.body, {})
    return replace(
        unit,
        footprints=_merge_footprints(accesses),
        work_per_iteration=None if retained_loop else max(work, 1),
        work_is_upper_bound=upper_bound or retained_loop,
    )


def _offset_extreme(offsets, name):
    values = tuple(dict.fromkeys(offsets))
    if not values:
        return None
    constants = tuple(_constant(value) for value in values)
    if all(value is not None for value in constants):
        return _integer((min if name == "min" else max)(constants))
    return values[0] if len(values) == 1 else IntrinsicCall(name, values, ScalarType.INTEGER)


def _chunk_plan(units, parameters):
    first = units[0].region.loops

    def domain(loops):
        return tuple(
            (
                _canonical(loop.lower, parameters),
                _canonical(loop.upper, parameters),
                loop.step if isinstance(loop.step, int) else _constant(loop.step),
            )
            for loop in loops
        )

    common = domain(first)
    if any(domain(unit.region.loops) != common for unit in units):
        return None, "chunking requires equal mapped domains across all units"
    if any(stride not in {-1, 1} for _, _, stride in common):
        return None, "chunking requires constant signed-unit strides"
    if any(unit.work_per_iteration is None for unit in units):
        return None, "retained loops are outside the bounded chunk prototype"
    footprints = _merge_footprints(value for unit in units for value in unit.footprints)
    if any(value.full_read or value.full_write for value in footprints):
        return None, "chunking requires rectangular physical footprints for every array"
    # Prefer the outer source dimension: slabs then retain inner contiguous work.
    for axis in range(len(first)):
        arrays, valid = [], True
        for footprint in footprints:
            boxes = (*footprint.reads, *footprint.writes)
            mappings = []
            for box in boxes:
                found = [(dimension, value) for dimension, value in enumerate(box.axes) if value.axis == axis]
                if len(found) != 1:
                    valid = False
                    break
                mappings.append(found[0])
            if not valid or not mappings:
                valid = False
                break
            dimension, mapping = mappings[0]
            if any((dim, value.coefficient) != (dimension, mapping.coefficient) for dim, value in mappings):
                valid = False
                break
            # Identical slab coordinates across reads/writes prevent every
            # cross-window RAW, WAR and WAW, even between ordered kernels.
            if footprint.writes and any(value.offset != mapping.offset for _, value in mappings):
                valid = False
                break
            read_offsets = [box.axes[dimension].offset for box in footprint.reads]
            write_offsets = [box.axes[dimension].offset for box in footprint.writes]
            arrays.append(
                ChunkArray(
                    footprint.symbol,
                    dimension,
                    mapping.coefficient,
                    _offset_extreme(read_offsets, "min"),
                    _offset_extreme(read_offsets, "max"),
                    _offset_extreme(write_offsets, "min"),
                    _offset_extreme(write_offsets, "max"),
                )
            )
        if valid and arrays:
            return ChunkPlan(axis, tuple(arrays), first), None
    return None, "no common slab axis without cross-chunk written-array dependence"


def scope_slab_plan(units, parameters):
    """Prove a complete full-layout chain without compact-buffer halo bounds.

    Written resources must have the same slab coordinate for every access.
    Immutable inputs may instead retain their exact full read union; this is
    required for overlapping halos and invariant reads before batches start.
    The caller supplies canonical symbols when composing direct numerical calls.
    """
    if not units:
        return None, "batching requires a nonempty numerical subchain"
    parameters = frozenset(parameters)
    first = units[0].region.loops
    common = tuple((_canonical(loop.lower, parameters), _canonical(loop.upper, parameters),
                    loop.step if isinstance(loop.step, int) else _constant(loop.step)) for loop in first)
    for unit in units:
        domain = tuple((_canonical(loop.lower, parameters), _canonical(loop.upper, parameters),
                        loop.step if isinstance(loop.step, int) else _constant(loop.step)) for loop in unit.region.loops)
        if domain != common:
            return None, "batching requires equal mapped domains across the complete subchain"
        if unit.work_per_iteration is None or unit.work_is_upper_bound:
            return None, "batching requires known unconditional numerical work"
    if any(stride not in {-1, 1} for _, _, stride in common):
        return None, "batching requires constant signed-unit strides"
    footprints = _merge_footprints(fp for unit in units for fp in unit.footprints)
    if any(fp.full_read or fp.full_write for fp in footprints):
        return None, "batching requires exact rectangular physical footprints"
    if any(fp.writes and not fp.exact for fp in footprints):
        return None, "batching requires proved exact output sections"
    for axis in range(len(first)):
        arrays, valid = [], True
        for footprint in footprints:
            mappings = []
            for box in (*footprint.reads, *footprint.writes):
                found = [(dimension, value) for dimension, value in enumerate(box.axes) if value.axis == axis]
                if len(found) != 1:
                    mappings = []
                    break
                mappings.append(found[0])
            if mappings:
                dimension, mapping = mappings[0]
                consistent = all((dim, value.coefficient, value.offset) ==
                                 (dimension, mapping.coefficient, mapping.offset) for dim, value in mappings)
            else:
                consistent = False
            if consistent:
                if footprint.writes and abs(mapping.coefficient) != 1:
                    valid = False
                    break
                arrays.append(ScopeSlabArray(footprint.symbol, dimension, mapping.coefficient, mapping.offset))
            elif footprint.writes:
                valid = False
                break
            else:
                arrays.append(ScopeSlabArray(footprint.symbol, None))
        if valid and arrays:
            return ScopeSlabPlan(axis, first, tuple(arrays)), None
    return None, "no common slab axis without cross-chunk written-array dependence"


def analyze_scope_slabs(function, plan):
    """Retain the ordinary legality checks, with full-layout readonly unions."""
    analysis = analyze_offload(function, plan)
    if not analysis.available:
        return None, analysis.reason, analysis.units
    slab, reason = scope_slab_plan(analysis.units, function.parameters)
    return slab, reason, analysis.units


def scope_slab_candidates(function, plan):
    """The existing bounded GPU intervals, each with its own slab proof."""
    analysis = analyze_offload(function, plan)
    candidates = []
    if analysis.available:
        for interval in analysis.intervals:
            slab, reason = scope_slab_plan(analysis.units[interval.start:interval.stop], function.parameters)
            candidates.append((interval, slab, reason))
    return analysis, tuple(candidates)


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
    return protected - unconditional, protected | unconditional


def analyze_offload(function: FunctionIR, plan: ExecutionPlan, *, max_interval: int = 4) -> OffloadAnalysis:
    """Analyze an already validated plan without changing default compilation.

    Feed an unfused plan to retain source units. A failed optional analysis is
    not a numerical rejection: the caller retains its ordinary/native path.
    """
    if not 1 <= max_interval <= 4:
        raise ValueError("offload intervals must be bounded between one and four units")
    if not plan.steps or any(not isinstance(step, ParallelRegion) for step in plan.steps):
        return OffloadAnalysis(False, "offload partitioning requires a nonempty flat all-device execution plan")
    parameters = frozenset(function.parameters)
    used_scalar_inputs = set()
    for region in plan.steps:
        if any(symbol not in parameters for symbol in (*region.read_symbols, *region.write_symbols)):
            return OffloadAnalysis(False, "array storage must be backed by entry parameters")
        if any(not symbol.rank and symbol not in parameters for symbol in region.captured_symbols):
            return OffloadAnalysis(False, "scalar state crossing a region boundary is unsupported")
        if any(symbol in parameters and not symbol.rank for symbol in block_writes(region.body)):
            return OffloadAnalysis(False, "offload inputs must remain immutable across units")
        protected, used = _protected_scalar_inputs(region, parameters)
        used_scalar_inputs.update(used)
        if protected:
            names = ", ".join(symbol.name for symbol in sorted(protected, key=lambda symbol: symbol.id))
            return OffloadAnalysis(
                False, "scalar inputs may be protected by conditional or empty retained-loop control: " + names
            )
        for loop in region.loops:
            bounds = (loop.lower, loop.upper)
            if not all(_safe_query(expression, parameters) for expression in bounds):
                return OffloadAnalysis(False, "mapped bounds require descriptor sizes or immutable scalar arithmetic")
            if not isinstance(loop.step, int) and not _safe_query(loop.step, parameters):
                return OffloadAnalysis(False, "mapped strides require immutable scalar arithmetic")
    unused = {symbol for symbol in parameters if not symbol.rank} - used_scalar_inputs
    if unused:
        names = ", ".join(symbol.name for symbol in sorted(unused, key=lambda symbol: symbol.id))
        return OffloadAnalysis(False, "unused scalar inputs are not safe through the numerical value ABI: " + names)
    units = tuple(
        _unit_footprints(Unit(index, region, (), None), parameters) for index, region in enumerate(plan.steps)
    )
    spans = {
        (start, stop)
        for start in range(len(units))
        for stop in range(start + 1, min(len(units), start + max_interval) + 1)
    }
    spans.add((0, len(units)))
    intervals = tuple(
        Interval(start, stop, _merge_footprints(value for unit in units[start:stop] for value in unit.footprints))
        for start, stop in sorted(spans)
    )
    chunk, chunk_reason = _chunk_plan(units, parameters)
    return OffloadAnalysis(True, None, units, intervals, chunk, chunk_reason)


def _expression_text(expression):
    if isinstance(expression, Literal):
        return expression.value
    if isinstance(expression, Reference):
        return expression.symbol.name
    if isinstance(expression, ArrayAccess):
        return f"{expression.symbol.name}({','.join(_expression_text(value) for value in expression.indices)})"
    if isinstance(expression, Size):
        return f"size({expression.symbol.name},{expression.dimension})"
    if isinstance(expression, Unary):
        return f"({expression.operator}{_expression_text(expression.operand)})"
    if isinstance(expression, Binary):
        return f"({_expression_text(expression.left)} {expression.operator} {_expression_text(expression.right)})"
    if isinstance(expression, IntrinsicCall):
        return f"{expression.name}({','.join(_expression_text(value) for value in expression.arguments)})"
    raise TypeError(f"unsupported query expression: {expression!r}")
