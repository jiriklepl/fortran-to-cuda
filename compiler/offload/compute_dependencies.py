"""Bounded, cost-only dependency features for one straight-line work item.

This occurrence graph is not a legality proof, a native instruction schedule,
or evidence of SIMD execution. Constant and invariant flags describe source
dependencies only: they do not assert that a native compiler folds or hoists
an operation. Existing operation counts and placement models remain separate.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

from compiler.ir import (
    ArrayAccess,
    Assignment,
    Binary,
    If,
    IntrinsicCall,
    Literal,
    Loop,
    Reference,
    ScalarType,
    Size,
    Unary,
)
from compiler.ir.intrinsics import REAL_MATH
from compiler.ir.plan import ParallelRegion

SCHEMA_VERSION = 1
MAX_OPERATIONS = 4096
MAX_EDGES = 16384
MAX_VISITS = 32768
MAX_DEFINITIONS = 8192
MAX_SYMBOLS = 8192
MAX_EXPRESSION_DEPTH = 128
_REAL = frozenset({ScalarType.REAL, ScalarType.REAL32})
_ARITHMETIC = {"+": "add", "-": "subtract", "*": "multiply"}
_COMPARISONS = frozenset({"==", "/=", "<", "<=", ">", ">=", ".eq.", ".ne.", ".lt.", ".le.", ".gt.", ".ge."})
_LOGICAL = frozenset({".and.", ".or.", ".eqv.", ".neqv."})


@dataclass(frozen=True)
class ComputeOperation:
    """One source numeric occurrence; IDs are positions in the graph tuple.

    Each operand retains its own ordered dependency group, including empty
    groups for leaves and repeated IDs for repeated operands. Zero-cost casts
    and integer/logical expressions forward their numeric dependencies.
    """

    family: str
    dtype: ScalarType
    operands: tuple[tuple[int, ...], ...]
    constant: bool
    invariant: bool
    # A variadic MIN/MAX is one source occurrence with N-1 pair operations.
    work_units: int = 1
    # Original typed literals survive same-type scalar copies. Constant
    # expressions and numeric conversions do not acquire guessed literals.
    literal_operands: tuple[tuple[str, str] | None, ...] = ()

    @property
    def predecessors(self) -> tuple[int, ...]:
        return tuple(index for operand in self.operands for index in operand)


@dataclass(frozen=True)
class ComputeDependencies:
    available: bool
    reason: str | None = None
    operations: tuple[ComputeOperation, ...] = ()
    outputs: tuple[tuple[int, ...], ...] = ()
    identity: str | None = None
    statement_count: int = 0
    edge_count: int = 0
    dependency_visit_count: int = 0
    visit_count: int = 0
    definition_count: int = 0
    schema_version: int = SCHEMA_VERSION
    # Includes zero-cost casts, leaves and assignment conversions, so an
    # operation-only precision check cannot overlook a narrower intermediate.
    # Address expressions are excluded from this numerical applicability fact.
    floating_dtypes: tuple[ScalarType, ...] = ()
    # Work outside the calibrated real-operation families, after excluding
    # integer definitions used exclusively to form array addresses.
    unpriced_numerical_operations: tuple[tuple[str, int], ...] = ()

    def weighted_span(self, weights: Mapping[str, float]) -> float:
        """Return the longest weighted source path, not a runtime estimate.

        All occurring families require a finite, nonnegative full-operation
        weight. In particular a division is not also charged as ordinary
        arithmetic. Variadic MIN/MAX receives its pair count as a conservative
        serial path; this makes no claim about native intrinsic lowering.
        """
        if not self.available:
            raise ValueError("dependency graph is unavailable")
        checked = {}
        for family in {operation.family for operation in self.operations}:
            value = weights.get(family)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("missing or invalid dependency weight for " + family)
            try:
                value = float(value)
            except OverflowError as error:
                raise ValueError("missing or invalid dependency weight for " + family) from error
            if not math.isfinite(value) or value < 0:
                raise ValueError("missing or invalid dependency weight for " + family)
            checked[family] = value
        spans = []
        for index, operation in enumerate(self.operations):
            if any(not 0 <= parent < index for parent in operation.predecessors):
                raise ValueError("dependency graph is not in topological order")
            span = checked[operation.family] * operation.work_units + max(
                (spans[parent] for parent in operation.predecessors), default=0.0
            )
            if not math.isfinite(span):
                raise ValueError("weighted dependency span overflows")
            spans.append(span)
        return max(spans, default=0.0)

    def to_dict(self, *, include_graph: bool = False) -> dict:
        """Return detached public diagnostics; unavailable work is not zero."""
        result = {
            "schema_version": self.schema_version,
            "available": self.available,
            "reason": self.reason,
            "identity": self.identity,
            "statement_count": self.statement_count,
            "visit_count": self.visit_count,
            "definition_count": self.definition_count,
            "dependency_visit_count": self.dependency_visit_count,
            "operation_count": None,
            "edge_count": None,
            "family_counts": None,
            "dtype_counts": None,
            "floating_dtypes": None,
            "unpriced_numerical_operations": None,
            "constant_operations": None,
            "invariant_operations": None,
            "unweighted_span": None,
            "maximum_same_depth_operations": None,
            "native_folding_proven": False,
            "native_simd_proven": False,
        }
        if self.available:
            depths = []
            for operation in self.operations:
                depths.append(operation.work_units + max((depths[p] for p in operation.predecessors), default=0))
            result.update(
                operation_count=len(self.operations),
                edge_count=self.edge_count,
                family_counts=dict(sorted(Counter(op.family for op in self.operations).items())),
                dtype_counts=dict(sorted(Counter(op.dtype.value for op in self.operations).items())),
                floating_dtypes=[dtype.value for dtype in self.floating_dtypes],
                unpriced_numerical_operations=dict(self.unpriced_numerical_operations),
                constant_operations=sum(op.constant for op in self.operations),
                invariant_operations=sum(op.invariant for op in self.operations),
                unweighted_span=max(depths, default=0),
                maximum_same_depth_operations=max(Counter(depths).values(), default=0),
            )
        if include_graph:
            result["operations"] = [
                {
                    "id": index,
                    "family": op.family,
                    "dtype": op.dtype.value,
                    "operands": [list(operand) for operand in op.operands],
                    "constant": op.constant,
                    "invariant": op.invariant,
                    "work_units": op.work_units,
                    "literal_operands": [list(value) if value is not None else None for value in op.literal_operands],
                }
                for index, op in enumerate(self.operations)
            ]
            result["outputs"] = [list(output) for output in self.outputs]
        return result


@dataclass(frozen=True)
class _Value:
    dtype: ScalarType
    dependencies: tuple[int, ...]
    constant: bool
    invariant: bool
    key: str
    address_safe: bool = False
    literal: tuple[str, str] | None = None
    unpriced: tuple[int, ...] = ()


class _Unavailable(Exception):
    pass


def _fingerprint(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


class _Analyzer:
    def __init__(self, region, max_nodes):
        self.region = region
        self.max_nodes = max_nodes
        self.operations = []
        self.outputs = []
        self.definitions = {}
        self.symbol_ids = {}
        self.iterators = {loop.iterator for loop in region.loops}
        self.private = set(region.private_symbols)
        self.accesses = {}
        self.written = set()
        self.edges = self.visits = self.definition_count = self.statements = 0
        self.trace = []
        self.floating_dtypes = set()
        self.unpriced = []
        self.address_unpriced = set()
        self.numerical_unpriced = set()

    def symbol_key(self, symbol):
        if symbol not in self.symbol_ids:
            if len(self.symbol_ids) >= MAX_SYMBOLS:
                raise _Unavailable("dependency symbol budget exceeded")
            self.symbol_ids[symbol] = len(self.symbol_ids)
        return self.symbol_ids[symbol], symbol.dtype.value, symbol.rank

    def visit(self, depth):
        self.visits += 1
        if self.visits > MAX_VISITS:
            raise _Unavailable("dependency expression visit budget exceeded")
        if depth > MAX_EXPRESSION_DEPTH:
            raise _Unavailable("dependency expression depth budget exceeded")

    def value(self, kind, dtype, values, *, extra=(), family=None, work_units=1, address_safe=False,
              unpriced_kind=None):
        constant = all(value.constant for value in values)
        invariant = all(value.invariant for value in values)
        operands = tuple(value.dependencies for value in values)
        dependencies = tuple(index for operand in operands for index in operand)
        # Bound even forwarded, unpriced dependency edges; otherwise a sequence
        # of logical/cast expressions could expand work without operation nodes.
        self.edges += len(dependencies) + sum(len(value.unpriced) for value in values)
        if self.edges > MAX_EDGES:
            raise _Unavailable("dependency edge budget exceeded")
        key = _fingerprint(kind, dtype.value, extra, tuple(value.key for value in values))
        if family:
            if len(self.operations) >= self.max_nodes:
                raise _Unavailable("dependency operation budget exceeded")
            self.operations.append(
                ComputeOperation(
                    family, dtype, operands, constant, invariant, work_units, tuple(value.literal for value in values)
                )
            )
            dependencies = (len(self.operations) - 1,)
        literal = values[0].literal if kind == "assignment" and values[0].dtype is dtype else None
        unpriced = tuple(dict.fromkeys(index for value in values for index in value.unpriced))
        if unpriced_kind is not None:
            if len(self.unpriced) >= self.max_nodes:
                raise _Unavailable("unpriced numerical operation budget exceeded")
            self.unpriced.append(unpriced_kind)
            unpriced += (len(self.unpriced) - 1,)
        return _Value(dtype, dependencies, constant, invariant, key, address_safe, literal, unpriced)

    def access(self, node, depth, *, write=False):
        if len(node.indices) != node.symbol.rank:
            raise _Unavailable("dependency array rank mismatch")
        indices = tuple(self.expression(index, depth + 1, address=True) for index in node.indices)
        if any(index.dtype is not ScalarType.INTEGER or not index.address_safe for index in indices):
            raise _Unavailable("dependency array coordinates are not exact unchanged integer expressions")
        key = tuple(index.key for index in indices)
        first, mixed = self.accesses.get(node.symbol, (key, False))
        mixed = mixed or first != key
        self.accesses[node.symbol] = first, mixed
        if not write and node.symbol in self.written:
            raise _Unavailable("dependency array read after write requires memory definition analysis")
        if (write or node.symbol in self.written) and mixed:
            raise _Unavailable("dependency mutable array uses different index definitions")
        if write:
            self.written.add(node.symbol)
        return _fingerprint("array", self.symbol_key(node.symbol), key)

    def expression(self, node, depth=0, *, address=False):
        value = self._expression(node, depth, address=address)
        if address:
            self.address_unpriced.update(value.unpriced)
        elif value.dtype in _REAL:
            self.floating_dtypes.add(value.dtype)
            self.numerical_unpriced.update(value.unpriced)
        return value

    def _expression(self, node, depth=0, *, address=False):
        self.visit(depth)
        if isinstance(node, Literal):
            return _Value(
                node.dtype,
                (),
                True,
                True,
                _fingerprint("literal", node.dtype.value, node.value),
                node.dtype is ScalarType.INTEGER,
                (node.dtype.value, node.value),
            )
        if isinstance(node, Reference):
            symbol = node.symbol
            if symbol.rank:
                raise _Unavailable("dependency whole-array reference is unsupported")
            if symbol in self.definitions:
                return self.definitions[symbol]
            if symbol in self.private and symbol not in self.iterators:
                raise _Unavailable("dependency private scalar is read before definition")
            return _Value(
                symbol.dtype,
                (),
                False,
                symbol not in self.iterators,
                _fingerprint("input", self.symbol_key(symbol)),
                symbol.dtype is ScalarType.INTEGER,
            )
        if isinstance(node, Size):
            if not 1 <= node.dimension <= node.symbol.rank:
                raise _Unavailable("dependency SIZE dimension is unsupported")
            return _Value(
                ScalarType.INTEGER,
                (),
                False,
                True,
                _fingerprint("size", self.symbol_key(node.symbol), node.dimension),
                True,
            )
        if isinstance(node, ArrayAccess):
            if address:
                raise _Unavailable("dependency indirect array coordinate is unsupported")
            return _Value(node.symbol.dtype, (), False, False, self.access(node, depth))
        if isinstance(node, Unary):
            operand = self.expression(node.operand, depth + 1, address=address)
            if node.operator not in {"+", "-", ".not."}:
                raise _Unavailable("unsupported dependency unary operation " + node.operator)
            if address and (operand.dtype is not ScalarType.INTEGER or node.operator == ".not."):
                raise _Unavailable("dependency address conversion is unsupported")
            family = "negate" if node.operator == "-" and operand.dtype in _REAL and not address else None
            return self.value(
                node.operator,
                operand.dtype,
                (operand,),
                family=family,
                address_safe=operand.address_safe and node.operator in {"+", "-"},
                unpriced_kind=(("logical:not" if node.operator == ".not." else "integer:negate")
                               if not address and operand.dtype not in _REAL and node.operator != "+" else None),
            )
        if isinstance(node, Binary):
            left = self.expression(node.left, depth + 1, address=address)
            right = self.expression(node.right, depth + 1, address=address)
            operator = node.operator.lower()
            if operator in _COMPARISONS | _LOGICAL:
                dtype, family = ScalarType.LOGICAL, None
            elif operator in {*_ARITHMETIC, "/"}:
                dtype = (
                    ScalarType.REAL
                    if ScalarType.REAL in {left.dtype, right.dtype}
                    else ScalarType.REAL32
                    if ScalarType.REAL32 in {left.dtype, right.dtype}
                    else ScalarType.INTEGER
                )
                family = (
                    (
                        _ARITHMETIC[operator]
                        if operator != "/"
                        else "divide_constant"
                        if right.constant
                        else "divide_dynamic"
                    )
                    if dtype in _REAL
                    else None
                )
            else:
                raise _Unavailable("unsupported dependency binary operation " + operator)
            if address and (dtype is not ScalarType.INTEGER or not left.address_safe or not right.address_safe):
                raise _Unavailable("dependency address conversion is unsupported")
            return self.value(
                operator,
                dtype,
                (left, right),
                family=family if not address else None,
                address_safe=dtype is ScalarType.INTEGER and left.address_safe and right.address_safe,
                unpriced_kind=(
                    "comparison:" + operator if operator in _COMPARISONS else
                    "logical:" + operator if operator in _LOGICAL else
                    "integer:" + operator if dtype is ScalarType.INTEGER else
                    "implicit_real_integer_conversion" if any(value.dtype is ScalarType.INTEGER
                                                               for value in (left, right)) else None
                ) if not address else None,
            )
        if isinstance(node, IntrinsicCall):
            name = node.name.lower()
            if name not in REAL_MATH | {"atan2", "abs", "min", "max", "real", "dble", "int"}:
                raise _Unavailable("unsupported dependency intrinsic " + name)
            arguments = node.arguments[:1] if name in {"real", "dble", "int"} else node.arguments
            values = tuple(self.expression(arg, depth + 1, address=address) for arg in arguments)
            if not values:
                raise _Unavailable("dependency intrinsic has no arguments")
            safe = (
                node.dtype is ScalarType.INTEGER
                and all(value.address_safe for value in values)
                and name in {"int", "abs", "min", "max"}
            )
            if address and not safe:
                raise _Unavailable("dependency address conversion is unsupported")
            family = name if node.dtype in _REAL and name not in {"real", "dble", "int"} and not address else None
            return self.value(
                name,
                node.dtype,
                values,
                family=family,
                work_units=len(values) - 1 if name in {"min", "max"} else 1,
                address_safe=safe,
                unpriced_kind=(
                    "conversion:" + values[0].dtype.value + "->" + node.dtype.value
                    if name in {"real", "dble", "int"} and values[0].dtype is not node.dtype else
                    "integer:" + name if node.dtype is ScalarType.INTEGER and name not in {"int"} else None
                ) if not address else None,
            )
        raise _Unavailable("unsupported dependency expression")

    def run(self):
        for statement in self.region.body.statements:
            self.statements += 1
            if isinstance(statement, If):
                raise _Unavailable("conditional dependency work is unknown")
            if isinstance(statement, Loop):
                raise _Unavailable("retained-loop dependency work is unknown")
            if not isinstance(statement, Assignment):
                raise _Unavailable("unsupported dependency statement")
            self.definition_count += 1
            if self.definition_count > MAX_DEFINITIONS:
                raise _Unavailable("dependency definition budget exceeded")
            value = self.expression(statement.value)
            target = statement.target
            if isinstance(target, (Reference, ArrayAccess)) and target.symbol.dtype in _REAL:
                self.floating_dtypes.add(target.symbol.dtype)
            if isinstance(target, Reference):
                if target.symbol in self.iterators or target.symbol not in self.private:
                    raise _Unavailable("dependency scalar output is not work-item private")
                stored = self.value(
                    "assignment",
                    target.symbol.dtype,
                    (value,),
                    address_safe=value.address_safe and target.symbol.dtype is ScalarType.INTEGER,
                    unpriced_kind=("conversion:" + value.dtype.value + "->" + target.symbol.dtype.value
                                   if value.dtype is not target.symbol.dtype else None),
                )
                stored = _Value(
                    stored.dtype,
                    stored.dependencies,
                    stored.constant,
                    stored.invariant,
                    stored.key,
                    stored.address_safe and stored.dtype is ScalarType.INTEGER,
                    stored.literal,
                    stored.unpriced,
                )
                if stored.dtype in _REAL:
                    self.numerical_unpriced.update(stored.unpriced)
                self.definitions[target.symbol] = stored
                target_key = _fingerprint("private", self.symbol_key(target.symbol))
            elif isinstance(target, ArrayAccess):
                target_key = self.access(target, 0, write=True)
                self.outputs.append(value.dependencies)
                self.numerical_unpriced.update(value.unpriced)
                if value.dtype is not target.symbol.dtype:
                    if len(self.unpriced) >= self.max_nodes:
                        raise _Unavailable("unpriced numerical operation budget exceeded")
                    self.unpriced.append("conversion:" + value.dtype.value + "->" + target.symbol.dtype.value)
            else:
                raise _Unavailable("unsupported dependency assignment target")
            self.trace.append((target_key, value.key, target.symbol.dtype.value))
        floating_dtypes = tuple(sorted(self.floating_dtypes, key=lambda dtype: dtype.value))
        unpriced = tuple(sorted(Counter(kind for index, kind in enumerate(self.unpriced)
            if index not in self.address_unpriced or index in self.numerical_unpriced).items()))
        identity = _fingerprint(SCHEMA_VERSION, self.trace, tuple(dtype.value for dtype in floating_dtypes), unpriced)
        return ComputeDependencies(
            True,
            operations=tuple(self.operations),
            outputs=tuple(self.outputs),
            identity=identity,
            statement_count=self.statements,
            edge_count=sum(len(op.predecessors) for op in self.operations),
            dependency_visit_count=self.edges,
            visit_count=self.visits,
            definition_count=self.definition_count,
            floating_dtypes=floating_dtypes,
            unpriced_numerical_operations=unpriced,
        )


def analyze_compute_dependencies(region: ParallelRegion, *, max_nodes: int = MAX_OPERATIONS) -> ComputeDependencies:
    """Describe exact straight-line numeric dependencies, conservatively.

    This budget is independent of the 256-operation planning limit. The graph
    has no persistent cache and consumes no source authority or runtime state.
    Unknown array RAW, changed subscript definitions, branches and retained
    loops make only these cost features unavailable; numerical legality and
    existing v2 counts/models are untouched.
    """
    if isinstance(max_nodes, bool) or not isinstance(max_nodes, int) or not 1 <= max_nodes <= MAX_OPERATIONS:
        raise ValueError(f"max_nodes must be an integer between 1 and {MAX_OPERATIONS}")
    if not isinstance(region, ParallelRegion):
        return ComputeDependencies(False, "dependency analysis requires a parallel region")
    analyzer = _Analyzer(region, max_nodes)
    try:
        return analyzer.run()
    except _Unavailable as error:
        return ComputeDependencies(
            False,
            str(error),
            statement_count=analyzer.statements,
            dependency_visit_count=analyzer.edges,
            visit_count=analyzer.visits,
            definition_count=analyzer.definition_count,
        )
