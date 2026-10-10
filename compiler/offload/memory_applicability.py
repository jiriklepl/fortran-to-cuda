"""Cost-only physical access classes; these never establish GPU legality.

The initial contract matches the numerical three-array memory fixture. It is
deliberately distinct from transfer footprints: a bounded footprint alone says
nothing about repeated loads, read/modify/write or native cache reuse.
"""
from __future__ import annotations

from dataclasses import dataclass

from compiler.analysis.dependence import Affine, affine_expression
from compiler.ir import (
    ArrayAccess,
    Assignment,
    Binary,
    CompilationError,
    IntrinsicCall,
    Literal,
    Reference,
    ScalarType,
    Size,
    Symbol,
    Unary,
)
from compiler.ir.integers import constant_integer
from compiler.ir.plan import ParallelRegion

POINTWISE_THREE_ARRAY = "pointwise_three_array_v1"
MAX_STATEMENTS = 4096
MAX_VISITS = 32768
MAX_DEPTH = 128
MAX_CAPTURES = 8192
MISSING_CONTRACTS = (
    ("zero_fill_v1", "one write resource; no independent cost validation"),
    ("copy_v1", "one read and one write resource; no independent cost validation"),
    ("read_modify_write_v1", "one resource read and written; no independent cost validation"),
    ("accumulate_v1", "read-only input plus read/write output; no independent cost validation"),
    ("stencil_v1", "multiple physical offsets or cross-cell reuse; no independent cost validation"),
)
_REAL = frozenset({ScalarType.REAL, ScalarType.REAL32})


@dataclass(frozen=True)
class TranslatedAccess:
    symbol: Symbol
    role: str
    loop_axes: tuple[int, ...]
    offsets: tuple[Affine, ...]

    def to_dict(self):
        return {"resource": self.symbol.name, "symbol_id": self.symbol.id, "role": self.role,
                "dtype": self.symbol.dtype.value, "rank": self.symbol.rank,
                "loop_axes": list(self.loop_axes),
                "physical_offsets": [{"constant": x.constant, "terms": [list(term) for term in x.terms]}
                                     for x in self.offsets]}


@dataclass(frozen=True)
class MemoryAccessRequirement:
    available: bool
    class_id: str | None = None
    reason: str | None = None
    accesses: tuple[TranslatedAccess, ...] = ()
    runtime_requirements: tuple[str, ...] = ()
    schema_version: int = 1

    def to_dict(self):
        return {"schema_version": self.schema_version, "available": self.available,
                "class_id": self.class_id, "reason": self.reason,
                "physical_accesses": [access.to_dict() for access in self.accesses],
                "runtime_requirements": list(self.runtime_requirements),
                "missing_contracts": dict(MISSING_CONTRACTS),
                "gpu_legality_proven": False, "runtime_alias_proven": False,
                "traffic_equals_unique_working_set_required": self.available}


class _Unavailable(Exception):
    pass


class _Classifier:
    def __init__(self, region):
        self.region = region
        self.visits = 0
        self.reads, self.writes = [], []
        self.private = set(region.private_symbols)
        self.iterator_names = {loop.iterator: f"i{axis}" for axis, loop in enumerate(region.loops)}
        self.iterators = {symbol: Affine(terms=((name, 1),)) for symbol, name in self.iterator_names.items()}
        self.environment = {symbol: Affine(terms=((f"p{symbol.id}", 1),))
                            for symbol in region.captured_symbols
                            if not symbol.rank and symbol.dtype is ScalarType.INTEGER}

    def count(self, expression):
        stack = [(expression, 0)]
        while stack:
            node, depth = stack.pop()
            self.visits += 1
            if self.visits > MAX_VISITS or depth > MAX_DEPTH:
                raise _Unavailable("physical access expression budget exceeded")
            yield node
            if isinstance(node, Binary):
                stack.extend(((node.right, depth + 1), (node.left, depth + 1)))
            elif isinstance(node, Unary):
                stack.append((node.operand, depth + 1))
            elif isinstance(node, IntrinsicCall):
                if node.name.lower() == "merge":
                    raise _Unavailable("conditional expression access has no memory cost contract")
                stack.extend((value, depth + 1) for value in reversed(node.arguments))
            elif isinstance(node, ArrayAccess):
                stack.extend((value, depth + 1) for value in reversed(node.indices))
            elif not isinstance(node, (Literal, Reference, Size)):
                raise _Unavailable("unsupported physical access expression")

    def affine(self, expression, location):
        try:
            return affine_expression(expression, self.environment, self.iterators, location)
        except (CompilationError, ValueError, OverflowError) as error:
            raise _Unavailable("physical subscript or bound is not proved affine") from error

    def access(self, expression, role, location):
        symbol = expression.symbol
        rank = len(self.region.loops)
        if symbol.dtype not in _REAL or symbol.rank != rank or len(expression.indices) != rank:
            raise _Unavailable("memory class requires equal-rank real point resources")
        axes, offsets = [], []
        names = {name: axis for axis, name in enumerate(self.iterator_names.values())}
        for subscript in expression.indices:
            affine = self.affine(subscript, location)
            varying = [(names[name], coefficient) for name, coefficient in affine.terms if name in names]
            if len(varying) != 1 or varying[0][1] != 1:
                raise _Unavailable("memory class requires one translated physical point per axis")
            axis = varying[0][0]
            axes.append(axis)
            offsets.append(Affine(affine.constant, tuple((name, value) for name, value in affine.terms if name not in names)))
        if len(set(axes)) != rank or axes[0] != rank - 1:
            raise _Unavailable("first Fortran array axis must follow the innermost source loop")
        return TranslatedAccess(symbol, role, tuple(axes), tuple(offsets))

    def visit_reads(self, expression, location):
        for node in self.count(expression):
            if isinstance(node, ArrayAccess):
                self.reads.append(self.access(node, "read", location))
            elif isinstance(node, Reference) and node.symbol.rank:
                raise _Unavailable("whole-array expressions require a separate memory contract")

    def run(self):
        for loop in self.region.loops:
            if type(loop.step) is not int:
                tuple(self.count(loop.step))
            step = loop.step if type(loop.step) is int else constant_integer(loop.step)
            if step != 1:
                raise _Unavailable("memory class requires positive unit source strides")
            for bound in (loop.lower, loop.upper):
                for node in self.count(bound):
                    if isinstance(node, ArrayAccess):
                        raise _Unavailable("array-valued loop bounds require a separate memory contract")
                if any(name in self.iterator_names.values() for name, _ in self.affine(bound, loop.location).terms):
                    raise _Unavailable("memory class requires a rectangular source domain")
        for statement in self.region.body.statements:
            if not isinstance(statement, Assignment):
                raise _Unavailable("conditional or retained-loop accesses have no memory cost contract")
            self.visit_reads(statement.value, statement.location)
            target = statement.target
            if isinstance(target, ArrayAccess):
                for subscript in target.indices:
                    self.visit_reads(subscript, statement.location)
                self.writes.append(self.access(target, "write", statement.location))
            elif isinstance(target, Reference) and target.symbol in self.private:
                if target.symbol.dtype is ScalarType.INTEGER:
                    try:
                        self.environment[target.symbol] = self.affine(statement.value, statement.location)
                    except _Unavailable:
                        self.environment.pop(target.symbol, None)
            else:
                raise _Unavailable("nonprivate scalar effects have no memory cost contract")
        reads = {access.symbol for access in self.reads}
        writes = {access.symbol for access in self.writes}
        if reads & writes:
            raise _Unavailable("read/modify/write or accumulation memory contract is missing")
        if len(writes) != 1 or len(self.writes) != 1:
            raise _Unavailable("memory class requires one write-only output point")
        if len(reads) != 2:
            raise _Unavailable("zero-fill, copy or other resource-count memory contract is missing")
        if len(self.reads) != 2:
            raise _Unavailable("repeated or stencil input accesses have no memory cost contract")
        accesses = (*self.reads, *self.writes)
        if len({access.symbol.dtype for access in accesses}) != 1:
            raise _Unavailable("mixed memory precision has no validated cost contract")
        if len({access.loop_axes for access in accesses}) != 1:
            raise _Unavailable("memory resources do not share one physical traversal")
        return MemoryAccessRequirement(True, POINTWISE_THREE_ARRAY, accesses=accesses, runtime_requirements=(
            "three distinct canonical allocations with proved nonoverlapping views",
            "original allocation bounds and defined read sections remain valid",
            "dense Fortran layout with contiguous traversal across active dimensions",
            "where a higher dimension varies, all preceding physical dimensions cover their full descriptor extent",
        ))


def memory_access_requirement(region):
    """Inspect original ordered ArrayAccess expressions, never ISL coordinates.

    This only supplies a cost applicability requirement. Native/GPU legality,
    guarded descriptor evaluation and runtime aliases remain separate proofs.
    """
    if (not isinstance(region, ParallelRegion) or not region.loops or
            len(region.loops) > 7 or len(region.body.statements) > MAX_STATEMENTS or
            len(region.captured_symbols) > MAX_CAPTURES):
        return MemoryAccessRequirement(False, reason="bounded numerical access region unavailable")
    if len({loop.iterator for loop in region.loops}) != len(region.loops):
        return MemoryAccessRequirement(False, reason="source loop iterators are not distinct")
    try:
        return _Classifier(region).run()
    except _Unavailable as error:
        return MemoryAccessRequirement(False, reason=str(error))
    except (CompilationError, OverflowError, ValueError) as error:
        return MemoryAccessRequirement(False, reason="invalid physical access arithmetic: " + str(error))
