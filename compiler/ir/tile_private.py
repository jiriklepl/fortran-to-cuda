"""Indexed, tile-local storage and bounded host-side scratch layout.

This representation does not scalarize arrays or authorize placement. Original
source and dependence authority is supplied separately by the frontend issuer.
Scratch descriptors are snapshots of original allocated storage, not re-evaluated
Fortran specification expressions.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

from compiler.ir.nodes import Block, CompilationError, Loop, ScalarType, Symbol

SIZE_MAX = (1 << 64) - 1
INTEGER_MIN, INTEGER_MAX = -(1 << 31), (1 << 31) - 1


class ScratchResourceError(ValueError):
    """A prework resource failure; a caller may retain the original source."""


def _size(value: int) -> int:
    if type(value) is not int or not 0 <= value <= SIZE_MAX:
        raise ScratchResourceError("tile scratch size exceeds size_t")
    return value


def _multiply(left: int, right: int) -> int:
    left, right = _size(left), _size(right)
    if right and left > SIZE_MAX // right:
        raise ScratchResourceError("tile scratch product overflows size_t")
    return left * right


def _align(value: int) -> int:
    if value > SIZE_MAX - 7:
        raise ScratchResourceError("tile scratch alignment overflows size_t")
    return (value + 7) & ~7


@dataclass(frozen=True)
class ScratchDescriptor:
    symbol: Symbol
    lower: tuple[int, ...]
    upper: tuple[int, ...]

    @property
    def extents(self) -> tuple[int, ...]:
        if (
            self.symbol.dtype not in (ScalarType.REAL32, ScalarType.REAL)
            or not 1 <= self.symbol.rank <= 3
            or len(self.lower) != self.symbol.rank
            or len(self.upper) != self.symbol.rank
            or any(type(x) is not int or not INTEGER_MIN <= x <= INTEGER_MAX for x in (*self.lower, *self.upper))
        ):
            raise ScratchResourceError("tile scratch requires original numeric descriptor bounds")
        result = tuple(max(0, hi - lo + 1) for lo, hi in zip(self.lower, self.upper, strict=True))
        if not all(result):
            raise ScratchResourceError("active tile scratch has empty original storage")
        return result

    @property
    def element_bytes(self) -> int:
        return 4 if self.symbol.dtype is ScalarType.REAL32 else 8


@dataclass(frozen=True)
class ScratchSlice:
    descriptor: ScratchDescriptor
    offset_bytes: int
    payload_bytes: int

    def element_offset(self, coordinates: tuple[int, ...]) -> int:
        """Fortran column-major indexing, including negative declared bounds."""
        if len(coordinates) != self.descriptor.symbol.rank:
            raise IndexError("tile scratch index rank differs")
        result, pitch = 0, 1
        for x, lo, extent in zip(coordinates, self.descriptor.lower, self.descriptor.extents, strict=True):
            if type(x) is not int or not lo <= x < lo + extent:
                raise IndexError("tile scratch index exceeds original bounds")
            result += (x - lo) * pitch
            pitch *= extent
        return result


@dataclass(frozen=True)
class ScratchBatchPlan:
    total_tiles: int
    tiles_per_batch: int
    bytes_per_tile: int
    lease_bytes: int
    arrays: tuple[ScratchSlice, ...]

    def batches(self) -> Iterator[tuple[int, int]]:
        first = 0
        while first < self.total_tiles:
            count = min(self.tiles_per_batch, self.total_tiles - first)
            yield first, count
            first += count

    def byte_offset(self, slot: int, array: int, coordinates: tuple[int, ...]) -> int:
        if type(slot) is not int or not 0 <= slot < self.tiles_per_batch:
            raise IndexError("tile scratch slot exceeds current lease")
        if type(array) is not int or not 0 <= array < len(self.arrays):
            raise IndexError("tile scratch array does not belong to current lease")
        item = self.arrays[array]
        return (
            slot * self.bytes_per_tile
            + item.offset_bytes
            + item.element_offset(coordinates) * item.descriptor.element_bytes
        )


def plan_scratch(
    total_tiles: int, descriptors: Callable[[], tuple[ScratchDescriptor, ...]], byte_budget: int
) -> ScratchBatchPlan:
    """Calculate a bounded lease before accepted work, without host copies.

    A zero-trip domain returns before invoking the descriptor supplier. Positive
    domains use snapshots supplied at their original reached position. The
    runtime scratch-acquire call must still enforce its combined field/scratch
    budget; this arithmetic never promises that a lease will succeed.
    """
    total_tiles = _size(total_tiles)
    if not total_tiles:
        return ScratchBatchPlan(0, 0, 0, 0, ())
    byte_budget = _size(byte_budget)
    original = tuple(descriptors())
    if not original or len({x.symbol for x in original}) != len(original):
        raise ScratchResourceError("tile scratch requires distinct private arrays")
    arrays, stride = [], 0
    for descriptor in original:
        elements = 1
        for extent in descriptor.extents:
            elements = _multiply(elements, extent)
        payload = _multiply(elements, descriptor.element_bytes)
        stride = _align(stride)
        arrays.append(ScratchSlice(descriptor, stride, payload))
        stride = _size(stride + payload)
    stride = _align(stride)
    count = min(total_tiles, byte_budget // stride)
    if not count:
        raise ScratchResourceError("tile scratch cannot fit one tile before work")
    return ScratchBatchPlan(total_tiles, count, stride, _multiply(count, stride), tuple(arrays))


@dataclass(frozen=True)
class TilePrivateRegion:
    """Original serial phases within one independently scheduled logical tile.

    Producer and consumer Blocks preserve their ordinary ordered IR trees. The
    enclosing original worksharing units remain separate same-stream kernels.
    A single CUDA thread owns a tile in the first mapping, so phase order needs
    no cross-thread synchronization. A cooperative mapping needs a new proof.
    """

    id: int
    parameters: tuple[Symbol, ...]
    private_symbols: tuple[Symbol, ...]
    scratch_symbols: tuple[Symbol, ...]
    outer_loops: tuple[Loop, ...]
    producer: Block
    consumer: Block
    source_proof: object

    def __post_init__(self):
        if (
            not 1 <= len(self.outer_loops) <= 3
            or not self.scratch_symbols
            or len(set(self.scratch_symbols)) != len(self.scratch_symbols)
            or set(self.parameters) & set(self.scratch_symbols)
            or any(
                not 1 <= x.rank <= 3 or x.dtype not in (ScalarType.REAL32, ScalarType.REAL)
                for x in self.scratch_symbols
            )
        ):
            raise CompilationError("tile-private IR requires separate indexed real scratch and original tile axes")
