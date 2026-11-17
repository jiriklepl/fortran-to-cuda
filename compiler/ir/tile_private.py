"""Indexed, tile-local storage and bounded host-side scratch layout.

This representation does not scalarize arrays or authorize placement. Original
source and dependence authority is supplied separately by the frontend issuer.
Scratch descriptors are snapshots of original allocated storage, not re-evaluated
Fortran specification expressions.
"""

from __future__ import annotations

from dataclasses import dataclass

from compiler.ir.nodes import Block, CompilationError, Loop, ScalarType, Symbol

SIZE_MAX = (1 << 64) - 1
INTEGER_MIN, INTEGER_MAX = -(1 << 31), (1 << 31) - 1


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
