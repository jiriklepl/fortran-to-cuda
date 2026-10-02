"""Describe the ordered parameter list shared by all generated interfaces."""

from __future__ import annotations

from dataclasses import dataclass

from compiler.ir import Symbol


@dataclass(frozen=True)
class AbiArgument:
    """One scalar, array, or extent in the shared parameter list."""

    symbol: Symbol
    dimension: int | None = None

    @property
    def name(self) -> str:
        return self.symbol.cpp_name if self.dimension is None else dimension_name(self.symbol, self.dimension)


def abi_arguments(symbols: tuple[Symbol, ...]) -> tuple[AbiArgument, ...]:
    return tuple(
        argument
        for symbol in symbols
        for argument in (
            AbiArgument(symbol),
            *(AbiArgument(symbol, dimension) for dimension in range(1, symbol.rank + 1)),
        )
    )


def dimension_name(symbol: Symbol, dimension: int) -> str:
    return f"{symbol.cpp_name}_dim{dimension}"
