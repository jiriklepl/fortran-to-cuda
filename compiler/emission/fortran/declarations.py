"""Render Fortran public declarations and their interoperable ABI forms."""

from __future__ import annotations

from dataclasses import dataclass

from compiler.emission.common.abi import AbiArgument
from compiler.ir import ScalarType, Symbol


@dataclass(frozen=True)
class FortranKinds:
    integer: str
    real: str
    real32: str
    extent: str
    logical: str


def public_declaration(symbol: Symbol, name: str) -> str:
    kind = {
        ScalarType.INTEGER: "integer",
        ScalarType.REAL: "real(knd)",
        ScalarType.REAL32: "real",
        ScalarType.LOGICAL: "logical",
    }[symbol.dtype]
    intent = symbol.intent or "inout"
    shape = f"({', '.join(':' for _ in range(symbol.rank))})" if symbol.rank else ""
    attributes = f", contiguous, intent({intent})" if symbol.rank else f", intent({intent})"
    return f"{kind}{attributes} :: {name}{shape}"


def abi_declaration(argument: AbiArgument, kinds: FortranKinds) -> str:
    if argument.dimension is not None:
        return f"integer({kinds.extent}), value, intent(in) :: {argument.name}"
    if argument.symbol.dtype == ScalarType.INTEGER:
        kind = f"integer({kinds.integer})"
    elif argument.symbol.dtype == ScalarType.LOGICAL:
        kind = f"logical({kinds.logical})"
    else:
        real_kind = kinds.real32 if argument.symbol.dtype == ScalarType.REAL32 else kinds.real
        kind = f"real({real_kind})"
    if argument.symbol.rank:
        intent = argument.symbol.intent or "inout"
        return f"{kind}, intent({intent}) :: {argument.name}(*)"
    return f"{kind}, value, intent(in) :: {argument.name}"


def abi_call(argument: AbiArgument, kinds: FortranKinds) -> str:
    name = argument.symbol.cpp_name
    if argument.dimension is not None:
        return f"size({name}, {argument.dimension}, kind={kinds.extent})"
    if argument.symbol.rank:
        return name
    cast = {ScalarType.INTEGER: "int", ScalarType.LOGICAL: "logical"}.get(argument.symbol.dtype, "real")
    kind = {
        ScalarType.INTEGER: kinds.integer,
        ScalarType.REAL: kinds.real,
        ScalarType.REAL32: kinds.real32,
        ScalarType.LOGICAL: kinds.logical,
    }[argument.symbol.dtype]
    return f"{cast}({name}, kind={kind})"
