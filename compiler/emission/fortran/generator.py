"""Emit the Fortran module that bridges public callers to the C ABI."""

from __future__ import annotations

from dataclasses import replace

from compiler.emission.common.abi import AbiArgument
from compiler.emission.common.sessions import session_names
from compiler.emission.fortran.declarations import FortranKinds, abi_call, abi_declaration, public_declaration
from compiler.emission.fortran.formatting import _fortran_line, _fortran_list
from compiler.emission.fortran.sessions import fortran_sessions
from compiler.ir import FunctionIR, ScalarType


def generate_fortran(function: FunctionIR, abi: tuple[AbiArgument, ...], *, offload=None) -> str:
    # Keep the public dummy labels for keyword callers. Conversion intrinsics
    # belong in a separate scope where user names cannot hide SIZE/INT/REAL.
    names = session_names(function)
    occupied = set(names.__dict__.values())
    occupied.update({symbol.name.lower() for symbol in function.parameters})
    occupied.update({function.name.lower(), function.module.lower(), "start_hot", "finish_hot", "knd"})

    def unique_name(base: str) -> str:
        name = base
        suffix = 0
        while name.lower() in occupied:
            suffix += 1
            name = f"{base}_{suffix}"
        occupied.add(name.lower())
        return name

    kinds = FortranKinds(
        integer=unique_name("fort_internal_c_integer"),
        real=unique_name("fort_internal_c_real"),
        real32=unique_name("fort_internal_c_real32"),
        extent=unique_name("fort_internal_c_extent"),
        logical=unique_name("fort_internal_c_logical"),
        token=unique_name("fort_internal_c_token"),
    )
    c_entry = unique_name("fort_internal_c_entry")
    bridge = unique_name("fort_internal_bridge")
    c_start = unique_name("fort_internal_c_start")
    c_finish = unique_name("fort_internal_c_finish")
    c_query = unique_name("fort_internal_c_query")
    query_bridge = unique_name("fort_internal_query_bridge")
    query_result = unique_name("fort_internal_query_result")
    declarations, interfaces, bodies = fortran_sessions(function, kinds, unique_name)
    lines = [
        f"module {function.module}",
        "  use iso_c_binding, only: &",
        f"    {kinds.integer} => c_int, &",
        f"    {kinds.real} => c_double, &",
        f"    {kinds.real32} => c_float, &",
        f"    {kinds.extent} => c_size_t, &",
        f"    {kinds.logical} => c_bool, &",
        f"    {kinds.token} => c_int64_t",
        "  implicit none",
        "  private",
        f"  public :: {function.name}, knd, start_hot, finish_hot",
        f"  integer, parameter :: knd = {kinds.real}",
        "",
        *declarations,
        "  interface",
        *interfaces,
    ]
    if offload is not None:
        lines.extend(_fortran_list(f"function {c_query}(", [a.name for a in abi],
            f") bind(C, name='cpp_{offload.query_name}') result({query_result})", 4))
        lines.extend(f"      import :: {kind}" for kind in
                     (kinds.integer, kinds.real, kinds.real32, kinds.extent, kinds.logical))
        lines.append(f"      integer({kinds.integer}) :: {query_result}")
        for argument in abi:
            readonly = replace(argument, symbol=replace(argument.symbol, intent="in"))
            declaration = abi_declaration(readonly, kinds)
            if argument.dimension is None and not argument.symbol.rank:
                declaration = declaration.replace(", value,", ",")
            lines.extend(_fortran_line(declaration, 6))
        lines.append(f"    end function {c_query}")
        public_at = next(i for i, line in enumerate(lines) if line.startswith("  public ::"))
        lines.insert(public_at, f"  public :: {offload.query_name}")
    lines.extend(
        _fortran_list(
            f"subroutine {c_entry}(",
            [argument.name for argument in abi],
            f") bind(C, name='cpp_{function.name}')",
            4,
        )
    )
    lines.extend(
        f"      import :: {kind}" for kind in (kinds.integer, kinds.real, kinds.real32, kinds.extent, kinds.logical)
    )
    for argument in abi:
        lines.extend(_fortran_line(abi_declaration(argument, kinds), 6))
    lines.extend(
        [
            f"    end subroutine {c_entry}",
            f"    subroutine {c_start}() bind(C, name='cpp_start_hot')",
            f"    end subroutine {c_start}",
            f"    subroutine {c_finish}() bind(C, name='cpp_finish_hot')",
            f"    end subroutine {c_finish}",
            "  end interface",
            "",
            "contains",
            "",
        ]
    )
    public_names = [symbol.name.lower() for symbol in function.parameters]
    private_names = [symbol.cpp_name for symbol in function.parameters]
    lines.extend(_fortran_list(f"subroutine {function.name}(", public_names, ")", 2))
    for symbol in function.parameters:
        lines.extend(_fortran_line(public_declaration(symbol, symbol.name.lower()), 4))
    lines.extend(_fortran_list(f"call {bridge}(", public_names, ")", 4))
    lines.extend([f"  end subroutine {function.name}", ""])
    lines.extend(_fortran_list(f"subroutine {bridge}(", private_names, ")", 2))
    lines.append("    intrinsic :: size, int, real, logical")
    for symbol in function.parameters:
        lines.extend(_fortran_line(public_declaration(symbol, symbol.cpp_name), 4))
    lines.extend(_fortran_list(f"call {c_entry}(", [abi_call(argument, kinds) for argument in abi], ")", 4))
    query_bodies = []
    if offload is not None:
        query_bodies += _fortran_list(f"logical function {offload.query_name}(", public_names,
                                     f") result({query_result})", 2)
        for symbol in function.parameters:
            query_bodies += _fortran_line(public_declaration(replace(symbol, intent="in"), symbol.name.lower()), 4)
        query_bodies += _fortran_list(f"{query_result} = {query_bridge}(", public_names, ")", 4)
        query_bodies += [f"  end function {offload.query_name}", ""]
        query_bodies += _fortran_list(f"logical function {query_bridge}(", private_names,
                                     f") result({query_result})", 2)
        query_bodies += ["    intrinsic :: size, int, real, logical"]
        for symbol in function.parameters:
            query_bodies += _fortran_line(public_declaration(replace(symbol, intent="in"), symbol.cpp_name), 4)
        query_args = []
        for argument in abi:
            value = abi_call(argument, kinds)
            if argument.dimension is None and not argument.symbol.rank:
                if argument.symbol in offload.query_scalars:
                    # Supported query inputs are default INTEGER captures.
                    # Passing their address defers reading protected inner
                    # bounds until the compiler's outer-domain guard permits it.
                    value = argument.symbol.cpp_name
                else:
                    zero = ".false." if argument.symbol.dtype is ScalarType.LOGICAL else "0"
                    value = value.replace(argument.symbol.cpp_name, zero)
            query_args.append(value)
        query_bodies += _fortran_list(f"{query_result} = {c_query}(", query_args, ") /= 0", 4)
        query_bodies += [f"  end function {query_bridge}", ""]
    lines.extend(
        [
            f"  end subroutine {bridge}",
            "",
            *query_bodies,
            "  subroutine start_hot()",
            f"    call {c_start}()",
            "  end subroutine start_hot",
            "",
            "  subroutine finish_hot()",
            f"    call {c_finish}()",
            "  end subroutine finish_hot",
            "",
            *bodies,
            f"end module {function.module}",
            "",
        ]
    )
    return "\n".join(lines)
