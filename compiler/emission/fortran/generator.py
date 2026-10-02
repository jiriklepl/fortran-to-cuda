"""Emit the Fortran module that bridges public callers to the C ABI."""

from __future__ import annotations

from compiler.emission.common.abi import AbiArgument
from compiler.emission.fortran.declarations import FortranKinds, abi_call, abi_declaration, public_declaration
from compiler.ir import FunctionIR


def _fortran_list(prefix: str, values: list[str], suffix: str, indentation: int) -> list[str]:
    """Continue every argument separately to stay within free-form line limits."""
    if not values:
        return [" " * indentation + prefix + suffix]
    lines = [" " * indentation + prefix + " &"]
    lines.extend(
        " " * (indentation + 4) + value + (", &" if index < len(values) - 1 else " &")
        for index, value in enumerate(values)
    )
    lines.append(" " * indentation + suffix)
    return lines


def _fortran_line(declaration: str, indentation: int) -> list[str]:
    line = " " * indentation + declaration
    if len(line) <= 132:
        return [line]
    # A rank-15 assumed shape and a long public dummy name can exceed the
    # free-form limit. A continuation before its name keeps both parts short.
    left, right = line.split("::", 1)
    return [left.rstrip() + " :: &", " " * (indentation + 4) + right.strip()]


def generate_fortran(function: FunctionIR, abi: tuple[AbiArgument, ...]) -> str:
    # Keep the public dummy labels for keyword callers. Conversion intrinsics
    # belong in a separate scope where user names cannot hide SIZE/INT/REAL.
    occupied = {symbol.name.lower() for symbol in function.parameters}
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
    )
    c_entry = unique_name("fort_internal_c_entry")
    bridge = unique_name("fort_internal_bridge")
    c_start = unique_name("fort_internal_c_start")
    c_finish = unique_name("fort_internal_c_finish")
    lines = [
        f"module {function.module}",
        "  use iso_c_binding, only: &",
        f"    {kinds.integer} => c_int, &",
        f"    {kinds.real} => c_double, &",
        f"    {kinds.real32} => c_float, &",
        f"    {kinds.extent} => c_size_t",
        "  implicit none",
        "  private",
        f"  public :: {function.name}, knd, start_hot, finish_hot",
        f"  integer, parameter :: knd = {kinds.real}",
        "",
        "  interface",
    ]
    lines.extend(
        _fortran_list(
            f"subroutine {c_entry}(",
            [argument.name for argument in abi],
            f") bind(C, name='cpp_{function.name}')",
            4,
        )
    )
    lines.append(f"      import :: {kinds.integer}, {kinds.real}, {kinds.real32}, {kinds.extent}")
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
    lines.append("    intrinsic :: size, int, real")
    for symbol in function.parameters:
        lines.extend(_fortran_line(public_declaration(symbol, symbol.cpp_name), 4))
    lines.extend(_fortran_list(f"call {c_entry}(", [abi_call(argument, kinds) for argument in abi], ")", 4))
    lines.extend(
        [
            f"  end subroutine {bridge}",
            "",
            "  subroutine start_hot()",
            f"    call {c_start}()",
            "  end subroutine start_hot",
            "",
            "  subroutine finish_hot()",
            f"    call {c_finish}()",
            "  end subroutine finish_hot",
            "",
            f"end module {function.module}",
            "",
        ]
    )
    return "\n".join(lines)
