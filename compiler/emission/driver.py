"""Coordinate source generation through a single shared ABI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from compiler.emission.c.generator import generate_cpp
from compiler.emission.common.abi import abi_arguments
from compiler.emission.cuda.generator import generate_cuda
from compiler.emission.fortran.generator import generate_fortran
from compiler.ir import CompilationError, FunctionIR, SourceLocation

if TYPE_CHECKING:
    from compiler.ir import ExecutionPlan


@dataclass(frozen=True)
class GeneratedSources:
    cuda: str
    cpp: str
    fortran: str


def generate_sources(
    function: FunctionIR, plan: ExecutionPlan, *, common_header: str = "common_functions.cuh"
) -> GeneratedSources:
    """Generate all output text from one ABI before the caller publishes files."""
    if function.name.lower() in {"start_hot", "finish_hot", "knd"}:
        raise CompilationError(
            f"Entry procedure '{function.name}' conflicts with the generated public interface "
            "(knd, start_hot, finish_hot)",
            SourceLocation(function.source),
        )
    if any(character in common_header for character in ('"', "\n", "\r")):
        raise CompilationError("Common header filename cannot contain quotes or newlines")
    abi = abi_arguments(function.parameters)
    return GeneratedSources(
        cuda=generate_cuda(function, plan, abi, common_header),
        cpp=generate_cpp(function, plan, abi, common_header),
        fortran=generate_fortran(function, abi),
    )
