"""Validated options shared by the public pipeline and command-line driver."""

from dataclasses import dataclass

from compiler.ir import CompilationError


@dataclass(frozen=True)
class CompilerOptions:
    opt_level: int = 1

    def __post_init__(self) -> None:
        if self.opt_level not in (0, 1):
            raise CompilationError("optimization level must be 0 or 1")
