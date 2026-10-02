"""Validated options shared by the public pipeline and command-line driver."""

from dataclasses import dataclass

from compiler.ir import CompilationError


@dataclass(frozen=True)
class CompilerOptions:
    opt_level: int = 1
    schedule: str | None = None
    tile_sizes: tuple[int, ...] = ()
    fallback: str = "error"
    indexing: str | None = None

    def __post_init__(self) -> None:
        if self.opt_level not in (0, 1):
            raise CompilationError("optimization level must be 0 or 1")
        if self.schedule not in (None, "source", "auto"):
            raise CompilationError("schedule must be source or auto")
        if any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in self.tile_sizes):
            raise CompilationError("tile sizes must be positive integers")
        if self.fallback not in ("error", "host"):
            raise CompilationError("fallback must be error or host")
        if self.indexing not in (None, "source", "auto"):
            raise CompilationError("indexing must be source or auto")

    @property
    def resolved_schedule(self) -> str:
        return self.schedule or ("auto" if self.opt_level else "source")

    @property
    def resolved_indexing(self) -> str:
        return self.indexing or ("auto" if self.opt_level else "source")
