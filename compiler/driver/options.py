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
    gpu_policy: str = "always"
    memory_model: str = "call"
    scope_transfers: str = "direct"
    scope_execution: str = "bounded"

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
        if self.gpu_policy not in {"always", "sections", "auto", "chunked", "hybrid"}:
            raise CompilationError("GPU policy must be always, sections, auto, chunked, or hybrid")
        if self.memory_model not in {"call", "scoped"}:
            raise CompilationError("memory model must be call or scoped")
        if self.memory_model == "scoped" and self.gpu_policy not in {"sections", "auto"}:
            raise CompilationError("scoped memory currently requires sections or auto GPU policy")
        if self.scope_transfers not in {"direct", "pinned", "pipelined", "auto"}:
            raise CompilationError("scope transfers must be direct, pinned, pipelined, or auto")
        if self.scope_transfers != "direct" and self.memory_model != "scoped":
            raise CompilationError("nondefault scope transfers require scoped memory")
        if self.scope_execution not in {"bounded", "reached"}:
            raise CompilationError("scope execution must be bounded or reached")
        if self.scope_execution != "bounded" and self.memory_model != "scoped":
            raise CompilationError("reached scope execution requires scoped memory")

    @property
    def resolved_schedule(self) -> str:
        return self.schedule or ("auto" if self.opt_level else "source")

    @property
    def resolved_indexing(self) -> str:
        return self.indexing or ("auto" if self.opt_level else "source")
