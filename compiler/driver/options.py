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
    gpu_policy: str = "sections"
    memory_model: str = "scoped"
    scope_transfers: str = "direct"
    scope_execution: str | None = None

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
        self._validate_execution_configuration()
        self._validate_transfers()
        if self.scope_transfers != "direct" and self.memory_model != "scoped":
            raise CompilationError("nondefault scope transfers require scoped memory")
        self._validate_scope_execution()

    def _validate_execution_configuration(self) -> None:
        if self.memory_model != "scoped":
            raise CompilationError("ordinary-call generation was removed; use caller-owned scoped memory")
        if self.gpu_policy not in {"sections", "auto"}:
            raise CompilationError("ordinary GPU policies were removed; use sections or auto")

    def _validate_transfers(self) -> None:
        if self.scope_transfers == "pipelined":
            raise CompilationError("pipelined slab batching was removed; use shared graph execution")
        if self.scope_transfers not in {"direct", "pinned", "auto", "coalesced"}:
            raise CompilationError("scope transfers must be direct, pinned, auto, or coalesced")

    def _validate_scope_execution(self) -> None:
        if self.scope_execution not in {None, "graph"}:
            raise CompilationError("source scopes use graph execution")
        if self.scope_execution == "graph" and self.memory_model != "scoped":
            raise CompilationError("graph scope execution requires scoped memory")

    @property
    def resolved_schedule(self) -> str:
        return self.schedule or ("auto" if self.opt_level else "source")

    @property
    def resolved_indexing(self) -> str:
        return self.indexing or ("auto" if self.opt_level else "source")
