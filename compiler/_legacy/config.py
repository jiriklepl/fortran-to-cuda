"""Configuration for retained historical proof and calibration regression tests.

These types deliberately do not travel through the public CLI or graph source
dispatcher. They keep original proof fixtures useful while their common source
authority is reused by the graph compiler.
"""
from dataclasses import dataclass, replace

from compiler.driver.options import CompilerOptions
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig


@dataclass(frozen=True)
class LegacyCompilerOptions(CompilerOptions):
    scope_execution: str | None = "bounded"

    def _validate_scope_execution(self):
        if self.scope_execution not in {None, "bounded", "reached", "graph"}:
            raise CompilationError("unknown private reference execution mode")
        if self.scope_execution in {"reached", "graph"} and self.memory_model != "scoped":
            raise CompilationError("reference reached and graph execution require scoped memory")


@dataclass(frozen=True)
class LegacyOffloadConfig(OffloadConfig):
    scope_execution: str = "bounded"
    exact_window: dict | None = None
    scope_measurement: str | None = None

    def _validate_scope_execution(self):
        if self.scope_execution not in {"bounded", "reached", "graph"}:
            raise ValueError("Unknown private reference scope execution mode")

    def __post_init__(self):
        super().__post_init__()
        if self.exact_window is not None:
            if type(self.exact_window) is not dict:
                raise ValueError("Exact window cost request must be an object")
            coverage = self.exact_window.get("coverage")
            if coverage not in (None, "original_selection", "checked_interval"):
                raise ValueError("Unknown exact window cost coverage")
            execution = "bounded" if coverage == "original_selection" else "reached"
            if self.scope_execution != execution or self.policy not in {"auto", "sections"} or self.collective:
                raise ValueError(f"Exact window requires a serial {execution} auto/sections owner")
        if self.scope_measurement is not None and (
                type(self.scope_measurement) is not str or self.scope_measurement not in {"native", "resident"}
                or self.policy != "sections" or self.scope_execution != "reached"
                or self.collective or self.exact_window is not None):
            raise ValueError("Scope measurement requires a native/resident serial reached sections owner")

    @property
    def reached_execution(self):
        return self.scope_execution in {"reached", "graph"}

    def for_numerical_entry(self, **updates):
        return replace(self, exact_window=None, scope_measurement=None, **updates)
