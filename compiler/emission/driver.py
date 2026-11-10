"""Coordinate source generation through a single shared ABI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from compiler.ir import CompilationError, FunctionIR
from compiler.numerical_contract import numerical_build_contract

if TYPE_CHECKING:
    from compiler.ir import ExecutionPlan


@dataclass(frozen=True)
class GeneratedSources:
    # Scoped entries publish only their shared ABI and runtime artifacts.
    # None means this ABI was not generated, rather than an empty source file.
    cuda: str | None = None
    cpp: str | None = None
    fortran: str | None = None
    offload: dict | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    scoped: dict | None = None
    numerical_contract: dict = field(default_factory=numerical_build_contract)


def generate_sources(
    function: FunctionIR, plan: ExecutionPlan, *, common_header: str = "common_functions.cuh", offload_config=None,
    memory_model: str = "scoped",
) -> GeneratedSources:
    """Emit the shared graph backend; callers own storage and publication."""
    import json
    from compiler.emission.common.resources import read_scoped_runtime
    from compiler.emission.cuda.scoped import generate_scoped
    from compiler.offload.config import OffloadConfig

    if memory_model != "scoped":
        raise CompilationError("ordinary-call generation was removed; use caller-owned scoped memory")
    config = offload_config or OffloadConfig(policy="sections")
    if config.policy not in {"sections", "auto"}:
        raise CompilationError("ordinary GPU policies were removed; use sections or auto with shared graph execution")
    if any(character in common_header for character in ('"', "\n", "\r")):
        raise CompilationError("Common header filename cannot contain quotes or newlines")
    artifacts, runtime = read_scoped_runtime()
    shared = generate_scoped(function, plan, config, common_header, runtime_id=runtime["runtime_id"])
    artifacts.update({"shared_entry.cu": shared.cuda, "shared_interface.f90": shared.fortran,
                      "scoped-runtime.json": json.dumps(runtime, indent=2) + "\n"})
    scoped = {**shared.report, "runtime": runtime, "numerical_contract": numerical_build_contract()}
    return GeneratedSources(artifacts=artifacts, scoped=scoped)
