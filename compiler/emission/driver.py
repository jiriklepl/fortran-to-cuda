"""Coordinate source generation through a single shared ABI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from compiler.emission.c.generator import generate_cpp
from compiler.emission.c.sessions import append_cpu_sessions
from compiler.emission.common.abi import abi_arguments
from compiler.emission.common.c_family import cpp_type
from compiler.emission.cuda.generator import generate_cuda
from compiler.emission.fortran.generator import generate_fortran
from compiler.ir import CompilationError, FunctionIR, SourceLocation
from compiler.memory import plan_memory
from compiler.numerical_contract import numerical_build_contract

if TYPE_CHECKING:
    from compiler.ir import ExecutionPlan


@dataclass(frozen=True)
class GeneratedSources:
    cuda: str
    cpp: str
    fortran: str
    offload: dict | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    scoped: dict | None = None
    numerical_contract: dict = field(default_factory=numerical_build_contract)


def generate_sources(
    function: FunctionIR, plan: ExecutionPlan, *, common_header: str = "common_functions.cuh", offload_config=None,
    memory_model: str = "call",
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
    if memory_model not in {"call", "scoped"}:
        raise CompilationError("memory model must be call or scoped")
    if memory_model != "scoped" and getattr(offload_config, "scope_transfers", "direct") != "direct":
        raise CompilationError("nondefault scope transfers require scoped memory")
    artifacts, scoped = {}, None
    if memory_model == "scoped":
        import json

        from compiler.emission.common.resources import read_scoped_runtime
        from compiler.emission.cuda.scoped import generate_scoped

        if offload_config is None or offload_config.policy not in {"sections", "auto"}:
            raise CompilationError("scoped memory requires an explicit sections or auto policy")
        artifacts, runtime = read_scoped_runtime()
        shared = generate_scoped(function, plan, offload_config, common_header, runtime_id=runtime["runtime_id"])
        artifacts.update({"shared_entry.cu": shared.cuda, "shared_interface.f90": shared.fortran,
                          "scoped-runtime.json": json.dumps(runtime, indent=2) + "\n"})
        scoped = {**shared.report, "runtime": runtime, "numerical_contract": numerical_build_contract()}
    abi = abi_arguments(function.parameters)
    memory = plan_memory(plan, function.parameters, acquisition_policy="dedicated")
    ordinary_memory = plan_memory(plan, function.parameters, acquisition_policy="pooled")
    offload = None
    if offload_config is not None and offload_config.policy != "always":
        from compiler.emission.cuda.offload import generate_offload
        offload = generate_offload(function, plan, offload_config)
        offload.report["numerical_contract"] = numerical_build_contract()
    cpp = generate_cpp(function, plan, abi, common_header) + append_cpu_sessions(function, plan, memory=memory)
    if offload is not None:
        from compiler.emission.c.declarations import cpp_declaration
        signature = ", ".join(cpp_declaration(a) if a.symbol.rank else
                              f"const {cpp_type(a.symbol)} &{a.name}" for a in abi)
        cpp += f'\nextern "C" int cpp_{offload.query_name}({signature}) {{ return 0; }}\n'
    return GeneratedSources(
        cuda=generate_cuda(function, plan, abi, common_header, memory=memory, ordinary_memory=ordinary_memory,
                           offload=offload),
        cpp=cpp,
        fortran=generate_fortran(function, abi, offload=offload),
        offload=None if offload is None else offload.report,
        artifacts=artifacts,
        scoped=scoped,
    )
