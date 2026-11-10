"""Historical ordinary-backend reference for isolated compiler tests.

No production driver or application adapter imports this module. This preserves
numerical reference fixtures while the public compiler uses shared residency.
It is not a compatibility entry point or a supported application interface.
"""
from __future__ import annotations
from compiler.emission.c.generator import generate_cpp
from compiler.emission.c.sessions import append_cpu_sessions
from compiler.emission.common.abi import abi_arguments
from compiler.emission.common.c_family import cpp_type
from compiler.emission.cuda.generator import generate_cuda
from compiler.emission.fortran.generator import generate_fortran
from compiler.emission.driver import GeneratedSources, generate_sources as generate_shared_sources
from compiler.memory import plan_memory
from compiler.numerical_contract import numerical_build_contract
from compiler.ir import CompilationError, SourceLocation


def generate_sources(function, plan, *, common_header="common_functions.cuh", offload_config=None,
                     memory_model="call"):
    if memory_model == "scoped":
        return generate_shared_sources(function, plan, common_header=common_header,
                                       offload_config=offload_config, memory_model=memory_model)
    if memory_model != "call":
        raise ValueError("unknown historical reference memory model")
    if function.name.lower() in {"start_hot", "finish_hot", "knd"}:
        raise CompilationError("Entry procedure conflicts with the generated public interface (knd, start_hot, finish_hot)",
                               SourceLocation(function.source))
    if any(c in common_header for c in ('"', "\n", "\r")):
        raise CompilationError("Common header filename cannot contain quotes or newlines")
    if getattr(offload_config, "scope_transfers", "direct") != "direct":
        raise CompilationError("nondefault scope transfers require scoped memory")
    artifacts, scoped = {}, None
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
