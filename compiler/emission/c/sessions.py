"""Render CPU sessions through the same lifecycle and coherence operations."""

from compiler.emission.c.generator import cpp_plan_lines
from compiler.emission.common.c_family import cpp_type
from compiler.emission.common.memory import render_memory
from compiler.emission.common.sessions import session_definitions
from compiler.emission.common.symbols import host_symbols
from compiler.ir import ExecutionPlan, FunctionIR
from compiler.memory import MemoryPlan, validate_memory


def append_cpu_sessions(function: FunctionIR, plan: ExecutionPlan, *, memory: MemoryPlan) -> str:
    validate_memory(memory, function.parameters, allow_pooled=False)
    run = [f"{cpp_type(symbol)} {symbol.cpp_name};" for symbol in host_symbols(function, plan)]
    run.extend(render_memory(memory.run, device=False, execute=lambda step: cpp_plan_lines(ExecutionPlan((step,)))))
    return "\n".join(
        [
            "",
            "namespace generated_kernels {",
            *session_definitions(function, run_body=run, device=False, memory=memory),
            "}",
            "",
        ]
    )
