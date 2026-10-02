"""Render CUDA execution and owned sessions from an explicit memory plan."""

from __future__ import annotations

from compiler.emission.c.declarations import cpp_declaration
from compiler.emission.common.abi import AbiArgument, abi_arguments
from compiler.emission.common.c_family import cpp_type, indent, render_assignment, render_expression
from compiler.emission.common.loops import sequential_block
from compiler.emission.common.sessions import (
    array_dimensions,
    buffer_name,
    session_definitions,
    session_names,
    workspace_registry_name,
)
from compiler.emission.common.symbols import host_symbols
from compiler.emission.cuda.kernels import generate_kernel, generate_launch
from compiler.ir import ConditionalRegion, ExecutionPlan, FunctionIR, HostBlock, ParallelRegion, SequentialRegion
from compiler.memory import MemoryOperation, MemoryPlan, plan_memory


def _memory_lines(operations: tuple[MemoryOperation, ...]) -> list[str]:
    lines = []
    for operation in operations:
        for symbol in operation.symbols:
            storage = buffer_name(symbol)
            if operation.kind == "host":
                lines.append(f"{symbol.cpp_name} = {storage}.host_data();")
            elif operation.kind == "device":
                lines.append(f"{symbol.cpp_name}_device = {storage}.device_data();")
            elif operation.kind in ("host_write", "device_write"):
                lines.append(f"{storage}.{operation.kind.replace('_write', '_written')}();")
        step = operation.step
        if operation.kind == "execute":
            if isinstance(step, ParallelRegion):
                lines.extend(generate_launch(step))
            elif isinstance(step, HostBlock):
                lines.extend(render_assignment(assignment) for assignment in step.assignments)
            elif isinstance(step, SequentialRegion):
                lines.extend(sequential_block(step.body, 0, [0]))
            else:
                raise TypeError(f"Unknown executable step: {type(step).__name__}")
        elif operation.kind == "branch":
            assert isinstance(step, ConditionalRegion)
            lines.append(f"if ({render_expression(step.condition)}) {{")
            lines.extend(indent(_memory_lines(operation.then_ops)))
            lines.append("} else {")
            lines.extend(indent(_memory_lines(operation.else_ops)))
            lines.append("}")
    return lines


def generate_cuda(
    function: FunctionIR,
    plan: ExecutionPlan,
    abi: tuple[AbiArgument, ...],
    common_header: str,
    *,
    memory: MemoryPlan | None = None,
) -> str:
    memory = memory or plan_memory(plan, function.parameters)
    names = session_names(function)
    arrays = tuple(symbol for symbol in function.parameters if symbol.rank)
    scalars = tuple(symbol for symbol in function.parameters if not symbol.rank)
    lines = [
        "#include <cuda_runtime.h>",
        "#include <cstddef>",
        "#include <cstdio>",
        "#include <cstdlib>",
        "#include <utility>",
        f'#include "{common_header}"',
        "",
        "namespace generated_kernels {",
        "using namespace indexing;",
        "using namespace timing;",
        "",
    ]
    for region in plan.regions:
        lines.extend(generate_kernel(region))
    run = [f"{cpp_type(symbol)} {symbol.cpp_name};" for symbol in host_symbols(function, plan)]
    run.append("measure_kernel_executions([&]() {")
    run.extend(indent(_memory_lines(memory.run)))
    run.append("});")
    lines.extend(session_definitions(function, abi, cuda_run=run, memory=memory))
    lines.extend(
        [
            'extern "C" void cpp_start_hot() { reset_timing_vectors(); }',
            'extern "C" void cpp_finish_hot() { print_timing_summary(); }',
            "",
            f'extern "C" void cpp_{function.name}(',
        ]
    )
    lines.extend(
        indent(
            [cpp_declaration(argument) + ("," if index + 1 < len(abi) else "") for index, argument in enumerate(abi)]
        )
    )
    lines.append(") {")
    lines.append(
        f"    const auto fort_internal_token = cpp_{names.create}({', '.join(a.name for a in abi_arguments(arrays))});"
    )
    arguments = ", ".join(["fort_internal_token", *[symbol.cpp_name for symbol in scalars]])
    lines.append(f"    cpp_{names.run}({arguments});")
    lines.append("    storage::synchronize();")
    outputs = next(operation.symbols for operation in memory.retrieve if operation.kind == "host")
    if outputs:
        lines.append(f"    auto& fort_internal_state = {workspace_registry_name(function)}.get(fort_internal_token);")
    for symbol in outputs:
        lines.append(f"    {buffer_name(symbol)}.update_host({symbol.cpp_name}, {array_dimensions(symbol)});")
    lines.extend([f"    cpp_{names.destroy}(fort_internal_token);", "}", "}", ""])
    return "\n".join(lines)
