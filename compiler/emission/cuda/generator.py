"""Render CUDA execution and owned sessions from an explicit memory plan."""

from __future__ import annotations

from compiler.emission.c.declarations import cpp_declaration
from compiler.emission.common.abi import AbiArgument, abi_arguments
from compiler.emission.common.c_family import cpp_type, indent, render_assignment
from compiler.emission.common.loops import sequential_block
from compiler.emission.common.memory import render_memory
from compiler.emission.common.sessions import (
    execution_definition,
    lifecycle_definition,
    lifecycle_name,
    session_definitions,
    workspace_state_name,
)
from compiler.emission.common.symbols import host_symbols
from compiler.emission.cuda.kernels import generate_kernel, generate_launch
from compiler.ir import ExecutionPlan, FunctionIR, HostBlock, ParallelRegion, SequentialRegion
from compiler.memory import MemoryPlan, plan_memory, validate_memory


def _execute(step) -> list[str]:
    if isinstance(step, ParallelRegion):
        return generate_launch(step)
    if isinstance(step, HostBlock):
        return [render_assignment(assignment) for assignment in step.assignments]
    if isinstance(step, SequentialRegion):
        return sequential_block(step.body, 0, [0])
    raise TypeError(f"Unknown executable step: {type(step).__name__}")


def generate_cuda(
    function: FunctionIR,
    plan: ExecutionPlan,
    abi: tuple[AbiArgument, ...],
    common_header: str,
    *,
    memory: MemoryPlan | None = None,
    ordinary_memory: MemoryPlan | None = None,
) -> str:
    if memory is None:
        memory = plan_memory(plan, function.parameters, acquisition_policy="dedicated")
        if ordinary_memory is None:
            ordinary_memory = plan_memory(plan, function.parameters, acquisition_policy="pooled")
    elif ordinary_memory is None:
        # A caller-supplied legacy plan remains authoritative for both APIs.
        ordinary_memory = memory
    validate_memory(memory, function.parameters, allow_pooled=False)
    validate_memory(ordinary_memory, function.parameters)
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

    def run_body(operations):
        return [
            "timing::record_run();",
            *[f"{cpp_type(symbol)} {symbol.cpp_name};" for symbol in host_symbols(function, plan)],
            *render_memory(operations, device=True, execute=_execute),
        ]

    lines.extend(session_definitions(function, run_body=run_body(memory.run), device=True, memory=memory))
    ordinary_helpers = {}
    for phase in ("create", "run", "retrieve", "destroy"):
        operations = getattr(ordinary_memory, phase)
        if operations == getattr(memory, phase):
            ordinary_helpers[phase] = lifecycle_name(function, phase)
            continue
        name = ordinary_helpers[phase] = lifecycle_name(function, "ordinary_" + phase)
        if phase == "run":
            lines.extend(execution_definition(function, run_body(operations), device=True, name=name))
        else:
            lines.extend(lifecycle_definition(function, phase, operations, device=True, name=name))
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
    lines.append("    timing::ProfiledCallGuard fort_internal_profile;")
    dimensions = ", ".join(a.name for a in abi_arguments(arrays) if a.dimension is not None)
    lines.append(f"    {workspace_state_name(function)} fort_internal_state{{{dimensions}}};")
    array_arguments = ", ".join(["fort_internal_state", *[argument.name for argument in abi_arguments(arrays)]])
    lines.append(f"    {ordinary_helpers['create']}({array_arguments});")
    arguments = ", ".join(["fort_internal_state", *[symbol.cpp_name for symbol in scalars]])
    lines.append(f"    {ordinary_helpers['run']}({arguments});")
    lines.append(f"    {ordinary_helpers['retrieve']}({array_arguments});")
    lines.append(f"    {ordinary_helpers['destroy']}(fort_internal_state);")
    lines.extend(["}", "}", ""])
    return "\n".join(lines)
