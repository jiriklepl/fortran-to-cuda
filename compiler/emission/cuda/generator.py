"""Emit CUDA host wrappers, allocation, and source-order synchronization."""

from __future__ import annotations

from typing import TYPE_CHECKING

from compiler.emission.c.declarations import cpp_declaration
from compiler.emission.common.abi import AbiArgument, dimension_name
from compiler.emission.common.c_family import cpp_type, indent, render_assignment, render_expression
from compiler.emission.common.symbols import host_symbols
from compiler.emission.cuda.kernels import generate_kernel, generate_launch
from compiler.emission.cuda.transfers import array_bytes, generate_transfer
from compiler.ir import ArrayAccess, ConditionalRegion, FunctionIR, HostBlock, ParallelRegion, Symbol, walk_expr

if TYPE_CHECKING:
    from compiler.ir import ExecutionPlan


def _plan_lines(plan: ExecutionPlan, arrays: tuple[Symbol, ...], device_dirty: set[Symbol]) -> list[str]:
    """Render ordered host/device work, synchronizing branch entry and joins."""
    lines = []
    for step in plan.steps:
        if isinstance(step, ParallelRegion):
            bounds = [expression for loop in step.loops for expression in (loop.lower, loop.upper)]
            bounds.extend(loop.step for loop in step.loops if not isinstance(loop.step, int))
            bound_reads = {
                node.symbol for expression in bounds for node in walk_expr(expression) if isinstance(node, ArrayAccess)
            }
            downloads = tuple(symbol for symbol in arrays if symbol in device_dirty & bound_reads)
            lines.extend(generate_transfer(downloads, to_device=False))
            device_dirty.difference_update(downloads)
            lines.extend(generate_launch(step))
            device_dirty.update(step.write_symbols)
        elif isinstance(step, HostBlock):
            # A partial host write must first preserve every device-produced
            # value elsewhere in the array, even if it does not read that value.
            needed = device_dirty & (set(step.read_symbols) | set(step.write_symbols))
            downloads = tuple(symbol for symbol in arrays if symbol in needed)
            lines.extend(generate_transfer(downloads, to_device=False))
            device_dirty.difference_update(downloads)
            lines.extend(render_assignment(assignment) for assignment in step.assignments)
            uploads = tuple(symbol for symbol in arrays if symbol in step.write_symbols)
            lines.extend(generate_transfer(uploads, to_device=True))
        elif isinstance(step, ConditionalRegion):
            # Both copies agree at branch boundaries. Each arm can then use
            # the ordinary source-order transfer logic independently.
            downloads = tuple(symbol for symbol in arrays if symbol in device_dirty)
            lines.extend(generate_transfer(downloads, to_device=False))
            device_dirty.clear()
            lines.append(f"if ({render_expression(step.condition)}) {{")
            for index, branch in enumerate((step.then_plan, step.else_plan)):
                if index:
                    lines.append("} else {")
                branch_dirty: set[Symbol] = set()
                lines.extend(indent(_plan_lines(branch, arrays, branch_dirty)))
                downloads = tuple(symbol for symbol in arrays if symbol in branch_dirty)
                lines.extend(indent(generate_transfer(downloads, to_device=False)))
            lines.append("}")
        else:
            raise TypeError(f"Unknown execution step: {type(step).__name__}")
    return lines


def generate_cuda(function: FunctionIR, plan: ExecutionPlan, abi: tuple[AbiArgument, ...], common_header: str) -> str:
    arrays = tuple(symbol for symbol in function.parameters if symbol.rank)
    lines = [
        "#include <cuda_runtime.h>",
        "#include <cstddef>",
        "#include <cstdio>",
        "#include <cstdlib>",
        "#include <utility>",
        "#ifdef USE_PINNED_MEMORY",
        "#include <unordered_set>",
        "#endif",
        "#define MEASURE_CUDA_EXECUTION_TIME",
        f'#include "{common_header}"',
        "",
        "namespace generated_kernels {",
        "using namespace indexing;",
        "using namespace timing;",
        "#ifdef USE_PINNED_MEMORY",
        "static std::unordered_set<const void*> fort_internal_pinned_ptrs;",
        "#endif",
        "",
    ]
    for region in plan.regions:
        lines.extend(generate_kernel(region))
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
            [cpp_declaration(argument) + ("," if index < len(abi) - 1 else "") for index, argument in enumerate(abi)]
        )
    )
    lines.append(") {")
    for symbol in arrays:
        dimensions = " * ".join(dimension_name(symbol, dimension) for dimension in range(1, symbol.rank + 1))
        lines.extend(
            indent(
                [
                    f"{cpp_type(symbol)}* {symbol.cpp_name}_device = nullptr;",
                    f"const std::size_t {array_bytes(symbol)} = sizeof({cpp_type(symbol)}) * {dimensions};",
                ]
            )
        )
    lines.append("    #ifdef USE_PINNED_MEMORY")
    for symbol in arrays:
        lines.extend(
            indent(
                [
                    f"if ({array_bytes(symbol)} > 0 && fort_internal_pinned_ptrs.insert({symbol.cpp_name}).second) {{",
                    f"    CUCH(cudaHostRegister(const_cast<{cpp_type(symbol)}*>({symbol.cpp_name}), {array_bytes(symbol)}, cudaHostRegisterPortable));",
                    "}",
                ]
            )
        )
    lines.extend(["    #endif", "    measure_alloc([&]() {"])
    lines.extend(
        indent(
            [
                f"if ({array_bytes(symbol)} > 0) CUCH(cudaMalloc(reinterpret_cast<void**>(&{symbol.cpp_name}_device), {array_bytes(symbol)}));"
                for symbol in arrays
            ],
            2,
        )
    )
    inputs = tuple(symbol for symbol in arrays if symbol.intent != "out")
    outputs = tuple(symbol for symbol in arrays if symbol.intent != "in")
    input_bytes = " + ".join(array_bytes(symbol) for symbol in inputs) or "0"
    output_bytes = " + ".join(array_bytes(symbol) for symbol in outputs) or "0"
    lines.extend(["    });", f"    measure_h2d({input_bytes}, [&]() {{"])
    lines.extend(
        indent(
            [
                f"if ({array_bytes(symbol)} > 0) CUCH(cudaMemcpy({symbol.cpp_name}_device, {symbol.cpp_name}, {array_bytes(symbol)}, cudaMemcpyHostToDevice));"
                for symbol in inputs
            ],
            2,
        )
    )
    lines.append("    });")
    lines.extend(indent([f"{cpp_type(symbol)} {symbol.cpp_name};" for symbol in host_symbols(function, plan)]))
    lines.append("    measure_kernel_executions([&]() {")
    lines.extend(indent(_plan_lines(plan, arrays, set()), 2))
    lines.extend(["    });", "    CUCH(cudaDeviceSynchronize());", f"    measure_d2h({output_bytes}, [&]() {{"])
    lines.extend(
        indent(
            [
                f"if ({array_bytes(symbol)} > 0) CUCH(cudaMemcpy({symbol.cpp_name}, {symbol.cpp_name}_device, {array_bytes(symbol)}, cudaMemcpyDeviceToHost));"
                for symbol in outputs
            ],
            2,
        )
    )
    lines.extend(["    });", "    measure_free([&]() {"])
    lines.extend(
        indent(
            [
                f"if ({symbol.cpp_name}_device != nullptr) CUCH(cudaFree({symbol.cpp_name}_device));"
                for symbol in arrays
            ],
            2,
        )
    )
    lines.extend(["    });", "}", "}", ""])
    return "\n".join(lines)
