"""Emit C++, CUDA, and the Fortran bridge from a checked computation plan."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from compiler.ir import (
    ArrayAccess,
    Assignment,
    Binary,
    Block,
    CompilationError,
    Expr,
    FunctionIR,
    Literal,
    Loop,
    Reference,
    ScalarType,
    Size,
    SourceLocation,
    Symbol,
    Unary,
    referenced_symbols,
    walk_expr,
)

if TYPE_CHECKING:
    from compiler.analysis import ExecutionPlan, ParallelRegion


@dataclass(frozen=True)
class GeneratedSources:
    cuda: str
    cpp: str
    fortran: str


@dataclass(frozen=True)
class _FortranKinds:
    integer: str
    real: str
    real32: str
    extent: str


@dataclass(frozen=True)
class _AbiArgument:
    """One entry in the shared C/Fortran parameter list."""

    symbol: Symbol
    dimension: int | None = None

    @property
    def name(self) -> str:
        return self.symbol.cpp_name if self.dimension is None else _dimension_name(self.symbol, self.dimension)

    @property
    def cpp_declaration(self) -> str:
        if self.dimension is not None:
            return f"std::size_t {self.name}"
        if self.symbol.rank:
            const = "const " if self.symbol.intent == "in" else ""
            return f"{const}{_cpp_type(self.symbol)}* __restrict__ {self.name}"
        return f"{_cpp_type(self.symbol)} {self.name}"

    def fortran_declaration(self, kinds: _FortranKinds) -> str:
        if self.dimension is not None:
            return f"integer({kinds.extent}), value, intent(in) :: {self.name}"
        if self.symbol.dtype == ScalarType.INTEGER:
            kind = f"integer({kinds.integer})"
        else:
            real_kind = kinds.real32 if self.symbol.dtype == ScalarType.REAL32 else kinds.real
            kind = f"real({real_kind})"
        if self.symbol.rank:
            intent = self.symbol.intent or "inout"
            return f"{kind}, intent({intent}) :: {self.name}(*)"
        return f"{kind}, value, intent(in) :: {self.name}"

    def fortran_call(self, kinds: _FortranKinds) -> str:
        name = self.symbol.cpp_name
        if self.dimension is not None:
            return f"size({name}, {self.dimension}, kind={kinds.extent})"
        if self.symbol.rank:
            return name
        cast = "int" if self.symbol.dtype == ScalarType.INTEGER else "real"
        kind = {ScalarType.INTEGER: kinds.integer, ScalarType.REAL: kinds.real, ScalarType.REAL32: kinds.real32}[
            self.symbol.dtype
        ]
        return f"{cast}({name}, kind={kind})"


def _abi_arguments(symbols: tuple[Symbol, ...]) -> tuple[_AbiArgument, ...]:
    return tuple(
        argument
        for symbol in symbols
        for argument in (
            _AbiArgument(symbol),
            *(_AbiArgument(symbol, dimension) for dimension in range(1, symbol.rank + 1)),
        )
    )


def _cpp_type(symbol: Symbol) -> str:
    return {ScalarType.INTEGER: "int", ScalarType.REAL: "double", ScalarType.REAL32: "float"}[symbol.dtype]


def _dimension_name(symbol: Symbol, dimension: int) -> str:
    return f"{symbol.cpp_name}_dim{dimension}"


def _expression(expression: Expr) -> str:
    if isinstance(expression, Literal):
        if expression.dtype is ScalarType.INTEGER:
            # Fortran integer literals are decimal even with leading zeros.
            return str(int(expression.value))
        # The frontend normalizes literals; accepting Fortran exponent spelling
        # here also makes standalone construction of the public IR convenient.
        value = expression.value.split("_", 1)[0].replace("d", "e").replace("D", "e")
        return value + "f" if expression.dtype is ScalarType.REAL32 else value
    if isinstance(expression, Reference):
        return expression.symbol.cpp_name
    if isinstance(expression, Size):
        return f"static_cast<int>({_dimension_name(expression.symbol, expression.dimension)})"
    if isinstance(expression, ArrayAccess):
        values = [_expression(index) for index in expression.indices]
        values.extend(_dimension_name(expression.symbol, dim) for dim in range(1, expression.symbol.rank + 1))
        return f"{expression.symbol.cpp_name}[F_IDX({', '.join(values)})]"
    if isinstance(expression, Unary):
        return f"({expression.operator}{_expression(expression.operand)})"
    if isinstance(expression, Binary):
        if expression.operator not in {"+", "-", "*", "/"}:
            raise CompilationError(f"unsupported emission operator: {expression.operator}")
        return f"({_expression(expression.left)} {expression.operator} {_expression(expression.right)})"
    raise CompilationError(f"unsupported IR expression: {type(expression).__name__}")


def _assignment(assignment: Assignment) -> str:
    return f"{_expression(assignment.target)} = {_expression(assignment.value)};"


def _indent(lines: list[str], depth: int = 1) -> list[str]:
    return ["    " * depth + line if line else "" for line in lines]


def _loop_step(loop: Loop) -> str:
    return str(loop.step) if isinstance(loop.step, int) else _expression(loop.step)


def _step_symbols(loop: Loop) -> frozenset[Symbol]:
    return frozenset() if isinstance(loop.step, int) else referenced_symbols(loop.step)


def _block_symbols(block: Block) -> set[Symbol]:
    used: set[Symbol] = set()
    for statement in block.statements:
        if isinstance(statement, Assignment):
            used.update(referenced_symbols(statement.value))
            used.update(referenced_symbols(statement.target))
        else:
            used.update(referenced_symbols(statement.lower))
            used.update(referenced_symbols(statement.upper))
            used.update(_step_symbols(statement))
            used.add(statement.iterator)
            used.update(_block_symbols(statement.body))
    return used


def _host_symbols(function: FunctionIR, plan: ExecutionPlan) -> tuple[Symbol, ...]:
    """Only locals used by host blocks or captured by a launch need host storage."""
    used: set[Symbol] = set()
    for step in plan.steps:
        if hasattr(step, "loops"):
            for loop in step.loops:
                used.update(referenced_symbols(loop.lower))
                used.update(referenced_symbols(loop.upper))
                used.update(_step_symbols(loop))
            used.update(symbol for symbol in step.captured_symbols if not symbol.rank)
        else:
            for assignment in step.assignments:
                used.update(referenced_symbols(assignment.value))
                used.update(referenced_symbols(assignment.target))
    return tuple(symbol for symbol in function.symbols if symbol in used and not symbol.parameter)


def _region_body(region: ParallelRegion) -> Block:
    return region.body if region.body is not None else Block(region.assignments)


def _region_symbols(region: ParallelRegion) -> tuple[Symbol, ...]:
    used = set(region.captured_symbols) | _block_symbols(_region_body(region))
    # Mapped bounds and strides are captured as separate scalar snapshots. Their
    # source expressions need not be evaluated again in the device kernel.
    used.difference_update(region.private_symbols)
    used.difference_update(loop.iterator for loop in region.loops)
    return tuple(sorted(used, key=lambda symbol: symbol.id))


def _loop_snapshot(loop: Loop, suffix: str, active: str | None = None, *, device: bool = False) -> list[str]:
    """Evaluate bounds once and count positive or negative Fortran DO trips."""
    lower = f"fort_internal_lower{suffix}"
    upper = f"fort_internal_upper{suffix}"
    stride = f"fort_internal_stride{suffix}"
    extent = f"fort_internal_extent{suffix}"

    def evaluate(expression: str) -> str:
        return f"({active}) ? ({expression}) : 0" if active else expression

    location = str(loop.location).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")
    lines = [
        f"const int {lower} = {evaluate(_expression(loop.lower))};",
        f"const int {upper} = {evaluate(_expression(loop.upper))};",
        f"const int {stride} = {evaluate(_loop_step(loop))};",
    ]
    condition = f"({active}) && {stride} == 0" if active else f"{stride} == 0"
    failure = (
        [f'    printf("%s\\n", "{location}: DO stride must be nonzero");', '    asm("trap;");']
        if device
        else [f'    std::cerr << "{location}: DO stride must be nonzero" << std::endl;', "    std::abort();"]
    )
    lines.extend(
        [
            f"if ({condition}) {{",
            *failure,
            "}",
            f"const std::size_t {extent} = {stride} > 0 && {upper} >= {lower}",
            f"    ? static_cast<std::size_t>((static_cast<long long>({upper}) - {lower}) / {stride} + 1)",
            f"    : {stride} < 0 && {lower} >= {upper}",
            f"        ? static_cast<std::size_t>((static_cast<long long>({lower}) - {upper})",
            f"            / -static_cast<long long>({stride}) + 1) : 0;",
        ]
    )
    return lines


def _iteration_value(suffix: str, ordinal: str) -> str:
    return (
        f"static_cast<int>(static_cast<long long>(fort_internal_lower{suffix})"
        f" + static_cast<long long>({ordinal}) * fort_internal_stride{suffix})"
    )


def _sequential_block(block: Block, depth: int, counter: list[int], *, device: bool = False) -> list[str]:
    lines: list[str] = []
    for statement in block.statements:
        if isinstance(statement, Assignment):
            lines.extend(_indent([_assignment(statement)], depth))
            continue
        suffix = f"_serial{counter[0]}"
        counter[0] += 1
        ordinal = f"fort_internal_ordinal{suffix}"
        lines.extend(_indent(["{"], depth))
        lines.extend(_indent(_loop_snapshot(statement, suffix, device=device), depth + 1))
        lines.extend(
            _indent(
                [
                    f"for (std::size_t {ordinal} = 0; {ordinal} < fort_internal_extent{suffix}; ++{ordinal}) {{",
                    f"    {statement.iterator.cpp_name} = {_iteration_value(suffix, ordinal)};",
                ],
                depth + 1,
            )
        )
        lines.extend(_sequential_block(statement.body, depth + 2, counter, device=device))
        lines.extend(
            _indent(
                [
                    "}",
                    f"{statement.iterator.cpp_name} = {_iteration_value(suffix, f'fort_internal_extent{suffix}')};",
                ],
                depth + 1,
            )
        )
        lines.extend(_indent(["}"], depth))
    return lines


def _mapped_snapshots(region: ParallelRegion) -> list[str]:
    lines: list[str] = []
    for dimension, loop in enumerate(region.loops):
        active = " && ".join(f"fort_internal_extent{previous} > 0" for previous in range(dimension))
        lines.extend(_loop_snapshot(loop, str(dimension), active or None))
    return lines


def _cpp_region(region: ParallelRegion) -> list[str]:
    lines = ["{", f"    // Verified parallel region {region.id}."]
    lines.extend(_indent(_mapped_snapshots(region)))
    pragma = "    #pragma omp parallel for"
    if len(region.loops) > 1:
        pragma += f" collapse({len(region.loops)})"
    lines.append(pragma)
    depth = 1
    for dimension in range(len(region.loops)):
        ordinal = f"fort_internal_ordinal{dimension}"
        lines.extend(
            _indent(
                [
                    f"for (std::size_t {ordinal} = 0;",
                    f"     {ordinal} < fort_internal_extent{dimension}; ++{ordinal}) {{",
                ],
                depth,
            )
        )
        depth += 1
    for dimension, loop in enumerate(region.loops):
        lines.extend(
            _indent(
                [
                    f"const int {loop.iterator.cpp_name} = {_iteration_value(str(dimension), f'fort_internal_ordinal{dimension}')};"
                ],
                depth,
            )
        )
    lines.extend(_indent([f"{_cpp_type(symbol)} {symbol.cpp_name};" for symbol in region.private_symbols], depth))
    lines.extend(_sequential_block(_region_body(region), depth, [0]))
    for _ in region.loops:
        depth -= 1
        lines.extend(_indent(["}"], depth))
    lines.append("}")
    return lines


def _generate_cpp(function: FunctionIR, plan: ExecutionPlan, abi: tuple[_AbiArgument, ...], common_header: str) -> str:
    lines = [
        "#include <cstdlib>",
        f'#include "{common_header}"',
        "",
        "namespace generated_kernels {",
        "using namespace generated_kernels::indexing;",
        "",
        'extern "C" void cpp_start_hot() {}',
        'extern "C" void cpp_finish_hot() {}',
        "",
        f'extern "C" void cpp_{function.name}(',
    ]
    lines.extend(
        _indent(
            [argument.cpp_declaration + ("," if index < len(abi) - 1 else "") for index, argument in enumerate(abi)]
        )
    )
    lines.append(") {")
    lines.extend(_indent([f"{_cpp_type(symbol)} {symbol.cpp_name};" for symbol in _host_symbols(function, plan)]))
    for step in plan.steps:
        if hasattr(step, "loops"):
            lines.extend(_indent(_cpp_region(step)))
        else:
            lines.extend(_indent([_assignment(assignment) for assignment in step.assignments]))
    lines.extend(["}", "}", ""])
    return "\n".join(lines)


def _kernel_name(region: ParallelRegion) -> str:
    return f"kernel_region_{region.id}_device"


def _cuda_region(region: ParallelRegion) -> list[str]:
    symbols = _region_symbols(region)
    parameters = [argument.cpp_declaration for argument in _abi_arguments(symbols)]
    for dimension in range(len(region.loops)):
        parameters.extend(
            [
                f"int fort_internal_lower{dimension}",
                f"int fort_internal_stride{dimension}",
                f"std::size_t fort_internal_extent{dimension}",
            ]
        )
    parameters.append("std::size_t fort_internal_total")
    lines = [f"__global__ void {_kernel_name(region)}("]
    lines.extend(
        _indent(
            [parameter + ("," if index < len(parameters) - 1 else "") for index, parameter in enumerate(parameters)]
        )
    )
    lines.extend(
        [
            ") {",
            "    std::size_t fort_internal_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;",
            "    if (fort_internal_index >= fort_internal_total) return;",
        ]
    )
    for dimension in reversed(range(len(region.loops))):
        name = region.loops[dimension].iterator.cpp_name
        lines.append(
            f"    const int {name} = "
            f"{_iteration_value(str(dimension), f'fort_internal_index % fort_internal_extent{dimension}')};"
        )
        lines.append(f"    fort_internal_index /= fort_internal_extent{dimension};")
    lines.extend(_indent([f"{_cpp_type(symbol)} {symbol.cpp_name};" for symbol in region.private_symbols]))
    lines.extend(_sequential_block(_region_body(region), 1, [0], device=True))
    lines.extend(["}", ""])
    return lines


def _cuda_launch(region: ParallelRegion) -> list[str]:
    lines = ["{", f"    // Source-order parallel region {region.id}."]
    lines.extend(_indent(_mapped_snapshots(region)))
    total = " * ".join(f"fort_internal_extent{dimension}" for dimension in range(len(region.loops)))
    lines.extend(
        [
            f"    const std::size_t fort_internal_total = {total};",
            "    if (fort_internal_total > 0) {",
            "        constexpr unsigned int fort_internal_threads = 256;",
            "        const unsigned int fort_internal_blocks = static_cast<unsigned int>(",
            "            (fort_internal_total - 1) / fort_internal_threads + 1);",
        ]
    )
    arguments: list[str] = []
    for argument in _abi_arguments(_region_symbols(region)):
        arguments.append(
            argument.name + "_device" if argument.dimension is None and argument.symbol.rank else argument.name
        )
    for dimension in range(len(region.loops)):
        arguments.extend(
            [
                f"fort_internal_lower{dimension}",
                f"fort_internal_stride{dimension}",
                f"fort_internal_extent{dimension}",
            ]
        )
    arguments.append("fort_internal_total")
    lines.append(f"        {_kernel_name(region)}<<<fort_internal_blocks, fort_internal_threads>>>(")
    lines.extend(
        _indent([argument + ("," if index < len(arguments) - 1 else "") for index, argument in enumerate(arguments)], 3)
    )
    lines.extend(["        );", "        CUCH(cudaGetLastError());", "    }", "}"])
    return lines


def _array_bytes(symbol: Symbol) -> str:
    return f"{symbol.cpp_name}_bytes"


def _cuda_transfer(symbols: tuple[Symbol, ...], *, to_device: bool) -> list[str]:
    if not symbols:
        return []
    direction = "h2d" if to_device else "d2h"
    cuda_direction = "cudaMemcpyHostToDevice" if to_device else "cudaMemcpyDeviceToHost"
    byte_count = " + ".join(_array_bytes(symbol) for symbol in symbols)
    lines = [f"measure_{direction}({byte_count}, [&]() {{"]
    for symbol in symbols:
        host = symbol.cpp_name
        device = f"{host}_device"
        destination, source = (device, host) if to_device else (host, device)
        lines.append(
            f"    if ({_array_bytes(symbol)} > 0) CUCH(cudaMemcpy({destination}, {source}, "
            f"{_array_bytes(symbol)}, {cuda_direction}));"
        )
    lines.append("});")
    return lines


def _generate_cuda(function: FunctionIR, plan: ExecutionPlan, abi: tuple[_AbiArgument, ...], common_header: str) -> str:
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
    for step in plan.steps:
        if hasattr(step, "loops"):
            lines.extend(_cuda_region(step))
    lines.extend(
        [
            'extern "C" void cpp_start_hot() { reset_timing_vectors(); }',
            'extern "C" void cpp_finish_hot() { print_timing_summary(); }',
            "",
            f'extern "C" void cpp_{function.name}(',
        ]
    )
    lines.extend(
        _indent(
            [argument.cpp_declaration + ("," if index < len(abi) - 1 else "") for index, argument in enumerate(abi)]
        )
    )
    lines.append(") {")
    for symbol in arrays:
        dimensions = " * ".join(_dimension_name(symbol, dimension) for dimension in range(1, symbol.rank + 1))
        lines.extend(
            _indent(
                [
                    f"{_cpp_type(symbol)}* {symbol.cpp_name}_device = nullptr;",
                    f"const std::size_t {_array_bytes(symbol)} = sizeof({_cpp_type(symbol)}) * {dimensions};",
                ]
            )
        )
    lines.append("    #ifdef USE_PINNED_MEMORY")
    for symbol in arrays:
        lines.extend(
            _indent(
                [
                    f"if ({_array_bytes(symbol)} > 0 && fort_internal_pinned_ptrs.insert({symbol.cpp_name}).second) {{",
                    f"    CUCH(cudaHostRegister(const_cast<{_cpp_type(symbol)}*>({symbol.cpp_name}), {_array_bytes(symbol)}, cudaHostRegisterPortable));",
                    "}",
                ]
            )
        )
    lines.extend(["    #endif", "    measure_alloc([&]() {"])
    lines.extend(
        _indent(
            [
                f"if ({_array_bytes(symbol)} > 0) CUCH(cudaMalloc(reinterpret_cast<void**>(&{symbol.cpp_name}_device), {_array_bytes(symbol)}));"
                for symbol in arrays
            ],
            2,
        )
    )
    inputs = tuple(symbol for symbol in arrays if symbol.intent != "out")
    outputs = tuple(symbol for symbol in arrays if symbol.intent != "in")
    input_bytes = " + ".join(_array_bytes(symbol) for symbol in inputs) or "0"
    output_bytes = " + ".join(_array_bytes(symbol) for symbol in outputs) or "0"
    lines.extend(["    });", f"    measure_h2d({input_bytes}, [&]() {{"])
    lines.extend(
        _indent(
            [
                f"if ({_array_bytes(symbol)} > 0) CUCH(cudaMemcpy({symbol.cpp_name}_device, {symbol.cpp_name}, {_array_bytes(symbol)}, cudaMemcpyHostToDevice));"
                for symbol in inputs
            ],
            2,
        )
    )
    lines.append("    });")
    lines.extend(_indent([f"{_cpp_type(symbol)} {symbol.cpp_name};" for symbol in _host_symbols(function, plan)]))
    lines.append("    measure_kernel_executions([&]() {")
    device_dirty: set[Symbol] = set()
    for step in plan.steps:
        if hasattr(step, "loops"):
            bounds = [expression for loop in step.loops for expression in (loop.lower, loop.upper)]
            bounds.extend(loop.step for loop in step.loops if not isinstance(loop.step, int))
            bound_reads = {
                node.symbol for expression in bounds for node in walk_expr(expression) if isinstance(node, ArrayAccess)
            }
            downloads = tuple(symbol for symbol in arrays if symbol in device_dirty & bound_reads)
            lines.extend(_indent(_cuda_transfer(downloads, to_device=False), 2))
            device_dirty.difference_update(downloads)
            lines.extend(_indent(_cuda_launch(step), 2))
            device_dirty.update(step.write_symbols)
        else:
            # A partial host write must first preserve every device-produced
            # value elsewhere in the array, even if it does not read that value.
            needed = device_dirty & (set(step.read_symbols) | set(step.write_symbols))
            downloads = tuple(symbol for symbol in arrays if symbol in needed)
            lines.extend(_indent(_cuda_transfer(downloads, to_device=False), 2))
            device_dirty.difference_update(downloads)
            lines.extend(_indent([_assignment(assignment) for assignment in step.assignments], 2))
            uploads = tuple(symbol for symbol in arrays if symbol in step.write_symbols)
            lines.extend(_indent(_cuda_transfer(uploads, to_device=True), 2))
    lines.extend(["    });", "    CUCH(cudaDeviceSynchronize());", f"    measure_d2h({output_bytes}, [&]() {{"])
    lines.extend(
        _indent(
            [
                f"if ({_array_bytes(symbol)} > 0) CUCH(cudaMemcpy({symbol.cpp_name}, {symbol.cpp_name}_device, {_array_bytes(symbol)}, cudaMemcpyDeviceToHost));"
                for symbol in outputs
            ],
            2,
        )
    )
    lines.extend(["    });", "    measure_free([&]() {"])
    lines.extend(
        _indent(
            [
                f"if ({symbol.cpp_name}_device != nullptr) CUCH(cudaFree({symbol.cpp_name}_device));"
                for symbol in arrays
            ],
            2,
        )
    )
    lines.extend(["    });", "}", "}", ""])
    return "\n".join(lines)


def _fortran_list(prefix: str, values: list[str], suffix: str, indentation: int) -> list[str]:
    """Continue every argument separately to stay within free-form line limits."""
    if not values:
        return [" " * indentation + prefix + suffix]
    lines = [" " * indentation + prefix + " &"]
    lines.extend(
        " " * (indentation + 4) + value + (", &" if index < len(values) - 1 else " &")
        for index, value in enumerate(values)
    )
    lines.append(" " * indentation + suffix)
    return lines


def _fortran_declaration(symbol: Symbol, name: str) -> str:
    kind = {ScalarType.INTEGER: "integer", ScalarType.REAL: "real(knd)", ScalarType.REAL32: "real"}[symbol.dtype]
    intent = symbol.intent or "inout"
    shape = f"({', '.join(':' for _ in range(symbol.rank))})" if symbol.rank else ""
    attributes = f", contiguous, intent({intent})" if symbol.rank else f", intent({intent})"
    return f"{kind}{attributes} :: {name}{shape}"


def _fortran_line(declaration: str, indentation: int) -> list[str]:
    line = " " * indentation + declaration
    if len(line) <= 132:
        return [line]
    # A rank-15 assumed shape and a long public dummy name can exceed the
    # free-form limit. A continuation before its name keeps both parts short.
    left, right = line.split("::", 1)
    return [left.rstrip() + " :: &", " " * (indentation + 4) + right.strip()]


def _generate_fortran(function: FunctionIR, abi: tuple[_AbiArgument, ...]) -> str:
    # Keep the public dummy labels for keyword callers. Conversion intrinsics
    # belong in a separate scope where user names cannot hide SIZE/INT/REAL.
    occupied = {symbol.name.lower() for symbol in function.parameters}
    occupied.update({function.name.lower(), function.module.lower(), "start_hot", "finish_hot", "knd"})

    def unique_name(base: str) -> str:
        name = base
        suffix = 0
        while name.lower() in occupied:
            suffix += 1
            name = f"{base}_{suffix}"
        occupied.add(name.lower())
        return name

    kinds = _FortranKinds(
        integer=unique_name("fort_internal_c_integer"),
        real=unique_name("fort_internal_c_real"),
        real32=unique_name("fort_internal_c_real32"),
        extent=unique_name("fort_internal_c_extent"),
    )
    c_entry = unique_name("fort_internal_c_entry")
    bridge = unique_name("fort_internal_bridge")
    c_start = unique_name("fort_internal_c_start")
    c_finish = unique_name("fort_internal_c_finish")
    lines = [
        f"module {function.module}",
        "  use iso_c_binding, only: &",
        f"    {kinds.integer} => c_int, &",
        f"    {kinds.real} => c_double, &",
        f"    {kinds.real32} => c_float, &",
        f"    {kinds.extent} => c_size_t",
        "  implicit none",
        "  private",
        f"  public :: {function.name}, knd, start_hot, finish_hot",
        f"  integer, parameter :: knd = {kinds.real}",
        "",
        "  interface",
    ]
    lines.extend(
        _fortran_list(
            f"subroutine {c_entry}(",
            [argument.name for argument in abi],
            f") bind(C, name='cpp_{function.name}')",
            4,
        )
    )
    lines.append(f"      import :: {kinds.integer}, {kinds.real}, {kinds.real32}, {kinds.extent}")
    for argument in abi:
        lines.extend(_fortran_line(argument.fortran_declaration(kinds), 6))
    lines.extend(
        [
            f"    end subroutine {c_entry}",
            f"    subroutine {c_start}() bind(C, name='cpp_start_hot')",
            f"    end subroutine {c_start}",
            f"    subroutine {c_finish}() bind(C, name='cpp_finish_hot')",
            f"    end subroutine {c_finish}",
            "  end interface",
            "",
            "contains",
            "",
        ]
    )
    public_names = [symbol.name.lower() for symbol in function.parameters]
    private_names = [symbol.cpp_name for symbol in function.parameters]
    lines.extend(_fortran_list(f"subroutine {function.name}(", public_names, ")", 2))
    for symbol in function.parameters:
        lines.extend(_fortran_line(_fortran_declaration(symbol, symbol.name.lower()), 4))
    lines.extend(_fortran_list(f"call {bridge}(", public_names, ")", 4))
    lines.extend([f"  end subroutine {function.name}", ""])
    lines.extend(_fortran_list(f"subroutine {bridge}(", private_names, ")", 2))
    lines.append("    intrinsic :: size, int, real")
    for symbol in function.parameters:
        lines.extend(_fortran_line(_fortran_declaration(symbol, symbol.cpp_name), 4))
    lines.extend(_fortran_list(f"call {c_entry}(", [argument.fortran_call(kinds) for argument in abi], ")", 4))
    lines.extend(
        [
            f"  end subroutine {bridge}",
            "",
            "  subroutine start_hot()",
            f"    call {c_start}()",
            "  end subroutine start_hot",
            "",
            "  subroutine finish_hot()",
            f"    call {c_finish}()",
            "  end subroutine finish_hot",
            "",
            f"end module {function.module}",
            "",
        ]
    )
    return "\n".join(lines)


def generate_sources(
    function: FunctionIR, plan: ExecutionPlan, *, common_header: str = "common_functions.cuh"
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
    abi = _abi_arguments(function.parameters)
    return GeneratedSources(
        cuda=_generate_cuda(function, plan, abi, common_header),
        cpp=_generate_cpp(function, plan, abi, common_header),
        fortran=_generate_fortran(function, abi),
    )
