"""Entry-specific ABI glue over the reusable owned-storage runtime."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from compiler.emission.c.declarations import cpp_declaration
from compiler.emission.common.abi import AbiArgument, abi_arguments, dimension_name
from compiler.emission.common.c_family import cpp_type, indent
from compiler.ir import ExecutionPlan, FunctionIR, Symbol
from compiler.memory import MemoryPlan, plan_memory


@dataclass(frozen=True)
class SessionNames:
    workspace: str
    create: str
    run: str
    update_device: str
    update_host: str
    destroy: str


def session_names(function: FunctionIR) -> SessionNames:
    occupied = {symbol.name.lower() for symbol in function.parameters}
    occupied.update(symbol.cpp_name.lower() for symbol in function.parameters)
    occupied.update((function.name.lower(), function.module.lower(), "start_hot", "finish_hot", "knd"))
    names = []
    for suffix in ("workspace", "create", "run", "update_device", "update_host", "destroy"):
        original = f"{function.name.lower()}_{suffix}"
        name = original
        salt = 0
        while len(name) > 63 or name in occupied:
            digest = sha256(f"{original}:{salt}".encode()).hexdigest()[:10]
            name = f"{function.name.lower()[:38]}_{digest}_{suffix}"
            salt += 1
        names.append(name)
        occupied.add(name)
    return SessionNames(*names)


def _signature(name: str, parameters: list[str], result: str = "void") -> list[str]:
    return [
        f'extern "C" {result} {name}(',
        *indent([p + ("," if i + 1 < len(parameters) else "") for i, p in enumerate(parameters)]),
        ") {",
    ]


def buffer_name(symbol: Symbol) -> str:
    return f"fort_internal_state.{symbol.cpp_name}"


def array_dimensions(symbol: Symbol) -> str:
    return "{" + ", ".join(dimension_name(symbol, axis) for axis in range(1, symbol.rank + 1)) + "}"


def workspace_registry_name(function: FunctionIR) -> str:
    return "fort_internal_" + session_names(function).workspace + "_registry"


def session_definitions(
    function: FunctionIR,
    abi: tuple[AbiArgument, ...],
    *,
    cuda_run: list[str] | None = None,
    memory: MemoryPlan,
) -> list[str]:
    """Emit inside generated_kernels, after kernels or the ordinary CPU entry."""
    names = session_names(function)
    arrays = next(operation.symbols for operation in memory.create if operation.kind == "acquire")
    inputs = next(operation.symbols for operation in memory.create if operation.kind == "device")
    scalars = tuple(symbol for symbol in function.parameters if not symbol.rank)
    array_abi = abi_arguments(arrays)
    dimensions = tuple(argument for argument in array_abi if argument.dimension is not None)
    state_type = "fort_internal_" + names.workspace + "_state"
    registry = workspace_registry_name(function)
    lines = [f"struct {state_type} {{"]
    lines.extend(indent([f"storage::Buffer<{cpp_type(symbol)}> {symbol.cpp_name};" for symbol in arrays]))
    ctor_args = ", ".join(cpp_declaration(argument) for argument in dimensions)
    ctor_init = ", ".join(f"{symbol.cpp_name}({array_dimensions(symbol)})" for symbol in arrays)
    lines.append(f"    {state_type}({ctor_args})" + (f" : {ctor_init}" if arrays else "") + " {}")
    lines.extend(["};", f"static storage::Registry<{state_type}> {registry};", ""])
    lines.extend(
        _signature("cpp_" + names.create, [cpp_declaration(argument) for argument in array_abi], "std::int64_t")
    )
    lines.append(f"    const auto fort_internal_token = {registry}.create({', '.join(a.name for a in dimensions)});")
    lines.append(f"    auto& fort_internal_state = {registry}.get(fort_internal_token);")
    for symbol in inputs:
        lines.append(f"    {buffer_name(symbol)}.update_device({symbol.cpp_name}, {array_dimensions(symbol)});")
    lines.extend(["    return fort_internal_token;", "}", ""])
    lines.extend(
        _signature(
            "cpp_" + names.run,
            ["std::int64_t fort_internal_token", *[cpp_declaration(AbiArgument(s)) for s in scalars]],
        )
    )
    lines.append(f"    auto& fort_internal_state = {registry}.get(fort_internal_token);")
    for symbol in arrays:
        pointer = f"{cpp_type(symbol)}* {symbol.cpp_name}"
        lines.append(
            f"    {pointer} = " + ("nullptr;" if cuda_run is not None else f"{buffer_name(symbol)}.host_data();")
        )
        if cuda_run is not None:
            lines.append(f"    {pointer}_device = nullptr;")
        for axis in range(1, symbol.rank + 1):
            lines.append(
                f"    const std::size_t {dimension_name(symbol, axis)} = {buffer_name(symbol)}.extent({axis - 1});"
            )
    if cuda_run is None:
        lines.append(f"    cpp_{function.name}({', '.join(argument.name for argument in abi)});")
        lines.extend(f"    {buffer_name(symbol)}.host_written();" for symbol in arrays if symbol.intent != "in")
    else:
        lines.extend(indent(cuda_run))
    lines.extend(["}", ""])
    # Every explicit update, including an empty optional update, validates and waits.
    lines.extend(_signature(f"cpp_{names.workspace}_validate", ["std::int64_t fort_internal_token"]))
    lines.extend([f"    {registry}.get(fort_internal_token);", "    storage::synchronize();", "}", ""])
    for direction in ("device", "host"):
        for symbol in arrays:
            arguments = [
                "std::int64_t fort_internal_token",
                f"{'const ' if direction == 'device' else ''}{cpp_type(symbol)}* {symbol.cpp_name}",
            ]
            arguments.extend(cpp_declaration(AbiArgument(symbol, axis)) for axis in range(1, symbol.rank + 1))
            lines.extend(_signature(f"cpp_{getattr(names, 'update_' + direction)}_{symbol.id}", arguments))
            lines.append(f"    auto& fort_internal_state = {registry}.get(fort_internal_token);")
            lines.append(
                f"    {buffer_name(symbol)}.update_{direction}({symbol.cpp_name}, {array_dimensions(symbol)});"
            )
            lines.extend(["    storage::synchronize();", "}", ""])
    lines.extend(_signature("cpp_" + names.destroy, ["std::int64_t fort_internal_token"]))
    lines.extend([f"    {registry}.destroy(fort_internal_token);", "}", ""])
    return lines


def append_cpu_sessions(function: FunctionIR, abi: tuple[AbiArgument, ...], *, memory: MemoryPlan | None = None) -> str:
    memory = memory or plan_memory(ExecutionPlan(()), function.parameters)
    return "\n".join(["", "namespace generated_kernels {", *session_definitions(function, abi, memory=memory), "}", ""])
