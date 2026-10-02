"""Entry-specific ABI glue over explicit lifecycle operations and owned state."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from compiler.emission.c.declarations import cpp_declaration
from compiler.emission.common.abi import AbiArgument, abi_arguments, dimension_name
from compiler.emission.common.c_family import cpp_type, indent
from compiler.emission.common.memory import render_memory
from compiler.ir import FunctionIR
from compiler.memory import MemoryOperation, MemoryPlan, validate_memory


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


def _signature(
    name: str, parameters: list[str], result: str = "void", *, linkage: str = 'extern "C"', profiled: bool = False
) -> list[str]:
    return [
        f"{linkage} {result} {name}(",
        *indent([p + ("," if i + 1 < len(parameters) else "") for i, p in enumerate(parameters)]),
        ") {",
        *(["    timing::ProfiledCallGuard fort_internal_profile;"] if profiled else []),
    ]


def workspace_registry_name(function: FunctionIR) -> str:
    return "fort_internal_" + session_names(function).workspace + "_registry"


def workspace_state_name(function: FunctionIR) -> str:
    return "fort_internal_" + session_names(function).workspace + "_state"


def lifecycle_name(function: FunctionIR, phase: str) -> str:
    return "fort_internal_" + session_names(function).workspace + "_" + phase


def execution_name(function: FunctionIR) -> str:
    return lifecycle_name(function, "run")


def session_definitions(function: FunctionIR, *, run_body: list[str], device: bool, memory: MemoryPlan) -> list[str]:
    """Render state helpers and exported sessions after target kernel definitions."""
    validate_memory(memory, function.parameters)
    names = session_names(function)
    arrays = tuple(symbol for symbol in function.parameters if symbol.rank)
    scalars = tuple(symbol for symbol in function.parameters if not symbol.rank)
    array_abi = abi_arguments(arrays)
    dimensions = tuple(argument for argument in array_abi if argument.dimension is not None)
    state_type = workspace_state_name(function)
    registry = workspace_registry_name(function)
    lines = [f"struct {state_type} {{"]
    lines.extend(indent([f"storage::BufferSlot<{cpp_type(symbol)}> {symbol.cpp_name};" for symbol in arrays]))
    lines.extend(indent([f"const std::size_t {argument.name};" for argument in dimensions]))
    ctor_args = ", ".join(cpp_declaration(argument) for argument in dimensions)
    ctor_init = ", ".join(f"{argument.name}({argument.name})" for argument in dimensions)
    lines.append(f"    {state_type}({ctor_args})" + (f" : {ctor_init}" if dimensions else "") + " {}")
    lines.extend(["};", f"static storage::Registry<{state_type}> {registry};", ""])
    state_argument = f"{state_type}& fort_internal_state"
    for phase, operations in (("create", memory.create), ("retrieve", memory.retrieve), ("destroy", memory.destroy)):
        arguments = [state_argument]
        if phase != "destroy":
            arguments.extend(cpp_declaration(argument) for argument in array_abi)
        lines.extend(_signature(lifecycle_name(function, phase), arguments, linkage="static"))
        lines.extend(indent(render_memory(operations, device=device)))
        lines.extend(["}", ""])
    lines.extend(
        _signature(
            execution_name(function),
            [state_argument, *[cpp_declaration(AbiArgument(s)) for s in scalars]],
            linkage="static",
        )
    )
    for symbol in arrays:
        lines.append(f"    {cpp_type(symbol)}* {symbol.cpp_name} = nullptr;")
        if device:
            lines.append(f"    {cpp_type(symbol)}* {symbol.cpp_name}_device = nullptr;")
        for axis in range(1, symbol.rank + 1):
            name = dimension_name(symbol, axis)
            lines.append(f"    const std::size_t {name} = fort_internal_state.{name};")
    lines.extend(indent(run_body))
    lines.extend(["}", ""])
    lines.extend(
        _signature(
            "cpp_" + names.create,
            [cpp_declaration(argument) for argument in array_abi],
            "std::int64_t",
            profiled=device,
        )
    )
    lines.append(f"    const auto fort_internal_token = {registry}.create({', '.join(a.name for a in dimensions)});")
    lines.append(f"    auto& fort_internal_state = {registry}.get(fort_internal_token);")
    arguments = ", ".join(["fort_internal_state", *[argument.name for argument in array_abi]])
    lines.extend(
        [f"    {lifecycle_name(function, 'create')}({arguments});", "    return fort_internal_token;", "}", ""]
    )
    lines.extend(
        _signature(
            "cpp_" + names.run,
            ["std::int64_t fort_internal_token", *[cpp_declaration(AbiArgument(s)) for s in scalars]],
            profiled=device,
        )
    )
    lines.append(f"    auto& fort_internal_state = {registry}.get(fort_internal_token);")
    arguments = ", ".join(["fort_internal_state", *[symbol.cpp_name for symbol in scalars]])
    lines.extend([f"    {execution_name(function)}({arguments});", "}", ""])
    lines.extend(_signature(f"cpp_{names.workspace}_validate", ["std::int64_t fort_internal_token"], profiled=device))
    lines.append(f"    {registry}.get(fort_internal_token);")
    lines.extend(indent(render_memory((MemoryOperation("sync"),), device=device)))
    lines.extend(["}", ""])
    for direction, kind in (("device", "upload"), ("host", "download")):
        for symbol in arrays:
            arguments = [
                "std::int64_t fort_internal_token",
                f"{'const ' if direction == 'device' else ''}{cpp_type(symbol)}* {symbol.cpp_name}",
            ]
            arguments.extend(cpp_declaration(AbiArgument(symbol, axis)) for axis in range(1, symbol.rank + 1))
            lines.extend(
                _signature(f"cpp_{getattr(names, 'update_' + direction)}_{symbol.id}", arguments, profiled=device)
            )
            lines.append(f"    auto& fort_internal_state = {registry}.get(fort_internal_token);")
            lines.extend(
                indent(render_memory((MemoryOperation(kind, (symbol,)), MemoryOperation("sync")), device=device))
            )
            lines.extend(["}", ""])
    lines.extend(_signature("cpp_" + names.destroy, ["std::int64_t fort_internal_token"], profiled=device))
    lines.extend(
        [
            "    if (!fort_internal_token) return;",
            f"    auto& fort_internal_state = {registry}.get(fort_internal_token);",
            f"    {lifecycle_name(function, 'destroy')}(fort_internal_state);",
            f"    {registry}.destroy(fort_internal_token);",
            "}",
            "",
        ]
    )
    return lines
