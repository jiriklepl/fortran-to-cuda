"""Render the C ABI used by both C++ and CUDA entry points."""

from compiler.emission.common.abi import AbiArgument
from compiler.emission.common.c_family import cpp_type


def cpp_declaration(argument: AbiArgument, *, array_views: bool = False) -> str:
    if argument.dimension is not None:
        return f"std::size_t {argument.name}"
    if argument.symbol.rank:
        const = "const " if argument.symbol.intent == "in" else ""
        if array_views:
            return f"fort_scoped::RootView<{const}{cpp_type(argument.symbol)}, {argument.symbol.rank}> {argument.name}"
        return f"{const}{cpp_type(argument.symbol)}* __restrict__ {argument.name}"
    return f"{cpp_type(argument.symbol)} {argument.name}"
