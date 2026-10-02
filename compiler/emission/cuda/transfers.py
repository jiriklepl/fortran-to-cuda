"""Render timed host/device transfers for arrays shared between plan steps."""

from compiler.ir import Symbol


def array_bytes(symbol: Symbol) -> str:
    return f"{symbol.cpp_name}_bytes"


def generate_transfer(symbols: tuple[Symbol, ...], *, to_device: bool) -> list[str]:
    if not symbols:
        return []
    direction = "h2d" if to_device else "d2h"
    cuda_direction = "cudaMemcpyHostToDevice" if to_device else "cudaMemcpyDeviceToHost"
    byte_count = " + ".join(array_bytes(symbol) for symbol in symbols)
    lines = [f"measure_{direction}({byte_count}, [&]() {{"]
    for symbol in symbols:
        host = symbol.cpp_name
        device = f"{host}_device"
        destination, source = (device, host) if to_device else (host, device)
        lines.append(
            f"    if ({array_bytes(symbol)} > 0) CUCH(cudaMemcpy({destination}, {source}, "
            f"{array_bytes(symbol)}, {cuda_direction}));"
        )
    lines.append("});")
    return lines
