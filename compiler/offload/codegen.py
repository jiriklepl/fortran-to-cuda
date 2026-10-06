"""Shared profile constants for generated selection and hybrid workers."""

import json

from compiler.emission.common.abi import dimension_name
from compiler.ir import Binary, IntrinsicCall, Literal, Reference, Size, Unary
from compiler.ir.integers import integer_literal
from compiler.offload.profile import ProfileError, compiler_identity


def query_expression(expression):
    """Wide, total arithmetic for analyzer-approved physical/bound expressions."""
    if isinstance(expression, Literal):
        return f"({integer_literal(expression.value)}.0L)"
    if isinstance(expression, Reference):
        return f"static_cast<long double>({expression.symbol.cpp_name})"
    if isinstance(expression, Size):
        return f"static_cast<long double>({dimension_name(expression.symbol, expression.dimension)})"
    if isinstance(expression, Unary):
        return f"({expression.operator}{query_expression(expression.operand)})"
    if isinstance(expression, Binary):
        value = f"({query_expression(expression.left)} {expression.operator} {query_expression(expression.right)})"
        return f"std::trunc({value})" if expression.operator == "/" else value
    if isinstance(expression, IntrinsicCall) and expression.name.lower() in {"min", "max"}:
        return f"std::{expression.name.lower()}({{{', '.join(query_expression(v) for v in expression.arguments)}}})"
    raise ValueError(f"Unsafe cost expression: {expression!r}")


def profile_expression(profile: dict | None) -> str:
    if profile is None:
        return "offload::Profile{}"
    try:
        identity = compiler_identity(profile)
    except ProfileError:
        return "offload::Profile{}"
    rates = profile["rates"]
    fields = {
        "valid": "true",
        "threads": str(profile["cpu_threads"]),
        "precision": str(profile["precision_bits"]),
        "uuid": json.dumps(profile["hardware"]["gpu_uuid"]),
        "cc": json.dumps(str(profile["hardware"]["compute_capability"])),
        **{key: json.dumps(value) if isinstance(value, str) else str(value)
           for key, value in identity.items()},
        "launch_seconds": rates["launch_latency_seconds"],
        "h2d_latency": rates["h2d_pageable"]["latency_seconds"],
        "h2d_bandwidth": rates["h2d_pageable"]["bandwidth_bytes_per_second"],
        "d2h_latency": rates["d2h_pageable"]["latency_seconds"],
        "d2h_bandwidth": rates["d2h_pageable"]["bandwidth_bytes_per_second"],
        "pinned_h2d_latency": rates["h2d_pinned"]["latency_seconds"],
        "pinned_h2d_bandwidth": rates["h2d_pinned"]["bandwidth_bytes_per_second"],
        "pinned_d2h_latency": rates["d2h_pinned"]["latency_seconds"],
        "pinned_d2h_bandwidth": rates["d2h_pinned"]["bandwidth_bytes_per_second"],
        "pack_bandwidth": rates["pack_bytes_per_second"],
        "unpack_bandwidth": rates["unpack_bytes_per_second"],
        "cpu_flops": rates["cpu_flops_per_second"],
        "cpu_bandwidth": rates["cpu_memory_bytes_per_second"],
        "cpu_worker_flops": rates["cpu_worker_flops_per_second"],
        "cpu_worker_bandwidth": rates["cpu_worker_memory_bytes_per_second"],
        "gpu_flops": rates["gpu_flops_per_second"],
        "gpu_bandwidth": rates["gpu_memory_bytes_per_second"],
        "pump_seconds": rates["pump_latency_seconds"],
    }
    assignments = [f"p.{key} = {value if isinstance(value, str) else repr(float(value))};"
                   for key, value in fields.items()]
    return "[]() { offload::Profile p; " + " ".join(assignments) + " return p; }()"
