"""Input validation and numerical checks for the stencil comparison."""

from __future__ import annotations

import argparse
import math
import re

ATOL = 1e-10


def grid(text: str) -> tuple[int, int, int]:
    try:
        dimensions = tuple(int(value) for value in re.split("[xX,]", text))
    except ValueError as error:
        raise argparse.ArgumentTypeError("grid must be NXxNYxNZ") from error
    if len(dimensions) != 3 or min(dimensions) < 1:
        raise argparse.ArgumentTypeError("grid must contain three positive dimensions")
    if math.prod(value + 2 for value in dimensions) > 2**31 - 1:
        raise argparse.ArgumentTypeError("grid exceeds signed 32-bit array indexing")
    return dimensions


def positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def nonnegative(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return value


def values(text: str, expected: int) -> list[float]:
    result = [float(token) for token in text.split()]
    if len(result) != expected:
        raise ValueError(f"expected {expected} output values, got {len(result)}")
    if not all(math.isfinite(value) for value in result):
        raise ValueError("output contains a non-finite value")
    return result


def compare_values(reference: list[float], actual: list[float]) -> float:
    if len(reference) != len(actual) or not reference:
        raise ValueError("comparison requires equal, nonempty output arrays")
    if not all(math.isfinite(value) for value in (*reference, *actual)):
        raise ValueError("comparison contains a non-finite value")
    difference = max(abs(left - right) for left, right in zip(reference, actual, strict=True))
    if difference > ATOL:
        raise ValueError(f"maximum absolute error {difference:.17g} exceeds {ATOL:g}")
    return difference


def timing_values(output: str) -> tuple[float, float]:
    fields: dict[str, float] = {}
    for line in output.splitlines():
        if not line.startswith(("total_ms ", "checksum ")):
            continue
        key, value = line.split()
        if key in fields:
            raise ValueError(f"duplicate timing field {key}")
        fields[key] = float(value)
    if set(fields) != {"total_ms", "checksum"}:
        raise ValueError(f"missing timing or checksum field: {output}")
    duration, checksum = fields["total_ms"], fields["checksum"]
    if not math.isfinite(duration) or duration <= 0 or not math.isfinite(checksum):
        raise ValueError(f"invalid timing output: {output}")
    return duration, checksum
