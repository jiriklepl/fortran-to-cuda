"""Input validation and numerical checks for the stencil comparison."""

from __future__ import annotations

import math

ATOL = 1e-10


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
