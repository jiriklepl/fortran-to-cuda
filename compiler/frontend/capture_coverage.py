"""Literal capture contracts, distinct from source-proven definitions.

Physical rectangles are zero-based half-open offsets in the original array.
Reached source proofs use logical inclusive coordinates. Converting between
them requires an explicit descriptor contract, checked again at registration;
neither allocation presence nor that contract establishes payload definitions.
"""

from compiler.frontend.native_sections import (
    RECTANGLE_LIMIT,
)
from compiler.ir import CompilationError


def expected_descriptor(fact, rank):
    """Validate an external descriptor constraint without reading an array."""
    if "expected_descriptor" not in fact:
        return None
    expected = fact["expected_descriptor"]
    if (not 1 <= rank <= 15 or not isinstance(expected, dict)
            or set(expected) != {"lower_bounds", "extents"}
            or any(not isinstance(expected[key], list) or len(expected[key]) != rank
                   for key in ("lower_bounds", "extents"))
            or any(type(value) is not int or not -(2**63) <= value < 2**63
                   for value in expected["lower_bounds"])
            or any(type(value) is not int or not 0 <= value < 2**63 for value in expected["extents"])):
        raise CompilationError("expected capture descriptor requires rank-matched signed64 lower bounds "
                               "and nonnegative signed64 extents")
    return expected


def physical_sections(fact, rank):
    """Validate the existing literal physical-section interface once."""
    boxes = fact.get("sections")
    if not isinstance(boxes, list) or len(boxes) > RECTANGLE_LIMIT:
        raise CompilationError("initialized capture sections exceed the bounded interface")
    for box in boxes:
        if not isinstance(box, dict) or set(box) != {"lower", "upper"}:
            raise CompilationError("initialized capture section requires lower/upper physical coordinates")
        lower, upper = box["lower"], box["upper"]
        if (not isinstance(lower, list) or not isinstance(upper, list)
                or len(lower) != rank or len(upper) != rank
                or any(type(value) is not int or not 0 <= value < 2**63 for value in [*lower, *upper])
                or any(lo > hi for lo, hi in zip(lower, upper, strict=True))):
            raise CompilationError("invalid initialized physical capture section")
    return boxes
