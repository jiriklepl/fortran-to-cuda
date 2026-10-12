"""Literal capture contracts, distinct from source-proven definitions.

Physical rectangles are zero-based half-open offsets in the original array.
Reached source proofs use logical inclusive coordinates. Converting between
them requires an explicit descriptor contract, checked again at registration;
neither allocation presence nor that contract establishes payload definitions.
"""

from compiler.frontend.native_sections import (
    RECTANGLE_LIMIT,
    NativeAxis,
    NativeBound,
    NativeRectangle,
    NativeResourceSections,
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


def reached_sections(binding, fact):
    """Translate asserted entry coverage; retain the original descriptor guard.

    Absolute logical coordinates avoid inventing bound-expression reads before
    original allocation guards. The assertion is never expanded to a bounding
    hull or credited as a source overwrite. Empty rectangles define no values.
    """
    boxes = physical_sections(fact, binding.rank)
    expected = expected_descriptor(fact, binding.rank)
    if expected is None:
        raise CompilationError("reached physical entry sections require an expected original descriptor")
    rectangles = []
    for box in boxes:
        if any(upper > extent for upper, extent in zip(box["upper"], expected["extents"], strict=True)):
            raise CompilationError("initialized physical capture section exceeds its expected descriptor")
        if any(lo == hi for lo, hi in zip(box["lower"], box["upper"], strict=True)):
            continue
        axes = []
        for origin, lower, upper in zip(expected["lower_bounds"], box["lower"], box["upper"], strict=True):
            lower, upper = origin + lower, origin + upper - 1
            if not -(2**31) <= lower <= upper < 2**31:
                raise CompilationError("reached physical definition exceeds the source INTEGER ABI")
            axes.append(NativeAxis(NativeBound("literal", lower), NativeBound("literal", upper)))
        rectangles.append(NativeRectangle(tuple(axes), binding.name))
    return NativeResourceSections(binding.root, binding.rank, tuple(expected["lower_bounds"]),
                                  (), (), tuple(rectangles))
