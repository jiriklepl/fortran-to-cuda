"""First registration from literal facts or authenticated activation geometry."""

from __future__ import annotations

import re
from dataclasses import dataclass

from compiler.frontend.native_sections import RECTANGLE_LIMIT, NativeResourceSections
from compiler.ir import CompilationError
from compiler.scopes.access import _statement_lines, build_native_access


@dataclass(frozen=True)
class RegistrationCode:
    specification: tuple[str, ...]
    prepare: tuple[str, ...]


def expected_descriptor(fact, rank):
    """Validate an external descriptor constraint, without defining payload."""
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


def expected_descriptor_conditions(fact, rank, name):
    """Render inquiries only for an already allocated, contiguous ABI-safe array."""
    expected = expected_descriptor(fact, rank)
    if expected is None:
        return ()

    def literal(value):
        return "(-9223372036854775807_c_int64_t - 1_c_int64_t)" if value == -(2**63) else f"{value}_c_int64_t"

    return tuple(condition for axis, (lower, extent) in enumerate(
        zip(expected["lower_bounds"], expected["extents"], strict=True), 1)
        for condition in (f"lbound({name}, {axis}, kind=c_int64_t) == {literal(lower)}",
                          f"size({name}, {axis}, kind=c_int64_t) == {literal(extent)}"))


def expected_descriptor_public(fact, rank):
    expected = expected_descriptor(fact, rank)
    return {} if expected is None else {
        "expected_descriptor": {key: list(values) for key, values in expected.items()},
        "descriptor_expectation_guard": {
            "checked_at": "original reached descriptor after allocation, contiguity and integer ABI guards; "
                          "before registration, query and borrowed dummy association",
            "reason": "capture expected descriptor mismatch",
            "fallback": "publish and close current owner; original suffix once; no reopening or prefix replay",
            "definition_authority": False}}


def build_registration(fact, rank, handle, prefix, key, layout, *, context="fort_context",
                       status="fort_status", on_error=("return",), parameters=None,
                       logical_lower_bounds=None):
    """Consume compiler-owned source sections only at the original descriptor.

    The caller supplies an already guarded local layout and runs this code only
    while the resource handle is zero. No undefined registration/reset cycle
    occurs. External physical section facts retain their existing interpretation.
    """
    if not re.fullmatch(r"fort_[a-z0-9_]{1,24}", prefix) or not 1 <= rank <= 15:
        raise CompilationError("registration requires a bounded compiler-private name and array rank")
    initialized = fact.get("initialized")
    source = fact.get("source_sections")
    spec, prepare = [], []
    if source is not None:
        if (initialized != "sections" or not isinstance(source, NativeResourceSections)
                or source.rank != rank or source.reads or source.writes or not source.overwrites):
            raise CompilationError("source activation registration requires typed exact definition sections")
        access = build_native_access(source, handle, prefix, context=context, status=status,
                                     on_error=on_error, parameters=parameters,
                                     logical_lower_bounds=logical_lower_bounds, original_layout=layout)
        spec += access.specification
        prepare += access.prepare
        pointer, count = access.access_name + "%overwrites", access.access_name + "%overwrite_count"
    elif initialized == "sections":
        boxes = fact.get("sections")
        if not isinstance(boxes, list) or len(boxes) > RECTANGLE_LIMIT:
            raise CompilationError("registration requires bounded literal physical section facts")
        spec += [f"type(fort_scope_section), target :: {prefix}_sections({max(1, len(boxes))})"]
        for index, box in enumerate(boxes, 1):
            if (not isinstance(box, dict) or set(box) != {"lower", "upper"}
                    or any(not isinstance(box[label], list) or len(box[label]) != rank
                           for label in ("lower", "upper"))
                    or any(type(value) is not int or not 0 <= value < 2**63
                           for label in ("lower", "upper") for value in box[label])
                    or any(lo > hi for lo, hi in zip(box["lower"], box["upper"], strict=True))):
                raise CompilationError("registration requires valid literal physical section coordinates")
            for label in ("lower", "upper"):
                name = f"{prefix}_{index}_{label}"
                spec += [f"integer(c_size_t), target :: {name}({rank})"]
                prepare += [name + " = [" + ", ".join(f"{value}_c_size_t" for value in box[label]) + "]",
                            f"{prefix}_sections({index})%{label} = c_loc({name})"]
        pointer = f"c_loc({prefix}_sections)" if boxes else "c_null_ptr"
        count = f"{len(boxes)}_c_size_t"
    elif initialized in {"whole", "none"}:
        prepare += [f"{status} = fort_scope_register({context}, {key}, &",
                    f"1_c_int64_t, {layout}, {int(initialized == 'whole')}_c_int, {handle})"]
        return RegistrationCode((), tuple(prepare))
    else:
        raise CompilationError("registration requires explicit initial definition coverage")
    prepare += [f"{status} = fort_scope_register_sections({context}, {key}, &",
                f"1_c_int64_t, {layout}, {pointer}, {count}, {handle})"]
    return RegistrationCode(tuple(line for statement in spec for line in _statement_lines(statement)),
                            tuple(line for statement in prepare for line in _statement_lines(statement)))
