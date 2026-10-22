"""First registration from literal facts or authenticated activation geometry."""

from __future__ import annotations

import re
from dataclasses import dataclass

from compiler.frontend.capture_coverage import expected_descriptor as expected_descriptor
from compiler.frontend.capture_coverage import physical_sections
from compiler.frontend.native_sections import NativeResourceSections
from compiler.ir import CompilationError
from compiler.scopes.access import _statement_lines, build_native_access


@dataclass(frozen=True)
class RegistrationCode:
    specification: tuple[str, ...]
    prepare: tuple[str, ...]


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


def _empty_registration(fact):
    """The same literal NONE policy is used for registration and its preflight."""
    return fact.get("initialized") == "none" and fact.get("source_sections") is None


def fresh_undefined_read_guard(facts, required, parameters, *, context, query_plan,
                              entry_guard, on_error, status="fort_status"):
    """Reject only the already required coarse preflight of a fresh owner.

    The caller has checked the original activation, descriptors and typed
    preparation guards, and authenticates all slots as belonging to this
    lexical owner. Context zero therefore retires even nonzero cached slots;
    no current definitions can survive there. A borrowed live registration or
    an inherited query must retain the ordinary runtime checks.

    This is the exact NONE policy used by ``build_registration`` below, not
    an inference about the original program's values. No reached admission or
    unit effect precedes its required-whole preflight. Empty roots retain that
    preflight because an empty whole read is valid.
    """
    if (not required or any(fact.get("requires_registered_handle") for fact in facts.values())):
        return ()
    lines = ["! Fresh owner required-whole read has no initial definitions.",
             f"if ({context} == 0) then", f"if (.not. {query_plan}) then"]
    for root in sorted(required):
        fact = facts[root]
        # The normal registration is still built and validates these facts.
        # Its nonpristine branch always registers NONE, including section
        # facts; a pristine literal NONE registration is empty as well.
        condition = ".true." if _empty_registration(fact) else f".not. {entry_guard}"
        lines += [f"if ({condition}) then",
                  f"if (all(shape({parameters[root]}, kind=c_size_t) > 0_c_size_t)) then",
                  f"{status} = FORT_SCOPE_UNINITIALIZED", *on_error, "endif", "endif"]
    return tuple([*lines, "endif", "endif"])


def build_registration(fact, rank, handle, prefix, key, layout, *, context="fort_context",
                       status="fort_status", on_error=("return",), parameters=None,
                       logical_lower_bounds=None, entry_guard=None):
    """Consume compiler-owned source sections only at the original descriptor.

    The caller supplies an already guarded local layout and runs this code only
    while the resource handle is zero. No undefined registration/reset cycle
    occurs. External physical section facts retain their existing interpretation.
    """
    if entry_guard is not None:
        initial = build_registration(fact, rank, handle, prefix, key, layout, context=context,
            status=status, on_error=on_error, parameters=parameters, logical_lower_bounds=logical_lower_bounds)
        # A closed epoch has no initialized coverage to inherit. Only a
        # separately authenticated reached read may subsequently admit values.
        empty = build_registration({"initialized": "none"}, rank, handle, prefix, key, layout,
            context=context, status=status, on_error=on_error)
        return RegistrationCode(initial.specification,
            (f"if ({entry_guard}) then", *initial.prepare, "else", *empty.prepare, "endif"))
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
        boxes = physical_sections(fact, rank)
        spec += [f"type(fort_scope_section), target :: {prefix}_sections({max(1, len(boxes))})"]
        for index, box in enumerate(boxes, 1):
            for label in ("lower", "upper"):
                name = f"{prefix}_{index}_{label}"
                spec += [f"integer(c_size_t), target :: {name}({rank})"]
                prepare += [name + " = [" + ", ".join(f"{value}_c_size_t" for value in box[label]) + "]",
                            f"{prefix}_sections({index})%{label} = c_loc({name})"]
        pointer = f"c_loc({prefix}_sections)" if boxes else "c_null_ptr"
        count = f"{len(boxes)}_c_size_t"
    elif initialized in {"whole", "none"}:
        prepare += [f"{status} = fort_scope_register({context}, {key}, &",
                    f"1_c_int64_t, {layout}, {int(not _empty_registration(fact))}_c_int, {handle})"]
        return RegistrationCode((), tuple(prepare))
    else:
        raise CompilationError("registration requires explicit initial definition coverage")
    prepare += [f"{status} = fort_scope_register_sections({context}, {key}, &",
                f"1_c_int64_t, {layout}, {pointer}, {count}, {handle})"]
    return RegistrationCode(tuple(line for statement in spec for line in _statement_lines(statement)),
                            tuple(line for statement in prepare for line in _statement_lines(statement)))
