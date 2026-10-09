"""One checked physical access description for native queries and execution."""

from __future__ import annotations

import re
from dataclasses import dataclass

from compiler.frontend.native_sections import RECTANGLE_LIMIT, NativeResourceSections, NativeSections
from compiler.ir import CompilationError
from compiler.ir.integers import INTEGER_MAX, INTEGER_MIN


@dataclass(frozen=True)
class NativeAccessCode:
    specification: tuple[str, ...]
    prepare: tuple[str, ...]
    access_name: str
    handle: str


def _statement_lines(statement):
    """Continue generated statements at whitespace outside quoted literals."""
    lines = []
    while len(statement) > 112:
        quote, boundary = None, None
        for index, character in enumerate(statement[:112]):
            if character in {"'", '"'}:
                quote = None if quote == character else character if quote is None else quote
            elif character == " " and quote is None and index > 3:
                boundary = index
        if boundary is None:
            raise CompilationError("native access statement contains an overlong indivisible token")
        lines.append(statement[:boundary].rstrip() + " &")
        statement = "  & " + statement[boundary+1:]
    return [*lines, statement]


def build_native_access(resource: NativeResourceSections, handle: str, prefix: str, *,
                        context="fort_context", status="fort_status", on_error=("return",), view=None):
    """Map original callee coordinates through the registered full layout.

    The returned TARGET arrays must remain in the caller's specification until
    host_end (plan_add instead copies them). This preparation has no payload
    access and does not mutate runtime coherence. A query must run it before any
    numerical work; later execution may never replay the original span on error.
    """
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,29}", prefix):
        raise CompilationError("native access prefix must be a short generated Fortran identifier")
    if resource.rank < 1 or len(resource.lower_bounds) != resource.rank:
        raise CompilationError("native access requires original array rank and logical lower bounds")
    rank = resource.rank
    access, layout, extents = (prefix + suffix for suffix in ("_access", "_layout", "_extents"))
    origin, lower, upper, empty = (prefix + suffix for suffix in ("_origin", "_lower", "_upper", "_empty"))
    layout_type = "fort_scope_view_layout_v1" if view is not None else "fort_scope_layout"
    specification = [f"type(fort_scope_access) :: {access}", f"type({layout_type}) :: {layout}",
                     f"integer(c_size_t), pointer :: {extents}(:)",
                     f"integer(c_int64_t) :: {origin}({rank}), {lower}({rank}), {upper}({rank})",
                     f"logical :: {empty}"]
    prepare = []

    def require(condition):
        prepare.extend([f"if ({condition}) then", f"{status} = FORT_SCOPE_BOUNDARY", *on_error, "endif"])

    if view is None:
        prepare += [f"{status} = fort_scope_layout_get({context}, {handle}, {layout})",
                    f"if ({status} /= FORT_SCOPE_OK) then", *on_error, "endif"]
        require(f"{layout}%rank /= {rank} .or. .not. c_associated({layout}%extents)")
    else:
        physical_origin = prefix + "_physical_origin"
        specification += [f"integer(c_size_t), pointer :: {physical_origin}(:)"]
        # The view and the eventual host/query hook must name the same root.
        # view_get establishes generation, type, bounds and full-root pitches
        # without touching numerical data or changing coherence.
        require(f"{view}%buffer /= {handle}")
        prepare += [f"{status} = fort_scope_view_get_v1({context}, {view}, {layout})",
                    f"if ({status} /= FORT_SCOPE_OK) then", *on_error, "endif"]
        require(f"{layout}%root%rank /= {rank} .or. .not. c_associated({layout}%extents) .or. "
                f".not. c_associated({layout}%origins)")
        prepare += [f"call c_f_pointer({layout}%origins, {physical_origin}, [{rank}])"]
    prepare += [f"call c_f_pointer({layout}%extents, {extents}, [{rank}])"]
    require(f"any({extents} < 0_c_size_t)")
    for axis, declared in enumerate(resource.lower_bounds, 1):
        if not INTEGER_MIN <= declared <= INTEGER_MAX:
            raise CompilationError("native access declared lower bound exceeds default INTEGER")
        # Empty dimensions have LBOUND=1 even for a negative dummy declaration.
        prepare += [f"{origin}({axis}) = {declared}_c_int64_t",
                    f"if ({extents}({axis}) == 0_c_size_t) {origin}({axis}) = 1_c_int64_t"]
    prepare += [f"{access} = fort_scope_access()"]

    def value(bound, axis, *, default):
        if bound is None:
            return default
        if bound.kind == "literal":
            if bound.value is None or not INTEGER_MIN <= bound.value <= INTEGER_MAX:
                raise CompilationError("native access literal exceeds default INTEGER")
            return f"{bound.value}_c_int64_t"
        if bound.dimension != axis or bound.kind not in {"lbound", "ubound", "size"}:
            raise CompilationError("native access inquiry is not a checked same-array dimension")
        require(f"{extents}({axis}) > {INTEGER_MAX}_c_size_t")
        if bound.kind == "size":
            if resource.lower_bounds[axis-1] != 1:
                raise CompilationError("native access SIZE bounds require declared lower bound one")
            return f"int({extents}({axis}), c_int64_t)"
        if bound.kind == "lbound":
            return f"{origin}({axis})"
        # Both operands fit default INTEGER, so this wider sum cannot overflow.
        expression = f"({origin}({axis}) + int({extents}({axis}), c_int64_t) - 1_c_int64_t)"
        require(f"{expression} < {INTEGER_MIN}_c_int64_t .or. {expression} > {INTEGER_MAX}_c_int64_t")
        return expression

    for label in ("reads", "writes", "overwrites"):
        boxes = getattr(resource, label)
        if not boxes:
            continue
        if len(boxes) > RECTANGLE_LIMIT:
            raise CompilationError("native access rectangle budget exceeded")
        suffix = {"reads": "r", "writes": "w", "overwrites": "o"}[label]
        sections, lows, highs, count = (prefix + "_" + suffix + tail
                                       for tail in ("_sections", "_lows", "_highs", "_count"))
        specification += [f"type(fort_scope_section), target :: {sections}({len(boxes)})",
                          f"integer(c_size_t), target :: {lows}({rank},{len(boxes)}), {highs}({rank},{len(boxes)})",
                          f"integer(c_size_t) :: {count}"]
        prepare += [f"{count} = 0_c_size_t"]
        for box in boxes:
            if len(box.axes) != rank:
                raise CompilationError("native access rectangle rank differs from its resource")
            prepare += [f"{empty} = .false."]
            for axis, specification_axis in enumerate(box.axes, 1):
                if specification_axis.point and (specification_axis.lower is None or specification_axis.upper is None):
                    raise CompilationError("native access point requires an exact literal or inquiry")
                lo, hi = f"{lower}({axis})", f"{upper}({axis})"
                if specification_axis.lower is None and specification_axis.upper is None and not specification_axis.point:
                    prepare += [f"{lo} = 0_c_int64_t", f"{hi} = int({extents}({axis}), c_int64_t)"]
                else:
                    logical_lo = value(specification_axis.lower, axis, default=f"{origin}({axis})")
                    logical_hi = value(specification_axis.upper, axis,
                                       default=f"({origin}({axis}) + int({extents}({axis}), c_int64_t) - 1_c_int64_t)")
                    # Omitted upper ends map directly to extent; no artificial
                    # default-INTEGER UBOUND or unchecked wider addition occurs.
                    prepare += [f"{lo} = {logical_lo} - {origin}({axis})"]
                    if specification_axis.upper is None and not specification_axis.point:
                        prepare += [f"{hi} = int({extents}({axis}), c_int64_t)"]
                    else:
                        prepare += [f"{hi} = {logical_hi} - {origin}({axis}) + 1_c_int64_t"]
                # Check every scalar/nonempty axis independently. An empty
                # range must not hide another axis's invalid point coordinate.
                prepare += [f"if ({hi} <= {lo}) then"]
                if specification_axis.point:
                    prepare += [f"{status} = FORT_SCOPE_BOUNDARY", *on_error]
                else:
                    prepare += [f"{empty} = .true."]
                prepare += ["else"]
                require(f"{lo} < 0_c_int64_t .or. {hi} > int({extents}({axis}), c_int64_t)")
                prepare += ["endif"]
            prepare += [f"if (.not. {empty}) then", f"{count} = {count} + 1_c_size_t"]
            for axis in range(1, rank+1):
                offset = f" + {physical_origin}({axis})" if view is not None else ""
                # Local endpoints were checked against the child extent. The
                # validated root origin plus either endpoint stays within the
                # canonical allocation, including noncontiguous rectangles.
                prepare += [f"{lows}({axis},{count}) = int({lower}({axis}), c_size_t){offset}",
                            f"{highs}({axis},{count}) = int({upper}({axis}), c_size_t){offset}"]
            prepare += [f"{sections}({count})%lower = c_loc({lows}(1,{count}))",
                        f"{sections}({count})%upper = c_loc({highs}(1,{count}))", "endif"]
        singular = {"reads": "read", "writes": "write", "overwrites": "overwrite"}[label]
        prepare += [f"{access}%{singular}_count = {count}",
                    f"if ({count} > 0_c_size_t) {access}%{label} = c_loc({sections})"]
    return NativeAccessCode(tuple(line for statement in specification for line in _statement_lines(statement)),
                            tuple(line for statement in prepare for line in _statement_lines(statement)), access, handle)


def build_native_accesses(sections: NativeSections, handles, prefix: str, **options):
    """Require unambiguous canonical mappings before refining any access."""
    if not sections.available:
        raise CompilationError(sections.reason or "native sections are unavailable")
    mapped = []
    for resource in sections.resources:
        if resource.resource not in handles:
            raise CompilationError("native section buffer mapping is unavailable: " + resource.resource)
        mapped.append(handles[resource.resource])
    if len(set(mapped)) != len(mapped):
        raise CompilationError("native section aliases require a proved common physical mapping")
    return tuple(build_native_access(resource, handle, prefix + "_" + str(index), **options)
                 for index, (resource, handle) in enumerate(zip(sections.resources, mapped, strict=True)))


def build_native_view_accesses(sections: NativeSections, views, handles, prefix: str, **options):
    """Project exact native effects through validated canonical-root views.

    An unavailable refinement is a compile boundary: a partial actual cannot
    silently acquire whole-root hooks. Partial procedure-entry INTENT(OUT)
    events remain separate, using views.forget_view at their original point.
    Native callee lower bounds are independent of prepared numerical view
    lower bounds; only physical origins and child extents come from the view.
    """
    if not sections.available:
        raise CompilationError(sections.reason or "native view sections are unavailable")
    mapped = []
    for resource in sections.resources:
        if resource.resource not in handles or resource.resource not in views:
            raise CompilationError("native view mapping is unavailable: " + resource.resource)
        mapped.append(handles[resource.resource])
    if len(set(mapped)) != len(mapped):
        raise CompilationError("native view aliases require a proved common physical mapping")
    codes = tuple(build_native_access(resource, handle, prefix + "_" + str(index),
                                      view=views[resource.resource], **options)
                  for index, (resource, handle) in enumerate(zip(sections.resources, mapped, strict=True)))
    # Distinct generated expressions may name the same token after a nested
    # call maps separate formals onto one root. Reject that ambiguity before
    # preparing any access; string inequality is not an alias proof.
    status = options.get("status", "fort_status")
    on_error = options.get("on_error", ("return",))
    preflight = []
    for index, handle in enumerate(mapped):
        for previous in mapped[:index]:
            preflight += [f"if ({handle} == {previous}) then", f"{status} = FORT_SCOPE_ALIAS", *on_error, "endif"]
    if preflight:
        first = codes[0]
        checked = tuple(line for statement in preflight for line in _statement_lines(statement))
        codes = (NativeAccessCode(first.specification, checked + first.prepare, first.access_name, first.handle), *codes[1:])
    return codes
