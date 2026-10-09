"""Checked borrowed views for source-backed rectangular numerical actuals.

Every arithmetic operand is checked against the numerical INTEGER ABI before
an operation. Two such operands fit the wider temporary even for multiplication;
the result is checked again before it can become an operand or a coordinate.
No source expression is evaluated by unguarded generated Fortran arithmetic.
"""

from dataclasses import dataclass

from compiler.ir import CompilationError

MINIMUM = "(-2147483647_c_int64_t - 1_c_int64_t)"
MAXIMUM = "2147483647_c_int64_t"


def view_call(call):
    if call.region is not None:
        return False
    return call.borrowed or any(mapping.section is not None for mapping in call.resolved.mappings)


def dependencies(call):
    if call.region is not None:
        return {}
    return {dependency.resource: dependency.binding
            for mapping in call.resolved.mappings if mapping.section is not None
            for dependency in mapping.section.dependencies if dependency.kind == "scalar_read"}


def check_call(builder, call):
    """Do not confuse an analyzable section with an executable native view."""
    if not view_call(call):
        return
    if builder.config.collective:
        raise CompilationError("rectangular numerical views require a serial source coordinator")
    if (not builder.numerical(call.procedure)
            and not builder.view_wrapper_supported(call.procedure, require_borrowed=False)
            and not builder.view_native_supported(call.procedure)):
        raise CompilationError("native rectangular source-call mapping requires projected native hooks")
    if call.procedure in builder.packages:
        raise CompilationError("normalized numerical package sections require composed original-coordinate views")
    if any(mapping.presence != "supplied" or mapping.formal_binding.attributes & {"allocatable", "optional"}
           for mapping in call.resolved.mappings):
        raise CompilationError("rectangular numerical views require present ordinary numerical formals")
    for binding in dependencies(call).values():
        if binding.signature() != ("integer", 4, 0):
            raise CompilationError("rectangular bound execution requires checked default INTEGER controls: " + binding.root)
    if not builder.numerical(call.procedure) and builder.view_native_supported(call.procedure):
        roots = [mapping.resource for mapping in call.resolved.mappings if mapping.formal_binding.rank]
        if len(roots) != len(set(roots)):
            raise CompilationError("native rectangular aliases require merged projected access descriptors")


def forget_view(name, rank, *, query, on_error):
    """Retain a wrapper's original partial INTENT(OUT) event."""
    function = "fort_scope_plan_forget_sections_v1" if query else "fort_scope_forget_sections_v1"
    return ["block", "type(fort_scope_view_layout_v1) :: fort_discard_layout",
            "type(fort_scope_section), target :: fort_discard_section(1)",
            "integer(c_size_t), pointer :: fort_discard_origin(:), fort_discard_extent(:)",
            f"integer(c_size_t), target :: fort_discard_upper({rank})",
            "integer(c_size_t) :: fort_discard_count",
            f"fort_status = fort_scope_view_get_v1(fort_context, {name}, fort_discard_layout)",
            "if (fort_status /= FORT_SCOPE_OK) then", *on_error, "endif",
            f"call c_f_pointer({name}%origins, fort_discard_origin, [{rank}])",
            f"call c_f_pointer({name}%extents, fort_discard_extent, [{rank}])",
            "fort_discard_upper = fort_discard_origin + fort_discard_extent",
            "fort_discard_section(1)%lower = c_loc(fort_discard_origin)",
            "fort_discard_section(1)%upper = c_loc(fort_discard_upper)",
            "fort_discard_count = 0_c_size_t",
            "if (all(fort_discard_extent > 0_c_size_t)) fort_discard_count = 1_c_size_t",
            f"fort_status = {function}(fort_context, {name}%buffer, c_loc(fort_discard_section), fort_discard_count)",
            "if (fort_status /= FORT_SCOPE_OK) then", *on_error, "endif", "end block"]


@dataclass(frozen=True)
class BorrowedView:
    specification: tuple[str, ...]
    prepare: tuple[str, ...]
    name: str
    public: dict


def build_view(mapping, handle, original_lower, parameters, prefix, *, on_error, parent_view=None):
    """Describe one root view without payload reads or a CUDA initialization."""
    rank = mapping.formal_binding.rank
    if not rank or mapping.binding is None:
        raise CompilationError("borrowed numerical view requires a source-backed array actual")
    view, root, layout, dims = (prefix + suffix for suffix in ("_view", "_root", "_layout", "_dims"))
    origins, extents, lowers = (prefix + suffix for suffix in ("_origins", "_extents", "_lowers"))
    spec = [f"type(fort_scope_view_v1) :: {view}", f"type(fort_scope_layout) :: {root}",
            f"type(fort_scope_view_layout_v1) :: {layout}", f"integer(c_size_t), pointer :: {dims}(:)",
            f"integer(c_size_t), target :: {origins}({rank}), {extents}({rank})",
            f"integer(c_int64_t), target :: {lowers}({rank})"]
    body = [f"fort_status = fort_scope_layout_get(fort_context, {handle}, {root})",
            "if (fort_status /= FORT_SCOPE_OK) then", *on_error, "endif"]

    def require(condition):
        body.extend([f"if ({condition}) then", "fort_status = FORT_SCOPE_BOUNDARY", *on_error, "endif"])

    require(f"{root}%rank /= {rank} .or. .not. c_associated({root}%extents)")
    if parent_view:
        parent_origin = prefix + "_parent_origin"
        spec.append(f"integer(c_size_t), pointer :: {parent_origin}(:)")
        require(f"{parent_view}%buffer /= {handle}")
        body += [f"fort_status = fort_scope_view_get_v1(fort_context, {parent_view}, {layout})",
                 "if (fort_status /= FORT_SCOPE_OK) then", *on_error, "endif",
                 f"call c_f_pointer({parent_view}%extents, {dims}, [{rank}])",
                 f"call c_f_pointer({parent_view}%origins, {parent_origin}, [{rank}])"]
    else:
        body += [f"call c_f_pointer({root}%extents, {dims}, [{rank}])"]
    require(f"any({dims} < 0_c_size_t) .or. any({dims} > 2147483647_c_size_t)")
    body += [f"{origins} = 0_c_size_t", f"{extents} = {dims}", f"{lowers} = 1_c_int64_t"]
    serial = 0

    def temporary(expression):
        nonlocal serial
        name = prefix + "_value_" + str(serial)
        serial += 1
        spec.append(f"integer(c_int64_t) :: {name}")
        body.append(name + " = " + expression)
        require(f"{name} < {MINIMUM} .or. {name} > {MAXIMUM}")
        return name

    parent_lowers, parent_uppers = {}, {}
    if mapping.section is not None:
        for axis in range(1, rank + 1):
            lo = temporary(f"{original_lower}({axis})")
            parent_lowers[axis] = lo
            parent_uppers[axis] = temporary(f"{lo} + int({dims}({axis}), c_int64_t) - 1_c_int64_t")

    def bound(value):
        if value.kind == "literal":
            if value.value is None or not -(2**31) <= value.value < 2**31:
                raise CompilationError("rectangular bound literal exceeds the numerical INTEGER ABI")
            return temporary(str(value.value) + "_c_int64_t" if value.value >= 0 else
                             "(-" + str(-value.value) + "_c_int64_t)")
        if value.kind == "scalar":
            return temporary("int(" + parameters[value.resource] + ", c_int64_t)")
        if value.kind in {"lbound", "ubound", "size"}:
            axis = value.dimension
            if value.kind == "lbound":
                return parent_lowers[axis]
            if value.kind == "ubound":
                return parent_uppers[axis]
            return temporary(f"int({dims}({axis}), c_int64_t)")
        if value.kind == "parenthesis":
            return bound(value.children[0])
        if value.kind == "unary":
            return temporary(value.operator + "(" + bound(value.children[0]) + ")")
        if value.kind == "binary":
            lhs, rhs = (bound(child) for child in value.children)
            return temporary(lhs + " " + value.operator + " " + rhs)
        raise CompilationError("unsupported checked rectangular bound expression")

    if mapping.section is not None:
        for axis, section_axis in enumerate(mapping.section.axes, 1):
            lo, hi = bound(section_axis.lower), bound(section_axis.upper)
            # A zero-size section has no referenced element. Normalize that
            # axis only; independent axes still pass their containment checks.
            body += [f"if ({hi} < {lo}) then", f"{origins}({axis}) = 0_c_size_t",
                     f"{extents}({axis}) = 0_c_size_t", "else"]
            require(f"{lo} < {parent_lowers[axis]} .or. {hi} > {parent_uppers[axis]}")
            body += [f"{origins}({axis}) = int({lo} - {parent_lowers[axis]}, c_size_t)",
                     f"{extents}({axis}) = int({hi} - {lo} + 1_c_int64_t, c_size_t)", "endif"]
    if parent_view:
        # Both operands are contained within the validated root extent. Their
        # sum therefore cannot overflow size_t or change the canonical layout.
        body += [f"{origins} = {origins} + {parent_origin}"]
    body += [f"{view} = fort_scope_view_v1()", f"{view}%rank = {rank}", f"{view}%buffer = {handle}",
             f"{view}%generation = {root}%generation", f"{view}%origins = c_loc({origins})",
             f"{view}%extents = c_loc({extents})", f"{view}%lower_bounds = c_loc({lowers})",
             f"fort_status = fort_scope_view_get_v1(fort_context, {view}, {layout})",
             "if (fort_status /= FORT_SCOPE_OK) then", *on_error, "endif"]
    return BorrowedView(tuple(spec), tuple(body), view,
                        {"formal": mapping.formal, "resource": mapping.resource,
                         "section": mapping.section.public() if mapping.section is not None else None,
                         "view_abi_version": 1, "logical_lower_bounds": [1] * rank,
                         "physical_layout": "borrowed canonical root pitches; no section packing",
                         "bounds": "checked original caller coordinates before numerical execution"})
