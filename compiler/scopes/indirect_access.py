"""Bounded Fortran inspectors for native indirect communication footprints."""

from __future__ import annotations

from compiler.frontend.indirect_sections import IndirectSections, INSPECTION_STEP_LIMIT
from compiler.frontend.native_sections import RECTANGLE_LIMIT
from compiler.ir import CompilationError
from compiler.ir.integers import INTEGER_MIN, INTEGER_MAX
from compiler.scopes.access import NativeAccessCode, _statement_lines, build_native_access


def build_indirect_accesses(sections: IndirectSections, handles, prefix, *, resource_names,
                            parameters=None, context="fort_context", status="fort_status",
                            on_error=("return",), logical_lower_bounds=None):
    """Prepare exact READ/WRITE unions at the original native boundary.

    Metadata objects must first be reserved against managed-buffer aliases.
    The generated loops read fixed fields using original Fortran selectors and
    make no runtime calls per point.  Checked descriptor/coordinate failures and
    an over-budget exact union return a boundary before numerical execution.
    Arrays referenced by each returned access must live through host_end.
    """
    if not sections.available:
        raise CompilationError(sections.reason or "indirect native sections are unavailable")
    mapped = []
    for resource in sections.resources:
        if resource.resource not in handles:
            raise CompilationError("indirect native buffer mapping is unavailable: " + resource.resource)
        mapped.append(handles[resource.resource])
    if len(set(mapped)) != len(mapped):
        raise CompilationError("indirect native aliases require a proved common physical mapping")
    required = {binding.root for binding in sections.metadata}
    required.update(binding.root for binding in sections.scalars if binding.rank)
    if not required <= resource_names.keys():
        raise CompilationError("indirect native original metadata/descriptor mapping is unavailable: "
                               + ", ".join(sorted(required - resource_names.keys())))
    parameters = parameters or {}
    required_scalars = {binding.root for binding in sections.scalars if not binding.rank}
    if not required_scalars <= parameters.keys():
        raise CompilationError("indirect native reached scalar mapping is unavailable: "
                               + ", ".join(sorted(required_scalars - parameters.keys())))
    return tuple(_build(sections, resource, handles[resource.resource], prefix + "_" + str(index),
                        resource_names, parameters, context, status, on_error, logical_lower_bounds)
                 for index, resource in enumerate(sections.resources))


def _build(sections, resource, handle, prefix, names, parameters, context, status, on_error, logical_lower_bounds):
    base = build_native_access(resource, handle, prefix, context=context, status=status,
                               on_error=on_error, logical_lower_bounds=logical_lower_bounds)
    specification, prepare = list(base.specification), list(base.prepare)
    rank, access = resource.rank, base.access_name
    origin, extents = prefix + "_origin", prefix + "_extents"
    lo, hi, item, count, different, axis, changed, contained = (
        prefix + suffix for suffix in ("_lo", "_hi", "_item", "_count", "_different", "_axis", "_changed", "_contained"))
    specification += [f"integer(c_size_t) :: {lo}({rank}), {hi}({rank})",
                      f"integer :: {item}, {count}, {different}, {axis}",
                      f"logical :: {changed}, {contained}"]
    scan_steps = prefix + "_scan_steps"
    specification.append(f"integer(c_int64_t) :: {scan_steps}")
    prepare += [f"{scan_steps} = 0_c_int64_t"]
    temporary_count = 0
    metadata = {binding.root: binding for binding in sections.metadata}
    descriptors = {**metadata, **{binding.root: binding for binding in sections.scalars if binding.rank}}

    def require(condition):
        prepare.extend([f"if ({condition}) then", f"{status} = FORT_SCOPE_BOUNDARY", *on_error, "endif"])

    require(f"any({extents} > {INTEGER_MAX}_c_size_t)")

    # No descriptor inquiries occur before allocation checks. Original lower
    # bounds and original object fields remain in their owning Fortran scope.
    for root, binding in sorted(descriptors.items()):
        if "allocatable" in binding.attributes:
            require(f".not. allocated({names[root]})")

    def checked(expression):
        nonlocal temporary_count
        temporary_count += 1
        name = prefix + "_value" + str(temporary_count)
        specification.append(f"integer(c_int64_t) :: {name}")
        prepare.append(f"{name} = {expression}")
        require(f"{name} < {INTEGER_MIN}_c_int64_t .or. {name} > {INTEGER_MAX}_c_int64_t")
        return name

    def scan_step():
        # Count entered iterations at every depth, including an outer loop
        # whose nested loop is empty. Pathological empty products therefore
        # cannot evade the bounded inspector-work budget.
        require(f"{scan_steps} >= {INSPECTION_STEP_LIMIT}_c_int64_t")
        prepare.append(f"{scan_steps} = {scan_steps} + 1_c_int64_t")

    def value(expression, iterators):
        if expression.kind == "literal":
            return str(expression.value) + "_c_int64_t"
        if expression.kind == "scalar":
            if expression.resource in iterators:
                return iterators[expression.resource]
            return checked(f"int({parameters[expression.resource]}, c_int64_t)")
        if expression.kind in {"size", "lbound", "ubound"}:
            keyword = ", kind=c_int64_t" if expression.kind in {"size", "lbound", "ubound"} else ""
            return checked(f"{expression.kind}({names[expression.resource]}, {expression.dimension}{keyword})")
        if expression.kind in {"binary", "unary"}:
            children = tuple(value(child, iterators) for child in expression.children)
            if expression.kind == "unary":
                return checked(expression.operator + "(" + children[0] + ")")
            # Operands fit the existing default INTEGER ABI; their widened
            # addition, subtraction and product fit signed int64 before each
            # original source intermediate is checked again.
            return checked("(" + children[0] + " " + expression.operator + " " + children[1] + ")")
        if expression.kind == "metadata":
            coordinates = tuple(value(child, iterators) for child in expression.children)
            for dimension, coordinate in enumerate(coordinates, 1):
                require(f"{coordinate} < lbound({names[expression.resource]}, {dimension}, kind=c_int64_t) .or. "
                        f"{coordinate} > ubound({names[expression.resource]}, {dimension}, kind=c_int64_t)")
            selector = names[expression.resource] + "(" + ", ".join(coordinates) + ")%" + expression.member
            return checked(f"int({selector}, c_int64_t)")
        raise CompilationError("indirect inspector expression has no checked lowering")

    # Each direction gets its own exact union. Overwrites deliberately stay
    # empty: arbitrary scatter assignments are may-writes, never a definition
    # proof. Separate opposite faces remain separate unless the union is itself
    # one rectangle (equal extents on every axis but one touching interval).
    for action in ("read", "write"):
        references = tuple(item for item in sections.references if item.resource == resource.resource and item.action == action)
        if not references:
            continue
        suffix = "_r" if action == "read" else "_w"
        low, high, boxes, total = (prefix + suffix + tail for tail in ("_lows", "_highs", "_sections", "_total"))
        specification += [f"integer(c_size_t), target :: {low}({rank},{RECTANGLE_LIMIT}), {high}({rank},{RECTANGLE_LIMIT})",
                          f"type(fort_scope_section), target :: {boxes}({RECTANGLE_LIMIT})",
                          f"integer :: {total}"]
        prepare += [f"{total} = 0"]
        for reference_index, reference in enumerate(references):
            iterators = {}
            for loop_index, loop in enumerate(reference.loops):
                iterator = prefix + suffix + "_i" + str(reference_index) + "_" + str(loop_index)
                specification.append(f"integer(c_int64_t) :: {iterator}")
                lower, upper = value(loop.lower, iterators), value(loop.upper, iterators)
                prepare += [f"do {iterator} = {lower}, {upper}, {loop.step}_c_int64_t"]
                iterators[loop.iterator] = iterator
                scan_step()
            if not reference.loops:
                scan_step()
            for dimension, coordinate in enumerate(reference.indices, 1):
                logical = value(coordinate, iterators)
                # The layout extent is size_t, but it was constrained by the
                # source ABI in build_native_access; test without subtraction
                # in unsigned arithmetic to retain negative original bounds.
                require(f"{logical} < {origin}({dimension}) .or. "
                        f"{logical} - {origin}({dimension}) >= int({extents}({dimension}), c_int64_t)")
                prepare += [f"{lo}({dimension}) = int({logical} - {origin}({dimension}), c_size_t)",
                            f"{hi}({dimension}) = {lo}({dimension}) + 1_c_size_t"]
            prepare += _append_exact(lo, hi, low, high, total, item, count, different, axis,
                                     changed, contained, rank, status, on_error)
            prepare += ["enddo"] * len(reference.loops)
        prepare += [f"do {item} = 1, {total}",
                    f"{boxes}({item})%lower = c_loc({low}(1,{item}))",
                    f"{boxes}({item})%upper = c_loc({high}(1,{item}))", "enddo",
                    f"{access}%{action}_count = int({total}, c_size_t)",
                    f"if ({total} > 0) {access}%{action}s = c_loc({boxes})"]
    return NativeAccessCode(tuple(line for statement in specification for line in _statement_lines(statement)),
                            tuple(line for statement in prepare for line in _statement_lines(statement)), access, handle)


def _append_exact(lo, hi, lows, highs, total, item, count, different, axis, changed, contained, rank, status, on_error):
    """Insert one box, merging only when its exact union is a rectangle.

    Prefixes may exceed the bounded representation even if a later completed
    plane could fit. Such order-sensitive loss of precision is conservative:
    it closes ownership rather than inventing a volume-sized envelope.
    """
    return [f"{contained} = .false.", f"{item} = 1", f"do while ({item} <= {total})",
            f"if (all({lo} >= {lows}(:,{item})) .and. all({hi} <= {highs}(:,{item}))) then",
            f"{contained} = .true.", "exit", "endif",
            f"{changed} = all({lows}(:,{item}) >= {lo}) .and. all({highs}(:,{item}) <= {hi})",
            f"{different} = 0", f"{axis} = 0", f"do {count} = 1, {rank}",
            f"if ({lo}({count}) /= {lows}({count},{item}) .or. {hi}({count}) /= {highs}({count},{item})) then",
            f"{different} = {different} + 1", f"{axis} = {count}", "endif", "enddo",
            f"if ({different} == 1) then",
            f"if ({lo}({axis}) <= {highs}({axis},{item}) .and. {lows}({axis},{item}) <= {hi}({axis})) then",
            f"{lo} = min({lo}, {lows}(:,{item}))", f"{hi} = max({hi}, {highs}(:,{item}))",
            f"{changed} = .true.", "endif", "endif", f"if ({changed}) then",
            f"{lows}(:,{item}) = {lows}(:,{total})", f"{highs}(:,{item}) = {highs}(:,{total})",
            f"{total} = {total} - 1", f"{item} = 1", "else", f"{item} = {item} + 1", "endif", "enddo",
            f"if (.not. {contained}) then", f"if ({total} == {RECTANGLE_LIMIT}) then",
            f"{status} = FORT_SCOPE_BOUNDARY", *on_error, "endif", f"{total} = {total} + 1",
            f"{lows}(:,{total}) = {lo}", f"{highs}(:,{total}) = {hi}", "endif"]
