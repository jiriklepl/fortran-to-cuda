"""Pointwise full-section assignments inside original independent loops.

Only full (:) axes are expanded here. Nontrivial slices and overlapping RHS
snapshots keep their own native boundary. The result is numerical syntax, not
source-effect authority; effects still come from the original assignment.
"""

from copy import copy

from fparser.common.readfortran import FortranStringReader
from fparser.two import Fortran2003 as F
from fparser.two.utils import Base

from compiler.frontend.source_effects import _children, _kind
from compiler.ir import CompilationError


def expand_full_sections(analysis, routine, node, variables, guards):
    """Return scalar numerical syntax after proving pointwise RHS semantics."""
    if _kind(node) != "Assignment_Stmt":
        return None
    target, _, value = node.items
    if _kind(target) not in {"Name", "Part_Ref"}:
        return None
    base = target.items[0] if _kind(target) == "Part_Ref" else target
    output = analysis._binding(routine.scope, base)
    if output is None or not output.rank:
        return None
    indices = tuple(_children(target.items[1])) if _kind(target) == "Part_Ref" else (None,) * output.rank
    axes = tuple(axis for axis, index in enumerate(indices, 1)
                 if index is None or _kind(index) == "Subscript_Triplet")
    if not axes:
        return None
    if len(indices) != output.rank or any(index is not None and _kind(index) == "Subscript_Triplet"
                                        and any(value is not None for value in index.items) for index in indices):
        raise CompilationError("nested array operation requires full section axes")
    iterators = []
    for _axis in axes:
        name = "fort_section_index_" + str(len(variables) + 1)
        if analysis._binding(routine.scope, name) or analysis._candidates(routine.scope, name):
            raise CompilationError("nested array operation iterator conflicts with original source")
        variables.append(name)
        iterators.append(name)
        if len(variables) > 32:
            raise CompilationError("nested section iterator budget exhausted")

    def signature(items):
        return tuple(":" if item is None or (_kind(item) == "Subscript_Triplet"
                                             and all(value is None for value in item.items))
                     else str(item).lower() for item in items)

    def element(reference, binding, *, writing=False):
        own = tuple(_children(reference.items[1])) if _kind(reference) == "Part_Ref" else (None,) * binding.rank
        if len(own) != binding.rank:
            raise CompilationError("nested array operation subscript rank is inconsistent")
        section_axes = tuple(axis for axis, index in enumerate(own, 1)
                             if index is None or _kind(index) == "Subscript_Triplet")
        if any(index is not None and _kind(index) == "Subscript_Triplet"
               and any(value is not None for value in index.items) for index in own):
            raise CompilationError("nested array operation requires full section axes")
        if not writing and binding.root == output.root and signature(own) != signature(indices):
            raise CompilationError("nested array operation requires a snapshot for a different output section")
        if section_axes and len(section_axes) != len(axes):
            raise CompilationError("nested array expression ranks do not conform")
        names = iter(iterators)
        dimensions = iter(axes)
        coordinates = []
        name = str(reference.items[0]) if _kind(reference) == "Part_Ref" else str(reference)
        for axis, index in enumerate(own, 1):
            if axis not in section_axes:
                coordinates.append(str(index))
                continue
            iterator, output_axis = next(names), next(dimensions)
            if binding.root == output.root and axis == output_axis:
                coordinate = iterator
            else:
                coordinate = f"lbound({name},{axis})+({iterator}-lbound({base},{output_axis}))"
                guards.append(f"size({name},{axis},kind=c_int64_t) == size({base},{output_axis},kind=c_int64_t)")
            coordinates.append(coordinate)
        return F.Part_Ref(name + "(" + ",".join(coordinates) + ")")

    def expression(item):
        if not isinstance(item, Base):
            return item
        kind = _kind(item)
        if kind in {"Name", "Part_Ref"}:
            name = item.items[0] if kind == "Part_Ref" else item
            binding = analysis._binding(routine.scope, name)
            if binding is None:
                raise CompilationError("nested array expression requires resolved scalar operands")
            return element(item, binding) if binding.rank else item
        if kind in {"Function_Reference", "Structure_Constructor", "Data_Ref"}:
            raise CompilationError("nested array expression requires simple numeric operands")
        if kind == "Actual_Arg_Spec":
            result = copy(item)
            result.items = (item.items[0], expression(item.items[1]))
            return result
        if kind == "Intrinsic_Function_Reference":
            name = str(item.items[0]).lower()
            if analysis._binding(routine.scope, name) or analysis._candidates(routine.scope, name):
                raise CompilationError("nested array expression intrinsic is shadowed")
            if name in {"size", "lbound", "ubound"}:
                return item  # Descriptor inquiries are scalar original inputs.
            if name not in {"abs", "min", "max", "sqrt", "exp", "log", "sin", "cos", "acos", "real", "int"}:
                raise CompilationError("nested array expression requires a proved elemental intrinsic")
            result = copy(item)
            result.items = (item.items[0], expression(item.items[1]))
            return result
        result = copy(item)
        if hasattr(item, "items"):
            result.items = tuple(expression(child) for child in item.items)
        return result

    assignment = str(element(target, output, writing=True)) + " = " + str(expression(value))
    body = [f"do {iterator}=lbound({base},{axis}),ubound({base},{axis})"
            for axis, iterator in reversed(tuple(zip(axes, iterators, strict=True)))]
    body += [assignment, *("enddo" for _ in axes)]
    return F.Block_Nonlabel_Do_Construct(FortranStringReader("\n".join(body) + "\n"))
