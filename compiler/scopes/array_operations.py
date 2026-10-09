"""Source-proven conformant array operations as ordinary numerical loops.

Only the reached assignment is borrowed. Original storage, allocation guards,
logical bounds and native RHS snapshot semantics remain authoritative. The
ordinary dependence proof rejects operations needing an overlapping snapshot.
"""

from dataclasses import dataclass
from hashlib import sha256
import copy

from fparser.two import Fortran2003 as F
from fparser.two.utils import Base, walk

from compiler.frontend.call_bindings import _bound
from compiler.frontend.component_bindings import component_access, source_scope_for
from compiler.frontend.source_effects import _children, _kind
from compiler.ir import CompilationError, SourceLocation
from compiler.scopes.numerical import Parameter
from compiler.scopes.regions import RegionExtraction, allocation_guard, _exception_observers, _validate_real_varargs
from compiler.scopes.segments import fortran_lines, statement_span


@dataclass(frozen=True)
class _Selector:
    binding: object
    axes: tuple
    extents: tuple[str, ...]


def extract_array_operation(analysis, routine, node, *, preceding=(), following=()):
    """Normalize one original assignment; runtime checks run at its source point."""
    analysis.inputs.verify()
    role = analysis._source_roles.get(routine.qualified)
    if (role is None or analysis.routines.get(routine.qualified) is not routine
            or routine.execution is not role[0] or analysis._routine_signature(routine) != role[1]
            or str(routine.execution) != role[2] or not any(node is item for item in walk(routine.execution))):
        raise CompilationError("array operation requires unchanged original source authority")
    if _kind(node) != "Assignment_Stmt":
        raise CompilationError("array operation requires an original assignment")
    if (any(issue != "specification effect unavailable: Derived_Type_Def" for issue in routine.issues)
            or analysis._unknown_exports(routine.scope)):
        raise CompilationError("array operation needs resolved original declarations and imports")
    target, _, value = node.items
    scope = source_scope_for(analysis, node, routine.scope)
    _validate_real_varargs(analysis, routine, node)
    if observers := _exception_observers(analysis):
        raise CompilationError("source-observable floating-point exception flags prevent GPU array operation: "
                               + "; ".join(observers))
    bindings, names, lower_names, guards = {}, {}, {}, []
    min64 = "(-2147483647_c_int64_t - 1_c_int64_t)"
    max64 = "2147483647_c_int64_t"

    def capture(binding):
        if binding is None:
            raise CompilationError("array operation contains an unresolved variable")
        boundary = analysis.resource_identity_boundary(binding)
        if boundary:
            raise CompilationError(boundary)
        if binding.attributes & {"pointer", "optional", "volatile", "asynchronous", "value"}:
            raise CompilationError("array operation association is uncertain: " + binding.root)
        if binding.signature()[:2] not in {("real", 4), ("real", 8), ("integer", 4), ("logical", 1)}:
            raise CompilationError("array operation numerical type is unsupported: " + binding.root)
        if binding.rank and binding.dtype == "logical":
            raise CompilationError("logical array numerical ABI is unsupported")
        if not binding.rank and "allocatable" in binding.attributes:
            raise CompilationError("array operation scalar descriptor is unsupported")
        if binding.root not in bindings:
            index = len(bindings)
            bindings[binding.root] = binding
            names[binding.root] = "fort_array_value_" + str(index)
            for axis in range(1, binding.rank + 1):
                lower_names[binding.root, axis] = "fort_array_lower_" + str(index) + "_" + str(axis)
        return names[binding.root]

    def checked(expression):
        guards.extend((f"{expression} >= {min64}", f"{expression} <= {max64}"))
        return expression

    def bound(expression, binding, axis, default=None):
        fact = _bound(analysis, source_scope_for(analysis, expression, scope), expression, binding, axis, default)

        def render(item):
            if item.kind == "literal":
                return str(item.value), checked(str(item.value) + "_c_int64_t" if item.value >= 0 else
                                               "(-" + str(-item.value) + "_c_int64_t)")
            if item.kind == "scalar":
                dependency, = (dependency for dependency in item.dependencies if dependency.kind == "scalar_read")
                name = capture(dependency.binding)
                return name, checked("int(" + dependency.binding.name + ",kind=c_int64_t)")
            if item.kind in {"lbound", "ubound", "size"}:
                name = capture(binding)
                dimension = item.dimension
                original = binding.name
                wide = checked(f"{item.kind}({original},{dimension},kind=c_int64_t)")
                if item.kind == "lbound":
                    return lower_names[binding.root, dimension], wide
                if item.kind == "ubound":
                    return f"({lower_names[binding.root, dimension]}+(size({name},{dimension})-1))", wide
                return f"size({name},{dimension})", wide
            children = [render(child) for child in item.children]
            if item.kind == "parenthesis":
                return f"({children[0][0]})", f"({children[0][1]})"
            if item.kind == "unary":
                return item.operator + f"({children[0][0]})", checked(item.operator + f"({children[0][1]})")
            if item.kind == "binary":
                return (f"({children[0][0]}{item.operator}{children[1][0]})",
                        checked(f"({children[0][1]}{item.operator}{children[1][1]})"))
            raise CompilationError("array operation bound is unsupported")
        return render(fact)

    def selector(reference):
        name = reference.items[0] if _kind(reference) == "Part_Ref" else reference
        binding = analysis._binding(source_scope_for(analysis, reference, scope), name)
        if binding is None or not binding.rank:
            raise CompilationError("array operation selector is not an original numeric array")
        capture(binding)
        access = component_access(analysis, source_scope_for(analysis, reference, scope), reference)
        indices = (tuple(_children(reference.items[1])) if _kind(reference) == "Part_Ref"
                   else access.indices if access is not None and access.indices else (None,) * binding.rank)
        if len(indices) != binding.rank:
            raise CompilationError("array operation selector rank is inconsistent")
        axes, extents = [], []
        for axis, index in enumerate(indices, 1):
            if index is None or _kind(index) == "Subscript_Triplet":
                lo, hi, step = index.items if index is not None else (None, None, None)
                stride = 1 if step is None else routine.scope.kinds.integer(step, SourceLocation(str(routine.scope.path)))
                if not stride or not -(2**31) < stride < 2**31:
                    raise CompilationError("array operation stride must be a nonzero supported INTEGER constant")
                low, low64 = bound(lo, binding, axis, "lbound" if stride > 0 else "ubound")
                high, high64 = bound(hi, binding, axis, "ubound" if stride > 0 else "lbound")
                distance = f"({high64}-{low64})" if stride > 0 else f"({low64}-{high64})"
                # Every default-INTEGER intermediate in the worker is checked
                # before association; wider two-operand arithmetic cannot overflow.
                checked(distance)
                # Clamp before division: Fortran INTEGER division truncates
                # toward zero, so MAX(0,d/s+1) is incorrect for -s<d<0.
                # The clamped numerator is nonnegative for every legal s.
                numerator64 = f"(max({distance},-1_c_int64_t)+{abs(stride)}_c_int64_t)"
                checked(numerator64)
                extent64 = f"({numerator64}/{abs(stride)}_c_int64_t)"
                distance_default = f"({high}-{low})" if stride > 0 else f"({low}-{high})"
                extent = f"((max({distance_default},-1)+{abs(stride)})/{abs(stride)})"
                checked(extent64)
                # Logical endpoints can fit INTEGER even when the offset
                # multiply/rebase in a normalized worker would overflow.
                checked(f"(max({extent64}-1_c_int64_t,0_c_int64_t)*{abs(stride)}_c_int64_t)")
                # Explicit empty sections do not reference their endpoints.
                minimum, maximum = (low64, high64) if stride > 0 else (high64, low64)
                guards.append(f"({extent64} == 0_c_int64_t .or. ({minimum} >= "
                              f"lbound({binding.name},{axis},kind=c_int64_t) .and. {maximum} <= "
                              f"ubound({binding.name},{axis},kind=c_int64_t)))")
                axes.append((low, stride, len(extents)))
                extents.append((extent, extent64))
            else:
                coordinate, coordinate64 = bound(index, binding, axis)
                guards.extend((f"{coordinate64} >= lbound({binding.name},{axis},kind=c_int64_t)",
                               f"{coordinate64} <= ubound({binding.name},{axis},kind=c_int64_t)"))
                axes.append((coordinate, None, None))
        return _Selector(binding, tuple(axes), tuple(extents))

    if _kind(target) not in {"Name", "Part_Ref", "Data_Ref"}:
        raise CompilationError("array operation requires a simple numeric array target")
    output = selector(target)
    if not output.extents:
        raise CompilationError("array operation target must retain at least one array dimension")
    if output.binding.intent == "in":
        raise CompilationError("array operation writes original INTENT(IN) storage")
    iterators = tuple("fort_array_index_" + str(axis) for axis in range(len(output.extents)))

    def element(selection):
        if selection.extents:
            if len(selection.extents) != len(output.extents):
                raise CompilationError("array expression ranks do not conform")
            guards.extend(f"{right[1]} == {left[1]}" for left, right in
                          zip(output.extents, selection.extents, strict=True))
        indices = []
        for axis, (coordinate, stride, ordinal) in enumerate(selection.axes, 1):
            logical = coordinate if stride is None else f"({coordinate}+({iterators[ordinal]}-1)*({stride}))"
            # A nonempty array dimension and its worker offset both fit the
            # existing INTEGER ABI; rebasing still needs an explicit check.
            guards.append(f"size({selection.binding.name},{axis},kind=c_int64_t) <= {max64}")
            indices.append(f"({logical})-{lower_names[selection.binding.root,axis]}+1")
        return F.Part_Ref(names[selection.binding.root] + "(" + ",".join(indices) + ")")

    def expression(item):
        kind = _kind(item)
        if kind in {"Name", "Part_Ref", "Data_Ref"}:
            name = item.items[0] if kind == "Part_Ref" else item
            binding = analysis._binding(source_scope_for(analysis, item, scope), name)
            if binding is None:
                raise CompilationError("array expression contains an unresolved variable or helper: " + str(name))
            if binding.rank:
                return element(selector(item))
            if kind not in {"Name", "Data_Ref"} or kind == "Data_Ref" and component_access(analysis, scope, item).indices:
                raise CompilationError("array expression uses an unsupported scalar selector")
            if "parameter" in binding.attributes and binding.dtype == "integer":
                return F.Level_2_Expr(str(routine.scope.kinds.integer(item, SourceLocation(str(routine.scope.path)))))
            return F.Name(capture(binding))
        if kind == "Intrinsic_Function_Reference":
            intrinsic = str(item.items[0]).lower()
            if analysis._binding(routine.scope, intrinsic) or analysis._candidates(routine.scope, intrinsic):
                raise CompilationError("array expression intrinsic is shadowed")
            args = tuple(_children(item.items[1]))
            if intrinsic in {"lbound", "ubound", "size"}:
                if intrinsic == "size" and len(args) == 1 and _kind(args[0]) in {"Name", "Data_Ref"}:
                    binding = analysis._binding(source_scope_for(analysis, args[0], scope), args[0])
                    if binding is None or not binding.rank:
                        raise CompilationError("array expression SIZE requires an original whole-array descriptor")
                    name = capture(binding)
                    checked(f"size({binding.name},kind=c_int64_t)")
                    return F.Intrinsic_Function_Reference("size(" + name + ")")
                if len(args) != 2 or _kind(args[0]) not in {"Name", "Data_Ref"}:
                    raise CompilationError("array expression inquiries require a constant scalar DIM")
                binding = analysis._binding(source_scope_for(analysis, args[0], scope), args[0])
                axis = routine.scope.kinds.integer(args[1], SourceLocation(str(routine.scope.path)))
                rendered, _ = bound(item, binding, axis)
                return F.Level_2_Expr(rendered)
            result = copy.copy(item)
            result.items = (item.items[0], expression(item.items[1]))
            return result
        if kind == "Actual_Arg_Spec":
            result = copy.copy(item)
            result.items = (item.items[0], expression(item.items[1]))
            return result
        if not isinstance(item, Base):
            return item
        if kind in {"Real_Literal_Constant", "Int_Literal_Constant"} and item.items[1] is not None:
            selector_kind = routine.scope.kinds.integer(item.items[1], SourceLocation(str(routine.scope.path)))
            result = copy.copy(item)
            result.items = (item.items[0], str(selector_kind))
            return result
        result = copy.copy(item)
        for attribute in ("content", "items"):
            if hasattr(item, attribute):
                children = getattr(item, attribute)
                setattr(result, attribute, tuple(expression(child) for child in children)
                        if isinstance(children, tuple) else [expression(child) for child in children])
        return result

    assignment = str(element(output)) + " = " + str(expression(value))
    if len(bindings) > 64:
        raise CompilationError("array operation capture budget exceeded")
    for intrinsic in ("int", "lbound", "ubound", "size", "max"):
        if analysis._binding(routine.scope, intrinsic) or analysis._candidates(routine.scope, intrinsic):
            raise CompilationError("array operation generated inquiry conflicts with an original intrinsic binding")
    declarations, parameters = [], []
    allocation_guards = []
    for root, binding in bindings.items():
        dtype = binding.dtype if binding.dtype in {"integer", "logical"} else f"real({binding.kind})"
        shape = "(" + ",".join(":" for _ in range(binding.rank)) + ")" if binding.rank else ""
        intent = "inout" if root == output.binding.root else "in"
        declarations.append(f"{dtype},intent({intent}) :: {names[root]}{shape}")
        parameters.append(Parameter(names[root], root, binding.rank))
        for axis in range(1, binding.rank + 1):
            parameters.append(Parameter(lower_names[root,axis], root, 0, axis, True))
            declarations.append(f"integer,intent(in) :: {lower_names[root,axis]}")
        if "allocatable" in binding.attributes and root.startswith(routine.qualified + "::"):
            allocation_guards.append(allocation_guard(analysis, routine, binding, preceding))
    body = [f"do {iterator}=1,{extent[0]}" for iterator, extent in reversed(tuple(zip(iterators, output.extents, strict=True)))]
    body += [assignment, *("enddo" for _ in iterators)]
    identity = sha256(("array-operation-v1\0" + analysis.sources[str(routine.scope.path)] + "\0" +
                       routine.qualified + "\0" + str(statement_span(node)) + "\0" + str(node)).encode()).hexdigest()
    module = "fort_array_operation_" + identity[:12]
    source = "\n".join(fortran_lines([f"module {module}", "implicit none", "contains", "subroutine region(" +
        ",".join(parameter.name for parameter in parameters) + ")", *declarations,
        "integer :: " + ",".join(iterators), *body, "end subroutine", "end module"])) + "\n"
    analysis.inputs.verify()
    return RegionExtraction((node,), statement_span(node), source, module + "::region", tuple(parameters),
                            tuple(bindings.values()), frozenset((output.binding.root,)), (),
                            tuple(allocation_guards), identity,
                            {"available": True, "caller_contract": "serial_source_scope",
                             "reason": "original conformant array assignment completes before continuation"},
                            runtime_guards=tuple(dict.fromkeys(guards)), operation_kind="array_assignment",
                            numerical_environment_required=any(binding.dtype == "real" for binding in bindings.values()))
