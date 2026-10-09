"""Bounded native sections from original ASTs, separate from whole effects.

Assignments and bounded affine counted loops are refined. The original native
computation is retained; these facts describe its physical array accesses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

from fparser.two.utils import walk

from compiler.ir import CompilationError, SourceLocation
from compiler.ir.integers import integer_literal
from compiler.ir.intrinsics import ARRAY_INQUIRIES, INTRINSICS, MODEL_INQUIRIES
from compiler.frontend.call_bindings import SourceDependency, _bound

RECTANGLE_LIMIT = 32


def _kind(node):
    return type(node).__name__


def _children(node):
    return getattr(node, "content", getattr(node, "items", ())) or ()


@dataclass(frozen=True)
class NativeBound:
    kind: str
    value: int | None = None
    dimension: int | None = None
    resource: str | None = None
    operator: str | None = None
    children: tuple[NativeBound, ...] = ()
    dependencies: tuple[SourceDependency, ...] = field(default=(), compare=False, repr=False)

    def public(self):
        return {"kind": self.kind, **({"value": self.value} if self.value is not None else {}),
                **({"dimension": self.dimension} if self.dimension is not None else {}),
                **({"resource": self.resource} if self.resource is not None
                   and self.kind in {"scalar", "lbound", "ubound", "size"} else {}),
                **({"operator": self.operator} if self.operator is not None else {}),
                **({"children": [child.public() for child in self.children]} if self.children else {})}


@dataclass(frozen=True)
class NativeIteration:
    lower: NativeBound
    upper: NativeBound
    step: int = 1

    def public(self):
        return {"lower": self.lower.public(), "upper": self.upper.public(), "step": self.step}


@dataclass(frozen=True)
class NativeAxis:
    lower: NativeBound | None = None
    upper: NativeBound | None = None
    point: bool = False

    def public(self):
        return {"kind": "point" if self.point else "unit_stride_range",
                "lower": self.lower.public() if self.lower else None,
                "upper": self.upper.public() if self.upper else None}


@dataclass(frozen=True)
class NativeRectangle:
    axes: tuple[NativeAxis, ...]
    node: object = field(compare=False, repr=False)
    iterations: tuple[NativeIteration, ...] = ()

    def public(self):
        return {"axes": [axis.public() for axis in self.axes], "source_access": str(self.node),
                **({"iteration_guards": [item.public() for item in self.iterations]} if self.iterations else {})}


@dataclass(frozen=True)
class NativeResourceSections:
    resource: str
    rank: int
    lower_bounds: tuple[int | None, ...]
    reads: tuple[NativeRectangle, ...]
    writes: tuple[NativeRectangle, ...]
    overwrites: tuple[NativeRectangle, ...]
    lower_bound_expressions: tuple[NativeBound | None, ...] = ()
    descriptor_lower_bounds: bool = False
    descriptor_origin: str = "allocation"

    @property
    def dependencies(self):
        result = {}
        def collect(bound):
            if bound:
                result.update(((item.resource, item.kind), item) for item in bound.dependencies)
                for child in bound.children:
                    collect(child)
        for bound in self.lower_bound_expressions:
            collect(bound)
        for label in ("reads", "writes", "overwrites"):
            for box in getattr(self, label):
                for axis in box.axes:
                    collect(axis.lower)
                    collect(axis.upper)
                for iteration in box.iterations:
                    collect(iteration.lower)
                    collect(iteration.upper)
        return tuple(result[key] for key in sorted(result))

    def public(self):
        return {"resource": self.resource, "rank": self.rank, "logical_lower_bounds": list(self.lower_bounds),
                "empty_dimension_lower_bound": 1,
                **({("original_allocation_lower_bounds" if self.descriptor_origin == "allocation"
                     else "original_dummy_lower_bounds"): True} if self.descriptor_lower_bounds else {}),
                **({"logical_lower_bound_expressions": [bound.public() if bound else None
                                                       for bound in self.lower_bound_expressions]}
                   if self.lower_bound_expressions else {}),
                "dependencies": [item.public() for item in self.dependencies],
                **{label: [box.public() for box in getattr(self, label)]
                   for label in ("reads", "writes", "overwrites")}}


@dataclass(frozen=True)
class NativeSections:
    available: bool
    reason: str | None = None
    resources: tuple[NativeResourceSections, ...] = ()

    def public(self):
        return {"schema_version": 2, "available": self.available, "reason": self.reason,
                "coordinate_system": "original logical bounds; checked zero-based full-storage half-open mapping",
                "rectangle_limit": RECTANGLE_LIMIT, "resources": [resource.public() for resource in self.resources]}


def analyze_native_sections(analysis, routine):
    """Derive exact bounded rectangles from original assignment/DO syntax.

    Every runtime bound remains a typed, source-backed expression. An admitted
    loop is Cartesian in storage coordinates; diagonals, mutable bounds and
    unbounded strided unions remain conservative whole-resource effects.
    """
    resources = {}
    location = SourceLocation(str(routine.scope.path))
    loops = {}
    scalar_writes = set()
    operation_count = 0
    completion = getattr(analysis, "_native_sections_completion", None)
    private_roots = set(completion.private_roots) if completion is not None else set()

    def scope_for(node):
        return analysis.source_scope_for(node, routine.scope)

    def check_module_ownership(module):
        if module is None:
            return
        specification = next((node for node in _children(module.node)
                              if _kind(node) == "Specification_Part"), None)
        if any(_kind(node) == "Comment" and str(node).lstrip().lower().startswith("!$omp")
               for node in walk(specification)):
            raise CompilationError("native module OpenMP ownership requires source proof")

    def literal(node):
        if _kind(node) == "Int_Literal_Constant" and node.items[1] is None:
            return integer_literal(str(node.items[0]), location, source_token=True)
        items = _children(node)
        if len(items) == 2 and str(items[0]) in {"+", "-"} and _kind(items[1]) == "Int_Literal_Constant":
            value = literal(items[1])
            return integer_literal(str(-value if str(items[0]) == "-" else value), location)
        raise CompilationError("native section bounds require checked affine INTEGER expressions")

    def convert(item):
        if item.kind == "literal":
            integer_literal(str(item.value), location)
        return NativeBound(item.kind, item.value, item.dimension, item.resource, item.operator,
                           tuple(convert(child) for child in item.children), item.dependencies)

    def constant(item):
        if item.kind == "literal":
            return item.value
        values = tuple(constant(child) for child in item.children)
        if not values or any(value is None for value in values):
            return None
        if item.kind == "parenthesis":
            return values[0]
        if item.kind == "unary":
            value = values[0] if item.operator == "+" else -values[0]
        elif item.kind == "binary":
            left, right = values
            value = left + right if item.operator == "+" else left - right if item.operator == "-" else left * right
        else:
            return None
        return integer_literal(str(value), location)

    def typed(node, binding, dimension=1, *, declaration=False):
        if node is None:
            return None
        # Keep the original compact literal representation and default INTEGER
        # range diagnostic; the shared section parser supplies scalar authority.
        try:
            return NativeBound("literal", literal(node))
        except CompilationError:
            pass
        kind, items = _kind(node), _children(node)
        if kind == "Parenthesis":
            child = typed(items[1], binding, dimension, declaration=declaration)
            result = NativeBound("parenthesis", children=(child,), dependencies=child.dependencies)
        elif len(items) == 2 and str(items[0]) in {"+", "-"}:
            child = typed(items[1], binding, dimension, declaration=declaration)
            result = NativeBound("unary", operator=str(items[0]), children=(child,), dependencies=child.dependencies)
        elif len(items) == 3 and str(items[1]) in {"+", "-", "*"}:
            children = tuple(typed(item, binding, dimension, declaration=declaration) for item in (items[0], items[2]))
            if str(items[1]) == "*" and not any(constant(child) is not None for child in children):
                raise CompilationError("native section requires affine scalar arithmetic")
            dependencies = {(item.resource, item.kind): item for child in children for item in child.dependencies}
            result = NativeBound("binary", operator=str(items[1]), children=children,
                                 dependencies=tuple(dependencies[key] for key in sorted(dependencies)))
        elif kind == "Data_Ref":
            scalar = analysis._binding(scope_for(node), node)
            if (scalar is None or scalar.rank or scalar.dtype != "integer" or scalar.kind not in {4, 8}
                    or scalar.attributes & {"optional", "pointer", "allocatable", "volatile", "asynchronous"}):
                raise CompilationError("native section bound component requires a stable INTEGER scalar")
            dependency = SourceDependency(scalar.root, "scalar_read", scalar)
            result = NativeBound("scalar", resource=scalar.root, dependencies=(dependency,))
        else:
            if kind == "Intrinsic_Function_Reference":
                arguments = _children(node.items[1])
                candidate = next((item.items[1] if _kind(item) == "Actual_Arg_Spec" else item
                                  for item in arguments if _kind(item) != "Actual_Arg_Spec"
                                  or str(item.items[0]).lower() == "array"), None)
                other = analysis._actual_binding(scope_for(node), candidate) if candidate is not None else None
                if other is not None and other.rank:
                    binding = other
            result = convert(_bound(analysis, scope_for(node), node, binding, dimension))
        for dependency in result.dependencies:
            if (not declaration and dependency.kind == "scalar_read" and dependency.resource in scalar_writes
                    and dependency.resource not in loops):
                raise CompilationError("native section bound changes within the reached segment: " + dependency.resource)
        return result

    def external(binding):
        return (binding.name in routine.arguments or binding.root in getattr(analysis, "_captured_local_roots", ())
                or not binding.root.startswith(routine.qualified + "::") or "save" in binding.attributes)

    def resource(binding):
        if binding.root not in resources:
            identity_boundary = analysis.resource_identity_boundary(binding)
            if identity_boundary:
                raise CompilationError(identity_boundary)
            check_module_ownership(analysis.modules.get(binding.root.split("::", 1)[0]))
            if binding.attributes & {"pointer", "optional", "volatile", "asynchronous"}:
                raise CompilationError("native section association or lifetime is uncertain: " + binding.root)
            descriptor = "allocatable" in binding.attributes
            descriptor_origin = "allocation"
            if descriptor:
                proof = analysis.descriptor_stability(routine.qualified)
                stable = any(item["resource"] == binding.root and item["stable"] for item in proof["resources"])
                if not stable and binding.root not in analysis.stable_module_allocatables:
                    raise CompilationError("native allocation descriptor lifetime requires source proof: " + binding.root)
                lowers, expressions = (None,) * binding.rank, ()
            else:
                expressions = tuple(typed(node, binding, axis, declaration=True) if node is not None else None
                                    for axis, node in enumerate(binding.lower_bound_nodes, 1))
                if len(expressions) != binding.rank:
                    raise CompilationError("native section requires original declared lower bounds: " + binding.root)
                lowers = tuple(1 if item is None else constant(item) for item in expressions)
                if all(value is not None for value in lowers):
                    expressions = ()
                elif not getattr(analysis, "_native_sections_include_entry", True):
                    # A parent's specification expression ran at its original
                    # entry. Mutable scalars may already have changed before
                    # this reached segment, so never reevaluate the expression
                    # to reconstruct an existing dummy's descriptor bounds.
                    source = getattr(analysis, "_descriptor_source", analysis)
                    original = source.routines[routine.qualified]
                    if binding.name in original.arguments:
                        descriptor, descriptor_origin = True, "original_dummy"
                        lowers, expressions = (None,) * binding.rank, ()
                    else:
                        raise CompilationError("native local dynamic bounds require their original descriptor")
            resources[binding.root] = {"binding": binding, "lowers": lowers, "lower_expressions": expressions,
                                       "descriptor": descriptor, "descriptor_origin": descriptor_origin,
                                       "reads": [], "writes": [], "overwrites": []}
        return resources[binding.root]

    def replace_bound(item, root, replacement):
        if item.kind == "scalar" and item.resource == root:
            return replacement
        children = tuple(replace_bound(child, root, replacement) for child in item.children)
        dependencies = tuple(dep for dep in item.dependencies if dep.resource != root)
        dependencies += tuple(dep for dep in replacement.dependencies if dep not in dependencies)
        return NativeBound(item.kind, item.value, item.dimension, item.resource, item.operator, children, dependencies)

    def coefficient(item, root):
        if item.kind == "scalar":
            return int(item.resource == root)
        if item.kind in {"literal", "lbound", "ubound", "size"}:
            return 0
        values = tuple(coefficient(child, root) for child in item.children)
        if item.kind == "parenthesis":
            return values[0]
        if item.kind == "unary":
            return values[0] if item.operator == "+" else -values[0]
        if item.kind == "binary":
            if item.operator == "+":
                return values[0] + values[1]
            if item.operator == "-":
                return values[0] - values[1]
            left, right = (constant(child) for child in item.children)
            if left is not None:
                return left * values[1]
            if right is not None:
                return right * values[0]
        raise CompilationError("native loop subscript is not affine")

    def point_axis(item, used):
        roots = [root for root in loops if any(dep.resource == root for dep in item.dependencies)]
        if not roots:
            return (NativeAxis(item, item, True),)
        if len(roots) != 1 or roots[0] in used:
            raise CompilationError("native loop footprints require independent rectangular iterator axes")
        root = roots[0]
        used.add(root)
        iteration, values = loops[root]
        if values is not None:
            return tuple(NativeAxis(replace_bound(item, root, NativeBound("literal", value)),
                                    replace_bound(item, root, NativeBound("literal", value)), True) for value in values)
        scale = coefficient(item, root)
        if scale not in {-1, 1}:
            raise CompilationError("native runtime loop footprints require unit physical stride")
        lower, upper = ((iteration.lower, iteration.upper) if iteration.step > 0
                        else (iteration.upper, iteration.lower))
        if scale < 0:
            lower, upper = upper, lower
        return (NativeAxis(replace_bound(item, root, lower), replace_bound(item, root, upper)),)

    def reference(node):
        scope = scope_for(node)
        if _kind(node) == "Data_Ref":
            from compiler.frontend.component_bindings import component_access
            access = component_access(analysis, scope, node)
            if access is None:
                raise CompilationError("native component storage is unresolved: " + str(node))
            return access.binding, access.indices
        base = node.items[0] if _kind(node) == "Part_Ref" else node
        return analysis._binding(scope, base), tuple(_children(node.items[1])) if _kind(node) == "Part_Ref" else ()

    def rectangles(node, binding, indices):
        guards = tuple(item[0] for item in loops.values())
        if not indices:
            return (NativeRectangle(tuple(NativeAxis() for _ in range(binding.rank)), node, guards),)
        if len(indices) != binding.rank:
            raise CompilationError("native section rank differs from the original array")
        axes, used = [], set()
        for dimension, index in enumerate(indices, 1):
            if _kind(index) == "Subscript_Triplet":
                lower, upper, step = index.items
                lo, hi = typed(lower, binding, dimension), typed(upper, binding, dimension)
                if any(dep.resource in loops for item in (lo, hi) if item for dep in item.dependencies):
                    raise CompilationError("native moving array sections require a proven rectangular union")
                stride = 1 if step is None else constant(typed(step, binding, dimension))
                if stride in {None, 0}:
                    raise CompilationError("native section requires a nonzero constant stride")
                if stride == 1:
                    axes.append((NativeAxis(lo, hi),))
                else:
                    start, stop = constant(lo) if lo else None, constant(hi) if hi else None
                    if start is None or stop is None:
                        raise CompilationError("native strided sections require bounded constant endpoints")
                    values = range(start, stop + (1 if stride > 0 else -1), stride)
                    if len(values) > RECTANGLE_LIMIT:
                        raise CompilationError("native section rectangle budget exceeded: " + binding.root)
                    axes.append(tuple(NativeAxis(NativeBound("literal", value), NativeBound("literal", value), True)
                                      for value in values))
            else:
                axes.append(point_axis(typed(index, binding, dimension), used))
        count = 1
        for axis in axes:
            count *= len(axis)
        if count > RECTANGLE_LIMIT:
            raise CompilationError("native section rectangle budget exceeded: " + binding.root)
        return tuple(NativeRectangle(tuple(axis), node, guards) for axis in product(*axes))

    def add(node, action):
        binding, indices = reference(node)
        if binding is None:
            raise CompilationError("native section storage is unresolved: " + str(node))
        if binding.root in private_roots:
            return
        if not binding.rank:
            if indices:
                raise CompilationError("native section reference may be an unresolved function or indexed scalar")
            return
        if not external(binding):
            raise CompilationError("native section refinement excludes local array storage: " + binding.root)
        entry = resource(binding)
        for box in rectangles(node, binding, indices):
            if box not in entry[action]:
                entry[action].append(box)
        if len(set((*entry["reads"], *entry["writes"], *entry["overwrites"]))) > RECTANGLE_LIMIT:
            raise CompilationError("native section rectangle budget exceeded: " + binding.root)

    def expression(node):
        if node is None or isinstance(node, (str, int)):
            return
        kind = _kind(node)
        if kind in {"Part_Ref", "Function_Reference", "Intrinsic_Function_Reference"}:
            name = str(node.items[0]).lower()
            scope = scope_for(node)
            if (analysis._binding(scope, name) is None
                    and not analysis._unknown_exports(scope)
                    and analysis._candidates(scope, name) == ["$intrinsic::ieee_arithmetic::ieee_is_nan"]):
                arguments = _children(node.items[1])
                if len(arguments) != 1:
                    raise CompilationError("native IEEE_IS_NAN requires its original unary argument")
                argument = arguments[0].items[1] if _kind(arguments[0]) == "Actual_Arg_Spec" else arguments[0]
                expression(argument)
                return
        if kind in {"Name", "Part_Ref", "Data_Ref"}:
            add(node, "reads")
            return
        if kind == "Intrinsic_Function_Reference":
            function, arguments = node.items
            name, scope = str(function).lower(), scope_for(node)
            if (analysis._binding(scope, name) or analysis._candidates(scope, name)
                    or analysis._unknown_exports(scope)
                    or name not in set(INTRINSICS) | ARRAY_INQUIRIES | MODEL_INQUIRIES | {
                        "sum", "product", "any", "all", "count", "minval", "maxval"}):
                raise CompilationError("native section RHS has unresolved function effects: " + name)
            arguments = _children(arguments)
            descriptor = name in ARRAY_INQUIRIES | MODEL_INQUIRIES
            if descriptor and (not arguments or _kind(arguments[0]) not in {"Name", "Data_Ref"}
                               or any(_kind(argument) == "Actual_Arg_Spec" for argument in arguments)):
                raise CompilationError("native descriptor/model inquiries require positional whole-variable arguments")
            if descriptor and analysis._binding(scope, arguments[0]) is None:
                raise CompilationError("native descriptor/model inquiry storage is unresolved")
            for index, argument in enumerate(arguments):
                if _kind(argument) == "Actual_Arg_Spec":
                    argument = argument.items[1]
                if index != 0 or not descriptor:
                    expression(argument)
            return
        if kind.endswith("Literal_Constant"):
            return
        if kind in {"Function_Reference", "Structure_Constructor"}:
            raise CompilationError("native section RHS has unsupported function effects")
        for child in _children(node):
            expression(child)

    def statements(nodes, conditional=False):
        nonlocal operation_count
        for node in nodes:
            if _kind(node) == "Comment":
                continue
            operation_count += 1
            if operation_count > analysis.operation_limit:
                raise CompilationError("native section operation budget exceeded")
            kind = _kind(node)
            if kind == "Assignment_Stmt":
                target, _, value = node.items
                if _kind(target) not in {"Name", "Part_Ref", "Data_Ref"}:
                    raise CompilationError("native section target is not a whole array or rectangular reference")
                target_binding, _indices = reference(target)
                if target_binding is not None and target_binding.root in loops:
                    raise CompilationError("native DO iterator is assigned within its active loop")
                expression(value)
                add(target, "writes")
                if not conditional:
                    add(target, "overwrites")
            elif kind == "Block_Nonlabel_Do_Construct":
                body = [item for item in node.content if _kind(item) != "Comment"]
                header = body[0]
                control = header.items[1]
                if (control is None or control.items[0] is not None or control.items[1] is None
                        or any(value is not None for value in control.items[2:])):
                    raise CompilationError("native sections require ordinary affine counted DO loops")
                iterator_node, bounds = control.items[1]
                iterator = analysis._binding(scope_for(header), iterator_node)
                if iterator is None or iterator.rank or iterator.dtype != "integer" or iterator.kind not in {4, 8}:
                    raise CompilationError("native loop iterator requires a source-backed INTEGER scalar")
                if iterator.root in loops:
                    raise CompilationError("native nested loop iterator is redefined")
                start, stop = (typed(value, iterator) for value in bounds[:2])
                if any(dep.resource in loops for item in (start, stop) for dep in item.dependencies):
                    raise CompilationError("native loop bounds require independent Cartesian loops")
                step = 1 if len(bounds) < 3 else constant(typed(bounds[2], iterator))
                if step in {None, 0}:
                    raise CompilationError("native loop requires a nonzero constant stride")
                values = None
                if abs(step) != 1:
                    lo, hi = constant(start), constant(stop)
                    if lo is None or hi is None:
                        raise CompilationError("native strided loops require bounded constant endpoints")
                    values = range(lo, hi + (1 if step > 0 else -1), step)
                    if len(values) > RECTANGLE_LIMIT:
                        raise CompilationError("native loop rectangle budget exceeded")
                loops[iterator.root] = NativeIteration(start, stop, step), values
                statements(body[1:-1], conditional)
                del loops[iterator.root]
            elif kind in {"If_Construct", "If_Stmt"} and completion is not None:
                # Keep complete joined teams intact. Potential effects from
                # alternate arms form exact bounded unions, but are not must
                # writes: native host_begin must preserve every untouched
                # possibility before host_end may publish that union.
                if kind == "If_Stmt":
                    expression(node.items[0])
                    statements((node.items[1],), True)
                else:
                    for item in node.content:
                        label = _kind(item)
                        if label in {"If_Then_Stmt", "Else_If_Stmt"}:
                            expression(item.items[0])
                        elif label not in {"Else_Stmt", "End_If_Stmt", "Comment"}:
                            statements((item,), True)
            elif kind not in {"Continue_Stmt"}:
                raise CompilationError("native section refinement requires assignment-only affine DO leaves")

    try:
        if routine.issues:
            raise CompilationError("native section specification or internal closure is uncertain: " + "; ".join(routine.issues))
        if completion is None and any(_kind(node) == "Comment" and str(node).lstrip().lower().startswith("!$omp")
                                      for node in walk(routine.scope.node)):
            raise CompilationError("native OpenMP participation and completion require source proof")
        check_module_ownership(routine.scope.parent)
        for node in walk(routine.execution):
            if _kind(node) == "Assignment_Stmt":
                target = node.items[0]
                if _kind(target) in {"Name", "Data_Ref"}:
                    binding = analysis._binding(scope_for(target), target)
                    if binding is not None and not binding.rank:
                        scalar_writes.add(binding.root)
        specification = next((node for node in _children(routine.scope.node)
                              if _kind(node) == "Specification_Part"), None)
        for declaration in _children(specification):
            if _kind(declaration) != "Type_Declaration_Stmt":
                continue
            _, attributes, entities = declaration.items
            dimension = next((attribute.items[1] for attribute in _children(attributes)
                              if _kind(attribute) == "Dimension_Attr_Spec"), None)
            for entity in _children(entities):
                shape = entity.items[1] if entity.items[1] is not None else dimension
                axes = _children(shape)
                binding = analysis._binding(routine.scope, entity.items[0])
                if binding is not None and binding.root in private_roots:
                    # The registered completion token proves this original
                    # fixed local storage stays private to the retained native
                    # team. It is not a synthetic assumed-shape capture.
                    continue
                if str(entity.items[0]).lower() in routine.arguments and not (
                        binding is not None and "allocatable" in binding.attributes) and any(
                            _kind(axis) != "Assumed_Shape_Spec" for axis in axes):
                    raise CompilationError("native physical sections require assumed-shape array dummies")
                for axis in axes:
                    for bound_node in axis.items:
                        if bound_node is not None:
                            typed(bound_node, binding, declaration=True)
        nodes = [node for node in _children(routine.execution) if _kind(node) != "Comment"]
        if not nodes:
            raise CompilationError("native section refinement requires assignment-only affine DO leaves")
        statements(nodes)
        for entry in resources.values():
            if (getattr(analysis, "_native_sections_include_entry", True)
                    and entry["binding"].intent == "out" and entry["reads"]):
                raise CompilationError("native INTENT(OUT) reads require original-position definition hooks")
        return NativeSections(True, resources=tuple(
            NativeResourceSections(root, entry["binding"].rank, entry["lowers"],
                                   *(tuple(entry[label]) for label in ("reads", "writes", "overwrites")),
                                   entry["lower_expressions"], entry["descriptor"], entry["descriptor_origin"])
            for root, entry in sorted(resources.items())))
    except CompilationError as error:
        return NativeSections(False, str(error))
