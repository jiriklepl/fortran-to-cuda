"""Bounded native sections from original ASTs, separate from whole effects.

Only straight-line assignments are refined. The original native computation is
retained; these facts describe its externally visible array accesses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fparser.two.utils import walk

from compiler.ir import CompilationError, SourceLocation
from compiler.ir.integers import integer_literal
from compiler.ir.intrinsics import ARRAY_INQUIRIES, INTRINSICS, MODEL_INQUIRIES

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

    def public(self):
        return {"kind": self.kind, **({"value": self.value} if self.value is not None else {}),
                **({"dimension": self.dimension} if self.dimension is not None else {})}


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

    def public(self):
        return {"axes": [axis.public() for axis in self.axes], "source_access": str(self.node)}


@dataclass(frozen=True)
class NativeResourceSections:
    resource: str
    rank: int
    lower_bounds: tuple[int, ...]
    reads: tuple[NativeRectangle, ...]
    writes: tuple[NativeRectangle, ...]
    overwrites: tuple[NativeRectangle, ...]

    def public(self):
        return {"resource": self.resource, "rank": self.rank, "logical_lower_bounds": list(self.lower_bounds),
                "empty_dimension_lower_bound": 1,
                **{label: [box.public() for box in getattr(self, label)]
                   for label in ("reads", "writes", "overwrites")}}


@dataclass(frozen=True)
class NativeSections:
    available: bool
    reason: str | None = None
    resources: tuple[NativeResourceSections, ...] = ()

    def public(self):
        return {"schema_version": 1, "available": self.available, "reason": self.reason,
                "coordinate_system": "original logical bounds; checked zero-based full-storage half-open mapping",
                "rectangle_limit": RECTANGLE_LIMIT, "resources": [resource.public() for resource in self.resources]}


def analyze_native_sections(analysis, routine):
    """Retain typed references from original Name/Part_Ref assignment sites."""
    resources = {}
    location = SourceLocation(str(routine.scope.path))

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
        raise CompilationError("native section bounds require default INTEGER literals or checked same-array inquiries")

    def external(binding):
        return (binding.name in routine.arguments or not binding.root.startswith(routine.qualified + "::")
                or "save" in binding.attributes)

    def resource(binding):
        if binding.root not in resources:
            check_module_ownership(analysis.modules.get(binding.root.split("::", 1)[0]))
            if binding.attributes & {"pointer", "allocatable", "optional", "volatile", "asynchronous"}:
                raise CompilationError("native section association or lifetime is uncertain: " + binding.root)
            lowers = tuple(1 if node is None else literal(node) for node in binding.lower_bound_nodes)
            if len(lowers) != binding.rank:
                raise CompilationError("native section requires original declared lower bounds: " + binding.root)
            resources[binding.root] = {"binding": binding, "lowers": lowers, "reads": [], "writes": [], "overwrites": []}
        return resources[binding.root]

    def bound(node, binding, dimension):
        if node is None:
            return None
        if _kind(node) == "Intrinsic_Function_Reference":
            function, arguments = node.items
            name = str(function).lower()
            args = _children(arguments)
            if (name not in {"lbound", "ubound", "size"} or len(args) != 2 or _kind(args[0]) != "Name"
                    or analysis._binding(routine.scope, name) or analysis._candidates(routine.scope, name)
                    or analysis._unknown_exports(routine.scope)):
                raise CompilationError("native section requires a checked same-array literal-dimension inquiry")
            other = analysis._binding(routine.scope, args[0])
            axis = literal(args[1])
            if other is None or other.root != binding.root or axis != dimension:
                raise CompilationError("native section inquiry must use its own array and dimension")
            if name == "size" and resource(binding)["lowers"][dimension-1] != 1:
                raise CompilationError("native section SIZE bounds require declared lower bound one")
            return NativeBound(name, dimension=dimension)
        return NativeBound("literal", literal(node))

    def rectangle(node, binding):
        if _kind(node) == "Name":
            return NativeRectangle(tuple(NativeAxis() for _ in range(binding.rank)), node)
        indices = _children(node.items[1])
        if len(indices) != binding.rank:
            raise CompilationError("native section rank differs from the original array")
        axes = []
        for dimension, index in enumerate(indices, 1):
            if _kind(index) == "Subscript_Triplet":
                lower, upper, step = index.items
                if step is not None and literal(step) != 1:
                    raise CompilationError("native section requires unit stride")
                axes.append(NativeAxis(bound(lower, binding, dimension), bound(upper, binding, dimension)))
            else:
                point = bound(index, binding, dimension)
                axes.append(NativeAxis(point, point, True))
        return NativeRectangle(tuple(axes), node)

    def add(node, action):
        base = node.items[0] if _kind(node) == "Part_Ref" else node
        binding = analysis._binding(routine.scope, base)
        if binding is None:
            raise CompilationError("native section storage is unresolved: " + str(base))
        if not binding.rank:
            if _kind(node) == "Part_Ref":
                raise CompilationError("native section reference may be an unresolved function or indexed scalar")
            return
        if not external(binding):
            raise CompilationError("native section refinement excludes local array storage: " + binding.root)
        entry = resource(binding)
        box = rectangle(node, binding)
        if box not in entry[action]:
            entry[action].append(box)
        if len(set((*entry["reads"], *entry["writes"], *entry["overwrites"]))) > RECTANGLE_LIMIT:
            raise CompilationError("native section rectangle budget exceeded: " + binding.root)

    def expression(node):
        if node is None or isinstance(node, (str, int)):
            return
        kind = _kind(node)
        if kind in {"Name", "Part_Ref"}:
            add(node, "reads")
            return
        if kind == "Intrinsic_Function_Reference":
            function, arguments = node.items
            name = str(function).lower()
            if (analysis._binding(routine.scope, name) or analysis._candidates(routine.scope, name)
                    or analysis._unknown_exports(routine.scope)
                    or name not in set(INTRINSICS) | ARRAY_INQUIRIES | MODEL_INQUIRIES | {
                        "sum", "product", "any", "all", "count", "minval", "maxval"}):
                raise CompilationError("native section RHS has unresolved function effects: " + name)
            arguments = _children(arguments)
            descriptor = name in ARRAY_INQUIRIES | MODEL_INQUIRIES
            if descriptor and (not arguments or _kind(arguments[0]) != "Name"
                               or any(_kind(argument) == "Actual_Arg_Spec" for argument in arguments)):
                # A section/expression argument can evaluate payload indices;
                # skipping it would omit reads needed before descriptor work.
                raise CompilationError("native descriptor/model inquiries require positional whole-variable arguments")
            if descriptor and analysis._binding(routine.scope, arguments[0]) is None:
                raise CompilationError("native descriptor/model inquiry storage is unresolved")
            for index, argument in enumerate(arguments):
                if _kind(argument) == "Actual_Arg_Spec":
                    argument = argument.items[1]
                if index == 0 and descriptor:
                    continue
                expression(argument)
            return
        if kind.endswith("Literal_Constant"):
            return
        if kind in {"Data_Ref", "Function_Reference", "Structure_Constructor"}:
            raise CompilationError("native section RHS has unsupported component or function effects")
        for child in _children(node):
            expression(child)

    try:
        if routine.issues:
            raise CompilationError("native section specification or internal closure is uncertain: " + "; ".join(routine.issues))
        if any(_kind(node) == "Comment" and str(node).lstrip().lower().startswith("!$omp")
               for node in walk(routine.scope.node)):
            raise CompilationError("native OpenMP participation and completion require source proof")
        check_module_ownership(routine.scope.parent)
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
                if str(entity.items[0]).lower() in routine.arguments and any(
                        _kind(axis) != "Assumed_Shape_Spec" for axis in axes):
                    raise CompilationError("native physical sections require assumed-shape array dummies")
                for axis in axes:
                    for bound_node in axis.items:
                        if bound_node is not None:
                            # Specification expressions run at native entry,
                            # before the assignments whose effects we refine.
                            literal(bound_node)
        nodes = [node for node in _children(routine.execution) if _kind(node) != "Comment"]
        if len(nodes) > analysis.operation_limit:
            raise CompilationError("native section assignment budget exceeded")
        if not nodes or any(_kind(node) != "Assignment_Stmt" for node in nodes):
            raise CompilationError("native section refinement requires straight-line assignment-only leaves")
        for node in nodes:
            target, _, value = node.items
            if _kind(target) not in {"Name", "Part_Ref"}:
                raise CompilationError("native section target is not a whole array or rectangular reference")
            expression(value)
            add(target, "writes")
            add(target, "overwrites")
        for entry in resources.values():
            if entry["binding"].intent == "out" and entry["reads"]:
                raise CompilationError("native INTENT(OUT) reads require original-position definition hooks")
        return NativeSections(True, resources=tuple(
            NativeResourceSections(root, entry["binding"].rank, entry["lowers"],
                                   *(tuple(entry[label]) for label in ("reads", "writes", "overwrites")))
            for root, entry in sorted(resources.items())))
    except CompilationError as error:
        return NativeSections(False, str(error))
