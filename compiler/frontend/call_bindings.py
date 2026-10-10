"""Source-backed call bindings without evaluating actuals or rebasing sections.

Resolved objects borrow the original syntax during analysis. Public records are
plain values suitable for summary caches; they never contain parser objects or
claim physical transfer coordinates without a runtime descriptor proof.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from compiler.ir import CompilationError, SourceLocation
from compiler.ir.integers import integer_literal


def _kind(node):
    return type(node).__name__


def _children(node):
    return getattr(node, "content", getattr(node, "items", ())) or ()


@dataclass(frozen=True)
class SourceDependency:
    resource: str
    kind: str
    binding: object = field(compare=False, repr=False)

    def public(self):
        return {"resource": self.resource, "kind": self.kind}


@dataclass(frozen=True)
class SourceSectionBound:
    kind: str
    expression: str
    node: object = field(compare=False, repr=False)
    value: int | None = None
    resource: str | None = None
    dimension: int | None = None
    operator: str | None = None
    children: tuple[SourceSectionBound, ...] = ()
    dependencies: tuple[SourceDependency, ...] = ()

    def public(self):
        return {"kind": self.kind, "expression": self.expression,
                **({"value": self.value} if self.value is not None else {}),
                **({"resource": self.resource} if self.resource is not None else {}),
                **({"dimension": self.dimension} if self.dimension is not None else {}),
                **({"operator": self.operator} if self.operator is not None else {}),
                **({"children": [child.public() for child in self.children]} if self.children else {}),
                "dependencies": [dependency.public() for dependency in self.dependencies]}


@dataclass(frozen=True)
class SourceSectionAxis:
    lower: SourceSectionBound
    upper: SourceSectionBound
    scalar: bool = False

    def public(self):
        return {"kind": "scalar_coordinate" if self.scalar else "unit_stride_range",
                "lower": self.lower.public(), "upper": self.upper.public(), "stride": 1}


@dataclass(frozen=True)
class RectangularSection:
    resource: str
    axes: tuple[SourceSectionAxis, ...]
    node: object = field(compare=False, repr=False)

    @property
    def logical_rank(self):
        return sum(not axis.scalar for axis in self.axes)

    @property
    def dependencies(self):
        values = {}
        for axis in self.axes:
            for bound in (axis.lower, axis.upper):
                values.update(((item.resource, item.kind), item) for item in bound.dependencies)
        return tuple(values[key] for key in sorted(values))

    def public(self):
        return {"resource": self.resource, "rank": len(self.axes), "logical_rank": self.logical_rank,
                "retained_axes": [axis for axis, item in enumerate(self.axes) if not item.scalar],
                "source_access": str(self.node),
                "coordinate_system": "original caller logical indices; physical mapping requires runtime descriptors",
                "axes": [axis.public() for axis in self.axes],
                "dependencies": [dependency.public() for dependency in self.dependencies],
                "bounds_evaluation": "at the original call under its original guards"}


@dataclass(frozen=True)
class SourceActualMapping:
    formal: str
    actual: object = field(compare=False, repr=False)
    formal_binding: object = field(compare=False, repr=False)
    binding: object = field(default=None, compare=False, repr=False)
    section: RectangularSection | None = None
    presence: str = "supplied"
    source_object: object = field(default=None, compare=False, repr=False)
    canonical_resource: str | None = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "canonical_resource", self.binding.root if self.binding is not None else None)

    @property
    def resource(self):
        return self.canonical_resource

    def public(self):
        storage = ("omitted" if self.actual is None else "rectangle" if self.section is not None else
                   "whole" if self.binding is not None else "expression")
        return {"formal": self.formal, "formal_resource": "argument::" + self.formal,
                "actual": str(self.actual) if self.actual is not None else None,
                "resource": self.resource, "storage": storage, "presence": self.presence,
                "formal_descriptor": self.formal_binding.public(),
                "actual_descriptor": self.binding.public() if self.binding is not None else None,
                "section": self.section.public() if self.section is not None else None,
                **({"source_object": self.source_object.public()} if self.source_object is not None else {}),
                "requirements": {"presence_preserving_forwarding": self.presence == "forwarded_optional",
                                 "descriptor_dependent_presence": self.presence in {"allocation_dependent", "association_dependent"},
                                 "original_actual_presence": bool(self.binding is not None
                                                                  and "optional" in self.binding.attributes),
                                 "original_allocation_descriptor": "allocatable" in self.formal_binding.attributes,
                                 "physical_section_mapping": self.section is not None}}


@dataclass(frozen=True)
class SourceCallArgument:
    position: int
    formal: str
    keyword: str | None
    actual: object = field(compare=False, repr=False)

    def public(self):
        return {"position": self.position, "formal": self.formal, "keyword": self.keyword,
                "actual": str(self.actual)}


@dataclass(frozen=True)
class ResolvedSourceCall:
    procedure: str
    formals: tuple[str, ...]
    actuals: tuple[object | None, ...] = field(compare=False, repr=False)
    mappings: tuple[SourceActualMapping, ...]
    original_arguments: tuple[SourceCallArgument, ...]
    target: str
    source_kind: str = "module"

    @property
    def bindings(self):
        return {"argument::" + item.formal: item.binding for item in self.mappings if item.binding is not None}

    @property
    def resource_mapping(self):
        return {"argument::" + item.formal: item.resource for item in self.mappings if item.resource is not None}

    def render_original_arguments(self, values=None):
        """Keep source keyword spelling/order while substituting formal values."""
        replacements = dict(zip(self.formals, values, strict=True)) if values is not None else None
        return tuple((item.keyword + " = " if item.keyword is not None else "") +
                     (str(replacements[item.formal]) if replacements is not None else str(item.actual))
                     for item in self.original_arguments)

    def public(self):
        return {"schema_version": 1, "procedure": self.procedure, "target": self.target, "source_kind": self.source_kind,
                "formal_arguments": list(self.formals),
                "actual_arguments": [str(actual) if actual is not None else None for actual in self.actuals],
                "original_arguments": [argument.public() for argument in self.original_arguments],
                "resource_mapping": self.resource_mapping,
                "resource_mappings": [mapping.public() for mapping in self.mappings],
                "actual_evaluation": "original call position and guards; no analysis-time evaluation"}


def _constant_bound(bound):
    if bound.kind == "literal":
        return True
    return (bound.kind in {"unary", "parenthesis", "binary"} and bool(bound.children)
            and all(_constant_bound(child) for child in bound.children))


def _dependencies(children):
    values = {(item.resource, item.kind): item for child in children for item in child.dependencies}
    return tuple(values[key] for key in sorted(values))


def _check_resource_identity(analysis, binding):
    reason = analysis.resource_identity_boundary(binding)
    if reason:
        raise CompilationError(reason)


def _bound(analysis, scope, node, binding, dimension, default=None):
    if node is None:
        dependency = SourceDependency(binding.root, "descriptor_read", binding)
        return SourceSectionBound(default, default + "(" + binding.name + "," + str(dimension) + ")", None,
                                  resource=binding.root, dimension=dimension, dependencies=(dependency,))
    kind, items = _kind(node), _children(node)
    location = SourceLocation(str(scope.path))
    if kind == "Int_Literal_Constant":
        width = 4
        if node.items[1] is not None:
            width = scope.kinds.integer(node.items[1], location)
        if width not in {4, 8}:
            raise CompilationError("section literal requires supported INTEGER kind")
        value = int(node.items[0])
        if not 0 <= value < 2 ** (width * 8 - 1):
            raise CompilationError("section INTEGER literal is out of range")
        return SourceSectionBound("literal", str(node), node, value=value)
    if kind in {"Name", "Data_Ref"}:
        scalar = analysis._binding(scope, node)
        if scalar is None or scalar.rank or scalar.dtype != "integer" or scalar.kind not in {4, 8}:
            raise CompilationError("section bound requires a source-backed INTEGER scalar: " + str(node))
        _check_resource_identity(analysis, scalar)
        if "parameter" in scalar.attributes:
            value = scope.kinds.integer(node, location)
            integer_literal(str(value), location)
            return SourceSectionBound("literal", str(node), node, value=value, resource=scalar.root)
        if scalar.attributes & {"optional", "pointer", "allocatable", "volatile", "asynchronous"}:
            raise CompilationError("section bound scalar presence or association is uncertain: " + str(node))
        dependency = SourceDependency(scalar.root, "scalar_read", scalar)
        return SourceSectionBound("scalar", str(node), node, resource=scalar.root, dependencies=(dependency,))
    if kind == "Parenthesis":
        child = _bound(analysis, scope, items[1], binding, dimension)
        return SourceSectionBound("parenthesis", str(node), node, children=(child,), dependencies=child.dependencies)
    if kind == "Intrinsic_Function_Reference":
        name, arguments = str(node.items[0]).lower(), _children(node.items[1])
        if (name not in {"lbound", "ubound", "size"} or len(arguments) != 2
                or analysis._binding(scope, name) or analysis._candidates(scope, name) or analysis._unknown_exports(scope)):
            raise CompilationError("section bounds require source-backed scalar arithmetic or array inquiries")
        values, position, keyword_seen = {}, 0, False
        for argument in arguments:
            if _kind(argument) == "Actual_Arg_Spec":
                keyword_seen = True
                keyword, value = str(argument.items[0]).lower(), argument.items[1]
            else:
                if keyword_seen or position >= 2:
                    raise CompilationError("section inquiry requires ARRAY and DIM bindings")
                keyword, value = ("array", "dim")[position], argument
                position += 1
            if keyword not in {"array", "dim"} or keyword in values:
                raise CompilationError("section inquiry requires ARRAY and DIM bindings")
            values[keyword] = value
        if set(values) != {"array", "dim"}:
            raise CompilationError("section inquiry requires ARRAY and DIM bindings")
        arguments = values["array"], values["dim"]
        array = analysis._actual_binding(scope, arguments[0])
        if array is not None:
            _check_resource_identity(analysis, array)
        if array is None or array.root != binding.root or not array.rank:
            raise CompilationError("section inquiry must use the original section array")
        axis = scope.kinds.integer(arguments[1], location)
        if not 1 <= axis <= array.rank:
            raise CompilationError("section inquiry dimension is out of range")
        dependency = SourceDependency(array.root, "descriptor_read", array)
        return SourceSectionBound(name, str(node), node, resource=array.root, dimension=axis, dependencies=(dependency,))
    if len(items) == 2 and str(items[0]) in {"+", "-"}:
        child = _bound(analysis, scope, items[1], binding, dimension)
        return SourceSectionBound("unary", str(node), node, operator=str(items[0]), children=(child,),
                                  dependencies=child.dependencies)
    if len(items) == 3 and str(items[1]) in {"+", "-", "*"}:
        children = (_bound(analysis, scope, items[0], binding, dimension),
                    _bound(analysis, scope, items[2], binding, dimension))
        if str(items[1]) == "*" and not any(_constant_bound(child) for child in children):
            raise CompilationError("section bounds require affine scalar arithmetic")
        return SourceSectionBound("binary", str(node), node, operator=str(items[1]), children=children,
                                  dependencies=_dependencies(children))
    raise CompilationError("section bounds require affine scalar arithmetic: " + str(node))


def _actual(analysis, scope, node):
    binding = analysis._actual_binding(scope, node)
    if binding is not None:
        _check_resource_identity(analysis, binding)
        return binding.signature(), binding, None
    if _kind(node) in {"Part_Ref", "Data_Ref"}:
        if _kind(node) == "Data_Ref":
            from compiler.frontend.component_bindings import component_access
            access = component_access(analysis, scope, node)
            binding, indices = (access.binding, access.indices) if access is not None else (None, ())
        else:
            binding = analysis._binding(scope, node.items[0])
            indices = tuple(_children(node.items[1]))
        if binding is not None and binding.rank:
            _check_resource_identity(analysis, binding)
            if len(indices) != binding.rank:
                raise CompilationError("array-element/section actual requires in-place mapping and coherence: " + str(node))
            axes = []
            for dimension, index in enumerate(indices, 1):
                if _kind(index) != "Subscript_Triplet":
                    coordinate = _bound(analysis, scope, index, binding, dimension)
                    axes.append(SourceSectionAxis(coordinate, coordinate, scalar=True))
                    continue
                lower, upper, stride = index.items
                if stride is not None and scope.kinds.integer(stride, SourceLocation(str(scope.path))) != 1:
                    raise CompilationError("source rectangular actual requires unit stride: " + str(node))
                axes.append(SourceSectionAxis(_bound(analysis, scope, lower, binding, dimension, "lbound"),
                                              _bound(analysis, scope, upper, binding, dimension, "ubound")))
            section = RectangularSection(binding.root, tuple(axes), node)
            if not section.logical_rank:
                raise CompilationError("array-element actual requires scalar payload coherence: " + str(node))
            return (*binding.signature()[:2], section.logical_rank), binding, section
    reason = analysis._actual_mapping_boundary(scope, node)
    if reason:
        raise CompilationError(reason)
    return analysis._signature(scope, node), None, None


def _arrange(callee, original):
    values, arguments, keyword_seen, position = {}, [], False, 0
    for original_position, item in enumerate(original):
        keyword = str(item.items[0]) if _kind(item) == "Actual_Arg_Spec" else None
        actual = item.items[1] if keyword is not None else item
        if keyword is not None:
            keyword_seen = True
            formal = keyword.lower()
            if formal not in callee.arguments:
                raise CompilationError("unknown source-call keyword: " + keyword)
        else:
            if keyword_seen:
                raise CompilationError("positional source-call argument follows a keyword")
            if position >= len(callee.arguments):
                raise CompilationError("too many source-call arguments")
            formal = callee.arguments[position]
            position += 1
        if formal in values:
            raise CompilationError("duplicate source-call argument: " + formal)
        values[formal] = actual
        arguments.append(SourceCallArgument(original_position, formal, keyword, actual))
    for formal in callee.arguments:
        binding = callee.scope.bindings.get(formal)
        if binding is None:
            raise CompilationError("source-call formal declaration is unavailable: " + formal)
        if formal not in values and "optional" not in binding.attributes:
            raise CompilationError("required source-call argument is missing: " + formal)
    return tuple(values.get(formal) for formal in callee.arguments), tuple(arguments)


def resolve_source_call(analysis, scope, call_node):
    """Resolve signatures, keywords and source rectangles without evaluating them."""
    target, arguments = call_node.items
    if _kind(target) != "Name":
        raise CompilationError("indirect calls are scope boundaries")
    original = tuple(_children(arguments))
    matches, errors = [], []
    candidates = analysis._candidates(scope, target)
    unavailable = [procedure for procedure in candidates if procedure not in analysis.routines]
    if unavailable:
        raise CompilationError("source call interface closure unavailable: " + ", ".join(unavailable))
    for procedure in candidates:
        callee = analysis.routines[procedure]
        try:
            if callee.source_kind == "external":
                interface = analysis.external_interface(callee)
                if not interface["available"]:
                    raise CompilationError(interface["reason"] + ": " + procedure)
                if any(_kind(item) == "Actual_Arg_Spec" for item in original):
                    raise CompilationError("external keyword arguments require a proven explicit interface: " + procedure)
            actuals, original_arguments = _arrange(callee, original)
            mappings = []
            for formal, actual in zip(callee.arguments, actuals, strict=True):
                declaration = callee.scope.bindings[formal]
                from compiler.frontend.component_bindings import _typename
                derived = _typename(declaration.dtype) is not None
                if actual is None:
                    if derived:
                        raise CompilationError("source object requires fixed nonoptional scalar original storage")
                    mappings.append(SourceActualMapping(formal, None, declaration, presence="omitted"))
                    continue
                signature, binding, section = _actual(analysis, scope, actual)
                source_object = None
                if derived:
                    from compiler.frontend.source_objects import source_object_mapping
                    source_object = source_object_mapping(analysis, declaration, binding, actual, call_node, scope)
                elif (signature is None or None in (declaration.kind, signature[1])
                        or declaration.signature() != signature):
                    raise CompilationError("source-call type, kind or rank mismatch: " + formal)
                if binding is None and (declaration.intent in {"out", "inout"} or "allocatable" in declaration.attributes):
                    raise CompilationError("source-call writable or descriptor actual requires original storage: " + formal)
                presence = ("forwarded_optional" if binding is not None and section is None
                            and "optional" in binding.attributes else "supplied")
                if presence == "forwarded_optional" and "optional" not in declaration.attributes:
                    raise CompilationError("optional actual requires an optional callee formal: " + formal)
                if binding is not None and section is None and presence == "supplied" and "optional" in declaration.attributes:
                    if "allocatable" in binding.attributes and "allocatable" not in declaration.attributes:
                        presence = "allocation_dependent"
                    elif "pointer" in binding.attributes and "pointer" not in declaration.attributes:
                        presence = "association_dependent"
                if "allocatable" in declaration.attributes and (binding is None or section is not None
                                                                or "allocatable" not in binding.attributes):
                    raise CompilationError("allocatable callee formal requires its original allocation descriptor: " + formal)
                mappings.append(SourceActualMapping(formal, actual, declaration, binding, section, presence, source_object))
            matches.append(ResolvedSourceCall(procedure, callee.arguments, actuals, tuple(mappings),
                                              original_arguments, str(target), callee.source_kind))
        except CompilationError as error:
            errors.append(error)
    if len(matches) != 1:
        if not matches and errors and len({str(error) for error in errors}) == 1:
            raise errors[0]
        raise CompilationError("source call is unresolved or ambiguous: " + str(target))
    return matches[0]
