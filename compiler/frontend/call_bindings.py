"""Source-backed call bindings without evaluating actuals or rebasing sections.

Resolved objects borrow the original syntax during analysis. Public records are
plain values suitable for summary caches; they never contain parser objects or
claim physical transfer coordinates without a runtime descriptor proof.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType

from compiler.ir import CompilationError, SourceLocation
from compiler.ir.integers import INTEGER_MAX, INTEGER_MIN, integer_literal


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
class SourceScalarElementActual:
    """Original native scalar association, never an array resource mapping."""
    binding: object = field(compare=False, repr=False)
    actual: object = field(compare=False, repr=False)
    index: SourceSectionBound
    bounds: tuple[int, int]
    initializer: object = field(compare=False, repr=False)
    declaration: object = field(compare=False, repr=False)
    declaration_work: int = field(compare=False, repr=False)

    def public(self):
        return {"resource": self.binding.root, "actual": str(self.actual),
                "backing_descriptor": self.binding.public(), "index": self.index.public(),
                "bounds": list(self.bounds), "association": "original CALL element association",
                "declaration_initializer": str(self.initializer),
                "declaration_scan_work": self.declaration_work,
                "payload_evaluation": "original scalar dummy uses; no call-entry value snapshot",
                "managed_storage_authority": False, "definition_authority": False}


@dataclass(frozen=True)
class SourceActualMapping:
    formal: str
    actual: object = field(compare=False, repr=False)
    formal_binding: object = field(compare=False, repr=False)
    binding: object = field(default=None, compare=False, repr=False)
    section: RectangularSection | None = None
    presence: str = "supplied"
    source_object: object = field(default=None, compare=False, repr=False)
    scalar_element: SourceScalarElementActual | None = None
    canonical_resource: str | None = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "canonical_resource", self.binding.root
                           if self.binding is not None and self.scalar_element is None else None)

    @property
    def resource(self):
        return self.canonical_resource

    def public(self):
        storage = ("omitted" if self.actual is None else "scalar_element" if self.scalar_element is not None else
                   "rectangle" if self.section is not None else
                   "whole" if self.binding is not None else "expression")
        return {"formal": self.formal, "formal_resource": "argument::" + self.formal,
                "actual": str(self.actual) if self.actual is not None else None,
                "resource": self.resource, "storage": storage, "presence": self.presence,
                "formal_descriptor": self.formal_binding.public(),
                "actual_descriptor": self.binding.public() if self.binding is not None else None,
                "section": self.section.public() if self.section is not None else None,
                **({"source_object": self.source_object.public()} if self.source_object is not None else {}),
                **({"scalar_element": self.scalar_element.public()} if self.scalar_element is not None else {}),
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
        return {"argument::" + item.formal: item.binding for item in self.mappings
                if item.binding is not None and item.scalar_element is None}

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
        return {"schema_version": 2, "procedure": self.procedure, "target": self.target, "source_kind": self.source_kind,
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
            return (*binding.signature()[:2], section.logical_rank), binding, section
    reason = analysis._actual_mapping_boundary(scope, node)
    if reason:
        raise CompilationError(reason)
    return analysis._signature(scope, node), None, None


def _arrange(callee, original, *, function=False):
    values, arguments, keyword_seen, position = {}, [], False, 0
    for original_position, item in enumerate(original):
        keyword = str(item.items[0]) if _kind(item) in ({"Actual_Arg_Spec", "Component_Spec"}
                                                       if function else {"Actual_Arg_Spec"}) else None
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
    return _resolve_source_invocation(analysis, scope, call_node)


def resolve_source_function(analysis, scope, expression):
    """Associate an original scalar function expression, without granting GPU legality."""
    return _resolve_source_invocation(analysis, scope, expression, function=True)


def _fixed_array_bounds(analysis, binding):
    """Prove bounded original explicit shape; never evaluate runtime bounds."""
    from compiler.frontend.source_functions import _validate_scope

    scope = binding.declaring_scope
    if (scope is None or scope.bindings.get(binding.name) is not binding
            or len(binding.shape_nodes) != binding.rank):
        raise CompilationError("native source array requires its original fixed shape")
    _validate_scope(analysis, scope)
    location, bounds, total = SourceLocation(str(scope.path)), [], 1
    for axis in binding.shape_nodes:
        if _kind(axis) != "Explicit_Shape_Spec":
            raise CompilationError("native source array requires original constant explicit bounds")
        try:
            lower = 1 if axis.items[0] is None else scope.kinds.integer(axis.items[0], location)
            upper = scope.kinds.integer(axis.items[1], location)
        except CompilationError as error:
            raise CompilationError("native source array requires original constant explicit bounds") from error
        extent = max(0, upper - lower + 1)
        if not (INTEGER_MIN <= lower <= INTEGER_MAX and INTEGER_MIN <= upper <= INTEGER_MAX
                and extent <= INTEGER_MAX):
            raise CompilationError("native source array bounds exceed the INTEGER ABI")
        total *= extent
        if total > 256:
            raise CompilationError("native source array exceeds the 256-element fixed storage budget")
        bounds.append((lower, upper))
    return tuple(bounds)


def _function_array_mapping(analysis, declaration, binding, section):
    formal_bounds = _fixed_array_bounds(analysis, declaration)
    if binding is None or binding.attributes & {
            "optional", "pointer", "allocatable", "target", "value", "volatile", "asynchronous"}:
        raise CompilationError("native source function array actual requires original fixed unaliased storage")
    actual_bounds = _fixed_array_bounds(analysis, binding)
    if section is not None:
        # The existing rectangular mapping retains caller coordinates and
        # rank-reducing axes. Only a fully constant, in-bounds view is admitted;
        # no temporary, sequence association or runtime bound is invented.
        def constant(bound):
            if bound.kind == "literal":
                value = bound.value
            else:
                values = tuple(constant(child) for child in bound.children)
                if bound.kind == "parenthesis":
                    value = values[0]
                elif bound.kind == "unary":
                    value = -values[0] if bound.operator == "-" else values[0]
                else:
                    left, right = values
                    value = {"+": left + right, "-": left - right, "*": left * right}[bound.operator]
            return integer_literal(str(value), SourceLocation(str(binding.declaring_scope.path)))

        selected = []
        for axis, (lower, upper) in zip(section.axes, actual_bounds, strict=True):
            values = []
            for bound in (axis.lower, axis.upper):
                if bound.kind in {"lbound", "ubound"} and bound.node is None:
                    value = lower if bound.kind == "lbound" else upper
                elif bound.node is not None and _constant_bound(bound):
                    # A PARAMETER belongs to the caller's lexical scope, which
                    # may differ from the array's defining module. _bound has
                    # already resolved it there; do not rebind its spelling.
                    value = constant(bound)
                else:
                    raise CompilationError("native source function array section requires fixed bounds")
                if not INTEGER_MIN <= value <= INTEGER_MAX:
                    raise CompilationError("native source function section bounds exceed the INTEGER ABI")
                values.append(value)
            lo, hi = values
            if lo < lower or lo > upper or hi < lower or hi > upper:
                raise CompilationError("native source function array section is outside original bounds")
            if not axis.scalar:
                selected.append((lo, hi))
        actual_bounds = tuple(selected)
    if tuple(max(0, hi - lo + 1) for lo, hi in formal_bounds) != tuple(
            max(0, hi - lo + 1) for lo, hi in actual_bounds):
        raise CompilationError("native source function array shape mismatch")


def _scalar_element_storage_scope(analysis, scope):
    from compiler.frontend.source_effects import _part
    from compiler.frontend.source_functions import _validate_scope, _walk

    _validate_scope(analysis, scope)
    visits = 0
    for node in _walk(_part(scope.node, "Specification_Part")):
        visits += 1
        if (type(node).__name__ in {"Equivalence_Stmt", "Common_Stmt", "Cray_Pointer_Stmt"}
                or type(node).__name__ == "Comment" and str(node).lstrip().lower().startswith("!$omp")):
            raise CompilationError("scalar element storage requires no original storage association or OpenMP directives")
    return visits


def _scalar_element_actual(analysis, scope, actual, binding, section, formal):
    from compiler.frontend.source_effects import _part
    from compiler.frontend.source_functions import _validate_resolution, _walk

    forbidden = {"pointer", "target", "allocatable", "optional", "value", "volatile", "asynchronous", "parameter"}
    owner = binding.declaring_scope
    if (type(actual).__name__ != "Part_Ref" or binding.rank != 1 or binding.dtype != "real"
            or binding.kind not in {4, 8} or binding.attributes & forbidden
            or owner is None or analysis.modules.get(owner.module) is not owner
            or formal.rank or formal.intent != "in" or formal.attributes & (forbidden - {"parameter"})):
        raise CompilationError("scalar element requires fixed module REAL storage and required scalar INTENT(IN)")
    index = section.axes[0].lower
    syntax = index.node
    if type(syntax).__name__ == "Name":
        selector = analysis._binding(scope, syntax)
        if (selector is None or selector.rank or selector.dtype != "integer" or selector.kind != 4
                or selector.attributes & forbidden or analysis.resource_identity_boundary(selector)
                or selector.declaring_scope is None):
            raise CompilationError("scalar element selector requires ordinary default INTEGER storage")
        selector_work = _scalar_element_storage_scope(analysis, selector.declaring_scope)
    elif not (index.kind == "literal" or index.kind == "unary" and len(index.children) == 1
              and index.children[0].kind == "literal"):
        raise CompilationError("scalar element selector requires an original INTEGER Name or literal")
    else:
        selector_work = 0
    if analysis._signature(scope, syntax) != ("integer", 4, 0):
        raise CompilationError("scalar element selector requires default INTEGER kind")
    bounds, = _fixed_array_bounds(analysis, binding)
    spec = _part(owner.node, "Specification_Part")
    initializer, declaration = None, None
    visits = _scalar_element_storage_scope(analysis, owner) + selector_work
    for node in _children(spec):
        visits += 1
        if visits > 65536:
            raise CompilationError("scalar element declaration scan budget exhausted")
        if type(node).__name__ != "Type_Declaration_Stmt":
            continue
        for entity in _children(node.items[2]):
            visits += 1
            if visits > 65536:
                raise CompilationError("scalar element declaration scan budget exhausted")
            if str(entity.items[0]).lower() == binding.name:
                initializer, declaration = entity.items[3], node
                shape = entity.items[1]
                if shape is None:
                    shape = next((attribute.items[1] for attribute in _children(node.items[1])
                                  if type(attribute).__name__ == "Dimension_Attr_Spec"), None)
                axes = tuple(_children(shape))
                if (len(axes) != 1 or binding.shape_nodes[0] is not axes[0]
                        or binding.lower_bound_nodes[0] is not axes[0].items[0]):
                    raise CompilationError("scalar element original declaration bounds authority changed")
                break
        if declaration is not None:
            break
    if initializer is None or type(initializer).__name__ != "Initialization" or str(initializer.items[0]) != "=":
        raise CompilationError("scalar element backing module table requires original declaration initialization")
    _validate_resolution(analysis, owner, _walk(initializer))
    return SourceScalarElementActual(binding, actual, index, bounds, initializer, declaration, visits)


def _match_signature(declaration, signature):
    if (signature is None or None in (declaration.kind, signature[1])
            or declaration.signature() != signature):
        raise CompilationError("source-call type, kind or rank mismatch: " + declaration.name)


def _resolve_candidates(analysis, scope, call_node, associate, *, function=False):
    target, arguments = call_node.items
    if _kind(target) not in ({"Name", "Type_Name"} if function else {"Name"}):
        raise CompilationError("indirect calls are scope boundaries")
    original = tuple(_children(arguments))
    matches, errors = [], []
    candidates = analysis._candidates(scope, target)
    pool = ({name: routine for name, routine in analysis.numerical_helpers.items()
             if routine.source_kind in {"function", "internal_function"}} if function else analysis.routines)
    unavailable = [procedure for procedure in candidates if procedure not in pool]
    if unavailable:
        raise CompilationError("source call interface closure unavailable: " + ", ".join(unavailable))
    for procedure in candidates:
        callee = pool[procedure]
        try:
            if callee.source_kind == "external":
                interface = analysis.external_interface(callee)
                if not interface["available"]:
                    raise CompilationError(interface["reason"] + ": " + procedure)
                if any(_kind(item) == "Actual_Arg_Spec" for item in original):
                    raise CompilationError("external keyword arguments require a proven explicit interface: " + procedure)
            actuals, original_arguments = _arrange(callee, original, function=function)
            associations = associate(callee, actuals)
            matches.append((callee, actuals, associations, original_arguments))
        except OriginalCallBudgetExceeded:
            raise
        except CompilationError as error:
            errors.append(error)
    if len(matches) != 1:
        if not matches and errors and len({str(error) for error in errors}) == 1:
            raise errors[0]
        raise CompilationError("source call is unresolved or ambiguous: " + str(target))
    return matches[0]


def _mapped_arguments(analysis, scope, call_node, callee, actuals, *, function=False):
    mappings = []
    for formal, actual in zip(callee.arguments, actuals, strict=True):
        declaration = callee.scope.bindings[formal]
        from compiler.frontend.component_bindings import _typename
        derived = _typename(declaration.dtype) is not None
        if function and (derived or declaration.intent != "in"
                         or declaration.dtype not in {"real", "integer", "logical"}
                         or declaration.kind not in {4, 8}
                         or declaration.attributes & {"optional", "pointer", "allocatable", "value",
                                                      "volatile", "asynchronous"}
                         or declaration.rank and "target" in declaration.attributes):
            raise CompilationError("native source function requires fixed INTENT(IN) arguments")
        if actual is None:
            if derived:
                raise CompilationError("source object requires fixed nonoptional scalar original storage")
            mappings.append(SourceActualMapping(formal, None, declaration, presence="omitted"))
            continue
        signature, binding, section = _actual(analysis, scope, actual)
        scalar_element = None
        if section is not None and not section.logical_rank:
            if function:
                raise CompilationError("array-element actual requires scalar payload coherence: "
                                       "source function association needs separate authority: " + str(actual))
            try:
                scalar_element = _scalar_element_actual(analysis, scope, actual, binding, section, declaration)
            except CompilationError as error:
                raise CompilationError("array-element actual requires scalar payload coherence: "
                                       + str(error) + ": " + str(actual)) from error
            section = None
        source_object = None
        if derived:
            from compiler.frontend.source_objects import source_object_mapping
            source_object = source_object_mapping(analysis, declaration, binding, actual, call_node, scope)
        else:
            _match_signature(declaration, signature)
        if function and declaration.rank:
            _function_array_mapping(analysis, declaration, binding, section)
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
        mappings.append(SourceActualMapping(formal, actual, declaration, binding, section, presence,
                                            source_object, scalar_element))
    return tuple(mappings)


def _resolve_source_invocation(analysis, scope, call_node, *, function=False):
    callee, actuals, mappings, original_arguments = _resolve_candidates(
        analysis, scope, call_node,
        lambda callee, actuals: _mapped_arguments(analysis, scope, call_node, callee, actuals, function=function),
        function=function)
    return ResolvedSourceCall(callee.qualified, callee.arguments, actuals, mappings, original_arguments,
                              str(call_node.items[0]), callee.source_kind)

# This association layer retains original source semantics only. In particular,
# its storage reference is not a rectangle, a capture, or a no-alias proof.
class OriginalCallBudgetExceeded(CompilationError):
    """Original association inspection exceeded a bounded source scan."""


_ORIGINAL_CALL_ISSUER = object()


@dataclass(frozen=True)
class OriginalActualType:
    dtype: str
    kind: int | None
    rank: int
    binding: object = field(default=None, compare=False, repr=False)
    schema: object = field(default=None, compare=False, repr=False)

    def public(self):
        return {"type": self.dtype, "kind": self.kind, "rank": self.rank,
                "storage_resource": self.binding.root if self.binding is not None else None,
                "declaring_type": ({"module": self.schema.scope.module, "name": self.schema.name}
                                   if self.schema is not None else None)}


@dataclass(frozen=True)
class OriginalArgumentAssociation:
    formal: str
    actual: object = field(compare=False, repr=False)
    formal_binding: object = field(compare=False, repr=False)
    actual_type: OriginalActualType | None

    @property
    def binding(self):
        return self.actual_type.binding if self.actual_type is not None else None

    def public(self):
        return {"formal": self.formal, "actual": str(self.actual) if self.actual is not None else None,
                "formal_descriptor": self.formal_binding.public(),
                "actual_type": self.actual_type.public() if self.actual_type is not None else None,
                "entry_definition_kill": self.formal_binding.intent == "out"}


@dataclass(frozen=True)
class ResolvedOriginalCall:
    procedure: str
    formals: tuple[str, ...]
    actuals: tuple = field(compare=False, repr=False)
    associations: tuple[OriginalArgumentAssociation, ...]
    original_arguments: tuple[SourceCallArgument, ...]
    target: str
    source_kind: str
    callee: object = field(compare=False, repr=False)
    _caller: object = field(compare=False, repr=False)
    _analysis: object = field(compare=False, repr=False)
    _scope: object = field(compare=False, repr=False)
    _original: object = field(compare=False, repr=False)
    _scopes: tuple = field(compare=False, repr=False)
    _roles: object = field(compare=False, repr=False)
    _schemas: tuple = field(compare=False, repr=False)
    _limits: tuple = field(compare=False, repr=False)
    _issuer: object = field(compare=False, repr=False)

    def public(self):
        return {"schema_version": 1, "procedure": self.procedure, "target": self.target,
                "source_kind": self.source_kind, "formal_arguments": list(self.formals),
                "actual_arguments": [str(actual) if actual is not None else None for actual in self.actuals],
                "original_arguments": [item.public() for item in self.original_arguments],
                "associations": [item.public() for item in self.associations],
                "actual_evaluation": "unchanged original call position and guards",
                "effect_authority": False, "capture_authority": False, "alias_authority": False,
                "completion_authority": False, "execution_authority": False}

    def validate(self, analysis, scope, original):
        from compiler.frontend.source_functions import _original, _validate_scope, _validate_scope_roles
        from compiler.frontend.source_objects import _members, _TypeBuilder

        if (self._issuer is not _ORIGINAL_CALL_ISSUER or analysis is not self._analysis
                or scope is not self._scope or original is not self._original
                or getattr(analysis, "_original_call_associations", {}).get((id(analysis), id(original))) is not self):
            raise CompilationError("original call requires exact issued source association")
        if self._limits != (analysis.depth_limit, analysis.procedure_limit, analysis.operation_limit):
            raise CompilationError("original call association inspection budgets changed")
        analysis.verify_source()
        if (_original(analysis, self._caller.qualified) is not self._caller
                or _original(analysis, self.procedure) is not self.callee):
            raise CompilationError("original call procedure authority changed")
        for retained in self._scopes:
            _validate_scope(analysis, retained)
        _validate_scope_roles(analysis, self._roles, {id(scope) for scope in self._scopes})
        builder = _TypeBuilder(analysis)
        for schema in self._schemas:
            record = builder.scope(schema.scope)[5].get(schema.name)
            if (record is None or record[0] is not schema or record[1] is not schema.node
                    or record[2] != str(schema.node) or record[3] != _members(schema)):
                raise CompilationError("original call declaring type authority changed")
        return self


class _OriginalTypes:
    """Bounded declaration/type inspection; no layout, payload or index evaluation."""
    def __init__(self, analysis):
        from compiler.frontend.source_objects import _TypeBuilder

        self.analysis, self.scopes, self.schemas = analysis, {}, {}
        self.visits = 0
        reader = self

        class Types(_TypeBuilder):
            def scope(self, scope):
                reader.scope(scope)
                return super().scope(scope)

        self.types = Types(analysis)

    def scope(self, scope):
        from compiler.frontend.source_functions import _validate_scope

        if scope is not None and id(scope) not in self.scopes:
            _validate_scope(self.analysis, scope)
            self.scopes[id(scope)] = scope
        return scope

    def schema(self, scope, dtype):
        from compiler.frontend.component_bindings import _typename

        name = _typename(dtype)
        if name is None:
            if dtype not in {"real", "integer", "logical"}:
                raise CompilationError("original call requires a known nonpolymorphic declared type")
            return None
        schema = self.types.resolve(scope, name)
        if schema is None or schema.reason:
            raise CompilationError("original call declaring type is unavailable or unsupported")
        self.schemas[id(schema)] = schema
        return schema

    def declared(self, binding):
        self.scope(binding.declaring_scope)
        if binding.declaring_scope.bindings.get(binding.name) is not binding:
            raise CompilationError("original call requires an exact declared binding")
        return OriginalActualType(binding.dtype, binding.kind, binding.rank, binding,
                                  self.schema(binding.declaring_scope, binding.dtype))

    def indices(self, scope, part, rank, depth):
        if _kind(part) != "Part_Ref":
            return rank
        indices = tuple(_children(part.items[1]))
        if len(indices) != rank:
            raise CompilationError("original call array subscripts do not match declared rank")
        logical_rank = 0
        for index in indices:
            values = index.items if _kind(index) == "Subscript_Triplet" else (index,)
            logical_rank += _kind(index) == "Subscript_Triplet"
            for value in values:
                if value is not None:
                    typed = self.actual(scope, value, depth + 1)
                    if typed.dtype != "integer" or typed.rank:
                        raise CompilationError("original call requires scalar INTEGER section indices")
        return logical_rank

    def storage(self, scope, node, depth):
        parts = tuple(node.items) if _kind(node) == "Data_Ref" else (node,)
        base = parts[0].items[0] if _kind(parts[0]) == "Part_Ref" else parts[0]
        if _kind(base) != "Name":
            return None
        binding = self.analysis._binding(scope, base)
        if binding is None:
            return None
        typed = self.declared(binding)
        rank = self.indices(scope, parts[0], typed.rank, depth)
        dtype, width, schema = typed.dtype, typed.kind, typed.schema
        for part in parts[1:]:
            self.tick(depth)
            if schema is None:
                raise CompilationError("original call component requires a declared source type")
            name = str(part.items[0] if _kind(part) == "Part_Ref" else part).lower()
            declaration = schema.members.get(name)
            if declaration is None:
                raise CompilationError("original call component declaration unavailable: " + name)
            spec, attributes, entity = declaration
            flags = {str(item).split("(")[0].lower() for item in attributes}
            if (schema.private or "private" in flags) and scope.module != schema.scope.module:
                raise CompilationError("original call component is inaccessible: " + name)
            dimension = next((item.items[1] for item in attributes
                              if _kind(item) == "Dimension_Component_Attr_Spec"), None)
            shape = entity.items[1] if entity.items[1] is not None else dimension
            field_rank = len(_children(shape))
            if rank and field_rank:
                raise CompilationError("original call requires at most one array part in a component reference")
            if _kind(spec) == "Intrinsic_Type_Spec":
                base_type, selector = spec.items
                dtype = str(base_type).lower()
                width = 8 if dtype == "double precision" else 4
                dtype = "real" if dtype == "double precision" else dtype
                if selector is not None:
                    width = schema.scope.kinds.integer(selector.items[1], SourceLocation(str(scope.path)))
            else:
                dtype, width = str(spec).lower(), None
            owner = schema.scope
            schema = self.schema(owner, dtype)
            rank += self.indices(scope, part, field_rank, depth)
        return OriginalActualType(dtype, width, rank, binding, schema)

    def tick(self, depth):
        self.visits += 1
        if depth > self.analysis.depth_limit or self.visits > self.analysis.operation_limit:
            raise OriginalCallBudgetExceeded("original call type inspection budget exhausted")

    def actual(self, scope, node, depth=0):
        self.tick(depth)
        self.scope(scope)
        if _kind(node) in {"Name", "Part_Ref", "Data_Ref"}:
            storage = self.storage(scope, node, depth)
            if storage is not None:
                return storage
        items = _children(node)
        if _kind(node) == "Parenthesis":
            typed = self.actual(scope, items[1], depth + 1)
            return OriginalActualType(typed.dtype, typed.kind, typed.rank, schema=typed.schema)
        if len(items) == 2 and str(items[0]) in {"+", "-"}:
            typed = self.actual(scope, items[1], depth + 1)
            if typed.dtype in {"integer", "real"} and typed.kind in {4, 8} and not typed.rank:
                return OriginalActualType(typed.dtype, typed.kind, 0)
        if len(items) == 3 and str(items[1]) in {"+", "-", "*", "/"}:
            left, right = (self.actual(scope, child, depth + 1) for child in (items[0], items[2]))
            if ((left.dtype, left.kind, left.rank) == (right.dtype, right.kind, right.rank)
                    and left.dtype in {"integer", "real"} and left.kind in {4, 8} and not left.rank):
                return OriginalActualType(left.dtype, left.kind, 0)
        if _kind(node) in {"Int_Literal_Constant", "Real_Literal_Constant", "Logical_Literal_Constant"}:
            signature = self.analysis._signature(scope, node)
            if signature is not None and signature[1] is not None:
                return OriginalActualType(*signature)
        raise CompilationError("original call actual type requires separate source expression authority: " + str(node))

    def associate(self, callee, actuals):
        result = []
        self.scope(callee.scope)
        for formal, actual in zip(callee.arguments, actuals, strict=True):
            declaration = callee.scope.bindings[formal]
            expected = self.declared(declaration)
            typed = self.actual(self.caller_scope, actual) if actual is not None else None
            if typed is not None:
                if expected.schema is not None:
                    if expected.schema is not typed.schema or expected.rank != typed.rank:
                        raise CompilationError("source-call canonical declaring type or rank mismatch: " + formal)
                else:
                    _match_signature(declaration, (typed.dtype, typed.kind, typed.rank))
                if typed.binding is None and (declaration.intent in {"out", "inout"}
                                              or declaration.attributes & {"pointer", "allocatable"}):
                    raise CompilationError("source-call writable or descriptor actual requires original storage: " + formal)
            result.append(OriginalArgumentAssociation(formal, actual, declaration, typed))
        return tuple(result)


def _resolve_original_invocation(analysis, scope, original, *, function=False):
    from compiler.frontend.source_functions import _original, _scope_roles, _validate_resolution, _walk

    analysis.verify_source()
    allowed = {"Part_Ref", "Function_Reference", "Structure_Constructor"} if function else {"Call_Stmt"}
    if _kind(original) not in allowed:
        raise CompilationError("original association requires the corresponding CALL or function syntax")
    existing = getattr(analysis, "_original_call_associations", {}).get((id(analysis), id(original)))
    if existing is not None:
        return existing.validate(analysis, scope, original)
    # Find the exact enclosing original routine, including lexical ASSOCIATE
    # scopes. No projected statement or equal-text clone can issue authority.
    pending, caller = scope, None
    while pending is not None:
        caller = next((routine for routine in analysis.numerical_helpers.values()
                       if routine.scope is pending), None)
        if caller is not None:
            break
        pending = pending.parent
    if (caller is None or _original(analysis, caller.qualified) is not caller
            or not any(node is original for node in _walk(caller.execution))
            or analysis.source_scope_for(original, caller.scope) is not scope):
        raise CompilationError("original call requires its exact original statement and lexical scope")
    roles = _scope_roles(analysis)
    if isinstance(roles, dict):
        roles = MappingProxyType(roles)
    reader = _OriginalTypes(analysis)
    reader.caller_scope = scope
    _validate_resolution(analysis, scope, _walk(original), tick=lambda _kind: reader.tick(0),
                         _scope_validator=reader.scope)
    callee, actuals, associations, arguments = _resolve_candidates(
        analysis, scope, original, reader.associate, function=function)
    if _original(analysis, callee.qualified) is not callee:
        raise CompilationError("original call requires its exact original callee")
    proof = ResolvedOriginalCall(callee.qualified, callee.arguments, actuals, associations, arguments,
                                str(original.items[0]), callee.source_kind, callee, caller, analysis, scope,
                                original, tuple(reader.scopes.values()), roles, tuple(reader.schemas.values()),
                                (analysis.depth_limit, analysis.procedure_limit, analysis.operation_limit),
                                _ORIGINAL_CALL_ISSUER)
    registry = getattr(analysis, "_original_call_associations", None)
    if registry is None:
        registry = analysis._original_call_associations = {}
    registry[id(analysis), id(original)] = proof
    try:
        return proof.validate(analysis, scope, original)
    except BaseException:
        del registry[id(analysis), id(original)]
        raise


def resolve_original_call(analysis, scope, call_node):
    """Match an unchanged original CALL without granting physical mapping."""
    return _resolve_original_invocation(analysis, scope, call_node)


def resolve_original_function(analysis, scope, expression):
    """Match original function arguments; result/effects require separate proofs."""
    return _resolve_original_invocation(analysis, scope, expression, function=True)
