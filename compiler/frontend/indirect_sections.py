"""Source-backed inspection plans for retained native indirect accesses.

The inspector reads original Fortran metadata, never a C approximation of a
derived type.  These plans refine communication only: they establish neither
scatter independence nor permission to move a native worksharing operation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fparser.two.utils import walk

from compiler.frontend.call_bindings import _bound
from compiler.frontend.native_sections import NativeResourceSections, RECTANGLE_LIMIT
from compiler.ir import CompilationError, SourceLocation
from compiler.ir.integers import integer_literal

INSPECTION_STEP_LIMIT = 1_048_576


def _kind(node):
    return type(node).__name__


def _children(node):
    return getattr(node, "content", getattr(node, "items", ())) or ()


@dataclass(frozen=True)
class IndirectExpression:
    kind: str
    value: int | None = None
    resource: str | None = None
    member: str | None = None
    operator: str | None = None
    dimension: int | None = None
    children: tuple[IndirectExpression, ...] = ()

    def public(self):
        result = {"kind": self.kind}
        for name in ("value", "resource", "member", "operator", "dimension"):
            if getattr(self, name) is not None:
                result[name] = getattr(self, name)
        if self.children:
            result["children"] = [item.public() for item in self.children]
        return result


@dataclass(frozen=True)
class IndirectLoop:
    iterator: str
    lower: IndirectExpression
    upper: IndirectExpression
    step: int = 1

    def public(self):
        return {"iterator": self.iterator, "lower": self.lower.public(),
                "upper": self.upper.public(), "step": self.step}


@dataclass(frozen=True)
class IndirectReference:
    resource: str
    action: str
    indices: tuple[IndirectExpression, ...]
    loops: tuple[IndirectLoop, ...]
    source: str = field(compare=False)

    def public(self):
        return {"resource": self.resource, "action": self.action,
                "indices": [item.public() for item in self.indices],
                "loops": [item.public() for item in self.loops], "source_access": self.source}


@dataclass(frozen=True)
class IndirectSections:
    available: bool
    reason: str | None = None
    resources: tuple[NativeResourceSections, ...] = ()
    references: tuple[IndirectReference, ...] = ()
    metadata: tuple[object, ...] = field(default=(), compare=False, repr=False)
    scalars: tuple[object, ...] = field(default=(), compare=False, repr=False)
    structured_identity: str | None = None
    selected_node_ids: tuple[str, ...] = ()

    def public(self):
        return {"schema_version": 1, "available": self.available, "reason": self.reason,
                "structured_identity": self.structured_identity,
                "selected_original_nodes": list(self.selected_node_ids),
                "rectangle_limit": RECTANGLE_LIMIT,
                "inspection_step_limit_per_resource": INSPECTION_STEP_LIMIT,
                "inspection_step": "one entered counted-loop iteration or unlooped access; all directions included",
                "preparation": "original reached native boundary; stable host metadata only",
                "failure": "boundary before execution; original native operation remains once",
                "coordinate_system": "original logical indices mapped to full-storage half-open rectangles",
                "proves_scatter_independence": False,
                "metadata_reservations": [binding.root for binding in self.metadata],
                "metadata_lifetime": "descriptor stable throughout the selected assignment/DO operation; original reached allocation check",
                "scalar_dependencies": [binding.root for binding in self.scalars],
                "resources": [item.public() for item in self.resources],
                "references": [item.public() for item in self.references]}


def analyze_indirect_sections(analysis, requested, selected, *, completion=None, capture_locals=False):
    """Inspect affine offsets of original INTEGER metadata fields in counted DOs.

    Selection and completion tokens retain original AST authority.  Initially
    metadata is a one-dimensional array of fixed scalar fields; all managed
    accesses are scalar points.  Unproved forms return an unavailable refinement
    and leave conservative communication or owner closure to the caller.
    """
    try:
        graph, identities = analysis._selected_source(requested, selected)
        if completion is not None:
            from compiler.frontend.native_completion import NativeCompletionProof
            try:
                from compiler.frontend.worksharing_completion import WorksharingNativeCompletionProof
                admitted = (NativeCompletionProof, WorksharingNativeCompletionProof)
            except ImportError:
                admitted = (NativeCompletionProof,)
            if not isinstance(completion, admitted):
                raise CompilationError("indirect sections require a registered native completion token")
            completion.validate(analysis, requested, identities)
        originals = tuple(node for identity in identities for node in graph.source_nodes(identity))
        if completion is None and any(_kind(item) == "Comment" and str(item).lstrip().lower().startswith("!$omp")
                                      for item in walk(analysis.routines[requested].execution)):
            raise CompilationError("indirect native OpenMP inspection requires original completion proof")
        projected, routine, _guard = analysis._segment_projection(
            requested, identities, include_entry=False, capture_locals=capture_locals)
        projected._native_metadata = True
        projected._native_sections_include_entry = False
        return _analyze(projected, routine, originals, completion, graph.identity, identities)
    except CompilationError as error:
        return IndirectSections(False, str(error))


def _analyze(analysis, routine, originals, completion, identity, identities):
    from compiler.frontend.component_bindings import component_access

    location = SourceLocation(str(routine.scope.path))
    private = set(completion.private_roots) if completion is not None else set()
    resources, metadata, scalars, references, loops = {}, {}, {}, [], []
    scalar_writes, object_writes = set(), set()
    operation_count = 0

    def scope(node):
        return analysis.source_scope_for(node, routine.scope)

    def reference(node):
        if _kind(node) == "Data_Ref":
            item = component_access(analysis, scope(node), node)
            if item is None:
                raise CompilationError("indirect section component storage is unresolved")
            return item.binding, item.indices
        base = node.items[0] if _kind(node) == "Part_Ref" else node
        binding = analysis._binding(scope(node), base)
        return binding, tuple(_children(node.items[1])) if _kind(node) == "Part_Ref" else ()

    for node in originals:
        for item in walk(node):
            if _kind(item) == "Assignment_Stmt":
                target = item.items[0]
                if _kind(target) not in {"Name", "Part_Ref", "Data_Ref"}:
                    raise CompilationError("indirect section target storage is unsupported")
                binding, _ = reference(target)
                if binding is None:
                    raise CompilationError("indirect section target storage is unresolved")
                if hasattr(binding, "native_metadata_object"):
                    object_writes.add(binding.native_metadata_object.root)
                elif not binding.rank:
                    scalar_writes.add(binding.root)
                else:
                    object_writes.add(binding.root)

    def stable(binding, *, metadata_object=False):
        if binding.attributes & {"pointer", "optional", "volatile", "asynchronous", "value"}:
            raise CompilationError("indirect section association or lifetime is uncertain: " + binding.root)
        boundary = analysis.resource_identity_boundary(binding)
        if boundary:
            raise CompilationError(boundary)
        owner = analysis.modules.get(binding.root.split("::", 1)[0])
        if owner is not None:
            specification = next((node for node in _children(owner.node)
                                  if _kind(node) == "Specification_Part"), None)
            if any(_kind(item) == "Comment" and str(item).lstrip().lower().startswith("!$omp")
                   for item in walk(specification)):
                raise CompilationError("indirect section module OpenMP ownership requires proof")
        if "allocatable" in binding.attributes and not metadata_object:
            proof = analysis.descriptor_stability(routine.qualified)
            if (binding.root not in analysis.stable_module_allocatables and not any(
                    item["resource"] == binding.root and item["stable"] for item in proof["resources"])):
                raise CompilationError("indirect section allocation descriptor lifetime requires proof: " + binding.root)
        if metadata_object:
            # The complete accepted traversal contains only assignments and
            # counted loops, with no calls, allocation or association changes;
            # metadata objects are never assignment targets. This is a local
            # lifetime proof for the reached scan and retained native unit.
            # Derived-object arrays need not become numeric GPU captures or
            # appear in the numeric descriptor-stability provider.
            if binding.root in object_writes:
                raise CompilationError("indirect index metadata changes in the reached operation: " + binding.root)
            if binding.rank != 1:
                raise CompilationError("indirect index metadata requires a rank-one object array")
            metadata[binding.root] = binding

    def literal(node):
        if _kind(node) == "Int_Literal_Constant" and node.items[1] is None:
            return integer_literal(str(node.items[0]), location, source_token=True)
        items = _children(node)
        if len(items) == 2 and str(items[0]) in {"+", "-"} and _kind(items[1]) == "Int_Literal_Constant":
            value = literal(items[1])
            return integer_literal(str(value if str(items[0]) == "+" else -value), location)
        raise CompilationError("indirect coordinate requires checked original INTEGER arithmetic")

    def scalar(binding, node):
        if (binding is None or binding.rank or binding.dtype != "integer" or binding.kind not in {4, 8}
                or binding.attributes & {"pointer", "allocatable", "optional", "volatile", "asynchronous"}):
            raise CompilationError("indirect coordinate requires a stable INTEGER scalar")
        if binding.root in scalar_writes:
            raise CompilationError("indirect coordinate scalar changes in the reached operation: " + binding.root)
        active_iterator = any(loop.iterator == binding.root for loop in loops)
        if binding.root in private and not active_iterator:
            raise CompilationError("indirect coordinate requires uniform original-team scalar state: " + binding.root)
        if not active_iterator:
            scalars[binding.root] = binding
        return IndirectExpression("scalar", resource=binding.root)

    def expression(node, *, metadata_allowed=True):
        kind, items = _kind(node), _children(node)
        if kind == "Int_Literal_Constant":
            return IndirectExpression("literal", value=literal(node))
        if kind == "Parenthesis":
            return expression(items[1], metadata_allowed=metadata_allowed)
        if len(items) == 2 and str(items[0]) in {"+", "-"}:
            return IndirectExpression("unary", operator=str(items[0]),
                                      children=(expression(items[1], metadata_allowed=metadata_allowed),))
        if len(items) == 3 and str(items[1]) in {"+", "-", "*"}:
            children = tuple(expression(value, metadata_allowed=metadata_allowed) for value in (items[0], items[2]))
            if str(items[1]) == "*" and not any(item.kind == "literal" for item in children):
                raise CompilationError("indirect coordinate requires affine constant multiplication")
            return IndirectExpression("binary", operator=str(items[1]), children=children)
        if kind == "Name":
            return scalar(analysis._binding(scope(node), node), node)
        if kind == "Data_Ref":
            access = component_access(analysis, scope(node), node)
            binding = access.binding
            if hasattr(binding, "native_metadata_object"):
                if not metadata_allowed:
                    raise CompilationError("indirect loop controls cannot read metadata payload")
                if binding.dtype != "integer" or binding.kind not in {4, 8}:
                    raise CompilationError("indirect index metadata field requires INTEGER storage")
                root = binding.native_metadata_object
                stable(root, metadata_object=True)
                indices = tuple(expression(item, metadata_allowed=False) for item in access.indices)
                return IndirectExpression("metadata", resource=root.root, member=access.path[0], children=indices)
            return scalar(binding, node)
        if kind == "Intrinsic_Function_Reference":
            name = str(node.items[0]).lower()
            arguments = _children(node.items[1])
            if (name not in {"size", "lbound", "ubound"} or len(arguments) not in {1, 2}
                    or any(_kind(item) == "Actual_Arg_Spec" for item in arguments)
                    or _kind(arguments[0]) != "Name" or analysis._binding(scope(node), name)
                    or analysis._candidates(scope(node), name) or analysis._unknown_exports(scope(node))):
                raise CompilationError("indirect loop inquiry requires an original unshadowed whole-array descriptor")
            binding = analysis._binding(scope(node), arguments[0])
            if binding is None or not binding.rank:
                raise CompilationError("indirect loop inquiry descriptor is unresolved")
            stable(binding, metadata_object=binding.dtype.startswith("type("))
            dimension = literal(arguments[1]) if len(arguments) == 2 else 1
            if len(arguments) == 1 and binding.rank != 1 or not 1 <= dimension <= binding.rank:
                raise CompilationError("indirect loop inquiry requires a supported literal dimension")
            if binding.dtype.startswith("type("):
                metadata[binding.root] = binding
            else:
                scalars[binding.root] = binding
            return IndirectExpression(name, resource=binding.root, dimension=dimension)
        raise CompilationError("indirect coordinate has unsupported or side-effecting computation: " + str(node))

    def descriptor(binding):
        if binding.root in resources:
            return
        if binding.dtype not in {"real", "integer", "logical"}:
            raise CompilationError("indirect managed payload requires numeric storage: " + binding.root)
        stable(binding)
        if (binding.name not in routine.arguments and binding.root.startswith(routine.qualified + "::")
                and binding.root not in getattr(analysis, "_captured_local_roots", ()) and "save" not in binding.attributes):
            raise CompilationError("indirect section local array requires original capture proof: " + binding.root)
        descriptor_bounds, origin = "allocatable" in binding.attributes, "allocation"
        lowers, expressions = (), ()
        if descriptor_bounds:
            lowers = (None,) * binding.rank
        else:
            declaring_scope = binding.declaring_scope or routine.scope
            converted = tuple(_bound(analysis, declaring_scope, item, binding, axis) if item is not None else None
                              for axis, item in enumerate(binding.lower_bound_nodes, 1))
            if len(converted) != binding.rank:
                raise CompilationError("indirect section requires original array lower bounds")
            def constant(item):
                if item is None:
                    return 1
                if item.kind == "literal":
                    return item.value
                values = tuple(constant(child) for child in item.children)
                if not values or any(value is None for value in values):
                    return None
                if item.kind == "parenthesis":
                    result = values[0]
                elif item.kind == "unary":
                    result = values[0] if item.operator == "+" else -values[0]
                elif item.kind == "binary":
                    result = (values[0] + values[1] if item.operator == "+" else values[0] - values[1]
                              if item.operator == "-" else values[0] * values[1])
                else:
                    return None
                return integer_literal(str(result), location)
            lowers = tuple(constant(item) for item in converted)
            if any(item is None for item in lowers) and binding.name in routine.arguments:
                descriptor_bounds, origin, lowers = True, "original_dummy", (None,) * binding.rank
            elif any(item is None for item in lowers):
                raise CompilationError("indirect section dynamic local bounds need their original descriptor")
        resources[binding.root] = NativeResourceSections(binding.root, binding.rank, lowers, (), (), (),
                                                       expressions, descriptor_bounds, origin)

    def add(node, action):
        binding, indices = reference(node)
        if binding is None:
            raise CompilationError("indirect native payload storage is unresolved: " + str(node))
        if hasattr(binding, "native_metadata_object"):
            stable(binding.native_metadata_object, metadata_object=True)
            return
        if not binding.rank or binding.root in private:
            return
        if not indices or len(indices) != binding.rank or any(_kind(item) == "Subscript_Triplet" for item in indices):
            raise CompilationError("indirect native refinement initially requires scalar point accesses")
        descriptor(binding)
        coordinates = tuple(expression(item) for item in indices)
        result = IndirectReference(binding.root, action, coordinates, tuple(loops), str(node))
        if result not in references:
            references.append(result)

    def reads(node):
        if node is None or isinstance(node, (str, int)):
            return
        kind = _kind(node)
        if kind in {"Name", "Part_Ref", "Data_Ref"}:
            add(node, "read")
            return
        if kind in {"Function_Reference", "Structure_Constructor"}:
            raise CompilationError("indirect native payload function effects are unsupported")
        if kind == "Intrinsic_Function_Reference":
            from compiler.ir.intrinsics import ARRAY_INQUIRIES, INTRINSICS, MODEL_INQUIRIES
            name = str(node.items[0]).lower()
            if (name not in set(INTRINSICS) | ARRAY_INQUIRIES | MODEL_INQUIRIES
                    or analysis._binding(scope(node), name) or analysis._candidates(scope(node), name)
                    or analysis._unknown_exports(scope(node))):
                raise CompilationError("indirect native intrinsic effects are unresolved")
            arguments = _children(node.items[1])
            if name in ARRAY_INQUIRIES | MODEL_INQUIRIES:
                if not arguments or _kind(arguments[0]) != "Name":
                    raise CompilationError("indirect native descriptor inquiry requires a whole variable")
                arguments = arguments[1:]
            for argument in arguments:
                reads(argument.items[1] if _kind(argument) == "Actual_Arg_Spec" else argument)
            return
        if kind.endswith("Literal_Constant"):
            return
        for child in _children(node):
            reads(child)

    def statements(nodes):
        nonlocal operation_count
        for node in nodes:
            kind = _kind(node)
            if kind == "Comment":
                continue
            operation_count += 1
            if operation_count > analysis.operation_limit:
                raise CompilationError("indirect native operation budget exceeded")
            if kind == "Assignment_Stmt":
                target, _, value = node.items
                reads(value)
                add(target, "write")
            elif kind == "Block_Nonlabel_Do_Construct":
                body = tuple(item for item in _children(node) if _kind(item) != "Comment")
                control = body[0].items[1]
                if (control is None or control.items[0] is not None or control.items[1] is None
                        or any(item is not None for item in control.items[2:])):
                    raise CompilationError("indirect native inspection requires ordinary counted DO loops")
                iterator_node, bounds = control.items[1]
                iterator = analysis._binding(scope(body[0]), iterator_node)
                if (iterator is None or iterator.rank or iterator.dtype != "integer" or iterator.kind not in {4, 8}
                        or iterator.root in scalar_writes or any(item.iterator == iterator.root for item in loops)):
                    raise CompilationError("indirect native iterator is mutable or unresolved")
                lower, upper = (expression(item, metadata_allowed=False) for item in bounds[:2])
                step = 1 if len(bounds) < 3 else literal(bounds[2])
                if step == 0:
                    raise CompilationError("indirect native iterator requires a nonzero constant stride")
                loops.append(IndirectLoop(iterator.root, lower, upper, step))
                statements(body[1:-1])
                loops.pop()
            elif kind != "Continue_Stmt":
                raise CompilationError("indirect native inspection requires assignment-only counted DO leaves")

    statements(originals)
    if not references or not any(item.kind == "metadata" for reference in references
                                 for coordinate in reference.indices for item in _expressions(coordinate)):
        raise CompilationError("indirect native refinement requires source-backed metadata indices")
    if len(references) > analysis.operation_limit:
        raise CompilationError("indirect native reference budget exceeded")
    return IndirectSections(True, resources=tuple(resources[key] for key in sorted(resources)),
                            references=tuple(references), metadata=tuple(metadata[key] for key in sorted(metadata)),
                            scalars=tuple(scalars[key] for key in sorted(scalars)),
                            structured_identity=identity, selected_node_ids=identities)


def _expressions(expression):
    yield expression
    for child in expression.children:
        yield from _expressions(child)
