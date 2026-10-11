"""Original whole-array call coordinates, conditional on reached execution.

This adapter authenticates associations without scanning a child's executable
closure. The reached planner owns child effects, branches and completion. No
coordinate token supplies storage, definition, execution or GPU authority.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from types import MappingProxyType

from fparser.two.utils import Base

from compiler.frontend.call_bindings import SourceDependency, _bound, resolve_source_call
from compiler.frontend.native_sections import RECTANGLE_LIMIT, NativeAxis, NativeBound, NativeIteration, NativeRectangle
from compiler.frontend.source_functions import _validate_resolution, _validate_scope, _walk
from compiler.frontend.structured_effects import _freeze, _thaw
from compiler.ir import CompilationError, SourceLocation
from compiler.ir.integers import integer_literal

NATIVE_CALL_COORDINATES_VERSION = 2
_UNSAFE = {"pointer", "optional", "volatile", "asynchronous", "value"}


class NativeCallCoordinatesUnsupported(CompilationError):
    """A valid original call needs to run after its owner closes."""


def _hash(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class _Budget:
    def __init__(self, analysis, consume=None):
        self.analysis, self.consume, self.work = analysis, consume, 0

    def tick(self, _label=None, depth=0):
        self.work += 1
        if self.work > self.analysis.operation_limit or depth > self.analysis.depth_limit:
            raise CompilationError("native call coordinate aggregate budget exhausted")
        if self.consume is not None:
            self.consume(1)


def _dependencies(children):
    values = {(item.resource, item.kind): item for child in children for item in child.dependencies}
    return tuple(values[key] for key in sorted(values))


def _convert(bound, budget, depth=0):
    budget.tick(depth=depth)
    return NativeBound(bound.kind, bound.value, bound.dimension, bound.resource, bound.operator,
                       tuple(_convert(child, budget, depth + 1) for child in bound.children), bound.dependencies)


def _arithmetic(operator, left, right):
    if left.kind == right.kind == "literal":
        value = left.value + right.value if operator == "+" else left.value - right.value
        try:
            integer_literal(str(value), SourceLocation("<original-call-coordinates>"))
        except CompilationError as error:
            raise NativeCallCoordinatesUnsupported("native call coordinate constant exceeds INTEGER ABI") from error
        return NativeBound("literal", value)
    if right.kind == "literal" and right.value == 0:
        return left
    if operator == "-" and left.public() == right.public():
        return NativeBound("literal", 0)
    if operator == "+" and left.kind == "literal" and left.value == 0:
        return right
    return NativeBound("binary", operator=operator, children=(left, right), dependencies=_dependencies((left, right)))


def _constant(bound):
    if bound.kind == "literal":
        return bound.value
    values = tuple(_constant(child) for child in bound.children)
    if not values or any(value is None for value in values):
        return None
    if bound.kind == "parenthesis":
        return values[0]
    if bound.kind == "unary" and bound.operator in {"+", "-"}:
        return values[0] if bound.operator == "+" else -values[0]
    if bound.kind == "binary" and bound.operator in {"+", "-", "*"}:
        left, right = values
        return left + right if bound.operator == "+" else left - right if bound.operator == "-" else left * right
    return None


def _call_resolution_nodes(original, *, tick):
    """Keep original syntax charges while resolving field names by their type."""
    stack, visits = [iter(((original, False),))], 0
    while stack:
        try:
            node, member = next(stack[-1])
        except StopIteration:
            stack.pop()
            continue
        visits += 1
        if visits > 65536:
            raise CompilationError("native call coordinate original-source scan budget exhausted")
        if isinstance(node, (tuple, list)):
            stack.append(iter((child, False) for child in node))
        elif isinstance(node, Base):
            kind = type(node).__name__
            # Charge this original occurrence without submitting its spelling
            # to the caller's USE namespace. component_access and _TypeBuilder
            # authenticate every admitted field below; no AST is fabricated.
            if member and kind == "Name":
                tick("syntax")
            else:
                yield node
            children = getattr(node, "content", getattr(node, "items", ())) or ()
            if kind == "Data_Ref":
                stack.append(iter((child, index > 0) for index, child in enumerate(children)))
            elif member and kind == "Part_Ref":
                stack.append(iter((child, index == 0) for index, child in enumerate(children)))
            else:
                stack.append(iter((child, False) for child in children))


@dataclass(frozen=True)
class _Association:
    formal: object = field(compare=False, repr=False)
    actual: object = field(compare=False, repr=False)
    formal_lowers: tuple[NativeBound, ...]
    actual_lowers: tuple[NativeBound, ...]

    def public(self):
        return {"formal_resource": self.formal.root, "actual_resource": self.actual.root,
                "rank": self.formal.rank, "formal_lower_origins": [item.public() for item in self.formal_lowers],
                "actual_lower_origins": [item.public() for item in self.actual_lowers]}


def _derive(analysis, caller, original_call, budget):
    routine = analysis._require_original(caller)
    if type(original_call).__name__ != "Call_Stmt":
        raise NativeCallCoordinatesUnsupported("native call coordinates require an original direct call")
    graph, selected = analysis._selected_source(caller, (original_call,))
    if len(selected) != 1 or graph.nodes[selected[0]].kind != "call":
        raise NativeCallCoordinatesUnsupported("native call coordinates require a complete original call node")
    _validate_scope(analysis, routine.scope)
    _validate_resolution(analysis, routine.scope, _call_resolution_nodes(original_call, tick=budget.tick),
                         tick=budget.tick)
    resolved = resolve_source_call(analysis, routine.scope, original_call)
    child = analysis._require_original(resolved.procedure)
    _validate_scope(analysis, child.scope)
    if len(resolved.mappings) > analysis.operation_limit:
        raise CompilationError("native call coordinate argument budget exhausted")
    arrays, scalars, scalar_roots, kills, immutable, components = [], {}, {}, [], [], []
    actual_roots = set()
    constant_lowers = {}

    def lower(binding, owner, axis, *, formal=False):
        if len(binding.lower_bound_nodes) != binding.rank:
            raise NativeCallCoordinatesUnsupported("native call coordinates require original declared bounds")
        # An allocation's lower bound belongs to its current original descriptor.
        # It is not the declaration's deferred-shape spelling or an address fact.
        if "allocatable" in binding.attributes:
            if formal:
                raise NativeCallCoordinatesUnsupported("native call coordinates exclude allocatable callee formals")
            return NativeBound("lbound", dimension=axis, resource=binding.root,
                               dependencies=(SourceDependency(binding.root, "descriptor_read", binding),))
        node = binding.lower_bound_nodes[axis - 1]
        if node is None:
            return NativeBound("literal", 1)
        # A DIMENSION attribute can share this exact expression among several
        # original declarations. Reuse only a successfully authenticated and
        # checked constant within this derivation; descriptors and mutable
        # specification inputs still require their original treatment below.
        key = (id(owner), id(node))
        if key in constant_lowers:
            budget.tick()
            return constant_lowers[key]
        _validate_resolution(analysis, owner, _walk(node), tick=budget.tick)
        try:
            source_bound = _bound(analysis, owner, node, binding, axis)
        except CompilationError as error:
            raise NativeCallCoordinatesUnsupported("native call coordinates require constant ordinary dummy bounds") from error
        value = _convert(source_bound, budget)
        constant = _constant(value)
        if constant is None:
            if formal:
                raise NativeCallCoordinatesUnsupported("native call coordinates require constant callee lower origins")
            # Ordinary runtime dummy bounds ran at original procedure entry.
            # Never reevaluate the specification expression at this later call.
            if binding.name not in routine.arguments:
                raise NativeCallCoordinatesUnsupported("native call coordinates require an original caller descriptor")
            return NativeBound("lbound", dimension=axis, resource=binding.root,
                               dependencies=(SourceDependency(binding.root, "descriptor_read", binding),))
        integer_literal(str(constant), SourceLocation(str(owner.path)))
        result = NativeBound("literal", constant)
        if not source_bound.dependencies and not value.dependencies:
            constant_lowers[key] = result
        return result

    for mapping in resolved.mappings:
        budget.tick()
        formal, actual = mapping.formal_binding, mapping.binding
        if (mapping.presence != "supplied" or mapping.actual is None or mapping.section is not None
                or formal.attributes & _UNSAFE or formal.dtype not in {"real", "integer", "logical"}
                or formal.kind not in {4, 8} or mapping.source_object is not None):
            raise NativeCallCoordinatesUnsupported("native call coordinates require fixed supplied numeric associations")
        if actual is not None:
            component = type(mapping.actual).__name__ == "Data_Ref"
            if component:
                from compiler.frontend.component_bindings import _typename, component_access
                from compiler.frontend.source_objects import _TypeBuilder

                if formal.rank or formal.intent != "in" or formal.dtype not in {"real", "logical"}:
                    raise NativeCallCoordinatesUnsupported("native call component actual requires readonly noninteger scalar")
                access = component_access(analysis, routine.scope, mapping.actual)
                if (access is None or access.indices or access.binding.rank
                        or access.binding.public() != actual.public()):
                    raise NativeCallCoordinatesUnsupported("native call component actual requires a fixed scalar field")
                authority = _TypeBuilder(analysis)
                schema = authority.binding(access.root_binding)
                for index, name in enumerate(access.path):
                    budget.tick(depth=index + 1)
                    # Exact type authentication compares every sibling member,
                    # even though only this selected field needs fixed storage.
                    # Charge that work without imposing layout legality on the
                    # unrelated fields.
                    for _ in schema.members:
                        budget.tick()
                    member = schema.members.get(name)
                    if member is None:
                        raise CompilationError("native call original component declaration changed")
                    if index + 1 < len(access.path):
                        nested = _typename(member[0])
                        schema = authority.resolve(schema.scope, nested) if nested else None
                        if schema is None:
                            raise CompilationError("native call original component type authority changed")
                for _ in range(authority.visits):
                    budget.tick()
                components.append(access.public())
            if ((type(mapping.actual).__name__ != "Name" and not component) or actual.attributes & _UNSAFE
                    or analysis.resource_identity_boundary(actual)):
                raise NativeCallCoordinatesUnsupported("native call coordinates require original whole Name storage")
            _validate_scope(analysis, actual.declaring_scope or routine.scope)
        if not formal.rank:
            if formal.intent != "in" or "allocatable" in formal.attributes:
                raise NativeCallCoordinatesUnsupported("native call coordinate scalar formals require INTENT(IN)")
            if actual is not None:
                scalar_roots[formal.root] = actual.root
            if formal.dtype == "integer":
                syntax = mapping.actual
                if (actual is None and type(syntax).__name__ != "Int_Literal_Constant"
                        and not (len(getattr(syntax, "items", ())) == 2
                                 and str(syntax.items[0]) in {"+", "-"}
                                 and type(syntax.items[1]).__name__ == "Int_Literal_Constant")):
                    raise NativeCallCoordinatesUnsupported("native call scalar expressions need original call-entry snapshot proof")
                try:
                    source_bound = _bound(analysis, routine.scope, mapping.actual, formal, 1)
                except CompilationError as error:
                    raise NativeCallCoordinatesUnsupported("native call coordinate scalar requires original INTEGER storage or literal") from error
                value = _convert(source_bound, budget)
                constant = _constant(value)
                if constant is not None:
                    integer_literal(str(constant), SourceLocation(str(routine.scope.path)))
                    value = NativeBound("literal", constant)
                scalars[formal.root] = value
            elif actual is None and not type(mapping.actual).__name__.endswith("Literal_Constant"):
                raise NativeCallCoordinatesUnsupported("native call coordinate scalar expression requires separate proof")
            continue
        if (actual is None or not 1 <= formal.rank <= 4 or formal.signature() != actual.signature()
                or "allocatable" in formal.attributes or formal.intent not in {"in", "out", "inout"}):
            raise NativeCallCoordinatesUnsupported("native call coordinates require equal-rank ordinary array associations")
        if "parameter" in actual.attributes:
            if formal.intent != "in":
                raise NativeCallCoordinatesUnsupported("immutable original array cannot be a writable call actual")
            immutable.append(mapping.public())
            continue
        if (len(formal.shape_nodes) != formal.rank
                or any(type(axis).__name__ != "Assumed_Shape_Spec" for axis in formal.shape_nodes)):
            raise NativeCallCoordinatesUnsupported("native call coordinates initially require assumed-shape array formals")
        if actual.root in actual_roots:
            raise NativeCallCoordinatesUnsupported("native call coordinates require distinct original array actual roots")
        actual_roots.add(actual.root)
        arrays.append(_Association(formal, actual,
            tuple(lower(formal, child.scope, axis, formal=True) for axis in range(1, formal.rank + 1)),
            tuple(lower(actual, actual.declaring_scope or routine.scope, axis)
                  for axis in range(1, actual.rank + 1))))
        if formal.intent == "out":
            kills.append(actual.root)
    record = {"schema_version": NATIVE_CALL_COORDINATES_VERSION, "kind": "original_native_call_coordinates",
              "caller": caller, "callee": resolved.procedure, "node_id": selected[0],
              "guard": list(graph.nodes[selected[0]].guard), "structured_identity": graph.identity,
              "analysis_identity": graph.authority_identity, "source_identity": analysis.inputs.identity(),
              "call": resolved.public(), "coordinate_associations": [item.public() for item in arrays],
              "scalar_mapping": dict(sorted(scalar_roots.items())),
              "scalar_bounds": {root: value.public() for root, value in sorted(scalars.items())},
              "original_scalar_components": components,
              "unmanaged_immutable_associations": immutable, "entry_definition_changes": kills,
              "source_work": budget.work, "bounds_evaluated": False, "execution_authorized": False,
              "entry_definition_authority": False, "allocation_freshness_authority": False,
              "gpu_legality_established": False, "requires_original_call_and_guards": True,
              "requires_registered_child_effect_and_completion_proofs": True,
              "requires_immutable_array_input_for_immutable_associations": bool(immutable),
              "requires_runtime_layout_alias_and_integer_checks": True,
              "mapping_does_not_apply_entry_kills": True,
              "limits": ["whole supplied Name equal-rank ordinary assumed-shape arrays",
                         "all associated OUT roots retained, including unused formals",
                         "immutable PARAMETER actuals stay original unmanaged associations",
                         "nontrivial empty-sensitive LBOUND/UBOUND inquiry substitutions rejected",
                         "partial caller-to-child transforms exclude unresolved caller descriptor origins",
                         "conditional child return coverage remains owned by its reached plan"]}
    return tuple(arrays), scalars, scalar_roots, tuple(kills), record


@dataclass(frozen=True)
class NativeCallCoordinatesProof:
    caller: str
    callee: str
    identity: str
    _analysis: object = field(compare=False, repr=False)
    _call: object = field(compare=False, repr=False)
    _arrays: tuple[_Association, ...] = field(compare=False, repr=False)
    _scalars: object = field(compare=False, repr=False)
    _scalar_roots: object = field(compare=False, repr=False)
    _kills: tuple[str, ...] = field(compare=False, repr=False)
    _record: object = field(compare=False, repr=False)

    @property
    def formal_roots(self):
        return tuple(item.formal.root for item in self._arrays)

    @property
    def actual_roots(self):
        return tuple(item.actual.root for item in self._arrays)

    @property
    def resource_mapping(self):
        return {item.formal.root: item.actual.root for item in self._arrays}

    @property
    def scalar_mapping(self):
        return dict(self._scalar_roots)

    @property
    def entry_definition_changes(self):
        return self._kills

    def public(self):
        return {"proof_identity": self.identity, **_thaw(self._record)}

    def validate(self, analysis, caller, original_call, *, consume=None):
        if (analysis is not self._analysis or caller != self.caller or original_call is not self._call
                or getattr(analysis, "_native_definition_proofs", {}).get(self.identity) is not self):
            raise CompilationError("native call coordinates lack registered exact original source authority")
        budget = _Budget(analysis, consume)
        if _hash(_derive(analysis, caller, original_call, budget)[4]) != self.identity:
            raise CompilationError("native call coordinate source authority changed")
        return self

    def caller_to_child(self, rectangles, *, consume=None):
        return self._map(rectangles, to_child=True, consume=consume)

    def child_to_caller(self, rectangles, *, consume=None):
        return self._map(rectangles, to_child=False, consume=consume)

    def _map(self, rectangles, *, to_child, consume):
        self.validate(self._analysis, self.caller, self._call, consume=consume)
        # Source authentication and geometry transfer are independently
        # bounded phases. Both still spend the same parent's aggregate work.
        budget = _Budget(self._analysis, consume)
        if (not isinstance(rectangles, (dict, MappingProxyType)) or len(rectangles) > self._analysis.operation_limit):
            raise CompilationError("native call coordinates require bounded resource rectangle mappings")
        associations = {item.actual.root if to_child else item.formal.root: item for item in self._arrays}
        reverse_scalars, ambiguous_scalars = {}, set()
        for formal, value in self._scalars.items():
            if value.kind == "scalar":
                if value.resource in reverse_scalars and reverse_scalars[value.resource].resource != formal:
                    ambiguous_scalars.add(value.resource)
                binding = self._analysis.routines[self.callee].scope.bindings[formal.removeprefix("argument::")]
                reverse_scalars[value.resource] = NativeBound("scalar", resource=formal,
                    dependencies=(SourceDependency(formal, "scalar_read", binding),))

        def bound(value, depth=0):
            if value is None:
                return None
            budget.tick(depth=depth)
            if value.kind == "literal":
                integer_literal(str(value.value), SourceLocation("<original-call-coordinates>"))
                return value
            if value.kind == "scalar":
                if to_child and value.resource in ambiguous_scalars:
                    raise NativeCallCoordinatesUnsupported("native call scalar coordinate association is ambiguous")
                result = reverse_scalars.get(value.resource) if to_child else self._scalars.get(value.resource)
                if result is not None:
                    return result
                return module_bound(value)
            if value.kind in {"lbound", "ubound", "size"}:
                association = associations.get(value.resource)
                if association is None:
                    return module_bound(value)
                if value.dimension is None or not 1 <= value.dimension <= association.formal.rank:
                    raise CompilationError("native call coordinate inquiry has invalid dimension")
                axis = value.dimension - 1
                if (value.kind != "size" and association.formal_lowers[axis].public()
                        != association.actual_lowers[axis].public()):
                    raise NativeCallCoordinatesUnsupported("native call coordinates cannot rewrite empty-sensitive LBOUND/UBOUND inquiry")
                target = association.formal if to_child else association.actual
                return NativeBound(value.kind, dimension=value.dimension, resource=target.root,
                                   dependencies=(SourceDependency(target.root, "descriptor_read", target),))
            if value.kind not in {"parenthesis", "unary", "binary"}:
                raise NativeCallCoordinatesUnsupported("native call coordinate bound needs separate source proof")
            if (value.kind == "parenthesis" and len(value.children) != 1
                    or value.kind == "unary" and (len(value.children) != 1 or value.operator not in {"+", "-"})
                    or value.kind == "binary" and (len(value.children) != 2 or value.operator not in {"+", "-", "*"})):
                raise CompilationError("native call coordinate expression tree is malformed")
            children = tuple(bound(child, depth + 1) for child in value.children)
            return NativeBound(value.kind, operator=value.operator, children=children, dependencies=_dependencies(children))

        def module_binding(root):
            module_name, separator, name = root.partition("::")
            module = self._analysis.modules.get(module_name)
            binding = module.bindings.get(name) if module is not None and separator else None
            if (binding is None or binding.root != root or binding.attributes & _UNSAFE
                    or binding.dtype not in {"real", "integer", "logical"} or binding.kind not in {4, 8}
                    or self._analysis.resource_identity_boundary(binding)):
                raise NativeCallCoordinatesUnsupported("native call coordinates have an unmapped original resource: " + root)
            _validate_scope(self._analysis, module)
            return binding

        def module_bound(value):
            binding = module_binding(value.resource or "")
            if (value.kind == "scalar" and binding.rank or value.kind != "scalar" and not binding.rank
                    or value.kind == "scalar" and binding.dtype != "integer"):
                raise NativeCallCoordinatesUnsupported("native call coordinate dependency type is unsupported")
            return value

        result, rectangle_count = {}, 0
        for root, boxes in rectangles.items():
            budget.tick()
            if not isinstance(root, str):
                raise CompilationError("native call coordinates require original resource names")
            association = associations.get(root)
            target = (association.formal if to_child else association.actual) if association else module_binding(root)
            if (not isinstance(boxes, tuple) or len(boxes) > RECTANGLE_LIMIT
                    or any(not isinstance(box, NativeRectangle) or len(box.axes) != target.rank for box in boxes)):
                raise CompilationError("native call coordinates require bounded original equal-rank rectangles")
            transformed = []
            for box in boxes:
                budget.tick()
                rectangle_count += 1
                if rectangle_count > RECTANGLE_LIMIT * len(rectangles):
                    raise CompilationError("native call coordinate rectangle budget exhausted")
                axes = []
                for index, axis in enumerate(box.axes):
                    def coordinate(value, association=association, index=index):
                        # Canonical dummy roots are not procedure-qualified.
                        # Carrying the caller's descriptor into child geometry
                        # could therefore impersonate a same-spelling child
                        # dummy. Whole/none coverage needs no such operand.
                        if (value is not None and association is not None and to_child
                                and association.actual_lowers[index].kind != "literal"):
                            raise NativeCallCoordinatesUnsupported(
                                "partial child coordinates need an authenticated caller descriptor origin snapshot")
                        value = bound(value)
                        if value is None or association is None:
                            return value
                        source_lower = association.actual_lowers[index] if to_child else association.formal_lowers[index]
                        target_lower = association.formal_lowers[index] if to_child else association.actual_lowers[index]
                        if source_lower.public() == target_lower.public():
                            return value
                        budget.tick()
                        budget.tick()
                        return _arithmetic("+", target_lower, _arithmetic("-", value, source_lower))
                    axes.append(NativeAxis(coordinate(axis.lower), coordinate(axis.upper), axis.point))
                iterations = tuple(NativeIteration(bound(item.lower), bound(item.upper), item.step)
                                   for item in box.iterations)
                transformed.append(NativeRectangle(tuple(axes), box.node, iterations))
            result[target.root] = tuple(transformed)
        return MappingProxyType(result)


def prove_native_call_coordinates(analysis, caller, original_call, *, consume=None):
    """Register original associations; child reached transfer remains separate."""
    budget = _Budget(analysis, consume)
    arrays, scalars, scalar_roots, kills, record = _derive(analysis, caller, original_call, budget)
    identity = _hash(record)
    registry = getattr(analysis, "_native_definition_proofs", None)
    if registry is None:
        registry = analysis._native_definition_proofs = {}
    if identity in registry:
        proof = registry[identity]
        # The complete fresh derivation above already authenticated the source
        # and matching record. A second derivation would make recorded work
        # depend on whether this token was created earlier.
        if (not isinstance(proof, NativeCallCoordinatesProof) or proof.identity != identity
                or proof._analysis is not analysis or proof.caller != caller or proof._call is not original_call
                or _hash(_thaw(proof._record)) != identity):
            raise CompilationError("native call coordinates lack registered exact original source authority")
        return proof
    if len(registry) >= analysis.procedure_limit:
        raise CompilationError("native call coordinate proof registry budget exhausted")
    proof = NativeCallCoordinatesProof(caller, record["callee"], identity, analysis, original_call, arrays,
                                       MappingProxyType(scalars), MappingProxyType(scalar_roots), kills, _freeze(record))
    registry[identity] = proof
    return proof
