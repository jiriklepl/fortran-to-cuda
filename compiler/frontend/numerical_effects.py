"""Guarded physical effects of an exact prepared original numerical region.

Only compiler-issued preparation can supply computation IR. The rectangles
describe original logical coordinates, not uploads or dependence surrogates.
They grant no storage, native-hook, numerical-execution or input-definition
authority. Transfer is conditional on the original preparation preconditions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from hashlib import sha256

from compiler.frontend.call_bindings import SourceDependency, _bound, _constant_bound
from compiler.frontend.native_definitions import NativeDefinitionStep, _transfer_steps, _unique
from compiler.frontend.native_sections import (
    RECTANGLE_LIMIT,
    NativeAxis,
    NativeBound,
    NativeIteration,
    NativeRectangle,
    NativeResourceSections,
    NativeSections,
)
from compiler.frontend.structured_effects import _freeze, _thaw
from compiler.ir import Binary, CompilationError, Literal, Reference, ScalarType, Size, SourceLocation, Unary
from compiler.ir.integers import INTEGER_MAX, INTEGER_MIN, integer_literal
from compiler.offload.analysis import _canonical

NUMERICAL_EFFECT_VERSION = 1
_SCAN_LIMIT = 65536


class NumericalEffectUnsupported(CompilationError):
    """Authenticated numerical geometry has no supported exact representation."""


def _hash(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class _Mapper:
    def __init__(self, analysis, routine, prepared, consume):
        self.analysis, self.routine, self.prepared = analysis, routine, prepared
        self.consume, self.work = consume, 0
        self.parameters = frozenset(prepared.function.parameters)
        self.symbols, self.bindings, self.lowers = {}, {}, {}
        self.original_specifications = {}
        parameters = {}
        for parameter in prepared.extraction.parameters:
            self.tick()
            name = parameter.name.lower()
            if name in parameters:
                raise CompilationError("numerical effect preparation has duplicate parameter authority")
            parameters[name] = parameter
        for binding in prepared.extraction.bindings:
            self.tick()
            if binding.root in self.bindings:
                raise CompilationError("numerical effect preparation has duplicate binding authority")
            self.bindings[binding.root] = binding
        if len(parameters) != len(self.parameters):
            raise CompilationError("numerical effect preparation parameter identities changed")
        for symbol in prepared.function.parameters:
            self.tick()
            parameter = parameters.get(symbol.name.lower())
            if parameter is None or symbol.rank != parameter.rank:
                raise CompilationError("numerical effect preparation parameter correspondence changed")
            self.symbols[symbol] = parameter
            if parameter.lower_bound_dimension is not None:
                key = parameter.resource, parameter.lower_bound_dimension
                if key in self.lowers or not parameter.runtime_lower_bound or symbol.rank:
                    raise CompilationError("numerical effect original lower-bound correspondence changed")
                self.lowers[key] = symbol

    def tick(self, count=1):
        self.work += count
        if self.consume is not None:
            self.consume(count)
        if self.work > _SCAN_LIMIT:
            raise CompilationError("numerical effect physical traversal budget exhausted")

    def binding(self, resource):
        self.tick()
        binding = self.bindings.get(resource)
        if binding is None:
            raise NumericalEffectUnsupported("numerical effect bound requires an original simple binding: " + resource)
        return binding

    def scan_expression(self, expression, depth=0):
        self.tick()
        if depth > self.analysis.depth_limit:
            raise NumericalEffectUnsupported("numerical effect physical bound depth is unsupported")
        if isinstance(expression, Binary):
            self.scan_expression(expression.left, depth + 1)
            self.scan_expression(expression.right, depth + 1)
        elif isinstance(expression, Unary):
            self.scan_expression(expression.operand, depth + 1)
        elif not isinstance(expression, (Literal, Reference, Size)):
            raise NumericalEffectUnsupported("numerical effect physical bound expression is unsupported")

    def canonical(self, expression):
        self.scan_expression(expression)
        # The existing affine helper builds an environment and descriptor
        # atoms for every parameter. Charge those visits as well as syntax.
        for symbol in self.parameters:
            self.tick(1 + symbol.rank)
        return _canonical(expression, self.parameters)

    def bound(self, expression, depth=0):
        self.tick()
        if depth > self.analysis.depth_limit:
            raise NumericalEffectUnsupported("numerical effect physical bound depth is unsupported")
        if isinstance(expression, Literal):
            if expression.dtype is not ScalarType.INTEGER:
                raise NumericalEffectUnsupported("numerical effect bounds require default INTEGER")
            try:
                value = int(expression.value)
            except (TypeError, ValueError) as error:
                raise NumericalEffectUnsupported("numerical effect bound literal is unsupported") from error
            if not INTEGER_MIN <= value <= INTEGER_MAX:
                raise NumericalEffectUnsupported("numerical effect bound literal exceeds default INTEGER")
            return NativeBound("literal", value)
        if isinstance(expression, Reference):
            parameter = self.symbols.get(expression.symbol)
            if parameter is None or expression.symbol.rank or expression.symbol.dtype is not ScalarType.INTEGER:
                raise NumericalEffectUnsupported("numerical effect bound requires an original INTEGER input")
            binding = self.binding(parameter.resource)
            if parameter.lower_bound_dimension is not None:
                if not 1 <= parameter.lower_bound_dimension <= binding.rank:
                    raise CompilationError("numerical effect original lower-bound dimension changed")
                return NativeBound("lbound", dimension=parameter.lower_bound_dimension, resource=binding.root,
                                   dependencies=(SourceDependency(binding.root, "descriptor_read", binding),))
            if binding.rank or binding.dtype != "integer" or binding.kind != 4:
                raise NumericalEffectUnsupported("numerical effect bound requires an original default INTEGER scalar")
            return NativeBound("scalar", resource=binding.root,
                               dependencies=(SourceDependency(binding.root, "scalar_read", binding),))
        if isinstance(expression, Size):
            parameter = self.symbols.get(expression.symbol)
            if parameter is None or not parameter.rank or not 1 <= expression.dimension <= parameter.rank:
                raise NumericalEffectUnsupported("numerical effect SIZE requires an original array descriptor")
            binding = self.binding(parameter.resource)
            return NativeBound("size", dimension=expression.dimension, resource=binding.root,
                               dependencies=(SourceDependency(binding.root, "descriptor_read", binding),))
        if isinstance(expression, Unary) and expression.operator in {"+", "-"}:
            children = (self.bound(expression.operand, depth + 1),)
            return NativeBound("unary", operator=expression.operator, children=children)
        if isinstance(expression, Binary) and expression.operator in {"+", "-", "*"}:
            children = (self.bound(expression.left, depth + 1), self.bound(expression.right, depth + 1))
            if expression.operator == "*" and not any(child.kind == "literal" for child in children):
                raise NumericalEffectUnsupported("numerical effect bound multiplication requires a literal factor")
            return NativeBound("binary", operator=expression.operator, children=children)
        raise NumericalEffectUnsupported("numerical effect physical bound expression is unsupported")

    def original(self, expression, binding, dimension):
        symbol = self.lowers.get((binding.root, dimension))
        if symbol is None:
            raise CompilationError("numerical effect original array lower-bound parameter changed")
        expression = Binary("-", Binary("+", expression, Reference(symbol)), Literal("1", ScalarType.INTEGER))
        return self.bound(self.canonical(expression))

    def iterations(self, unit):
        result = []
        for loop in unit.region.loops:
            self.tick()
            step = loop.step
            if isinstance(step, Literal) and step.dtype is ScalarType.INTEGER:
                try:
                    step = int(step.value)
                except (TypeError, ValueError):
                    step = None
            if type(step) is not int or step not in {-1, 1}:
                raise NumericalEffectUnsupported("numerical effect iteration guards require unit constant stride")
            result.append(NativeIteration(self.bound(self.canonical(loop.lower)),
                                          self.bound(self.canonical(loop.upper)), step))
        return tuple(result)

    def rectangles(self, boxes, unit, binding, node, iterations):
        if len(boxes) > RECTANGLE_LIMIT:
            raise CompilationError("numerical effect rectangle budget exhausted")
        result = []
        for box in boxes:
            self.tick()
            if box.active_units != (unit.index,):
                raise NumericalEffectUnsupported("numerical effect active-unit alternatives require separate guards")
            if len(box.lower) != binding.rank or len(box.upper) != binding.rank or len(box.axes) != binding.rank:
                raise CompilationError("numerical effect physical footprint rank changed")
            axes = []
            for dimension, (lower, upper, mapping) in enumerate(zip(box.lower, box.upper, box.axes, strict=True), 1):
                self.tick()
                if (mapping.axis is None and mapping.coefficient != 0
                        or mapping.axis is not None and (not 0 <= mapping.axis < len(unit.region.loops)
                                                        or mapping.coefficient not in {-1, 1})):
                    raise NumericalEffectUnsupported("numerical effect physical axis is unsupported")
                axes.append(NativeAxis(self.original(lower, binding, dimension),
                                       self.original(upper, binding, dimension), mapping.axis is None))
            result.append(NativeRectangle(tuple(axes), node, iterations))
        return _unique(tuple(result))

    def declared_lowers(self, binding):
        from compiler.frontend.numerical_preparation import _walk as source_walk
        from compiler.frontend.source_functions import _validate_resolution, _validate_scope, _walk

        component = getattr(binding, "component_object", None)
        if component is not None:
            from compiler.frontend.component_bindings import _typename, component_access
            from compiler.frontend.source_objects import _TypeBuilder

            access = component_access(self.analysis, self.routine.scope, binding.component_selector)
            if (access is None or access.root_binding is not component or access.binding.public() != binding.public()
                    or len(binding.shape_nodes) != binding.rank or len(binding.lower_bound_nodes) != binding.rank):
                raise CompilationError("numerical effect original component bound authority changed")
            authority = _TypeBuilder(self.analysis)
            schema = authority.binding(component)
            for index, name in enumerate(access.path):
                self.tick()
                if index >= self.analysis.depth_limit:
                    raise CompilationError("numerical effect original component depth exhausted")
                for _member in schema.members:
                    self.tick()
                member = schema.members.get(name)
                if member is None:
                    raise CompilationError("numerical effect original component declaration changed")
                if index + 1 < len(access.path):
                    nested = _typename(member[0])
                    schema = authority.resolve(schema.scope, nested) if nested else None
                    if schema is None:
                        raise CompilationError("numerical effect original component type authority changed")
            self.tick(authority.visits)
            for axis, original, lower, original_lower in zip(binding.shape_nodes, access.binding.shape_nodes,
                    binding.lower_bound_nodes, access.binding.lower_bound_nodes, strict=True):
                self.tick()
                if axis is not original or lower is not original_lower:
                    raise CompilationError("numerical effect original shape/lower-bound node authority changed")
            # ComponentAccess already evaluated these fixed declaration
            # constants in the original defining type's kind namespace.
            self.tick(binding.rank)
            lowers = tuple(integer_literal(value, SourceLocation(str(self.routine.scope.path)))
                           for value in access.binding.lower_bounds)
            return lowers, False, "original_dummy"

        owner = binding.declaring_scope
        if (owner is None or len(binding.lower_bound_nodes) != binding.rank
                or len(binding.shape_nodes) != binding.rank):
            raise CompilationError("numerical effect requires original declared lower-bound authority")
        self.tick()
        _validate_scope(self.analysis, owner)
        originals = self.original_specifications.get(id(owner))
        if originals is None:
            specification = next((node for node in getattr(owner.node, "content", ())
                                  if type(node).__name__ == "Specification_Part"), None)
            originals = {id(node): node for node in source_walk(specification, self)}
            self.original_specifications[id(owner)] = originals
        for axis, lower in zip(binding.shape_nodes, binding.lower_bound_nodes, strict=True):
            self.tick()
            original_lower = axis.items[1] if type(axis).__name__ == "Assumed_Size_Spec" else axis.items[0]
            if originals.get(id(axis)) is not axis or lower is not original_lower:
                raise CompilationError("numerical effect original shape/lower-bound node authority changed")
        if "allocatable" in binding.attributes:
            return (None,) * binding.rank, True, "allocation"
        lowers, dynamic = [], False
        for dimension, node in enumerate(binding.lower_bound_nodes, 1):
            self.tick()
            if node is None:
                lowers.append(1)
                continue
            _validate_resolution(self.analysis, owner, _walk(node), tick=lambda _role: self.tick())
            source = _bound(self.analysis, owner, node, binding, dimension)
            if not _constant_bound(source):
                # Ordinary dummy specification expressions ran at procedure
                # entry. Use the original reached descriptor; never evaluate
                # a mutable lower-bound scalar again at this numerical group.
                if binding.name not in self.routine.arguments:
                    raise NumericalEffectUnsupported("numerical effect local dynamic bounds require an original descriptor mapping")
                dynamic = True
                lowers.append(None)
                continue
            value = owner.kinds.integer(node, SourceLocation(str(owner.path)))
            lowers.append(integer_literal(str(value), SourceLocation(str(owner.path))))
        return ((None,) * binding.rank, True, "original_dummy") if dynamic else (tuple(lowers), False, "original_dummy")

    def resource(self, footprint, unit, node, iterations):
        self.tick()
        parameter = self.symbols.get(footprint.symbol)
        if parameter is None or not parameter.rank or parameter.lower_bound_dimension is not None:
            raise CompilationError("numerical effect physical resource correspondence changed")
        binding = self.binding(parameter.resource)
        if binding.rank != parameter.rank or not 1 <= binding.rank <= 4:
            raise NumericalEffectUnsupported("numerical effect original resource rank is unsupported")
        if footprint.full_read or footprint.full_write:
            raise NumericalEffectUnsupported("numerical effect requires bounded physical read/write footprints")
        reads = self.rectangles(footprint.reads, unit, binding, node, iterations)
        writes = self.rectangles(footprint.writes, unit, binding, node, iterations)
        lowers, descriptor, origin = self.declared_lowers(binding)
        result = NativeResourceSections(binding.root, binding.rank, lowers,
                                        reads, writes, writes if footprint.exact else (),
                                        descriptor_lower_bounds=descriptor, descriptor_origin=origin)
        if any(item.kind == "descriptor_read" and item.resource != binding.root for item in result.dependencies):
            raise NumericalEffectUnsupported("numerical effect cross-resource descriptor bounds require separate rendering")
        return result


def _derive(analysis, procedure, selection, prepared, consume):
    from compiler.frontend.numerical_preparation import PreparedOriginalRegion

    if (not isinstance(selection, tuple) or not selection or len(selection) > analysis.operation_limit
            or any(isinstance(node, str) for node in selection)):
        raise CompilationError("numerical effects require a bounded exact original statement tuple")
    if not isinstance(prepared, PreparedOriginalRegion):
        raise CompilationError("numerical effects require compiler-issued original preparation")
    prepared.validate(analysis, procedure, selection, consume=consume)
    if not prepared.all_runtime_guards_typed:
        raise NumericalEffectUnsupported("numerical effect requires typed authority for every original runtime guard")
    routine = analysis._require_original(procedure)
    graph, identities = analysis._selected_source(procedure, selection)
    if not prepared.offload.available:
        raise NumericalEffectUnsupported("numerical effect physical analysis unavailable: "
                                         + str(prepared.offload.reason))
    units = prepared.offload.units
    if not units or len(units) > analysis.operation_limit:
        raise CompilationError("numerical effect original unit budget exhausted")
    mapper = _Mapper(analysis, routine, prepared, consume)
    steps, accumulated, unit_records = [], {}, []
    scalar_changes = tuple(prepared.scalar_changes)
    if len(scalar_changes) > analysis.operation_limit:
        raise CompilationError("numerical effect original scalar-change budget exhausted")
    for position, unit in enumerate(units):
        mapper.tick()
        if unit.index != position:
            raise CompilationError("numerical effect physical unit order changed")
        iterations = mapper.iterations(unit)
        resources = tuple(mapper.resource(footprint, unit, selection[0], iterations) for footprint in unit.footprints)
        if len(resources) > analysis.operation_limit or len({item.resource for item in resources}) != len(resources):
            raise CompilationError("numerical effect physical resource budget or identity changed")
        sections = NativeSections(True, resources=resources)
        steps.append(NativeDefinitionStep(identities[0], "numerical_unit", graph.nodes[identities[0]].guard,
                                          sections, scalar_changes))
        for resource in resources:
            mapper.tick()
            previous = accumulated.get(resource.resource)
            if previous is None:
                accumulated[resource.resource] = resource
            else:
                if (previous.rank, previous.lower_bounds, previous.descriptor_lower_bounds, previous.descriptor_origin
                        ) != (resource.rank, resource.lower_bounds, resource.descriptor_lower_bounds,
                              resource.descriptor_origin):
                    raise CompilationError("numerical effect original coordinate origin changed across units")
                accumulated[resource.resource] = replace(resource,
                    reads=_unique((*previous.reads, *resource.reads)),
                    writes=_unique((*previous.writes, *resource.writes)),
                    overwrites=_unique((*previous.overwrites, *resource.overwrites)))
        unit_records.append({"unit_index": unit.index, "sections": sections.public()})
    sections = NativeSections(True, resources=tuple(accumulated[root] for root in sorted(accumulated)))
    effects = sum(len(item.reads) + len(item.writes) + len(item.overwrites)
                  for step in steps for item in step.sections.resources) + len(scalar_changes)
    if effects > analysis.operation_limit:
        raise CompilationError("numerical effect physical operation budget exhausted")
    record = {"schema_version": NUMERICAL_EFFECT_VERSION, "proof_role": "prepared_numerical_physical_effects",
              "procedure": procedure, "analysis_identity": graph.authority_identity,
              "structured_identity": graph.identity, "source_identity": analysis.inputs.identity(),
              "selected_node_ids": list(identities), "preparation_identity": prepared.identity,
              "sections": sections.public(), "units": unit_records, "steps": [step.public() for step in steps],
              "resources": [{key: value for key, value in item.public().items()
                             if key not in {"reads", "writes", "overwrites", "dependencies"}}
                            for item in sections.resources],
              "source_demands": [prepared.identity], "entry_definition_changes": [],
              "scalar_changes": list(scalar_changes), "ordered_effect_count": effects,
              "preconditions": [item.public() for item in prepared.preconditions],
              "physical_visits": mapper.work, "rectangle_limit": RECTANGLE_LIMIT,
              "transfer_scope": "only original reached execution with all preparation preconditions true",
              "joined_reads": "every unit uses the same original group-entry definition snapshot",
              "input_definitions_authorized": False, "execution_authorized": False,
              "native_hook_authorized": False, "gpu_legality_established": False,
              "descriptor_lifetime_authorized": False, "alias_authorized": False,
              "unconditional_return_definitions_authorized": False, "bounds_evaluated": False}
    return tuple(steps), sections, record


@dataclass(frozen=True)
class NumericalEffectProof:
    procedure: str
    identity: str
    steps: tuple[NativeDefinitionStep, ...]
    sections: NativeSections
    _analysis: object = field(repr=False, compare=False)
    _selection: tuple = field(repr=False, compare=False)
    _prepared: object = field(repr=False, compare=False)
    _record: object = field(repr=False, compare=False)

    @property
    def prepared(self):
        return self._prepared

    @property
    def preconditions(self):
        return self._prepared.preconditions

    def public(self):
        return {"proof_identity": self.identity, **_thaw(self._record)}

    def validate(self, analysis, procedure, original_selection, prepared=None, *, consume=None):
        prepared = self._prepared if prepared is None else prepared
        if (analysis is not self._analysis or procedure != self.procedure or prepared is not self._prepared
                or not isinstance(original_selection, tuple) or len(original_selection) != len(self._selection)
                or any(left is not right for left, right in zip(original_selection, self._selection, strict=True))
                or getattr(analysis, "_numerical_effect_proofs", {}).get(self.identity) is not self):
            raise CompilationError("numerical effects lack registered exact original preparation authority")
        derived = _thaw(_freeze(_derive(analysis, procedure, original_selection, prepared, consume)[2]))
        if derived != _thaw(self._record):
            raise CompilationError("numerical effect source or physical preparation authority changed")
        return self

    def transfer(self, entry_rectangles=None, *, consume=None):
        """Definitions only on the original preparation-precondition-true path."""
        self.validate(self._analysis, self.procedure, self._selection, consume=consume)
        return _transfer_steps(self.steps, {item.resource: item.rank for item in self.sections.resources},
                               entry_rectangles, joined=True,
                               operation_limit=self._analysis.operation_limit, consume=consume)


def prove_numerical_effects(analysis, procedure, original_selection, prepared, *, consume=None):
    """Register guarded source-backed physical effects, never initialized facts."""
    steps, sections, record = _derive(analysis, procedure, original_selection, prepared, consume)
    record = _thaw(_freeze(record))
    identity = _hash(record)
    registry = getattr(analysis, "_numerical_effect_proofs", None)
    if registry is None:
        registry = analysis._numerical_effect_proofs = {}
    if identity in registry:
        proof = registry[identity]
        if (not isinstance(proof, NumericalEffectProof) or proof._analysis is not analysis
                or proof.procedure != procedure or proof._prepared is not prepared
                or len(proof._selection) != len(original_selection)
                or any(left is not right for left, right in zip(proof._selection, original_selection, strict=True))
                or _thaw(proof._record) != record):
            raise CompilationError("numerical effect cached proof lacks exact original preparation authority")
        return proof
    if len(registry) >= analysis.procedure_limit:
        raise CompilationError("numerical effect proof registry budget exhausted")
    proof = NumericalEffectProof(procedure, identity, steps, sections, analysis, original_selection,
                                  prepared, _freeze(record))
    registry[identity] = proof
    return proof
