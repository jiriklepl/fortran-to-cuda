"""Exact values required by a well-defined original reached numerical group.

These are conditional preconditions, not a proof that an earlier native prefix
initialized anything.  Only unconditional source reads preceding every write
to the same storage are eligible.  Original preparation guards, storage,
lifetime, alias and execution authority remain separate requirements.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from compiler.frontend.native_definitions import _covers, _unique
from compiler.frontend.native_sections import NativeSections
from compiler.frontend.numerical_effects import _hash, _Mapper, prove_numerical_effects
from compiler.frontend.structured_effects import _freeze, _thaw
from compiler.ir import ArrayAccess, Assignment, Binary, CompilationError, IntrinsicCall, ScalarType, walk_expr
from compiler.offload.analysis import _access_box, _index_terms, _merge_boxes, _resolve

REACHED_READ_REQUIREMENTS_VERSION = 1


class ReachedReadRequirementsUnsupported(CompilationError):
    """Original reads cannot justify the numerical upload requirement exactly."""


@dataclass(frozen=True)
class ReachedReadUnit:
    index: int
    sections: NativeSections

    def public(self):
        return {"unit_index": self.index, "sections": self.sections.public()}


def _upward_reads(unit, mapper):
    """Use the same affine mapper as physical analysis, retaining source order."""
    definitions, written, reads, writes = {}, set(), {}, {}
    iterators = {loop.iterator: index for index, loop in enumerate(unit.region.loops)}

    def read(expression):
        for node in walk_expr(expression):
            mapper.tick()
            if (isinstance(node, IntrinsicCall) and node.name.lower() == "merge"
                    or isinstance(node, Binary) and node.operator.lower() in {".and.", ".or.", "and", "or"}):
                raise ReachedReadRequirementsUnsupported(
                    "reached reads cannot assume operands of a conditional expression are evaluated")
            if not isinstance(node, ArrayAccess) or node.symbol in written:
                continue
            box = _access_box(_resolve(node, definitions), unit, mapper.parameters)
            if box is None:
                raise ReachedReadRequirementsUnsupported("reached reads require exact affine source accesses")
            reads.setdefault(node.symbol, []).append(box)

    for statement in unit.region.body.statements:
        mapper.tick()
        if not isinstance(statement, Assignment):
            raise ReachedReadRequirementsUnsupported(
                "reached reads require unconditional assignments; conditional and retained-loop work remains native")
        read(statement.value)
        if isinstance(statement.target, ArrayAccess):
            for index in statement.target.indices:
                read(index)
            box = _access_box(_resolve(statement.target, definitions), unit, mapper.parameters)
            if box is None:
                raise ReachedReadRequirementsUnsupported("reached reads require exact affine source writes")
            writes.setdefault(statement.target.symbol, []).append(box)
            written.add(statement.target.symbol)
        else:
            symbol = statement.target.symbol
            value = _resolve(statement.value, definitions)
            if symbol.dtype is ScalarType.INTEGER and _index_terms(value, iterators, mapper.parameters) is not None:
                definitions[symbol] = value
            else:
                definitions.pop(symbol, None)
    # A preceding iteration must not be able to supply a supposedly incoming
    # value.  Exact per-iteration self updates are sufficient; shifted reads or
    # multiple target mappings need a separate dependence proof.
    for symbol in reads.keys() & writes.keys():
        mappings = {box.axes for box in (*reads[symbol], *writes[symbol])}
        if len(mappings) != 1:
            raise ReachedReadRequirementsUnsupported(
                "reached read/write storage requires identical per-iteration self-update mappings")
    return {symbol: _merge_boxes(boxes) for symbol, boxes in reads.items()}, set(writes)


def _derive(analysis, procedure, selection, prepared, consume):
    physical = prove_numerical_effects(analysis, procedure, selection, prepared, consume=consume)
    mapper = _Mapper(analysis, analysis.routines[procedure], prepared, consume)
    units, accesses, accumulated = [], [], {}
    for unit in prepared.offload.units:
        upward, written = _upward_reads(unit, mapper)
        accesses.append((set(upward), written))
        resources = []
        iterations = mapper.iterations(unit)
        for footprint in unit.footprints:
            mapper.tick()
            if footprint.full_upload or not footprint.exact:
                raise ReachedReadRequirementsUnsupported(
                    "reached reads cannot authorize whole or synthetic preservation uploads")
            resource = mapper.resource(footprint, unit, selection[0], iterations)
            binding = mapper.binding(resource.resource)
            reads = mapper.rectangles(upward.get(footprint.symbol, ()), unit, binding, selection[0], iterations)
            uploads = mapper.rectangles(footprint.uploads, unit, binding, selection[0], iterations)
            if any(not any(_covers(read, upload, consume=mapper.tick) for read in reads) for upload in uploads):
                raise ReachedReadRequirementsUnsupported(
                    "numerical upload is not covered by exact original reads before writes: " + resource.resource)
            if not reads:
                continue
            resource = replace(resource, reads=reads, writes=(), overwrites=())
            resources.append(resource)
            previous = accumulated.get(resource.resource)
            if previous is not None:
                resource = replace(resource, reads=_unique((*previous.reads, *resource.reads)))
            accumulated[resource.resource] = resource
        units.append(ReachedReadUnit(unit.index, NativeSections(True, resources=tuple(resources))))
    # Every unit's requirement refers to the same original joined-group entry.
    # Never derive incoming values from a preceding unit, including NOWAIT.
    for index, (reads, writes) in enumerate(accesses):
        for other_reads, other_writes in accesses[index + 1:]:
            mapper.tick()
            if writes & (other_reads | other_writes) or other_writes & reads:
                raise ReachedReadRequirementsUnsupported(
                    "reached joined units require independent array reads and writes at group entry")
    sections = NativeSections(True, resources=tuple(accumulated[root] for root in sorted(accumulated)))
    record = {"schema_version": REACHED_READ_REQUIREMENTS_VERSION,
              "proof_role": "exact_original_reached_read_preconditions", "procedure": procedure,
              "source_identity": analysis.inputs.identity(), "preparation_identity": prepared.identity,
              "numerical_effect_identity": physical.identity,
              "units": [unit.public() for unit in units], "sections": sections.public(),
              "preconditions": [item.public() for item in prepared.preconditions],
              "physical_visits": mapper.work,
              "validity": "well-defined original reached execution after all original preparation guards",
              "joined_reads": "independent units share the original group-entry snapshot",
              "prefix_definitions_authorized": False, "capture_authorized": False,
              "descriptor_lifetime_authorized": False, "alias_authorized": False,
              "execution_authorized": False, "bounds_evaluated": False}
    return tuple(units), sections, _thaw(_freeze(record))


@dataclass(frozen=True)
class ReachedReadRequirementsProof:
    procedure: str
    identity: str
    units: tuple[ReachedReadUnit, ...]
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
                or getattr(analysis, "_reached_read_requirements_proofs_owner", None) is not analysis
                or getattr(analysis, "_reached_read_requirements_proofs", {}).get(self.identity) is not self):
            raise CompilationError("reached read requirements lack registered exact original preparation authority")
        _units, _sections, record = _derive(analysis, procedure, original_selection, prepared, consume)
        if (record != _thaw(self._record) or self.identity != _hash(record)
                or [unit.public() for unit in self.units] != record["units"]
                or self.sections.public() != record["sections"]):
            raise CompilationError("reached read source or requirement payload authority changed")
        bindings = {item.root: item for item in prepared.extraction.bindings}

        def bound_authority(bound):
            if bound is None:
                return
            for dependency in bound.dependencies:
                if dependency.binding is not bindings.get(dependency.resource):
                    raise CompilationError("reached read original dependency binding authority changed")
            for child in bound.children:
                bound_authority(child)

        for sections in (self.sections, *(unit.sections for unit in self.units)):
            for resource in sections.resources:
                if any(box.node is not original_selection[0] for box in resource.reads):
                    raise CompilationError("reached read original access authority changed")
                for box in resource.reads:
                    for axis in (*box.axes, *box.iterations):
                        bound_authority(axis.lower)
                        bound_authority(axis.upper)
        return self


def prove_reached_read_requirements(analysis, procedure, original_selection, prepared, *, consume=None):
    """Issue exact reached-read preconditions, never facts about an opaque prefix."""
    units, sections, record = _derive(analysis, procedure, original_selection, prepared, consume)
    identity = _hash(record)
    registry = getattr(analysis, "_reached_read_requirements_proofs", None)
    if registry is None:
        registry = analysis._reached_read_requirements_proofs = {}
        analysis._reached_read_requirements_proofs_owner = analysis
    if getattr(analysis, "_reached_read_requirements_proofs_owner", None) is not analysis:
        raise CompilationError("reached read requirements cannot borrow another source session registry")
    if identity in registry:
        return registry[identity].validate(analysis, procedure, original_selection, prepared, consume=consume)
    if len(registry) >= analysis.procedure_limit:
        raise CompilationError("reached read requirement proof registry budget exhausted")
    proof = ReachedReadRequirementsProof(procedure, identity, units, sections, analysis, original_selection,
                                         prepared, _freeze(record))
    registry[identity] = proof
    return proof
