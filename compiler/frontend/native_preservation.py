"""Source-bound preservation of existing native definition state.

This is a dormant-state transfer, not host/device coherence. Ordinary native
writes preserve already defined points, even through aliases; they do not grant
new definitions. Entry undefinition and unknown effects remove coverage. Bounds
retain their original scalar and descriptor dependencies. No address, condition,
or payload is evaluated by this analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType

from compiler.frontend.native_definitions import _rectangles
from compiler.frontend.native_demand import _Declarations, _separation, _validate_context
from compiler.frontend.native_group_structure import NATIVE_GROUP_SCAN_LIMIT
from compiler.frontend.resource_reachability import _digest, prove_resource_reachability
from compiler.ir import CompilationError

VERSION = 1
_ISSUER = object()


def _same_or_possible_alias(analysis, access, binding, context, declarations):
    native = access.storage
    if native.binding is binding:
        return True
    if not access.external:
        return False
    if context is not None:
        native = replace(native, binding=context.canonical(native.binding))
        binding = context.canonical(binding)
        if native.binding is binding:
            return True
    try:
        _separation(analysis, native, binding, declarations)
    except CompilationError:
        return True
    return False


def _losses(analysis, reachability, demanded, context):
    """Only hazards require separation; ordinary payload writes are harmless."""
    if reachability.barriers:
        return tuple((binding.root, ("unknown native effects",)) for binding in demanded)
    declarations, result, work = _Declarations(), [], 0
    for binding in demanded:
        reasons = set()
        for access in reachability.accesses:
            if access.kind != "undefine" and not (access.kind == "write" and not binding.rank):
                continue
            work += 1
            if work > NATIVE_GROUP_SCAN_LIMIT:
                raise CompilationError("native preservation hazard comparison budget exhausted")
            if _same_or_possible_alias(analysis, access, binding, context, declarations):
                reasons.add("entry undefinition" if access.kind == "undefine" else "scalar value write")
        if reasons:
            result.append((binding.root, tuple(sorted(reasons))))
    return tuple(result)


def _dependencies(box, demanded):
    """Retain descriptor reads as well as scalar reads, with exact bindings."""
    dependencies, pending, work = set(), [], 0
    for axis in box.axes:
        pending.extend(value for value in (axis.lower, axis.upper) if value is not None)
    for iteration in box.iterations:
        pending.extend((iteration.lower, iteration.upper))
    while pending:
        value = pending.pop()
        work += 1
        if work > NATIVE_GROUP_SCAN_LIMIT:
            raise CompilationError("native preservation bound scan budget exhausted")
        pending.extend(value.children)
        local_dependencies = set()
        for dependency in value.dependencies:
            binding = demanded.get(dependency.resource)
            if (binding is None or binding is not dependency.binding
                    or dependency.kind not in {"scalar_read", "descriptor_read"}
                    or dependency.kind == "scalar_read" and binding.rank):
                raise CompilationError("native preservation bound lacks an exact demanded dependency")
            dependencies.add((dependency.resource, dependency.kind))
            local_dependencies.add((dependency.resource, dependency.kind))
        if value.kind in {"scalar", "lbound", "ubound", "size"}:
            kind = "scalar_read" if value.kind == "scalar" else "descriptor_read"
            if (value.resource, kind) not in local_dependencies:
                raise CompilationError("native preservation bound lacks its original dependency")
    return dependencies


@dataclass(frozen=True)
class NativePreservationTransfer:
    definitions: object
    complete: bool
    invalidations: tuple

    def public(self):
        return {"defined_sections": {root: [box.public() for box in boxes]
                                     for root, boxes in self.definitions.items()},
                "complete": self.complete,
                "invalidations": [{"resource": root, "reasons": list(reasons)}
                                  for root, reasons in self.invalidations],
                "new_definition_authority": False, "execution_authorized": False,
                "entry_rectangles_require_independent_admission": True}


@dataclass(frozen=True)
class NativePreservationProof:
    procedure: str
    identity: str
    demanded: tuple = field(repr=False, compare=False)
    reachability: object = field(repr=False, compare=False)
    invalidations: tuple
    _analysis: object = field(repr=False, compare=False)
    _selection: tuple = field(repr=False, compare=False)
    _context: object = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)

    @property
    def complete(self):
        return not self.reachability.barriers

    def _payload(self):
        return {"version": VERSION, "procedure": self.procedure,
                "reachability": self.reachability.public(),
                "demanded": [id(binding) for binding in self.demanded],
                "invalidations": self.invalidations,
                "context": self._context.identity if self._context is not None else None}

    def validate(self, analysis, procedure, originals, demanded):
        if (self._issuer is not _ISSUER or type(self) is not NativePreservationProof
                or analysis is not self._analysis or procedure != self.procedure
                or getattr(analysis, "_native_preservation_registry_owner", None) is not analysis
                or getattr(analysis, "_native_preservation_proofs", {}).get(self.identity) is not self):
            raise CompilationError("native preservation requires exact registered source authority")
        self.reachability.validate(analysis, procedure, originals, demanded)
        _validate_context(self._context, analysis, procedure, originals, demanded)
        if _digest(self._payload()) != self.identity:
            raise CompilationError("native preservation registered payload changed")
        return self

    def transfer(self, coverage):
        self.validate(self._analysis, self.procedure, self._selection, self.demanded)
        if not isinstance(coverage, (dict, MappingProxyType)):
            raise CompilationError("native preservation needs an original resource coverage mapping")
        bindings = {binding.root: binding for binding in self.demanded}
        if any(root not in bindings or not bindings[root].rank for root in coverage):
            raise CompilationError("native preservation coverage requires demanded array bindings")
        losses = dict(self.invalidations)
        preserved = {}
        for root, boxes in coverage.items():
            _rectangles(boxes)
            if any(len(box.axes) != bindings[root].rank for box in boxes):
                raise CompilationError("native preservation coverage rank differs from original storage")
            kept = []
            for box in boxes:
                dependencies = _dependencies(box, bindings)
                # Complete graphs cannot change descriptors: those actions are
                # explicit barriers. INTENT(OUT) alone invalidates payload, not
                # a nonallocatable actual's descriptor.
                scalar_changed = any(kind == "scalar_read" and dependency in losses
                                     for dependency, kind in dependencies)
                if self.complete and root not in losses and not scalar_changed:
                    kept.append(box)
            preserved[root] = tuple(kept)
        return NativePreservationTransfer(MappingProxyType(preserved), self.complete, self.invalidations)

    def public(self):
        return {"schema_version": VERSION, "identity": self.identity, "procedure": self.procedure,
                "reachability_identity": self.reachability.identity,
                "interval_context_identity": self._context.identity if self._context is not None else None,
                "complete": self.complete,
                "invalidations": [{"resource": root, "reasons": list(reasons)}
                                  for root, reasons in self.invalidations],
                "semantics": "unchanged native execution preserves only independently admitted entry coverage",
                "ordinary_payload_writes_grant_definitions": False,
                "original_scalar_and_descriptor_dependencies_retained": True,
                "new_definition_authority": False, "execution_authorized": False,
                "physical_mapping_authority": False, "descriptor_lifetime_authority": False,
                "alias_authority": False, "cost_authority": False}


def prove_native_preservation(analysis, procedure, originals, demanded, *, context=None):
    """Issue a source-authenticated conservative filter of entry definitions.

    Unknown effects clear all coverage. This is useful for path-sensitive
    dormant plans, but it never permits an issuer to treat an incomplete native
    leaf as preserving a later optimized entry. Runtime guards and physical
    registration remain separate authorities at their original safe positions.
    """
    _validate_context(context, analysis, procedure, originals, demanded)
    reachability = prove_resource_reachability(analysis, procedure, originals, demanded)
    if len({binding.root for binding in demanded}) != len(demanded):
        raise CompilationError("native preservation demands require distinct original resource roots")
    invalidations = _losses(analysis, reachability, demanded, context)
    value = NativePreservationProof(procedure, "", demanded, reachability, invalidations,
                                    analysis, originals, context, _ISSUER)
    value = replace(value, identity=_digest(value._payload()))
    registry = getattr(analysis, "_native_preservation_proofs", None)
    if registry is None or getattr(analysis, "_native_preservation_registry_owner", None) is not analysis:
        registry = analysis._native_preservation_proofs = {}
        analysis._native_preservation_registry_owner = analysis
    if value.identity in registry:
        return registry[value.identity].validate(analysis, procedure, originals, demanded)
    if len(registry) >= analysis.procedure_limit:
        raise CompilationError("native preservation proof registry budget exhausted")
    registry[value.identity] = value
    try:
        return value.validate(analysis, procedure, originals, demanded)
    except BaseException:
        if registry.get(value.identity) is value:
            del registry[value.identity]
        raise
