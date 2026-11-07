"""Original, call-free specification reads before a checked procedure entry.

This is a source projection, not permission to enter a child or read a buffer.
The caller must separately bind its canonical allocations and acquire the reads
before Fortran evaluates the original bounds. Body completion, scratch lifetime,
alias separation and initialization are deliberately not inferred here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256

from compiler.frontend.resource_reachability import ResourceAccess, _children, _kind, _limits, _Scan
from compiler.frontend.source_functions import (
    _original,
    _scope_roles,
    _validate_scope,
    _validate_scope_roles,
    _walk,
)
from compiler.ir import CompilationError

_ISSUER = object()
_UNSAFE = {"pointer", "target", "optional", "volatile", "asynchronous", "value", "codimension"}
_ALIASES = {"Common_Stmt", "Equivalence_Stmt", "Namelist_Stmt", "Cray_Pointer_Stmt"}
_EXPRESSIONS = {"Parenthesis", "Level_2_Expr", "Level_2_Unary_Expr", "Add_Operand", "Mult_Operand"}
_SHAPES = {"Explicit_Shape_Spec", "Assumed_Shape_Spec", "Deferred_Shape_Spec", "Assumed_Size_Spec"}


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _expression(scan, routine, node, depth=0):
    """Restrict entry expressions before using the existing effect scanner."""
    if node is None or isinstance(node, (str, int)):
        return
    if depth > scan.analysis.depth_limit:
        raise CompilationError("specification reads exceed the original expression depth budget")
    scan.tick()
    kind = _kind(node)
    if kind in {"Int_Literal_Constant", "Signed_Int_Literal_Constant"}:
        return
    if kind in {"Name", "Part_Ref"}:
        scope = scan.scope(scan.analysis.source_scope_for(node, routine.scope))
        name = node.items[0] if kind == "Part_Ref" else node
        binding = scan.analysis._binding(scope, name)
        if (binding is None or binding.dtype != "integer" or binding.attributes & _UNSAFE
                or binding.declaring_scope is routine.scope and binding.name not in routine.arguments
                and "parameter" not in binding.attributes):
            raise CompilationError("specification reads require original independent integer input bindings")
        scan.scope(binding.declaring_scope)
        if kind == "Name":
            if binding.rank:
                raise CompilationError("specification reads require a scalar original bound value")
            return
        indices = _children(node.items[1])
        if not binding.rank or len(indices) != binding.rank:
            raise CompilationError("specification reads reject source functions and non-scalar array references")
        for index in indices:
            if _kind(index) in {"Subscript_Triplet", "Vector_Subscript"}:
                raise CompilationError("specification reads require scalar original array subscripts")
            _expression(scan, routine, index, depth + 1)
        return
    if kind not in _EXPRESSIONS | _SHAPES | {"list", "tuple"}:
        raise CompilationError("specification reads require call-free original integer bounds")
    for child in _children(node):
        _expression(scan, routine, child, depth + 1)


def _derive(analysis, routine):
    if (getattr(analysis, "_descriptor_source", None) is not None
            or _original(analysis, routine.qualified) is not routine):
        raise CompilationError("specification reads require the exact original procedure")
    inputs, identity, roles = analysis.inputs, analysis.inputs.identity(), _scope_roles(analysis)
    scan = _Scan(analysis, ())
    scan.scope(routine.scope)
    specification = next((node for node in _children(routine.scope.node)
                          if _kind(node) == "Specification_Part"), None)
    originals = tuple(_walk(specification))
    if any(_kind(node) in _ALIASES or _kind(node) == "Comment" and str(node).lstrip().startswith("!$")
           for node in originals):
        raise CompilationError("specification reads require original unaliased declaration storage")
    declarations = []
    for binding in routine.scope.bindings.values():
        if binding.dtype not in {"integer", "real", "logical"} or binding.attributes & _UNSAFE:
            raise CompilationError("specification reads require original numeric declarations")
        for bound in binding.shape_nodes:
            _expression(scan, routine, bound)
        scan.frame("declaration", tuple(binding.shape_nodes),
                   lambda binding=binding: [scan.expression(routine, bound, {}) for bound in binding.shape_nodes],
                   declaration=binding)
        declarations.append(binding)
    scan.specification_authority(routine)
    if (scan.barriers or scan.host_blockers or scan.family_blockers or scan.template_calls
            or scan.terminal_calls or scan.host_calls or scan.completion
            or any(type(item) is not ResourceAccess or item.kind != "read" or item.storage.components
                   for node in scan.nodes for item in node.items)):
        raise CompilationError("specification reads have unproved original calls, aliases or effects")
    accesses = tuple(scan.accesses)
    record = {"version": 1, "kind": "original_call_free_specification_reads", "procedure": routine.qualified,
              "source_inputs": identity, "original_procedure_sha256": sha256(str(routine.scope.node).encode()).hexdigest(),
              "declarations": [node.public() for node in scan.nodes],
              "array_resources": sorted({item.storage.resource for item in accesses if item.storage.rank}),
              "phase": "before original procedure entry and specification evaluation",
              "bounds_evaluation": "original Fortran only; this projection does not evaluate expressions",
              "region_authority": "storage may-read projection; no subscript bounds or exact region proof",
              "requires_caller_association_and_coherence": True,
              "definition_authority": False, "alias_authority": False, "lifetime_authority": False,
              "body_completion_authority": False, "payload_access_authorized": False, "execution_authorized": False}
    encoded = _encoded(record)
    # Serialization may execute callbacks. Recheck the consumed source objects,
    # namespace roles and live inputs before publishing any retained authority.
    current = tuple(_walk(specification))
    if (len(current) != len(originals) or any(a is not b for a, b in zip(current, originals, strict=True))
            or _original(analysis, routine.qualified) is not routine
            or any(routine.scope.bindings.get(binding.name) is not binding for binding in declarations)):
        raise CompilationError("specification reads changed original declaration authority before publication")
    _validate_scope_roles(analysis, roles, scan.scopes)
    for scope in scan.scopes.values():
        # scope() remembers prior validation; the final fence must revalidate.
        _validate_scope(analysis, scope)
    if analysis.inputs is not inputs or analysis.inputs.identity() != identity:
        raise CompilationError("specification reads changed source inputs before publication")
    analysis.verify_publication()
    return encoded, originals, accesses


@dataclass(frozen=True, slots=True)
class SpecificationReadsProof:
    identity: str
    accesses: tuple = field(repr=False, compare=False)
    _analysis: object = field(repr=False, compare=False)
    _routine: object = field(repr=False, compare=False)
    _record: str = field(repr=False, compare=False)
    _originals: tuple = field(repr=False, compare=False)
    _limits: tuple = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)

    def validate(self, analysis, routine):
        if (self._issuer is not _ISSUER or analysis is not self._analysis or routine is not self._routine
                or self._limits != _limits(analysis)
                or getattr(analysis, "_specification_reads", {}).get(self.identity) is not self):
            raise CompilationError("specification reads require their exact compiler-issued authority")
        record, originals, accesses = _derive(analysis, routine)
        if (record != self._record or len(originals) != len(self._originals)
                or any(a is not b for a, b in zip(originals, self._originals, strict=True))
                or len(accesses) != len(self.accesses)
                or any(a.original is not b.original or a.storage.binding is not b.storage.binding
                       or a.public() != b.public() for a, b in zip(accesses, self.accesses, strict=True))
                or getattr(analysis, "_specification_reads", {}).get(self.identity) is not self):
            raise CompilationError("specification reads changed original source, occurrences or bindings")
        analysis.verify_publication()
        return self

    def public(self):
        return {"identity": self.identity, **json.loads(self._record)}


def prove_specification_reads(analysis, routine):
    registry = getattr(analysis, "_specification_reads", None)
    if registry is None:
        registry = analysis._specification_reads = {}
    for proof in tuple(registry.values()):
        if proof._routine is routine:
            return proof.validate(analysis, routine)
    record, originals, accesses = _derive(analysis, routine)
    if getattr(analysis, "_specification_reads", None) is not registry:
        raise CompilationError("specification reads changed their issuer registry before publication")
    proof = SpecificationReadsProof(sha256(record.encode()).hexdigest(), accesses, analysis, routine,
                                    record, originals, _limits(analysis), _ISSUER)
    registry[proof.identity] = proof
    return proof.validate(analysis, routine)
