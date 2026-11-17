"""Original-source proofs for bounded, read-only parallel reductions.

These records establish source form and effects, not runtime placement. Real
extrema and reassociated sums deliberately retain distinct native contracts.
No routine name supplied by an application grants numerical legality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json

from compiler.frontend.structured_effects import _thaw
from compiler.ir import CompilationError

REDUCTION_PROOF_VERSION = 1
_REDUCTIONS = frozenset({"minval", "maxval", "all", "any", "sum"})
_IEEE_NAN = "$intrinsic::ieee_arithmetic::ieee_is_nan"


def _kind(node):
    return type(node).__name__


def _children(node):
    return getattr(node, "content", getattr(node, "items", ())) or ()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class ReductionProof:
    procedure: str
    node_id: str
    structured_identity: str
    identity: str
    available: bool
    reason: str | None
    _record: object = field(repr=False, compare=False)
    original_node: object = field(repr=False, compare=False)

    def public(self):
        return {"schema_version": REDUCTION_PROOF_VERSION, "procedure": self.procedure,
                "node_id": self.node_id, "structured_identity": self.structured_identity,
                "proof_identity": self.identity, "available": self.available,
                "source_analysis_available": self.available,
                "execution_supported": False,
                "execution_reason": "source proof only; no generated reduction worker or calibrated placement",
                "reason": self.reason, **_thaw(self._record)}

    def validate(self, analysis):
        """Recheck authority before attaching a generated implementation."""
        graph = analysis.structure(self.procedure)
        if (analysis._reduction_proofs.get(self.identity) is not self
                or graph.identity != self.structured_identity or graph.node_id(self.original_node) != self.node_id
                or str(self.original_node) != self._record["original_source"]):
            raise CompilationError("reduction proof no longer has original source authority")
        return self


def _function(node):
    while _kind(node) == "Parenthesis":
        node = node.items[1]
    if _kind(node) not in {"Intrinsic_Function_Reference", "Function_Reference", "Part_Ref"}:
        raise CompilationError("reduction requires one original intrinsic assignment")
    return node, str(node.items[0]).lower(), tuple(_children(node.items[1]))


def _arguments(original, formals):
    result, order, position, keyword_seen = {}, [], 0, False
    for item in original:
        if _kind(item) == "Actual_Arg_Spec":
            keyword_seen = True
            formal, value = str(item.items[0]).lower(), item.items[1]
            keyword = str(item.items[0])
        else:
            if keyword_seen or position >= len(formals):
                raise CompilationError("reduction arguments must retain valid original keyword order")
            formal, value, keyword = formals[position], item, None
            position += 1
        if formal not in formals or formal in result:
            raise CompilationError("reduction has an unknown or duplicate argument: " + formal)
        result[formal] = value
        order.append({"formal": formal, "keyword": keyword, "source": str(value)})
    if formals[0] not in result:
        raise CompilationError("reduction is missing its original array argument")
    return result, order


def _reference(analysis, scope, node):
    while _kind(node) == "Parenthesis":
        node = node.items[1]
    if _kind(node) == "Data_Ref":
        from compiler.frontend.component_bindings import component_access
        access = component_access(analysis, scope, node)
        if access is None:
            raise CompilationError("reduction component storage is unresolved")
        binding, indices = access.binding, access.indices
    elif _kind(node) == "Part_Ref":
        binding, indices = analysis._binding(scope, node.items[0]), tuple(_children(node.items[1]))
    elif _kind(node) == "Name":
        binding, indices = analysis._binding(scope, node), ()
    else:
        raise CompilationError("reduction value requires a whole array or proven rectangular section")
    if binding is None or not binding.rank:
        raise CompilationError("reduction value requires a source-backed array")
    if binding.attributes & {"pointer", "optional", "volatile", "asynchronous"}:
        raise CompilationError("reduction array association or presence requires additional proof")
    if indices:
        if len(indices) != binding.rank:
            raise CompilationError("reduction section rank differs from original storage")
        rank = sum(_kind(index) == "Subscript_Triplet" for index in indices)
    else:
        rank = binding.rank
    if rank < 1:
        raise CompilationError("reduction array argument is scalar")
    reason = analysis.resource_identity_boundary(binding)
    if reason:
        raise CompilationError(reason)
    return node, binding, rank
