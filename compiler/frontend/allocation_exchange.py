"""Exact source authority for a completed three-MOVE_ALLOC exchange.

The proof changes descriptor association only. It grants no payload definitions,
GPU placement, initialized storage, or permission to generalize unknown mutation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256

from compiler.frontend.call_bindings import resolve_original_call
from compiler.frontend.source_effects import _children, _kind
from compiler.ir import CompilationError

_ISSUER = object()
_FORBIDDEN = {"pointer", "optional", "volatile", "asynchronous", "value", "save", "target", "protected", "parameter"}


def _body(routine):
    nodes = tuple(node for node in _children(routine.execution) if _kind(node) != "Comment")
    comments = tuple(node for node in _children(routine.execution) if _kind(node) == "Comment")
    if any(str(node).lstrip().startswith("!$") for node in comments):
        return ()
    return nodes


def _looks_like_exchange(routine):
    body = _body(routine)
    return len(body) == 3 and all(_kind(node) == "Call_Stmt" and str(node.items[0]).lower() == "move_alloc"
                                  for node in body)


def _derive(analysis, routine, node, resolved):
    resolved.validate(analysis, analysis.source_scope_for(node, routine.scope), node)
    child = resolved.callee
    if not _looks_like_exchange(child) or len(child.arguments) != 2 or len(resolved.associations) != 2:
        raise CompilationError("allocation exchange requires exactly three original MOVE_ALLOC calls")
    if (analysis._binding(child.scope, "move_alloc") is not None
            or analysis._candidates(child.scope, "move_alloc") or analysis._unknown_exports(child.scope)):
        raise CompilationError("allocation exchange requires the unshadowed MOVE_ALLOC intrinsic")
    formals = tuple(child.scope.bindings[name] for name in child.arguments)
    locals_ = tuple(binding for name, binding in child.scope.bindings.items() if name not in child.arguments)
    if len(locals_) != 1:
        raise CompilationError("allocation exchange requires one automatic temporary")
    temporary, = locals_
    if any(binding.rank < 1 or "allocatable" not in binding.attributes or binding.attributes & _FORBIDDEN
           or binding.signature() != formals[0].signature() for binding in (*formals, temporary)):
        raise CompilationError("allocation exchange requires matching ordinary allocatable arrays")
    if any(binding.intent != "inout" for binding in formals):
        raise CompilationError("allocation exchange requires INOUT formals")
    if formals[0].signature()[:2] not in {("real", 4), ("real", 8), ("integer", 4)}:
        raise CompilationError("allocation exchange element type is unsupported")
    # An initialized or saved temporary is not the automatic unallocated
    # descriptor whose three moves exchange the two incoming allocations.
    from compiler.frontend.source_effects import _part
    for declaration in _children(_part(child.scope.node, "Specification_Part")):
        if _kind(declaration) in {"Save_Stmt", "Data_Stmt", "Common_Stmt", "Equivalence_Stmt"}:
            raise CompilationError("allocation exchange has persistent or associated local state")
        if (_kind(declaration) == "Type_Declaration_Stmt"
                and any(entity.items[3] is not None for entity in _children(declaration.items[2]))):
            raise CompilationError("allocation exchange has initialized local state")
    expected = ((formals[0].name, temporary.name), (formals[1].name, formals[0].name),
                (temporary.name, formals[1].name))
    for call, pair in zip(_body(child), expected, strict=True):
        arguments = tuple(_children(call.items[1]))
        if (len(arguments) != 2 or any(_kind(item) != "Name" for item in arguments)
                or tuple(str(item).lower() for item in arguments) != pair):
            raise CompilationError("allocation exchange MOVE_ALLOC order or arguments are unsupported")
    actuals = tuple(item.binding for item in resolved.associations)
    if any(binding is None or _kind(item.actual) != "Name" or "allocatable" not in binding.attributes
           or binding.attributes & _FORBIDDEN or binding.signature() != formals[0].signature()
           for item, binding in zip(resolved.associations, actuals, strict=True)):
        raise CompilationError("allocation exchange requires exact whole allocatable actuals")
    if actuals[0].root == actuals[1].root:
        raise CompilationError("allocation exchange actuals must have distinct canonical storage slots")
    graph, selected = analysis._selected_source(routine.qualified, (node,))
    return {"version": 1, "procedure": routine.qualified, "source_graph": graph.identity,
            "selected_node_ids": list(selected), "callee": child.qualified,
            "source_inputs": analysis.inputs.identity(), "call": resolved.public(),
            "body": [str(item) for item in _body(child)], "resources": [binding.root for binding in actuals],
            "transition": "exchange allocation associations; physical generations and replicas survive",
            "payload_reads": False, "payload_writes": False, "initialization_authorized": False,
            "runtime_guard": "both current layouts match opposite live allocations after the original call",
            "on_unproved_mutation": "publish and retire before original source operation"}


def _encoded(record):
    return json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True, slots=True)
class AllocationExchangeProof:
    identity: str
    procedure: str
    bindings: tuple
    _analysis: object = field(repr=False, compare=False)
    _routine: object = field(repr=False, compare=False)
    _node: object = field(repr=False, compare=False)
    _resolved: object = field(repr=False, compare=False)
    _record: str = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)

    def validate(self, analysis, routine, node):
        if (self._issuer is not _ISSUER or analysis is not self._analysis or routine is not self._routine
                or node is not self._node or getattr(analysis, "_allocation_exchange_proofs", {}).get(self.identity) is not self):
            raise CompilationError("allocation exchange requires its exact compiler-issued original proof")
        if _encoded(_derive(analysis, routine, node, self._resolved)) != self._record:
            raise CompilationError("allocation exchange source or association authority changed")
        if any(binding is not item.binding for binding, item in zip(self.bindings, self._resolved.associations, strict=True)):
            raise CompilationError("allocation exchange original binding identity changed")
        return self

    def public(self):
        return {"identity": self.identity, **json.loads(self._record)}


def allocation_exchange(analysis, routine, node):
    """Return a proved narrow exchange, or leave other calls to normal analysis."""
    if _kind(node) != "Call_Stmt" or _kind(node.items[0]) != "Name":
        return None
    scope = analysis.source_scope_for(node, routine.scope)
    candidates = analysis._candidates(scope, str(node.items[0]).lower())
    if not any(name in analysis.routines and _looks_like_exchange(analysis.routines[name]) for name in candidates):
        return None
    for proof in getattr(analysis, "_allocation_exchange_proofs", {}).values():
        if proof._node is node:
            return proof.validate(analysis, routine, node)
    resolved = resolve_original_call(analysis, scope, node)
    encoded = _encoded(_derive(analysis, routine, node, resolved))
    proof = AllocationExchangeProof(sha256(encoded.encode()).hexdigest(), routine.qualified,
        tuple(item.binding for item in resolved.associations), analysis, routine, node, resolved, encoded, _ISSUER)
    registry = getattr(analysis, "_allocation_exchange_proofs", None)
    if registry is None:
        registry = analysis._allocation_exchange_proofs = {}
    registry[proof.identity] = proof
    return proof.validate(analysis, routine, node)
