"""Original scalar SYSTEM_CLOCK calls retain host effects and completion order."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256

from fparser.two.utils import walk

from compiler.frontend.source_effects import _children, _kind, _part
from compiler.ir import CompilationError

_ISSUER = object()


def _encoded(record):
    return json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _origin(analysis):
    seen = set()
    while getattr(analysis, "_descriptor_source", None) is not None:
        if id(analysis) in seen:
            raise CompilationError("SYSTEM_CLOCK source projection is cyclic")
        seen.add(id(analysis))
        analysis = analysis._descriptor_source
    return analysis


def is_system_clock(analysis, scope, node):
    if _kind(node) != "Call_Stmt" or _kind(node.items[0]) != "Name" or str(node.items[0]).lower() != "system_clock":
        return False
    owner, shadowed = scope, False
    while owner is not None:
        shadowed |= any("system_clock" in names for names in (
            owner.externals, owner.procedure_arguments, owner.procedures, owner.generics, owner.ambiguous_imports))
        owner = owner.parent
    if (shadowed or analysis._binding(scope, "system_clock") or analysis._candidates(scope, "system_clock")
            or analysis._unknown_exports(scope)):
        raise CompilationError("SYSTEM_CLOCK requires its unshadowed standard intrinsic binding")
    return True


def _derive(analysis, procedure, node):
    routine = analysis._require_original(procedure)
    scope = analysis.source_scope_for(node, routine.scope)
    if not is_system_clock(analysis, scope, node):
        raise CompilationError("scalar clock proof requires an original SYSTEM_CLOCK call")
    if any(_kind(item) in {"Common_Stmt", "Equivalence_Stmt"}
           for item in walk(_part(routine.scope.node, "Specification_Part"))):
        raise CompilationError("SYSTEM_CLOCK local outputs require independent storage")
    outputs, position, keyword_seen = {}, 0, False
    for argument in _children(node.items[1]):
        if _kind(argument) == "Actual_Arg_Spec":
            keyword_seen = True
            formal, value = str(argument.items[0]).lower(), argument.items[1]
        else:
            if keyword_seen or position >= 3:
                raise CompilationError("SYSTEM_CLOCK argument association is unsupported")
            formal, value = ("count", "count_rate", "count_max")[position], argument
            position += 1
        if formal not in {"count", "count_rate", "count_max"} or formal in outputs or _kind(value) != "Name":
            raise CompilationError("SYSTEM_CLOCK requires distinct original scalar output arguments")
        binding = analysis._binding(scope, value)
        if (binding is None or binding.declaring_scope is not routine.scope or binding.name in routine.arguments
                or binding.rank
                or binding.dtype not in ({"integer", "real"} if formal == "count_rate" else {"integer"})
                or binding.attributes & {"target", "pointer", "allocatable", "optional", "parameter", "volatile",
                                         "asynchronous", "value"}):
            raise CompilationError("SYSTEM_CLOCK requires independent original local INTEGER/REAL scalar outputs")
        if any(existing is binding for existing in outputs.values()):
            raise CompilationError("SYSTEM_CLOCK outputs cannot alias")
        outputs[formal] = binding
    if not outputs:
        raise CompilationError("SYSTEM_CLOCK requires at least one original scalar output")
    graph, selected = analysis._selected_source(procedure, (node,))
    return {"version": 1, "procedure": procedure, "source_graph": graph.identity,
        "selected_node_ids": list(selected), "source_inputs": analysis.inputs.identity(),
        "intrinsic": "SYSTEM_CLOCK", "outputs": {formal: binding.public() for formal, binding in outputs.items()},
        "execution": "unchanged original intrinsic at its original reached point",
        "completion": "join preceding device work without publishing payloads before the host clock",
        "payload_access_authorized": False, "definitions_authorized": False, "placement_authorized": False}


@dataclass(frozen=True, slots=True)
class OriginalScalarClockProof:
    identity: str
    _analysis: object = field(repr=False, compare=False)
    _procedure: str = field(repr=False, compare=False)
    _node: object = field(repr=False, compare=False)
    _record: str = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)

    def validate(self, analysis, procedure, node):
        analysis = _origin(analysis)
        if (self._issuer is not _ISSUER or self._analysis is not analysis or self._procedure != procedure
                or self._node is not node or getattr(analysis, "_scalar_clock_proofs", {}).get(self.identity) is not self):
            raise CompilationError("SYSTEM_CLOCK requires its exact compiler-issued original proof")
        if _encoded(_derive(analysis, procedure, node)) != self._record:
            raise CompilationError("SYSTEM_CLOCK source or scalar associations changed")
        return self

    def public(self):
        return {"identity": self.identity, **json.loads(self._record)}


def original_scalar_clock(analysis, procedure, node):
    analysis = _origin(analysis)
    record = _encoded(_derive(analysis, procedure, node))
    registry = getattr(analysis, "_scalar_clock_proofs", None)
    if registry is None:
        registry = analysis._scalar_clock_proofs = {}
    for proof in registry.values():
        if proof._node is node:
            return proof.validate(analysis, procedure, node)
    proof = OriginalScalarClockProof(sha256(record.encode()).hexdigest(), analysis, procedure, node, record, _ISSUER)
    registry[proof.identity] = proof
    return proof.validate(analysis, procedure, node)
