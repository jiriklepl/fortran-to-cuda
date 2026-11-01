"""An exact private receiver may retain qualified opaque library scratch.

This proof concerns original host object identity and observers. It does not
authorize a device capture, read the object, infer definitions, or establish
library/numerical equivalence. The explicit supplied library qualification
must independently establish private scratch ownership/reinitialization.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256

from fparser.two.utils import walk

from compiler.frontend.source_effects import _kind, _part
from compiler.ir import CompilationError

_ISSUER = object()


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _span(node):
    item = getattr(node, "item", None)
    return tuple(getattr(item, "fort_original_span", item.span)) if item is not None else None


def _derive(analysis, routine, node, resolved, formal, contract):
    from compiler.frontend.device_tasks import _keys, _reference

    _keys(contract, ("formal", "constructor", "qualification"))
    if contract["formal"] != formal:
        raise CompilationError("opaque receiver contract differs from the original formal")
    resolved.validate(analysis, analysis.source_scope_for(node, routine.scope), node)
    association = next((item for item in resolved.associations if item.formal == formal), None)
    binding = association.binding if association is not None else None
    if (association is None or binding is None or _kind(association.actual) != "Name"
            or binding.rank or binding.declaring_scope is not routine.scope or "save" not in binding.attributes
            or binding.dtype in {"integer", "real", "logical", "complex", "character"}
            or binding.attributes & {"pointer", "target", "allocatable", "optional", "volatile", "asynchronous", "value"}
            or association.actual_type is None or association.actual_type.schema is None
            or association.formal_binding.rank or association.formal_binding.intent != "inout"):
        raise CompilationError("opaque library receiver requires its exact local SAVE derived-type association")
    if _part(routine.scope.node, "Internal_Subprogram_Part") is not None:
        raise CompilationError("opaque library receiver cannot escape through contained host association")
    if any(_kind(item) == "Namelist_Stmt" for item in walk(_part(routine.scope.node, "Specification_Part"))):
        # READ/WRITE NML names the group in executable source, not its objects.
        # An execution-Name observer check alone cannot establish privacy.
        raise CompilationError("opaque library receiver requires independent storage without NAMELIST exposure")
    if any(_kind(item) in {"Common_Stmt", "Equivalence_Stmt"}
           for item in walk(_part(routine.scope.node, "Specification_Part"))):
        raise CompilationError("opaque library receiver requires independent local storage without association")
    if any(_kind(item) in {"Final_Binding", "Type_Bound_Procedure_Part"}
           for item in walk(association.actual_type.schema.node)):
        raise CompilationError("opaque library receiver finalization or bound methods require separate observability proof")
    constructor = contract["constructor"]
    _keys(constructor, ("span", "statement_sha256"))
    if (not isinstance(constructor["span"], list) or len(constructor["span"]) != 2
            or any(type(line) is not int for line in constructor["span"])):
        raise CompilationError("opaque library receiver requires an exact original constructor span")
    candidates = [item for item in walk(routine.execution) if _kind(item) == "Assignment_Stmt"
                  and _span(item) == tuple(constructor["span"])]
    if len(candidates) != 1:
        raise CompilationError("opaque library receiver constructor is unavailable or ambiguous")
    assignment, = candidates
    lhs, _, rhs = assignment.items
    if (_kind(lhs) != "Name" or analysis._binding(analysis.source_scope_for(lhs, routine.scope), lhs) is not binding
            or _kind(rhs) not in {"Function_Reference", "Part_Ref", "Structure_Constructor"}
            or sha256(str(assignment).encode()).hexdigest() != constructor["statement_sha256"]):
        raise CompilationError("opaque library receiver constructor changed or copies another object")
    permitted = {id(lhs), id(association.actual)}
    for name in walk(routine.execution):
        if (_kind(name) == "Name" and analysis._binding(analysis.source_scope_for(name, routine.scope), name) is binding
                and id(name) not in permitted):
            raise CompilationError("opaque library receiver has an unqualified observer, mutation or escape")
    _reference(contract["qualification"])
    graph, selected = analysis._selected_source(routine.qualified, (node,))
    constructor_graph, constructor_nodes = analysis._selected_source(routine.qualified, (assignment,))
    if graph is not constructor_graph:
        raise CompilationError("opaque library receiver lost its original source graph")
    return {"version": 1, "procedure": routine.qualified, "source_graph": graph.identity,
            "source_inputs": analysis.inputs.identity(), "selected_node_ids": list(selected),
            "formal": formal, "resource": binding.root, "descriptor": binding.public(),
            "actual_type": association.actual_type.public(),
            "constructor": {**constructor, "selected_node_ids": list(constructor_nodes),
                            "execution": "unchanged original source and guards"},
            "qualification": contract["qualification"],
            "observers": "only original constructor LHS and exact contracted receiver argument",
            "workspace_effects": "explicit supplied private-scratch qualification; never a readonly object claim",
            "capture_authorized": False, "definitions_authorized": False, "gpu_access_authorized": False,
            "metadata_reads_authorized": False, "storage": "original SAVE declaration; no copy or C address export"}


@dataclass(frozen=True, slots=True)
class OriginalOpaqueStateProof:
    identity: str
    binding: object
    _analysis: object = field(repr=False, compare=False)
    _routine: object = field(repr=False, compare=False)
    _node: object = field(repr=False, compare=False)
    _resolved: object = field(repr=False, compare=False)
    _formal: str = field(repr=False, compare=False)
    _contract: str = field(repr=False, compare=False)
    _record: str = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)

    def validate(self, analysis, routine, node, resolved=None):
        if (self._issuer is not _ISSUER or analysis is not self._analysis or routine is not self._routine
                or node is not self._node or resolved is not None and resolved is not self._resolved
                or getattr(analysis, "_opaque_library_state_proofs", {}).get(self.identity) is not self):
            raise CompilationError("opaque receiver requires its exact compiler-issued original proof")
        if _encoded(_derive(analysis, routine, node, self._resolved, self._formal,
                            json.loads(self._contract))) != self._record:
            raise CompilationError("opaque receiver source, constructor, observers or qualification changed")
        return self

    def public(self):
        return {"identity": self.identity, **json.loads(self._record)}


def original_opaque_state(analysis, routine, node, resolved, formal, contract):
    encoded = _encoded(_derive(analysis, routine, node, resolved, formal, contract))
    for proof in getattr(analysis, "_opaque_library_state_proofs", {}).values():
        if proof._node is node and proof._formal == formal and proof._contract == _encoded(contract):
            return proof.validate(analysis, routine, node, resolved)
    binding = next(item.binding for item in resolved.associations if item.formal == formal)
    proof = OriginalOpaqueStateProof(sha256(encoded.encode()).hexdigest(), binding, analysis, routine, node, resolved,
                                    formal, _encoded(contract), encoded, _ISSUER)
    registry = getattr(analysis, "_opaque_library_state_proofs", None)
    if registry is None:
        registry = analysis._opaque_library_state_proofs = {}
    registry[proof.identity] = proof
    return proof.validate(analysis, routine, node, resolved)
