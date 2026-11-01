"""Project an original fresh-allocation prefix for definition preservation.

Successful private allocations create storage disjoint from live incoming
arrays. This is useful only inside the exact first-allocation guard; it never
exempts an arbitrary write to a SAVE allocation. All remaining statements need
the ordinary native preservation proof. Original execution is unchanged.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from hashlib import sha256

from compiler.frontend.allocation_epoch_eligibility import OriginalAllocationGate, _prearm_clock
from compiler.frontend.local_allocatable_storage import original_local_allocation
from compiler.frontend.optional_scalar_control import (
    _same_scope_state,
    _scope_state,
    is_optional_scalar_control,
    original_optional_scalar_control,
)
from compiler.frontend.scalar_clock import is_system_clock
from compiler.frontend.source_effects import _children, _kind
from compiler.frontend.source_functions import _original, _validate_scope
from compiler.ir import CompilationError

_ISSUER = object()


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _attachments(value):
    return tuple((item.name, getattr(value, item.name)) for item in fields(value))


def _source_scope_states(projection):
    scopes = {id(projection._gate._routine.scope): projection._gate._routine.scope}
    for proof, _registry in projection._retained:
        for scope in getattr(proof, "_scopes", ()):
            scopes[id(scope)] = scope
    return tuple(_scope_state(scope) for scope in scopes.values())


def _same_source_scope_states(projection, states):
    current = _source_scope_states(projection)
    return (len(current) == len(states) and all(a[0] is b[0] for a, b in zip(current, states, strict=True))
            and all(_same_scope_state(state) for state in states))


def _derive(analysis, procedure, segments, gate):
    if type(gate) is not OriginalAllocationGate or gate._analysis is not analysis:
        raise CompilationError("private entry prefix requires its exact original allocation gate")
    routine = _original(analysis, procedure)
    if gate._routine is not routine:
        raise CompilationError("private entry prefix cannot borrow another original routine's gate")
    gate.validate()
    guard = next((node for node in _children(routine.execution) if _kind(node) != "Comment"), None)
    if (not isinstance(segments, tuple) or len(segments) != 1 or any(not isinstance(segment, tuple) or not segment
            for segment in segments) or sum(node is guard for segment in segments for node in segment) != 1):
        raise CompilationError("private entry prefix requires the complete exact first-allocation guard")
    resources = gate._resources
    locals_ = tuple(original_local_allocation(analysis, routine, binding) for binding in resources)
    steps, native, retained = [], [], [(gate, "_allocation_gates")]
    retained.extend((proof, "_local_allocation_proofs") for proof in locals_)
    graph = analysis.structure(procedure)
    for segment in segments:
        current, identities = analysis._selected_source(procedure, segment)
        if current is not graph or tuple(graph.nodes[graph.root_id].children[:len(identities)]) != identities:
            raise CompilationError("private entry prefix requires a complete original root prefix")

    def step(kind, node, proofs=()):
        steps.append((kind, node, proofs))

    def clock(node):
        call = node.items[1] if _kind(node) == "If_Stmt" else node
        if _kind(call) != "Call_Stmt" or not is_system_clock(analysis, routine.scope, call):
            return False
        _node, clock_proof, descriptor = _prearm_clock(analysis, routine, node, resources)
        retained.extend(((clock_proof, "_scalar_clock_proofs"), (descriptor, "_descriptor_window_proofs")))
        step("original_scalar_clock", node, (clock_proof, descriptor))
        return True

    for segment in segments:
        for node in segment:
            if _kind(node) == "Comment" and not str(node).lstrip().startswith("!$"):
                step("original_comment", node)
                continue
            if node is not guard:
                if is_optional_scalar_control(node):
                    proof = original_optional_scalar_control(analysis, procedure, node)
                    retained.append((proof, "_optional_scalar_controls"))
                    step("original_optional_scalar_control", node, (proof,))
                elif not clock(node):
                    native.append((node,))
                    step("ordinary_native", node)
                continue
            header, *body, end = guard.content
            # The independent gate authenticates all predicates, allocations,
            # bounds, private literal writes, clocks, and the final arm. It
            # rejects ELSE arms and any other work in this first guard.
            native.append((header,))
            step("ordinary_native", header)
            significant = tuple(item for item in body if _kind(item) != "Comment")
            allocations = significant[:len(resources)]
            for allocation, local in zip(allocations, locals_, strict=True):
                step("fresh_private_allocation", allocation, (local,))
            for item in body:
                if any(item is allocation for allocation in allocations):
                    continue
                if _kind(item) == "Comment":
                    step("original_comment", item)
                elif (_kind(item) == "Assignment_Stmt" and _kind(item.items[0]) == "Name"
                      and any(analysis._binding(routine.scope, item.items[0]) is binding for binding in resources)):
                    # Only a literal overwrite after all fresh allocations is
                    # permitted by the gate. It defines no incoming resource.
                    step("fresh_private_literal_overwrite", item)
                elif not clock(item):
                    native.append((item,))
                    step("ordinary_native", item)
            step("original_end_if", end)
    public_steps = []
    for kind, node, proofs in steps:
        identity = None
        if kind not in {"original_comment", "original_end_if"}:
            current, identities = analysis._selected_source(procedure, (node,))
            if current is not graph or len(identities) != 1:
                raise CompilationError("private entry prefix step requires one exact original graph node")
            identity = identities[0]
        public_steps.append({"kind": kind, "source_node_id": identity,
                             "original_source_sha256": sha256(str(node).encode()).hexdigest(),
                             "proof_identities": [proof.identity for proof in proofs]})
    record = {"version": 3,
              "kind": "original_private_allocation_prefix_projection", "procedure": procedure,
              "source_inputs": analysis.inputs.identity(), "source_graph": graph.identity,
              "allocation_gate_identity": gate.identity,
              "private_resources": [binding.root for binding in resources],
              "local_declaration_identities": [proof.identity for proof in locals_],
              "original_segments": [list(analysis._selected_source(procedure, segment)[1]) for segment in segments],
              "steps": public_steps,
              "semantics": "filter existing incoming definitions across unchanged source; successful fresh private storage only",
              "ordinary_native_effects_require_independent_preservation": True,
              "private_resources_cannot_receive_preserved_entry_facts": True,
              "new_definition_authority": False, "execution_authorized": False,
              "payload_access_authorized": False, "descriptor_lifetime_authority": False,
              "alias_authority": False, "cross_exit_retention_authorized": False}
    return _encoded(record), tuple(native), tuple(steps), tuple(retained)


@dataclass(frozen=True, slots=True)
class PrivateEntryPrefixProjection:
    identity: str
    native_segments: tuple = field(repr=False, compare=False)
    private_bindings: tuple = field(repr=False, compare=False)
    _analysis: object = field(repr=False, compare=False)
    _procedure: str
    _segments: tuple = field(repr=False, compare=False)
    _gate: object = field(repr=False, compare=False)
    _steps: tuple = field(repr=False, compare=False)
    _retained: tuple = field(repr=False, compare=False)
    _record: str = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)

    def public(self):
        return {"identity": self.identity, **json.loads(self._record)}

    def validate(self, analysis, procedure, segments):
        attached = _attachments(self)
        scope_states = _source_scope_states(self)
        dependencies = tuple((proof, _attachments(proof), name, getattr(analysis, name, None), proof.identity)
                             for proof, name in self._retained)
        if (type(self) is not PrivateEntryPrefixProjection or self._issuer is not _ISSUER
                or analysis is not self._analysis or procedure != self._procedure
                or getattr(analysis, "_private_entry_prefixes_owner", None) is not analysis
                or getattr(analysis, "_private_entry_prefixes", {}).get(self.identity) is not self
                or not isinstance(segments, tuple) or len(segments) != len(self._segments)
                or any(not isinstance(a, tuple) or len(a) != len(b)
                       or any(x is not y for x, y in zip(a, b, strict=True))
                       for a, b in zip(segments, self._segments, strict=True))):
            raise CompilationError("private entry prefix requires its exact registered original source")
        record, native, steps, retained = _derive(analysis, procedure, segments, self._gate)
        if (record != self._record or sha256(record.encode()).hexdigest() != self.identity
                or self.private_bindings is not self._gate._resources
                or len(native) != len(self.native_segments)
                or any(len(a) != len(b) or any(x is not y for x, y in zip(a, b, strict=True))
                       for a, b in zip(native, self.native_segments, strict=True))
                or len(steps) != len(self._steps)
                or any(a != b or x is not y or len(p) != len(q)
                       or any(u is not v for u, v in zip(p, q, strict=True))
                       for (a, x, p), (b, y, q) in zip(steps, self._steps, strict=True))
                or len(retained) != len(self._retained)
                or any(a is not b or x != y for (a, x), (b, y) in zip(retained, self._retained, strict=True))):
            raise CompilationError("private entry prefix changed its original projection")
        analysis.verify_publication()
        analysis.verify_source()
        analysis.inputs.verify_live()
        _validate_scope(analysis, _original(analysis, procedure).scope)
        if (any(getattr(self, name) is not value for name, value in attached)
                or not _same_source_scope_states(self, scope_states)
                or getattr(analysis, "_private_entry_prefixes_owner", None) is not analysis
                or getattr(analysis, "_private_entry_prefixes", {}).get(self.identity) is not self
                or any(any(getattr(proof, name) is not value for name, value in saved)
                       or getattr(analysis, registry_name, None) is not registry or registry.get(identity) is not proof
                       for proof, saved, registry_name, registry, identity in dependencies)):
            raise CompilationError("private entry prefix dependency changed at final publication")
        return self


def prove_private_entry_prefix(analysis, procedure, segments, gate):
    """Issue source projection only; native effects and entry facts stay separate."""
    record, native, steps, retained = _derive(analysis, procedure, segments, gate)
    identity = sha256(record.encode()).hexdigest()
    if getattr(analysis, "_private_entry_prefixes_owner", None) is not analysis:
        analysis._private_entry_prefixes_owner = analysis
        analysis._private_entry_prefixes = {}
    registry = analysis._private_entry_prefixes
    if identity not in registry:
        registry[identity] = PrivateEntryPrefixProjection(identity, native, gate._resources, analysis, procedure,
            segments, gate, steps, retained, record, _ISSUER)
    return registry[identity].validate(analysis, procedure, segments)
