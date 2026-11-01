"""Compose existing descriptor proofs over every retained original path edge.

Only the exact source-bound barred CALLs are excluded. Their original sticky
clearings remain in the inventory. No branch is selected speculatively, and an
unsupported additional edge rejects the entire composition. Native execution,
payload reads/writes and the mandatory publication frontier remain independent.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from hashlib import sha256

from compiler.frontend.allocation_epoch_eligibility import OriginalAllocationGate
from compiler.frontend.descriptor_windows import prove_descriptor_window
from compiler.frontend.source_effects import _children, _kind
from compiler.frontend.supported_paths import OriginalSupportedPathSelection
from compiler.ir import CompilationError

_ISSUER = object()


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _attachments(proof):
    return tuple((item.name, getattr(proof, item.name)) for item in fields(proof))


def _attached(proof, attachments):
    return all(getattr(proof, name) is original for name, original in attachments)


def _derive(analysis, path, gate, roots, module_roots):
    if (type(path) is not OriginalSupportedPathSelection or type(gate) is not OriginalAllocationGate
            or path._analysis is not analysis or gate._analysis is not analysis or path._routine is not gate._routine):
        raise CompilationError("supported path storage requires its exact source path and sticky gate")
    path_attachments, gate_attachments = _attachments(path), _attachments(gate)
    path.validate()
    gate.validate()
    descriptor_attachments = []

    def retain(proof):
        # Preserve the authentic issuer-return state before any later issuer
        # or validation callback can change an already consumed token.
        descriptor_attachments.append((proof, _attachments(proof)))
        return proof
    routine = path._routine
    graph, selected = analysis._selected_source(routine.qualified, path._originals)
    barred, original_nodes = {}, {}
    pending, covered, leaves = list(reversed(selected)), set(), []
    while pending:
        identity = pending.pop()
        if identity in covered:
            raise CompilationError("supported path storage requires distinct original graph occurrences")
        covered.add(identity)
        node = graph.nodes[identity]
        originals = graph.source_nodes(identity)
        for original in originals:
            original_nodes[id(original)] = identity
        if node.kind == "sequence":
            pending.extend(reversed(node.children))
        elif node.kind == "branch":
            pending.extend(reversed(tuple(child for condition, body in node.alternatives
                                          for child in (condition, body) if child is not None)))
        elif node.kind in {"call", "operation", "boundary"}:
            # A source-graph boundary is not a storage mutation verdict. The
            # ordinary descriptor issuer below must independently inspect the
            # entire exact node (including clocks and intrinsic output). Its
            # lifetime evidence grants no native execution/completion authority.
            leaves.append(identity)
        else:
            raise CompilationError("supported path storage needs separate control/completion authority: " + node.kind)
    for original in gate._barriers:
        identity = original_nodes.get(id(original))
        actual = graph.source_nodes(identity) if identity is not None else ()
        if identity is None or graph.nodes[identity].kind != "call" or len(actual) != 1 or actual[0] is not original:
            raise CompilationError("supported path storage lost an exact original barred CALL")
        barred[identity] = original
    components = []
    for identity in leaves:
        if identity not in barred:
            # This is the unchanged checked descriptor issuer. No parser
            # payload, public JSON, callee spelling or flag replaces it.
            proof = retain(prove_descriptor_window(analysis, routine.qualified, (identity,), roots, module_roots=module_roots))
            components.append((identity, proof))
    if not components:
        raise CompilationError("supported path storage has no independently retained original edge")
    # A private source gate cannot veto independently generated owners that
    # share module/argument allocations. Every excluded CALL therefore needs
    # the unchanged descriptor/no-escape issuer for each allocation outside
    # the gate's private frame. Unknown excluded effects may concern the
    # private gated resources, but cannot silently expose a shared allocation.
    private = {binding.root for binding in gate._resources}
    exposed_roots = tuple(root for root in roots if root not in private)
    exposed_modules = tuple(root for root in module_roots if root not in private)
    exposures = tuple((identity, retain(prove_descriptor_window(analysis, routine.qualified, (identity,),
        exposed_roots, module_roots=exposed_modules))) for identity in barred
        if exposed_roots or exposed_modules)
    exposure_map = dict(exposures)
    exposure_resources = sorted((*exposed_roots, *exposed_modules))
    metadata = []
    for original, clock, _private_descriptor in gate._prearm:
        _original_graph, identities = analysis._selected_source(routine.qualified, (original,))
        if _original_graph is not graph or len(identities) != 1:
            raise CompilationError("supported path pre-arm metadata lost its exact original graph node")
        identity, = identities
        descriptor = retain(prove_descriptor_window(analysis, routine.qualified, (identity,), roots, module_roots=module_roots))
        metadata.append((identity, original, clock, descriptor))
    # The sticky flag protects future invocations only if the same private
    # allocations cannot escape or change through an unselected prefix/tail.
    # The once-only fresh-allocation guard has its separate exact gate proof;
    # every other original top-level node uses the ordinary lifetime issuer
    # for the complete capture inventory, including shared module inputs.
    significant = tuple(node for node in _children(routine.execution) if _kind(node) != "Comment")
    first_guard = significant[0]
    outside = []
    for original in significant:
        if original is first_guard or any(original is node for node in path._originals):
            continue
        _original_graph, identities = analysis._selected_source(routine.qualified, (original,))
        if _original_graph is not graph or len(identities) != 1:
            raise CompilationError("supported path lost complete original outside-frame source membership")
        identity, = identities
        outside.append((identity, retain(prove_descriptor_window(analysis, routine.qualified, (identity,),
            roots, module_roots=module_roots))))
    closure = {}
    for _identity, proof in components:
        for name, source_graph in proof.public()["source_call_closure"].items():
            if name in closure and closure[name] != source_graph:
                raise CompilationError("supported path storage has conflicting original child closure identities")
            closure[name] = source_graph
    record = {"version": 1, "kind": "original_supported_path_descriptor_composition", "procedure": routine.qualified,
              "source_inputs": analysis.inputs.identity(), "source_graph": graph.identity,
              "selected_node_ids": list(selected), "path_identity": path.identity, "gate_identity": gate.identity,
              "roots": list(roots), "module_roots": list(module_roots),
              "components": [{"node_id": node, "descriptor_identity": proof.identity} for node, proof in components],
              "outside_interval_components": [{"node_id": node, "descriptor_identity": proof.identity}
                                              for node, proof in outside],
              "barred_calls": [{"node_id": node, "original_source_sha256": sha256(str(original).encode()).hexdigest(),
                                "exposure_resources": exposure_resources,
                                "exposure_descriptor_identity": exposure_map[node].identity if node in exposure_map else None}
                               for node, original in barred.items()],
              "source_call_closure": closure, "complete_retained_graph": True,
              "excluded_edge_policy": "publish and close before unchanged CALL once; source gate vetoes all later activations",
              "additional_unproved_edge": "reject entire supported path",
              "entry_definition_authority": False, "allocation_presence_authority": False,
              "payload_access_authorized": False, "execution_authorized": False,
              "alias_authority": False, "cross_frontier_retention_authorized": False}
    if metadata:
        from compiler.frontend.supported_paths import _stamp
        record.update(version=2, gate_metadata_edges=[{
            "node_id": identity, "source_span": _stamp(original, ())["original_span"],
            "original_source_sha256": sha256(str(original).encode()).hexdigest(),
            "clock_identity": clock.identity, "descriptor_identity": descriptor.identity}
            for identity, original, clock, descriptor in metadata])
    encoded = _encode(record)
    path.validate()
    gate.validate()
    for node, proof in components:
        proof.validate(analysis, routine.qualified, (node,))
    for node, proof in exposures:
        proof.validate(analysis, routine.qualified, (node,))
    for node, proof in outside:
        proof.validate(analysis, routine.qualified, (node,))
    for identity, original, clock, descriptor in metadata:
        call = original.items[1] if _kind(original) == "If_Stmt" else original
        clock.validate(analysis, routine.qualified, call)
        descriptor.validate(analysis, routine.qualified, (identity,))
    analysis.verify_publication()
    if (not _attached(path, path_attachments) or not _attached(gate, gate_attachments)
            or getattr(analysis, "_supported_path_selections", {}).get(path.identity) is not path
            or getattr(analysis, "_allocation_gates", {}).get(gate.identity) is not gate
            or sha256(path._record.encode()).hexdigest() != path.identity
            or sha256(gate._record.encode()).hexdigest() != gate.identity
            or any(not _attached(proof, attached)
                   or getattr(analysis, "_descriptor_window_proofs", {}).get(proof.identity) is not proof
                   for proof, attached in descriptor_attachments)):
        raise CompilationError("supported path storage independent issuer changed at final publication")
    return encoded, graph, tuple(components), tuple(barred), exposures, tuple(outside), tuple(metadata)


@dataclass(frozen=True, slots=True)
class SupportedPathDescriptorComposition:
    identity: str
    _analysis: object = field(repr=False, compare=False)
    _path: object = field(repr=False, compare=False)
    _gate: object = field(repr=False, compare=False)
    _roots: tuple = field(repr=False, compare=False)
    _module_roots: tuple = field(repr=False, compare=False)
    _graph: object = field(repr=False, compare=False)
    _components: tuple = field(repr=False, compare=False)
    _barred: tuple = field(repr=False, compare=False)
    _exposures: tuple = field(repr=False, compare=False)
    _outside: tuple = field(repr=False, compare=False)
    _metadata: tuple = field(repr=False, compare=False)
    _record: str = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)

    def validate(self):
        attachments = _attachments(self)
        analysis = self._analysis
        if (self._issuer is not _ISSUER or type(self) is not SupportedPathDescriptorComposition
                or type(self._record) is not str
                or sha256(self._record.encode()).hexdigest() != self.identity
                or getattr(analysis, "_supported_path_storage", {}).get(self.identity) is not self):
            raise CompilationError("supported path storage requires its exact registered composition")
        record, graph, components, barred, exposures, outside, metadata = _derive(analysis, self._path, self._gate, self._roots, self._module_roots)
        if (record != self._record or graph is not self._graph or barred != self._barred
                or len(components) != len(self._components)
                or any(a != b or p is not q for (a, p), (b, q) in zip(components, self._components, strict=True))
                or len(exposures) != len(self._exposures)
                or any(a != b or p is not q for (a, p), (b, q) in zip(exposures, self._exposures, strict=True))
                or len(outside) != len(self._outside)
                or any(a != b or p is not q for (a, p), (b, q) in zip(outside, self._outside, strict=True))
                or len(metadata) != len(self._metadata)
                or any(a != b or x is not y or c is not d or p is not q
                       for (a, x, c, p), (b, y, d, q) in zip(metadata, self._metadata, strict=True))
                or getattr(analysis, "_supported_path_storage", {}).get(self.identity) is not self
                or not _attached(self, attachments)
                or sha256(self._record.encode()).hexdigest() != self.identity):
            raise CompilationError("supported path storage source, retained edge or descriptor authority changed")
        return self

    def public(self):
        return {"identity": self.identity, **json.loads(self._record)}

    def resource_state(self, resource):
        self.validate()
        if resource not in (*self._roots, *self._module_roots):
            raise CompilationError("supported path storage resource exceeds its independent descriptor inventory")
        states = [proof.resource_state(resource) for _node, proof in self._components]
        if any(state != states[0] for state in states):
            # No guessed intersection or whole payload fact is manufactured.
            raise CompilationError("supported path storage component declaration states disagree")
        return states[0]

    def descriptor_for(self, selection, *, analysis=None):
        """Return the ordinary issuer token only for a wholly retained query."""
        self.validate()
        analysis = self._analysis if analysis is None else analysis
        original = self._analysis
        if (analysis.inputs is not original.inputs or analysis.routines.keys() != original.routines.keys()
                or any(analysis.routines[name] is not routine for name, routine in original.routines.items())
                or analysis.modules.keys() != original.modules.keys()
                or any(analysis.modules[name] is not scope for name, scope in original.modules.items())):
            raise CompilationError("supported path descriptor consumer changed its exact original source objects")
        graph, identities = self._analysis._selected_source(self._path._routine.qualified, selection)
        covered, pending = set(), list(identities)
        while pending:
            identity = pending.pop()
            if identity in covered:
                continue
            covered.add(identity)
            node = graph.nodes[identity]
            pending.extend(node.children)
            pending.extend(child for condition, body in node.alternatives for child in (condition, body) if child is not None)
        retained = {node for node, _proof in self._components}
        leaves = {node for node in covered if graph.nodes[node].kind in {"call", "operation", "boundary"}}
        if graph is not self._graph or not leaves or not leaves <= retained:
            raise CompilationError("supported path storage query includes an excluded or foreign original edge")
        return prove_descriptor_window(analysis, self._path._routine.qualified, identities,
                                       self._roots, module_roots=self._module_roots)


def prove_supported_path_storage(analysis, path, gate, roots, *, module_roots=()):
    if (type(path) is not OriginalSupportedPathSelection or type(gate) is not OriginalAllocationGate
            or path._analysis is not analysis or gate._analysis is not analysis or path._routine is not gate._routine):
        raise CompilationError("supported path storage requires its exact original source analysis and tokens")
    roots, module_roots = tuple(roots), tuple(module_roots)
    if any(not isinstance(root, str) for root in (*roots, *module_roots)) or len(set((*roots, *module_roots))) != len(roots)+len(module_roots):
        raise CompilationError("supported path storage requires exact distinct original resource roots")
    registry = getattr(analysis, "_supported_path_storage", None)
    if registry is None:
        registry = analysis._supported_path_storage = {}
    if type(registry) is not dict:
        raise CompilationError("supported path storage requires its private issuer registry")
    for proof in tuple(registry.values()):
        if type(proof) is not SupportedPathDescriptorComposition:
            raise CompilationError("supported path storage cannot consume public metadata as authority")
        if proof._path is path and proof._gate is gate and proof._roots == roots and proof._module_roots == module_roots:
            return proof.validate()
    record, graph, components, barred, exposures, outside, metadata = _derive(analysis, path, gate, roots, module_roots)
    if getattr(analysis, "_supported_path_storage", None) is not registry:
        raise CompilationError("supported path storage registry changed before publication")
    proof = SupportedPathDescriptorComposition(sha256(record.encode()).hexdigest(), analysis, path, gate, roots,
        module_roots, graph, components, barred, exposures, outside, metadata, record, _ISSUER)
    registry[proof.identity] = proof
    return proof.validate()
