"""Preserve independent invocation-entry definitions at a reached local group.

This is neither an INTENT-based definition nor an upload-envelope admission.
The existing native preservation engine filters already supplied WHOLE entry
facts across the exact original prefix and reached branch predicates. Fresh
registration and every child entry kill remain independent.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from hashlib import sha256

from compiler.frontend.native_preservation import prove_native_preservation
from compiler.frontend.resource_reachability import _authenticate_binding
from compiler.frontend.source_effects import _kind
from compiler.frontend.source_functions import _hash, _original
from compiler.frontend.source_span_exit import prove_original_sequence_exit
from compiler.ir import CompilationError

_ISSUER = object()


class EntryPreservationUnsupported(CompilationError):
    """Existing entry definitions cannot be carried to this source position."""


def _prefix_segments(analysis, procedure, originals, *, position=None):
    """Retain ordered exact guarded segments on the reached original IF path.

    Earlier IF bodies are included in full. Bodies of untaken sibling arms of
    an enclosing IF are excluded, while every preceding predicate remains.
    Loops and associations need separate backedge/association authority.
    """
    if position is None:
        exit_proof = prove_original_sequence_exit(analysis, procedure, originals)
    else:
        from compiler.frontend.source_position import OriginalReachedPositionProof

        if type(position) is not OriginalReachedPositionProof:
            raise CompilationError("entry preservation requires its typed original reached position")
        exit_proof = position.validate(analysis, procedure, originals)
    graph = analysis.structure(procedure)
    cursor, result, visited = exit_proof.public()["selected_node_ids"][0], (), set()
    while True:
        if cursor in visited or len(visited) > graph.node_limit:
            raise CompilationError("entry preservation original ancestry is cyclic or exhausted")
        visited.add(cursor)
        parents = [node for node in graph.nodes.values() if node.kind == "sequence" and cursor in node.children]
        if len(parents) != 1:
            raise EntryPreservationUnsupported("entry preservation requires an exact original sequence ancestry")
        parent = parents[0]
        before = tuple(original for identity in parent.children[:parent.children.index(cursor)]
                       for original in graph.source_nodes(identity))
        if before:
            result = (before, *result)
        if parent.id == graph.root_id:
            return exit_proof, result
        branches = [(node, ordinal) for node in graph.nodes.values() if node.kind == "branch"
                    for ordinal, (_condition, body) in enumerate(node.alternatives) if body == parent.id]
        if len(branches) != 1:
            raise EntryPreservationUnsupported("entry preservation cannot cross a loop backedge or association")
        branch, ordinal = branches[0]
        source, = graph.source_nodes(branch.id)
        if _kind(source) != "If_Construct":
            raise EntryPreservationUnsupported("entry preservation needs a complete original IF path")
        headers = [node for node in source.content
                   if _kind(node) in {"If_Then_Stmt", "Else_If_Stmt", "Else_Stmt", "End_If_Stmt"}]
        if ordinal >= len(headers)-1:
            raise CompilationError("entry preservation lost its original branch ordinal")
        # Each predicate belongs to its original condition guard. Flattening
        # these with an outer prefix would forge one reached source segment.
        predicates = tuple((node,) for node in headers[:ordinal+1] if _kind(node) != "Else_Stmt")
        result = (*predicates, *result)
        cursor = branch.id


def _prefix(analysis, procedure, originals):
    """Diagnostic original nodes; authority remains in the exact segments."""
    exit_proof, segments = _prefix_segments(analysis, procedure, originals)
    return exit_proof, tuple(node for segment in segments for node in segment)


def _eligible(analysis, procedure, bindings, facts):
    routine = _original(analysis, procedure)
    if (not isinstance(bindings, tuple) or len({id(binding) for binding in bindings}) != len(bindings)
            or len({binding.root for binding in bindings}) != len(bindings)
            or not isinstance(facts, dict) or not isinstance(facts.get("captures"), dict)):
        raise CompilationError("entry preservation requires exact independent capture facts and bindings")
    eligible = []
    for binding in bindings:
        _authenticate_binding(analysis, binding)
        fact = facts["captures"].get(binding.root)
        if (not binding.rank or binding.dtype not in {"real", "integer", "logical"}
                or binding.name in routine.arguments and binding.intent == "out"
                or binding.attributes & {"pointer", "target", "optional", "volatile", "asynchronous", "value"}
                or not isinstance(fact, dict) or fact.get("initialized") != "whole"
                or fact.get("lifetime", "invocation") != "invocation"):
            continue
        if (fact.get("storage") != "stable" or fact.get("escapes") is not False
                or fact.get("allocation_changes") is not False
                or "source_sections" in fact or "requires_registered_handle" in fact):
            raise CompilationError("entry preservation lacks independently stable invocation storage")
        eligible.append(binding)
    return tuple(eligible)


def _native_attachments(analysis, preservations):
    """Fence the retained native receipts consumed by this definition filter."""
    result = []
    for preservation in preservations:
        for proof, registry_name in ((preservation, "_native_preservation_proofs"),
                                     (preservation.reachability, "_resource_reachability_proofs"),
                                     (preservation.reachability.native_graph, "_native_effect_graphs")):
            registry = getattr(analysis, registry_name, None)
            if not isinstance(registry, dict) or registry.get(proof.identity) is not proof:
                raise CompilationError("entry preservation lost its retained native source receipt")
            result.append((proof, tuple((item.name, getattr(proof, item.name)) for item in fields(proof)),
                           registry_name, registry, proof.identity))
        # This issuer always uses the original routine bindings directly.
        if preservation._context is not None:
            raise CompilationError("entry preservation cannot acquire a substituted interval context")
    return tuple(result)


def _projected_segments(analysis, procedure, segments, position, projection):
    if projection is None:
        return segments, ()
    from compiler.frontend.private_entry_prefix import PrivateEntryPrefixProjection

    if position is None or type(projection) is not PrivateEntryPrefixProjection:
        raise CompilationError("entry preservation requires its typed root-prefix projection")
    projection.validate(analysis, procedure, segments)
    return projection.native_segments, projection.private_bindings


def _projection_attachments(analysis, projection):
    if projection is None:
        return ()
    from compiler.frontend.private_entry_prefix import PrivateEntryPrefixProjection

    if type(projection) is not PrivateEntryPrefixProjection:
        raise CompilationError("entry preservation requires its typed root-prefix projection")
    result = []
    for proof, name in ((projection, "_private_entry_prefixes"), *projection._retained):
        registry = getattr(analysis, name, None)
        if not isinstance(registry, dict) or registry.get(proof.identity) is not proof:
            raise CompilationError("entry preservation lost its retained prefix projection")
        result.append((proof, tuple((part.name, getattr(proof, part.name)) for part in fields(proof)),
                       name, registry, proof.identity))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class PreservedEntryDefinitions:
    identity: str
    procedure: str
    preserved: tuple = field(repr=False, compare=False)
    _analysis: object = field(repr=False, compare=False)
    _originals: tuple = field(repr=False, compare=False)
    _bindings: tuple = field(repr=False, compare=False)
    _prefix: tuple = field(repr=False, compare=False)
    _exit: object = field(repr=False, compare=False)
    _segments: tuple = field(repr=False, compare=False)
    _preservations: tuple = field(repr=False, compare=False)
    _facts_identity: str
    _record: str = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)
    _position: object = field(default=None, repr=False, compare=False)
    _projection: object = field(default=None, repr=False, compare=False)

    def public(self):
        if type(self._record) is not str:
            raise CompilationError("entry preservation changed its encoded original record")
        try:
            return {"identity": self.identity, **json.loads(self._record)}
        except (TypeError, ValueError) as error:
            raise CompilationError("entry preservation changed its encoded original record") from error

    def validate(self, analysis, procedure, originals, bindings, facts):
        from compiler.frontend.source_position import OriginalReachedPositionProof
        from compiler.frontend.source_span_exit import OriginalSequenceExitProof

        # The anchor is a retained independent issuer, not just a pointer in
        # this receipt. Keep its exact attachments before any charged callback
        # and check them after this consumer's final publication callback.
        anchor = self._exit if self._position is None else self._position
        if type(anchor) is not (OriginalSequenceExitProof if self._position is None else OriginalReachedPositionProof):
            raise CompilationError("entry preservation requires its typed original position")
        anchor_fields = tuple((item.name, getattr(anchor, item.name)) for item in fields(anchor))
        anchor_registry_name = "_original_sequence_exits" if self._position is None else "_original_reached_positions"
        anchor_registry = getattr(analysis, anchor_registry_name, None)
        anchor_identity = anchor.identity
        position_owner = getattr(analysis, "_original_reached_positions_owner", None)
        native_attachments = _native_attachments(analysis, self._preservations)
        native_owner = getattr(analysis, "_native_preservation_registry_owner", None)
        projection_attachments = _projection_attachments(analysis, self._projection)
        projection_owner = getattr(analysis, "_private_entry_prefixes_owner", None)
        projection_scopes = ()
        if self._projection is not None:
            from compiler.frontend.private_entry_prefix import _same_source_scope_states, _source_scope_states
            projection_scopes = _source_scope_states(self._projection)
        attachment_names = ("identity", "procedure", "preserved", "_analysis", "_originals", "_bindings",
                            "_prefix", "_exit", "_segments", "_preservations", "_facts_identity", "_record", "_issuer",
                            "_position", "_projection")
        attachments = tuple(getattr(self, name) for name in attachment_names)
        attached_facts_identity = self._facts_identity
        if (self._issuer is not _ISSUER or analysis is not self._analysis or procedure != self.procedure
                or getattr(analysis, "_entry_preservations_owner", None) is not analysis
                or getattr(analysis, "_entry_preservations", {}).get(self.identity) is not self
                or _hash(facts) != self._facts_identity
                or not isinstance(bindings, tuple) or len(bindings) != len(self._bindings)
                or any(a is not b for a, b in zip(bindings, self._bindings, strict=True))
                or not isinstance(originals, tuple) or len(originals) != len(self._originals)
                or any(a is not b for a, b in zip(originals, self._originals, strict=True))):
            raise CompilationError("entry preservation requires its exact registered original facts and source")
        if type(self._record) is not str or sha256(self._record.encode()).hexdigest() != self.identity:
            raise CompilationError("entry preservation changed its encoded original record")
        record = self.public()
        exit_proof, segments = _prefix_segments(analysis, procedure, tuple(originals), position=self._position)
        prefix = tuple(node for segment in segments for node in segment)
        original_segments = segments
        segments, private = _projected_segments(analysis, procedure, segments, self._position, self._projection)
        eligible = _eligible(analysis, procedure, bindings, facts)
        eligible = tuple(binding for binding in eligible if not any(binding is item for item in private))
        anchor = self._exit if self._position is None else self._position
        if (exit_proof is not anchor or self._position is not None and self._exit is not None
                or len(prefix) != len(self._prefix)
                or any(a is not b for a, b in zip(prefix, self._prefix, strict=True))
                or len(segments) != len(self._segments) or len(segments) != len(self._preservations)
                or any(len(a) != len(b) or any(x is not y for x, y in zip(a, b, strict=True))
                       for a, b in zip(segments, self._segments, strict=True))):
            raise CompilationError("entry preservation original prefix or branch path changed")
        if len(record["segments"]) != len(segments):
            raise CompilationError("entry preservation changed its issued segment inventory")
        for segment, preservation, segment_record in zip(segments, self._preservations, record["segments"], strict=True):
            preservation.validate(analysis, procedure, segment, eligible)
            _graph, identities = analysis._selected_source(procedure, segment)
            if (list(identities) != list(segment_record["selected_node_ids"])
                    or preservation.identity != segment_record["preservation_identity"]):
                raise CompilationError("entry preservation changed its issued exact native segment proof")
        lost = {root for proof in self._preservations for root, _reasons in proof.invalidations}
        complete = all(proof.complete for proof in self._preservations)
        current = tuple(binding for binding in eligible if complete and binding.root not in lost)
        if (len(current) != len(self.preserved) or any(a is not b for a, b in zip(current, self.preserved, strict=True))
                or type(self._record) is not str or sha256(self._record.encode()).hexdigest() != self.identity
                or _entry_record(analysis, procedure, exit_proof, segments, self._preservations, current,
                                 self._facts_identity, position=self._position, projection=self._projection,
                                 original_segments=original_segments) != self._record):
            raise CompilationError("entry preservation retained definition state changed")
        analysis.verify_publication()
        if (_hash(facts) != attached_facts_identity or self._facts_identity != attached_facts_identity
                or any(getattr(self, name) is not value for name, value in zip(attachment_names, attachments, strict=True))
                or getattr(analysis, "_entry_preservations_owner", None) is not analysis
                or getattr(analysis, "_entry_preservations", {}).get(self.identity) is not self):
            raise CompilationError("entry preservation facts or native proof attachment changed at publication")
        analysis.verify_source()
        analysis.inputs.verify_live()
        if (any(getattr(anchor, name) is not value for name, value in anchor_fields)
                or getattr(analysis, anchor_registry_name, None) is not anchor_registry
                or anchor_registry.get(anchor_identity) is not anchor
                or self._position is not None
                and getattr(analysis, "_original_reached_positions_owner", None) is not position_owner):
            raise CompilationError("entry preservation retained original position changed at publication")
        if (getattr(analysis, "_native_preservation_registry_owner", None) is not native_owner
                or any(any(getattr(proof, name) is not value for name, value in attached)
                       or getattr(analysis, registry_name, None) is not registry
                       or registry.get(identity) is not proof
                       for proof, attached, registry_name, registry, identity in native_attachments)):
            raise CompilationError("entry preservation retained native source receipt changed at publication")
        if (self._projection is not None and getattr(analysis, "_private_entry_prefixes_owner", None) is not projection_owner
                or self._projection is not None and not _same_source_scope_states(self._projection, projection_scopes)
                or any(any(getattr(proof, name) is not value for name, value in attached)
                       or getattr(analysis, registry_name, None) is not registry
                       or registry.get(identity) is not proof
                       for proof, attached, registry_name, registry, identity in projection_attachments)):
            raise CompilationError("entry preservation retained prefix projection changed at publication")
        return self

    def permits(self, binding):
        return any(binding is original for original in self.preserved)


def _entry_record(analysis, procedure, anchor, segments, preservations, preserved, facts_identity, *, position=None,
                  projection=None, original_segments=()):
    """Encode authenticated state; this pure projection grants no new facts."""
    graph = analysis.structure(procedure)
    segment_records = []
    for segment, preservation in zip(segments, preservations, strict=True):
        original_graph, identities = analysis._selected_source(procedure, segment)
        if original_graph is not graph:
            raise CompilationError("entry preservation lost its exact original segment graph")
        segment_records.append({"selected_node_ids": list(identities), "preservation_identity": preservation.identity})
    record = {"version": 4 if projection is not None else 2 if position is None else 3,
        "kind": "original_invocation_entry_definition_preservation", "procedure": procedure,
        "source_inputs": analysis.inputs.identity(), "source_graph": graph.identity,
        "segments": segment_records, "entry_capture_facts_identity": facts_identity,
        "preserved_resources": [binding.root for binding in preserved],
        "definition_source": "independent original invocation-entry WHOLE facts after original entry undefinition",
        "prefix": "exact prior original statements and reached predicates; no enclosing loop or association",
        "semantics": "preserve existing definedness; no stale replica or unchanged-value assumption",
        "new_definition_authority": False, "execution_authorized": False, "payload_access_authorized": False,
        "descriptor_lifetime_authority": False, "alias_authority": False, "cross_exit_retention_authorized": False}
    record["sequence_exit_identity" if position is None else "reached_position_identity"] = anchor.identity
    if projection is not None:
        record["prefix_projection"] = projection.public()
        record["original_prefix_segments"] = [list(analysis._selected_source(procedure, segment)[1])
                                             for segment in original_segments]
    return json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)


def prove_preserved_entry_definitions(analysis, procedure, originals, bindings, facts, *, position=None,
                                      prefix_projection=None):
    """Filter independent entry facts; unsupported paths publish no new token."""
    facts_identity = _hash(facts)
    analysis.verify_source()
    originals, bindings = tuple(originals), tuple(bindings)
    eligible = _eligible(analysis, procedure, bindings, facts)
    if not eligible:
        return None
    exit_proof, segments = _prefix_segments(analysis, procedure, originals, position=position)
    original_segments = segments
    prefix = tuple(node for segment in segments for node in segment)
    segments, private = _projected_segments(analysis, procedure, segments, position, prefix_projection)
    eligible = tuple(binding for binding in eligible if not any(binding is item for item in private))
    if not eligible:
        return None
    if not segments and position is None:
        raise EntryPreservationUnsupported("entry preservation requires an authenticated original reached prefix")
    preservations = tuple(prove_native_preservation(analysis, procedure, segment, eligible)
                          for segment in segments)
    lost = {root for proof in preservations for root, _reasons in proof.invalidations}
    complete = all(proof.complete for proof in preservations)
    preserved = tuple(binding for binding in eligible if complete and binding.root not in lost)
    if not preserved:
        return None
    encoded = _entry_record(analysis, procedure, exit_proof, segments, preservations, preserved,
                            facts_identity, position=position, projection=prefix_projection,
                            original_segments=original_segments)
    identity = sha256(encoded.encode()).hexdigest()
    if getattr(analysis, "_entry_preservations_owner", None) is not analysis:
        analysis._entry_preservations_owner = analysis
        analysis._entry_preservations = {}
    registry = analysis._entry_preservations
    if identity not in registry:
        registry[identity] = PreservedEntryDefinitions(identity, procedure, preserved, analysis, originals, bindings,
            prefix, exit_proof if position is None else None, segments, preservations, facts_identity, encoded,
            _ISSUER, position, prefix_projection)
    return registry[identity].validate(analysis, procedure, originals, bindings, facts)
