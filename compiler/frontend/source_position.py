"""An authenticated reached root-sequence position, without exit authority.

This identifies original source preceding a selected interval. It does not
permit that interval to execute, keep an allocation alive, or preserve facts.
Those remain independent consumers of this exact source position.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from hashlib import sha256

from compiler.frontend.source_functions import _original
from compiler.frontend.source_span_exit import _span
from compiler.ir import CompilationError

_ISSUER = object()


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _unit_span(original):
    """Containers retain all original statement spans, including their end."""
    pending, spans, seen = [original], [], set()
    while pending:
        node = pending.pop()
        if id(node) in seen:
            raise CompilationError("reached position original unit contains repeated source objects")
        seen.add(id(node))
        if getattr(node, "item", None) is not None:
            span = _span(node)
            if (any(type(value) is not int for value in span) or span[0] <= 0 or span[1] < span[0]):
                raise CompilationError("reached position requires original physical source spans")
            spans.append(span)
        else:
            pending.extend(getattr(node, "content", ()))
    if not spans:
        raise CompilationError("reached position original unit has no physical source span")
    return min(span[0] for span in spans), max(span[1] for span in spans)


def _ordinary_comment(node):
    return type(node).__name__ == "Comment" and not str(node).lstrip().startswith("!$")


def _derive(analysis, procedure, originals):
    routine = _original(analysis, procedure)
    graph, selected = analysis._selected_source(procedure, originals)
    parent = graph.nodes[graph.root_id]
    units = tuple(graph.source_nodes(identity) for identity in selected)
    semantic = tuple(node for node in originals if not _ordinary_comment(node))
    expanded = tuple(node for unit in units for node in unit if not _ordinary_comment(node))
    if (not originals or not selected or parent.kind != "sequence" or any(not unit for unit in units)
            or len(set(selected)) != len(selected) or len(semantic) != len(expanded)
            or any(a is not b for a, b in zip(semantic, expanded, strict=True))):
        raise CompilationError("reached position requires complete exact original root units")
    try:
        start = parent.children.index(selected[0])
    except ValueError as error:
        raise CompilationError("reached position requires the original root sequence") from error
    if tuple(parent.children[start:start+len(selected)]) != selected:
        raise CompilationError("reached position requires one contiguous original root interval")
    spans = []
    for unit in units:
        physical = tuple(_unit_span(original) for original in unit if not _ordinary_comment(original))
        spans.append((min(span[0] for span in physical), max(span[1] for span in physical)))
    spans = tuple(spans)
    if any(a[1] >= b[0] for a, b in zip(spans[:-1], spans[1:], strict=True)):
        raise CompilationError("reached position lost its independent original source spans")
    record = {"version": 1, "kind": "original_root_sequence_position", "procedure": procedure,
        "source_inputs": analysis.inputs.identity(), "source_graph": graph.identity,
        "parent_sequence": parent.id, "selected_node_ids": list(selected),
        "activation_node_id": selected[0], "source_span": [spans[0][0], spans[-1][1]],
        "original_units": [{"node_id": identity, "source_span": list(span),
                            "original_source_sha256": sha256("\n".join(str(node) for node in unit).encode()).hexdigest()}
                           for identity, unit, span in zip(selected, units, spans, strict=True)],
        "activation": "before the first selected original root unit after its complete original prefix",
        "prefix_scope": "original root sequence only; no enclosing loop, association or branch",
        "execution_authorized": False, "entry_definition_authority": False,
        "descriptor_lifetime_authority": False, "payload_access_authorized": False,
        "exit_authority": False, "cross_exit_retention_authorized": False, "alias_authority": False}
    return _encode(record), graph, routine


@dataclass(frozen=True, slots=True)
class OriginalReachedPositionProof:
    identity: str
    _analysis: object = field(repr=False, compare=False)
    _procedure: str
    _originals: tuple = field(repr=False, compare=False)
    _graph: object = field(repr=False, compare=False)
    _routine: object = field(repr=False, compare=False)
    _record: str = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)

    def public(self):
        if type(self._record) is not str:
            raise CompilationError("reached position changed its encoded original record")
        try:
            return {"identity": self.identity, **json.loads(self._record)}
        except (TypeError, ValueError) as error:
            raise CompilationError("reached position changed its encoded original record") from error

    def validate(self, analysis, procedure, originals):
        attachments = tuple((item.name, getattr(self, item.name)) for item in fields(self))
        if (type(self) is not OriginalReachedPositionProof or self._issuer is not _ISSUER
                or analysis is not self._analysis or procedure != self._procedure
                or not isinstance(originals, tuple) or len(originals) != len(self._originals)
                or any(a is not b for a, b in zip(originals, self._originals, strict=True))
                or getattr(analysis, "_original_reached_positions_owner", None) is not analysis
                or getattr(analysis, "_original_reached_positions", {}).get(self.identity) is not self):
            raise CompilationError("reached position requires its exact registered original source")
        record, graph, routine = _derive(analysis, procedure, originals)
        if (record != self._record or sha256(record.encode()).hexdigest() != self.identity
                or graph is not self._graph or routine is not self._routine):
            raise CompilationError("reached position changed its original source or interval")
        analysis.verify_publication()
        current, current_graph, current_routine = _derive(analysis, procedure, originals)
        if (current != record or current_graph is not graph or current_routine is not routine
                or any(getattr(self, name) is not value for name, value in attachments)
                or getattr(analysis, "_original_reached_positions_owner", None) is not analysis
                or getattr(analysis, "_original_reached_positions", {}).get(self.identity) is not self):
            raise CompilationError("reached position source or authority changed at publication")
        analysis.verify_source()
        analysis.inputs.verify_live()
        return self


def prove_original_reached_position(analysis, procedure, originals):
    """Issue a position receipt, without demanding an IF suffix or task count."""
    analysis.verify_source()
    originals = tuple(originals)
    record, graph, routine = _derive(analysis, procedure, originals)
    identity = sha256(record.encode()).hexdigest()
    if getattr(analysis, "_original_reached_positions_owner", None) is not analysis:
        analysis._original_reached_positions_owner = analysis
        analysis._original_reached_positions = {}
    registry = analysis._original_reached_positions
    if identity in registry:
        return registry[identity].validate(analysis, procedure, originals)
    value = OriginalReachedPositionProof(identity, analysis, procedure, originals, graph, routine, record, _ISSUER)
    registry[identity] = value
    try:
        return registry[identity].validate(analysis, procedure, originals)
    except BaseException:
        if registry.get(identity) is value:
            del registry[identity]
        raise
