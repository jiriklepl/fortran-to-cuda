"""An exact original branch-sequence exit, without access or lifetime facts."""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256

from compiler.frontend.source_effects import _kind
from compiler.frontend.source_functions import _hash, _original
from compiler.frontend.structured_effects import _freeze_source_record, _thaw
from compiler.ir import CompilationError

_ISSUER = object()


def _span(node):
    item = getattr(node, "item", None)
    span = getattr(item, "fort_original_span", getattr(item, "span", None))
    if span is None or len(span) != 2:
        raise CompilationError("local sequence exit requires its exact original edit anchor")
    return tuple(span)


def _derive(analysis, procedure, originals):
    _original(analysis, procedure)
    graph, selected = analysis._selected_source(procedure, originals)
    if (len(selected) < 2 or len(selected) != len(originals)
            or any(graph.nodes[identity].kind != "call" for identity in selected)):
        raise CompilationError("local sequence exit requires distinct complete original CALLs")
    if any(len(graph.source_nodes(identity)) != 1 or graph.source_nodes(identity)[0] is not original
           for identity, original in zip(selected, originals, strict=True)):
        raise CompilationError("local sequence exit lost its original CALL selection")
    parents = [node for node in graph.nodes.values() if node.kind == "sequence"
               and len(node.children) >= len(selected) and tuple(node.children[-len(selected):]) == selected]
    if len(parents) != 1:
        raise CompilationError("local sequence exit requires one exact contiguous sequence suffix")
    parent = parents[0]
    branches = [(node, ordinal) for node in graph.nodes.values() if node.kind == "branch"
                for ordinal, (_condition, body) in enumerate(node.alternatives) if body == parent.id]
    if len(branches) != 1:
        raise CompilationError("local sequence exit currently requires an original IF branch")
    branch, ordinal = branches[0]
    source, = graph.source_nodes(branch.id)
    if _kind(source) != "If_Construct":
        raise CompilationError("local sequence exit requires a complete original IF construct")
    headers = [node for node in source.content
               if _kind(node) in {"If_Then_Stmt", "Else_If_Stmt", "Else_Stmt", "End_If_Stmt"}]
    if ordinal + 1 >= len(headers) or _kind(headers[-1]) != "End_If_Stmt":
        raise CompilationError("local sequence exit lost its original branch boundary")
    header, anchor = headers[ordinal:ordinal+2]
    first, last, boundary = _span(originals[0]), _span(originals[-1]), _span(anchor)
    if first[0] <= _span(header)[1] or last[1] >= boundary[0]:
        raise CompilationError("local sequence exit has no independent original insertion span")
    record = {"version": 1, "kind": "original_branch_sequence_exit", "procedure": procedure,
              "source_inputs": analysis.inputs.identity(), "source_graph": graph.identity,
              "parent_sequence": parent.id, "branch_node_id": branch.id, "branch_ordinal": ordinal,
              "selected_node_ids": list(selected), "last_call_node_id": selected[-1],
              "source_span": [first[0], last[1]],
              "exit_anchor": {"parser_kind": _kind(anchor), "source_span": list(boundary),
                              "original_source_sha256": sha256(str(anchor).encode()).hexdigest()},
              "cleanup": "publish observable storage and wait; destroy domain and handles immediately after last CALL",
              "activation": "inside unchanged original branch after predicate evaluation",
              "execution_authorized": False, "entry_definition_authority": False,
              "cross_exit_retention_authorized": False, "alias_authority": False}
    analysis.verify_publication()
    return record, graph, source, anchor


@dataclass(frozen=True, slots=True)
class OriginalSequenceExitProof:
    identity: str
    _analysis: object = field(repr=False, compare=False)
    _procedure: str
    _originals: tuple = field(repr=False, compare=False)
    _graph: object = field(repr=False, compare=False)
    _branch: object = field(repr=False, compare=False)
    _anchor: object = field(repr=False, compare=False)
    _record: object = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)

    def public(self):
        return {"identity": self.identity, **_thaw(self._record)}

    def validate(self, analysis=None, procedure=None, originals=None):
        analysis = self._analysis if analysis is None else analysis
        procedure = self._procedure if procedure is None else procedure
        originals = self._originals if originals is None else tuple(originals)
        if (self._issuer is not _ISSUER or analysis is not self._analysis or procedure != self._procedure
                or len(originals) != len(self._originals)
                or any(a is not b for a, b in zip(originals, self._originals, strict=True))
                or getattr(analysis, "_original_sequence_exits", {}).get(self.identity) is not self):
            raise CompilationError("local sequence exit requires its exact registered original source")
        record, graph, branch, anchor = _derive(analysis, procedure, originals)
        if (graph is not self._graph or branch is not self._branch or anchor is not self._anchor
                or _hash(record) != self.identity or _freeze_source_record(analysis, record) != self._record):
            raise CompilationError("local sequence exit source, branch or cleanup anchor changed")
        return self


def prove_original_sequence_exit(analysis, procedure, originals):
    originals = tuple(originals)
    record, graph, branch, anchor = _derive(analysis, procedure, originals)
    identity = _hash(record)
    registry = getattr(analysis, "_original_sequence_exits", None)
    if registry is None:
        registry = analysis._original_sequence_exits = {}
    if identity not in registry:
        registry[identity] = OriginalSequenceExitProof(identity, analysis, procedure, originals, graph,
            branch, anchor, _freeze_source_record(analysis, record), _ISSUER)
    return registry[identity].validate(analysis, procedure, originals)
