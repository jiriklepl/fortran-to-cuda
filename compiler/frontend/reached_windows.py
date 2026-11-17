"""One-shot source-position state proofs; no execution or owner reopening.

The contract selects contiguous original statements at entry or in the first
traversal of one serial counted loop. It reuses ordered native effects and
source-relative descriptor proofs, meets prefix branches, and declines unknown
closures. Supplied invocation facts require independent admission. Registration,
runtime alias checks, mandatory closure and numerical legality remain separate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256


from compiler.ir import CompilationError

REACHED_WINDOW_VERSION = 16
_EFFECTS = {"read", "write", "overwrite", "descriptor_read", "definition_change"}
_FORBIDDEN = {"pointer", "optional", "volatile", "asynchronous", "value"}


def _hash(value):
    try:
        return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    except (TypeError, ValueError, RecursionError) as error:
        raise CompilationError("reached window requires finite source facts") from error


def _single_if_action_source(analysis, routine, graph, identity, *, consume=None):
    """Bind a conditional edit to one intact original IF, never its inner span."""
    if consume is not None:
        consume(1)
    if graph.procedure != routine.qualified or identity not in graph.nodes:
        raise CompilationError("conditional prefix action lacks its original procedure authority")
    node, originals = graph.nodes[identity], graph.source_nodes(identity)
    if (node.kind != "branch" or len(originals) != 1 or type(originals[0]).__name__ != "If_Stmt"
            or len(node.alternatives) != 1):
        raise CompilationError("conditional prefix action requires one exact original single-line IF")
    original = originals[0]
    condition_id, body_id = node.alternatives[0]
    if condition_id is None or body_id not in graph.nodes:
        raise CompilationError("conditional prefix action lacks its original condition and body")
    body = graph.nodes[body_id]
    if body.kind != "sequence" or len(body.children) != 1:
        raise CompilationError("conditional prefix action requires one exact original action")
    action_id, = body.children
    condition, action = original.items
    if (condition_id not in graph.nodes or action_id not in graph.nodes
            or len(graph.source_nodes(condition_id)) != 1 or graph.source_nodes(condition_id)[0] is not condition
            or len(graph.source_nodes(action_id)) != 1 or graph.source_nodes(action_id)[0] is not action):
        raise CompilationError("conditional prefix action differs from its original condition or action authority")
    item = getattr(original, "item", None)
    span = getattr(item, "fort_original_span", getattr(item, "span", None))
    if (not isinstance(span, tuple) or len(span) != 2 or any(type(value) is not int for value in span)
            or not 1 <= span[0] <= span[1] or tuple(node.span) != span):
        raise CompilationError("conditional prefix action lacks its exact original edit span")
    if str(routine.scope.path) not in analysis.sources:
        raise CompilationError("conditional prefix action lacks its original source file")
    lines = analysis.source_text(routine.scope.path).splitlines(keepends=True)
    if span[1] > len(lines):
        raise CompilationError("conditional prefix action exceeds its original source span")
    source = lines[span[0] - 1:span[1]]
    unsafe = None
    if getattr(item, "label", None) is not None:
        unsafe = "single original IF action has a labeled edit span"
    for line in source:
        if consume is not None:
            consume(len(line))
        if line.lstrip().startswith("#"):
            unsafe = "single original IF action crosses a preprocessor edit boundary"
        elif ";" in line:
            # Reject even a quoted semicolon conservatively. Replacing a whole
            # physical line must never discard a neighboring original statement.
            unsafe = "single original IF action has a potentially shared semicolon edit span"
    return {"condition_node_id": condition_id, "action_node_id": action_id,
            "condition_source_sha256": _hash([str(condition)]),
            "action_source_sha256": _hash([str(action)]), "action_source_kind": type(action).__name__,
            "original_span_sha256": sha256("".join(source).encode()).hexdigest(),
            "condition_evaluated_once": True, "action_evaluated_once": True}, unsafe


@dataclass(frozen=True)
class _ResolvedNumericalRegion:
    """One prepared/effect pair, consumed only within a validated plan pass."""

    procedure: str
    selection: tuple
    prepared: object
    effects: object
