"""Original orphan worksharing structure; no storage or completion authority.

Only the complete original callee's descriptor scanner may consume this token.
Its loop bodies, calls and lifetime effects still need their ordinary checks.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from hashlib import sha256

from fparser.two.utils import walk

from compiler.frontend.descriptor_teams import (
    _counted_loop,
    _group_syntax_checkpoint,
    _implicit_namespace_checkpoint,
    _same_group_syntax,
    _same_implicit_namespace,
    _worksharing_sequence,
)
from compiler.frontend.native_group_structure import (
    NATIVE_GROUP_SCAN_LIMIT,
    NativeGroupCandidate,
    directive,
    grouped_original_sequence,
)
from compiler.frontend.optional_scalar_control import _same_scope_state, _scope_state
from compiler.frontend.source_functions import _validate_scope
from compiler.ir import CompilationError


class WorksharingStorageUnsupported(CompilationError):
    """Original structure is outside this finite descriptor-only contract."""


ORIGINAL_WORKSHARING_STORAGE_VERSION = 1
DESCRIPTOR_WINDOW_ORIGINAL_WORKSHARING_VERSION = 23
_ISSUER = object()
_REGISTRY = "_original_worksharing_storage"
_OWNER = "_original_worksharing_storage_owner"
_RECEIPTS = "_original_worksharing_storage_receipts"


class _IssuedReceipt(tuple):
    """An immutable exact-object baseline borrowed with its issued proof.

    Generation transactions copy the mutable registry map, but must not
    recursively detach these original proof-field identities. This frontend
    object follows the existing borrowed source-authority ownership policy.
    """
    __slots__ = ()

    def __new__(cls, proof, attached):
        return tuple.__new__(cls, (proof, attached))


def _kind(node):
    return type(node).__name__


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _attached(value):
    return tuple((part.name, getattr(value, part.name)) for part in fields(value))


def _same_attached(value, saved):
    return all(getattr(value, name) is item for name, item in saved)


def _source_anchors(analysis):
    inputs = analysis.inputs
    attached = tuple(vars(inputs).items())
    pending, seen, containers = [value for _name, value in attached], set(), []
    while pending:
        value = pending.pop()
        if id(value) in seen or not isinstance(value, (Mapping, tuple, list, set, frozenset)):
            continue
        seen.add(id(value))
        if isinstance(value, Mapping):
            items = tuple(value.items())
            containers.append((value, "mapping", items))
            pending.extend(item for _key, item in items)
        elif isinstance(value, (set, frozenset)):
            containers.append((value, "set", frozenset(value)))
        else:
            items = tuple(value)
            containers.append((value, "sequence", items))
            pending.extend(items)
    session = getattr(analysis, "_source_session", None)
    session_fields = _attached(session) if session is not None else ()
    maps = tuple((value, tuple(value.items())) for _name, value in session_fields if isinstance(value, Mapping))
    return inputs, attached, tuple(containers), session, session_fields, maps


def _same_source_anchors(analysis, saved):
    inputs, attached, containers, session, session_fields, maps = saved
    return (analysis.inputs is inputs and len(vars(inputs)) == len(attached)
            and all(vars(inputs).get(name) is value for name, value in attached)
            and getattr(analysis, "_source_session", None) is session
            and (session is None or _same_attached(session, session_fields))
            and all(len(value) == len(items) and all(value.get(key) is item for key, item in items)
                    for value, items in maps)
            and all(len(value) == len(items) and (
                all(key in value and value[key] is item for key, item in items) if kind == "mapping"
                else value == items if kind == "set"
                else all(a is b for a, b in zip(value, items, strict=True)))
                for value, kind, items in containers))


def _graph_checkpoint(graph):
    tables = tuple((name, getattr(graph, name), tuple(getattr(graph, name).items()))
                   for name in ("nodes", "_originals", "_ids", "native_groups"))
    nodes = tuple((node, _attached(node)) for node in graph.nodes.values())
    return _attached(graph), tables, nodes


def _same_graph(graph, saved):
    attached, tables, nodes = saved
    return (_same_attached(graph, attached)
            and all(getattr(graph, name) is table and len(table) == len(items)
                    and all(table.get(key) is value for key, value in items)
                    for name, table, items in tables)
            and all(_same_attached(node, fields) for node, fields in nodes))


def _registry(analysis):
    registry = getattr(analysis, _REGISTRY, None)
    if registry is None or getattr(analysis, _OWNER, None) is not analysis:
        registry = {}
        setattr(analysis, _REGISTRY, registry)
        setattr(analysis, _OWNER, analysis)
        setattr(analysis, _RECEIPTS, {})
    if type(registry) is not dict or type(getattr(analysis, _RECEIPTS, None)) is not dict:
        raise CompilationError("worksharing storage requires its compiler-owned registry")
    return registry


def _registry_receipt(analysis, registry):
    issued = getattr(analysis, _RECEIPTS)
    if len(issued) != len(registry):
        raise CompilationError("worksharing storage lost its independent issuer receipt")
    rows = tuple((key, proof, _attached(proof)) for key, proof in registry.items()
                 if type(proof) is OriginalWorksharingStorage)
    if (len(rows) != len(registry) or any(type(issued.get(key)) is not _IssuedReceipt
            or issued[key][0] is not proof or not _same_attached(proof, issued[key][1])
            for key, proof, _saved in rows)):
        raise CompilationError("worksharing storage registry contains foreign authority")
    return rows, issued, tuple(issued.items())


def _same_registry(analysis, registry, saved):
    rows, issued, receipts = saved
    return (getattr(analysis, _RECEIPTS, None) is issued and len(issued) == len(receipts)
            and all(issued.get(key) is value for key, value in receipts) and _same_registry_rows(analysis, registry, rows))


def _same_registry_rows(analysis, registry, rows):
    return (getattr(analysis, _REGISTRY, None) is registry
            and getattr(analysis, _OWNER, None) is analysis and len(registry) == len(rows)
            and all(registry.get(key) is proof and _same_attached(proof, saved)
                    for key, proof, saved in rows))


def _span(node):
    item = getattr(node, "item", None)
    span = getattr(item, "fort_original_span", getattr(item, "span", None))
    if span is None or len(span) != 2 or any(type(n) is not int for n in span):
        raise CompilationError("worksharing storage requires exact original source positions")
    return list(span)


def _stamp(node):
    return {"source_span": _span(node), "original_source": str(node),
            "syntax_sha256": sha256(str(node).encode()).hexdigest()}


def _regions(analysis, routine):
    """Only top-level original orphan loops; complete teams remain separate."""
    from compiler.scopes.participation import _nodes

    peeled = {}
    sequence = grouped_original_sequence(_nodes(routine), peeled)
    nodes = []
    for item in sequence:
        if isinstance(item, NativeGroupCandidate):
            nodes.append(item)
        else:
            nodes.extend(_worksharing_sequence((item,), peeled))
    result, index = [], 0
    while index < len(nodes):
        node = nodes[index]
        if isinstance(node, NativeGroupCandidate):
            index += 1
            continue  # No claim about this group's storage or completion.
        text = directive(node)
        if text is None:
            index += 1
            continue
        match = re.fullmatch(r"do(?:\s+collapse\s*\(\s*([1-4])\s*\))?", text)
        if match is None:
            raise WorksharingStorageUnsupported("worksharing storage requires a restricted original orphan DO")
        collapse = int(match.group(1) or "1")
        opening = node
        index += 1
        while index < len(nodes) and _kind(nodes[index]) == "Comment" and directive(nodes[index]) is None:
            index += 1
        if index >= len(nodes) or _kind(nodes[index]) != "Block_Nonlabel_Do_Construct":
            raise WorksharingStorageUnsupported("worksharing storage requires its associated ordinary counted loop")
        loop, current = nodes[index], nodes[index]
        for depth in range(collapse):
            body = _counted_loop(current, peeled)
            control = body[0].items[1].items[1]
            binding = analysis._binding(routine.scope, control[0])
            if (binding is None or binding.declaring_scope is not routine.scope
                    or binding.name in routine.arguments or binding.dtype != "integer"
                    or binding.kind not in {4, 8} or binding.rank or binding.intent is not None
                    or binding.attributes & {"pointer", "target", "allocatable", "volatile",
                                             "asynchronous", "optional", "save", "parameter"}):
                raise WorksharingStorageUnsupported("worksharing storage iterator requires its original ordinary INTEGER local")
            if depth + 1 < collapse:
                nested = [item for item in body[1:-1] if _kind(item) != "Comment"]
                if len(nested) != 1 or _kind(nested[0]) != "Block_Nonlabel_Do_Construct":
                    raise WorksharingStorageUnsupported("worksharing storage COLLAPSE requires its exact perfect counted nest")
                current = nested[0]
        prefix = []
        for item in loop.content:
            if _kind(item) != "Comment":
                break
            prefix.append(item)
        for nested in walk(loop):
            if (directive(nested) is not None and not any(nested is item for item in prefix)) or _kind(nested) in {
                    "Call_Stmt", "Return_Stmt", "Exit_Stmt", "Cycle_Stmt", "Goto_Stmt",
                    "Computed_Goto_Stmt", "Assigned_Goto_Stmt", "Stop_Stmt", "Error_Stop_Stmt"}:
                raise WorksharingStorageUnsupported("worksharing storage loop has an unsupported nested directive, call or exit")
        index += 1
        while index < len(nodes) and _kind(nodes[index]) == "Comment" and directive(nodes[index]) is None:
            index += 1
        if index >= len(nodes) or directive(nodes[index]) not in {"end do", "end do nowait"}:
            raise WorksharingStorageUnsupported("worksharing storage requires its exact original END DO or END DO NOWAIT")
        ending = nodes[index]
        header, end_loop = _counted_loop(loop, peeled)[0], _counted_loop(loop, peeled)[-1]
        if not _span(opening)[1] < _span(header)[0] <= _span(end_loop)[1] < _span(ending)[0]:
            raise WorksharingStorageUnsupported("worksharing storage original loop/end positions do not match")
        result.append((opening, loop, ending, collapse))
        index += 1
        if len(result) > NATIVE_GROUP_SCAN_LIMIT:
            raise CompilationError("worksharing storage original source budget exhausted")
    if not result:
        raise WorksharingStorageUnsupported("worksharing storage requires an original orphan DO in the complete callee")
    return tuple(result)


def _derive(analysis, routine):
    if (getattr(analysis, "_descriptor_source", None) is not None
            or analysis.routines.get(routine.qualified) is not routine):
        raise CompilationError("worksharing storage requires its exact original analysis and callee")
    anchors = _source_anchors(analysis)
    cached = tuple((name, getattr(analysis, name), key, graph, _graph_checkpoint(graph))
                   for name in ("_structures", "_live_structures")
                   for key, graph in getattr(analysis, name).items() if key[0] == routine.qualified)
    roots = (routine.scope.node,)
    syntax = _group_syntax_checkpoint(roots)
    namespace = _implicit_namespace_checkpoint(analysis, routine, syntax)
    routine_map = analysis.routines

    def checkpoint():
        if (not _same_source_anchors(analysis, anchors)
                or analysis.routines is not routine_map or routine_map.get(routine.qualified) is not routine
                or not _same_group_syntax(syntax) or not _same_implicit_namespace(analysis, namespace)
                or any(getattr(analysis, name) is not table or table.get(key) is not graph
                       or not _same_graph(graph, saved) for name, table, key, graph, saved in cached)):
            raise CompilationError("worksharing storage original source or namespace attachment changed")

    analysis._require_original(routine.qualified)
    checkpoint()
    graph = analysis.structure(routine.qualified)
    graph_fields = _graph_checkpoint(graph)
    checkpoint()
    state = _scope_state(routine.scope)
    regions = _regions(analysis, routine)
    checkpoint()
    statements = routine.scope.node.content
    header = next((node for node in statements if _kind(node) == "Subroutine_Stmt"), None)
    ending = next((node for node in reversed(statements) if _kind(node) == "End_Subroutine_Stmt"), None)
    if header is None or ending is None:
        raise CompilationError("worksharing storage requires one complete original subroutine callee")
    callee_span = [_span(header)[0], _span(ending)[1]]
    source_path = str(routine.scope.path)
    original_lines = analysis.source_text(routine.scope.path).splitlines(keepends=True)
    record = {"version": ORIGINAL_WORKSHARING_STORAGE_VERSION,
        "kind": "original_orphan_worksharing_storage_structure", "procedure": routine.qualified,
        "source_inputs": analysis.inputs.identity(), "structured_identity": graph.identity,
        "whole_original_procedure_node_id": graph.root_id,
        "callee_source": {"path": source_path, "source_sha256": analysis.sources[source_path],
            "source_span": callee_span,
            "original_callee_sha256": sha256("".join(original_lines[callee_span[0]-1:callee_span[1]]).encode()).hexdigest()},
        "regions": [{"directive": _stamp(opening), "associated_loop": {
            "source_span": [_span(_counted_loop(loop, {})[0])[0], _span(_counted_loop(loop, {})[-1])[1]],
            "syntax_sha256": sha256(str(loop).encode()).hexdigest()},
            "end_directive": _stamp(ending), "collapse": collapse,
            "retained_nowait": directive(ending) == "end do nowait"}
            for opening, loop, ending, collapse in regions],
        "requires_complete_original_callee_and_body_rescan": True,
        "requires_whole_original_native_callee": True,
        "original_native_source_unchanged": True, "internal_cuts_authorized": False,
        "descriptor_preservation_authorized": False, "storage_lifetime_authority": False,
        "caller_participation_authority": False, "work_completion_authority": False,
        "execution_authorized": False, "payload_access_authorized": False,
        "entry_definition_authority": False, "alias_authority": False,
        "gpu_legality_established": False}
    encoded = _encoded(record)
    _validate_scope(analysis, routine.scope)
    if not _same_scope_state(state):
        raise CompilationError("worksharing storage original namespace changed during source validation")
    checkpoint()
    analysis.verify_publication()
    analysis.verify_source()
    analysis.inputs.verify_live()
    checkpoint()
    if not _same_graph(graph, graph_fields):
        raise CompilationError("worksharing storage graph attachment changed at publication")
    return encoded, regions, syntax, namespace, graph, graph_fields, anchors


@dataclass(frozen=True, slots=True)
class OriginalWorksharingStorage:
    identity: str
    _analysis: object = field(repr=False, compare=False)
    _routine: object = field(repr=False, compare=False)
    _record: str = field(repr=False, compare=False)
    _regions: tuple = field(repr=False, compare=False)
    _syntax: object = field(repr=False, compare=False)
    _namespace: object = field(repr=False, compare=False)
    _graph: object = field(repr=False, compare=False)
    _graph_fields: tuple = field(repr=False, compare=False)
    _anchors: object = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)

    def checkpoint(self):
        analysis, routine = self._analysis, self._routine
        registry = getattr(analysis, _REGISTRY, None)
        if (type(self) is not OriginalWorksharingStorage or self._issuer is not _ISSUER
                or type(registry) is not dict or getattr(analysis, _OWNER, None) is not analysis
                or registry.get(self.identity) is not self
                or type(getattr(analysis, _RECEIPTS, None)) is not dict
                or type(getattr(analysis, _RECEIPTS).get(self.identity)) is not _IssuedReceipt
                or getattr(analysis, _RECEIPTS)[self.identity][0] is not self
                or not _same_attached(self, getattr(analysis, _RECEIPTS)[self.identity][1])
                or analysis.routines.get(routine.qualified) is not routine
                or type(self._record) is not str or sha256(self._record.encode()).hexdigest() != self.identity
                or not _same_source_anchors(analysis, self._anchors)
                or not _same_group_syntax(self._syntax) or not _same_implicit_namespace(analysis, self._namespace)
                or not _same_graph(self._graph, self._graph_fields)):
            raise CompilationError("worksharing storage requires its unchanged original compiler-issued authority")
        return _attached(self), registry

    def require_checkpoint(self, saved):
        attached, registry = saved
        if not _same_attached(self, attached) or getattr(self._analysis, _REGISTRY, None) is not registry:
            raise CompilationError("worksharing storage retained receipt changed at publication")
        self.checkpoint()

    def validate(self, analysis, routine):
        saved = self.checkpoint()
        if analysis is not self._analysis or routine is not self._routine:
            raise CompilationError("worksharing storage belongs to another original callee or analysis")
        record, regions, _syntax, _namespace, graph, _graph_state, _anchors = _derive(analysis, routine)
        if (record != self._record or graph is not self._graph or len(regions) != len(self._regions)
                or any(any(a is not b for a, b in zip(current[:3], old[:3], strict=True))
                       or current[3] != old[3] for current, old in zip(regions, self._regions, strict=True))):
            raise CompilationError("worksharing storage original loop inventory changed")
        self.require_checkpoint(saved)
        return self

    def public(self):
        return {"proof_identity": self.identity, **json.loads(self._record)}


def original_worksharing_storage(analysis, routine):
    """Freshly scan once, then publish only under exact retained issuer receipts."""
    registry = _registry(analysis)
    prior = _registry_receipt(analysis, registry)
    record, regions, syntax, namespace, graph, graph_fields, anchors = _derive(analysis, routine)
    if not _same_registry(analysis, registry, prior):
        raise CompilationError("worksharing storage registry changed during original source derivation")
    identity = sha256(record.encode()).hexdigest()
    if identity in registry:
        proof = registry[identity]
        if (proof._issuer is not _ISSUER or proof._analysis is not analysis or proof._routine is not routine
                or proof._record != record or proof._graph is not graph
                or len(proof._regions) != len(regions)
                or any(any(a is not b for a, b in zip(old[:3], current[:3], strict=True))
                       for old, current in zip(proof._regions, regions, strict=True))):
            raise CompilationError("worksharing storage query found a foreign original token")
        proof.checkpoint()
        return proof
    if len(registry) >= len(analysis.routines):
        raise CompilationError("worksharing storage registry exceeds exact original source inventory")
    proof = OriginalWorksharingStorage(identity, analysis, routine, record, regions, syntax,
                                       namespace, graph, graph_fields, anchors, _ISSUER)
    registry[identity] = proof  # Exact dict; no delegated publication callback.
    getattr(analysis, _RECEIPTS)[identity] = _IssuedReceipt(proof, _attached(proof))
    proof.checkpoint()
    return proof
