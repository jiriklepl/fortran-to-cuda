"""Exact original structured path selection, without lifetime or GPU authority.

The selection inventories *every* edge of a contiguous original interval. A
selected CALL is only a discovery candidate. Its consumer must independently
prove every retained edge, publish/close before excluded effects, and preserve
any allocation-owned exclusion barrier across later context recreation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256

from fparser.two.utils import Base

from compiler.frontend.call_bindings import _arrange, _OriginalTypes, resolve_original_call
from compiler.frontend.source_functions import _scope_roles, _validate_scope, _validate_scope_roles
from compiler.ir import CompilationError

_ISSUER = object()
_EXCLUDED_CONTROL = {"Block_Nonlabel_Do_Construct", "Block_Label_Do_Construct", "Forall_Construct",
                     "Where_Construct", "Associate_Construct", "Select_Type_Construct", "Select_Rank_Construct",
                     "Select_Case_Construct", "Return_Stmt", "Goto_Stmt", "Computed_Goto_Stmt",
                     "Assigned_Goto_Stmt", "Exit_Stmt", "Cycle_Stmt"}


def _kind(node):
    return type(node).__name__


def _children(node):
    return node if isinstance(node, (tuple, list)) else getattr(node, "content", getattr(node, "items", ())) or ()


def _syntax(roots):
    pending = [(node, (index,), ()) for index, node in reversed(tuple(enumerate(roots)))]
    while pending:
        node, path, ancestors = pending.pop()
        if not isinstance(node, (Base, tuple, list)):
            continue
        if id(node) in ancestors:
            raise CompilationError("supported path requires acyclic original syntax")
        yield node, path
        pending.extend((child, (*path, index), (*ancestors, id(node)))
                       for index, child in reversed(tuple(enumerate(_children(node)))))


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _stamp(node, path):
    item = getattr(node, "item", None)
    return {"syntax_path": list(path), "parser_kind": _kind(node), "source": str(node),
            "syntax_sha256": sha256(str(node).encode()).hexdigest(),
            "original_span": list(getattr(item, "fort_original_span", item.span)) if item is not None else None}


def _association_receipt(analysis, routine, node, resolved):
    scope = analysis.source_scope_for(node, routine.scope)
    resolved.validate(analysis, scope, node)
    actuals, arguments = _arrange(resolved.callee, tuple(_children(node.items[1])))
    if (resolved.formals != resolved.callee.arguments or len(resolved.actuals) != len(actuals)
            or any(a is not b for a, b in zip(resolved.actuals, actuals, strict=True))
            or len(resolved.associations) != len(actuals) or len(resolved.original_arguments) != len(arguments)
            or any(a.actual is not b.actual or a.public() != b.public()
                   for a, b in zip(resolved.original_arguments, arguments, strict=True))):
        raise CompilationError("supported path requires unchanged original CALL actual objects")
    types = _OriginalTypes(analysis)
    for association, formal, actual in zip(resolved.associations, resolved.formals, actuals, strict=True):
        if (association.formal != formal or association.actual is not actual
                or association.formal_binding is not resolved.callee.scope.bindings[formal]):
            raise CompilationError("supported path original CALL association changed")
        fresh = types.actual(scope, actual) if actual is not None else None
        if ((fresh is None) != (association.actual_type is None) or fresh is not None
                and (fresh.public() != association.actual_type.public() or fresh.binding is not association.binding
                     or fresh.schema is not association.actual_type.schema)):
            raise CompilationError("supported path original CALL actual type changed")
    return (id(resolved), tuple(id(node) for node in resolved.actuals),
            tuple((id(item), id(item.actual_type), id(item.formal_binding), id(item.binding))
                  for item in resolved.associations), tuple(id(item) for item in resolved.original_arguments),
            _encoded(resolved.public()))


def _derive(analysis, routine, originals, device_roots):
    if (getattr(analysis, "_descriptor_source", None) is not None
            or analysis._require_original(routine.qualified) is not routine):
        raise CompilationError("supported path requires its exact original routine")
    if (not isinstance(originals, tuple) or not originals or _kind(originals[0]) != "Call_Stmt"
            or _kind(originals[-1]) != "Call_Stmt"):
        raise CompilationError("supported path requires original CALL endpoints")
    source = tuple(_children(routine.execution))
    positions = [index for index in range(len(source)) if len(source[index:index+len(originals)]) == len(originals)
                 and all(a is b for a, b in zip(source[index:index+len(originals)], originals, strict=True))]
    if len(positions) != 1:
        raise CompilationError("supported path requires one complete contiguous original interval")
    first, stop = positions[0], positions[0]+len(originals)
    if (not isinstance(device_roots, tuple) or not device_roots or any(not isinstance(name, str)
            or name != name.lower() or name not in analysis.routines for name in device_roots)
            or len(set(device_roots)) != len(device_roots)):
        raise CompilationError("supported path discovery requires distinct original procedure identities")
    inputs, input_identity = analysis.inputs, analysis.inputs.identity()
    roles, scopes, resolutions = _scope_roles(analysis), {}, []

    def authenticate(scope):
        _validate_scope(analysis, scope)
        scopes[id(scope)] = scope

    authenticate(routine.scope)
    graph, selected = analysis._selected_source(routine.qualified, originals)
    calls, controls, native, consumed = [], [], [], []
    for node, path in _syntax(originals):
        kind = _kind(node)
        if kind in _EXCLUDED_CONTROL:
            raise CompilationError("supported path structured control requires a separate original proof: " + kind)
        if kind == "Comment" and str(node).lstrip().startswith("!$"):
            raise CompilationError("supported path root concurrency requires a separate joined proof")
        if kind == "Call_Stmt":
            authenticate(analysis.source_scope_for(node, routine.scope))
            try:
                resolved = resolve_original_call(analysis, analysis.source_scope_for(node, routine.scope), node)
            except CompilationError:
                # Unresolved original calls are preserved as *unproved* edges.
                # No consumer may turn their presence into noescape authority.
                record = {**_stamp(node, path), "resolved": None, "device_discovery_selected": False}
            else:
                receipt = _association_receipt(analysis, routine, node, resolved)
                resolutions.append((resolved, node, receipt))
                record = {**_stamp(node, path), "resolved": resolved.public(),
                          "device_discovery_selected": resolved.procedure in device_roots}
            calls.append(record)
            consumed.append((node, path))
        elif kind in {"If_Then_Stmt", "Else_If_Stmt", "Else_Stmt", "If_Stmt"}:
            controls.append(_stamp(node, path))
            consumed.append((node, path))
        elif kind.endswith("_Stmt") and kind not in {"End_If_Stmt"}:
            native.append(_stamp(node, path))
            consumed.append((node, path))
    selected_names = {row["resolved"]["procedure"] for row in calls if row["device_discovery_selected"]}
    if selected_names != set(device_roots):
        raise CompilationError("supported path must contain every exact selected original procedure")
    record = {"version": 1, "kind": "original_supported_path_selection", "procedure": routine.qualified,
              "source_inputs": input_identity, "structured_identity": graph.identity,
              "selected_node_ids": list(selected), "device_roots": list(device_roots),
              "source_interval": {"first_ordinal": first, "stop_ordinal": stop,
                  "following_original": _stamp(source[stop], (stop,)) if stop < len(source) else None},
              "calls": calls, "controls": controls, "other_native_statements": native,
              "dispatch": "original source order and predicates retained; no early path substitution",
              "required_edge_policy": "independently checked retention or publication/close before original execution; no GPU resumption after an unproved edge",
              "required_future_epoch_policy": "unproved exposure remains barred for every possibly exposed canonical allocation until independent retirement/noescape proof",
              "discovery_only": True, "lifetime_authorized": False, "entry_definition_authority": False,
              "payload_access_authorized": False, "execution_authorized": False,
              "placement_authorized": False, "motion_authorized": False,
              "cross_frontier_retention_authorized": False}
    encoded = _encoded(record)
    fresh = {path: node for node, path in _syntax(originals)}
    if any(fresh.get(path) is not node for node, path in consumed):
        raise CompilationError("supported path consumed original occurrence changed before publication")
    for resolved, node, receipt in resolutions:
        if _association_receipt(analysis, routine, node, resolved) != receipt:
            raise CompilationError("supported path original CALL association changed before publication")
    _validate_scope_roles(analysis, roles, scopes)
    for scope in scopes.values():
        authenticate(scope)
    if analysis.inputs is not inputs or analysis.inputs.identity() != input_identity:
        raise CompilationError("supported path original source inputs changed before publication")
    analysis.verify_publication()
    return encoded, tuple(consumed), tuple(receipt for _resolved, _node, receipt in resolutions)


@dataclass(frozen=True, slots=True)
class OriginalSupportedPathSelection:
    identity: str
    _analysis: object = field(repr=False, compare=False)
    _routine: object = field(repr=False, compare=False)
    _originals: tuple = field(repr=False, compare=False)
    _device_roots: tuple = field(repr=False, compare=False)
    _record: str = field(repr=False, compare=False)
    _consumed: tuple = field(repr=False, compare=False)
    _resolutions: tuple = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)

    def validate(self):
        analysis = self._analysis
        if (self._issuer is not _ISSUER or type(self) is not OriginalSupportedPathSelection
                or type(self._record) is not str or sha256(self._record.encode()).hexdigest() != self.identity
                or getattr(analysis, "_supported_path_selections", {}).get(self.identity) is not self):
            raise CompilationError("supported path requires its exact registered original selection")
        record, consumed, resolutions = _derive(analysis, self._routine, self._originals, self._device_roots)
        if (record != self._record or len(consumed) != len(self._consumed)
                or any(a is not b or ap != bp for (a, ap), (b, bp) in zip(consumed, self._consumed, strict=True))
                or len(resolutions) != len(self._resolutions)
                or resolutions != self._resolutions
                or getattr(analysis, "_supported_path_selections", {}).get(self.identity) is not self):
            raise CompilationError("supported path source, binding or consumed edge authority changed")
        analysis.verify_publication()
        return self

    def public(self):
        return {"identity": self.identity, **json.loads(self._record)}


def prove_supported_path(analysis, routine, originals, device_roots):
    registry = getattr(analysis, "_supported_path_selections", None)
    if registry is None:
        registry = analysis._supported_path_selections = {}
    if type(registry) is not dict:
        raise CompilationError("supported path requires its private original-analysis issuer registry")
    for proof in tuple(registry.values()):
        if type(proof) is not OriginalSupportedPathSelection or proof._analysis is not analysis:
            raise CompilationError("supported path registry cannot consume public metadata as authority")
        if (proof._routine is routine and len(proof._originals) == len(originals)
                and all(a is b for a, b in zip(proof._originals, originals, strict=True))
                and proof._device_roots == device_roots):
            return proof.validate()
    record, consumed, resolutions = _derive(analysis, routine, originals, device_roots)
    if getattr(analysis, "_supported_path_selections", None) is not registry:
        raise CompilationError("supported path registry changed before publication")
    proof = OriginalSupportedPathSelection(sha256(record.encode()).hexdigest(), analysis, routine, originals,
        device_roots, record, consumed, resolutions, _ISSUER)
    registry[proof.identity] = proof
    return proof.validate()


def requested_supported_path(analysis, routine, request, device_roots):
    """Resolve a discovery request into fresh original path and gate tokens.

    Anchors and binding names restrict source selection. Neither the JSON
    nor a public proof identity can populate an authority registry.
    """
    from compiler.frontend.allocation_epoch_eligibility import prove_allocation_gate

    def anchor(value):
        if (not isinstance(value, dict) or set(value) != {"span", "sha256", "callee"}
                or not isinstance(value["span"], list) or len(value["span"]) != 2
                or any(type(item) is not int or item < 1 for item in value["span"])
                or value["span"][0] > value["span"][1] or not isinstance(value["sha256"], str)
                or len(value["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in value["sha256"])
                or not isinstance(value["callee"], str)):
            raise CompilationError("supported path request requires exact original CALL anchors")
        matches = [node for node, path in _syntax((routine.execution,)) if _kind(node) == "Call_Stmt"
                   and _stamp(node, path)["original_span"] == value["span"]
                   and sha256(str(node).encode()).hexdigest() == value["sha256"]]
        if len(matches) != 1:
            raise CompilationError("supported path request lost its exact original CALL anchor")
        node, = matches
        resolved = resolve_original_call(analysis, analysis.source_scope_for(node, routine.scope), node)
        _association_receipt(analysis, routine, node, resolved)
        if resolved.procedure != value["callee"]:
            raise CompilationError("supported path request changed its original resolved callee")
        return node

    if (not isinstance(request, dict) or set(request) != {"version", "source", "first", "last", "eligibility"}
            or type(request["version"]) is not int or request["version"] != 1
            or not isinstance(request["source"], dict) or set(request["source"]) != {"path", "sha256"}
            or request["source"]["path"] != str(routine.scope.path)
            or request["source"]["sha256"] != analysis.sources[str(routine.scope.path)]):
        raise CompilationError("supported path request requires its exact original source identity")
    eligibility = request["eligibility"]
    if (not isinstance(eligibility, dict) or set(eligibility) != {"binding", "resources", "barred_calls"}
            or not isinstance(eligibility["binding"], str) or eligibility["binding"] not in routine.scope.bindings
            or not isinstance(eligibility["resources"], list) or not eligibility["resources"]
            or any(not isinstance(root, str) for root in eligibility["resources"])
            or len(set(eligibility["resources"])) != len(eligibility["resources"])
            or not isinstance(eligibility["barred_calls"], list) or not eligibility["barred_calls"]):
        raise CompilationError("supported path request requires exact original sticky-gate bindings")
    first, last = anchor(request["first"]), anchor(request["last"])
    original = tuple(_children(routine.execution))
    positions = [index for index, node in enumerate(original) if node is first or node is last]
    if len(positions) != 2 or original[positions[0]] is not first or original[positions[1]] is not last:
        raise CompilationError("supported path request requires original top-level CALL endpoints")
    path = prove_supported_path(analysis, routine, original[positions[0]:positions[1]+1], tuple(device_roots))
    resources = []
    for root in eligibility["resources"]:
        matches = [binding for binding in routine.scope.bindings.values() if binding.root == root]
        if len(matches) != 1:
            raise CompilationError("supported path request lost an original local allocation identity")
        resources.append(matches[0])
    barriers = tuple(anchor(value) for value in eligibility["barred_calls"])
    source_calls = tuple(node for node, _location in _syntax(path._originals) if _kind(node) == "Call_Stmt")
    positions = tuple(next(index for index, original in enumerate(source_calls) if original is node)
                      for node in barriers if any(original is node for original in source_calls))
    if (len(positions) != len(barriers) or tuple(sorted(set(positions))) != positions):
        raise CompilationError("supported path barred CALL anchors require exact distinct original source order")
    gate = prove_allocation_gate(analysis, routine, routine.scope.bindings[eligibility["binding"]], resources, barriers)
    contained = {id(node) for node, _location in _syntax(path._originals)}
    if any(id(node) not in contained for node in barriers):
        raise CompilationError("supported path request barred CALL lies outside the exact selected interval")
    analysis.verify_publication()
    return path, gate
