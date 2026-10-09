"""Source-authoritative control skeletons for demanded effect analysis.

The graph retains source order and guards without expanding call closures.
An available graph can contain explicit boundaries; it does not establish that
an arbitrary original call can execute within a resident owner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from types import MappingProxyType

from fparser.two.utils import walk

from compiler.frontend.call_bindings import resolve_source_call
from compiler.ir import CompilationError

STRUCTURED_EFFECT_VERSION = 1


def _kind(node):
    return type(node).__name__


def _children(node):
    return getattr(node, "content", getattr(node, "items", ())) or ()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value):
    if isinstance(value, (dict, MappingProxyType)):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw(item) for item in value]
    return value


def _span(node):
    spans = [getattr(item.item, "fort_original_span", item.item.span)
             for item in walk(node) if getattr(item, "item", None) is not None]
    spans = [span for span in spans if span is not None]
    return (min(span[0] for span in spans), max(span[1] for span in spans)) if spans else ()


@dataclass(frozen=True)
class EffectNode:
    id: str
    kind: str
    guard: tuple[str, ...] = ()
    span: tuple[int, ...] = ()
    children: tuple[str, ...] = ()
    alternatives: tuple[tuple[str | None, str], ...] = ()
    details: dict = field(default_factory=dict, compare=False, repr=False)

    def public(self):
        return {"id": self.id, "kind": self.kind, "guard": list(self.guard),
                "span": list(self.span), "children": list(self.children),
                "alternatives": [{"condition": condition, "body": body}
                                 for condition, body in self.alternatives], **_thaw(self.details)}


@dataclass(frozen=True)
class StructuredSummary:
    procedure: str
    identity: str
    authority_identity: str
    root_id: str
    entry_id: str
    nodes: object
    available: bool
    reasons: tuple[str, ...]
    node_limit: int
    _originals: dict = field(compare=False, repr=False)
    _ids: dict = field(compare=False, repr=False)
    version: int = STRUCTURED_EFFECT_VERSION

    def node_id(self, original_node, role="statement"):
        key = (id(original_node), role)
        if key not in self._ids:
            raise CompilationError("structured effect node lacks original source authority")
        return self._ids[key]

    def source_nodes(self, node_id):
        if node_id not in self.nodes:
            raise CompilationError("structured effect node is unavailable")
        return self._originals[node_id]

    def public(self):
        return {"schema_version": self.version, "procedure": self.procedure,
                "structured_identity": self.identity, "analysis_identity": self.authority_identity,
                "root": self.root_id, "entry": self.entry_id, "available": self.available,
                "reasons": list(self.reasons), "node_limit": self.node_limit,
                "nodes": [node.public() for node in self.nodes.values()]}


def build_structure(analysis, routine, authority_identity):
    nodes, originals, ids, reasons = {}, {}, {}, []
    # Entry/root containers are not source operations. The remaining bounded
    # graph retains individual call sites, conditions and native operations.
    limit = analysis.operation_limit + 2

    def add(path, kind, source=(), guard=(), *, role="statement", aliases=(), **values):
        identity = routine.qualified + "/" + path
        if len(nodes) >= limit:
            raise CompilationError("bounded structured source node budget exhausted")
        source = tuple(source)
        node = EffectNode(identity, kind, tuple(guard),
                          _span(source[0]) if len(source) == 1 else
                          ((_span(source[0])[0], _span(source[-1])[-1])
                           if source and _span(source[0]) and _span(source[-1]) else ()),
                          tuple(values.pop("children", ())), tuple(values.pop("alternatives", ())), _freeze(values))
        nodes[identity], originals[identity] = node, source
        for original in (*source, *aliases):
            key = id(original), role
            if key in ids and ids[key] != identity:
                raise CompilationError("structured source node has ambiguous original identity")
            ids[key] = identity
        return identity

    def sequence(items, path, guard=(), depth=0):
        if depth > analysis.depth_limit:
            return add(path, "boundary", items, guard,
                       reason="bounded structured source control depth exhausted")
        children = []
        for index, original in enumerate(items):
            kind, child_path = _kind(original), path + "/" + str(index)
            if kind == "Comment":
                if str(original).lstrip().lower().startswith("!$omp"):
                    children.append(add(child_path, "operation", (original,), guard,
                                        evaluation="directive", completion="original joined source group required"))
                continue
            if kind == "If_Construct":
                alternatives, current, header, previous = [], [], None, []
                for item in original.content:
                    label = _kind(item)
                    if label in {"If_Then_Stmt", "Else_If_Stmt", "Else_Stmt", "End_If_Stmt"}:
                        if header is not None:
                            branch_guard = (*guard, *(".not.(" + c + ")" for c in previous))
                            if _kind(header) != "Else_Stmt":
                                condition = header.items[0]
                                condition_id = add(child_path + "/condition" + str(len(alternatives)),
                                                   "operation", (condition,), branch_guard,
                                                   role="condition", aliases=(header,), evaluation="condition")
                                branch_guard = (*branch_guard, str(condition))
                                previous.append(str(condition))
                            else:
                                condition_id = None
                            body_id = sequence(current, child_path + "/body" + str(len(alternatives)),
                                               branch_guard, depth + 1)
                            alternatives.append((condition_id, body_id))
                        current, header = [], item
                    else:
                        current.append(item)
                children.append(add(child_path, "branch", (original,), guard, alternatives=alternatives))
            elif kind == "If_Stmt":
                condition, action = original.items
                condition_id = add(child_path + "/condition", "operation", (condition,), guard,
                                   role="condition", aliases=(original,), evaluation="condition")
                body_id = sequence((action,), child_path + "/body", (*guard, str(condition)), depth + 1)
                children.append(add(child_path, "branch", (original,), guard,
                                    alternatives=((condition_id, body_id),)))
            elif kind in {"Block_Nonlabel_Do_Construct", "Block_Label_Do_Construct"}:
                body = [item for item in original.content if _kind(item) != "Comment"]
                if not body:
                    children.append(add(child_path, "boundary", (original,), guard, reason="empty loop syntax"))
                    continue
                header = body[0]
                header_id = add(child_path + "/header", "operation", (header,), guard,
                                role="header", evaluation="loop_header")
                body_id = sequence(body[1:-1], child_path + "/body", (*guard, str(header)), depth + 1)
                children.append(add(child_path, "loop", (original,), guard,
                                    children=(header_id, body_id), control=str(header)))
            elif kind == "Associate_Construct":
                record = getattr(analysis, "_associate_scopes", {}).get(id(original))
                if record is None or not record.available:
                    children.append(add(child_path, "boundary", (original,), guard,
                                        reason=getattr(record, "reason", None) or "unproved lexical ASSOCIATE selector"))
                else:
                    header = original.content[0]
                    selector_id = add(child_path + "/selector", "operation", (header,), guard,
                                      role="selector", evaluation="associate_selector")
                    body_id = sequence(original.content[1:-1], child_path + "/body", guard, depth + 1)
                    children.append(add(child_path, "associate", (original,), guard,
                                        children=(selector_id, body_id), association=record.public()))
            elif kind == "Call_Stmt":
                scope = analysis.source_scope_for(original, routine.scope)
                try:
                    resolved = resolve_source_call(analysis, scope, original)
                    callee = analysis.routines[resolved.procedure]
                    changes = []
                    for mapping in resolved.mappings:
                        if mapping.formal_binding.intent == "out":
                            changes.append({"kind": "definition_change",
                                "formal_resource": "argument::" + mapping.formal,
                                "resource": mapping.resource, "storage": mapping.public()["storage"],
                                "section": mapping.public()["section"], "presence": mapping.presence,
                                "guard": list(guard), "position": "callee entry after original actual evaluation"})
                    children.append(add(child_path, "call", (original,), guard,
                                        **resolved.public(), definition_events=changes,
                                        source_target_identity=analysis.sources[str(callee.scope.path)]))
                except CompilationError as error:
                    # Reviewed opaque contracts are materialized using the
                    # ordinary effect proof at the original call position.
                    target = str(original.items[0]).lower()
                    matches = analysis._candidates(scope, target)
                    contracted = len(matches) == 1 and matches[0] in analysis.contracts
                    children.append(add(child_path, "call" if contracted else "boundary", (original,), guard,
                                        **({"procedure": matches[0], "contract": True} if contracted else {}),
                                        reason=None if contracted else str(error)))
            elif kind in {"Assignment_Stmt", "Continue_Stmt"}:
                children.append(add(child_path, "operation", (original,), guard, evaluation="statement"))
            else:
                children.append(add(child_path, "boundary", (original,), guard,
                                    reason="native ordering/effects unavailable: " + kind))
        return add(path, "sequence", tuple(items), guard, children=children, role="sequence")

    specification = next((node for node in _children(routine.scope.node)
                          if _kind(node) == "Specification_Part"), None)
    entry_id = add("entry", "entry", (specification,) if specification is not None else (),
                   role="entry", definition_changes=[binding.root for binding in routine.scope.bindings.values()
                       if binding.name in routine.arguments and binding.intent == "out"])
    try:
        root_id = sequence(_children(routine.execution), "body")
    except CompilationError as error:
        reasons.append(str(error))
        # A truncated graph is never advertised as an executable skeleton.
        nodes, originals, ids = {entry_id: nodes[entry_id]}, {entry_id: originals[entry_id]}, {}
        root_id = add("body", "boundary", _children(routine.execution),
                      reason="bounded structured source node budget exhausted")
    public = {"version": STRUCTURED_EFFECT_VERSION, "procedure": routine.qualified,
              "analysis_identity": authority_identity, "entry": entry_id, "root": root_id,
              "available": not reasons, "reasons": reasons, "nodes": [node.public() for node in nodes.values()]}
    identity = sha256(_canonical(public).encode()).hexdigest()
    return StructuredSummary(routine.qualified, identity, authority_identity, root_id, entry_id,
                             MappingProxyType(nodes), not reasons, tuple(reasons), limit,
                             MappingProxyType(originals), MappingProxyType(ids))


def admit_cached_structure(public, expected):
    """A cache transports facts; freshly bound source nodes grant authority."""
    if type(public) is not dict or public != expected.public():
        raise ValueError("cached structured source proof differs from original authority")


def mapped_resource(resource, mapping):
    """Map a formal root and its proven component suffix at a delimiter."""
    if resource in mapping:
        return mapping[resource]
    candidates = [(formal, actual) for formal, actual in mapping.items()
                  if resource.startswith(formal + "%")]
    if not candidates:
        return resource
    formal, actual = max(candidates, key=lambda pair: len(pair[0]))
    return None if actual is None else actual + resource[len(formal):]


def descriptor_stability(analysis, requested):
    """Prove original allocatable descriptors unchanged on every source path.

    This proof is independent of payload effects and does not authorize an
    unallocated/present/aliased actual. Those checks remain at original callers.
    Unknown effects conservatively prevent borrowing any relevant descriptor.
    """
    cache, active, unknown = {}, [], set()

    def prove(name):
        if name in cache:
            return cache[name]
        routine = analysis.routines[name]
        resources = {binding.root: [] for binding in routine.scope.bindings.values()
                     if binding.name in routine.arguments and "allocatable" in binding.attributes}
        if name in active or len(active) >= analysis.depth_limit or len(cache) + len(active) >= analysis.procedure_limit:
            unknown.add(name)
            return {root: ["bounded descriptor closure or recursion unavailable"] for root in resources}
        graph = analysis.structure(name)
        if not graph.available:
            unknown.add(name)
            return {root: ["bounded descriptor source skeleton unavailable"] for root in resources}
        active.append(name)

        def reject(reason, roots=None):
            if roots is None:
                unknown.add(name)
            for root in resources if roots is None else roots:
                if root in resources and reason not in resources[root]:
                    resources[root].append(reason)

        for binding in routine.scope.bindings.values():
            if binding.root not in resources:
                continue
            if binding.intent not in {"in", "inout"}:
                reject("original allocatable INTENT(OUT) or unspecified descriptor may change at entry", (binding.root,))
            if binding.attributes & {"pointer", "volatile", "asynchronous"}:
                reject("descriptor association or observation is uncertain", (binding.root,))
        for node in graph.nodes.values():
            if node.kind == "boundary":
                reject(node.details.get("reason") or "unknown descriptor effects")
            elif node.kind == "call":
                if node.details.get("contract"):
                    contract = analysis.contracts.get(node.details["procedure"], {})
                    if not (contract.get("complete") is True and contract.get("descriptor_changes") is False
                            and contract.get("lifetime") == "stable" and contract.get("escapes") is False
                            and contract.get("ordering") == "serial" and isinstance(contract.get("identity"), str)
                            and contract["identity"] and isinstance(contract.get("effects"), list)):
                        reject("opaque call lacks a complete stable descriptor contract")
                    continue
                child = node.details["procedure"]
                child_resources = prove(child)
                if child in unknown:
                    reject("callee descriptor/escape closure is incomplete: " + child)
                for mapping in node.details["resource_mappings"]:
                    actual = mapping["resource"]
                    formal = mapping["formal_resource"]
                    if actual in resources and formal in child_resources and child_resources[formal]:
                        reject("callee descriptor changes are unproved: " + child, (actual,))
            elif node.kind == "operation" and node.details.get("evaluation") == "statement":
                for original in graph.source_nodes(node.id):
                    if _kind(original) != "Assignment_Stmt":
                        continue
                    target, _equals, value = original.items
                    scope = analysis.source_scope_for(original, routine.scope)
                    if _kind(target) in {"Name", "Data_Ref"}:
                        try:
                            binding = analysis._binding(scope, target)
                        except CompilationError:
                            binding = None
                        if binding is not None and binding.root in resources:
                            reject("whole allocatable assignment may reallocate original storage", (binding.root,))
                    for expression in walk(value):
                        kind = _kind(expression)
                        if kind in {"Function_Reference", "Structure_Constructor"}:
                            reject("function descriptor/escape effects are unproved")
                        elif kind == "Part_Ref":
                            binding = analysis._binding(scope, expression.items[0])
                            if binding is None or not binding.rank:
                                reject("function descriptor/escape effects are unproved")
                        elif kind == "Intrinsic_Function_Reference":
                            intrinsic = str(expression.items[0]).lower()
                            if analysis._binding(scope, intrinsic) or analysis._candidates(scope, intrinsic):
                                reject("shadowed intrinsic descriptor effects are unproved")
        active.pop()
        cache[name] = resources
        return resources

    resources = prove(requested)
    graph = analysis.structure(requested)
    return {"procedure": requested, "structured_identity": graph.identity,
            "analysis_identity": graph.authority_identity,
            "complete": graph.available and requested not in unknown and all(not reasons for reasons in resources.values()),
            "resources": [{"resource": root, "stable": not reasons,
                           "original_descriptor_required": True, "reasons": reasons,
                           "execution_requires_runtime_guard": True}
                          for root, reasons in sorted(resources.items())],
            "requirements": ["original whole allocation descriptor", "original allocation and presence guards",
                             "independent alias and lifetime proof", "no allocation identity change"]}
