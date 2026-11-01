"""Exact owned record-array element associations; no payload or execution grant.

The slice describes one original combined PARALLEL DO inside a complete callee. A local
PRIVATE pointer may refer only to one element of its exact record dummy.
Consumers must still scan the whole original body and hidden module effects.
Consumers compose this token with an independent complete team/storage scan.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, fields
from hashlib import sha256

from fparser.two.utils import walk

from compiler.frontend.call_bindings import _resolve_candidates
from compiler.frontend.component_bindings import _typename
from compiler.frontend.descriptor_associations import _attached, _attachments
from compiler.frontend.descriptor_budget import DEFAULT_SOURCE_SCAN_LIMIT, SOURCE_SYNTAX_PASSES
from compiler.frontend.descriptor_teams import _original_group
from compiler.frontend.native_demand import _Declarations
from compiler.frontend.native_group_structure import NativeGroupCandidate, directive, grouped_original_sequence
from compiler.frontend.proof_arena import SourcePayload, _register_family
from compiler.frontend.source_dependency_receipts import (
    DependencyCapture,
    _PrivateDependencies,
    _resolve_dependencies,
    _SourceOwnership,
)
from compiler.frontend.source_functions import _original, _validate_resolution, _validate_scope, original_name_absent
from compiler.frontend.source_objects import _TypeBuilder
from compiler.ir import CompilationError

VERSION = 2
_ISSUER = object()


class OwnedRecordRegionUnsupported(CompilationError):
    """Unsupported original association; source/publication failures remain hard."""


_UNSAFE = {"optional", "volatile", "asynchronous", "value", "external", "intrinsic"}


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _kind(node):
    return type(node).__name__


def _children(node):
    return node if isinstance(node, (tuple, list)) else getattr(node, "content", getattr(node, "items", ())) or ()


def _binding(analysis, scope, node):
    return analysis._binding(analysis.source_scope_for(node, scope), node)


def _original_binding(analysis, binding, types):
    scope = binding.declaring_scope
    _validate_scope(analysis, scope)
    authority = types.scope(scope)
    if scope.bindings.get(binding.name) is not binding:
        raise CompilationError("record region requires its exact original binding")
    if _typename(binding.dtype) is not None:
        original = authority[4].get(binding.name)
        if original is None or original[0] is not binding or original[1] != _encoded(binding.public()):
            raise CompilationError("record region declaring storage changed")
    return scope


def _owned(analysis, binding, types, *, record):
    scope = _original_binding(analysis, binding, types)
    module = analysis.modules.get(getattr(scope, "module", None))
    routine = analysis.numerical_helpers.get(getattr(scope, "qualified", None))
    formal = (not record and routine is not None and routine.scope is scope
              and routine.source_kind in {"module", "internal"}
              and binding.name in routine.arguments and binding.intent in {"in", "inout"})
    header = next((node for node in _children(scope.node) if _kind(node) in {"Subroutine_Stmt", "Function_Stmt"}), None)
    if formal and any(_kind(node) == "Language_Binding_Spec" for node in walk(header)):
        raise OwnedRecordRegionUnsupported("record region allocatable ownership excludes foreign caller association")
    if ((module is not scope and not formal) or "allocatable" not in binding.attributes
            or binding.attributes & (_UNSAFE | {"pointer"})
            or not _Declarations().independent(scope)):
        raise OwnedRecordRegionUnsupported("record region requires independent original allocatable ownership")
    if record:
        if binding.rank != 1 or "target" not in binding.attributes or _typename(binding.dtype) is None:
            raise OwnedRecordRegionUnsupported("record region requires a rank-one allocatable TARGET record owner")
    elif (binding.dtype not in {"real", "integer"} or binding.kind not in {4, 8}
          or not 1 <= binding.rank <= 4 or "target" in binding.attributes):
        raise OwnedRecordRegionUnsupported("record region numeric separation requires independent non-TARGET primitive allocation")


def _raw_capture(analysis, caller, call, numeric):
    """Retain known original inputs before delegated source verification."""
    from compiler.frontend.source_effects import Routine

    capture = DependencyCapture(analysis, components=True)
    routine = analysis.routines.get(caller) or analysis.numerical_helpers.get(caller)
    if (not isinstance(routine, Routine) or _kind(call) != "Call_Stmt" or _kind(call.items[0]) != "Name"
            or type(numeric) is not tuple or not numeric):
        raise CompilationError("record region requires its exact original named CALL and allocations")

    def retain(scope):
        current = scope
        while current is not None:
            capture.retain_scope(current)
            current = current.parent

    capture.retain_routine(routine)
    retain(routine.scope)
    for binding in numeric:
        retain(binding.declaring_scope)
    scope = analysis.source_scope_for(call, routine.scope)
    _validate_resolution(analysis, scope, tuple(walk(call)), _scope_validator=retain)
    for name in analysis._candidates(scope, call.items[0]):
        child = analysis.routines.get(name) or analysis.numerical_helpers.get(name)
        if isinstance(child, Routine):
            capture.retain_routine(child)
            retain(child.scope)
    return capture


def _derive(analysis, caller, call, numeric, *, tick=None, roles=False,
            _capture=None, _defer_publication=False, _with_capture=False):
    capture = _raw_capture(analysis, caller, call, numeric) if _capture is None else _capture
    caller = _original(analysis, caller)
    if (analysis._require_original(caller.qualified) is not caller or _kind(call) != "Call_Stmt"
            or sum(node is call for node in walk(caller.execution)) != 1):
        raise CompilationError("record region requires an exact original caller occurrence")
    if type(numeric) is not tuple or not numeric or len({id(item) for item in numeric}) != len(numeric):
        raise CompilationError("record region requires distinct exact numeric allocation bindings")
    types = _TypeBuilder(analysis)

    def retain_scope(scope):
        current = scope
        while current is not None:
            capture.retain_scope(current)
            _validate_scope(analysis, current)
            current = current.parent

    def retain(binding):
        retain_scope(binding.declaring_scope)
        _original_binding(analysis, binding, types)

    for binding in numeric:
        retain(binding)
        _owned(analysis, binding, types, record=False)
    call_scope = analysis.source_scope_for(call, caller.scope)
    capture.retain_syntax(("call", id(call)), (call,))

    def associate(callee, actuals):
        if callee.source_kind not in {"module", "internal"}:
            raise CompilationError("record region requires an exact original source callee")
        candidates = [(name, actual) for name, actual in zip(callee.arguments, actuals, strict=True)
                      if _typename(callee.scope.bindings[name].dtype) is not None]
        if len(candidates) != 1:
            raise OwnedRecordRegionUnsupported("record region first slice requires exactly one record formal")
        name, actual = candidates[0]
        formal = callee.scope.bindings[name]
        owner = _binding(analysis, call_scope, actual) if _kind(actual) == "Name" else None
        if owner is None:
            raise OwnedRecordRegionUnsupported("record region requires a whole original record allocation actual")
        retain(owner)
        retain(formal)
        _owned(analysis, owner, types, record=True)
        if (formal.rank != 1 or formal.intent != "inout" or "target" not in formal.attributes
                or formal.attributes & (_UNSAFE | {"allocatable", "pointer", "save"})):
            raise OwnedRecordRegionUnsupported("record region requires a rank-one TARGET INOUT dummy")
        left = types.resolve(formal.declaring_scope, _typename(formal.dtype))
        right = types.resolve(owner.declaring_scope, _typename(owner.dtype))
        if left is None or right is None or left.node is not right.node or left.scope is not right.scope:
            raise OwnedRecordRegionUnsupported("record region canonical declaring type mismatch")
        schema = types.schema(left)
        if any(_kind(node) in {"Sequence_Stmt", "Language_Binding_Spec"} for node in walk(left.node)):
            raise OwnedRecordRegionUnsupported("record region excludes sequence or foreign record representation")
        if not schema["fields"] or any(item["shape"] or item["type"] not in {"real", "integer", "logical"}
                                      or item["kind"] not in {4, 8} for item in schema["fields"]):
            raise OwnedRecordRegionUnsupported("record region requires fixed primitive scalar fields only")
        for other, value in zip(callee.arguments, actuals, strict=True):
            binding = callee.scope.bindings[other]
            if binding is formal:
                continue
            if (_kind(value) not in {"Name", "Int_Literal_Constant", "Real_Literal_Constant", "Logical_Literal_Constant"}
                    or binding.attributes & (_UNSAFE | {"pointer", "target"})
                    or "allocatable" in binding.attributes and (_kind(value) != "Name"
                        or "allocatable" not in _binding(analysis, call_scope, value).attributes)
                    or binding.dtype not in {"real", "integer", "logical"}
                    or binding.kind not in {4, 8}
                    or analysis._signature(call_scope, value) != binding.signature()):
                raise OwnedRecordRegionUnsupported("record region other actuals require unchanged primitive source association")
            retain(binding)
        return owner, formal, schema, left

    callee, _actuals, association, _arguments = _resolve_candidates(analysis, call_scope, call, associate)
    owner, formal, schema, original_schema = association
    capture.retain_routine(callee)
    capture.retain_syntax(("type", id(original_schema.node)), (original_schema.node,))
    for binding in callee.scope.bindings.values():
        retain(binding)
    pointers = [binding for binding in callee.scope.bindings.values() if "pointer" in binding.attributes]
    if len(pointers) != 1:
        raise OwnedRecordRegionUnsupported("record region requires exactly one local element pointer")
    pointer, = pointers
    if (pointer.declaring_scope is not callee.scope or pointer.name in callee.arguments or pointer.rank
            or pointer.attributes & (_UNSAFE | {"allocatable", "save", "target"})
            or types.resolve(callee.scope, _typename(pointer.dtype)) is not original_schema):
        raise OwnedRecordRegionUnsupported("record region pointer must be original local unsaved scalar storage")
    originals = tuple(_children(callee.execution))
    groups = [item for item in grouped_original_sequence(originals, {}) if isinstance(item, NativeGroupCandidate)]
    if len(groups) != 1:
        raise OwnedRecordRegionUnsupported("record region requires one exact original root joined team")
    group = _original_group(callee, groups[0].original_nodes)
    body_graph, body_selected = analysis._selected_source(callee.qualified, originals)
    texts = [directive(node) for node in walk(group.original_nodes) if directive(node)]
    if len(texts) != 2 or texts[-1] != "end parallel do":
        raise OwnedRecordRegionUnsupported("record region requires one complete original joined PARALLEL DO")
    opening = re.fullmatch(r"parallel do\s+private\(([^()]*)\)(?:\s+schedule\((?:static|guided)(?:,[1-9][0-9]*)?\))?", texts[0])
    if opening is None:
        raise OwnedRecordRegionUnsupported("record region requires explicit original PRIVATE storage without asynchronous clauses")
    private = {name.strip() for name in opening[1].split(",")}
    loops = [node for node in group.original_nodes if _kind(node) == "Block_Nonlabel_Do_Construct"]
    if len(loops) != 1:
        raise OwnedRecordRegionUnsupported("record region requires one exact original associated counted loop")
    loop, = loops
    body = [node for node in _children(loop) if _kind(node) != "Comment"]
    header, *statements, _end = body
    control = header.items[1]
    if control is None or control.items[0] is not None or control.items[1] is None or any(control.items[2:]):
        raise OwnedRecordRegionUnsupported("record region requires its ordinary original counted iterator")
    iterator = _binding(analysis, callee.scope, control.items[1][0])
    if (iterator is None or iterator.dtype != "integer" or iterator.kind not in {4, 8} or iterator.rank
            or pointer.name not in private or iterator.name not in private):
        raise OwnedRecordRegionUnsupported("record region element pointer and iterator must be explicitly PRIVATE")
    for name in private:
        binding = callee.scope.bindings.get(name)
        if (binding is None or binding.name in callee.arguments
                or binding is not pointer and (binding.dtype not in {"real", "integer", "logical"}
                    or binding.attributes & (_UNSAFE | {"pointer", "target", "allocatable", "save"}))):
            raise OwnedRecordRegionUnsupported("record region PRIVATE list contains unproved original storage")
    first = statements[0] if statements else None
    if _kind(first) != "Pointer_Assignment_Stmt":
        raise OwnedRecordRegionUnsupported("record region association must dominate every original element-pointer use")
    lhs, bounds, rhs = first.items
    indices = _children(rhs.items[1]) if _kind(rhs) == "Part_Ref" else ()
    if (bounds is not None or _kind(lhs) != "Name" or _binding(analysis, callee.scope, lhs) is not pointer
            or _kind(rhs) != "Part_Ref" or _binding(analysis, callee.scope, rhs.items[0]) is not formal
            or len(indices) != 1 or _kind(indices[0]) != "Name"
            or _binding(analysis, callee.scope, indices[0]) is not iterator):
        raise OwnedRecordRegionUnsupported("record region requires exact local pointer to original dummy element")
    field_names = {item["name"] for item in schema["fields"]}
    permitted_names = {id(lhs), id(rhs.items[0])}
    effects, field_nodes = [], []
    dominated = {id(node): node for node in walk(statements[1:])}
    nodes = walk(callee.execution)
    allowance = max(DEFAULT_SOURCE_SCAN_LIMIT, SOURCE_SYNTAX_PASSES * len(nodes))
    for visits, node in enumerate(nodes, 1):
        if visits > allowance:
            raise CompilationError("record region original source traversal budget exhausted")
        if tick is not None:
            tick()
        if _kind(node) == "Assignment_Stmt":
            scope = analysis.source_scope_for(node, callee.scope)
            _validate_resolution(analysis, scope, (), names=("assignment(=)",), _scope_validator=retain_scope)
            if not original_name_absent(analysis, scope, "assignment(=)"):
                raise OwnedRecordRegionUnsupported("record region implicit defined assignment requires independent closure authority")
        kind = _kind(node)
        scope = analysis.source_scope_for(node, callee.scope)
        if kind == "Pointer_Assignment_Stmt" and node is not first:
            raise OwnedRecordRegionUnsupported("record region pointer association cannot be replaced or exported")
        if kind == "Intrinsic_Function_Reference":
            name = str(node.items[0]).lower()
            args = _children(node.items[1])
            if (name in {"size", "lbound", "ubound"} and args and _kind(args[0]) == "Name"
                    and _binding(analysis, scope, args[0]) is formal
                    and not analysis._binding(scope, name) and not analysis._candidates(scope, name)
                    and not analysis._unknown_exports(scope)):
                permitted_names.add(id(args[0]))
        if _kind(node) == "Data_Ref" and _kind(node.items[0]) == "Name" and _binding(analysis, callee.scope, node.items[0]) is pointer:
            if id(node) not in dominated or dominated[id(node)] is not node:
                raise OwnedRecordRegionUnsupported("record region pointer use must be dominated inside its exact original loop")
            if len(node.items) != 2 or _kind(node.items[1]) != "Name" or str(node.items[1]).lower() not in field_names:
                raise OwnedRecordRegionUnsupported("record region pointer use exceeds its exact fixed scalar fields")
            permitted_names.add(id(node.items[0]))
            effects.append(str(node))
            field_nodes.append(node)
    for node in walk(callee.execution):
        if (_kind(node) == "Name" and any(_binding(analysis, callee.scope, node) is item for item in (pointer, formal))
                and id(node) not in permitted_names):
            raise OwnedRecordRegionUnsupported("record region whole object/address use or pointer escape is unproved")
    record = {"version": VERSION, "kind": "original_owned_fixed_record_element_region",
        "caller": caller.qualified, "callee": callee.qualified, "source_inputs": analysis.inputs.identity(),
        "original_call": str(call), "record_owner": owner.root, "record_formal": formal.root,
        "complete_original_body": {"source_graph": body_graph.identity, "selected_node_ids": list(body_selected),
                                   "source": "\n".join(str(node) for node in originals)},
        "declaring_type": schema, "pointer": pointer.root, "original_association": str(first),
        "original_directives": texts, "private_bindings": sorted(private), "original_field_occurrences": effects,
        "disjoint_numeric_allocations": sorted(binding.root for binding in numeric),
        "separation_basis": "original allocatable ownership; fixed record inline fields versus primitive allocation; no storage association",
        "association_only": True, "complete_original_body_and_callees_must_be_scanned": True,
        "original_host_execution_required": True, "payload_access_authorized": False,
        "native_execution_authorized": False, "numerical_completion_authorized": False,
        "entry_definition_authority": False, "allocation_presence_proved": False}
    record = _encoded(record)
    if not _defer_publication:
        _publish_capture(analysis, capture)
    retained = (caller, callee, owner, formal, pointer, first, tuple(group.original_nodes), tuple(field_nodes), originals)
    if _with_capture:
        return record, retained, capture
    if roles:
        return record, retained
    return record


def _publish_capture(analysis, capture):
    analysis.verify_publication()
    analysis.inputs.verify_live()
    capture.receipt().require()


@dataclass(frozen=True, slots=True)
class _RecordIssue:
    family: object
    proof: object
    fields: tuple
    receipt_fields: tuple


def _issued(proof, analysis):
    return (type(proof) is OwnedRecordRegion and type(proof._issuer) is _RecordIssue
            and proof._issuer.family is _ISSUER and proof._issuer.proof is proof
            and proof._analysis is analysis
            and all(getattr(proof, name) is value for name, value in proof._issuer.fields)
            and all(getattr(owner, name) is value for owner, name, value in proof._issuer.receipt_fields))


def _same_roles(left, right):
    return (len(left) == len(right) and all(a is b for a, b in zip(left[:6], right[:6], strict=True))
            and all(len(a) == len(b) and all(x is y for x, y in zip(a, b, strict=True))
                    for a, b in zip(left[6:], right[6:], strict=True)))


@dataclass(frozen=True, slots=True)
class OwnedRecordRegion:
    identity: str
    _analysis: object = field(repr=False)
    _caller: str
    _call: object = field(repr=False)
    _numeric: tuple = field(repr=False)
    _record: str = field(repr=False)
    _issuer: object = field(repr=False)
    _roles: tuple = field(repr=False)
    _dependencies: object = field(default=None, repr=False)
    _context: object = field(default=None, repr=False)
    _arena: object = field(default=None, repr=False)
    _derived: object = field(default=None, repr=False)

    def public(self):
        return {"identity": self.identity, **json.loads(self._record)}

    def validate(self, analysis=None):
        analysis = self._analysis if analysis is None else analysis
        if self._context is not None:
            return _require_region_handle(self, analysis)
        registry = _registry(analysis)
        if (not _issued(self, analysis)
                or getattr(analysis, "_owned_record_regions_owner", None) is not analysis
                or type(self.identity) is not str or registry.get(self.identity) is not self
                or sha256(self._record.encode()).hexdigest() != self.identity):
            raise CompilationError("record region requires its exact registered original issuer")
        attached = tuple((part.name, getattr(self, part.name)) for part in fields(self))
        registry = analysis._owned_record_regions
        record, roles = _derive(analysis, self._caller, self._call, self._numeric, roles=True)
        if (not _same_roles(roles, self._roles) or record != self._record
                or not _issued(self, analysis) or any(getattr(self, name) is not value for name, value in attached)
                or analysis._owned_record_regions is not registry or registry.get(self.identity) is not self
                or analysis._owned_record_regions_owner is not analysis):
            raise CompilationError("record region source or consuming attachment changed")
        self._dependencies.require()
        return self


def prove_owned_record_region(analysis, caller, call, numeric, *, tick=None):
    """Issue exact association evidence; body/storage permissions stay separate."""
    if getattr(analysis, "_source_session", None) is not None:
        return _arena_region(analysis, caller, call, numeric, tick=tick)
    owner = getattr(analysis, "_owned_record_regions_owner", None)
    if owner is not None and owner is not analysis:
        raise CompilationError("record region issuer registry belongs to another original analysis")
    registry = getattr(analysis, "_owned_record_regions", None)
    if registry is None:
        registry = analysis._owned_record_regions = {}
        analysis._owned_record_regions_owner = analysis
    if type(registry) is not dict or any(type(key) is not str or type(value) is not OwnedRecordRegion
                                         for key, value in registry.items()):
        raise CompilationError("record region registry contains foreign source authority")
    if any(not _issued(value, analysis) for value in registry.values()):
        raise CompilationError("record region cached issuer belongs to foreign source authority")
    before = tuple(registry.items())
    states = tuple((proof, _attachments(proof)) for _key, proof in before)
    record, roles, capture = _derive(analysis, caller, call, numeric, tick=tick, roles=True,
                                     _defer_publication=True, _with_capture=True)
    _publish_capture(analysis, capture)
    _require_issuing_registry(analysis, registry, before, states)
    identity = sha256(record.encode()).hexdigest()
    if identity in registry:
        proof = registry[identity]
        if (not _issued(proof, analysis) or proof.identity != identity or proof._record != record
                or proof._caller != caller or proof._call is not call or type(proof._numeric) is not tuple
                or len(proof._numeric) != len(numeric)
                or any(a is not b for a, b in zip(proof._numeric, numeric, strict=True))
                or not _same_roles(proof._roles, roles)):
            raise CompilationError("record region cached issuer or consuming original query is foreign")
        return proof
    proof = OwnedRecordRegion(identity, analysis, caller, call, numeric, record, None, roles, capture.receipt())
    _issue(proof)
    registry[identity] = proof
    return proof


def _registry(analysis):
    registry = getattr(analysis, "_owned_record_regions", None)
    if (type(registry) is not dict or any(type(key) is not str or type(value) is not OwnedRecordRegion
                                        for key, value in registry.items())):
        raise CompilationError("record region requires its exact registered original issuer")
    return registry


class RecordRegionCheckpoint:
    """Private consuming baseline, never authority or a serialized proof."""
    def __init__(self, analysis, proof, extra_routines=(), *, calls=(), terminating_calls=()):
        registry = _registry(analysis)
        if (not _issued(proof, analysis)
                or getattr(analysis, "_owned_record_regions_owner", None) is not analysis
                or type(proof.identity) is not str or registry.get(proof.identity) is not proof):
            raise CompilationError("record region checkpoint needs exact registered association")
        self.analysis, self.proof = analysis, proof
        self.registry = analysis._owned_record_regions
        self.registry_entries = tuple(self.registry.items())
        self.proof_fields = _attachments(proof)
        self.calls = tuple((token, _attachments(token), analysis._original_call_associations)
                           for token in calls)
        self.terminating_calls = tuple((token, _attachments(token), analysis._descriptor_terminating_calls)
                                       for token in terminating_calls)
        capture = DependencyCapture(analysis, components=True, inherited=proof._dependencies)
        for routine in extra_routines:
            capture.retain_routine(routine)
            for binding in routine.scope.bindings.values():
                scope = binding.declaring_scope
                while scope is not None:
                    capture.retain_scope(scope)
                    scope = scope.parent
        self.dependencies = capture.receipt()

    def require(self):
        analysis, proof = self.analysis, self.proof
        current = _registry(analysis)
        if (any(not _attached(token, saved) or analysis._original_call_associations is not registry
                    or registry.get((id(analysis), id(token._original))) is not token
                    for token, saved, registry in self.calls)
                or any(not _attached(token, saved) or analysis._descriptor_terminating_calls is not registry
                    or registry.get((token._procedure, id(token._call))) is not token
                    for token, saved, registry in self.terminating_calls)
                or not _issued(proof, analysis) or current is not self.registry
                or analysis._owned_record_regions_owner is not analysis
                or len(self.registry) != len(self.registry_entries)
                or any(self.registry.get(key) is not value for key, value in self.registry_entries)
                or not _attached(proof, self.proof_fields)):
            raise CompilationError("record region consumed source, namespace or issuer changed")
        self.dependencies.require()


def _issue(proof):
    attached = tuple((part.name, getattr(proof, part.name)) for part in fields(proof) if part.name != "_issuer")
    owners = (proof._dependencies,)
    if type(proof._dependencies) is _PrivateDependencies:
        owners += (proof._dependencies.ownership,)
    receipt = tuple((owner, part.name, getattr(owner, part.name)) for owner in owners for part in fields(owner))
    object.__setattr__(proof, "_issuer", _RecordIssue(_ISSUER, proof, attached, receipt))


def _require_region_attachments(proof, *, registry=None):
    """Private callback-free fence after the publisher's final live-file check."""
    analysis, context = proof._analysis, proof._context
    if (not _issued(proof, analysis)
            or getattr(analysis, "_source_authority_context", None) is not context
            or getattr(analysis, "_source_session", None) is not context.session):
        raise CompilationError("record region retained authority changed")
    current_registry = _registry(analysis)
    if (registry is not None and current_registry is not registry
            or analysis._owned_record_regions_owner is not analysis):
        raise CompilationError("record region consuming attachment changed")
    active = getattr(analysis, "_source_derivation_arena", None)
    current, accepted, handle = active, False, None
    key = "original_owned_record_region", id(analysis), proof._derived.query
    while current is not None:
        if current is proof._arena:
            accepted = True
        if handle is None:
            handle = current._handles.get(key)
        current = current._parent
    if (not accepted or handle is not proof or active._context is not context
            or active._records.get((proof._derived.family, proof._derived.query)) is not proof._derived
            or current_registry.get(proof.identity) is not proof):
        raise CompilationError("record region consuming attachment changed")
    proof._dependencies.require()


def _require_region_handle(proof, analysis, *, tick=None):
    if not _issued(proof, analysis) or proof._context is None:
        raise CompilationError("record region requires its exact registered original issuer")
    registry = _registry(analysis)
    if (getattr(analysis, "_owned_record_regions_owner", None) is not analysis
            or registry.get(proof.identity) is not proof):
        raise CompilationError("record region requires its exact registered original issuer")
    proof._context.require(analysis)
    arena = proof._context.derivation_arena(analysis)
    if not arena.accepts(proof._arena):
        raise CompilationError("record region belongs to a stale generation arena")
    arena.require(proof._context, proof._derived)
    arena.require_handle(("original_owned_record_region", id(analysis), proof._derived.query), proof)
    if tick is not None:
        tick()
    analysis.verify_publication()
    analysis.inputs.verify_live()
    _require_region_attachments(proof, registry=registry)
    return proof


def _role_recipe(roles):
    return [*(id(value) for value in roles[:6]), *([id(value) for value in group] for group in roles[6:])]


def _resolve_roles(context, recipe):
    from compiler.frontend.source_effects import Binding, Routine

    session = context.session
    objects = tuple(session.objects.get(identity) for identity in recipe[:5])
    if any(not isinstance(value, expected) or id(value) != identity
           for value, expected, identity in zip(objects, (Routine, Routine, Binding, Binding, Binding), recipe[:5], strict=True)):
        raise CompilationError("record arena role is not exact original session storage")
    return (*objects, session.original_node(recipe[5]),
            *(tuple(session.original_node(identity) for identity in group) for group in recipe[6:]))


def _region_arena_derivation(analysis, caller, call, numeric, tick, capture):
    record, roles, capture = _derive(analysis, caller, call, numeric, tick=tick, _capture=capture,
                                    _defer_publication=True, _with_capture=True)
    encoded = _encoded({"public_record": record, "public_identity": sha256(record.encode()).hexdigest(),
                        "roles": _role_recipe(roles), "dependencies": capture.recipe()})
    _publish_capture(analysis, capture)
    return SourcePayload(encoded, (id(call),))


def _arena_region(analysis, caller, call, numeric, *, tick=None):
    owner = getattr(analysis, "_owned_record_regions_owner", None)
    if owner is not None and owner is not analysis:
        raise CompilationError("record region issuer registry belongs to another original analysis")
    if getattr(analysis, "_owned_record_regions", None) is None:
        analysis._owned_record_regions = {}
        analysis._owned_record_regions_owner = analysis
    registry = _registry(analysis)
    if owner is None and not registry:
        analysis._owned_record_regions_owner = analysis
    if (analysis._owned_record_regions_owner is not analysis
            or any(not _issued(proof, analysis) for proof in registry.values())):
        raise CompilationError("record region cached issuer belongs to foreign source authority")
    before = tuple(registry.items())
    states = tuple((proof, _attachments(proof)) for _key, proof in before)
    if type(numeric) is not tuple or not numeric:
        raise CompilationError("record region requires distinct exact numeric allocation bindings")
    query = caller, id(call), tuple(id(value) for value in numeric)
    matching = tuple(proof for proof in registry.values()
                     if proof._caller == caller and proof._call is call
                     and len(proof._numeric) == len(numeric)
                     and all(a is b for a, b in zip(proof._numeric, numeric, strict=True)))
    active_context = getattr(analysis, "_source_authority_context", None)
    existing = next((proof for proof in matching if proof._context is active_context),
                    matching[-1] if matching else None)
    ownership = (existing._dependencies.ownership
                 if existing is not None and existing._context is not None
                 and type(existing._dependencies) is _PrivateDependencies
                 and existing._context is getattr(analysis, "_source_authority_context", None)
                 else _SourceOwnership.capture(analysis))
    capture = (None if existing is not None and existing._context is not None
               and existing._context is active_context else _raw_capture(analysis, caller, call, numeric))
    context = analysis.source_authority_context()
    context.require(analysis)
    if (context.session.original_node(id(call)) is not call
            or any(not context.session.contains(value) for value in numeric)):
        raise CompilationError("record region requires its exact original caller and allocation roles")
    arena = context.derivation_arena(analysis)
    if existing is not None and existing._context is context:
        result = _require_region_handle(existing, analysis, tick=tick)
        ownership.require()
        _require_issuing_registry(analysis, registry, before, states)
        _require_region_attachments(existing, registry=registry)
        return result
    derived = arena.derive(_REGION_FAMILY, query, (caller, call, numeric, tick, capture), analysis)
    context.require(analysis)
    payload = json.loads(derived.record)
    ownership.require()
    dependencies = _resolve_dependencies(context, payload["dependencies"], ownership)
    roles = _resolve_roles(context, payload["roles"])
    _require_issuing_registry(analysis, registry, before, states)
    previous = registry.get(payload["public_identity"])
    if previous is not None and previous is not existing:
        raise CompilationError("record region cached issuer or consuming original query is foreign")
    from compiler.frontend.storage_mutation_effects import source_registry_budget
    if existing is None and len(registry) >= source_registry_budget(analysis):
        raise CompilationError("record region source-sized token registry budget exhausted")
    proof = OwnedRecordRegion(payload["public_identity"], analysis, caller, call, numeric,
                              payload["public_record"], None, roles, dependencies, context, arena, derived)
    _issue(proof)
    dependencies.require()
    arena.register_handle(("original_owned_record_region", id(analysis), query), proof)
    registry[proof.identity] = proof
    return _require_region_handle(proof, analysis)


_REGION_FAMILY = _register_family("original_owned_record_region", 1, _region_arena_derivation, legacy_import=True)


def _require_issuing_registry(analysis, registry, before, states):
    if (_registry(analysis) is not registry or analysis._owned_record_regions_owner is not analysis
            or len(registry) != len(before) or any(registry.get(key) is not proof for key, proof in before)
            or any(not _issued(proof, analysis) or not _attached(proof, state) for proof, state in states)):
        raise CompilationError("record region registry or retained authority changed during derivation")
