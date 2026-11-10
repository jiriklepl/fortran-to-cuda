"""Returned-domain authority from the same exhaustive checked source inventory.

The draft permits only undefined prospective registrations. It cannot emit or
authorize native access. Publication requires a sealed proof covering every
original execution node with a checked route or a reached domain retirement.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256

from fparser.two.utils import walk

from compiler.frontend.call_bindings import resolve_source_call
from compiler.frontend.source_effects import _kind, _part
from compiler.frontend.source_functions import _validate_scope
from compiler.frontend.storage_mutation_effects import (
    _preserves_record,
    prove_storage_mutation_effects,
    source_registry_budget,
)
from compiler.ir import CompilationError
from compiler.scopes.segments import statement_span


class _ReturnIssuer:
    __slots__ = ()

    def public(self):
        return {"kind": "checked conditional returned-domain issuer", "version": 1}


_ISSUER = _ReturnIssuer()


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value):
    return sha256(_encoded(value).encode()).hexdigest()


def _same_originals(left, right):
    return len(left) == len(right) and all(a is b for a, b in zip(left, right, strict=True))


def _scalar_component_source(analysis, scope, mapping):
    """Authenticate a scalar view through its original object and field declarations."""
    from compiler.frontend.component_bindings import _typename, component_access
    from compiler.frontend.source_objects import _TypeBuilder

    binding, formal = mapping.binding, mapping.formal_binding
    if (binding is None or binding.rank or formal.rank or formal.intent != "in"
            or binding.dtype not in {"real", "integer", "logical"}
            or binding.attributes & {"pointer", "allocatable", "optional", "volatile", "asynchronous"}
            or formal.attributes & {"pointer", "allocatable", "optional", "volatile", "asynchronous", "value"}
            or _kind(mapping.actual) != "Data_Ref"):
        return None
    access = component_access(analysis, scope, mapping.actual)
    if (access is None or access.indices or access.binding.rank
            or getattr(binding, "component_object", None) is not access.root_binding
            or getattr(binding, "component_selector", None) is not mapping.actual
            or binding.public() != access.binding.public()):
        raise CompilationError("returned-domain scalar component lost its original source association")
    authority = _TypeBuilder(analysis)
    schema = authority.binding(access.root_binding)
    originals = [access.root_binding, mapping.actual]
    for index, name in enumerate(access.path):
        if index >= analysis.depth_limit:
            raise CompilationError("returned-domain scalar component declaration depth exhausted")
        declaration = schema.members.get(name)
        if declaration is None:
            raise CompilationError("returned-domain scalar component declaration changed")
        dtype, attributes, entity = declaration
        originals.extend((schema.node, dtype, *attributes, entity))
        if index + 1 < len(access.path):
            nested = _typename(dtype)
            schema = authority.resolve(schema.scope, nested) if nested else None
            if schema is None:
                raise CompilationError("returned-domain scalar component nested type changed")
    return tuple(originals)


@dataclass(frozen=True)
class _ScalarComponentActual:
    mapping: object
    binding: object
    originals: tuple
    descriptor: str

    def matches(self, analysis, scope, retained, current):
        # A fresh view may represent the same original component. Replacing
        # the retained mapping or its projection never creates that authority.
        if (retained is not self.mapping or retained.binding is not self.binding
                or _encoded(retained.binding.public()) != self.descriptor):
            return False
        return (_same_originals(self.originals, _scalar_component_source(analysis, scope, retained) or ())
                and _same_originals(self.originals, _scalar_component_source(analysis, scope, current) or ()))


@dataclass(frozen=True)
class CheckedLexicalCall:
    """The original argumentless lexical CALL, with no synthetic association."""
    parent: object
    callee: object
    node: object
    resolved: object
    record: object
    patch: object
    _issuer: object = field(repr=False, compare=False)

    def validate(self, member):
        from compiler.frontend.immutable_graph import original_type
        from compiler.scopes.lexical import LexicalOwner

        child, analysis = self.callee, member.builder.analysis
        span = statement_span(self.node)
        if (self._issuer is not _ISSUER or member is not self.parent
                or original_type(self) is not CheckedLexicalCall
                or member._issued_lexical_calls.get(id(self.node)) is not self
                or not any(receipt is self for receipt in member.source_lexical_calls)
                or child.parent is not member or child.owner is not member.owner
                or original_type(child) is not LexicalOwner
                or member.owner.members.get(child.routine.qualified) is not child
                or analysis.routines.get(child.routine.qualified) is not child.routine
                or child.routine.scope.parent is not member.routine.scope
                or child.routine.source_kind != "internal" or child.routine.arguments
                or child.resources is not member.resources
                or child.canonical_bindings is not member.canonical_bindings
                or child.resource_slot_namespace is not member.resource_slot_namespace
                or not any(row is self.record for row in member.internal_calls)
                or member.builder.resolved_calls.get((member.routine.qualified, span)) != self.record
                or not any(first == span[0] and last == span[1] and patch is self.patch
                           for first, last, patch in member.patches)):
            raise CompilationError("lexical CALL lost its exact original source and namespace receipt")
        scope = analysis.source_scope_for(self.node, member.routine.scope)
        current = resolve_source_call(analysis, scope, self.node)
        graph, selected = analysis._selected_source(member.routine.qualified, (self.node,))
        expected = {**current.public(), "caller": member.routine.qualified,
                    "source": str(member.routine.scope.path), "first_line": span[0], "last_line": span[1],
                    "execution": "original internal procedure; host-associated owner control"}
        if (scope is not member.routine.scope or graph is not member.graph or not selected
                or current.procedure != child.routine.qualified or current.actuals
                or current.public() != self.resolved.public() or self.record != expected):
            raise CompilationError("lexical CALL changed its original source selection or association")
        return self

    def public(self):
        self.validate(self.parent)
        graph, selected = self.parent.builder.analysis._selected_source(
            self.parent.routine.qualified, (self.node,))
        return {"source_call": dict(self.record), "caller_graph": graph.identity,
                "selected_node_ids": list(selected)}


def retain_lexical_call(member, node, child, resolved, record, patch):
    receipt = CheckedLexicalCall(member, child, node, resolved, record, patch, _ISSUER)
    if id(node) in member._issued_lexical_calls:
        raise CompilationError("lexical CALL duplicates its original source receipt")
    member.source_lexical_calls.append(receipt)
    member._issued_lexical_calls[id(node)] = receipt
    return receipt.validate(member)


@dataclass(frozen=True)
class CheckedBorrowedCall:
    node: object
    callee: object
    resolved: object
    marker: str
    source: str
    handle_bindings: tuple
    scalar_components: tuple
    _issuer: object = field(repr=False, compare=False)
    scalar_element_coordinates: object = field(default=None, repr=False, compare=False)

    def validate(self, member):
        first, last = statement_span(self.node)
        if (self._issuer is not _ISSUER or member.owner.members.get(self.callee.routine.qualified) is not self.callee
                or not any(a == first and b == last and text is self.source for a, b, text in member.patches)
                or getattr(member, "external_control_placeholders", {}).get(self.marker) != (self.callee, member.context)):
            raise CompilationError("returned-domain call lost its exact checked control forwarding")
        from compiler.scopes.specification_entry import call_record
        call_record(member, self)
        analysis = member.builder.analysis
        scope = analysis.source_scope_for(self.node, member.routine.scope)
        current = resolve_source_call(analysis, scope, self.node)
        elements = tuple(item.scalar_element for item in current.mappings if item.scalar_element is not None)
        if elements:
            from compiler.frontend.native_call_coordinates import NativeCallCoordinatesProof

            proof = self.scalar_element_coordinates
            if type(proof) is not NativeCallCoordinatesProof:
                raise CompilationError("returned-domain scalar element requires its registered original coordinates")
            proof.validate(analysis, member.routine.qualified, self.node)
            authorized = proof.scalar_elements
            if (len(elements) != len(authorized) or any(
                    current.actual is not retained.actual or current.binding is not retained.binding
                    or current.public() != retained.public()
                    for current, retained in zip(elements, authorized, strict=True))):
                raise CompilationError("returned-domain scalar element changed its original association")
        elif self.scalar_element_coordinates is not None:
            raise CompilationError("returned-domain scalar element coordinates lack their original association")
        components = dict(self.scalar_components)
        if (current.public() != self.resolved.public() or len(current.mappings) != len(self.resolved.mappings)
                or any(a.actual is not b.actual or a.formal_binding is not b.formal_binding
                       or (not components[index].matches(analysis, scope, b, a) if index in components
                           else a.binding is not b.binding)
                       for index, (a, b) in enumerate(zip(current.mappings, self.resolved.mappings, strict=True)))):
            raise CompilationError("returned-domain call changed its original actual association")
        actuals = {"argument::" + item.formal: item.binding for item in current.mappings if item.formal_binding.rank}
        bindings = dict(self.handle_bindings)
        for root, (binding, slot) in self.callee.resources.items():
            expected = member.canonical(actuals.get(root, self.callee.canonical(binding))).root
            number = bindings.get(slot)
            forwarded = [value for value, local in member.resources.values() if local == number]
            if (number is None or len(forwarded) != 1
                    or member.canonical(forwarded[0]).root != expected):
                raise CompilationError("returned-domain call changed its canonical allocation forwarding")
        return self


def retain_call(member, node, callee, resolved, marker, source, handle_bindings, *, source_projection=False,
                scalar_element_coordinates=None):
    returned_domain = getattr(member.owner.builder, "_conditional_return_draft", None) is not None
    if returned_domain or source_projection:
        analysis = member.builder.analysis
        scope = analysis.source_scope_for(node, member.routine.scope)
        components = []
        for index, mapping in enumerate(resolved.mappings):
            originals = _scalar_component_source(analysis, scope, mapping)
            if originals is not None:
                components.append((index, _ScalarComponentActual(
                    mapping, mapping.binding, originals, _encoded(mapping.binding.public()))))
        receipt = CheckedBorrowedCall(node, callee, resolved, marker, source,
                                      tuple(handle_bindings.items()), tuple(components), _ISSUER,
                                      scalar_element_coordinates)
        if returned_domain:
            member.return_calls.append(receipt)
        return receipt
    return None


def _entry_exit(member):
    """Only lifecycle outside the reached execution inventory is checked here."""
    routine, analysis = member.routine, member.builder.analysis
    from compiler.scopes.specification_entry import projection
    specification = projection(member)
    _validate_scope(analysis, routine.scope)
    spec = _part(routine.scope.node, "Specification_Part")
    aliases = {"Common_Stmt", "Equivalence_Stmt", "Namelist_Stmt", "Cray_Pointer_Stmt",
               "Procedure_Declaration_Stmt", "Proc_Declaration_Stmt", "Bind_Stmt"}
    if any(_kind(node) in aliases or (_kind(node) == "Comment" and str(node).lstrip().startswith("!$"))
           for node in walk(spec)):
        raise CompilationError("conditional return requires original unaliased entry declarations")
    opaque = set(getattr(member, "opaque_saved_storage", ()))
    for binding in routine.scope.bindings.values():
        if binding.root in opaque:
            continue  # Existing exact qualified opaque-state issuer remains mandatory.
        if (binding.dtype not in {"integer", "real", "logical"}
                or binding.attributes & {"pointer", "target", "volatile", "asynchronous", "value"}):
            raise CompilationError("conditional return entry/finalization requires separate source authority: " + binding.root)
        if binding.name in routine.arguments and binding.intent == "out" and "allocatable" in binding.attributes:
            raise CompilationError("conditional return cannot cross allocatable OUT entry deallocation")
        for bound in (*binding.shape_nodes, *binding.lower_bound_nodes):
            if specification is not None:
                continue  # Separate source proof; every caller must publish before entry.
            for node in walk(bound):
                kind = _kind(node)
                if kind in {"Data_Ref", "Function_Reference", "Structure_Constructor"}:
                    raise CompilationError("conditional return specification evaluation needs separate authority")
                if kind == "Part_Ref":
                    raise CompilationError("conditional return specification cannot read array payload or source functions")
                if kind == "Name":
                    value = analysis._binding(routine.scope, node)
                    if value is not None and value.rank:
                        raise CompilationError("conditional return specification cannot read array payload")
        if (member is not member.owner and binding.name not in routine.arguments and binding.rank
                and "save" not in binding.attributes and binding.root in member.resources):
            raise CompilationError("conditional return cannot retain child-local storage beyond its original exit")
    return {"procedure": routine.qualified, "original_body_sha256": sha256(str(routine.scope.node).encode()).hexdigest(),
            "specification": "original numeric scalar/descriptor declarations; no hidden calls, aliases or finalization",
            **({"specification_reads": specification.public()} if specification is not None else {}),
            "exit": "no retained child-local allocation; original RETURN keeps caller controls",
            "opaque_resources": sorted(opaque), "definition_authority": False}


def check_entry(member):
    # At admission this checks entry semantics before any caller association.
    # It is repeated after resource discovery to check implicit-exit lifetimes.
    if getattr(member.owner.builder, "_conditional_return_draft", None) is not None:
        _entry_exit(member)


class ConditionalReturnDraft:
    def __init__(self, owner, issuer, *, selected_root=None):
        self.owner, self._issuer = owner, issuer
        self.selected_root = selected_root
        self._selected_root_receipt = selected_root
        self._consumers = []
        self._native_preservations = {}
        self._empty_native_calls = {}

    def selection_identity(self):
        self.validate()
        return (self.selected_root.identity if self.selected_root is not None else
                self.owner.native_ancestor_selection.identity)

    def consumer_key(self, builder):
        self.validate()
        # GenerationTransaction rebinds the owning builder while preserving
        # opaque authority objects. An integer id(builder) is not a portable
        # key across that legitimate adoption.
        current = next((row for row in self._consumers if row[0] is builder), None)
        if current is None:
            current = builder, object()
            self._consumers.append(current)
        return current[1]

    def validate(self):
        owner = self.owner
        if (self._issuer is not _ISSUER or getattr(owner.builder, "_conditional_return_draft", None) is not self
                or owner.parent is not None or owner.builder.config.scope_execution != "graph"):
            raise CompilationError("prospective returned-domain capture requires its original checked graph")
        if self.selected_root is not None:
            from compiler.scopes.hierarchical_regions import SelectedRoot
            if (type(self.selected_root) is not SelectedRoot
                    or self.selected_root is not self._selected_root_receipt):
                raise CompilationError("selected returned-domain capture requires its original root selection")
            self.selected_root.validate(owner)
        else:
            if (getattr(owner, "native_ancestor_selection", None) is None
                    or owner.native_ancestor_selection.public()["version"] != 1):
                raise CompilationError("prospective returned-domain capture requires its original checked graph")
            owner.native_ancestor_selection.validate(owner.builder.analysis)
        return self

    def capture(self, binding, fact):
        self.validate()
        original = self.owner.builder.visible_binding(self.owner.routine, binding.root)
        lifetimes = {"invocation", "selected_window"} if self.selected_root is not None else {"selected_window"}
        if original is not binding or fact.get("lifetime", "invocation") not in lifetimes:
            raise CompilationError("prospective returned-domain capture requires its exact original allocation")
        return {**{key: value for key, value in fact.items()
                   if key not in {"sections", "source_sections", "requires_registered_handle"}},
                **({"lifetime": "selected_window"} if self.selected_root is not None else {}),
                "initialized": "none"}


def prepare(owner):
    selection = getattr(owner, "native_ancestor_selection", None)
    if selection is not None and selection.public()["version"] == 1:
        owner.builder._conditional_return_draft = ConditionalReturnDraft(owner, _ISSUER)


def prepare_selected(owner, selected_root):
    """The same route proof, scoped to one original contiguous root region."""
    if getattr(owner.builder, "_conditional_return_draft", None) is not None:
        raise CompilationError("selected returned-domain owner already has a lifetime draft")
    owner.builder._conditional_return_draft = ConditionalReturnDraft(owner, _ISSUER, selected_root=selected_root)
    owner.builder._conditional_return_draft.validate()



@dataclass(frozen=True)
class SameFamilyNativeCallLifecycle:
    """Retirement route, conditional on the unchanged first-admission rely."""
    identity: str
    _member: object = field(repr=False, compare=False)
    _draft: object = field(repr=False, compare=False)
    _view: object = field(repr=False, compare=False)
    _originals: tuple = field(repr=False, compare=False)
    _facts: str = field(repr=False, compare=False)
    _entry: str = field(repr=False, compare=False)
    _record: str = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)

    def public(self):
        return {"identity": self.identity, **json.loads(self._record)}

    def validate(self, analysis=None, procedure=None):
        member, draft = self._member, self._draft.validate()
        analysis = member.builder.analysis if analysis is None else analysis
        procedure = member.routine.qualified if procedure is None else procedure
        if (self._issuer is not _ISSUER or analysis is not member.builder.analysis
                or procedure != member.routine.qualified or draft.owner is not member.owner
                or getattr(member.owner.builder,"provider_only_lifetime",None) is not None
                or draft._native_preservations.get(self.identity) is not self
                or _encoded(member.owner.builder.facts) != self._facts
                or _encoded(_entry_exit(member)) != self._entry):
            raise CompilationError("same-family CALL changed its original admission/capture/entry contract")
        self._view.validate(analysis, procedure, self._originals)
        if not self._view.public()["available"]:
            raise CompilationError("same-family CALL has an unproved descriptor/publication/completion effect")
        record = _native_call_preservation_record(member, self._originals, self._view)
        if _encoded(record) != self._record or _hash(record) != self.identity:
            raise CompilationError("same-family CALL lost its exact selected-window preservation route")
        # Current/future numerical registrations retain their own checked facts;
        # this route grants none of their lifetime/layout/alias/read authority.
        if not set(record["writable_actual_families"]) <= set(member.owner.resources):
            raise CompilationError("same-family CALL writable actual is outside the inherited managed family")
        from compiler.scopes.storage_epoch import resource_inventory

        managed, _reservations, _opaque = resource_inventory(member.owner)
        for root, (binding, _slot) in member.owner.resources.items():
            if root in managed:
                fact = member.owner.builder.capture_fact(binding)
                draft.capture(binding, fact)
        return self



def _native_call_writable_families(member, originals, view):
    """Original lexical associations to existing captures, never field geometry."""
    from compiler.frontend.resource_reachability import _host_arrange, _original
    graph = view._graph
    if len(graph.host_instances)<2 or graph.host_instances[1].call.original is not originals[0]:
        raise CompilationError("same-family CALL lost its original outer lexical association")
    edge = graph.host_instances[1].call
    callee = _original(member.builder.analysis, graph.host_instances[1].procedure)
    families=set()
    for name,actual in _host_arrange(edge,callee):
        declaration=callee.scope.bindings.get(name)
        if declaration is None or declaration.intent=="in" or actual is None:
            continue  # Fixed procedure actuals have their own exact closure; absence is not storage.
        if declaration.dtype not in {"real","integer","logical"} or _kind(actual) not in {"Name","Part_Ref"}:
            raise CompilationError("same-family CALL writable actual requires an existing primitive capture")
        base=actual.items[0] if _kind(actual)=="Part_Ref" else actual
        binding=member.builder.analysis._binding(edge.scope,base)
        if binding is None:
            raise CompilationError("same-family CALL writable original binding is unavailable")
        binding=member.canonical(binding)
        fact=member.owner.builder.capture_fact(binding)
        member.owner.builder._conditional_return_draft.capture(binding,fact)
        families.add(binding.root)
    return sorted(families)


def _native_call_preservation_record(member, originals, view):
    draft = member.owner.builder._conditional_return_draft.validate()
    graph, selected = member.builder.analysis._selected_source(member.routine.qualified, originals)
    first, last = statement_span(originals[0])
    return {"version": 2, "kind": "same_family_native_call_retirement", "procedure": member.routine.qualified,
            "source_graph": graph.identity, "selected_node_ids": list(selected),
            "source_span": [first,last], "original_source": member.original(originals), "original_source_sha256": sha256(member.original(originals).encode()).hexdigest(),
            "original_call": str(originals[0]), "completion_identity": view.identity,
            "native_graph_identity": view._graph.identity,
            "selection_identity": draft.selection_identity(),
            "capture_facts_identity": _hash(member.owner.builder.facts),
            "entry_exit_identity": _hash(_entry_exit(member)),
            "writable_actual_families": _native_call_writable_families(member, originals, view),
            "entry_rely": "same conditional allocation-family access exclusion as first checked activation",
            "retirement": "full owning domain before untouched original CALL; next activation remains fresh NONE",
            "completion_scope": "normal return only", "no_imports_or_publication": True,
            "initial_quiescence_authorized": False, "payload_access_authorized": False,
            "stable_storage_authorized": False, "entry_definition_authority": False,
            "alias_authority": False, "motion_authorized": False, "cross_cut_replica_retention_authorized": False}


def _preserved_native_call(member, originals):
    # A provider-only close may keep its domain alive; this role requires the
    # existing resident close to retire the complete owning allocation family.
    if getattr(member.owner.builder,"provider_only_lifetime",None) is not None:
        return None
    if (len(originals) != 1 or _kind(originals[0]) != "Call_Stmt"
            or getattr(getattr(originals[0],"item",None),"label",None) is not None):
        return None
    from compiler.frontend.resource_reachability import prove_native_effect_graph
    native = prove_native_effect_graph(member.builder.analysis, member.routine.qualified, originals, project_payload=False)
    view = native.same_family_call_view()
    if not view.public()["available"]:
        return None
    draft = member.owner.builder._conditional_return_draft.validate()
    record = _native_call_preservation_record(member, originals, view)
    identity = _hash(record)
    cached = draft._native_preservations.get(identity)
    if cached is not None:
        return cached  # Prospective until exhaustive returned-owner coverage validates every managed family.
    if len(draft._native_preservations) >= source_registry_budget(member.builder.analysis):
        raise CompilationError("same-family retirement route budget exhausted")
    proof = SameFamilyNativeCallLifecycle(identity,member,draft,view,originals,
        _encoded(member.owner.builder.facts),_encoded(_entry_exit(member)),_encoded(record),_ISSUER)
    draft._native_preservations[identity] = proof
    return proof  # The existing exhaustive coverage/patch consumer is the admission gate.


def boundary_lifecycle(member, originals, *, diagnostic=None):
    """Classify a cut without granting permission to execute its payload."""
    if getattr(member.owner.builder, "_conditional_return_draft", None) is None:
        return None, None
    analysis, procedure = member.builder.analysis, member.routine.qualified
    completion, team = None, None
    try:
        preserved = _preserved_native_call(member, originals)
        if preserved is not None:
            analysis.verify_publication()
            if diagnostic is not None:
                diagnostic.update(classification="same_family_native_call_preservation", preservation_identity=preserved.identity)
            return preserved, None
    except CompilationError as error:
        if diagnostic is not None:
            diagnostic["same_family_preservation_decline"] = str(error)
    try:
        graph, selected = analysis._selected_source(procedure, originals)
        from compiler.frontend.native_group_structure import NativeGroupCandidate, grouped_original_sequence
        grouped = grouped_original_sequence(originals, {})
        joined = (len(grouped) == 1 and isinstance(grouped[0], NativeGroupCandidate)
                  and len(grouped[0].original_nodes) == len(originals)
                  and all(left is right for left, right in zip(grouped[0].original_nodes, originals, strict=True)))
        if joined:
            from compiler.frontend.descriptor_teams import prove_descriptor_team
            from compiler.frontend.native_completion import prove_graph_joined_completion
            from compiler.frontend.resource_reachability import prove_native_effect_graph
            # All three issuers retain the same complete original group. The
            # native graph has no storage demand and cannot lend payload or
            # alias authority to this reached retirement.
            native = prove_native_effect_graph(analysis, procedure, originals)
            completion = prove_graph_joined_completion(analysis, procedure, originals, native)
            team = prove_descriptor_team(analysis, procedure, originals)
            if completion.selected_node_ids != selected or team.selected_node_ids != selected:
                raise CompilationError("joined boundary changed its exact original team selection")
        proof = prove_storage_mutation_effects(analysis, procedure, selection=originals, joined_storage=team,
            native_graph=native if joined and completion._ordinary_join is not None else None)
        from compiler.frontend.call_bindings import _original_call_control
        if not joined and _original_call_control(originals) is not None and not proof._record["complete"]:
            from compiler.frontend.native_completion import native_call_storage_structures, prove_native_call_retirement
            from compiler.frontend.resource_reachability import prove_native_effect_graph

            try:
                native = prove_native_effect_graph(analysis, procedure, originals)
                structures = native_call_storage_structures(analysis, native)
                if structures:
                    candidate = prove_storage_mutation_effects(analysis, procedure, selection=originals, joined_storage=structures,
                        native_graph=native if any(getattr(team, "_native_graph", None) is native
                                                   for team in structures) else None)
                    completed = prove_native_call_retirement(analysis, procedure, originals, native, candidate)
                    proof, completion = candidate, completed
            except CompilationError as error:
                # The optional composition cannot erase an already classified
                # original escape/frontier or replace it with a wrapper error.
                if diagnostic is not None:
                    diagnostic["native_call_retirement_decline"] = str(error)
    except CompilationError as error:
        if diagnostic is not None:
            diagnostic.update(classification="unavailable_original_storage_projection", reason=str(error))
        return None, None
    record = proof._record
    if diagnostic is not None:
        diagnostic.update(storage_projection_identity=proof.identity,
                          classification="proved_lifecycle" if record["complete"] and not record["escapes"]
                          else "unproved_lifetime_or_escape")
        frontiers = []
        for closure in record["closure"]:
            for item in (*closure["frontiers"], *closure["escapes"]):
                original = item["original_source"]
                frontiers.append({"procedure": closure["procedure"], "kind": item["kind"],
                    "resource": item["resource"], "source_sha256": sha256(original.encode()).hexdigest(),
                    "source_excerpt": original[:1024], "excerpt_truncated": len(original) > 1024})
                break
            if frontiers:
                break
        if not frontiers:
            for item in (*record["frontiers"], *record["escapes"])[:1]:
                original = item["original_source"]
                frontiers.append({"procedure": member.routine.qualified, "kind": item["kind"],
                    "resource": item["resource"], "source_sha256": sha256(original.encode()).hexdigest(),
                    "source_excerpt": original[:1024], "excerpt_truncated": len(original) > 1024})
        diagnostic["first_source_frontier"] = frontiers[0] if frontiers else None
    if record["complete"] and not record["escapes"]:
        if completion is not None:
            completion.validate(analysis, procedure, selected)
            analysis.verify_publication()
        return proof, completion
    return None, None


@dataclass(frozen=True, slots=True)
class ConditionalUnitDescriptorProof:
    identity: str
    procedure: str
    _analysis: object = field(repr=False, compare=False)
    _consumer: object = field(repr=False, compare=False)
    _consumer_identity: int = field(repr=False, compare=False)
    _bindings: tuple = field(repr=False, compare=False)
    _selected: tuple
    _storage: object = field(repr=False, compare=False)
    _record: str = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)

    def public(self):
        return {"proof_identity": self.identity, **json.loads(self._record)}

    @property
    def module_roots(self):
        return tuple(self.public()["module_roots"])

    def validate(self, analysis, procedure, selected):
        if (self._issuer is not _ISSUER or id(self._consumer) != self._consumer_identity
                or analysis is not self._analysis or procedure != self.procedure
                or tuple(selected) != self._selected
                or getattr(analysis, "_conditional_descriptor_consumers", {}).get(self.identity) is not self._consumer
                or getattr(analysis, "_conditional_unit_descriptors", {}).get(self.identity) is not self):
            raise CompilationError("conditional unit descriptor proof requires its exact registered original selection")
        self._storage.validate()
        record = json.loads(self._record)
        if (sha256(self._record.encode()).hexdigest() != self.identity
                or not _preserves_record(self._storage._record) or
                _encoded(_descriptor_record(analysis, procedure, self._selected, self._storage, self._bindings,
                    record["owner_entry"], record["owner_selection_identity"])) != self._record):
            raise CompilationError("conditional unit descriptor source, association or lifetime changed")
        return self


def _descriptor_record(analysis, procedure, selected, storage, bindings, owner_entry, selection_identity):
    graph, selected = analysis._selected_source(procedure, selected)
    resources, modules = [], []
    for binding in bindings:
        root = binding.root
        if binding.declaring_scope.bindings.get(binding.name) is not binding:
            raise CompilationError("conditional descriptor lost its original declaration binding")
        if not binding.rank or "allocatable" not in binding.attributes:
            continue
        resources.append({"resource": root, "stable": True, "original_descriptor_required": True,
            "descriptor_changes_within_selection": False, "escapes_within_selection": False,
            "requires_original_allocation_presence_and_alias_guards": True,
            "execution_requires_runtime_guard": True, "entry_definition_authority": False})
        if any(binding.declaring_scope is module for module in analysis.modules.values()):
            original = getattr(analysis, "_source_module_bindings", {}).get(root)
            if original is None or original[0] is not binding:
                raise CompilationError("conditional descriptor lost its exact original module binding")
            modules.append(root)
        # Local SAVE and automatic allocations remain local. Selected-unit
        # stability never turns either into module storage or return lifetime.
    return {"schema_version": 1, "kind": "checked_native_unit_descriptor_projection", "procedure": procedure,
            "owner_entry": owner_entry,
            "owner_selection_identity": selection_identity,
            "structured_identity": graph.identity, "analysis_identity": graph.authority_identity,
            "selected_node_ids": list(selected), "source_call_closure": {
                item["procedure"]: analysis.structure(item["procedure"]).identity for item in storage._record["closure"]},
            "resources": resources, "module_roots": modules, "storage_projection_identity": storage.identity,
            "entry_exit_lifecycle_authorized": False, "payload_access_authorized": False,
            "entry_definition_authority": False, "execution_authorized": False,
            "retention_requires_sealed_return_contract": True}


def _joined_descriptor_storage(analysis, procedure, graph, selected):
    """Consume one complete original join, without granting payload effects."""
    from compiler.frontend.descriptor_teams import _original_group, prove_descriptor_team
    from compiler.frontend.native_completion import prove_graph_joined_completion
    from compiler.frontend.native_group_structure import NativeGroupCandidate, grouped_original_sequence
    from compiler.frontend.resource_reachability import prove_native_effect_graph

    originals = tuple(node for identity in selected for node in graph.source_nodes(identity))
    grouped = grouped_original_sequence(originals, {})
    if (len(grouped) != 1 or not isinstance(grouped[0], NativeGroupCandidate)
            or len(grouped[0].original_nodes) != len(originals)
            or any(a is not b for a, b in zip(grouped[0].original_nodes, originals, strict=True))):
        return None
    # Graph identities omit ordinary comments. Recover the complete original
    # parser selection before asking the exact joined-team issuer; never relax
    # that issuer's source-object checks or lend a join to a smaller selection.
    originals = _original_group(analysis._require_original(procedure), originals,
                                graph_selection=True).original_nodes
    original_graph, original_selected = analysis._selected_source(procedure, originals)
    if original_graph is not graph or original_selected != selected:
        raise CompilationError("conditional descriptor changed its original joined selection")
    team = prove_descriptor_team(analysis, procedure, originals)
    if team.selected_node_ids != selected:
        raise CompilationError("conditional descriptor lost its complete original joined selection")
    storage = prove_storage_mutation_effects(analysis, procedure, selection=originals, joined_storage=team)
    if _preserves_record(storage._record):
        return storage
    # Only storage that needs the graph's stronger original-call/record
    # evidence requests payload traversal. Descriptor stability alone must not
    # depend on the unrelated payload-analysis control-depth budget.
    native = prove_native_effect_graph(analysis, procedure, originals)
    completion = prove_graph_joined_completion(analysis, procedure, originals, native)
    if completion.selected_node_ids != selected or team.selected_node_ids != selected:
        raise CompilationError("conditional descriptor lost its complete original joined selection")
    return prove_storage_mutation_effects(analysis, procedure, selection=originals, joined_storage=team,
        native_graph=native if completion._ordinary_join is not None else None)


def descriptor_for(builder, analysis, procedure, selection):
    draft = getattr(builder, "_conditional_return_draft", None)
    if draft is None:
        return None
    draft.validate()
    original = draft.owner.builder.analysis
    if (analysis.routines.get(procedure) is not builder.entry or analysis.inputs is not original.inputs
            or any(analysis.routines.get(name) is not routine
                                                    for name, routine in original.routines.items())):
        raise CompilationError("conditional descriptor projection changed original source objects")
    graph, selected = analysis._selected_source(procedure, selection)
    storage = prove_storage_mutation_effects(analysis, procedure, selection=tuple(selected))
    if not _preserves_record(storage._record):
        # A standalone storage walk cannot discharge OpenMP concurrency or
        # record-array reads. Use the existing exact joined-group authorities;
        # an unknown call, missing join or descriptor mutation still declines.
        try:
            storage = _joined_descriptor_storage(analysis, procedure, graph, selected)
        except CompilationError:
            return None
        if storage is None or not _preserves_record(storage._record):
            return None
    bindings = []
    for root in storage._record["resources"]:
        try:
            binding = builder.visible_binding(analysis.routines[procedure], root)
        except CompilationError:
            continue  # Private callee locals are not caller capture capabilities.
        if binding.rank and "allocatable" in binding.attributes:
            bindings.append(binding)
    consumer = draft.consumer_key(builder)
    record = _encoded(_descriptor_record(analysis, procedure, selected, storage, tuple(bindings),
        draft.owner.routine.qualified, draft.selection_identity()))
    identity = sha256(record.encode()).hexdigest()
    registry = getattr(analysis, "_conditional_unit_descriptors", {})
    analysis._conditional_unit_descriptors = registry
    if identity not in registry:
        if len(registry) >= source_registry_budget(analysis):
            raise CompilationError("conditional unit descriptor registry budget exhausted")
        receipts = getattr(analysis, "_conditional_descriptor_consumers", None)
        if receipts is None:
            receipts = analysis._conditional_descriptor_consumers = {}
        if identity in receipts:
            raise CompilationError("conditional descriptor lost its independent original consumer receipt")
        receipts[identity] = consumer
        registry[identity] = ConditionalUnitDescriptorProof(identity, procedure, analysis, consumer, id(consumer), tuple(bindings), tuple(selected),
                                                            storage, record, _ISSUER)
        return registry[identity]
    elif registry[identity]._consumer is not consumer:
        raise CompilationError("conditional descriptor token belongs to another original owner/builder")
    return registry[identity].validate(analysis, procedure, selected)


def _coverage(member):
    from compiler.scopes.lexical import LexicalBoundaryPatch

    graph, covered, routes = member.graph, set(), []
    draft = member.owner.builder._conditional_return_draft.validate()
    selected_root = draft.selected_root if member is member.owner else None
    roots = selected_root.selected if selected_root is not None else (graph.root_id,)
    allowed, inventory = set(), list(roots)
    while inventory:
        identity = inventory.pop()
        if identity in allowed:
            continue
        allowed.add(identity)
        node = graph.nodes[identity]
        inventory.extend(node.children)
        inventory.extend(child for condition, body in node.alternatives for child in (condition, body) if child)

    def mark(originals, kind, **extra):
        source, selected = member.builder.analysis._selected_source(member.routine.qualified, originals)
        if source is not graph or not selected or not set(selected) <= allowed:
            raise CompilationError("conditional return lost its exact original source selection")
        pending = list(selected)
        while pending:
            identity = pending.pop()
            if identity in covered:
                continue
            covered.add(identity)
            node = graph.nodes[identity]
            pending.extend(node.children)
            pending.extend(child for condition, body in node.alternatives for child in (condition, body) if child)
        routes.append({"kind": kind, "selected_node_ids": list(selected), **extra})

    for ordinal, unit in enumerate(member.units):
        mark(member.builder.inline.original_selection(unit.nodes), "checked_unit", ordinal=ordinal)
    for ordinal, (nodes, _scope) in enumerate(member.graph_native_tasks):
        mark(member.builder.inline.original_selection(nodes), "checked_native", ordinal=ordinal)
    for task in (*member.device_tasks, *member.source_device_tasks):
        mark(task.nodes if hasattr(task, "nodes") else (task.node,), "checked_device", proof_identity=task.proof.identity)
    for transfer in member.storage_transfers:
        transfer.proof.validate(member.builder.analysis, member.routine, transfer.node)
        mark((transfer.node,), "checked_identity_transfer", proof_identity=transfer.proof.identity)
    for _first, _last, patch in member.patches:
        if isinstance(patch, LexicalBoundaryPatch):
            patch.validate(member, require_source=True)
            if patch.lifecycle is None:
                # A finite invocation control cannot discharge an outstanding
                # alias/task on a later invocation. Until there is an issued
                # persistent barrier/join contract, this entire conditional
                # owner must remain native, even if the branch is inactive.
                raise CompilationError("conditional return cannot reenter after unproved lifetime/escape; "
                                       "a synchronous noescape contract or persistent barrier is required")
            if patch.lifecycle is not None:
                patch.lifecycle.validate()
            mark(patch.originals, "reached_retirement", lifetime=("proved_lifecycle" if patch.lifecycle else
                 "unproved_lifetime_or_escape"), rearm_allowed=patch.lifecycle is not None,
                 **({"loop_entry": dict(patch.counted_entry.authority)} if patch.counted_entry is not None else {}),
                 **({"call_preservation": patch.lifecycle.public()}
                    if type(patch.lifecycle) is SameFamilyNativeCallLifecycle else {}),
                 **({"empty_call_guard": patch.empty_call.public()} if patch.empty_call is not None else {}))
    for call in member.source_lexical_calls:
        call.validate(member)
        namespace = call.callee.resource_slot_namespace
        mark((call.node,), "checked_lexical_child", procedure=call.callee.routine.qualified,
             source_call=dict(call.record),
             namespace_owner=None if namespace is None else namespace.routine.qualified)
    from compiler.scopes.specification_entry import call_record
    for call in member.return_calls:
        call.validate(member)
        forward = dict(call.handle_bindings)
        canonical_handles, canonical_resources = {}, {}
        for root, (_binding, slot) in call.callee.resources.items():
            binding, = [binding for binding, local in member.resources.values() if local == forward[slot]]
            canonical = member.canonical(binding).root
            canonical_handles[str(slot)] = member.owner.resources[canonical][1]
            canonical_resources[root] = canonical
        specification = call_record(member, call)
        mark((call.node,), "checked_child", procedure=call.callee.routine.qualified,
             canonical_handles=canonical_handles, canonical_resources=canonical_resources,
             handle_forwarding={str(slot): number for slot, number in call.handle_bindings},
             **({"specification_entry": specification} if specification is not None else {}))
    for header, proof in member.return_conditions:
        current = member.graph.node_id(header, role="condition")
        if proof.effects or not proof.summary.get("complete"):
            # Payload conditions are represented by their checked publication unit.
            if current not in covered:
                raise CompilationError("conditional return condition lacks its native publication")
        else:
            mark((current,), "checked_condition", summary_identity=proof.summary["summary_identity"])
    counted = {control["header_identity"] for control in member.counted_controls}
    pending, visited = list(roots), set()
    while pending:
        identity = pending.pop()
        if identity in visited or identity in covered:
            continue
        visited.add(identity)
        node = graph.nodes[identity]
        if node.kind in {"sequence", "entry", "branch"} or node.kind == "loop" and any(child in counted for child in node.children):
            pass
        elif identity in counted:
            routes.append({"kind": "checked_counted_header", "selected_node_ids": [identity]})
            continue
        else:
            originals = graph.source_nodes(identity)
            if not (member.parent is not None and len(originals) == 1 and _kind(originals[0]) == "Return_Stmt"
                    and str(originals[0]).strip().lower() == "return"):
                raise CompilationError("conditional return has an uncovered original execution node: " + identity)
            routes.append({"kind": "checked_return", "selected_node_ids": [identity]})
            continue
        pending.extend(node.children)
        pending.extend(child for condition, body in node.alternatives for child in (condition, body) if child)
    return {"procedure": member.routine.qualified, "source_graph": graph.identity,
            "entry_exit": _entry_exit(member), "routes": routes}


def _record(owner):
    draft = owner.builder._conditional_return_draft.validate()
    if draft.owner is not owner:
        raise CompilationError("returned-domain proof changed its exact issuing owner")
    selected_root = draft.selected_root
    preservation = any(type(patch.lifecycle) is SameFamilyNativeCallLifecycle
                       for member in owner.members.values() for _first,_last,patch in member.patches
                       if hasattr(patch,"lifecycle"))
    empty_call = any(getattr(patch, "empty_call", None) is not None
                     for member in owner.members.values() for _first, _last, patch in member.patches)
    lexical = any(member.source_lexical_calls for member in owner.members.values())
    specification = any(getattr(member, "specification_reads", None) is not None for member in owner.members.values())
    return {"version": 5 if specification else 4 if lexical else 3 if preservation else 2 if selected_root is not None else 1,
            **({"specification_entry_policy_version": 2} if specification else {}),
            **({"lexical_child_policy_version": 1} if lexical else {}),
            **({"native_call_preservation_policy_version": 3 if empty_call else 2} if preservation else {}),
            "kind": "checked_conditional_return", "entry": owner.routine.qualified,
            "source_inputs": owner.builder.analysis.inputs.identity(),
            "selection_identity": draft.selection_identity(),
            **({"root_selection": selected_root.public()} if selected_root is not None else {}),
            "members": [_coverage(member) for member in owner.members.values()],
            "controls": {"retention_allowed": owner.retention_allowed, "module_control_version": 5,
                         "propagation": "INOUT by original checked CALL; false is never rearmed in this invocation"},
            "retained_return": "only checked effects and coherent identity transfers on the executed path",
            "unproved_boundary": "publish and retire before original code; disable the unchecked suffix and callers",
            "unproved_boundary_admission": "entire conditional owner remains native; no cross-invocation escape completion is assumed",
            "known_retirement": "publish and retire; new registrations have no inferred definitions",
            "escape_completion_assumed": False, "entry_definition_authority": False,
            "payload_access_authorized": False, "motion_authorized": False, "placement_authorized": False,
            "cross_cut_replica_retention_authorized": False}


@dataclass(frozen=True, slots=True)
class ConditionalReturnProof:
    identity: str
    _owner: object = field(repr=False, compare=False)
    _record: str = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)

    def public(self):
        return {"identity": self.identity, **json.loads(self._record)}

    def validate(self):
        owner = self._owner
        if (self._issuer is not _ISSUER or getattr(owner, "return_contract", None) is not self
                or _encoded(_record(owner)) != self._record):
            raise CompilationError("returned-domain contract lost its exhaustive checked source coverage")
        return self


def seal(owner):
    if getattr(owner.builder, "_conditional_return_draft", None) is None:
        return None
    record = _encoded(_record(owner))
    proof = ConditionalReturnProof(sha256(record.encode()).hexdigest(), owner, record, _ISSUER)
    owner.return_contract = proof
    return proof.validate()
