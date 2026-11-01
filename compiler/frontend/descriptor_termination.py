"""Descriptor evidence for an unchanged, fully known terminating CALL.

Diagnostic I/O and termination remain observable original operations. This
token never authorizes execution, continuation, motion, capture or definition.
The initial slice deliberately excludes nested calls and structured control.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields

from compiler.frontend.call_bindings import OriginalCallBudgetExceeded, _resolve_candidates
from compiler.frontend.proof_arena import SourcePayload, _register_family
from compiler.frontend.source_dependency_receipts import (
    DependencyCapture,
    _PrivateDependencies,
    _resolve_dependencies,
    _SourceOwnership,
)
from compiler.frontend.source_functions import (
    _hash,
    _original,
    _scope_roles,
    _validate_resolution,
    _validate_scope,
    _validate_scope_roles,
    _walk,
)
from compiler.frontend.structured_effects import _freeze, _thaw
from compiler.ir import CompilationError

_ISSUER = object()
_FORBIDDEN = {"pointer", "target", "allocatable", "optional", "save", "volatile", "asynchronous", "value"}


class DescriptorTerminationUnsupported(CompilationError):
    """A complete terminating descriptor-only contract was not established."""


def _kind(node):
    return type(node).__name__


def _part(node, kind):
    return next((item for item in getattr(node, "content", ()) if _kind(item) == kind), None)


def _resolution_inputs(node):
    nodes = tuple(_walk(node))
    names = tuple(sorted({str(item.items[1]).lower() for item in nodes
                          if _kind(item) in {"Int_Literal_Constant", "Real_Literal_Constant", "Logical_Literal_Constant"}
                          and item.items[1] is not None and str(item.items[1]).isidentifier()}))
    return nodes, names


class _Reader(DependencyCapture):
    def __init__(self, analysis, tick=None, resolution_tick=None):
        super().__init__(analysis)
        self.entries, self.scopes, self.roles, self.routines = 0, {}, None, {}
        self._tick, self._resolution_tick = tick, resolution_tick

    def routine(self, routine):
        if self.retain_routine(routine):
            self.routines[id(routine)] = routine


    def tick(self, _kind=None):
        self.entries += 1
        if self.entries > 65536:
            raise OriginalCallBudgetExceeded("terminating descriptor source scan budget exhausted")
        if self._tick is not None:
            try:
                self._tick()
            except CompilationError as error:
                # Generic candidate matching must not reinterpret the shared
                # consumer traversal bound as an ordinary overload mismatch.
                raise OriginalCallBudgetExceeded(str(error)) from error


    def scope(self, scope):
        if scope is not None:
            self.retain_scope(scope)
            _validate_scope(self.analysis, scope)
            self.scopes[id(scope)] = scope





    def require(self):
        self.receipt().require()

    def finish(self):
        for routine in self.routines.values():
            if _original(self.analysis, routine.qualified) is not routine:
                raise CompilationError("terminating descriptor original routine changed")
        for scope in self.scopes.values():
            _validate_scope(self.analysis, scope)
        _validate_scope_roles(self.analysis, self.roles, self.scopes)
        self.analysis.verify_publication()
        self.analysis.inputs.verify_live()
        self.require()

    def resolution(self, scope, node):
        self.scope(scope)
        nodes, names = _resolution_inputs(node)
        _validate_resolution(self.analysis, scope, nodes, names=names,
                             tick=self._resolution_tick or self.tick, _scope_validator=self.scope)

    def character_formal(self, routine, binding):
        # Default-kind assumed-length scalar only. No deferred length,
        # nondefault kind, writable character temporary or descriptor mapping.
        for declaration in getattr(_part(routine.scope.node, "Specification_Part"), "content", ()):
            if _kind(declaration) != "Type_Declaration_Stmt":
                continue
            dtype, _attrs, entities = declaration.items
            if not any(str(entity.items[0]).lower() == binding.name for entity in entities.items):
                continue
            if _kind(dtype) != "Intrinsic_Type_Spec" or str(dtype.items[0]).lower() != "character":
                break
            selector = dtype.items[1]
            return (_kind(selector) == "Length_Selector" and _kind(selector.items[1]) == "Type_Param_Value"
                    and str(selector.items[1]) == "*")
        return False

    def formal(self, routine, binding):
        self.tick()
        if (binding.name not in routine.arguments or routine.scope.bindings.get(binding.name) is not binding
                or binding.declaring_scope is not routine.scope or binding.rank or binding.intent != "in"
                or binding.attributes & _FORBIDDEN):
            raise DescriptorTerminationUnsupported("terminating descriptor formal requires ordinary scalar INTENT(IN)")
        if binding.dtype == "character":
            if not self.character_formal(routine, binding):
                raise DescriptorTerminationUnsupported("terminating descriptor character requires default assumed length")
        elif binding.dtype not in {"real", "integer", "logical"} or binding.kind not in {4, 8}:
            raise DescriptorTerminationUnsupported("terminating descriptor formal requires a fixed numeric scalar")

    def associate(self, caller, call, callee, actuals):
        self.routine(callee)
        self.scope(callee.scope)
        if _original(self.analysis, callee.qualified) is not callee or callee.source_kind != "module":
            raise DescriptorTerminationUnsupported("terminating descriptor requires exact original module procedure")
        records = []
        for name, actual in zip(callee.arguments, actuals, strict=True):
            binding = callee.scope.bindings[name]
            self.formal(callee, binding)
            if actual is None:
                raise DescriptorTerminationUnsupported("terminating descriptor arguments cannot be absent")
            scope = self.analysis.source_scope_for(call, caller.scope)
            if binding.dtype == "character":
                if _kind(actual) != "Char_Literal_Constant" or actual.items[1] is not None:
                    raise DescriptorTerminationUnsupported("terminating descriptor character actual must remain an original literal")
            else:
                source = self.analysis._binding(scope, actual) if _kind(actual) == "Name" else None
                if (_kind(actual) not in {"Name", "Int_Literal_Constant", "Real_Literal_Constant", "Logical_Literal_Constant"}
                        or self.analysis._signature(scope, actual) != binding.signature()
                        or source is not None and source.attributes & _FORBIDDEN):
                    raise DescriptorTerminationUnsupported("terminating descriptor numeric actual requires fixed original scalar storage or literal")
            records.append({"formal": name, "formal_descriptor": binding.public(), "original_actual": str(actual),
                            "evaluation": "unchanged original CALL position and guards; no analysis-time payload evaluation"})
        return records

    def unit(self, routine, node):
        if _kind(node) != "Name":
            raise DescriptorTerminationUnsupported("terminating descriptor diagnostic unit requires an original standard export")
        spelling, owner = str(node).lower(), routine.scope
        while owner is not None:
            self.scope(owner)
            sources = self.analysis.use_sources(owner, spelling)
            if (spelling in owner.bindings or spelling in owner.procedures or spelling in owner.generics
                    or spelling in owner.externals or spelling in owner.procedure_arguments
                    or spelling in owner.ambiguous_imports or sources is None or sources):
                raise DescriptorTerminationUnsupported("terminating descriptor diagnostic unit is shadowed or unresolved")
            choices = set()
            for use in getattr(_part(owner.node, "Specification_Part"), "content", ()):
                self.tick()
                if _kind(use) != "Use_Stmt" or str(use.items[2]).lower() != "iso_fortran_env":
                    continue
                nature, _colon, module, only, names = use.items
                if (str(nature).lower() == "non_intrinsic"
                        or str(nature).lower() != "intrinsic" and str(module).lower() in self.analysis.modules):
                    raise DescriptorTerminationUnsupported("terminating descriptor unit must come from intrinsic ISO_FORTRAN_ENV")
                unrestricted = names is None or str(only).upper().replace(" ", "") != ",ONLY:"
                renamed = set()
                for item in getattr(names, "items", ()):
                    local, remote = ((str(item.items[1]).lower(), str(item.items[2]).lower())
                                     if _kind(item) == "Rename" else (str(item).lower(), str(item).lower()))
                    if _kind(item) == "Rename":
                        renamed.add(remote)
                    if local == spelling and remote in {"error_unit", "output_unit"}:
                        choices.add(remote)
                if unrestricted and spelling in {"error_unit", "output_unit"} and spelling not in renamed:
                    choices.add(spelling)
            if choices:
                if len(choices) != 1:
                    raise DescriptorTerminationUnsupported("terminating descriptor diagnostic unit is ambiguous")
                return "$intrinsic::iso_fortran_env::" + next(iter(choices))
            owner = owner.parent
        raise DescriptorTerminationUnsupported("terminating descriptor diagnostic unit has no original intrinsic import")

    def output(self, routine, node):
        self.tick()
        if _kind(node) == "Char_Literal_Constant" and node.items[1] is None:
            return {"original_expression": str(node), "kind": "original default character literal"}
        if (_kind(node) in {"Int_Literal_Constant", "Real_Literal_Constant", "Logical_Literal_Constant"}
                and self.analysis._signature(routine.scope, node) in {
                    ("integer", 4, 0), ("integer", 8, 0), ("real", 4, 0), ("real", 8, 0), ("logical", 4, 0)}):
            return {"original_expression": str(node), "kind": "original fixed numeric literal"}
        binding = self.analysis._binding(routine.scope, node) if _kind(node) == "Name" else None
        if binding is None:
            raise DescriptorTerminationUnsupported("terminating descriptor output requires an exact readonly formal or literal")
        self.formal(routine, binding)
        return {"original_expression": str(node), "original_binding": binding.public(), "effect": "original scalar read"}

    def body(self, routine):
        self.scope(routine.scope)
        header = _part(routine.scope.node, "Subroutine_Stmt")
        if any(_kind(node) == "Language_Binding_Spec" for node in _walk(header)):
            raise DescriptorTerminationUnsupported("terminating descriptor foreign ABI requires separate source authority")
        for node in _walk(routine.execution):
            self.tick()
            if _kind(node) == "Comment" and str(node).lstrip().lower().startswith("!$omp"):
                raise DescriptorTerminationUnsupported("terminating descriptor original OpenMP requires separate completion authority")
        spec = _part(routine.scope.node, "Specification_Part")
        for node in _walk(spec):
            self.tick()
            if (_kind(node) in {"Save_Stmt", "Common_Stmt", "Equivalence_Stmt", "Namelist_Stmt", "Cray_Pointer_Stmt",
                               "Bind_Stmt", "Codimension_Attr_Spec", "Initialization", "Proc_Declaration_Stmt"}
                    or _kind(node) == "Comment" and str(node).lstrip().lower().startswith("!$omp")):
                raise DescriptorTerminationUnsupported("terminating descriptor declaration has unproved lifetime or association")
        if set(routine.scope.bindings) != set(routine.arguments):
            raise DescriptorTerminationUnsupported("terminating descriptor local state requires separate lifetime authority")
        for binding in routine.scope.bindings.values():
            self.formal(routine, binding)
        statements = [node for node in getattr(routine.execution, "content", ()) if _kind(node) != "Comment"]
        if not statements or _kind(statements[-1]) not in {"Stop_Stmt", "Error_Stop_Stmt"}:
            raise DescriptorTerminationUnsupported("terminating descriptor requires an unconditional final original STOP or ERROR STOP")
        events = []
        for index, node in enumerate(statements):
            self.tick()
            self.resolution(routine.scope, node)
            if index == len(statements) - 1:
                code = node.items[1]
                if code is not None and (type(code).__name__ != "Stop_Code" or not str(code).isdigit()):
                    raise DescriptorTerminationUnsupported("terminating descriptor stop code requires an original numeric literal")
                events.append({"kind": "original_termination", "source": str(node), "returns_to_caller": False})
                continue
            if _kind(node) != "Write_Stmt":
                raise DescriptorTerminationUnsupported("terminating descriptor full original body has unproved storage/control effects")
            values, position, keyword_seen = {}, 0, False
            for control in node.items[0].items:
                key, value = control.items
                if key is None:
                    if keyword_seen or position >= 2:
                        raise DescriptorTerminationUnsupported("terminating descriptor diagnostic WRITE control is unsupported")
                    key = ("unit", "fmt")[position]
                    position += 1
                else:
                    keyword_seen = True
                key = str(key).lower()
                if key not in {"unit", "fmt"} or key in values:
                    raise DescriptorTerminationUnsupported("terminating descriptor diagnostic WRITE control is unsupported")
                values[key] = value
            if set(values) != {"unit", "fmt"} or _kind(values["fmt"]) != "Char_Literal_Constant" or values["fmt"].items[1] is not None:
                raise DescriptorTerminationUnsupported("terminating descriptor diagnostic WRITE requires its original literal format")
            events.append({"kind": "original_external_diagnostic_write", "source": str(node),
                           "unit": self.unit(routine, values["unit"]), "format": str(values["fmt"]),
                           "outputs": [self.output(routine, output) for output in getattr(node.items[1], "items", ())]})
        return events


def _derive(analysis, caller, call, *, tick=None, resolution_tick=None):
    # Establish consumed raw facts before the first delegated source/session
    # verification. Candidate discovery uses the existing indexed lookup and
    # grants no authority; the authenticated resolution below still decides.
    from compiler.frontend.source_effects import Routine

    reader = _Reader(analysis, tick, resolution_tick)
    routine = analysis.routines.get(caller) or analysis.numerical_helpers.get(caller)
    if not isinstance(routine, Routine) or _kind(call) != "Call_Stmt" or _kind(call.items[0]) != "Name":
        raise CompilationError("terminating descriptor requires its exact original CALL")
    reader.routine(routine)
    scope = analysis.source_scope_for(call, routine.scope)
    nodes, names = _resolution_inputs(call)
    _validate_resolution(analysis, scope, nodes, names=names, _scope_validator=reader.retain_scope)
    candidates = analysis._candidates(scope, call.items[0])
    if len(candidates) > analysis.procedure_limit:
        raise CompilationError("terminating descriptor original candidate budget exhausted")
    for name in candidates:
        child = analysis.routines.get(name) or analysis.numerical_helpers.get(name)
        if isinstance(child, Routine):
            reader.routine(child)
            reader.retain_scope(child.scope)
    reader.roles = _scope_roles(analysis)
    analysis.verify_source()
    if (_original(analysis, caller) is not routine
            or not any(node is call for node in _walk(routine.execution))):
        raise CompilationError("terminating descriptor requires its exact original CALL")
    reader.require()
    occurrence = getattr(analysis, "_source_object_calls", {}).get(id(call))
    if (occurrence is None or occurrence[0] is not call or occurrence[1] != str(call)
            or occurrence[2] is not routine.scope or type(occurrence[3]) is not int):
        raise CompilationError("terminating descriptor requires its exact original callsite role")
    reader.routine(routine)
    scope = analysis.source_scope_for(call, routine.scope)
    reader.resolution(scope, call)
    if len(analysis._candidates(scope, call.items[0])) > analysis.procedure_limit:
        raise CompilationError("terminating descriptor original candidate budget exhausted")
    try:
        callee, _actuals, associations, arguments = _resolve_candidates(
            analysis, scope, call, lambda child, actuals: reader.associate(routine, call, child, actuals))
    except OriginalCallBudgetExceeded:
        raise
    except CompilationError as error:
        raise DescriptorTerminationUnsupported("terminating descriptor original scalar association is unavailable") from error
    events = reader.body(callee)
    return {"schema_version": 2, "proof_role": "original_terminating_call_descriptor",
            "procedure": caller, "source_inputs": analysis.inputs.identity(), "original_call": str(call),
            "original_call_ordinal": occurrence[3],
            "callee": callee.qualified, "callee_source_sha256": analysis.sources[str(callee.scope.path)],
            "callee_body_sha256": _hash(str(callee.execution)), "associations": associations,
            "original_arguments": [item.public() for item in arguments], "original_observable_events": events,
            "whole_original_declaration_and_body_required": True, "original_order_and_branch_required": True,
            "analysis_limits": {"operations": analysis.operation_limit, "depth": analysis.depth_limit,
                                "procedures": analysis.procedure_limit, "source_scan": 65536},
            "requires_original_call_execution": True, "returns_to_caller": False,
            "continuation_authorized": False, "execution_authorized": False, "motion_authorized": False,
            "payload_access_authorized": False, "capture_authorized": False, "entry_definition_authority": False,
            "gpu_legality_established": False, "internal_cuts_authorized": False}, callee, reader


@dataclass(frozen=True, slots=True)
class _TerminationIssue:
    family: object
    proof: object
    attachments: tuple
    receipt_fields: tuple


def _issued(proof, analysis):
    return (type(proof) is DescriptorTerminatingCallProof and type(proof._issuer) is _TerminationIssue
            and proof._issuer.family is _ISSUER and proof._issuer.proof is proof
            and proof._analysis is analysis
            and all(getattr(proof, name) is value for name, value in proof._issuer.attachments)
            and all(getattr(owner, name) is value for owner, name, value in proof._issuer.receipt_fields))


def _registry(analysis):
    registry = getattr(analysis, "_descriptor_terminating_calls", None)
    if registry is None:
        registry = analysis._descriptor_terminating_calls = {}
    if (type(registry) is not dict or any(type(key) is not tuple or len(key) != 2
            or type(key[0]) is not str or type(key[1]) is not int for key in registry)):
        raise CompilationError("terminating descriptor requires its original compiler registry")
    return registry


@dataclass(frozen=True, slots=True)
class DescriptorTerminatingCallProof:
    identity: str
    _analysis: object = field(repr=False, compare=False)
    _procedure: str
    _call: object = field(repr=False, compare=False)
    _callee: object = field(repr=False, compare=False)
    _record: object = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)
    _source_receipt: object = field(repr=False, compare=False)
    _context: object = field(default=None, repr=False, compare=False)
    _arena: object = field(default=None, repr=False, compare=False)
    _derived: object = field(default=None, repr=False, compare=False)

    def require_attachments(self):
        if not _issued(self, self._analysis):
            raise CompilationError("terminating descriptor lost its original issuer")
        if self._context is not None:
            _require_arena_attachments(self)
        else:
            self._source_receipt.require()

    def public(self):
        return {"identity": self.identity, **_thaw(self._record)}

    def validate(self, analysis, procedure, call, *, tick=None, resolution_tick=None):
        if self._context is not None:
            return _require_arena_handle(self, analysis, procedure, call, tick=tick)
        registry = _registry(analysis)
        if (not _issued(self, analysis) or procedure != self._procedure or call is not self._call
                or registry.get((procedure, id(call))) is not self):
            raise CompilationError("terminating descriptor requires its exact registered original token")
        self.require_attachments()
        saved = tuple((part.name, getattr(self, part.name)) for part in fields(self))
        record, callee, reader = _derive(analysis, procedure, call, tick=tick, resolution_tick=resolution_tick)
        if callee is not self._callee or _hash(record) != self.identity or _freeze(record) != self._record:
            raise CompilationError("terminating descriptor original source or namespace authority changed")
        reader.finish()
        self.require_attachments()
        if (not _issued(self, analysis) or _registry(analysis) is not registry
                or registry.get((procedure, id(call))) is not self
                or any(getattr(self, name) is not value for name, value in saved)):
            raise CompilationError("terminating descriptor retained issuer changed at publication")
        return self


def prove_descriptor_termination(analysis, procedure, call, *, tick=None, resolution_tick=None):
    if getattr(analysis, "_source_session", None) is not None:
        return _arena_termination(analysis, procedure, call, tick=tick, resolution_tick=resolution_tick)
    registry = _registry(analysis)
    existing = registry.get((procedure, id(call)))
    if existing is not None:
        if type(existing) is not DescriptorTerminatingCallProof:
            raise CompilationError("terminating descriptor registry contains foreign authority")
        return existing.validate(analysis, procedure, call, tick=tick, resolution_tick=resolution_tick)
    from compiler.frontend.storage_mutation_effects import source_registry_budget
    if len(registry) >= source_registry_budget(analysis):
        raise CompilationError("terminating descriptor source-sized token registry budget exhausted")
    record, callee, reader = _derive(analysis, procedure, call, tick=tick, resolution_tick=resolution_tick)
    proof = DescriptorTerminatingCallProof(_hash(record), analysis, procedure, call, callee, _freeze(record), None, reader.receipt())
    _issue(proof)
    saved = tuple((part.name, getattr(proof, part.name)) for part in fields(proof))
    reader.finish()
    if (_registry(analysis) is not registry or (procedure, id(call)) in registry
            or not _issued(proof, analysis)
            or any(getattr(proof, name) is not value for name, value in saved)):
        raise CompilationError("terminating descriptor issuing registry changed at publication")
    registry[procedure, id(call)] = proof
    return proof


def _issue(proof):
    attached = tuple((part.name, getattr(proof, part.name)) for part in fields(proof) if part.name != "_issuer")
    owners = (proof._source_receipt,)
    if type(proof._source_receipt) is _PrivateDependencies:
        owners += (proof._source_receipt.ownership,)
    receipt = tuple((owner, part.name, getattr(owner, part.name)) for owner in owners for part in fields(owner))
    object.__setattr__(proof, "_issuer", _TerminationIssue(_ISSUER, proof, attached, receipt))


def _require_arena_handle(proof, analysis, procedure, call, *, tick=None):
    if (not _issued(proof, analysis) or proof._context is None
            or proof._procedure != procedure or proof._call is not call):
        raise CompilationError("terminating descriptor requires its exact arena-issued original token")
    registry = _registry(analysis)
    if registry.get((procedure, id(call))) is not proof:
        raise CompilationError("terminating descriptor requires its exact registered original token")
    proof._context.require(analysis)
    arena = proof._context.derivation_arena(analysis)
    if not arena.accepts(proof._arena):
        raise CompilationError("terminating descriptor belongs to a stale generation arena")
    arena.require(proof._context, proof._derived)
    arena.require_handle(("original_terminating_call", id(analysis), proof._derived.query), proof)
    if tick is not None:
        try:
            tick()
        except CompilationError as error:
            raise OriginalCallBudgetExceeded(str(error)) from error
    # Context, handle and budget callbacks precede one final pure pass over
    # private raw receipts. Shared arena facts contain only encoded snapshots.
    _require_arena_attachments(proof, registry=registry)
    return proof


def _require_arena_attachments(proof, *, registry=None):
    """Callback-free final fence after the consuming publisher's live read."""
    analysis, context = proof._analysis, proof._context
    if (not _issued(proof, analysis) or getattr(analysis, "_source_authority_context", None) is not context
            or getattr(analysis, "_source_session", None) is not context.session):
        raise CompilationError("terminating descriptor retained immutable context changed")
    current_registry = _registry(analysis)
    if registry is not None and current_registry is not registry:
        raise CompilationError("terminating descriptor retained registry changed")
    active = getattr(analysis, "_source_derivation_arena", None)
    current, accepted, handle = active, False, None
    key = "original_terminating_call", id(analysis), proof._derived.query
    while current is not None:
        if current is proof._arena:
            accepted = True
        if handle is None:
            handle = current._handles.get(key)
        current = current._parent
    if (not accepted or handle is not proof or active._context is not context
            or active._records.get((proof._derived.family, proof._derived.query)) is not proof._derived
            or current_registry.get((proof._procedure, id(proof._call))) is not proof):
        raise CompilationError("terminating descriptor retained arena handle changed")
    proof._source_receipt.require()


def _termination_arena_derivation(analysis, procedure, call, tick, resolution_tick):
    # The initial reader consumes the sealed owner's closed source/policy and
    # existing verifier callbacks, preserving the pre-callback raw baseline.
    record, callee, reader = _derive(analysis, procedure, call, tick=tick, resolution_tick=resolution_tick)
    recipe = reader.recipe()
    encoded = json.dumps({"public_record": record, "public_identity": _hash(record),
                          "callee": callee.qualified, "dependencies": recipe},
                         sort_keys=True, separators=(",", ":"), allow_nan=False)
    reader.finish()
    return SourcePayload(encoded, (id(call),))


def _arena_termination(analysis, procedure, call, *, tick=None, resolution_tick=None):
    registry = _registry(analysis)
    query = procedure, id(call)
    existing = registry.get(query)
    if existing is not None and not _issued(existing, analysis):
        raise CompilationError("terminating descriptor registry contains foreign authority")
    ownership = (existing._source_receipt.ownership
                 if existing is not None and existing._context is not None
                 and type(existing._source_receipt) is _PrivateDependencies
                 and existing._context is getattr(analysis, "_source_authority_context", None)
                 else _SourceOwnership.capture(analysis))
    context = analysis.source_authority_context()
    context.require(analysis)
    if context.session.original_node(id(call)) is not call:
        raise CompilationError("terminating descriptor requires its exact original CALL")
    arena = context.derivation_arena(analysis)
    if existing is not None and existing._context is context:
        existing.validate(analysis, procedure, call, tick=tick, resolution_tick=resolution_tick)
        ownership.require()
        _require_arena_attachments(existing, registry=registry)
        return existing
    derived = arena.derive(_TERMINATION_FAMILY, query, (procedure, call, tick, resolution_tick), analysis)
    context.require(analysis)
    payload = json.loads(derived.record)
    ownership.require()
    dependencies = _resolve_dependencies(context, payload["dependencies"], ownership)
    callee = context.session.routine(analysis, payload["callee"])
    from compiler.frontend.storage_mutation_effects import source_registry_budget
    if _registry(analysis) is not registry or registry.get(query) is not existing:
        raise CompilationError("terminating descriptor issuing registry changed before publication")
    if existing is None and len(registry) >= source_registry_budget(analysis):
        raise CompilationError("terminating descriptor source-sized token registry budget exhausted")
    proof = DescriptorTerminatingCallProof(payload["public_identity"], analysis, procedure, call, callee,
                                         _freeze(payload["public_record"]), None, dependencies,
                                         context, arena, derived)
    _issue(proof)
    dependencies.require()
    arena.register_handle(("original_terminating_call", id(analysis), query), proof)
    registry[query] = proof
    return proof.validate(analysis, procedure, call)


_TERMINATION_FAMILY = _register_family("original_terminating_call", 1, _termination_arena_derivation,
                                      legacy_import=True)
