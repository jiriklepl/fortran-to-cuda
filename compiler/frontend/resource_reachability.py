"""Bounded source storage reachability, independent of complete effect summaries.

An issued result describes conservative may-accesses of one original selection.
Unrelated names are *not* a nonalias proof: external storage carries explicit
separation obligations. This module grants no capture, completion, execution,
definition, numerical or placement authority.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256

from compiler.frontend.call_bindings import (
    OriginalCallBudgetExceeded,
    resolve_original_call,
    resolve_original_function,
)
from compiler.frontend.component_bindings import _schema, _typename
from compiler.frontend.native_group_structure import (
    NATIVE_GROUP_SCAN_LIMIT,
    NativeGroupCandidate,
    grouped_original_sequence,
)
from compiler.frontend.source_effects import Binding
from compiler.frontend.source_functions import _original, _validate_scope
from compiler.frontend.source_objects import _members, _TypeBuilder
from compiler.ir import CompilationError
from compiler.ir.intrinsics import ARRAY_INQUIRIES, INTRINSICS, MODEL_INQUIRIES

VERSION = 9


class ResourceReachabilityBudgetExceeded(CompilationError):
    """A bounded demand declined; this is not a source-authenticity failure."""


_ISSUER = object()
_INTRINSICS = frozenset(INTRINSICS) | ARRAY_INQUIRIES | MODEL_INQUIRIES | {
    "allocated", "present", "sum", "product", "any", "all", "count", "minval", "maxval", "dot_product", "norm2", "matmul", "transpose"}


def _kind(node):
    if isinstance(node, tuple):
        return "tuple"
    if isinstance(node, list):
        return "list"
    return type(node).__name__


def _children(node):
    if isinstance(node, (tuple, list)):
        return tuple(node)
    return getattr(node, "content", getattr(node, "items", ())) or ()


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class StorageComponent:
    name: str
    dtype: str
    rank: int
    attributes: frozenset[str]
    schema: object = field(repr=False, compare=False)
    declaration: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class StoragePath:
    """Original declaration and selected fields, without a numerical layout."""
    binding: Binding = field(repr=False, compare=False)
    components: tuple[StorageComponent, ...] = ()

    @property
    def resource(self):
        return self.binding.root + "".join("%" + item.name for item in self.components)

    @property
    def rank(self):
        return self.components[-1].rank if self.components else self.binding.rank

    @property
    def dtype(self):
        return self.components[-1].dtype if self.components else self.binding.dtype

    @property
    def attributes(self):
        return self.binding.attributes | frozenset(flag for item in self.components for flag in item.attributes)

    def public(self):
        return {"resource": self.resource, "declaring_scope": getattr(self.binding.declaring_scope, "qualified",
                self.binding.declaring_scope.module), "base_resource": self.binding.root,
                "component_path": [item.name for item in self.components], "rank": self.rank,
                "type": self.dtype, "attributes": sorted(self.attributes), "numerical_capture_authority": False}


@dataclass(frozen=True)
class ResourceAccess:
    storage: StoragePath
    kind: str
    procedure: str
    source: str
    external: bool
    original: object = field(repr=False, compare=False)

    def public(self):
        return {"storage": self.storage.public(), "kind": self.kind, "procedure": self.procedure,
                "source": self.source, "external": self.external}


@dataclass(frozen=True)
class ReachabilityBarrier:
    kind: str
    procedure: str
    source: str
    reason: str

    def public(self):
        return vars(self).copy()


@dataclass(frozen=True)
class StorageDisjointnessObligation:
    native: StoragePath
    demanded: Binding = field(repr=False, compare=False)
    actions: tuple[str, ...]

    def public(self):
        return {"native_storage": self.native.public(), "demanded_resource": self.demanded.root,
                "demanded_declaring_scope": getattr(self.demanded.declaring_scope, "qualified",
                                                     self.demanded.declaring_scope.module),
                "actions": list(self.actions), "discharged": False,
                "requirement": "exact storage-association proof or safe reached runtime range separation",
                "distinct_source_names_prove_disjointness": False}


@dataclass(frozen=True)
class SourceCallEdge:
    caller: str
    callee: str
    original: object = field(repr=False, compare=False)
    resolved: object = field(repr=False, compare=False)

    def public(self):
        return {"caller": self.caller, "callee": self.callee, "source": str(self.original),
                "source_call": self.resolved.public()}


@dataclass(frozen=True)
class NativeTerminatingCall:
    """One original observable, nonreturning call with issued source evidence."""
    procedure: str
    original: object = field(repr=False, compare=False)
    proof: object = field(repr=False, compare=False)

    def validate(self, analysis):
        from compiler.frontend.descriptor_termination import DescriptorTerminatingCallProof
        if type(self.proof) is not DescriptorTerminatingCallProof:
            raise CompilationError("native terminal requires its issued original termination proof")
        self.proof.validate(analysis, self.procedure, self.original)
        self.proof.require_attachments()
        return self

    def public(self):
        return {"kind": "original_observable_terminal_call", "procedure": self.procedure,
                "source": str(self.original), "termination": self.proof.public(),
                "returns_to_caller": False, "original_order_and_branch_required": True,
                "motion_authority": False, "internal_cut_authority": False}


@dataclass(frozen=True)
class NativeTemplateCall:
    edge: SourceCallEdge
    associations: tuple = field(repr=False, compare=False)

    def public(self):
        return {"kind": "call", **self.edge.public(),
                "parameter_associations": [_association_public(formal, path) for formal, path in self.associations]}


def _association_public(formal, path):
    return {"formal_resource": formal.root,
            "formal_scope": getattr(formal.declaring_scope, "qualified", formal.declaring_scope.module),
            "actual_storage": path.public() if isinstance(path, StoragePath) else None,
            "expression_actual": path is None, "omitted": path is _ABSENT}


@dataclass(frozen=True)
class _AbsentActual:
    """Absence is distinct from an evaluated expression without backing storage."""


_ABSENT = _AbsentActual()


@dataclass(frozen=True)
class NativePresenceConditional:
    binding: Binding = field(repr=False, compare=False)
    original: object = field(repr=False, compare=False)
    present: tuple
    absent: tuple

    def public(self):
        return {"kind": "original_presence_conditional", "resource": self.binding.root,
                "source": str(self.original), "present": [item.public() for item in self.present],
                "absent": [item.public() for item in self.absent]}


@dataclass(frozen=True)
class ConfinedNativePointer:
    """One original local descriptor, rebound before every loop-iteration use."""
    binding: Binding = field(repr=False, compare=False)
    target: StoragePath
    loop: object = field(repr=False, compare=False)
    assignment: object = field(repr=False, compare=False)
    iterator: Binding = field(repr=False, compare=False)

    def public(self):
        return {"resource": self.binding.root, "target": self.target.public(),
                "source": str(self.assignment), "iterator": self.iterator.root,
                "coverage": "original scalar record pointer confined to one counted loop",
                "payload_is_private": False, "physical_section_authority": False}


@dataclass(frozen=True)
class NativeDeferredCompletion:
    procedure: str
    originals: tuple = field(repr=False, compare=False)
    reason: str

    def public(self):
        return {"kind": "graph_completion_obligation", "procedure": self.procedure,
                "source": str(self.originals), "ordinary_completion_reason": self.reason}


@dataclass(frozen=True)
class NativeAllocationObligation:
    """An original numeric allocation assignment, never stable payload access."""
    procedure: str
    original: object = field(repr=False, compare=False)
    target: object = field(repr=False, compare=False)
    binding: Binding = field(repr=False, compare=False)

    def public(self):
        return {"kind": "native_allocation_assignment_obligation", "procedure": self.procedure,
                "source": str(self.original), "target": str(self.target), "resource": self.binding.root,
                "storage_mutation_proof_required": True, "payload_access_authorized": False}


@dataclass(frozen=True)
class NativeWorksharingObligation:
    """Exact whole-callee orphan structure; participation remains an obligation."""
    procedure: str
    original: object = field(repr=False, compare=False)
    proof: object = field(repr=False, compare=False)

    def public(self):
        return {"kind": "native_orphan_worksharing_obligation", "procedure": self.procedure,
                "source": str(self.original), "structure_identity": self.proof.identity,
                "whole_original_callee_required": True, "serial_caller_required": True,
                "completion_authorized": False, "payload_access_authorized": False}


@dataclass(frozen=True)
class NativeEffectNode:
    """Original control/statement topology with local, parameterized effects."""
    kind: str
    original: object = field(repr=False, compare=False)
    items: tuple = ()
    declaration: object = field(default=None, repr=False, compare=False)

    def public(self):
        return {"kind": self.kind, "source": str(self.original),
                "declaration": self.declaration.public() if self.declaration is not None else None,
                "items": [{"kind": "access", **item.public()} if isinstance(item, ResourceAccess)
                          else {"kind": "barrier", **item.public()} if isinstance(item, ReachabilityBarrier)
                          else item.public() for item in self.items]}


@dataclass(frozen=True)
class NativeEffectTemplate:
    procedure: str
    identity: str
    nodes: tuple[NativeEffectNode, ...]
    calls: tuple[NativeTemplateCall, ...]
    terminations: tuple[NativeTerminatingCall, ...]
    completion: tuple = field(repr=False, compare=False)
    confined_pointers: tuple[ConfinedNativePointer, ...]
    source_work: tuple
    _analysis: object = field(repr=False, compare=False)
    _routine: object = field(repr=False, compare=False)
    _selection: object = field(repr=False, compare=False)
    _scopes: tuple = field(repr=False, compare=False)
    _schemas: tuple = field(repr=False, compare=False)
    _limits: tuple = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)
    worksharing: tuple = field(default=(), repr=False, compare=False)

    def validate(self, analysis):
        if (self._issuer is not _ISSUER or analysis is not self._analysis
                or getattr(analysis, "_native_effect_templates", {}).get(_template_key(analysis, self.procedure,
                                                                                   self._selection)) is not self
                or self._limits != _limits(analysis)
                or _original(analysis, self.procedure) is not self._routine):
            raise CompilationError("native effect template lacks exact original source authority")
        for scope in self._scopes:
            _validate_scope(analysis, scope)
        for schema in self._schemas:
            _authenticate_schema(analysis, schema)
        for call in self.calls:
            call.edge.resolved.validate(analysis, call.edge.resolved._scope, call.edge.original)
        for terminal in self.terminations:
            terminal.validate(analysis)
        for name, selected, proof in self.completion:
            proof.validate(analysis, name, selected)
        for proof in self.worksharing:
            if proof._analysis is not analysis or proof._routine is not self._routine or self._selection is not None:
                raise CompilationError("native orphan structure requires its complete original callee")
            proof.checkpoint()
        return self

    def public(self):
        return {"identity": self.identity, "procedure": self.procedure,
                "scope": "complete callee" if self._selection is None else "exact original root selection",
                "nodes": [node.public() for node in self.nodes], "source_work": dict(self.source_work),
                "confined_pointers": [item.public() for item in self.confined_pointers],
                "completion_identities": [proof.identity for _name, _selected, proof in self.completion],
                "original_worksharing_requirements": [proof.public() for proof in self.worksharing]}


@dataclass(frozen=True)
class NativeCallInstance:
    ordinal: int
    template: NativeEffectTemplate
    parent: int | None
    call: NativeTemplateCall | None
    associations: tuple = field(repr=False, compare=False)

    def public(self):
        return {"ordinal": self.ordinal, "template_identity": self.template.identity, "parent": self.parent,
                "call": self.call.edge.public() if self.call is not None else None,
                "storage_substitutions": [_association_public(formal, path) for formal, path in self.associations]}


@dataclass(frozen=True)
class NativeEffectGraph:
    identity: str
    procedure: str
    templates: tuple[NativeEffectTemplate, ...]
    instances: tuple[NativeCallInstance, ...]
    source_work: tuple
    instantiation_work: tuple
    completion_eligible: bool
    deferred_completion: tuple[NativeDeferredCompletion, ...]
    terminal_calls: tuple
    _analysis: object = field(repr=False, compare=False)
    _selection: tuple = field(repr=False, compare=False)
    _source_identity: str = field(repr=False, compare=False)
    _limits: tuple = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)
    retirement_effects_eligible: bool = False
    lifetime_obligations: tuple = field(default=(), repr=False, compare=False)

    def validate(self, analysis, procedure, original_selection):
        if (self._issuer is not _ISSUER or analysis is not self._analysis or procedure != self.procedure
                or not isinstance(original_selection, tuple) or len(original_selection) != len(self._selection)
                or any(left is not right for left, right in zip(original_selection, self._selection, strict=True))
                or self._limits != _limits(analysis)
                or getattr(analysis, "_native_effect_graphs", {}).get(self.identity) is not self):
            raise CompilationError("native effect graph lacks exact issued source authority")
        analysis._selected_source(procedure, original_selection)
        for template in self.templates:
            template.validate(analysis)
        return self

    def public(self):
        templates = len(self.templates)
        return {"schema_version": 4, "identity": self.identity, "procedure": self.procedure,
                "source_identity": json.loads(self._source_identity),
                "templates": [template.public() for template in self.templates],
                "instances": [instance.public() for instance in self.instances],
                "source_work": dict(self.source_work), "instantiation_work": dict(self.instantiation_work),
                "effect_and_association_barriers_absent": self.completion_eligible,
                "retirement_effect_barriers_absent": self.retirement_effects_eligible,
                "retirement_lifetime_obligations": [{"instance": ordinal, **item.public()}
                    for ordinal, item in self.lifetime_obligations],
                "joined_completion_obligations": [item.public() for item in self.deferred_completion],
                "terminal_calls": [{"instance": ordinal, **item.public()} for ordinal, item in self.terminal_calls],
                "completion_scope": "normal return only; original diagnostics and termination remain observable",
                "motion_authority": False, "internal_cut_authority": False,
                "counting": "each unique original template is bounded separately; root selections and complete callees remain distinct",
                "limits": {"depth": self._limits[0], "procedures": self._limits[1],
                           "unique_templates": templates,
                           "per_template": {"source_statements": self._limits[2], "source_nodes": self._limits[3]},
                           "aggregate": {"source_statements": templates * self._limits[2],
                                         "source_nodes": templates * self._limits[3]},
                           "call_instances": self._limits[2],
                           "substitution_and_obligation_steps": self._limits[3],
                           "source_policy_version": self._limits[4]},
                "workload_key_authority": False, "execution_authority": False,
                "alias_authority": False, "cost_authority": False}


def _limits(analysis):
    return analysis.depth_limit, analysis.procedure_limit, analysis.operation_limit, NATIVE_GROUP_SCAN_LIMIT, VERSION


def _template_key(analysis, name, selection):
    return id(analysis), name, None if selection is None else tuple(map(id, selection))


@dataclass(frozen=True)
class ResourceReachabilityProof:
    procedure: str
    identity: str
    accesses: tuple[ResourceAccess, ...]
    obligations: tuple[StorageDisjointnessObligation, ...]
    barriers: tuple[ReachabilityBarrier, ...]
    calls: tuple[SourceCallEdge, ...]
    native_graph: NativeEffectGraph
    demanded: tuple[Binding, ...] = field(repr=False, compare=False)
    _analysis: object = field(repr=False, compare=False)
    _selection: tuple = field(repr=False, compare=False)
    _routines: tuple = field(repr=False, compare=False)
    _scopes: tuple = field(repr=False, compare=False)
    _schemas: tuple = field(repr=False, compare=False)
    _completion: tuple = field(repr=False, compare=False)
    _limits: tuple = field(repr=False, compare=False)
    _work: tuple = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)

    def public(self):
        effects = []
        for binding in self.demanded:
            actions = sorted({item.kind for item in self.accesses if item.storage.binding is binding})
            effects.append({"resource": binding.root, "declaring_scope": getattr(binding.declaring_scope, "qualified",
                            binding.declaring_scope.module), "may_access": actions})
        return {"schema_version": VERSION, "proof_identity": self.identity, "procedure": self.procedure,
                "accesses": [item.public() for item in self.accesses], "demanded_effects": effects,
                "storage_disjointness_obligations": [item.public() for item in self.obligations],
                "barriers": [item.public() for item in self.barriers], "calls": [item.public() for item in self.calls],
                "selection_covered": not self.barriers, "scan_work": dict(self._work),
                "native_graph": self.native_graph.public(),
                "whole_procedure_effects_complete": False, "execution_authorized": False,
                "capture_authorized": False, "definition_authorized": False, "alias_authorized": False,
                "scope": "exact original selection and statically resolved source callees; original optional associations",
                "control": "unchanged original source; PRESENT follows exact supplied/omitted associations only"}

    def validate(self, analysis, procedure, original_selection, demanded):
        if (self._issuer is not _ISSUER or analysis is not self._analysis or procedure != self.procedure
                or not isinstance(original_selection, tuple) or len(original_selection) != len(self._selection)
                or any(left is not right for left, right in zip(original_selection, self._selection, strict=True))
                or not isinstance(demanded, tuple) or len(demanded) != len(self.demanded)
                or any(left is not right for left, right in zip(demanded, self.demanded, strict=True))
                or getattr(analysis, "_resource_reachability_proofs", {}).get(self.identity) is not self):
            raise CompilationError("resource reachability lacks exact registered source and demand authority")
        if self._limits != _limits(analysis):
            raise CompilationError("resource reachability proof budgets changed")
        analysis.verify_source()
        self.native_graph.validate(analysis, procedure, original_selection)
        for name, routine in self._routines:
            if _original(analysis, name) is not routine:
                raise CompilationError("resource reachability original procedure changed")
        for scope in self._scopes:
            _validate_scope(analysis, scope)
        for schema in self._schemas:
            _authenticate_schema(analysis, schema)
        for binding in self.demanded:
            _authenticate_binding(analysis, binding)
        for edge in self.calls:
            edge.resolved.validate(analysis, edge.resolved._scope, edge.original)
        for name, selection, proof in self._completion:
            proof.validate(analysis, name, selection)
        return self


def _authenticate_binding(analysis, binding):
    if (not isinstance(binding, Binding) or binding.declaring_scope is None
            or binding.declaring_scope.bindings.get(binding.name) is not binding):
        raise CompilationError("resource demand requires an exact original Binding and declaring scope")
    _validate_scope(analysis, binding.declaring_scope)
    reason = analysis.resource_identity_boundary(binding)
    if reason:
        raise CompilationError(reason)


def _authenticate_schema(analysis, schema):
    record = _TypeBuilder(analysis).scope(schema.scope)
    original = record[5].get(schema.name)
    if (original is None or original[0] is not schema or original[1] is not schema.node
            or original[2] != str(schema.node) or original[3] != _members(schema)):
        raise CompilationError("resource reachability component declaration changed")


class _Scan:
    def __init__(self, analysis, demanded):
        self.analysis, self.demanded = analysis, demanded
        self.accesses, self.barriers, self.calls, self.completion = [], [], [], []
        self.routines, self.scopes, self.schemas = {}, {}, {}
        self.visits = self.operations = 0
        self.active = []
        self.specifications = set()
        self.nodes, self.frames = [], []
        self.template_calls, self.terminal_calls = [], []
        self.confined_pointers = {}
        self.joined_depth = 0
        self.local_loops = []
        self.intrinsic_declarations = {}
        self.worksharing = {}

    def explicit_intrinsics(self, scope):
        """Read exact original declarations absent from the ordinary name map."""
        self.scope(scope)
        if id(scope) not in self.intrinsic_declarations:
            names = set()
            specification = next((node for node in _children(scope.node)
                                  if _kind(node) == "Specification_Part"), None)
            # A nested INTERFACE body has its own lexical specification. Its
            # declarations cannot establish an intrinsic in this scope.
            for node in _children(specification):
                self.tick()
                if _kind(node) == "Intrinsic_Stmt":
                    names.update(str(item).lower() for item in _children(node.items[-1]))
                elif (_kind(node) == "Type_Declaration_Stmt"
                      and any(str(item).lower() == "intrinsic" for item in _children(node.items[1]))):
                    names.update(str(item.items[0]).lower() for item in _children(node.items[2]))
            self.intrinsic_declarations[id(scope)] = names
        return self.intrinsic_declarations[id(scope)]

    def original_intrinsic(self, scope, name):
        """Resolve native read effects without trusting intrinsic spellings.

        Explicit local INTRINSIC declarations shadow host names. Otherwise the
        existing closed-name lookup retains all consulted scopes and rejects
        unresolved EXTERNAL, generic and imported alternatives. A data Binding
        alone cannot authorize a typed host/USE intrinsic across a nearer
        procedure declaration; those forms remain conservative.
        """
        spelling = str(name).lower()
        owner, declared = scope, spelling
        if owner is not None and declared in self.explicit_intrinsics(owner):
            local = owner.bindings.get(declared)
            if (declared in _INTRINSICS
                    and declared not in owner.externals | owner.procedure_arguments
                    and declared not in owner.procedures and declared not in owner.generics
                    and declared not in owner.imports and declared not in owner.ambiguous_imports
                    and (local is None or not local.rank and local.intent is None
                         and not local.attributes - {"intrinsic", "public", "private"})):
                return declared
            return None
        if spelling not in _INTRINSICS:
            return None
        from compiler.frontend.source_functions import original_name_absent
        return spelling if original_name_absent(
            self.analysis, scope, spelling, _scope_validator=self.scope) else None

    def local_loop(self, children, *, called):
        """Retain a complete callee-local target, not a termination proof.

        If the unchanged callee returns, its ordinary local loop has already
        completed. EXIT/CYCLE to that loop cannot bypass its caller's join.
        Root selections, concurrent controls and enclosing joined teams cannot
        borrow this rule. The original body and every possible effect remain
        in the graph, including statements after a conditional transfer.
        """
        body = tuple(node for node in children if _kind(node) != "Comment")
        if (len(body) < 2 or _kind(body[0]) != "Nonlabel_Do_Stmt"
                or _kind(body[-1]) != "End_Do_Stmt"):
            return None, None, False
        header, ending = body[0], body[-1]
        name = getattr(getattr(header, "item", None), "name", None)
        name = str(name).lower() if name is not None else None
        end_name = ending.items[1]
        end_name = str(end_name).lower() if end_name is not None else None
        control = header.items[1]
        ordinary = control is None
        if control is not None and _kind(control) == "Loop_Control":
            parts = _children(control)
            ordinary = (len(parts) in {3, 4} and parts[2] in {None, ","}
                        and (len(parts) == 3 or parts[3] is None)
                        and (parts[0] is None) != (parts[1] is None))
        unique = name is None or not any(item[1] == name for item in self.local_loops)
        return header, name, bool(called and not self.joined_depth and ordinary
                                 and end_name == name and unique)

    def local_transfer(self, routine, node, *, called):
        target_name = node.items[1]
        targets = ([item for item in self.local_loops if item[1] == str(target_name).lower()]
                   if target_name is not None else self.local_loops[-1:])
        if (not called or self.joined_depth or len(targets) != 1 or not targets[0][2]):
            self.barrier("control", routine, node,
                         "EXIT/CYCLE requires its complete original callee-local ordinary loop target")

    def record(self, item):
        (self.frames[-1] if self.frames else self.nodes).append(item)

    def frame(self, kind, original, operation, *, declaration=None):
        self.frames.append([])
        try:
            operation()
            items = tuple(self.frames[-1])
        finally:
            self.frames.pop()
        self.record(NativeEffectNode(kind, original, items, declaration))

    def tick(self, statement=False):
        self.visits += 1
        self.operations += int(statement)
        if self.visits > NATIVE_GROUP_SCAN_LIMIT or self.operations > self.analysis.operation_limit:
            raise ResourceReachabilityBudgetExceeded("resource reachability source scan budget exhausted")

    def barrier(self, kind, routine, node, reason):
        item = ReachabilityBarrier(kind, routine.qualified, str(node), reason)
        if item not in self.barriers:
            self.barriers.append(item)
            self.record(item)

    def scope(self, scope):
        pending = scope
        while pending is not None:
            if id(pending) not in self.scopes:
                _validate_scope(self.analysis, pending)
                self.scopes[id(pending)] = pending
            pending = pending.parent
        return scope

    def confined_loop_pointers(self, routine):
        """Recognize a closed lexical association, never a general pointer alias.

        The descriptor must be a non-SAVE local scalar record pointer. Its sole
        association is the first operation of one counted loop, to one element
        of a nonpointer TARGET array indexed by that loop's iterator. Every use
        is a component access within that loop; whole-pointer passing, inquiries,
        reassociation, escape and use after the loop retain the ordinary barrier.
        The target is still a whole storage path: no alias or bounds proof is
        inferred from the syntax.
        """
        nodes, parents, pending = [], {}, [(routine.execution, None)]
        while pending:
            node, parent = pending.pop()
            self.tick()
            nodes.append(node)
            parents[id(node)] = parent
            pending.extend((child, node) for child in reversed(_children(node)))
        assignments = [node for node in nodes if _kind(node) == "Pointer_Assignment_Stmt"]
        for assignment in assignments:
            left, bounds, right = assignment.items
            if _kind(left) != "Name" or bounds is not None or _kind(right) != "Part_Ref":
                continue
            scope = self.scope(self.analysis.source_scope_for(assignment, routine.scope))
            binding = self.analysis._binding(scope, left)
            target = self.analysis._binding(scope, right.items[0])
            loop = parents.get(id(assignment))
            if (binding is None or target is None or _kind(loop) != "Block_Nonlabel_Do_Construct"
                    or binding.declaring_scope is not routine.scope or binding.name in routine.arguments
                    or binding.rank or binding.attributes != frozenset({"pointer"})
                    or not _typename(binding.dtype) or target.rank != 1 or "target" not in target.attributes
                    or target.attributes & {"pointer", "optional", "volatile", "asynchronous"}):
                continue
            body = tuple(node for node in _children(loop) if _kind(node) != "Comment")
            control = body[0].items[1] if body and _kind(body[0]) == "Nonlabel_Do_Stmt" else None
            indices = _children(right.items[1])
            if (len(body) < 3 or body[1] is not assignment or _kind(body[-1]) != "End_Do_Stmt"
                    or control is None or control.items[1] is None or len(indices) != 1
                    or _kind(indices[0]) != "Name" or str(indices[0]).lower() != str(control.items[1][0]).lower()):
                continue
            iterator = self.analysis._binding(self.analysis.source_scope_for(control, routine.scope), control.items[1][0])
            if (iterator is None or iterator.declaring_scope is not routine.scope or iterator.rank
                    or iterator.dtype != "integer" or iterator.attributes & {"pointer", "save", "optional"}):
                continue
            if sum(_kind(item.items[0]) == "Name" and str(item.items[0]).lower() == binding.name
                   for item in assignments) != 1:
                continue
            schema = _schema(self.analysis, binding.declaring_scope, _typename(binding.dtype))
            target_schema = _schema(self.analysis, target.declaring_scope, _typename(target.dtype))
            if schema is None or schema is not target_schema or schema.reason:
                continue
            _authenticate_schema(self.analysis, schema)
            self.schemas[id(schema)] = schema
            uses = [node for node in nodes if _kind(node) == "Name" and str(node).lower() == binding.name]
            confined = True
            for use in uses:
                if use is left:
                    continue
                parent = parents.get(id(use))
                if _kind(parent) != "Data_Ref" or parent.items[0] is not use:
                    confined = False
                    break
                enclosing = parent
                while enclosing is not None and enclosing is not loop:
                    enclosing = parents.get(id(enclosing))
                if enclosing is None:
                    confined = False
                    break
            if not confined:
                continue
            # Pointer initialization is not a fresh iteration-local association.
            specification = next((part for part in _children(routine.scope.node)
                                  if _kind(part) == "Specification_Part"), None)
            pending = [specification]
            initialized = False
            while pending:
                node = pending.pop()
                self.tick()
                if (_kind(node) == "Entity_Decl" and str(node.items[0]).lower() == binding.name
                        and node.items[-1] is not None):
                    initialized = True
                pending.extend(_children(node))
            if not initialized:
                self.confined_pointers[id(binding)] = ConfinedNativePointer(
                    binding, StoragePath(target), loop, assignment, iterator)

    def presence_condition(self, routine, expression):
        positive = True
        if _kind(expression) == "Parenthesis":
            expression = expression.items[1]
        if len(_children(expression)) == 2 and str(expression.items[0]).lower() == ".not.":
            expression, positive = expression.items[1], False
        if _kind(expression) not in {"Intrinsic_Function_Reference", "Part_Ref"}:
            return None
        name, arguments = expression.items
        scope = self.scope(self.analysis.source_scope_for(expression, routine.scope))
        args = _children(arguments)
        if (str(name).lower() != "present" or len(args) != 1 or _kind(args[0]) != "Name"
                or self.analysis._binding(scope, name) or self.analysis._candidates(scope, name)
                or self.analysis._unknown_exports(scope)):
            return None
        binding = self.analysis._binding(scope, args[0])
        if binding is None or "optional" not in binding.attributes or binding.name not in routine.arguments:
            return None
        return binding, positive

    def presence_branch(self, routine, node, condition, then, otherwise, mapping, *, called, depth):
        presence = self.presence_condition(routine, condition)
        if presence is None:
            return False
        branches = []
        for body in (then, otherwise):
            self.frames.append([])
            try:
                self.sequence(routine, body, mapping, called=called, depth=depth + 1)
                branches.append(tuple(self.frames[-1]))
            finally:
                self.frames.pop()
        binding, positive = presence
        present, absent = branches if positive else reversed(branches)
        self.record(NativePresenceConditional(binding, node, present, absent))
        return True

    def storage(self, routine, node, mapping):
        scope = self.scope(self.analysis.source_scope_for(node, routine.scope))
        parts = tuple(node.items) if _kind(node) == "Data_Ref" else (node,)
        base = parts[0].items[0] if _kind(parts[0]) == "Part_Ref" else parts[0]
        binding = self.analysis._binding(scope, base)
        if binding is None:
            raise CompilationError("unresolved original storage: " + str(base))
        if binding.declaring_scope is None:
            raise CompilationError("original storage declaration scope is unavailable: " + binding.name)
        self.scope(binding.declaring_scope)
        path = mapping.get(id(binding), StoragePath(binding))
        if path is None:
            raise CompilationError("expression actual has no original storage association: " + str(base))
        current_dtype, defining = binding.dtype, binding.declaring_scope
        components = list(path.components)
        for part in parts[1:]:
            self.tick()
            typename = _typename(current_dtype)
            schema = _schema(self.analysis, defining, typename) if typename else None
            if schema is None or schema.reason:
                raise CompilationError("unresolved or polymorphic source component: " + str(part))
            _authenticate_schema(self.analysis, schema)
            self.schemas[id(schema)] = schema
            self.scope(schema.scope)
            name = str(part.items[0] if _kind(part) == "Part_Ref" else part).lower()
            declaration = schema.members.get(name)
            if declaration is None:
                raise CompilationError("source component declaration unavailable: " + name)
            dtype, attributes, entity = declaration
            flags = frozenset(str(item).split("(")[0].lower() for item in attributes)
            if (schema.private or "private" in flags) and scope.module != schema.scope.module:
                raise CompilationError("source component is inaccessible: " + name)
            dimension = next((item.items[1] for item in attributes if _kind(item) == "Dimension_Component_Attr_Spec"), None)
            shape = entity.items[1] if entity.items[1] is not None else dimension
            current_dtype, defining = str(dtype).lower(), schema.scope
            components.append(StorageComponent(name, current_dtype, len(_children(shape)), flags, schema, entity))
        return StoragePath(path.binding, tuple(components)), binding

    def access(self, routine, node, mapping, action, *, assignment=None):
        scope = self.scope(self.analysis.source_scope_for(node, routine.scope))
        if _kind(node) == "Name":
            binding = self.analysis._binding(scope, node)
            if binding is not None and id(binding) in mapping and mapping[id(binding)] is None:
                if action not in {"read", "descriptor"}:
                    self.barrier("association", routine, node, "writable expression actual has no original storage")
                return  # Its original value expression was scanned at the CALL.
        try:
            path, binding = self.storage(routine, node, mapping)
        except CompilationError as error:
            self.barrier("storage", routine, node, str(error))
            return
        if "parameter" in path.attributes:
            return
        if path.attributes & {"pointer", "volatile", "asynchronous"}:
            self.barrier("association", routine, node, "pointer, optional or asynchronous storage needs separate authority")
        local = (binding.declaring_scope is routine.scope and binding.name not in routine.arguments
                 and "save" not in binding.attributes and id(binding) not in mapping)
        item = ResourceAccess(path, action, routine.qualified, str(node), not local, node)
        self.accesses.append(item)
        self.record(item)
        if _typename(path.dtype) and action in {"read", "write"}:
            self.barrier("implicit_call", routine, node, "whole derived-object operation may invoke defined assignment or finalization")
        terminal_attributes = path.components[-1].attributes if path.components else path.binding.attributes
        whole = _kind(node) == "Name" or _kind(node) == "Data_Ref" and _kind(node.items[-1]) == "Name"
        if action == "write" and "allocatable" in terminal_attributes and whole:
            if (_kind(node) == "Name" and _kind(assignment) == "Assignment_Stmt"
                    and assignment.items[0] is node and binding.dtype in {"integer", "real", "logical"}
                    and not path.components and not binding.attributes & {"pointer", "target", "volatile", "asynchronous"}):
                self.record(NativeAllocationObligation(routine.qualified, assignment, node, binding))
            else:
                self.barrier("allocation", routine, node, "whole allocatable assignment may reallocate original storage")
        parts = node.items if _kind(node) == "Data_Ref" else (node,)
        for part in parts:
            if _kind(part) == "Part_Ref":
                for index in _children(part.items[1]):
                    self.expression(routine, index, mapping)

    def expression(self, routine, node, mapping, action="read"):
        if node is None or isinstance(node, (str, int)):
            return
        self.tick()
        kind = _kind(node)
        if kind.endswith("Literal_Constant"):
            return
        reference_binding = (self.analysis._binding(self.analysis.source_scope_for(node, routine.scope), node.items[0])
                             if kind == "Part_Ref" else None)
        if kind in {"Name", "Data_Ref"} or reference_binding is not None and reference_binding.rank:
            self.access(routine, node, mapping, action)
        elif kind in {"Intrinsic_Function_Reference", "Function_Reference", "Part_Ref", "Structure_Constructor"}:
            name, args = node.items
            scope = self.scope(self.analysis.source_scope_for(node, routine.scope))
            try:
                intrinsic = self.original_intrinsic(scope, name)
            except CompilationError as error:
                self.barrier("call", routine, node, str(error))
                return
            if intrinsic is not None:
                for index, argument in enumerate(_children(args)):
                    if _kind(argument) == "Actual_Arg_Spec":
                        argument = argument.items[1]
                    inquiry = index == 0 and intrinsic in ARRAY_INQUIRIES | MODEL_INQUIRIES | {"allocated", "present"}
                    self.expression(routine, argument, mapping, "descriptor" if inquiry else "read")
            else:
                self.call(routine, node, mapping, function=True)
        else:
            if ("Defined" in kind or kind in {"Level_5_Expr", "Coindexed_Named_Object"}
                    or kind not in {"list", "tuple", "Parenthesis", "Level_1_Expr", "Level_2_Expr",
                        "Level_2_Unary_Expr", "Level_3_Expr", "Level_4_Expr", "Add_Operand", "Mult_Operand",
                        "And_Operand", "Or_Operand", "Equiv_Operand", "Subscript_Triplet", "Loop_Control",
                        "Explicit_Shape_Spec", "Assumed_Shape_Spec", "Assumed_Size_Spec", "Deferred_Shape_Spec",
                        "Array_Constructor", "Ac_Value_List", "Section_Subscript_List", "Return_Stmt",
                        "Case_Selector", "Case_Value_Range_List", "Case_Value_Range"}):
                self.barrier("expression", routine, node, "expression requires separate implicit-call or storage authority")
            for child in _children(node):
                self.expression(routine, child, mapping, action)

    def terminating_call(self, routine, node, mapping):
        from compiler.frontend.descriptor_termination import (
            DescriptorTerminationUnsupported,
            prove_descriptor_termination,
        )
        try:
            proof = prove_descriptor_termination(self.analysis, routine.qualified, node,
                                                 tick=self.tick, resolution_tick=lambda *_args: self.tick())
        except OriginalCallBudgetExceeded as error:
            raise ResourceReachabilityBudgetExceeded(str(error)) from error
        except DescriptorTerminationUnsupported:
            return False
        terminal = NativeTerminatingCall(routine.qualified, node, proof).validate(self.analysis)
        # The issuer covers the complete diagnostic/STOP body and rejects all
        # hidden calls, writes, local state and lifetime actions. Only numeric
        # original actual reads need projection into this caller's graph.
        for argument in _children(node.items[1]):
            if _kind(argument) == "Actual_Arg_Spec":
                argument = argument.items[1]
            self.expression(routine, argument, mapping)
        for statement in _children(proof._callee.execution):
            self.tick(statement=_kind(statement).endswith("_Stmt"))
        self.terminal_calls.append(terminal)
        self.record(terminal)
        return True

    def call(self, routine, node, mapping, *, function=False):
        scope = self.scope(self.analysis.source_scope_for(node, routine.scope))
        candidates = self.analysis._candidates(scope, node.items[0])
        pool = self.analysis.numerical_helpers if function else self.analysis.routines
        for candidate in candidates:
            callee = pool.get(candidate)
            if callee is not None:
                self.scope(callee.scope)
                if set(callee.arguments) & callee.scope.procedure_arguments:
                    self.barrier("callback", routine, node, "procedure-dummy association requires exact callback authority")
                    return
        # This syntax hint selects an existing proof attempt only. A failed
        # complete issuer falls through to the unchanged ordinary traversal.
        terminal_hint = not function and any(
            (body := tuple(item for item in _children(callee.execution) if _kind(item) != "Comment"))
            and _kind(body[-1]) in {"Stop_Stmt", "Error_Stop_Stmt"}
            for name in candidates if (callee := pool.get(name)) is not None)
        if terminal_hint and self.terminating_call(routine, node, mapping):
            return
        try:
            resolved = (resolve_original_function if function else resolve_original_call)(self.analysis, scope, node)
        except OriginalCallBudgetExceeded as error:
            raise ResourceReachabilityBudgetExceeded(str(error)) from error
        except CompilationError as error:
            if not function and not terminal_hint and self.terminating_call(routine, node, mapping):
                return
            self.barrier("call", routine, node, str(error))
            return
        if resolved.procedure.startswith("$external::"):
            self.barrier("external_call", routine, node, "external linkage/interposition needs independent original call authority")
            return
        edge = SourceCallEdge(routine.qualified, resolved.procedure, node, resolved)
        self.calls.append(edge)
        associations = []
        for item in resolved.associations:
            self.scope(item.formal_binding.declaring_scope)
            if item.actual is None:
                associations.append((item.formal_binding, _ABSENT))
                continue
            if item.formal_binding.attributes & {"pointer", "allocatable", "value"}:
                self.barrier("association", routine, node, "callee descriptor/presence association needs separate authority")
            if item.binding is not None:
                try:
                    path, _binding = self.storage(routine, item.actual, mapping)
                except CompilationError as error:
                    self.barrier("association", routine, item.actual, str(error))
                    continue
                associations.append((item.formal_binding, path))
                self.access(routine, item.actual, mapping, "association")
            else:
                associations.append((item.formal_binding, None))
                self.expression(routine, item.actual, mapping)
        call = NativeTemplateCall(edge, tuple(associations))
        self.template_calls.append(call)
        self.record(call)

    def sequence(self, routine, nodes, mapping, *, called, depth=0):
        if depth > self.analysis.depth_limit:
            raise ResourceReachabilityBudgetExceeded("resource reachability control depth budget exhausted")
        peeled = {}
        for item in grouped_original_sequence(nodes, peeled):
            if isinstance(item, NativeGroupCandidate):
                originals = item.original_nodes
                def group(originals=originals):
                    try:
                        proof = self.analysis.joined_completion(routine.qualified, originals)
                        self.completion.append((routine.qualified, proof.selected_node_ids, proof))
                    except CompilationError as error:
                        self.record(NativeDeferredCompletion(routine.qualified, originals, str(error)))
                    self.joined_depth += 1
                    try:
                        for node in originals:
                            if _kind(node) != "Comment":
                                self.statement(routine, node, mapping, called=called, depth=depth + 1,
                                               ignore_directives=True)
                    finally:
                        self.joined_depth -= 1
                self.frame("joined_group", originals, group)
            else:
                self.statement(routine, item, mapping, called=called, depth=depth)

    def statement(self, routine, node, mapping, *, called, depth=0, ignore_directives=False):
        self.frame(_kind(node), node, lambda: self._statement(routine, node, mapping, called=called,
                   depth=depth, ignore_directives=ignore_directives))

    def orphan_worksharing(self, routine, node):
        from compiler.frontend.descriptor_worksharing import original_worksharing_storage

        try:
            if routine.qualified not in self.worksharing:
                self.worksharing[routine.qualified] = original_worksharing_storage(self.analysis, routine)
            proof = self.worksharing[routine.qualified]
            proof.checkpoint()
        except CompilationError:
            return False
        if not any(node is opening or node is ending for opening, _loop, ending, _collapse in proof._regions):
            return False
        self.record(NativeWorksharingObligation(routine.qualified, node, proof))
        return True

    def _statement(self, routine, node, mapping, *, called, depth=0, ignore_directives=False):
        self.tick(statement=_kind(node).endswith("_Stmt"))
        kind = _kind(node)
        if kind == "Comment":
            if (str(node).lstrip().lower().startswith("!$omp") and not ignore_directives
                    and (not called or not self.orphan_worksharing(routine, node))):
                self.barrier("completion", routine, node, "unowned original OpenMP directive")
        elif kind == "Assignment_Stmt":
            self.access(routine, node.items[0], mapping, "write", assignment=node)
            self.expression(routine, node.items[2], mapping)
        elif kind == "Call_Stmt":
            self.call(routine, node, mapping)
        elif kind == "If_Stmt":
            if self.presence_branch(routine, node, node.items[0], (node.items[1],), (), mapping,
                                    called=called, depth=depth):
                return
            self.expression(routine, node.items[0], mapping)
            self.statement(routine, node.items[1], mapping, called=called, depth=depth + 1)
        elif kind in {"If_Construct", "Block_Nonlabel_Do_Construct", "Case_Construct"}:
            children = tuple(child for child in _children(node)
                             if not (ignore_directives and _kind(child) == "Comment"
                                     and str(child).lstrip().lower().startswith("!$omp")))
            if kind == "If_Construct":
                labels = tuple(_kind(child) for child in children)
                if (labels and labels[0] == "If_Then_Stmt" and labels[-1] == "End_If_Stmt"
                        and "Else_If_Stmt" not in labels):
                    split = labels.index("Else_Stmt") if "Else_Stmt" in labels else len(children) - 1
                    if self.presence_branch(routine, node, children[0].items[0], children[1:split],
                                            children[split + 1:-1], mapping, called=called, depth=depth):
                        return
            if kind == "Block_Nonlabel_Do_Construct":
                mapping = {**mapping, **{identity: pointer.target
                           for identity, pointer in self.confined_pointers.items() if pointer.loop is node}}
                self.local_loops.append(self.local_loop(children, called=called))
                try:
                    self.sequence(routine, children, mapping, called=called, depth=depth + 1)
                finally:
                    self.local_loops.pop()
            else:
                self.sequence(routine, children, mapping, called=called, depth=depth + 1)
        elif kind in {"If_Then_Stmt", "Else_If_Stmt", "Nonlabel_Do_Stmt", "Select_Case_Stmt", "Case_Stmt"}:
            if kind == "Nonlabel_Do_Stmt":
                control = next((part for part in _children(node) if _kind(part) == "Loop_Control"), None)
                if control is None or control.items[1] is None:
                    if not (self.local_loops and self.local_loops[-1][0] is node and self.local_loops[-1][2]):
                        self.barrier("control", routine, node, "non-counted loop needs separate completion authority")
                else:
                    self.access(routine, control.items[1][0], mapping, "write")
            for child in _children(node):
                self.expression(routine, child, mapping)
        elif kind in {"Else_Stmt", "End_If_Stmt", "End_Do_Stmt", "End_Select_Stmt", "Continue_Stmt"}:
            pass
        elif kind in {"Exit_Stmt", "Cycle_Stmt"}:
            self.local_transfer(routine, node, called=called)
        elif kind == "Return_Stmt" and called and not self.joined_depth:
            self.expression(routine, node, mapping)
        elif kind in {"Allocate_Stmt", "Deallocate_Stmt", "Pointer_Assignment_Stmt"}:
            pointer = next((item for item in self.confined_pointers.values() if item.assignment is node), None)
            if pointer is not None:
                self.access(routine, node.items[2], mapping, "association")
                return
            self.barrier("association", routine, node, "allocation or pointer association changes need separate authority")
            self.expression(routine, node, mapping, "association")
        else:
            self.barrier("statement", routine, node, "unsupported original statement or nonlocal completion")

    def specification_authority(self, routine):
        for owner in tuple(self.scopes.values()):
            if id(owner) in self.specifications:
                continue
            self.specifications.add(id(owner))
            specification = next((part for part in _children(owner.node) if _kind(part) == "Specification_Part"), None)
            pending = [specification]
            while pending:
                node = pending.pop()
                self.tick()
                pending.extend(reversed(_children(node)))
                if _kind(node) in {"Common_Stmt", "Equivalence_Stmt"}:
                    self.barrier("storage_association", routine, node, "COMMON/EQUIVALENCE requires independent storage identity")
                elif _kind(node) == "Comment" and str(node).lstrip().lower().startswith("!$omp"):
                    self.barrier("completion", routine, node, "declaration OpenMP ownership requires separate authority")
    def routine(self, name, mapping, selection=None):
        if name in self.active or len(self.active) >= self.analysis.depth_limit:
            raise ResourceReachabilityBudgetExceeded("resource reachability call depth or recursion budget exhausted")
        routine = _original(self.analysis, name)
        self.routines[name] = routine
        if len(self.routines) > self.analysis.procedure_limit:
            raise ResourceReachabilityBudgetExceeded("resource reachability procedure budget exhausted")
        self.scope(routine.scope)
        self.specification_authority(routine)
        self.active.append(name)
        self.confined_loop_pointers(routine)
        if selection is None:
            header = next((part for part in _children(routine.scope.node)
                           if _kind(part) in {"Subroutine_Stmt", "Function_Stmt"}), None)
            pending = [header]
            while pending:
                part = pending.pop()
                self.tick()
                if _kind(part) == "Language_Binding_Spec":
                    self.barrier("external_call", routine, part, "C binding needs independent linkage and escape authority")
                pending.extend(reversed(_children(part)))
            for binding in routine.scope.bindings.values():
                if binding.dtype == "character":
                    self.barrier("specification", routine, routine.scope.node,
                                 "CHARACTER length and entry specification effects need separate authority")
                self.frame("declaration", tuple(binding.shape_nodes),
                           lambda binding=binding: [self.expression(routine, shape, mapping)
                                                    for shape in binding.shape_nodes], declaration=binding)
                if (_typename(binding.dtype) and binding.name not in routine.arguments
                        and id(binding) not in self.confined_pointers):
                    self.barrier("implicit_call", routine, routine.scope.node,
                                 "local derived-object lifetime needs initialization/finalization authority")
            selection = tuple(_children(routine.execution))
            called = True
        else:
            called = False
        self.sequence(routine, selection, mapping, called=called)
        self.specification_authority(routine)
        self.active.pop()


def _template(analysis, procedure, selection):
    registry = getattr(analysis, "_native_effect_templates", None)
    if registry is None:
        registry = analysis._native_effect_templates = {}
    key = _template_key(analysis, procedure, selection)
    if key in registry:
        return registry[key].validate(analysis)
    from compiler.frontend.storage_mutation_effects import source_registry_budget
    if len(registry) >= source_registry_budget(analysis):
        raise ResourceReachabilityBudgetExceeded("native effect template registry budget exhausted")
    scan = _Scan(analysis, ())
    scan.routine(procedure, {}, selection)
    routine = scan.routines[procedure]
    root = NativeEffectNode("procedure" if selection is None else "selection",
                            routine.execution if selection is None else selection, tuple(scan.nodes))
    work = (("syntax_nodes", scan.visits), ("statements", scan.operations))
    identity = _digest({"version": VERSION, "source_identity": analysis.inputs.identity(),
                        "procedure": procedure, "selection": None if selection is None else list(map(id, selection)),
                        "source_work": dict(work), "nodes": root.public()})
    value = NativeEffectTemplate(procedure, identity, (root,), tuple(scan.template_calls), tuple(scan.terminal_calls), tuple(scan.completion),
        tuple(scan.confined_pointers.values()), work,
        analysis, routine, selection, tuple(scan.scopes.values()), tuple(scan.schemas.values()), _limits(analysis), _ISSUER,
        tuple(scan.worksharing.values()))
    registry[key] = value
    try:
        return value.validate(analysis)
    except BaseException:
        if registry.get(key) is value:
            del registry[key]
        raise


class _Instantiate:
    """Instantiate typed effects and call edges; never traverse callee syntax."""
    def __init__(self, analysis):
        self.analysis = analysis
        self.templates, self.instances, self.active = {}, [], []
        self.accesses, self.barriers, self.calls, self.completion = [], [], [], []
        self.routines, self.scopes, self.schemas = {}, {}, {}
        self.source_visits = self.source_operations = self.steps = 0
        self.deferred_completion, self.terminal_calls = [], []
        self.retirement_barriers, self.lifetime_obligations = [], []

    def tick(self):
        self.steps += 1
        if self.steps > NATIVE_GROUP_SCAN_LIMIT:
            raise ResourceReachabilityBudgetExceeded("resource reachability instantiation budget exhausted")

    def template(self, procedure, selection=None):
        value = _template(self.analysis, procedure, selection)
        if value.identity not in self.templates:
            self.templates[value.identity] = value
            self.routines[procedure] = value._routine
            if len(self.routines) > self.analysis.procedure_limit:
                raise ResourceReachabilityBudgetExceeded("resource reachability procedure budget exhausted")
            counts = dict(value.source_work)
            self.source_visits += counts["syntax_nodes"]
            self.source_operations += counts["statements"]
            # _Scan already enforces both source limits for this template.
            # Summed work describes the complete graph; reusing one template's
            # ceiling here creates an artificial whole-closure size window.
            # Distinct templates remain bounded by source procedures/calls,
            # and instantiation/substitution retains its independent budget.
            self.scopes.update((id(scope), scope) for scope in value._scopes)
            self.schemas.update((id(schema), schema) for schema in value._schemas)
            self.completion.extend(value.completion)
        return value

    def substitute(self, path, mapping):
        self.tick()
        if path is None or path is _ABSENT:
            return path
        replacement = mapping.get(id(path.binding))
        if replacement is None:
            return path
        _formal, storage = replacement
        if storage is None or storage is _ABSENT:
            return storage
        return StoragePath(storage.binding, (*storage.components, *path.components))

    def barrier(self, value):
        if value not in self.barriers:
            self.barriers.append(value)
        if value not in self.retirement_barriers:
            self.retirement_barriers.append(value)

    def access(self, item, mapping):
        storage = self.substitute(item.storage, mapping)
        if storage is _ABSENT:
            if item.kind != "association":
                self.barrier(ReachabilityBarrier("association", item.procedure, item.source,
                             "absent optional actual accessed outside its original PRESENT branch"))
            return
        if storage is None:
            # Expression values were evaluated by the original outer CALL.
            # Readonly forwarding needs no invented backing resource. An
            # eventual write or entry kill still makes the graph ineligible.
            if item.kind not in {"read", "descriptor", "association"}:
                self.barrier(ReachabilityBarrier("association", item.procedure, item.source,
                             "writable expression actual has no original storage"))
            return
        if "parameter" in storage.attributes:
            return
        if storage.attributes & {"pointer", "volatile", "asynchronous", "optional"}:
            self.barrier(ReachabilityBarrier("association", item.procedure, item.source,
                         "pointer, optional or asynchronous storage needs separate authority"))
        self.accesses.append(ResourceAccess(storage, item.kind, item.procedure, item.source,
                                            item.external, item.original))

    def items(self, items, instance, mapping):
        for item in items:
            self.tick()
            if isinstance(item, NativeEffectNode):
                self.items(item.items, instance, mapping)
            elif isinstance(item, ResourceAccess):
                self.access(item, mapping)
            elif isinstance(item, ReachabilityBarrier):
                self.barrier(item)
            elif isinstance(item, (NativeAllocationObligation, NativeWorksharingObligation)):
                self.lifetime_obligations.append((instance.ordinal, item))
                if isinstance(item, NativeAllocationObligation):
                    barrier = ReachabilityBarrier("allocation", item.procedure, str(item.target),
                        "whole allocatable assignment may reallocate original storage")
                else:
                    item.proof.checkpoint()
                    barrier = ReachabilityBarrier("completion", item.procedure, str(item.original),
                        "unowned original OpenMP directive")
                # Preserve the ordinary payload path's stronger eligibility.
                # Only an independent whole-call retirement proof may discharge
                # these typed obligations, with full storage and completion.
                if barrier not in self.barriers:
                    self.barriers.append(barrier)
            elif isinstance(item, NativeDeferredCompletion):
                self.deferred_completion.append((instance, item))
            elif isinstance(item, NativePresenceConditional):
                actual = mapping.get(id(item.binding))
                if actual is None or (isinstance(actual[1], StoragePath) and "optional" in actual[1].attributes):
                    self.barrier(ReachabilityBarrier("association", instance.template.procedure, str(item.original),
                                 "original optional presence requires an exact supplied or omitted call association"))
                    self.items(item.present, instance, mapping)
                    self.items(item.absent, instance, mapping)
                else:
                    self.items(item.absent if actual[1] is _ABSENT else item.present, instance, mapping)
            elif isinstance(item, NativeTerminatingCall):
                item.validate(self.analysis)
                if len(self.instances) - 1 + len(self.terminal_calls) >= self.analysis.operation_limit:
                    raise ResourceReachabilityBudgetExceeded("resource reachability call instantiation budget exhausted")
                callee = item.proof._callee
                self.routines[callee.qualified] = callee
                if len(self.routines) > self.analysis.procedure_limit:
                    raise ResourceReachabilityBudgetExceeded("resource reachability procedure budget exhausted")
                self.terminal_calls.append((instance.ordinal, item))
            elif isinstance(item, NativeTemplateCall):
                self.calls.append(item.edge)
                child_mapping = dict(mapping)
                for formal, path in item.associations:
                    actual = self.substitute(path, mapping)
                    child_mapping[id(formal)] = formal, actual
                    if actual is _ABSENT and "optional" not in formal.attributes:
                        self.barrier(ReachabilityBarrier("association", item.edge.caller, str(item.edge.original),
                                     "absent actual requires an original OPTIONAL callee formal"))
                    if isinstance(actual, StoragePath) and "optional" in formal.attributes:
                        terminal = actual.components[-1].attributes if actual.components else actual.binding.attributes
                        if terminal & {"allocatable", "pointer"}:
                            self.barrier(ReachabilityBarrier("association", item.edge.caller, str(item.edge.original),
                                         "optional allocatable/pointer actual needs descriptor-dependent presence authority"))
                    if isinstance(actual, StoragePath) and formal.intent == "out":
                        for action in ("undefine", "write"):
                            self.accesses.append(ResourceAccess(actual, action, item.edge.caller,
                                str(item.edge.original), True, item.edge.original))
                self.instance(item.edge.callee, child_mapping, parent=instance.ordinal, call=item)
            else:
                raise CompilationError("native effect template contains an unknown node")

    def instance(self, procedure, mapping, *, parent=None, call=None, selection=None):
        if procedure in self.active:
            raise ResourceReachabilityBudgetExceeded("resource reachability recursive call graph is unavailable")
        if len(self.active) >= self.analysis.depth_limit:
            raise ResourceReachabilityBudgetExceeded("resource reachability call depth budget exhausted")
        # The root selection is not a source CALL occurrence. Each subsequent
        # instance retains one exact original edge, even with shared templates.
        if call is not None and len(self.instances) - 1 + len(self.terminal_calls) >= self.analysis.operation_limit:
            raise ResourceReachabilityBudgetExceeded("resource reachability call instantiation budget exhausted")
        template = self.template(procedure, selection)
        instance = NativeCallInstance(len(self.instances), template, parent, call, tuple(mapping.values()))
        self.instances.append(instance)
        self.active.append(procedure)
        try:
            self.items(template.nodes, instance, mapping)
        finally:
            self.active.pop()

    def graph(self, procedure, selection):
        source_work = (("syntax_nodes", self.source_visits), ("statements", self.source_operations),
                       ("procedures", len(self.routines)), ("templates", len(self.templates)))
        instance_work = (("call_instances", len(self.instances) - 1 + len(self.terminal_calls)),
                         ("terminal_instances", len(self.terminal_calls)), ("substitution_steps", self.steps))
        identity = _digest({"version": VERSION, "source_identity": self.analysis.inputs.identity(),
            "procedure": procedure, "selection": list(map(id, selection)), "limits": _limits(self.analysis),
            "templates": list(self.templates), "instances": [item.public() for item in self.instances]})
        registry = getattr(self.analysis, "_native_effect_graphs", None)
        if registry is None:
            registry = self.analysis._native_effect_graphs = {}
        if identity in registry:
            return registry[identity].validate(self.analysis, procedure, selection)
        from compiler.frontend.storage_mutation_effects import source_registry_budget
        if len(registry) >= source_registry_budget(self.analysis):
            raise ResourceReachabilityBudgetExceeded("native effect graph registry budget exhausted")
        source_identity = json.dumps(self.analysis.inputs.identity(), sort_keys=True, separators=(",", ":"))
        deferred = tuple({(item.procedure, tuple(map(id, item.originals))): item
                          for _instance, item in self.deferred_completion}.values())
        value = NativeEffectGraph(identity, procedure, tuple(self.templates.values()), tuple(self.instances),
                                  source_work, instance_work, not self.barriers, deferred, tuple(self.terminal_calls), self.analysis, selection, source_identity,
                                  _limits(self.analysis), _ISSUER, not self.retirement_barriers,
                                  tuple(self.lifetime_obligations))
        registry[identity] = value
        try:
            return value.validate(self.analysis, procedure, selection)
        except BaseException:
            if registry.get(identity) is value:
                del registry[identity]
            raise


def _instantiate_native_graph(analysis, procedure, original_selection):
    if not isinstance(original_selection, tuple) or not original_selection:
        raise CompilationError("native effect graph requires an exact nonempty original tuple")
    analysis._selected_source(procedure, original_selection)
    scan = _Instantiate(analysis)
    scan.instance(procedure, {}, selection=original_selection)
    return scan, scan.graph(procedure, original_selection)


def prove_native_effect_graph(analysis, procedure, original_selection):
    """Issue the same source graph without inventing a storage demand.

    This grants no demand coverage, alias separation or native payload access.
    The existing joined-completion issuer must still check every obligation.
    """
    _scan, graph = _instantiate_native_graph(analysis, procedure, original_selection)
    return graph


def prove_resource_reachability(analysis, procedure, original_selection, demanded):
    """Issue conservative storage reachability for an exact original selection.

    Demands are exact original Binding objects (including scalar dependencies),
    never names copied from an effect report. All aliases remain obligations.
    Unique source statements and instantiated CALL occurrences each retain the
    existing operation bound; source-node and substitution/obligation work each
    retain the existing scan bound. Cached templates are authenticated and
    charged on every query. This replaces repeated callee syntax expansion.
    """
    if not isinstance(original_selection, tuple) or not original_selection:
        raise CompilationError("resource reachability requires an exact nonempty original tuple")
    if not isinstance(demanded, tuple) or not demanded or len({id(item) for item in demanded}) != len(demanded):
        raise CompilationError("resource reachability requires distinct exact demanded bindings")
    if len(demanded) > analysis.operation_limit:
        raise ResourceReachabilityBudgetExceeded("resource reachability demand budget exhausted")
    for binding in demanded:
        _authenticate_binding(analysis, binding)
    scan, graph = _instantiate_native_graph(analysis, procedure, original_selection)
    for obligation in graph.deferred_completion:
        try:
            from compiler.frontend.native_completion import prove_graph_joined_completion
            proof = prove_graph_joined_completion(analysis, obligation.procedure, obligation.originals, graph)
            scan.completion.append((obligation.procedure, proof.selected_node_ids, proof))
        except CompilationError as error:
            scan.barrier(ReachabilityBarrier("completion", obligation.procedure, str(obligation.originals),
                                           obligation.reason + "; native graph: " + str(error)))
    obligations = {}
    for access in scan.accesses:
        if not access.external or access.kind not in {"read", "write", "association", "undefine"}:
            continue
        for binding in demanded:
            scan.tick()
            if access.storage.binding is binding:
                continue
            key = id(access.storage.binding), tuple(id(item.declaration) for item in access.storage.components), id(binding)
            storage, actions = obligations.setdefault(key, (access.storage, set()))
            actions.add(access.kind)
    required = tuple(StorageDisjointnessObligation(storage, next(binding for binding in demanded if id(binding) == key[-1]),
                                                   tuple(sorted(actions)))
                     for key, (storage, actions) in obligations.items())
    work = (*graph.source_work, *graph.instantiation_work,
            ("substitution_and_obligation_steps", scan.steps))
    identity = _digest({"version": VERSION, "procedure": procedure,
        "sources": analysis.inputs.identity(), "selection": list(map(id, original_selection)),
        "demanded": list(map(id, demanded)), "accesses": [item.public() for item in scan.accesses],
        "barriers": [item.public() for item in scan.barriers], "work": dict(work)})
    registry = getattr(analysis, "_resource_reachability_proofs", None)
    if registry is None:
        registry = analysis._resource_reachability_proofs = {}
    if identity in registry:
        return registry[identity].validate(analysis, procedure, original_selection, demanded)
    from compiler.frontend.storage_mutation_effects import source_registry_budget
    if len(registry) >= source_registry_budget(analysis):
        raise ResourceReachabilityBudgetExceeded("resource reachability proof registry budget exhausted")
    proof = ResourceReachabilityProof(procedure, identity, tuple(scan.accesses), required, tuple(scan.barriers),
        tuple(scan.calls), graph, demanded, analysis, original_selection, tuple(scan.routines.items()), tuple(scan.scopes.values()),
        tuple(scan.schemas.values()), tuple(scan.completion),
        _limits(analysis), work, _ISSUER)
    registry[identity] = proof
    try:
        return proof.validate(analysis, procedure, original_selection, demanded)
    except BaseException:
        if registry.get(identity) is proof:
            del registry[identity]
        raise
