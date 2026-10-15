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

VERSION = 2


class ResourceReachabilityBudgetExceeded(CompilationError):
    """A bounded demand declined; this is not a source-authenticity failure."""


_ISSUER = object()
_INTRINSICS = frozenset(INTRINSICS) | ARRAY_INQUIRIES | MODEL_INQUIRIES | {
    "allocated", "present", "sum", "product", "any", "all", "count", "minval", "maxval", "dot_product"}


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
class ResourceReachabilityProof:
    procedure: str
    identity: str
    accesses: tuple[ResourceAccess, ...]
    obligations: tuple[StorageDisjointnessObligation, ...]
    barriers: tuple[ReachabilityBarrier, ...]
    calls: tuple[SourceCallEdge, ...]
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
                "whole_procedure_effects_complete": False, "execution_authorized": False,
                "capture_authorized": False, "definition_authorized": False, "alias_authorized": False,
                "scope": "exact original selection and statically resolved source callees; all original branches",
                "control": "unchanged original source; no predicates evaluated, hoisted or assumed"}

    def validate(self, analysis, procedure, original_selection, demanded):
        if (self._issuer is not _ISSUER or analysis is not self._analysis or procedure != self.procedure
                or not isinstance(original_selection, tuple) or len(original_selection) != len(self._selection)
                or any(left is not right for left, right in zip(original_selection, self._selection, strict=True))
                or not isinstance(demanded, tuple) or len(demanded) != len(self.demanded)
                or any(left is not right for left, right in zip(demanded, self.demanded, strict=True))
                or getattr(analysis, "_resource_reachability_proofs", {}).get(self.identity) is not self):
            raise CompilationError("resource reachability lacks exact registered source and demand authority")
        if self._limits != (analysis.depth_limit, analysis.procedure_limit, analysis.operation_limit, NATIVE_GROUP_SCAN_LIMIT):
            raise CompilationError("resource reachability proof budgets changed")
        analysis.verify_source()
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

    def tick(self, statement=False):
        self.visits += 1
        self.operations += int(statement)
        if self.visits > NATIVE_GROUP_SCAN_LIMIT or self.operations > self.analysis.operation_limit:
            raise ResourceReachabilityBudgetExceeded("resource reachability source scan budget exhausted")

    def barrier(self, kind, routine, node, reason):
        item = ReachabilityBarrier(kind, routine.qualified, str(node), reason)
        if item not in self.barriers:
            self.barriers.append(item)

    def scope(self, scope):
        pending = scope
        while pending is not None:
            if id(pending) not in self.scopes:
                _validate_scope(self.analysis, pending)
                self.scopes[id(pending)] = pending
            pending = pending.parent
        return scope

    def storage(self, routine, node, mapping):
        scope = self.scope(self.analysis.source_scope_for(node, routine.scope))
        parts = tuple(node.items) if _kind(node) == "Data_Ref" else (node,)
        base = parts[0].items[0] if _kind(parts[0]) == "Part_Ref" else parts[0]
        binding = self.analysis._binding(scope, base)
        if binding is None:
            raise CompilationError("unresolved original storage: " + str(base))
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

    def access(self, routine, node, mapping, action):
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
        if path.attributes & {"pointer", "volatile", "asynchronous", "optional"}:
            self.barrier("association", routine, node, "pointer, optional or asynchronous storage needs separate authority")
        local = (binding.declaring_scope is routine.scope and binding.name not in routine.arguments
                 and "save" not in binding.attributes and id(binding) not in mapping)
        self.accesses.append(ResourceAccess(path, action, routine.qualified, str(node), not local, node))
        if _typename(path.dtype) and action in {"read", "write"}:
            self.barrier("implicit_call", routine, node, "whole derived-object operation may invoke defined assignment or finalization")
        terminal_attributes = path.components[-1].attributes if path.components else path.binding.attributes
        whole = _kind(node) == "Name" or _kind(node) == "Data_Ref" and _kind(node.items[-1]) == "Name"
        if action == "write" and "allocatable" in terminal_attributes and whole:
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
        if kind in {"Name", "Data_Ref"} or kind == "Part_Ref" and self.analysis._binding(self.analysis.source_scope_for(node, routine.scope), node.items[0]) is not None:
            self.access(routine, node, mapping, action)
        elif kind in {"Intrinsic_Function_Reference", "Function_Reference", "Part_Ref", "Structure_Constructor"}:
            name, args = node.items
            scope = self.scope(self.analysis.source_scope_for(node, routine.scope))
            spelling = str(name).lower()
            if (spelling in _INTRINSICS and not self.analysis._binding(scope, name)
                    and not self.analysis._candidates(scope, name) and not self.analysis._unknown_exports(scope)):
                for index, argument in enumerate(_children(args)):
                    if _kind(argument) == "Actual_Arg_Spec":
                        argument = argument.items[1]
                    inquiry = index == 0 and spelling in ARRAY_INQUIRIES | MODEL_INQUIRIES | {"allocated", "present"}
                    self.expression(routine, argument, mapping, "descriptor" if inquiry else "read")
            else:
                self.call(routine, node, mapping, function=True)
        else:
            if ("Defined" in kind or kind in {"Level_5_Expr", "Coindexed_Named_Object"}
                    or kind not in {"list", "tuple", "Parenthesis", "Level_1_Expr", "Level_2_Expr",
                        "Level_2_Unary_Expr", "Level_3_Expr", "Level_4_Expr", "Add_Operand", "Mult_Operand",
                        "And_Operand", "Or_Operand", "Equiv_Operand", "Subscript_Triplet", "Loop_Control",
                        "Explicit_Shape_Spec", "Assumed_Shape_Spec", "Assumed_Size_Spec", "Deferred_Shape_Spec",
                        "Array_Constructor", "Ac_Value_List", "Section_Subscript_List", "Return_Stmt"}):
                self.barrier("expression", routine, node, "expression requires separate implicit-call or storage authority")
            for child in _children(node):
                self.expression(routine, child, mapping, action)

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
        try:
            resolved = (resolve_original_function if function else resolve_original_call)(self.analysis, scope, node)
        except OriginalCallBudgetExceeded as error:
            raise ResourceReachabilityBudgetExceeded(str(error)) from error
        except CompilationError as error:
            self.barrier("call", routine, node, str(error))
            return
        if resolved.procedure.startswith("$external::"):
            self.barrier("external_call", routine, node, "external linkage/interposition needs independent original call authority")
            return
        self.calls.append(SourceCallEdge(routine.qualified, resolved.procedure, node, resolved))
        child_mapping = dict(mapping)
        for item in resolved.associations:
            self.scope(item.formal_binding.declaring_scope)
            if item.actual is None:
                self.barrier("association", routine, node, "omitted optional actual needs separate presence authority")
                continue
            if item.formal_binding.attributes & {"pointer", "optional", "allocatable", "value"}:
                self.barrier("association", routine, node, "callee descriptor/presence association needs separate authority")
            if item.binding is not None:
                try:
                    path, _binding = self.storage(routine, item.actual, mapping)
                except CompilationError as error:
                    self.barrier("association", routine, item.actual, str(error))
                    continue
                child_mapping[id(item.formal_binding)] = path
                self.access(routine, item.actual, mapping, "association")
                if item.formal_binding.intent == "out":
                    self.access(routine, item.actual, mapping, "undefine")
                    self.access(routine, item.actual, mapping, "write")
            else:
                child_mapping[id(item.formal_binding)] = None
                self.expression(routine, item.actual, mapping)
        self.routine(resolved.procedure, child_mapping)

    def sequence(self, routine, nodes, mapping, *, called, depth=0):
        if depth > self.analysis.depth_limit:
            raise ResourceReachabilityBudgetExceeded("resource reachability control depth budget exhausted")
        peeled = {}
        for item in grouped_original_sequence(nodes, peeled):
            if isinstance(item, NativeGroupCandidate):
                originals = item.original_nodes
                try:
                    proof = self.analysis.joined_completion(routine.qualified, originals)
                    self.completion.append((routine.qualified, proof.selected_node_ids, proof))
                except CompilationError as error:
                    self.barrier("completion", routine, originals, str(error))
                for node in originals:
                    if _kind(node) != "Comment":
                        self.statement(routine, node, mapping, called=called, depth=depth + 1,
                                       ignore_directives=True)
            else:
                self.statement(routine, item, mapping, called=called, depth=depth)

    def statement(self, routine, node, mapping, *, called, depth=0, ignore_directives=False):
        self.tick(statement=_kind(node).endswith("_Stmt"))
        kind = _kind(node)
        if kind == "Comment":
            if str(node).lstrip().lower().startswith("!$omp") and not ignore_directives:
                self.barrier("completion", routine, node, "unowned original OpenMP directive")
        elif kind == "Assignment_Stmt":
            self.access(routine, node.items[0], mapping, "write")
            self.expression(routine, node.items[2], mapping)
        elif kind == "Call_Stmt":
            self.call(routine, node, mapping)
        elif kind == "If_Stmt":
            self.expression(routine, node.items[0], mapping)
            self.statement(routine, node.items[1], mapping, called=called, depth=depth + 1)
        elif kind in {"If_Construct", "Block_Nonlabel_Do_Construct", "Case_Construct"}:
            children = tuple(child for child in _children(node)
                             if not (ignore_directives and _kind(child) == "Comment"
                                     and str(child).lstrip().lower().startswith("!$omp")))
            self.sequence(routine, children, mapping, called=called, depth=depth + 1)
        elif kind in {"If_Then_Stmt", "Else_If_Stmt", "Nonlabel_Do_Stmt", "Select_Case_Stmt", "Case_Stmt"}:
            if kind == "Nonlabel_Do_Stmt":
                control = next((part for part in _children(node) if _kind(part) == "Loop_Control"), None)
                if control is None or control.items[1] is None:
                    self.barrier("control", routine, node, "non-counted loop needs separate completion authority")
                else:
                    self.access(routine, control.items[1][0], mapping, "write")
            for child in _children(node):
                self.expression(routine, child, mapping)
        elif kind in {"Else_Stmt", "End_If_Stmt", "End_Do_Stmt", "End_Select_Stmt", "Continue_Stmt"}:
            pass
        elif kind == "Return_Stmt" and called:
            self.expression(routine, node, mapping)
        elif kind in {"Allocate_Stmt", "Deallocate_Stmt", "Pointer_Assignment_Stmt"}:
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
                for shape in binding.shape_nodes:
                    self.expression(routine, shape, mapping)
                if _typename(binding.dtype) and binding.name not in routine.arguments:
                    self.barrier("implicit_call", routine, routine.scope.node,
                                 "local derived-object lifetime needs initialization/finalization authority")
            selection = tuple(_children(routine.execution))
            called = True
        else:
            called = False
        self.sequence(routine, selection, mapping, called=called)
        self.specification_authority(routine)
        self.active.pop()


def prove_resource_reachability(analysis, procedure, original_selection, demanded):
    """Issue conservative storage reachability for an exact original selection.

    Demands are exact original Binding objects (including scalar dependencies),
    never names copied from an effect report. All aliases remain obligations.
    """
    if not isinstance(original_selection, tuple) or not original_selection:
        raise CompilationError("resource reachability requires an exact nonempty original tuple")
    if not isinstance(demanded, tuple) or not demanded or len({id(item) for item in demanded}) != len(demanded):
        raise CompilationError("resource reachability requires distinct exact demanded bindings")
    if len(demanded) > analysis.operation_limit:
        raise ResourceReachabilityBudgetExceeded("resource reachability demand budget exhausted")
    for binding in demanded:
        _authenticate_binding(analysis, binding)
    analysis._selected_source(procedure, original_selection)
    scan = _Scan(analysis, demanded)
    scan.routine(procedure, {}, original_selection)
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
    work = (("syntax_nodes", scan.visits), ("statements", scan.operations), ("procedures", len(scan.routines)))
    identity = _digest({"version": VERSION, "procedure": procedure,
        "sources": analysis.inputs.identity(), "selection": list(map(id, original_selection)),
        "demanded": list(map(id, demanded)), "accesses": [item.public() for item in scan.accesses],
        "barriers": [item.public() for item in scan.barriers], "work": dict(work)})
    registry = getattr(analysis, "_resource_reachability_proofs", None)
    if registry is None:
        registry = analysis._resource_reachability_proofs = {}
    if identity in registry:
        return registry[identity].validate(analysis, procedure, original_selection, demanded)
    if len(registry) >= analysis.procedure_limit:
        raise ResourceReachabilityBudgetExceeded("resource reachability proof registry budget exhausted")
    proof = ResourceReachabilityProof(procedure, identity, tuple(scan.accesses), required, tuple(scan.barriers),
        tuple(scan.calls), demanded, analysis, original_selection, tuple(scan.routines.items()), tuple(scan.scopes.values()),
        tuple(scan.schemas.values()), tuple(scan.completion),
        (analysis.depth_limit, analysis.procedure_limit, analysis.operation_limit, NATIVE_GROUP_SCAN_LIMIT), work, _ISSUER)
    registry[identity] = proof
    try:
        return proof.validate(analysis, procedure, original_selection, demanded)
    except BaseException:
        if registry.get(identity) is proof:
            del registry[identity]
        raise
