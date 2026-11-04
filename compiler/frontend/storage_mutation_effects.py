"""Original-source descriptor mutation projection, independent of payload work.

This is the storage component of source effects, not a native execution summary.
Exact original associations project formal descriptor events into caller roots;
module/host roots remain visible even when a call also has ordinary array dummies.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from hashlib import sha256

from fparser.two.utils import Base

from compiler.frontend.call_bindings import (
    _fixed_array_bounds,
    _original_call_control,
    _OriginalTypes,
    _resolve_candidates,
    resolve_original_call,
    resolve_original_function,
)
from compiler.frontend.descriptor_budget import DEFAULT_SOURCE_SCAN_LIMIT, SOURCE_SYNTAX_PASSES, _syntax_inventory
from compiler.frontend.source_functions import (
    _hash,
    _original,
    _scope_roles,
    _validate_resolution,
    _validate_scope,
    _validate_scope_roles,
)
from compiler.frontend.structured_effects import _freeze_source_record, _thaw
from compiler.ir import CompilationError
from compiler.ir.intrinsics import ARRAY_INQUIRIES, INTRINSICS, MODEL_INQUIRIES

_ISSUER = object()
_UNSET = object()
STORAGE_ENCLOSING_LEXICAL_CONTEXT_VERSION = 8
STORAGE_FIXED_ARRAY_RESULT_VERSION = 11
STORAGE_TERMINATING_CALL_VERSION = 12
STORAGE_SELECT_CASE_VERSION = 13
STORAGE_READONLY_RECORD_VERSION = 14
STORAGE_ASSIGNMENT_APPLICABILITY_VERSION = 15
STORAGE_RECORD_ARRAY_READ_VERSION = 18
STORAGE_NATIVE_CALL_STRUCTURES_VERSION = 19
STORAGE_NATIVE_LIFETIME_CONTEXT_VERSION = 20
_UNSAFE = {"pointer", "target", "volatile", "asynchronous", "optional", "value"}
_ALIASES = {"Common_Stmt", "Equivalence_Stmt", "Namelist_Stmt", "Cray_Pointer_Stmt", "Bind_Stmt", "Codimension_Attr_Spec",
            "Procedure_Declaration_Stmt", "Proc_Declaration_Stmt"}
_INTRINSICS = set(INTRINSICS) | ARRAY_INQUIRIES | MODEL_INQUIRIES | {
    "allocated", "present", "shape", "sum", "product", "any", "all", "count", "minval", "maxval",
    "norm2", "dot_product", "matmul", "transpose", "minloc", "maxloc", "reshape", "spread", "cshift", "eoshift"}


def _kind(node):
    return type(node).__name__


def _children(node):
    return node if isinstance(node, (tuple, list)) else getattr(node, "content", getattr(node, "items", ())) or ()


def source_registry_budget(analysis):
    """Storage token work is bounded by source size, not retained task count."""
    session = getattr(analysis, "_source_session", None)
    if session is not None:
        # The immutable session already indexes exact syntax and declarations.
        # Its complete object inventory is an O(1) conservative upper bound.
        syntax = len(session.objects)
    else:
        syntax, declarations = _syntax_inventory(tuple(routine.scope.node
            for routine in analysis.numerical_helpers.values()))
        syntax += declarations
    return max(DEFAULT_SOURCE_SCAN_LIMIT, (SOURCE_SYNTAX_PASSES + analysis.operation_limit) * max(1, syntax))


def _preserves_record(record):
    """Pure arithmetic on an already authenticated projection, not authority."""
    private = set(record["private_local_resources"])
    return not record["frontiers"] and not any(event["resource"] not in private
        for bucket in ("mutations", "escapes") for event in record[bucket])


def _part(node, kind):
    return next((child for child in getattr(node, "content", ()) if _kind(child) == kind), None)


def _resource(binding):
    # Dummy spellings are intentionally reused by SourceEffects. A sibling
    # internal procedure can access its host's dummy through a second route;
    # that storage must never be rebound as an unrelated child's same-named
    # dummy when composing a call. Only exact declared-formal keys are mapped.
    if binding.root.startswith("argument::") and hasattr(binding.declaring_scope, "qualified"):
        return binding.declaring_scope.qualified + "::" + binding.root
    return binding.root


def _consumed_call_receipt(resolved):
    """Exact already-resolved edges, checked again at the publication fence."""
    return (id(resolved.callee), id(resolved._caller), resolved.procedure, resolved.formals,
            resolved.target, resolved.source_kind, tuple(id(node) for node in resolved.actuals),
            tuple((id(item.actual), id(item.formal_binding), id(item.binding), _hash(item.public()))
                  for item in resolved.associations),
            tuple((id(item.actual), _hash(item.public())) for item in resolved.original_arguments),
            _hash(resolved.public()))


class _Scan:
    def __init__(self, analysis):
        self.analysis = analysis
        from compiler.frontend.descriptor_intrinsics import intrinsic_source_attachments
        self.source_receipt = intrinsic_source_attachments(analysis)
        self.syntax_receipts = {}
        from compiler.frontend.source_dependency_receipts import DependencyCapture
        self.declaration_dependencies = DependencyCapture(analysis, components=True)
        self.readonly_records, self.readonly_associations = {}, []
        self.record_array_reads = []
        self.component_actuals = []
        self.inapplicable_assignments = []
        self.assignment_namespace_checked = False
        self.in_specification = False
        self.roles, self.scopes = _scope_roles(analysis), {}
        self.members, self.active, self.inventories = {}, [], {}
        self.declarations = {}
        self.call_receipts = []
        self.terminating_calls = {}
        self.terminating_receipts = {}
        self.library_calls = {}
        self.fixed_associations, self.association_stack = {}, []
        self.fixed_blocks = {}
        self.fixed_array_results = {}
        self.fixed_array_result_receipts = []
        self.select_cases = []
        self.joined_storage = None
        self.joined_routine = None
        self.joined_directives = {}
        self.joined_nodes = {}
        self.joined_closure = None
        self.enclosing_context = None
        self.record_region = None
        self.native_lifetime = None
        self.record_region_attached = None
        self.record_scopes, self.record_declarations = {}, {}
        self.component_scope_checks = set()
        self.component_session = getattr(analysis, "_source_session", None)
        self.visits, self.allowance = 0, 0
        self.limits = (analysis.depth_limit, analysis.procedure_limit, analysis.operation_limit)

    def tick(self, _role=None):
        self.visits += 1
        if self.visits > max(DEFAULT_SOURCE_SCAN_LIMIT, self.allowance):
            raise CompilationError("storage mutation original-source traversal budget exhausted")

    def walk(self, root):
        pending = [root]
        while pending:
            node = pending.pop()
            if not isinstance(node, (Base, tuple, list)):
                continue
            self.tick()
            if isinstance(node, Base):
                yield node
            pending.extend(reversed(_children(node)))

    def scope(self, scope):
        if scope is not None and id(scope) not in self.scopes:
            from compiler.frontend.descriptor_intrinsics import intrinsic_scope_attachments
            self.record_scopes[id(scope)] = intrinsic_scope_attachments(scope, syntax=False)
            _validate_scope(self.analysis, scope)
            self.scopes[id(scope)] = scope

    def resolve(self, routine, node):
        scope = self.analysis.source_scope_for(node, routine.scope)
        _validate_resolution(self.analysis, scope, self.walk(node), tick=self.tick, _scope_validator=self.scope)
        return scope

    def inventory(self, routine):
        if routine.qualified in self.inventories:
            return
        from compiler.frontend.descriptor_teams import _group_syntax_checkpoint
        self.syntax_receipts[routine.qualified] = _group_syntax_checkpoint((routine.scope.node,))
        syntax, declarations = _syntax_inventory((_part(routine.scope.node, "Subroutine_Stmt"),
            _part(routine.scope.node, "Function_Stmt"), _part(routine.scope.node, "Specification_Part"), routine.execution))
        # The same source-sized formula as descriptor demands, with an upper
        # bound of one semantic/resolution unit per original parser node.
        allowance = SOURCE_SYNTAX_PASSES*syntax + self.analysis.operation_limit*(syntax + declarations)
        self.inventories[routine.qualified] = {"procedure": routine.qualified, "unique_original_parser_nodes": syntax,
            "original_declaration_units": declarations, "per_unit_lookup_limit": self.analysis.operation_limit,
            "allowance": allowance}
        self.allowance += allowance

    def expression_path(self, routine, expression):
        pending, found = [(routine.execution, ())], []
        while pending:
            node, path = pending.pop()
            if not isinstance(node, (Base, tuple, list)):
                continue
            self.tick()
            if node is expression:
                found.append(path)
            pending.extend((child, (*path, index)) for index, child in reversed(tuple(enumerate(_children(node)))))
        if len(found) != 1:
            raise CompilationError("storage function query requires its exact unique original executable expression")
        return list(found[0])

    @staticmethod
    def result(routine):
        return {"procedure": routine.qualified, "body_sha256": _hash(str(routine.scope.node)),
                **{key: [] for key in ("mutations", "escapes", "frontiers", "definition_kills", "observables", "calls", "resources", "private_local_resources")}}

    @staticmethod
    def event(result, bucket, kind, node, resource="*"):
        event = {"kind": kind, "resource": resource, "original_source": str(node)}
        if event not in result[bucket]:
            result[bucket].append(event)

    def binding(self, result, binding, node):
        if binding is not None and id(binding) not in self.record_declarations:
            from compiler.frontend.descriptor_associations import _attachments
            self.record_declarations[id(binding)] = binding, _attachments(binding)
        if binding is None:
            self.event(result, "frontiers", "unresolved original storage", node)
            return None
        lexical = self.lexical_alias(binding)
        if lexical is not None:
            proof, access = lexical
            if binding.dtype not in {"real", "integer", "logical"} or binding.kind not in {4, 8}:
                self.event(result, "frontiers", "whole lexical object requires separate storage authority", node)
                return binding
            root = access.root_binding
            self.scope(root.declaring_scope)
            for declaration in self.walk(_part(root.declaring_scope.node, "Specification_Part")):
                if _kind(declaration) in _ALIASES:
                    self.event(result, "frontiers", "unproved declaring storage association", declaration)
            reason = self.analysis.resource_identity_boundary(root)
            if reason:
                self.event(result, "frontiers", reason, node)
            if access.binding.root not in result["resources"]:
                result["resources"].append(access.binding.root)
            return binding
        self.scope(binding.declaring_scope)
        if binding.declaring_scope.bindings.get(binding.name) is not binding:
            raise CompilationError("storage mutation original binding authority changed")
        if self.record_region is not None and any(binding is item for item in self.record_region._roles[3:5]):
            if self.record_region._roles[2].root not in result["resources"]:
                result["resources"].append(self.record_region._roles[2].root)
            return binding
        if self.native_lifetime is not None and self.native_lifetime.binding(binding):
            if _resource(binding) not in result["resources"]:
                result["resources"].append(_resource(binding))
            return binding
        from compiler.frontend.component_bindings import _typename
        if _typename(binding.dtype) is not None and binding.intent == "in":
            try:
                self.readonly_record(binding)
            except CompilationError as error:
                self.event(result, "frontiers", "unproved original readonly record declaration: " + str(error), node)
            else:
                if _resource(binding) not in result["resources"]:
                    result["resources"].append(_resource(binding))
                return binding
        unsafe = binding.attributes & _UNSAFE
        if "optional" in unsafe:
            # Presence controls whether the unchanged original program may
            # evaluate a dummy. It does not turn an ordinary numeric touch
            # into allocation mutation or address export. This projection
            # never evaluates that descriptor or grants payload/presence
            # authority; original associations and every closure statement
            # still receive their ordinary checks below.
            name = getattr(binding.declaring_scope, "qualified", None)
            if name in self.analysis.numerical_helpers:
                routine = _original(self.analysis, name)
                if routine.scope is binding.declaring_scope and binding.name in routine.arguments:
                    unsafe -= {"optional"}
        if binding.dtype not in {"real", "integer", "logical"} or unsafe:
            self.event(result, "frontiers", "unproved type/association/escape", node)
        reason = self.analysis.resource_identity_boundary(binding)
        if reason:
            self.event(result, "frontiers", reason, node)
        if _resource(binding) not in result["resources"]:
            result["resources"].append(_resource(binding))
        # Hidden module and host aliases must retain their exact declarations.
        key = id(binding.declaring_scope)
        if key not in self.declarations:
            self.declarations[key] = tuple(declaration for declaration in
                self.walk(_part(binding.declaring_scope.node, "Specification_Part")) if _kind(declaration) in _ALIASES)
        for declaration in self.declarations[key]:
            self.event(result, "frontiers", "unproved declaring storage association", declaration)
        return binding

    def readonly_scope(self, scope):
        self.declaration_dependencies.retain_scope(scope)
        self.scope(scope)

    def component_scope(self, scope, check):
        """Share a completed static check only within this immutable source demand."""
        self.readonly_scope(scope)
        session = getattr(self.analysis, "_source_session", None)
        if session is not self.component_session:
            raise CompilationError("storage component original source session changed")
        if session is None:
            check()
            return
        session.verify(self.analysis)
        if not session.contains(scope):
            raise CompilationError("storage component declaration scope is outside original source")
        if id(scope) not in self.component_scope_checks:
            check()
            # DependencyCapture retains the complete specification and schema
            # before this walk; publication must still require those receipts.
            self.component_scope_checks.add(id(scope))

    def assignment_inapplicable(self, routine, scope, node, result):
        """Exclude only concrete record LHS dummies from primitive assignment.

        This is type applicability, never a defined-assignment effect summary.
        A possible match, polymorphism or an open namespace stays unavailable.
        """
        from compiler.frontend.component_bindings import _typename
        from compiler.frontend.descriptor_associations import _attached, _attachments
        from compiler.frontend.source_functions import _resolution, original_name_absent

        target = node.items[0]
        excluded = {}
        scanner = self

        class Types(_OriginalTypes):
            def scope(self, consulted):
                scanner.readonly_scope(consulted)
                return super().scope(consulted)

        try:
            types = Types(self.analysis)
            supplied = None

            def incompatible(owner, spelling):
                nonlocal supplied
                # No type inspection is needed when the closed namespace has
                # no generic definition. A visible definition must be excluded
                # by this exact original LHS before intrinsic handling proceeds.
                if supplied is None:
                    if _kind(target) not in {"Name", "Part_Ref"}:
                        return False
                    supplied = types.actual(scope, target)
                    binding = supplied.binding
                    if supplied.dtype not in {"real", "integer", "logical"} or binding is None:
                        return False
                    declaring = binding.declaring_scope
                    original = self.analysis._source_function_scopes.get(id(declaring))
                    if (original is None or original[4].get(binding.name) != (binding, declaring)
                            or _resolution(declaring) != original[3]
                            or (supplied.dtype, supplied.kind) != (binding.dtype, binding.kind)):
                        raise CompilationError("assignment target requires its unchanged original primitive declaration")
                # Existing original namespace resolution supplies these exact
                # specifics. Retain declarations before delegated verification.
                candidates = self.analysis._candidates(owner, spelling)
                if not candidates:
                    return False
                for name in candidates:
                    self.tick()
                    candidate = self.analysis.routines.get(name)
                    if candidate is None:
                        return False
                    self.declaration_dependencies.retain_routine(candidate)
                    self.readonly_scope(candidate.scope)
                    arguments = candidate.arguments
                    if _original(self.analysis, name) is not candidate or candidate.arguments is not arguments:
                        raise CompilationError("assignment specific requires its original procedure declaration")
                    if len(arguments) != 2:
                        return False
                    formal = candidate.scope.bindings.get(arguments[0])
                    if (formal is None or _typename(formal.dtype) is None
                            or formal.intent not in {"out", "inout"}):
                        return False
                    attached = _attachments(formal)
                    declared = types.types.binding(formal)
                    if not _attached(formal, attached):
                        raise CompilationError("assignment specific declaration changed during type inspection")
                    if declared.reason:
                        return False
                    excluded[name] = {"procedure": name, "original_lhs_type": formal.dtype,
                        "declaring_type": getattr(declared.scope, "qualified", declared.scope.module) + "::" + declared.name}
                return True

            self.assignment_namespace_checked = True
            absent = original_name_absent(self.analysis, scope, "assignment(=)",
                _scope_validator=self.readonly_scope, _generic_inapplicable=incompatible)
            if not absent:
                return False
            if not excluded:
                return True
            self.inapplicable_assignments.append({"procedure": routine.qualified, "original_source": str(node),
                "actual_lhs_type": supplied.dtype, "excluded_specifics": [excluded[key] for key in sorted(excluded)],
                "reason": "concrete derived LHS dummy cannot accept original primitive LHS",
                "payload_access_authorized": False, "execution_authorized": False})
            return True
        except CompilationError as error:
            self.event(result, "frontiers", "unproved assignment applicability: " + str(error), node)
            return False

    def readonly_record(self, binding, *, formal=True):
        from compiler.frontend.storage_component_reads import readonly_record_declaration

        key = id(binding), formal
        if key not in self.readonly_records:
            context = self.native_lifetime if (not formal and self.native_lifetime is not None
                                              and self.native_lifetime.group_only) else None
            schema = readonly_record_declaration(self.analysis, binding,
                retain_scope=self.readonly_scope, tick=self.tick, formal=formal, _storage_context=context)
            self.readonly_records[key] = schema
        return self.readonly_records[key]

    def readonly_record_association(self, routine, resolved, item, result):
        """Original type association plus complete child scan, never noescape by INTENT."""
        from compiler.frontend.component_bindings import _typename

        formal, actual = item.formal_binding, item.binding
        if _typename(formal.dtype) is None:
            return False
        try:
            expected = self.readonly_record(formal)
            if actual is None or _kind(item.actual) not in {"Name", "Part_Ref"}:
                raise CompilationError("readonly record actual must be an original scalar variable or array element")
            supplied = self.readonly_record(actual, formal=False)
            if supplied is not expected:
                raise CompilationError("readonly record requires its exact canonical declaring type")
            if _kind(item.actual) == "Name" and actual.rank:
                raise CompilationError("readonly record actual must select one original scalar element")
            if _kind(item.actual) == "Part_Ref":
                scope = self.analysis.source_scope_for(item.actual, routine.scope)
                types = _OriginalTypes(self.analysis)
                if not actual.rank or types.indices(scope, item.actual, actual.rank, 0):
                    raise CompilationError("readonly record actual requires scalar original subscripts")
                for consulted in types.scopes.values():
                    self.readonly_scope(consulted)
                for _ in range(types.visits + types.types.visits):
                    self.tick()
                for index in _children(item.actual.items[1]):
                    self.expression(routine, index, result)
        except CompilationError as error:
            self.event(result, "frontiers", "unproved original readonly record association: " + str(error), item.actual)
            return False
        for binding in (formal, actual):
            if _resource(binding) not in result["resources"]:
                result["resources"].append(_resource(binding))
        self.readonly_associations.append({"caller": routine.qualified, "callee": resolved.procedure,
            "formal": formal.name, "actual": str(item.actual), "actual_resource": _resource(actual),
            "declaring_type": getattr(expected.scope, "qualified", expected.scope.module) + "::" + expected.name,
            "association": "exact original scalar INTENT(IN) record variable; complete original callee effects retained",
            "allocation_or_definition_inferred": False, "payload_access_authorized": False,
            "execution_authorized": False, "alias_authority": False})
        return True

    def numeric_component_association(self, routine, resolved, item, result):
        """Classify one fixed field actual; the caller still scans the complete callee."""
        if (_kind(resolved._original) != "Call_Stmt" or _kind(item.actual) != "Data_Ref"
                or _kind(item.actual.items[0]) != "Part_Ref" or item.binding is None
                or item.binding.attributes & _UNSAFE):
            return None
        from compiler.frontend.storage_component_reads import numeric_component_actual

        try:
            self.readonly_record(item.binding, formal=False)
            component = numeric_component_actual(self.analysis, routine, resolved, item,
                visit_index=lambda index: self.expression(routine, index, result),
                visit_scope=self.readonly_scope, tick=self.tick, _scope_scan=self.component_scope)
        except CompilationError as error:
            self.event(result, "frontiers", "unproved original numeric component actual: " + str(error), item.actual)
            return None
        resource = _resource(component.root_binding) + component.resource[len(component.root_binding.root):]
        self.component_actuals.append({"caller": routine.qualified, **component.public(), "resource": resource})
        if resource not in result["resources"]:
            result["resources"].append(resource)
        return resource

    def lexical_alias(self, binding):
        for proof in reversed(self.association_stack):
            declaration = proof._association
            for name, access in zip(declaration.scope.bindings, declaration.selectors, strict=True):
                if declaration.scope.bindings[name] is binding:
                    return proof, access
        return None

    def record_array_inquiry(self, routine, node, result):
        """Classify exact intrinsic descriptor reads, without reading elements."""
        from compiler.frontend.component_bindings import _typename
        from compiler.frontend.storage_component_reads import record_array_declaration

        name = str(node.items[0]).lower()
        if name not in {"size", "lbound", "ubound", "shape"}:
            return False
        formals = ("array", "kind") if name == "shape" else ("array", "dim", "kind")
        arguments = {}
        keyword_seen = False
        for ordinal, actual in enumerate(_children(node.items[1])):
            if _kind(actual) == "Actual_Arg_Spec":
                keyword_seen = True
                keyword, actual = str(actual.items[0]).lower(), actual.items[1]
            elif not keyword_seen and ordinal < len(formals):
                keyword = formals[ordinal]
            else:
                return False
            if keyword not in formals or keyword in arguments:
                return False
            arguments[keyword] = actual
        array = arguments.get("array")
        if _kind(array) != "Name":
            return False
        scope = self.analysis.source_scope_for(node, routine.scope)
        binding = self.analysis._binding(scope, array)
        context = self.native_lifetime if self.native_lifetime is not None and self.native_lifetime.group_only else None
        target_read = context is not None and context.readonly_target(binding)
        if (binding is None or not binding.rank or _typename(binding.dtype) is None
                or binding.attributes & (_UNSAFE - {"target"} if target_read else _UNSAFE)):
            # Existing exact owned-region authority may already classify an
            # associated root. Ordinary binding() retains that decision; this
            # new unassociated-array slice cannot replace or relax it.
            return False
        # Every selector still traverses the ordinary complete effects scanner.
        # An unknown or mutating selector cannot hide behind a descriptor read.
        for keyword, actual in arguments.items():
            if keyword != "array":
                self.expression(routine, actual, result)
        try:
            if target_read:
                context.require_target_read(routine,array,binding,descriptor=True)
            self.readonly_record(binding, formal=False)
            record_array_declaration(self.analysis, binding, retain_scope=self.readonly_scope, tick=self.tick,
                                     _storage_context=context)
            types = _OriginalTypes(self.analysis)
            for keyword, actual in arguments.items():
                if keyword != "array":
                    typed = types.actual(scope, actual)
                    if typed.dtype != "integer" or typed.rank:
                        raise CompilationError("record array inquiry requires scalar INTEGER selectors")
            for consulted in types.scopes.values():
                self.readonly_scope(consulted)
            for _ in range(types.visits + types.types.visits):
                self.tick()
        except CompilationError as error:
            self.event(result, "frontiers", "unproved original record array inquiry: " + str(error), node)
            return True
        if _resource(binding) not in result["resources"]:
            result["resources"].append(_resource(binding))
        self.record_array_reads.append({"kind": "original_record_array_inquiry", "procedure": routine.qualified,
            "source_expression": str(node), "resource": _resource(binding), "intrinsic": name,
            "allocation_or_definition_inferred": False, "payload_access_authorized": False,
            "execution_authorized": False, "alias_authority": False})
        return True

    def native_field(self, routine, node, result):
        path = self.native_lifetime.field(routine, node) if self.native_lifetime is not None else None
        if path is not None:
            self.binding(result, path.binding, node)
            resource = _resource(path.binding) + "".join("%" + item.name for item in path.components)
            if resource not in result["resources"]:
                result["resources"].append(resource)
            # Exact graph occurrence owns the field; still inspect every index.
            for part in node.items:
                if _kind(part) == "Part_Ref":
                    self.expression(routine, part.items[1], result)
            return resource
        return None

    def expression(self, routine, node, result):
        if node is None or isinstance(node, (str, int)):
            return
        self.tick()
        kind = _kind(node)
        scope = self.analysis.source_scope_for(node, routine.scope)
        if kind == "Data_Ref":
            if self.native_field(routine, node, result) is not None:
                return
            if self.record_region is not None and any(node is item for item in self.record_region._roles[7]):
                self.binding(result, self.record_region._roles[4], node)
                return
            from compiler.frontend.storage_component_reads import numeric_component_read

            try:
                head = node.items[0]
                array_element = _kind(head) == "Part_Ref"
                root = self.analysis._binding(scope, head.items[0] if array_element else head)
                lexical = self.lexical_alias(root)
                if array_element:
                    self.readonly_record(root, formal=False)
                elif root is not None and root.intent == "in" and root.dtype not in {"real", "integer", "logical"}:
                    self.readonly_record(root)
                component = numeric_component_read(self.analysis, routine, node,
                    visit_index=lambda index: self.expression(routine, index, result),
                    visit_scope=self.scope, tick=self.tick,
                    lexical_alias=None if lexical is None else lexical[0],
                    readonly_record=root if not array_element and (id(root), True) in self.readonly_records else None,
                    _scope_scan=self.component_scope,
                    _storage_context=self.native_lifetime if (self.native_lifetime is not None
                        and self.native_lifetime.group_only) else None)
            except CompilationError as error:
                self.event(result, "frontiers", "unproved original component read: " + str(error), node)
                return
            # The declaration-only helper does not produce a payload Binding.
            # In particular, passing its derived root through binding() would
            # conflate lifetime classification with generic object capture.
            resource = component.resource
            if (id(component.root_binding), True) in self.readonly_records:
                resource = _resource(component.root_binding) + resource[len(component.root_binding.root):]
            if resource not in result["resources"]:
                result["resources"].append(resource)
            if array_element:
                self.record_array_reads.append({"procedure": routine.qualified, **component.public()})
            return
        if kind in {"Proc_Component_Ref", "Coindexed_Named_Object"} or kind.startswith("Defined_"):
            self.event(result, "frontiers", "derived/indirect/defined-operator effects", node)
            return
        if kind in {"Part_Ref", "Function_Reference", "Structure_Constructor", "Intrinsic_Function_Reference"}:
            target, arguments = node.items
            binding = self.analysis._binding(scope, target)
            if kind == "Part_Ref" and binding is not None and binding.rank:
                self.binding(result, binding, node)
                self.expression(routine, arguments, result)
                return
            candidates = self.analysis._candidates(scope, target)
            if candidates:
                self.invocation(routine, node, result, function=True)
                return
            if (kind != "Intrinsic_Function_Reference" or binding is not None
                    or self.analysis._unknown_exports(scope) or str(target).lower() not in _INTRINSICS):
                self.event(result, "frontiers", "unknown function or address escape", node)
            elif self.record_array_inquiry(routine, node, result):
                return
            self.expression(routine, arguments, result)
            return
        if kind == "Name":
            binding = self.analysis._binding(scope, node)
            if binding is not None:
                self.binding(result, binding, node)
                if (id(binding), True) in self.readonly_records and not self.in_specification:
                    self.event(result, "frontiers", "whole record value requires an exact readonly source association", node)
            elif self.analysis._candidates(scope, node):
                self.event(result, "escapes", "procedure actual/address association", node)
            return
        for child in _children(node):
            self.expression(routine, child, result)

    def invocation(self, routine, node, result, *, function=False):
        scope = self.resolve(routine, node)
        for name in self.analysis._candidates(scope, node.items[0]):
            candidate = self.analysis.numerical_helpers.get(name)
            if candidate is not None and any(argument not in candidate.scope.bindings for argument in candidate.arguments):
                self.event(result, "frontiers", "procedure dummy/unknown callback association", node)
                self.expression(routine, node.items[1], result)
                return
        try:
            resolved = (resolve_original_function if function else resolve_original_call)(self.analysis, scope, node)
        except CompilationError as error:
            if not function and self.terminating_invocation(routine, node, result):
                return
            self.event(result, "frontiers", "unresolved original call: " + str(error), node)
            self.expression(routine, node.items[1], result)
            return
        from compiler.frontend.descriptor_associations import _attached, _attachments
        retained = _attachments(resolved)
        # Cached association tokens authenticate source/namespace roles, but
        # their consumed fields must also remain tied to the exact original
        # argument nodes and declared bindings. Reuse the same original type
        # and candidate association code, not a payload/function-effect proof.
        types = _OriginalTypes(self.analysis)
        types.caller_scope = scope
        callee, actuals, associations, arguments = _resolve_candidates(
            self.analysis, scope, node, types.associate, function=function)
        for consulted in types.scopes.values():
            self.scope(consulted)
        for _ in range(types.visits):
            self.tick()
        if (not _attached(resolved, retained) or resolved.callee is not callee or resolved.procedure != callee.qualified
                or resolved.target != str(node.items[0]) or resolved.source_kind != callee.source_kind
                or resolved._caller is not routine
                or resolved.formals != callee.arguments or len(resolved.actuals) != len(actuals)
                or any(a is not b for a, b in zip(resolved.actuals, actuals, strict=True))
                or len(resolved.associations) != len(associations)
                or len(resolved.original_arguments) != len(arguments)
                or any(a.actual is not b.actual or a.formal_binding is not b.formal_binding or a.binding is not b.binding
                       or a.public() != b.public() for a, b in zip(resolved.associations, associations, strict=True))
                or any(a.actual is not b.actual or a.public() != b.public()
                       for a, b in zip(resolved.original_arguments, arguments, strict=True))):
            raise CompilationError("storage mutation consumed original call association changed")
        self.call_receipts.append((resolved, _consumed_call_receipt(resolved), retained))
        if not function:
            from compiler.frontend.library_storage_calls import library_storage_call

            library = library_storage_call(self.analysis, routine, node, resolved)
            if library is not None:
                self.library_calls[library.identity] = library
                arrays = {item["name"] for item in library.public()["array_formals"]}
                mapping = {}
                for item in resolved.associations:
                    if item.formal not in arrays:
                        continue
                    self.binding(result, item.formal_binding, node)
                    self.expression(routine, item.actual, result)
                    mapping[_resource(item.formal_binding)] = _resource(item.binding)
                    if item.formal_binding.intent == "out":
                        self.event(result, "definition_kills", "original INTENT(OUT) entry kill", node,
                                   _resource(item.binding))
                result["calls"].append({"original_source": str(node), "association": resolved.public(),
                    "canonical_resources": mapping, "qualified_library_storage_identity": library.identity})
                return
        if resolved.callee.source_kind == "external":
            self.event(result, "frontiers", "external ABI requires separate storage contract", node)
        mapping = {}
        for item in resolved.associations:
            formal, actual = item.formal_binding, item.binding
            if self.native_lifetime is not None:
                record_actual = self.native_lifetime.association(resolved, item)
                component_path = self.native_lifetime.component_actual(routine, resolved, item)
                if record_actual is not None:
                    self.binding(result, formal, node)
                    self.binding(result, actual, item.actual)
                    mapping[_resource(formal)] = _resource(actual)
                    continue
                if component_path is not None:
                    self.binding(result, formal, node)
                    mapping[_resource(formal)] = self.native_field(routine, item.actual, result)
                    continue
            component = self.numeric_component_association(routine, resolved, item, result)
            if component is not None:
                self.binding(result, formal, node)
                mapping[_resource(formal)] = component
                continue
            if not self.readonly_record_association(routine, resolved, item, result):
                self.binding(result, formal, node)
                self.expression(routine, item.actual, result)
                if actual is not None:
                    self.binding(result, actual, item.actual)
            if ("allocatable" in formal.attributes and item.actual is not None
                    and (actual is None or "allocatable" not in actual.attributes or _kind(item.actual) != "Name")):
                self.event(result, "frontiers", "allocatable formal requires exact whole allocation actual", node)
            mapping[_resource(formal)] = _resource(actual) if actual is not None else None
        child = self.procedure(resolved.procedure)
        result["calls"].append({"original_source": str(node), "association": resolved.public(),
                                "canonical_resources": mapping})
        def project(resource):
            if resource in mapping:
                return mapping[resource]
            for formal, actual in mapping.items():
                if resource.startswith(formal + "%"):
                    return None if actual is None else actual + resource[len(formal):]
            return resource

        for resource in child["resources"]:
            projected = project(resource)
            if projected is not None and projected not in result["resources"]:
                result["resources"].append(projected)
        for bucket in ("mutations", "escapes", "frontiers", "definition_kills", "observables"):
            for original in child[bucket]:
                resource = original["resource"]
                projected = project(resource)
                if projected is not None:
                    event = {**original, "resource": projected, "via": routine.qualified + " -> " + resolved.procedure}
                    if event not in result[bucket]:
                        result[bucket].append(event)

    def terminating_invocation(self, routine, node, result):
        from compiler.frontend.call_bindings import OriginalCallBudgetExceeded
        from compiler.frontend.descriptor_termination import (
            DescriptorTerminationUnsupported,
            prove_descriptor_termination,
        )

        try:
            proof = prove_descriptor_termination(self.analysis, routine.qualified, node,
                                                  tick=self.tick, resolution_tick=self.tick)
        except OriginalCallBudgetExceeded:
            raise
        except DescriptorTerminationUnsupported:
            return False
        from compiler.frontend.descriptor_associations import _attachments
        self.terminating_receipts[routine.qualified, id(node)] = proof, _attachments(proof)
        child = proof._callee
        if child.qualified in self.active:
            raise CompilationError("storage mutation bounded terminating call recursion unavailable")
        if child.qualified not in self.members:
            if (len(self.active) >= self.analysis.depth_limit
                    or len(self.members) + len(self.active) >= self.analysis.procedure_limit):
                raise CompilationError("storage mutation bounded terminating call closure unavailable")
            self.inventory(child)
            self.scope(child.scope)
            member = self.result(child)
            member["qualified_original_terminating_callee"] = True
            # The existing issuer independently checked every original
            # declaration/body statement. It permits only readonly literal
            # scalar associations, diagnostic WRITE and final STOP; it has no
            # nested calls or hidden storage effects to omit here.
            for event in proof.public()["original_observable_events"]:
                self.event(member, "observables", event["kind"], event["source"])
            self.members[child.qualified] = member
        self.terminating_calls[routine.qualified, id(node)] = proof
        result["calls"].append({"original_source": str(node), "association": {
            "procedure": child.qualified, "original_terminating_descriptor_identity": proof.identity,
            "effect_authority": False, "execution_authority": False}, "canonical_resources": {}})
        for event in self.members[child.qualified]["observables"]:
            projected = {**event, "via": routine.qualified + " -> " + child.qualified}
            if projected not in result["observables"]:
                result["observables"].append(projected)
        return True

    def whole_target(self, routine, node, result):
        scope = self.analysis.source_scope_for(node, routine.scope)
        base = node.items[0] if _kind(node) == "Part_Ref" else node
        binding = self.binding(result, self.analysis._binding(scope, base), node)
        if binding is not None and (id(binding), True) in self.readonly_records:
            self.event(result, "frontiers", "readonly record target requires separate mutation authority", node)
        return binding

    def allocation(self, routine, node, result):
        self.resolve(routine, node)
        objects = node.items[1] if _kind(node) == "Allocate_Stmt" else node.items[0]
        for obj in _children(objects):
            target = obj.items[0] if _kind(obj) == "Allocation" else obj
            binding = self.whole_target(routine, target, result)
            if binding is not None:
                self.event(result, "mutations", "allocate" if _kind(node) == "Allocate_Stmt" else "deallocate",
                           node, _resource(binding))
        # Bounds, SOURCE/MOLD and status expressions can contain source calls.
        self.expression(routine, node.items, result)

    def forall_header(self, routine, node, result):
        """Visit original concurrent assignment control without evaluating it."""
        scope = self.resolve(routine, node)
        if _kind(node) != "Forall_Header" or len(node.items) != 2:
            self.event(result, "frontiers", "unsupported original FORALL header", node)
            self.expression(routine, node, result)
            return
        triplets, mask = node.items
        for triplet in _children(triplets):
            if _kind(triplet) != "Forall_Triplet_Spec" or len(triplet.items) != 4:
                self.event(result, "frontiers", "unsupported original FORALL triplet", triplet)
                self.expression(routine, triplet, result)
                continue
            index, lower, upper, stride = triplet.items
            binding = self.binding(result, self.analysis._binding(scope, index), index)
            if (_kind(index) != "Name" or binding is None or binding.dtype != "integer" or binding.rank
                    or binding.attributes & (_UNSAFE | {"allocatable", "parameter"})):
                self.event(result, "frontiers", "FORALL index needs its original independent INTEGER declaration", index)
            for bound in (lower, upper, stride):
                self.expression(routine, bound, result)
        self.expression(routine, mask, result)

    def statement(self, routine, node, result):
        if node is None:
            return
        self.tick()
        kind = _kind(node)
        if kind == "Block_Construct":
            from compiler.frontend.descriptor_blocks import DescriptorBlockUnsupported, original_fixed_block

            if routine is not self.joined_routine or self.joined_nodes.get(id(node)) is not node:
                self.event(result, "frontiers", "BLOCK requires complete original joined storage authority", node)
                for child in _children(node):
                    self.statement(routine, child, result)
                return
            try:
                proof = original_fixed_block(self.analysis, routine, node, scope_validator=self.scope, tick=self.tick)
            except DescriptorBlockUnsupported as error:
                self.event(result, "frontiers", "unproved original BLOCK declarations: " + str(error), node)
                for child in _children(node):
                    self.statement(routine, child, result)
                return
            attached = tuple((part.name, getattr(proof, part.name)) for part in fields(proof))
            self.fixed_blocks[id(node)] = proof, attached, proof.public()
            for binding in proof._scope.bindings.values():
                self.binding(result, binding, node)
            for child in _children(node):
                if _kind(child) == "Specification_Part":
                    for declaration in _children(child):
                        if _kind(declaration) == "Type_Declaration_Stmt":
                            for entity in declaration.items[2].items:
                                self.expression(routine, entity.items[3], result)
                elif _kind(child) not in {"Block_Stmt", "End_Block_Stmt"}:
                    self.statement(routine, child, result)
            return
        if kind == "Associate_Construct":
            from compiler.frontend.descriptor_associations import (
                _ReadonlyRecordAssociationRequired,
                original_fixed_association,
            )

            try:
                try:
                    proof = original_fixed_association(self.analysis, routine, node,
                        scope_validator=self.scope, tick=self.tick)
                except _ReadonlyRecordAssociationRequired as error:
                    if type(error) is not _ReadonlyRecordAssociationRequired:
                        raise
                    proof = original_fixed_association(self.analysis, routine, node,
                        scope_validator=self.readonly_scope, tick=self.tick, declaration_only=True)
                    for access in proof._association.selectors:
                        self.readonly_record(access.root_binding, formal=False)
            except CompilationError as error:
                self.event(result, "frontiers", "unproved original lexical association: " + str(error), node)
                for child in _children(node):
                    self.statement(routine, child, result)
                return
            attached = tuple((part.name, getattr(proof, part.name)) for part in fields(proof))
            self.fixed_associations[id(node)] = proof, attached, proof.public()
            if self.enclosing_context is not None:
                self.enclosing_context.consume(node, proof)
            self.association_stack.append(proof)
            try:
                for child in tuple(_children(node))[1:-1]:
                    self.statement(routine, child, result)
            finally:
                self.association_stack.pop()
            return
        if kind == "Assignment_Stmt":
            scope = self.resolve(routine, node)
            if not self.assignment_inapplicable(routine, scope, node, result):
                self.event(result, "frontiers", "defined assignment needs its original source association", node)
            target, _equals, value = node.items
            binding = self.whole_target(routine, target, result) if _kind(target) in {"Name", "Part_Ref"} else None
            if binding is not None and _kind(target) == "Name" and "allocatable" in binding.attributes:
                self.event(result, "mutations", "whole allocatable assignment may replace storage", node, _resource(binding))
            if (_kind(target) == "Data_Ref" and self.record_region is not None
                    and any(target is item for item in self.record_region._roles[7])):
                self.expression(routine, target, result)
            elif _kind(target) == "Data_Ref" and self.native_field(routine, target, result) is not None:
                pass
            elif _kind(target) == "Data_Ref":
                # Even a primitive allocatable component assignment may replace
                # storage. The READ-only declaration service grants no LHS or
                # descriptor-mutation classification, including section writes.
                self.event(result, "frontiers", "component assignment requires original mutation authority", target)
            else:
                self.expression(routine, target, result)
            self.expression(routine, value, result)
            return
        if kind == "Call_Stmt":
            scope = self.resolve(routine, node)
            name = str(node.items[0]).lower()
            if (name == "move_alloc" and not self.analysis._binding(scope, node.items[0])
                    and not self.analysis._candidates(scope, node.items[0]) and not self.analysis._unknown_exports(scope)):
                arguments = list(_children(node.items[1]))
                if len(arguments) != 2 or any(_kind(actual) != "Name" for actual in arguments):
                    self.event(result, "frontiers", "MOVE_ALLOC requires exact FROM/TO allocation names", node)
                for actual in arguments:
                    binding = self.whole_target(routine, actual, result)
                    if binding is not None:
                        self.event(result, "mutations", "move_alloc", node, _resource(binding))
                return
            self.invocation(routine, node, result)
            return
        if kind in {"Allocate_Stmt", "Deallocate_Stmt"}:
            self.allocation(routine, node, result)
            return
        if (kind == "Pointer_Assignment_Stmt" and self.record_region is not None
                and node is self.record_region._roles[5]):
            self.resolve(routine, node)
            self.expression(routine, node.items[2].items[1], result)
            return
        if (kind == "Pointer_Assignment_Stmt" and self.native_lifetime is not None
                and self.native_lifetime.pointer_assignment(routine, node)):
            self.resolve(routine, node)
            self.expression(routine, node.items[2].items[1], result)
            return
        if kind in {"Pointer_Assignment_Stmt", "Nullify_Stmt"}:
            self.event(result, "escapes", "pointer association", node)
            self.resolve(routine, node)
            self.expression(routine, node.items, result)
            return
        if kind in {"Write_Stmt", "Print_Stmt"}:
            # Initial I/O slice: numeric/literal external output only. Its
            # observability still requires a reached execution hook or cut.
            self.resolve(routine, node)
            self.event(result, "observables", "original diagnostic output", node)
            if any(_kind(child) in {"Io_Implied_Do", "Io_Implied_Do_Control"} for child in self.walk(node)):
                self.event(result, "frontiers", "unproved I/O implied control", node)
            for child in self.walk(node):
                if (_kind(child) == "Io_Control_Spec" and child.items[0] is not None
                        and str(child.items[0]).lower() not in {"unit", "fmt", "iostat", "iomsg", "err", "advance", "rec", "pos"}):
                    self.event(result, "frontiers", "unproved or asynchronous I/O borrowing", node)
            self.expression(routine, node.items, result)
            return
        if kind in {"Stop_Stmt", "Error_Stop_Stmt", "Return_Stmt", "Exit_Stmt", "Cycle_Stmt"}:
            self.resolve(routine, node)
            self.event(result, "observables", "original control/termination", node)
            self.expression(routine, node.items, result)
            return
        if kind == "Comment":
            if (str(node).lstrip().startswith("!$") and not
                    ((routine is self.joined_routine and self.joined_directives.get(id(node)) is node)
                     or (self.joined_closure is not None and self.joined_closure.permits(routine, node)))):
                self.event(result, "frontiers", "concurrency requires separate joined storage authority", node)
            return
        if kind == "Case_Construct":
            # This is a storage-effect union, not branch evaluation. Keep the
            # original selector, every CASE bound and every arm, including a
            # possible no-match continuation when there is no DEFAULT.
            self.select_cases.append({"procedure": routine.qualified, "original_source": str(node),
                "coverage": "original selector, every CASE bound and every arm body",
                "branch_selection_authorized": False, "guaranteed_match_authorized": False})
            for child in _children(node):
                self.statement(routine, child, result)
            return
        if kind in {"Execution_Part", "If_Construct", "Block_Nonlabel_Do_Construct", "Block_Label_Do_Construct"}:
            for child in _children(node):
                self.statement(routine, child, result)
            return
        if kind == "Forall_Construct":
            # FORALL completes its original assignments before continuation.
            # It is not a detached task or permission to speculate payloads.
            # The SAME scanner still visits every header, mask and statement,
            # including any allocation/association or unknown function effect.
            for child in _children(node):
                self.statement(routine, child, result)
            return
        if kind == "Forall_Construct_Stmt":
            self.forall_header(routine, node.items[1], result)
            return
        if kind == "Forall_Stmt":
            self.forall_header(routine, node.items[0], result)
            self.statement(routine, node.items[1], result)
            return
        if kind in {"If_Then_Stmt", "Else_If_Stmt", "Nonlabel_Do_Stmt", "If_Stmt", "Select_Case_Stmt", "Case_Stmt"}:
            self.resolve(routine, node)
            if kind == "If_Stmt":
                self.expression(routine, node.items[0], result)
                self.statement(routine, node.items[1], result)
            else:
                self.expression(routine, node.items, result)
            return
        if kind in {"End_If_Stmt", "Else_Stmt", "End_Do_Stmt", "Continue_Stmt", "End_Forall_Stmt", "End_Associate_Stmt", "End_Select_Stmt"}:
            return
        self.event(result, "frontiers", "unsupported original lifecycle/control: " + kind, node)
        # Do not hide a later known mutation behind an unsupported statement.
        self.expression(routine, node.items if hasattr(node, "items") else (), result)

    def procedure(self, name):
        if name in self.members:
            return self.members[name]
        if (name in self.active or len(self.active) >= self.analysis.depth_limit
                or len(self.members) + len(self.active) >= self.analysis.procedure_limit):
            raise CompilationError("storage mutation bounded call closure/recursion unavailable")
        routine = _original(self.analysis, name)
        self.inventory(routine)
        self.scope(routine.scope)
        result = self.result(routine)
        self.active.append(name)
        header = _part(routine.scope.node, "Subroutine_Stmt") or _part(routine.scope.node, "Function_Stmt")
        function_result = (str(header.items[3].items[0] if header.items[3] is not None else header.items[1]).lower()
                           if routine.source_kind in {"function", "internal_function"} else None)
        for issue in routine.issues:
            self.event(result, "frontiers", "unproved original declaration: " + issue, routine.scope.node)
        specification = _part(routine.scope.node, "Specification_Part")
        for node in self.walk(specification):
            if _kind(node) in _ALIASES:
                self.event(result, "frontiers", "unproved declaration association", node)
            if _kind(node) in {"Data_Stmt", "Save_Stmt"}:
                self.event(result, "frontiers", "implicit saved declaration requires separate lifecycle classification", node)
            if _kind(node) in {"Type_Declaration_Stmt", "Parameter_Stmt"}:
                self.resolve(routine, node)
                prior_specification = self.in_specification
                self.in_specification = True
                try:
                    self.expression(routine, node.items, result)
                finally:
                    self.in_specification = prior_specification
        for binding in routine.scope.bindings.values():
            self.binding(result, binding, routine.scope.node)
            for bound in (*binding.shape_nodes, *binding.lower_bound_nodes):
                if bound is not None:
                    self.resolve(routine, bound)
                    self.expression(routine, bound, result)
            if (binding.name not in routine.arguments and binding.dtype in {"real", "integer", "logical"}
                    and not binding.attributes & (_UNSAFE | {"save"})):
                result["private_local_resources"].append(_resource(binding))
            if binding.name in routine.arguments and binding.intent == "out":
                self.event(result, "definition_kills", "original INTENT(OUT) entry kill", routine.scope.node, _resource(binding))
                if "allocatable" in binding.attributes:
                    self.event(result, "mutations", "allocatable INTENT(OUT) entry deallocation", routine.scope.node, _resource(binding))
            elif ("allocatable" in binding.attributes and "save" not in binding.attributes
                  and binding.name not in routine.arguments and binding.name != function_result):
                self.event(result, "mutations", "local allocation implicit exit deallocation", routine.scope.node, _resource(binding))
        if function_result is not None:
            binding = routine.scope.bindings.get(function_result)
            if binding is None or "allocatable" in binding.attributes:
                self.event(result, "frontiers", "function result transfer/lifecycle requires separate authority", header)
            elif binding.rank:
                # A bounded primitive value result owns no allocatable or
                # associated storage. Its original specification, every body
                # statement and hidden callee still receive this same scan.
                if (binding.dtype not in {"real", "integer", "logical"} or binding.kind not in {4, 8}
                        or not 1 <= binding.rank <= 4
                        or binding.attributes & (_UNSAFE | {"save", "parameter"})):
                    self.event(result, "frontiers", "function array result requires ordinary fixed value storage", header)
                else:
                    from compiler.frontend.descriptor_associations import _attachments
                    from compiler.frontend.descriptor_intrinsics import intrinsic_scope_attachments
                    from compiler.frontend.descriptor_teams import _group_syntax_checkpoint

                    # Retain consumed declaration/constant-bound inputs before
                    # helpers or public-record callbacks can observe them.
                    scope_receipt = intrinsic_scope_attachments(binding.declaring_scope)
                    # The same constant namespaces may contain host/module
                    # PARAMETER expressions outside this routine's syntax.
                    # Retain their parser children without duplicating the
                    # existing INTEGER constant evaluator's lookup rules.
                    constant_syntax = _group_syntax_checkpoint((*binding.shape_nodes, *binding.lower_bound_nodes,
                        *(value
                        for _scope, _parent, values, _intrinsic in scope_receipt[3]
                        for value in values.values())))
                    self.fixed_array_result_receipts.append((binding, _attachments(binding),
                        scope_receipt, constant_syntax))
                    try:
                        bounds = _fixed_array_bounds(self.analysis, binding)
                    except CompilationError as error:
                        self.event(result, "frontiers", "unproved fixed function result: " + str(error), header)
                    else:
                        self.fixed_array_results[name] = {
                            "procedure": name, "original_result": binding.public(), "bounds": bounds,
                            "storage": "original bounded primitive value result; no allocation or association transfer",
                            "payload_access_authorized": False, "execution_authorized": False,
                            "entry_definition_authority": False, "lifetime_authorized": False}
        if any(_kind(node) == "Language_Binding_Spec" for node in self.walk(header)):
            self.event(result, "frontiers", "foreign ABI", header)
        self.statement(routine, routine.execution, result)
        self.active.pop()
        self.members[name] = result
        return result

    def finish(self):
        if self.limits != (self.analysis.depth_limit, self.analysis.procedure_limit, self.analysis.operation_limit):
            raise CompilationError("storage mutation source policy changed during projection")
        _validate_scope_roles(self.analysis, self.roles, self.scopes)
        for scope in self.scopes.values():
            _validate_scope(self.analysis, scope)
        # Selections and function-expression roots need the same final source
        # fence as complete callees, even when they are not closure members.
        for name in self.inventories:
            _original(self.analysis, name)
        from compiler.frontend.descriptor_associations import _attached
        for resolved, receipt, retained in self.call_receipts:
            resolved.validate(self.analysis, resolved._scope, resolved._original)
            if not _attached(resolved, retained) or _consumed_call_receipt(resolved) != receipt:
                raise CompilationError("storage mutation consumed original call changed before publication")
        self.analysis.verify_publication()


def _enclosing_association_frames(routine, originals):
    """Find exact original parents; names and canonical text are insufficient."""
    requested = {id(node): node for node in originals}
    found, visits, pending = {}, 0, [(routine.execution, ())]
    while pending:
        node, parents = pending.pop()
        if not isinstance(node, (Base, tuple, list)):
            continue
        visits += 1
        if requested.get(id(node)) is node:
            if id(node) in found:
                raise CompilationError("joined storage has repeated original lexical membership")
            found[id(node)] = parents
        pending.extend((child, (*parents, node)) for child in reversed(_children(node)))
    paths = [found.get(id(node), ()) for node in originals]
    if not any(_kind(parent) == "Associate_Construct" for path in paths for parent in path):
        return (), visits
    if (not paths or any(not path for path in paths)
            or any(len(path) != len(paths[0]) or any(a is not b for a, b in zip(path, paths[0], strict=True))
                   for path in paths)):
        raise CompilationError("joined storage requires one exact original enclosing lexical sequence")
    frames = paths[0][1:]
    if any(_kind(parent) != "Associate_Construct" for parent in frames):
        raise CompilationError("joined storage enclosing lexical frame requires separate source authority")
    return frames, visits


class _EnclosingJoinedContext:
    """One fresh scan's consuming receipts, never a registered authority cache."""

    def __init__(self, scanner, routine, graph, team, frames):
        from compiler.frontend.descriptor_associations import _attachments, _syntax_attachments
        from compiler.frontend.descriptor_block_storage import _scopes

        self.scanner, self.routine, self.graph, self.team, self.frames = scanner, routine, graph, team, frames
        self.team_fields = _attachments(team)
        self.team_registry = scanner.analysis._descriptor_team_proofs
        self.graph_fields = _attachments(graph)
        self.graph_maps = tuple((name, getattr(graph, name), tuple(getattr(graph, name).items()))
                                for name in ("nodes", "_originals", "_ids", "native_groups"))
        self.graph_nodes = tuple((node, _attachments(node)) for node in graph.nodes.values())
        self.syntax = _syntax_attachments(routine.execution)
        self.associations = {}
        self.rows = []
        # This helper saves raw fields/maps/entries before delegated serializers.
        self.namespaces = _scopes(scanner.analysis)
        self.checkpoint()

    def consume(self, original, proof):
        from compiler.frontend.descriptor_associations import _attachments

        self.checkpoint()
        if not any(original is frame for frame in self.frames):
            return
        if (proof._analysis is not self.scanner.analysis or proof._routine is not self.routine
                or proof._original is not original or id(original) in self.associations):
            raise CompilationError("joined storage changed its exact enclosing association")
        association = proof._association
        receipt = (proof, _attachments(proof), association, _attachments(association),
                   tuple((access, _attachments(access)) for access in association.selectors))
        self.associations[id(original)] = receipt
        graph, selected = self.scanner.analysis._selected_source(self.routine.qualified, (original,))
        members = graph.source_nodes(selected[0]) if len(selected) == 1 else ()
        if graph is not self.graph or len(members) != 1 or members[0] is not original:
            raise CompilationError("joined storage lost its exact enclosing association graph node")
        self.rows.append({"source_node_id": selected[0], "association": proof.public(),
                          "complete_original_body_required": True})
        self.checkpoint()

    def checkpoint(self):
        from compiler.frontend.descriptor_associations import _attached, _same_syntax
        from compiler.frontend.descriptor_block_storage import _same_scope_fields

        analysis = self.scanner.analysis
        if (not _attached(self.team, self.team_fields)
                or getattr(analysis, "_descriptor_team_proofs", None) is not self.team_registry
                or self.team_registry.get(self.team.identity) is not self.team
                or not _attached(self.graph, self.graph_fields)
                or any(getattr(self.graph, name) is not mapping or len(mapping) != len(items)
                       or any(mapping.get(key) is not value for key, value in items)
                       for name, mapping, items in self.graph_maps)
                or any(not _attached(node, attached) for node, attached in self.graph_nodes)
                or not _same_syntax(self.syntax) or not _same_scope_fields(analysis, self.namespaces)
                or any(not _attached(proof, attached) or proof._association is not association
                       or not _attached(association, state)
                       or any(not _attached(access, saved) for access, saved in selectors)
                       for proof, attached, association, state, selectors in self.associations.values())):
            raise CompilationError("joined storage enclosing lexical source attachment changed")

    def finish(self, record):
        from compiler.frontend.descriptor_block_storage import _same_scopes

        self.checkpoint()
        for frame in self.frames:
            receipt = self.associations.get(id(frame))
            if receipt is None or self.scanner.fixed_associations.get(id(frame), (None,))[0] is not receipt[0]:
                raise CompilationError("joined storage did not scan its complete original enclosing body")
        if (record.get("enclosing_lexical_context") != self.rows
                or len(self.rows) != len(self.frames)
                or not _same_scopes(self.scanner.analysis, self.namespaces)):
            raise CompilationError("joined storage enclosing lexical context changed at publication")
        self.scanner.analysis.verify_publication()
        self.checkpoint()


def _same_joined_storage(left, right):
    if type(left) is tuple or type(right) is tuple:
        return (type(left) is tuple and type(right) is tuple and len(left) == len(right)
                and all(a is b for a, b in zip(left, right, strict=True)))
    return left is right


def _joined_dependencies(value):
    return value if type(value) is tuple else (value,)


class _JoinedCalleeStructures:
    """Scan-local consumption of existing complete original callee structures."""
    def __init__(self, scanner, routine, graph, identities, proofs):
        from compiler.frontend.descriptor_associations import _attachments
        from compiler.frontend.descriptor_teams import DescriptorTeamProof
        from compiler.frontend.descriptor_worksharing import OriginalWorksharingStorage
        from compiler.frontend.native_group_structure import directive

        selected = tuple(node for identity in identities for node in graph.source_nodes(identity))
        if _original_call_control(selected) is None or not proofs:
            raise CompilationError("native callee structures require one complete original CALL/control selection")
        if len({id(proof) for proof in proofs}) != len(proofs):
            raise CompilationError("native callee structures require distinct original receipts")
        self.scanner, self.root, self.proofs = scanner, routine, proofs
        self.directives, self.used, self.receipts = {}, set(), []
        for proof in proofs:
            if type(proof) is DescriptorTeamProof:
                if (proof._analysis is not scanner.analysis or proof._record_region is not None
                        or (proof._native_graph is not None and (scanner.native_lifetime is None
                            or proof._native_graph is not scanner.native_lifetime.graph))):
                    raise CompilationError("native callee team requires exact ordinary original storage authority")
                child = _original(scanner.analysis, proof.procedure)
                proof.validate(scanner.analysis, child.qualified, proof.selected_node_ids)
                _child_graph, selected_ids = scanner.analysis._selected_source(child.qualified, proof._originals)
                if selected_ids != proof.selected_node_ids or _enclosing_association_frames(child, proof._originals)[0]:
                    raise CompilationError("native callee team requires its exact complete original group")
                nodes = tuple(node for original in proof._originals for node in scanner.walk(original)
                              if directive(node) is not None)
                checkpoint = None
            elif type(proof) is OriginalWorksharingStorage:
                child = proof._routine
                if proof._analysis is not scanner.analysis or _original(scanner.analysis, child.qualified) is not child:
                    raise CompilationError("native orphan structure requires its original called procedure")
                proof.validate(scanner.analysis, child)
                nodes = tuple(node for opening, _loop, ending, _collapse in proof._regions for node in (opening, ending))
                checkpoint = proof.checkpoint()
            else:
                raise CompilationError("native callee structure is not an issued team or orphan receipt")
            if child is routine or not nodes:
                raise CompilationError("native callee structures cannot authorize a selected root directive")
            for node in nodes:
                key = child.qualified, id(node)
                if key in self.directives:
                    raise CompilationError("native callee structures overlap their original directives")
                self.directives[key] = node, proof
            self.receipts.append((proof, _attachments(proof), checkpoint))

    def permits(self, routine, node):
        row = self.directives.get((routine.qualified, id(node)))
        if row is None:
            return False
        if row[0] is not node or not self.scanner.active or self.scanner.active[-1] != routine.qualified:
            raise CompilationError("native worksharing requires a complete original callee scan")
        self.used.add(id(row[1]))
        return True

    def checkpoint(self, *, complete=False):
        from compiler.frontend.descriptor_associations import _attached
        from compiler.frontend.descriptor_teams import DescriptorTeamProof

        for proof, attached, checkpoint in self.receipts:
            if not _attached(proof, attached):
                raise CompilationError("native callee structure attachment changed")
            if type(proof) is DescriptorTeamProof:
                proof.validate(self.scanner.analysis, proof.procedure, proof.selected_node_ids)
            else:
                proof.require_checkpoint(checkpoint)
        if complete and self.used != {id(proof) for proof in self.proofs}:
            raise CompilationError("native callee structure was not consumed by the complete original CALL")


def _joined_context(scanner, routine, graph, identities, proof):
    from compiler.frontend.descriptor_teams import DescriptorTeamProof
    from compiler.frontend.native_group_structure import directive

    if type(proof) is tuple:
        scanner.joined_closure = _JoinedCalleeStructures(scanner, routine, graph, identities, proof)
        scanner.joined_storage = proof
        scanner.joined_closure.checkpoint()
        return None
    if (type(proof) is not DescriptorTeamProof or proof._analysis is not scanner.analysis
            or proof.procedure != routine.qualified
            or sha256(proof._record.encode()).hexdigest() != proof.identity):
        raise CompilationError("joined storage requires its exact same-analysis selected team token")
    record_region = proof._record_region
    if record_region is None and tuple(identities) != proof.selected_node_ids:
        raise CompilationError("joined storage requires its exact same-analysis selected team token")
    originals = tuple(node for identity in proof.selected_node_ids for node in graph.source_nodes(identity))
    # Ordinary comments have authenticated source membership but no graph
    # operation. Keep every active sentinel/directive and original body node.
    # The team validation below still checks the full retained source tuple.
    def semantic(nodes):
        return tuple(node for node in nodes if _kind(node) != "Comment" or str(node).lstrip().startswith("!$"))

    original_semantic, retained_semantic = semantic(originals), semantic(proof._originals)
    selected_graph, selected = scanner.analysis._selected_source(routine.qualified, proof._originals)
    if (selected_graph is not graph or selected != proof.selected_node_ids
            or len(original_semantic) != len(retained_semantic)
            or any(a is not b for a, b in zip(original_semantic, retained_semantic, strict=True))):
        raise CompilationError("joined storage requires the entire original team selection")
    if record_region is not None:
        from compiler.frontend.storage_record_regions import OwnedRecordRegion
        if type(record_region) is not OwnedRecordRegion or record_region._roles[1] is not routine:
            raise CompilationError("record storage requires its exact original callee body issuer")
        body = tuple(_children(routine.execution))
        retained = record_region._roles[8]
        body_graph, body_selected = scanner.analysis._selected_source(routine.qualified, retained)
        if (body_graph is not graph or tuple(identities) != body_selected
                or len(body) != len(retained) or any(a is not b for a, b in zip(body, retained, strict=True))):
            raise CompilationError("record storage requires the complete exact original callee body")
    frames, visits = _enclosing_association_frames(routine, proof._originals)
    if frames:
        scanner.enclosing_context = _EnclosingJoinedContext(scanner, routine, graph, proof, frames)
        for _ in range(visits):
            scanner.tick()
        scanner.enclosing_context.checkpoint()
    from compiler.frontend.descriptor_associations import _attachments
    if proof._record_region is not None:
        scanner.record_region = proof._record_region
        from compiler.frontend.storage_record_regions import RecordRegionCheckpoint
        scanner.record_region_attached = (_attachments(proof),
            RecordRegionCheckpoint(scanner.analysis, scanner.record_region))
    proof.validate(scanner.analysis, routine.qualified, proof.selected_node_ids)
    scanner.joined_storage, scanner.joined_routine = proof, routine
    scanner.joined_nodes = {id(node): node for node in scanner.walk(originals)}
    scanner.joined_directives = {key: node for key, node in scanner.joined_nodes.items() if directive(node) is not None}
    if scanner.enclosing_context is not None:
        scanner.enclosing_context.checkpoint()
        return frames[0]
    return None


def _derive(analysis, procedure, expression=None, selection=None, *, joined_storage=None, native_graph=None, _defer_publication=False):
    analysis.verify_source()
    scanner = _Scan(analysis)
    if native_graph is not None:
        from compiler.frontend.descriptor_teams import DescriptorTeamProof
        from compiler.frontend.native_lifetime_context import NativeLifetimeContext
        if (selection is None or expression is not None
                or type(joined_storage) not in {tuple,DescriptorTeamProof}):
            raise CompilationError("native lifetime scan requires exact whole CALL or joined group and independent storage")
        scanner.native_lifetime = NativeLifetimeContext(analysis, native_graph,
            joined_storage=joined_storage if type(joined_storage) is DescriptorTeamProof else None)
        scanner.native_lifetime.require_selection(procedure, selection)
    routine = _original(analysis, procedure)
    original_expression = None
    selected_record = None
    if joined_storage is not None and (selection is None or expression is not None):
        raise CompilationError("joined storage is restricted to one exact original selected team")
    if selection is not None:
        if expression is not None:
            raise CompilationError("storage selection and function projections are distinct queries")
        graph, identities = analysis._selected_source(procedure, selection)
        if not identities:
            raise CompilationError("storage selection requires exact original executable nodes")
        scanner.inventory(routine)
        scanner.scope(routine.scope)
        root = scanner.result(routine)
        enclosing = (_joined_context(scanner, routine, graph, identities, joined_storage)
                     if joined_storage is not None else None)
        if scanner.record_region is not None:
            # Whole specification, entry/exit lifecycle, original body and all
            # callees use the unchanged complete-procedure scanner. The team
            # exception above covers only its exact registered directive nodes.
            root = scanner.procedure(procedure)
        elif enclosing is not None:
            # The fixed association receipt requires the complete original body.
            # Only the exact selected team's directive objects are authorized;
            # every prefix/tail statement and callee is scanned as well.
            scanner.statement(routine, enclosing, root)
        else:
            for identity in identities:
                node = graph.nodes[identity]
                for original in graph.source_nodes(identity):
                    if node.details.get("evaluation") in {"condition", "loop_header"}:
                        scanner.resolve(routine, original)
                        scanner.expression(routine, original, root)
                    else:
                        scanner.statement(routine, original, root)
        selected_record = {"structured_identity": graph.identity, "selected_node_ids": list(identities)}
    elif expression is None:
        root = scanner.procedure(procedure)
    else:
        scanner.inventory(routine)
        scanner.scope(routine.scope)
        path = scanner.expression_path(routine, expression)
        if _kind(expression) not in {"Part_Ref", "Function_Reference", "Structure_Constructor"}:
            raise CompilationError("storage function query requires an exact original source function call")
        root = scanner.result(routine)
        scanner.invocation(routine, expression, root, function=True)
        original_expression = {"parser_kind": _kind(expression), "source": str(expression), "syntax_path": path,
                               "calls": root["calls"]}
    original_roots = {_resource(binding): binding.root for name, binding in routine.scope.bindings.items()
                      if name in routine.arguments}
    projected = {key: [{**event, "resource": original_roots.get(event["resource"], event["resource"])} for event in root[key]]
                 for key in ("mutations", "escapes", "frontiers", "definition_kills", "observables")}
    projected["resources"] = [original_roots.get(resource, resource) for resource in root["resources"]]
    private = {resource for member in scanner.members.values() for resource in member["private_local_resources"]}
    record = {"version": 1, "kind": "original_storage_mutation_effects", "procedure": procedure,
            "source_inputs": analysis.inputs.identity(), "closure": list(scanner.members.values()),
            **projected,
            "private_local_resources": sorted(original_roots.get(resource, resource) for resource in private),
            "alias_requirement": "different formal/global names do not establish disjointness; only original nonescaping private locals are excluded",
            "resource_coordinates": "qualified original dummy declarations through exact call mappings; public root dummies retain argument roots",
            "complete": not root["frontiers"],
            "budget": {"formula": "max(65536,sum(8*S+lookup_limit*(S+declarations)))",
                       "source": list(scanner.inventories.values()), "limit": max(DEFAULT_SOURCE_SCAN_LIMIT, scanner.allowance)},
            "analysis_limits": list(scanner.limits), "payload_access_authorized": False,
            "entry_definition_authority": False, "capture_authorized": False, "execution_authorized": False,
            "motion_authorized": False, "gpu_legality_established": False, "alias_authority": False}
    if original_expression is not None:
        record.update(version=2, kind="original_storage_function_mutation_effects",
                      original_expression=original_expression,
                      native_execution_requirement="independent exact native effects or reached publication and retirement before original execution")
    if selected_record is not None:
        record.update(version=3, kind="original_selected_storage_mutation_effects", original_selection=selected_record,
                      entry_exit_lifecycle_authorized=False,
                      native_execution_requirement="independent exact native effects or reached publication and retirement before original execution")
    if joined_storage is not None and type(joined_storage) is not tuple:
        record.update(version=4, kind="original_joined_selected_storage_mutation_effects",
                      joined_storage_authority=joined_storage.public())
    if scanner.library_calls:
        if selected_record is None:
            raise CompilationError("qualified library storage requires an exact selected original interval")
        record.update(version=5, kind="original_qualified_library_storage_mutation_effects",
                      qualified_library_storage=[proof.public() for proof in scanner.library_calls.values()])
    if scanner.fixed_associations:
        record.update(version=6, kind="original_fixed_association_storage_mutation_effects",
                      fixed_scalar_associations=[public for _proof, _saved_attachment, public in scanner.fixed_associations.values()])
    if scanner.fixed_blocks:
        record.update(version=7, kind="original_joined_block_storage_mutation_effects",
                      fixed_scalar_blocks=[public for _proof, _saved_attachment, public in scanner.fixed_blocks.values()])
    if scanner.enclosing_context is not None:
        record.update(version=STORAGE_ENCLOSING_LEXICAL_CONTEXT_VERSION,
                      kind="original_enclosing_lexical_storage_mutation_effects",
                      enclosing_lexical_context=list(scanner.enclosing_context.rows),
                      enclosing_body_coverage="complete original enclosing associations; only exact selected-team directives")
    if scanner.record_region is not None:
        record.update(version=10, kind="original_owned_record_storage_mutation_effects",
                      owned_record_region=scanner.record_region.public())
    if scanner.fixed_array_results:
        record.update(version=STORAGE_FIXED_ARRAY_RESULT_VERSION,
                      fixed_array_function_results=list(scanner.fixed_array_results.values()))
    if scanner.terminating_calls:
        record.update(version=STORAGE_TERMINATING_CALL_VERSION,
                      original_terminating_calls=[proof.public() for proof in scanner.terminating_calls.values()])
    if scanner.select_cases:
        record.update(version=STORAGE_SELECT_CASE_VERSION, original_select_cases=scanner.select_cases)
    if scanner.readonly_records:
        record.update(version=STORAGE_READONLY_RECORD_VERSION,
                      readonly_record_associations=scanner.readonly_associations,
                      readonly_record_types=sorted({getattr(schema.scope, "qualified", schema.scope.module) + "::" + schema.name
                                                    for schema in scanner.readonly_records.values()}))
    if scanner.inapplicable_assignments:
        record.update(version=STORAGE_ASSIGNMENT_APPLICABILITY_VERSION,
                      original_inapplicable_assignments=scanner.inapplicable_assignments)
    if scanner.record_array_reads:
        record.update(version=STORAGE_RECORD_ARRAY_READ_VERSION,
                      original_record_array_reads=scanner.record_array_reads)
    if scanner.component_actuals:
        record.update(version=STORAGE_RECORD_ARRAY_READ_VERSION,
                      original_numeric_component_associations=scanner.component_actuals)
    if type(joined_storage) is tuple:
        scanner.joined_closure.checkpoint(complete=True)
        record.update(version=STORAGE_NATIVE_CALL_STRUCTURES_VERSION,
                      kind="original_native_call_storage_mutation_effects",
                      native_callee_structure_authorities=[proof.public() for proof in joined_storage])
    if scanner.native_lifetime is not None:
        record.update(version=STORAGE_NATIVE_LIFETIME_CONTEXT_VERSION,
                      native_lifetime_context=scanner.native_lifetime.public())
    if _defer_publication:
        # The issuer consumes this ephemeral scanner synchronously. Neither it
        # nor its callable fence is stored in an analysis/token registry.
        return record, scanner
    scanner.finish()
    return record


@dataclass(frozen=True, slots=True)
class StorageMutationEffects:
    identity: str
    _analysis: object = field(repr=False, compare=False)
    _procedure: str
    _record: object = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)
    _expression: object = field(default=None, repr=False, compare=False)
    _selection: object = field(default=None, repr=False, compare=False)
    _joined_storage: object = field(default=None, repr=False, compare=False)
    _calls: tuple = field(default=(), repr=False, compare=False)
    _terminating_calls: tuple = field(default=(), repr=False, compare=False)
    _call_attachments: tuple = field(default=(), repr=False, compare=False)
    _call_registries: tuple = field(default=(), repr=False, compare=False)
    _source_fact: object = field(default=None, repr=False, compare=False)
    _native_graph: object = field(default=None, repr=False, compare=False)
    _native_completion: object = field(default=None, repr=False, compare=False)
    _native_completion_fields: tuple = field(default=(), repr=False, compare=False)
    _native_read_dependencies: object = field(default=None, repr=False, compare=False)

    def public(self):
        return {"identity": self.identity, **_thaw(self._record)}

    def validate(self, analysis=None, procedure=None, expression=_UNSET, *, selection=_UNSET, joined_storage=_UNSET):
        from compiler.frontend.descriptor_associations import _attached, _attachments
        from compiler.frontend.native_completion import _require_graph_completion_attachments
        saved = _attachments(self)
        completion, fields, dependencies = self._native_completion,self._native_completion_fields,self._native_read_dependencies
        team = self._joined_storage
        result = self._validate(analysis,procedure,expression,selection=selection,joined_storage=joined_storage)
        if not _attached(self,saved):
            raise CompilationError("storage mutation retained source/role fields changed during validation")
        if completion is not None:
            dependencies.require()
            _require_graph_completion_attachments(completion,fields)
            registry = getattr(self._analysis,"_descriptor_team_proofs",None)
            if type(registry) is not dict or dict.get(registry,team.identity) is not team:
                raise CompilationError("native joined storage lost its exact original team registry entry")
        if not _attached(self,saved):
            raise CompilationError("storage mutation retained source/role fields changed at final validation")
        return result

    def _validate(self, analysis=None, procedure=None, expression=_UNSET, *, selection=_UNSET, joined_storage=_UNSET):
        completion, fields, dependencies = self._native_completion,self._native_completion_fields,self._native_read_dependencies
        fact = self._source_fact
        analysis = self._analysis if analysis is None else analysis
        procedure = self._procedure if procedure is None else procedure
        expression = self._expression if expression is _UNSET else expression
        selection = self._selection if selection is _UNSET else selection
        joined_storage = self._joined_storage if joined_storage is _UNSET else joined_storage
        self._require_registration(analysis, procedure, expression, selection, joined_storage)
        lifetime = self._record.get("native_lifetime_context")
        if ((lifetime is None) != (self._native_graph is None)
                or (lifetime is not None and lifetime["native_graph_identity"] != self._native_graph.identity)
                or (self._source_fact is not None and self._source_fact.query[-1] != id(self._native_graph))):
            raise CompilationError("storage mutation changed its original native lifetime graph")
        if self._native_read_dependencies is not None:
            from compiler.frontend.source_dependency_receipts import _SourceReceipt
            if (type(self._native_read_dependencies) is not _SourceReceipt
                    or self._native_read_dependencies.analysis is not analysis):
                raise CompilationError("native storage read requires its original declaration dependencies")
            self._native_read_dependencies.require()
        if self._native_graph is not None:
            self._native_graph.validate(analysis, procedure, selection)
        from compiler.frontend.native_completion import _require_graph_completion_attachments
        group = lifetime is not None and lifetime.get("whole_original_group_retirement_required")
        if (bool(group) != (self._native_completion is not None)
                or bool(group) != (self._native_read_dependencies is not None)):
            raise CompilationError("native joined storage changed its required retirement completion")
        if self._native_completion is not None:
            _require_graph_completion_attachments(self._native_completion,self._native_completion_fields)
            if (self._native_completion._graph is not self._native_graph
                    or self._native_completion._call_storage is not None
                    or self._native_completion._ordinary_join is None
                    or lifetime.get("completion_identity") != self._native_completion.identity):
                raise CompilationError("native joined storage changed its original retirement role")
            self._native_completion.validate(analysis,procedure,self._native_completion.selected_node_ids)
        if fact is not None:
            result = fact.require(self)
            if completion is not None:
                _require_graph_completion_attachments(completion,fields)
                dependencies.require()
            return result
        if getattr(analysis, "_source_session", None) is not None:
            raise CompilationError("mutable diagnostic proof cannot enter a sealed authority context")
        record, scanner = _derive(analysis, procedure, expression, selection,
                                  joined_storage=joined_storage, native_graph=self._native_graph, _defer_publication=True)
        self._validate_record(analysis, procedure, expression, selection, record, joined_storage)
        _publish_derivation(analysis, record, scanner)
        self._require_registration(analysis, procedure, expression, selection, joined_storage)
        if self._native_read_dependencies is not None:
            self._native_read_dependencies.require()
        return self

    def _require_registration(self, analysis, procedure, expression, selection, joined_storage):
        if (self._issuer is not _ISSUER or analysis is not self._analysis or procedure != self._procedure
                or expression is not self._expression or selection is not self._selection
                or not _same_joined_storage(joined_storage, self._joined_storage)
                or getattr(analysis, "_storage_mutation_effects", {}).get(self.identity) is not self):
            raise CompilationError("storage mutation requires its exact registered source projection")
        if (self._source_fact is not None and self._source_fact.query[3] != (
                ("native_callees", tuple(map(id, joined_storage))) if type(joined_storage) is tuple else id(joined_storage))):
            raise CompilationError("storage mutation changed its retained native callee structures")
        if self._source_fact is None and any(getattr(analysis, name, None) is not registry for name, registry in self._call_registries):
            raise CompilationError("storage mutation retained call registry changed")
        _require_retained_calls(analysis, self._calls, self._terminating_calls)
        from compiler.frontend.descriptor_associations import _attached
        if (len(self._call_attachments) != len(self._calls) + len(self._terminating_calls)
                or any(token is not expected or not _attached(token, saved)
                       for expected, (token, saved) in zip((*self._calls, *self._terminating_calls),
                                                          self._call_attachments, strict=True))):
            raise CompilationError("storage mutation consumed original call association retained fields changed")

    def _validate_record(self, analysis, procedure, expression, selection, record, joined_storage):
        self._require_registration(analysis, procedure, expression, selection, joined_storage)
        expected = record.get("original_selection")
        if expected is not None or self._selection is not None:
            if expected is None or self._selection is None:
                raise CompilationError("storage mutation retained selection attachment changed")
            # Reusing the fresh effect derivation must not stop authenticating
            # the token's own retained source attachment. Original AST nodes
            # and canonical node-ID queries may denote the same selection;
            # equal-text foreign parser objects never do.
            graph, retained = analysis._selected_source(procedure, self._selection)
            if (graph.identity != expected["structured_identity"]
                    or list(retained) != expected["selected_node_ids"]):
                raise CompilationError("storage mutation retained selection attachment changed")
        if _hash(record) != self.identity or _freeze_source_record(analysis, record) != self._record:
            raise CompilationError("storage mutation original source or resolution changed")
        return self

    def descriptor_actuals(self, routine, expression):
        """Consume exact readonly record associations, not whole object access.

        Every ordinary actual remains in the window's existing classifier.
        Only a record actual already authenticated by this complete projection
        contributes its original scalar index expressions instead of a second
        generic whole-object classification.
        """
        self._require_registration(self._analysis, routine.qualified, expression, None, None)
        matches = [call for call in self._calls if call._caller is routine and call._original is expression]
        if len(matches) != 1:
            raise CompilationError("storage readonly actuals require their exact original source function call")
        resolved = matches[0]
        rows = self._record.get("readonly_record_associations", ())
        result = []
        for item in resolved.associations:
            from compiler.frontend.component_bindings import _typename

            if _typename(item.formal_binding.dtype) is None:
                if item.actual is not None:
                    result.append(item.actual)
                continue
            actual = item.binding
            if (actual is None or _kind(item.actual) not in {"Name", "Part_Ref"}
                    or item.formal_binding.intent != "in" or item.formal_binding.rank
                    or item.formal_binding.attributes & (_UNSAFE | {"allocatable"})
                    or actual.attributes & _UNSAFE
                    or not any(row["caller"] == routine.qualified and row["callee"] == resolved.procedure
                               and row["formal"] == item.formal_binding.name
                               and row["actual"] == str(item.actual)
                               and row["actual_resource"] == _resource(actual) for row in rows)):
                raise CompilationError("storage readonly actuals require the retained exact typed record association")
            if _kind(item.actual) == "Part_Ref":
                result.extend(_children(item.actual.items[1]))
        return tuple(result)

    def preserves_descriptors(self, analysis=None, procedure=None, expression=_UNSET, *,
                              selection=_UNSET, joined_storage=_UNSET):
        """Check the requested association and effects in one fresh validation."""
        self.validate(analysis, procedure, expression, selection=selection, joined_storage=joined_storage)
        return _preserves_record(self._record)

    def descriptor_unchanged(self, resource):
        return self.descriptors_unchanged((resource,))

    def descriptors_unchanged(self, resources):
        """Check a buffer set against one freshly validated storage proof."""
        if type(resources) is not tuple or any(type(resource) is not str for resource in resources):
            raise CompilationError("storage descriptor query requires an exact resource tuple")
        self.validate()
        record = self._record
        private = set(record["private_local_resources"])
        wanted = set(resources)
        return wanted <= set(record["resources"]) and not record["frontiers"] and not any(
            event["resource"] in wanted or event["resource"] not in private
            for bucket in ("mutations", "escapes") for event in record[bucket])


def _require_retained_calls(analysis, calls, terminating_calls):
    from compiler.frontend.call_bindings import ResolvedOriginalCall
    from compiler.frontend.descriptor_termination import DescriptorTerminatingCallProof, _issued

    if type(calls) is not tuple or type(terminating_calls) is not tuple:
        raise CompilationError("storage mutation requires its retained original call tuples")
    original = getattr(analysis, "_original_call_associations", {})
    terminal = getattr(analysis, "_descriptor_terminating_calls", {})
    if (type(original) is not dict or type(terminal) is not dict
            or any(type(key) is not tuple or len(key) != 2 or any(type(value) is not int for value in key)
                   for key in original)
            or any(type(key) is not tuple or len(key) != 2 or type(key[0]) is not str or type(key[1]) is not int
                   for key in terminal)):
        raise CompilationError("storage mutation requires compiler-owned call registries")
    if (any(type(token) is not ResolvedOriginalCall or token._analysis is not analysis
            or original.get((id(analysis), id(token._original))) is not token for token in calls)
            or any(type(token) is not DescriptorTerminatingCallProof or not _issued(token, analysis)
            or terminal.get((token._procedure, id(token._call))) is not token for token in terminating_calls)):
        raise CompilationError("storage mutation retained original call authority changed")


def _publish_derivation(analysis, record, scanner):
    """Finish the SAME fresh scan after pure record/registry comparison."""
    from compiler.frontend.descriptor_associations import _attached
    calls = tuple(token for token, _receipt, _fields in scanner.call_receipts)
    terminating = tuple(scanner.terminating_calls.values())
    retained = tuple((token, saved) for token, _receipt, saved in scanner.call_receipts) + tuple(
        scanner.terminating_receipts.values())
    _require_retained_calls(analysis, calls, terminating)
    scanner.finish()
    if scanner.native_lifetime is not None:
        scanner.native_lifetime.checkpoint()
        if record.get("native_lifetime_context") != scanner.native_lifetime.public():
            raise CompilationError("native lifetime context changed before storage publication")
    for proof in terminating:
        proof.validate(analysis, proof._procedure, proof._call, tick=scanner.tick, resolution_tick=scanner.tick)
    if terminating and record.get("original_terminating_calls") != [proof.public() for proof in terminating]:
        raise CompilationError("storage terminating call public source changed before publication")
    for proof in scanner.library_calls.values():
        proof.validate()
    if scanner.library_calls and record.get("qualified_library_storage") != [
            proof.public() for proof in scanner.library_calls.values()]:
        raise CompilationError("qualified library storage changed before source publication")
    if scanner.joined_closure is not None:
        scanner.joined_closure.checkpoint(complete=True)
        if record.get("native_callee_structure_authorities") != [proof.public() for proof in scanner.joined_storage]:
            raise CompilationError("native callee structures changed before source publication")
    elif scanner.joined_storage is not None:
        proof = scanner.joined_storage
        proof.validate(analysis, record["procedure"], proof.selected_node_ids)
        if (record.get("joined_storage_authority") != proof.public()
                or sha256(proof._record.encode()).hexdigest() != proof.identity):
            raise CompilationError("joined storage authority changed before source publication")
    for proof, _saved_attachment, _public in scanner.fixed_associations.values():
        proof.validate(scope_validator=scanner.scope, tick=scanner.tick)
    if scanner.fixed_associations:
        analysis._require_original_global()
        if (record.get("fixed_scalar_associations") != [public for _proof, _saved_attachment, public in scanner.fixed_associations.values()]
                or any(proof.public() != public or any(getattr(proof, name) is not value for name, value in attached)
                       for proof, attached, public in scanner.fixed_associations.values())):
            raise CompilationError("storage lexical association source changed before publication")
    for proof, _saved_attachment, _public in scanner.fixed_blocks.values():
        proof.validate(scope_validator=scanner.scope, tick=scanner.tick)
    if scanner.fixed_blocks:
        from compiler.frontend.descriptor_blocks import _same_declaration

        analysis._require_original_global()
    if analysis.inputs.identity() != record["source_inputs"]:
        raise CompilationError("storage mutation original input metadata changed before publication")
    if scanner.fixed_blocks and (
                record.get("fixed_scalar_blocks") != [public for _proof, _saved_attachment, public in scanner.fixed_blocks.values()]
                or any(proof.public() != public or any(getattr(proof, name) is not value for name, value in attached)
                       or getattr(analysis, "_descriptor_blocks", {}).get(proof.identity) is not proof
                       or getattr(analysis, "_descriptor_blocks_owner", None) is not analysis
                       or not _same_declaration(analysis, proof._declaration)
                       for proof, attached, public in scanner.fixed_blocks.values())):
        raise CompilationError("storage BLOCK declaration source changed before publication")
    if scanner.enclosing_context is not None:
        scanner.enclosing_context.finish(record)
    if scanner.record_region is not None:
        scanner.record_region.validate(analysis)
        if record.get("owned_record_region") != scanner.record_region.public():
            raise CompilationError("record storage original association changed before publication")
    declaration_dependencies = scanner.declaration_dependencies.receipt() if (
        scanner.readonly_records or scanner.assignment_namespace_checked or scanner.component_scope_checks) else None
    analysis.verify_publication()
    analysis.inputs.verify_live()
    # No lookup/validation/public serializer follows this final source read.
    from compiler.frontend.descriptor_intrinsics import (
        same_intrinsic_scope_attachments,
        same_intrinsic_source_attachments,
    )
    from compiler.frontend.descriptor_teams import _same_group_syntax

    if (scanner.limits != (analysis.depth_limit, analysis.procedure_limit, analysis.operation_limit)
            or not same_intrinsic_source_attachments(analysis, scanner.source_receipt)
            or any(not same_intrinsic_scope_attachments(saved) for saved in scanner.record_scopes.values())
            or any(not _attached(binding, saved) for binding, saved in scanner.record_declarations.values())
            or any(not _attached(proof, saved) for proof, saved, _public in scanner.fixed_associations.values())
            or any(not _same_group_syntax(saved) for saved in scanner.syntax_receipts.values())):
        raise CompilationError("storage mutation consumed original source context changed at publication")
    if declaration_dependencies is not None:
        declaration_dependencies.require()
    if scanner.enclosing_context is not None:
        scanner.enclosing_context.checkpoint()
    if scanner.joined_closure is not None:
        scanner.joined_closure.checkpoint(complete=True)
    if scanner.record_region is not None:
        team_attached, checkpoint = scanner.record_region_attached
        checkpoint.require()
        if (not _attached(scanner.joined_storage, team_attached)
                or scanner.joined_storage._record_region is not scanner.record_region):
            raise CompilationError("record storage consumed original association changed at publication")
    if any(not _attached(binding, saved) or not same_intrinsic_scope_attachments(scope)
           or not _same_group_syntax(constants)
           for binding, saved, scope, constants in scanner.fixed_array_result_receipts):
        raise CompilationError("storage fixed function result declaration changed at publication")
    if scanner.native_lifetime is not None:
        scanner.native_lifetime.require_checkpoint()
    _require_retained_calls(analysis, calls, terminating)
    for proof in terminating:
        proof.require_attachments()
    if any(not _attached(token, saved) for token, saved in retained):
        raise CompilationError("storage mutation consumed call attachment changed at publication")


def prove_storage_mutation_effects(analysis, procedure, *, expression=None, selection=None, joined_storage=None, native_graph=None):
    from compiler.frontend.source_facts import STORAGE, issue, lookup, token_dependencies

    selection = tuple(selection) if selection is not None else None
    sealed = getattr(analysis, "_source_session", None) is not None
    context = analysis.source_authority_context() if sealed else None
    selected = analysis._selected_source(procedure, selection)[1] if selection is not None else None
    joined_key = ("native_callees", tuple(map(id, joined_storage))) if type(joined_storage) is tuple else id(joined_storage)
    query = procedure, id(expression), selected, joined_key, id(native_graph)
    previous = lookup(analysis, STORAGE, query)
    if previous is not None:
        return previous.validate(analysis, procedure, expression, selection=previous._selection,
                                 joined_storage=joined_storage)
    record, scanner = _derive(analysis, procedure, expression, selection,
                              joined_storage=joined_storage, native_graph=native_graph, _defer_publication=True)
    identity = _hash(record)
    if not hasattr(analysis, "_storage_mutation_effects"):
        analysis._storage_mutation_effects = {}
    registry = analysis._storage_mutation_effects
    if type(registry) is not dict:
        raise CompilationError("storage mutation requires its compiler-owned token registry")
    if (context is not None and identity in registry
            and type(registry[identity]) is StorageMutationEffects
            and registry[identity]._analysis is analysis and registry[identity]._source_fact is None):
        # A pre-seal diagnostic token cannot acquire production authority.
        # The completed fresh scan below issues a distinct immutable handle.
        del registry[identity]
    if identity not in registry:
        if len(registry) >= source_registry_budget(analysis):
            raise CompilationError("storage mutation source projection registry budget exhausted")
        proof = StorageMutationEffects(identity, analysis, procedure, _freeze_source_record(analysis, record),
                                       _ISSUER, expression, selection, joined_storage,
                                       tuple(token for token, _receipt, _fields in scanner.call_receipts),
                                       tuple(scanner.terminating_calls.values()),
                                       tuple((token, saved) for token, _receipt, saved in scanner.call_receipts)
                                       + tuple(scanner.terminating_receipts.values()),
                                       tuple((name, getattr(analysis, name)) for name, used in (
                                           ("_original_call_associations", scanner.call_receipts),
                                           ("_descriptor_terminating_calls", scanner.terminating_calls)) if used),
                                       _native_graph=native_graph,
                                       _native_completion=(scanner.native_lifetime.group_completion
                                           if scanner.native_lifetime is not None else None),
                                       _native_completion_fields=(scanner.native_lifetime._joined[3]
                                           + scanner.native_lifetime._joined[2]
                                           + ((analysis,"_descriptor_team_proofs",scanner.native_lifetime._joined[4]),)
                                           if scanner.native_lifetime is not None and scanner.native_lifetime.group_only else ()),
                                       _native_read_dependencies=(scanner.declaration_dependencies.receipt()
                                           if scanner.native_lifetime is not None and scanner.native_lifetime.group_only else None))
        _publish_derivation(analysis, record, scanner)
        if analysis._storage_mutation_effects is not registry or identity in registry:
            raise CompilationError("storage mutation issuing registry changed before publication")
        # Only now is the completed projection available to other consumers.
        registry[identity] = proof
        if context is not None:
            context.require(analysis)
            retained = token_dependencies((*proof._calls, *proof._terminating_calls,
                *_joined_dependencies(joined_storage), scanner.record_region, *scanner.library_calls.values(),
                *(scanner.native_lifetime.prerequisites() if scanner.native_lifetime is not None else ()),
                *(item[0] for item in scanner.fixed_associations.values()),
                *(item[0] for item in scanner.fixed_blocks.values())))
            proof = issue(analysis, STORAGE, query, proof, retained)
        return proof
    proof = registry[identity]
    # Equal exact selections can arrive in a fresh container. The authenticated
    # record binds their original graph node IDs; the registered token retains
    # its own original selection container for standalone validate().
    retained_selection = proof._selection
    proof._validate_record(analysis, procedure, expression, retained_selection, record, joined_storage)
    _publish_derivation(analysis, record, scanner)
    if analysis._storage_mutation_effects is not registry:
        raise CompilationError("storage mutation issuing registry changed before publication")
    proof._require_registration(analysis, procedure, expression, retained_selection, joined_storage)
    if context is not None:
        context.require(analysis)
        retained = token_dependencies((*proof._calls, *proof._terminating_calls,
            *_joined_dependencies(joined_storage), scanner.record_region, *scanner.library_calls.values(),
                *(scanner.native_lifetime.prerequisites() if scanner.native_lifetime is not None else ()),
            *(item[0] for item in scanner.fixed_associations.values()),
            *(item[0] for item in scanner.fixed_blocks.values())))
        proof = issue(analysis, STORAGE, query, proof, retained)
    return proof
