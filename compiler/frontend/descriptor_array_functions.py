"""Storage-only closure evidence for unchanged numeric array functions.

This separate token admits readonly assumed-shape inputs and fixed private
workspace passed to exact source children. It supplies no numerical mapping,
PURE, execution, payload, initialization or motion authority. Ordinary effect
summaries remain unchanged, including unresolved capture lifetime obligations.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from compiler.frontend.call_bindings import (
    _arrange,
    _fixed_array_bounds,
    resolve_original_call,
    resolve_original_function,
)
from compiler.frontend.source_functions import (
    _hash,
    _origin,
    _original,
    _scope_roles,
    _source_identity,
    _validate_resolution,
    _validate_scope,
    _validate_scope_roles,
    _walk,
)
from compiler.frontend.structured_effects import _freeze, _thaw
from compiler.ir import CompilationError
from compiler.ir.intrinsics import ARRAY_INQUIRIES, INTRINSICS, MODEL_INQUIRIES

_ISSUER = object()
_UNSAFE = {"pointer", "target", "optional", "save", "volatile", "asynchronous", "value"}
_ALIASES = {"Common_Stmt", "Equivalence_Stmt", "Namelist_Stmt", "Cray_Pointer_Stmt", "Bind_Stmt", "Codimension_Attr_Spec"}
_INTRINSICS = set(INTRINSICS) | ARRAY_INQUIRIES | MODEL_INQUIRIES | {
    "sum", "product", "any", "all", "count", "minval", "maxval", "dot_product"}
_EXPRESSION = {"Level_2_Unary_Expr", "Level_2_Expr", "Level_3_Expr", "Level_4_Expr", "Level_5_Expr",
               "Add_Operand", "Mult_Operand", "And_Operand", "Or_Operand", "Equiv_Operand",
               "Parenthesis", "Subscript_Triplet", "Section_Subscript_List", "Actual_Arg_Spec_List"}
_OPERATORS = {"+", "-", "*", "/", "**", "(", ")", ".true.", ".false.", ".not.", ".and.",
              ".or.", ".eqv.", ".neqv.", "==", "/=", "<", ">", "<=", ">=", ".eq.", ".ne.",
              ".lt.", ".gt.", ".le.", ".ge."}


def _kind(node):
    return type(node).__name__


def _children(node):
    return node if isinstance(node, (tuple, list)) else getattr(node, "content", getattr(node, "items", ())) or ()


def _part(node, kind):
    return next((item for item in getattr(node, "content", ()) if _kind(item) == kind), None)


class _Reader:
    def __init__(self, analysis, *, resolution_tick=None):
        self.analysis, self.resolution_tick = analysis, resolution_tick
        self.roles, self.scopes, self.members, self.active = _scope_roles(analysis), {}, {}, []
        self.specifications = {}
        self.reads, self.calls, self.operations, self.visits = {}, [], 0, 0

    def tick(self, _role=None):
        self.visits += 1
        if self.visits > 65536:
            raise CompilationError("descriptor array-function original-source scan budget exhausted")
        if self.resolution_tick is not None:
            self.resolution_tick(_role or "syntax")

    def scope(self, scope):
        if id(scope) not in self.scopes:
            _validate_scope(self.analysis, scope)
            self.scopes[id(scope)] = scope

    def specification(self, scope, *, private=False):
        if id(scope) not in self.specifications:
            kinds = set()
            for node in _walk(_part(scope.node, "Specification_Part")):
                self.tick()
                kinds.add(_kind(node))
                if _kind(node) in _ALIASES:
                    raise CompilationError("descriptor array-function declaring storage association is unproved")
            self.specifications[id(scope)] = kinds
        if private and "Data_Stmt" in self.specifications[id(scope)]:
            raise CompilationError("descriptor array-function private DATA storage is persistent")

    def resolve(self, routine, node):
        _validate_resolution(self.analysis, self.analysis.source_scope_for(node, routine.scope), _walk(node),
                             tick=self.tick, _scope_validator=self.scope)

    def operation(self):
        self.operations += 1
        if self.operations > self.analysis.operation_limit:
            raise CompilationError("descriptor array-function closure operation budget exhausted")

    def binding(self, binding, *, local=False, formal=False):
        unsafe = _UNSAFE if local else _UNSAFE - {"save"}
        if (binding is None or binding.dtype not in {"real", "integer", "logical"}
                or binding.kind not in {4, 8} or binding.attributes & unsafe):
            raise CompilationError("descriptor array-function requires ordinary nonassociated numeric storage")
        self.scope(binding.declaring_scope)
        if binding.declaring_scope.bindings.get(binding.name) is not binding:
            raise CompilationError("descriptor array-function original binding authority changed")
        self.specification(binding.declaring_scope)
        if local and "allocatable" in binding.attributes:
            raise CompilationError("descriptor array-function private workspace cannot allocate or escape")
        if binding.rank:
            if formal and binding.intent == "in" and all(_kind(axis) == "Assumed_Shape_Spec" for axis in binding.shape_nodes):
                if "allocatable" in binding.attributes:
                    raise CompilationError("descriptor array-function dummy must be ordinary assumed shape")
            elif local:
                _fixed_array_bounds(self.analysis, binding)
        return binding

    def declarations(self, routine, *, function):
        self.scope(routine.scope)
        self.specification(routine.scope, private=True)
        header = _part(routine.scope.node, "Function_Stmt" if function else "Subroutine_Stmt")
        if any(_kind(node) == "Language_Binding_Spec" for node in _walk(header)):
            raise CompilationError("descriptor array-function foreign procedure ABI is unproved")
        if function:
            prefix = {str(item).lower() for item in getattr(header.items[0], "items", ()) or ()}
            if "impure" in prefix:
                raise CompilationError("descriptor array-function explicit IMPURE requires separate authority")
            name = str(header.items[3].items[0] if header.items[3] is not None else header.items[1]).lower()
            result = self.binding(routine.scope.bindings.get(name), local=True)
            if result.rank:
                raise CompilationError("descriptor array-function result must remain a fixed numeric scalar")
        for binding in routine.scope.bindings.values():
            formal = binding.name in routine.arguments
            self.binding(binding, local=True, formal=formal)
            if formal and (binding.intent != "in" if function else binding.intent not in {"in", "out"}):
                raise CompilationError("descriptor array-function external inputs must be readonly and outputs fixed private workspace")
            if formal and binding.intent == "out" and binding.rank:
                _fixed_array_bounds(self.analysis, binding)
            if formal and binding.intent == "in":
                # Original descriptor setup can evaluate assumed-shape lower
                # bounds. Scan those expressions too; a user function there
                # needs specification-expression authority not supplied here.
                for axis in binding.shape_nodes:
                    if _kind(axis) == "Assumed_Shape_Spec" and axis.items[0] is not None:
                        self.expression(routine, axis.items[0])
        if routine.issues:
            raise CompilationError("descriptor array-function original declarations are incomplete: " + "; ".join(routine.issues))

    def read(self, routine, binding):
        self.binding(binding)
        if (binding.declaring_scope is routine.scope and binding.name not in routine.arguments
                or "parameter" in binding.attributes):
            return
        self.reads[routine.qualified, binding.root] = {"procedure": routine.qualified,
                                    "original_binding": binding.public(), "effect": "readonly original storage",
                                    "descriptor_evidence_only": True, "payload_access_authorized": False,
                                    "requires_original_evaluation_and_lifetime": True,
                                    "persistent_storage_read_only": "save" in binding.attributes}

    def write(self, routine, node):
        self.tick()
        name = node.items[0] if _kind(node) == "Part_Ref" else node
        if _kind(name) != "Name" or _kind(node) not in {"Name", "Part_Ref"}:
            raise CompilationError("descriptor array-function writes require private ordinary storage")
        binding = self.analysis._binding(self.analysis.source_scope_for(node, routine.scope), name)
        self.binding(binding, local=True, formal=binding is not None and binding.name in routine.arguments)
        if (binding.declaring_scope is not routine.scope or "parameter" in binding.attributes
                or binding.name in routine.arguments and binding.intent != "out"):
            raise CompilationError("descriptor array-function caller or module storage write is not readonly")
        if _kind(node) == "Part_Ref":
            if not binding.rank:
                raise CompilationError("descriptor array-function indexed write requires an original array")
            self.expression(routine, node.items[1])

    def invocation(self, routine, node, *, function):
        scope = self.analysis.source_scope_for(node, routine.scope)
        resolved = (resolve_original_function if function else resolve_original_call)(self.analysis, scope, node)
        callee = _original(self.analysis, resolved.procedure)
        actuals, _arguments = _arrange(callee, tuple(_children(node.items[1])), function=function)
        if (resolved.callee is not callee or resolved.formals != callee.arguments
                or len(resolved.actuals) != len(actuals)
                or any(left is not right for left, right in zip(resolved.actuals, actuals, strict=True))
                or len(resolved.associations) != len(actuals)
                or any(item.formal != name or item.actual is not actual
                       or item.formal_binding is not callee.scope.bindings[name]
                       for item, name, actual in zip(resolved.associations, callee.arguments, actuals, strict=True))):
            raise CompilationError("descriptor array-function original argument object authority changed")
        # All writable associations terminate in exact nonassociated private
        # storage (or a fixed OUT dummy already subject to this same rule).
        for association in resolved.associations:
            formal, actual = association.formal_binding, association.actual
            if actual is None:
                raise CompilationError("descriptor array-function arguments must all be present")
            if formal.intent == "out":
                if function or _kind(actual) != "Name":
                    raise CompilationError("descriptor array-function outputs require exact private whole storage")
                self.write(routine, actual)
                binding = association.binding
                if formal.rank and _fixed_array_bounds(self.analysis, formal) != _fixed_array_bounds(self.analysis, binding):
                    raise CompilationError("descriptor array-function private workspace shape mismatch")
                if any(other is not association and other.binding is binding for other in resolved.associations):
                    raise CompilationError("descriptor array-function private workspace overlaps another argument")
            elif formal.intent == "in":
                self.expression(routine, actual)
            else:
                raise CompilationError("descriptor array-function child requires explicit readonly inputs or private outputs")
        self.routine(callee, function=function)
        self.calls.append({"caller": routine.qualified, "source_call": resolved.public(),
                           "unchanged_original_execution": True, "physical_mapping_authorized": False})
        return resolved

    def expression(self, routine, node):
        self.tick()
        if node is None:
            return
        if isinstance(node, str):
            if node.lower() not in _OPERATORS:
                raise CompilationError("descriptor array-function unknown original operator")
            return
        if isinstance(node, (tuple, list)):
            for child in node:
                self.expression(routine, child)
            return
        kind = _kind(node)
        scope = self.analysis.source_scope_for(node, routine.scope)
        if kind in {"Int_Literal_Constant", "Real_Literal_Constant", "Logical_Literal_Constant"}:
            return
        if kind == "Name":
            self.read(routine, self.analysis._binding(scope, node))
        elif kind == "Actual_Arg_Spec":
            self.expression(routine, node.items[1])
        elif kind in {"Part_Ref", "Function_Reference", "Structure_Constructor", "Intrinsic_Function_Reference"}:
            name, arguments = node.items
            binding = self.analysis._binding(scope, name)
            if kind == "Part_Ref" and binding is not None and binding.rank:
                self.read(routine, binding)
                self.expression(routine, arguments)
            elif kind != "Intrinsic_Function_Reference":
                self.invocation(routine, node, function=True)
            else:
                intrinsic = str(name).lower()
                if (binding is not None or self.analysis._candidates(scope, name)
                        or self.analysis._unknown_exports(scope) or intrinsic not in _INTRINSICS):
                    raise CompilationError("descriptor array-function intrinsic resolution is unproved")
                self.expression(routine, arguments)
        elif kind in _EXPRESSION:
            for child in _children(node):
                self.expression(routine, child)
        else:
            raise CompilationError("descriptor array-function original expression requires separate authority: " + kind)

    def body(self, routine, node, depth=0):
        self.tick()
        if depth > self.analysis.depth_limit:
            raise CompilationError("descriptor array-function original control depth exhausted")
        kind = _kind(node)
        if kind in {"Execution_Part", "If_Construct", "Block_Nonlabel_Do_Construct"}:
            for child in _children(node):
                self.body(routine, child, depth + (kind != "Execution_Part"))
            return
        self.operation()
        self.resolve(routine, node)
        if kind == "Assignment_Stmt":
            self.write(routine, node.items[0])
            self.expression(routine, node.items[2])
        elif kind == "Call_Stmt":
            self.invocation(routine, node, function=False)
        elif kind in {"If_Then_Stmt", "Else_If_Stmt"}:
            self.expression(routine, node.items[0])
        elif kind == "If_Stmt":
            self.expression(routine, node.items[0])
            self.body(routine, node.items[1], depth + 1)
        elif kind == "Nonlabel_Do_Stmt":
            control = node.items[1]
            if (control is None or control.items[0] is not None or control.items[1] is None
                    or control.items[3] is not None):
                raise CompilationError("descriptor array-function requires original counted loop control")
            variable, limits = control.items[1]
            self.write(routine, variable)
            self.expression(routine, limits)
        elif kind in {"End_If_Stmt", "Else_Stmt", "End_Do_Stmt", "Continue_Stmt"}:
            return
        elif kind == "Return_Stmt" and all(value is None for value in node.items):
            return  # Complete original suffix is still scanned.
        elif kind == "Comment" and not str(node).lstrip().lower().startswith("!$omp"):
            return
        else:
            raise CompilationError("descriptor array-function closure effect is unproved: " + kind)

    def routine(self, routine, *, function):
        if routine.qualified in self.active or len(self.active) >= self.analysis.depth_limit:
            raise CompilationError("descriptor array-function recursive or deep original closure")
        if routine.qualified in self.members:
            return
        if len(self.members) >= self.analysis.procedure_limit:
            raise CompilationError("descriptor array-function original closure procedure budget exhausted")
        if _original(self.analysis, routine.qualified) is not routine:
            raise CompilationError("descriptor array-function original procedure authority changed")
        self.members[routine.qualified] = routine
        self.declarations(routine, function=function)
        self.active.append(routine.qualified)
        try:
            self.body(routine, routine.execution)
        finally:
            self.active.pop()

    def publish(self):
        for routine in self.members.values():
            if _original(self.analysis, routine.qualified) is not routine:
                raise CompilationError("descriptor array-function original closure changed")
            self.scope(routine.scope)
            _validate_resolution(self.analysis, routine.scope, _walk(routine.scope.node))
        for scope in self.scopes.values():
            _validate_scope(self.analysis, scope)
        _validate_scope_roles(self.analysis, self.roles, self.scopes)
        self.analysis.verify_publication()


def _derive(analysis, procedure, expression, *, resolution_tick=None):
    caller = _original(analysis, procedure)
    position = next((index for index, node in enumerate(_walk(caller.execution)) if node is expression), None)
    if position is None:
        raise CompilationError("descriptor array-function requires its exact original expression")
    reader = _Reader(analysis, resolution_tick=resolution_tick)
    reader.resolve(caller, expression)
    resolved = reader.invocation(caller, expression, function=True)
    reader.publish()
    record = {"source_expression": str(expression), "source_position": position, "source_call": resolved.public(),
              "closure": [{"procedure": name, "source_identity": _source_identity(analysis, routine)}
                          for name, routine in sorted(reader.members.items())],
              "readonly_storage": [reader.reads[key] for key in sorted(reader.reads)], "original_calls": reader.calls,
              "operations": reader.operations, "source_scan_count": reader.visits,
              "complete_original_closure_scanned": True, "requires_original_execution": True,
              "summary_completion_authorized": False, "descriptor_evidence_only": True,
              "argument_evaluation": "unchanged original expression, order and control",
              "execution_authorized": False, "payload_access_authorized": False, "entry_definition_authority": False,
              "capture_authorized": False, "pure_scalar_authorized": False, "gpu_legality_established": False,
              "numerical_lowering_authorized": False, "motion_authorized": False, "alias_authority": False}
    return resolved, record


@dataclass(frozen=True)
class DescriptorArraySourceFunctionProof:
    procedure: str
    callee: str
    identity: str
    source_identity: str
    limits: tuple
    resolved: object = field(repr=False, compare=False)
    _expression: object = field(repr=False, compare=False)
    _record: object = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)

    def public(self):
        return {"schema_version": 2, "proof_role": "descriptor_readonly_array_source_function",
                "proof_identity": self.identity, "procedure": self.procedure, "callee": self.callee,
                "source_identity": self.source_identity, **_thaw(self._record)}

    def validate(self, analysis, procedure, expression, *, resolution_tick=None):
        origin = _origin(analysis)
        if (self._issuer is not _ISSUER or procedure != self.procedure or expression is not self._expression
                or origin._descriptor_functions.get(self.identity) is not self
                or self.limits != (origin.operation_limit, origin.procedure_limit, origin.depth_limit)):
            raise CompilationError("descriptor array-function requires exact registered original authority")
        resolved, record = _derive(origin, procedure, expression, resolution_tick=resolution_tick)
        if (self.resolved is not resolved or resolved.procedure != self.callee
                or _hash(record) != _hash(_thaw(self._record))
                or _source_identity(origin, _original(origin, procedure)) != self.source_identity):
            raise CompilationError("descriptor array-function original closure authority changed")
        return self


def prove_descriptor_array_function(analysis, procedure, expression):
    origin = _origin(analysis)
    resolved, record = _derive(origin, procedure, expression)
    source = _source_identity(origin, _original(origin, procedure))
    identity = _hash({"version": 2, "source": source, "record": record})
    registry = origin._descriptor_functions
    if identity in registry:
        return registry[identity].validate(origin, procedure, expression)
    if len(registry) >= origin.operation_limit:
        raise CompilationError("descriptor array-function proof registry budget exhausted")
    proof = DescriptorArraySourceFunctionProof(procedure, resolved.procedure, identity, source,
        (origin.operation_limit, origin.procedure_limit, origin.depth_limit), resolved, expression, _freeze(record), _ISSUER)
    registry[identity] = proof
    return proof
