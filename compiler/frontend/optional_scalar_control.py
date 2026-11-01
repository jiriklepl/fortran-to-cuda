"""A standard PRESENT guard with only original scalar-local assignments.

Both presence alternatives leave incoming array definitions untouched. This
source token neither evaluates presence nor replaces or reorders the control.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from hashlib import sha256

from fparser.two.utils import walk

from compiler.frontend.source_effects import _children, _kind, _part
from compiler.frontend.source_functions import _original, _resolution, _validate_scope
from compiler.ir import CompilationError

_ISSUER = object()
_UNSAFE = {"pointer", "target", "allocatable", "optional", "volatile", "asynchronous", "value", "parameter"}
_EXPRESSIONS = {"Name", "Int_Literal_Constant", "Real_Literal_Constant", "Logical_Literal_Constant",
                "Level_2_Expr", "Level_2_Unary_Expr", "Add_Operand", "Mult_Operand", "Power_Operand",
                "Parenthesis", "Level_1_Expr", "Level_4_Expr", "Relational_Expr", "And_Operand",
                "Or_Operand", "Equiv_Operand"}


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _scope_state(scope):
    # Retain actual objects; integer IDs alone can be recycled during callbacks.
    return (scope, scope.node, scope.parent, scope.bindings, tuple(scope.bindings.items()), _resolution(scope))


def _same_scope_state(state):
    scope, node, parent, bindings, originals, resolution = state
    return (scope.node is node and scope.parent is parent and scope.bindings is bindings
            and len(bindings) == len(originals) and all(bindings.get(name) is binding for name, binding in originals)
            and _resolution(scope) == resolution)


def is_optional_scalar_control(node):
    if _kind(node) != "If_Construct" or not node.content or _kind(node.content[0]) != "If_Then_Stmt":
        return False
    expression = node.content[0].items[0]
    return (_kind(expression) == "Intrinsic_Function_Reference" and str(expression.items[0]).lower() == "present")


def _derive(analysis, procedure, node):
    routine = _original(analysis, procedure)
    scope = routine.scope
    _validate_scope(analysis, scope)
    if not is_optional_scalar_control(node):
        raise CompilationError("optional scalar control requires its original PRESENT guard")
    body = tuple(part for part in node.content if _kind(part) != "Comment")
    if ([_kind(part) for part in body] != ["If_Then_Stmt", "Assignment_Stmt", "Else_Stmt",
                                           "Assignment_Stmt", "End_If_Stmt"]
            or any(_kind(part) == "Comment" and str(part).lstrip().startswith("!$") for part in walk(node))):
        raise CompilationError("optional scalar control requires exactly two original scalar assignments")
    condition = body[0].items[0]
    owner = scope
    while owner is not None:
        if any("present" in names for names in (owner.externals, owner.procedure_arguments,
                owner.procedures, owner.generics, owner.ambiguous_imports)):
            raise CompilationError("optional scalar control requires unshadowed standard PRESENT")
        owner = owner.parent
    if (analysis._binding(scope, "present") or analysis._candidates(scope, "present")
            or analysis._unknown_exports(scope)):
        raise CompilationError("optional scalar control requires unshadowed standard PRESENT")
    arguments = _children(condition.items[1])
    if len(arguments) != 1 or _kind(arguments[0]) != "Name":
        raise CompilationError("optional scalar control requires one original optional scalar dummy")
    optional = analysis._binding(scope, arguments[0])
    if (optional is None or optional.declaring_scope is not scope or optional.name not in routine.arguments
            or "optional" not in optional.attributes or optional.rank or optional.dtype not in {"real", "integer", "logical"}
            or optional.kind not in {4, 8} or optional.attributes & (_UNSAFE - {"optional"})):
        raise CompilationError("optional scalar control requires an ordinary numeric or logical optional scalar")
    if any(_kind(item) in {"Common_Stmt", "Equivalence_Stmt", "Cray_Pointer_Stmt", "Bind_Stmt"}
           for item in walk(_part(scope.node, "Specification_Part"))):
        raise CompilationError("optional scalar control requires independent original local outputs")
    outputs, scopes = [], {id(scope): scope}
    for ordinal, statement in enumerate((body[1], body[3])):
        target = statement.items[0]
        binding = analysis._binding(scope, target) if _kind(target) == "Name" else None
        if (binding is None or binding.declaring_scope is not scope or binding.name in routine.arguments
                or binding.rank or binding.dtype not in {"real", "integer", "logical"}
                or binding.kind not in {4, 8} or binding.attributes & _UNSAFE):
            raise CompilationError("optional scalar control can only assign ordinary original local scalars")
        outputs.append(binding)
        for expression in walk(statement.items[2]):
            if expression is None or isinstance(expression, str):
                continue
            if _kind(expression) not in _EXPRESSIONS:
                raise CompilationError("optional scalar control expressions cannot call or reference array payloads")
            if _kind(expression) == "Name":
                read = analysis._binding(scope, expression)
                if (read is None or read.rank or read.dtype not in {"real", "integer", "logical"}
                        or read.kind not in {4, 8} or read.attributes & (_UNSAFE - {"optional", "parameter"})
                        or "optional" in read.attributes and (ordinal != 0 or read is not optional)):
                    raise CompilationError("optional scalar control reads require ordinary scalars and a reached PRESENT arm")
                _validate_scope(analysis, read.declaring_scope)
                scopes[id(read.declaring_scope)] = read.declaring_scope
    if outputs[0] is not outputs[1]:
        raise CompilationError("optional scalar control alternatives must assign the same original local scalar")
    graph, selected = analysis._selected_source(procedure, (node,))
    record = {"version": 1, "kind": "original_present_scalar_local_control", "procedure": procedure,
              "source_inputs": analysis.inputs.identity(), "source_graph": graph.identity,
              "selected_node_ids": list(selected), "original_source_sha256": sha256(str(node).encode()).hexdigest(),
              "optional_dummy": optional.public(), "local_output": outputs[0].public(),
              "presence": "both original alternatives retained; optional read only in the PRESENT arm",
              # A scalar actual can alias an element of a caller's array.
              # Reading it preserves definedness but still needs coherence.
              "incoming_array_definition_changes": [],
              "scalar_accesses_require_independent_native_coherence": True,
              "new_definition_authority": False, "execution_authorized": False,
              "payload_access_authorized": False, "descriptor_lifetime_authority": False,
              "alias_authority": False, "source_motion_authorized": False}
    for consulted in scopes.values():
        _validate_scope(analysis, consulted)
    return _encoded(record), tuple(scopes.values())


@dataclass(frozen=True, slots=True)
class OriginalOptionalScalarControlProof:
    identity: str
    _analysis: object = field(repr=False, compare=False)
    _procedure: str
    _node: object = field(repr=False, compare=False)
    _scopes: tuple = field(repr=False, compare=False)
    _record: str = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)

    def public(self):
        return {"identity": self.identity, **json.loads(self._record)}

    def validate(self, analysis, procedure, node):
        attached = tuple((part.name, getattr(self, part.name)) for part in fields(self))
        scope_states = tuple(_scope_state(scope) for scope in self._scopes)
        if (type(self) is not OriginalOptionalScalarControlProof or self._issuer is not _ISSUER
                or analysis is not self._analysis or procedure != self._procedure or node is not self._node
                or getattr(analysis, "_optional_scalar_controls_owner", None) is not analysis
                or getattr(analysis, "_optional_scalar_controls", {}).get(self.identity) is not self):
            raise CompilationError("optional scalar control requires its exact issued original source")
        record, scopes = _derive(analysis, procedure, node)
        if (record != self._record or len(scopes) != len(self._scopes)
                or any(a is not b for a, b in zip(scopes, self._scopes, strict=True))
                or sha256(self._record.encode()).hexdigest() != self.identity):
            raise CompilationError("optional scalar control changed its source or scalar declarations")
        analysis.verify_publication()
        analysis.verify_source()
        analysis.inputs.verify_live()
        for scope in scopes:
            _validate_scope(analysis, scope)
        if (any(getattr(self, name) is not value for name, value in attached)
                or not all(_same_scope_state(state) for state in scope_states)
                or getattr(analysis, "_optional_scalar_controls_owner", None) is not analysis
                or getattr(analysis, "_optional_scalar_controls", {}).get(self.identity) is not self):
            raise CompilationError("optional scalar control changed at final publication")
        return self


def original_optional_scalar_control(analysis, procedure, node):
    record, scopes = _derive(analysis, procedure, node)
    identity = sha256(record.encode()).hexdigest()
    if getattr(analysis, "_optional_scalar_controls_owner", None) is not analysis:
        analysis._optional_scalar_controls_owner = analysis
        analysis._optional_scalar_controls = {}
    registry = analysis._optional_scalar_controls
    if identity not in registry:
        registry[identity] = OriginalOptionalScalarControlProof(identity, analysis, procedure, node, scopes, record, _ISSUER)
    return registry[identity].validate(analysis, procedure, node)
