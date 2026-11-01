"""Exact once-only PURE constructors of private fixed host metadata.

This proof retains the original assignment and guard. It never grants payload
readability, capture, allocation lifetime, GPU execution, or definition facts.
A caller must separately synchronize the listed original metadata reads.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256

from fparser.two.utils import walk

from compiler.frontend.call_bindings import _fixed_array_bounds, resolve_original_function
from compiler.frontend.component_bindings import _field_binding
from compiler.frontend.source_effects import _children, _kind, _part
from compiler.ir import CompilationError

_ISSUER = object()
_NUMERIC = {"integer", "real", "logical"}
_UNSAFE = {"pointer", "target", "allocatable", "optional", "volatile", "asynchronous", "value"}


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _entity(scope, name):
    matches = [entity for statement in walk(_part(scope.node, "Specification_Part"))
               if _kind(statement) == "Type_Declaration_Stmt"
               for entity in _children(statement.items[2]) if str(entity.items[0]).lower() == name]
    if len(matches) != 1:
        raise CompilationError("metadata constructor requires one exact independent declaration")
    return matches[0]


def _literal(node):
    if _kind(node) in {"Int_Literal_Constant", "Real_Literal_Constant", "Logical_Literal_Constant"}:
        return True
    items = _children(node)
    return (len(items) == 2 and str(items[0]) in {"+", "-"} and _literal(items[1]))


def _shape(analysis, binding):
    return tuple(hi - lo + 1 for lo, hi in _fixed_array_bounds(analysis, binding)) if binding.rank else ()


def _guard(analysis, routine, assignment, binding):
    # NAMELIST I/O names its group, hiding reads/writes of the SAVE flag from
    # executable Name references. A reset could reconstruct live allocatables.
    if any(_kind(node) == "Namelist_Stmt"
           for node in walk(_part(routine.scope.node, "Specification_Part"))):
        raise CompilationError("metadata constructor once-only state cannot have NAMELIST association")
    body = tuple(_children(routine.execution))
    guards = [item for item in body if _kind(item) == "If_Construct"
              and any(child is assignment for child in _children(item))]
    if len(guards) != 1:
        raise CompilationError("metadata constructor requires a direct top-level once-only guard")
    guard, = guards
    children = tuple(child for child in _children(guard) if _kind(child) != "Comment")
    if len(children) < 4 or _kind(children[0]) != "If_Then_Stmt" or children[1] is not assignment:
        raise CompilationError("metadata constructor must be first in its original once-only guard")
    condition = children[0].items[0]
    if (len(_children(condition)) != 2 or str(condition.items[0]).lower() != ".not."
            or _kind(condition.items[1]) != "Name"):
        raise CompilationError("metadata constructor requires IF (.NOT. local SAVE flag)")
    flag = analysis._binding(routine.scope, condition.items[1])
    if (flag is None or flag.declaring_scope is not routine.scope or flag.name in routine.arguments
            or flag.dtype != "logical" or flag.rank or "save" not in flag.attributes or flag.attributes & _UNSAFE):
        raise CompilationError("metadata constructor requires an independent local SAVE logical flag")
    init = _entity(routine.scope, flag.name).items[3]
    if init is None or str(init.items[0]) != "=" or str(init.items[1]).lower() != ".false.":
        raise CompilationError("metadata constructor guard must start with a literal false initializer")
    set_flag = children[2]
    if (_kind(set_flag) != "Assignment_Stmt" or _kind(set_flag.items[0]) != "Name"
            or analysis._binding(routine.scope, set_flag.items[0]) is not flag
            or str(set_flag.items[2]).lower() != ".true."):
        raise CompilationError("metadata constructor must immediately set its once-only flag")
    if any(_kind(child) in {"Else_If_Stmt", "Else_Stmt"} for child in children):
        raise CompilationError("metadata constructor once-only guard cannot have alternatives")
    permitted = {id(condition.items[1]), id(set_flag.items[0])}
    if any(_kind(node) == "Name" and analysis._binding(analysis.source_scope_for(node, routine.scope), node) is flag
           and id(node) not in permitted for node in walk(routine.execution)):
        raise CompilationError("metadata constructor flag has another observer, mutation or escape")
    # Exclude alternate entry/control transfer and reentry before setting the
    # flag. Calls later in the guard still keep their independent effect proofs.
    if any(_kind(node) in {"Entry_Stmt", "Goto_Stmt", "Computed_Goto_Stmt", "Assigned_Goto_Stmt"}
           for node in walk(routine.scope.node)):
        raise CompilationError("metadata constructor requires structured single-entry control")
    before = body[:body.index(guard)]
    if any(_kind(node) in {"Call_Stmt", "Part_Ref", "Function_Reference", "Structure_Constructor"}
           for node in walk(before)):
        raise CompilationError("metadata constructor prefix may reenter before once-only initialization")
    receiver_init = _entity(routine.scope, binding.name).items[3]
    if receiver_init is not None:
        raise CompilationError("metadata constructor receiver cannot have another explicit initializer")
    return flag


def _derive(analysis, routine, assignment, opaque):
    opaque.validate(analysis, routine, opaque._node, opaque._resolved)
    lhs, _, expression = assignment.items if _kind(assignment) == "Assignment_Stmt" else (None, None, None)
    if (not any(node is assignment for node in walk(routine.execution)) or _kind(lhs) != "Name"
            or analysis._binding(routine.scope, lhs) is not opaque.binding
            or opaque.public()["constructor"]["statement_sha256"] != sha256(str(assignment).encode()).hexdigest()):
        raise CompilationError("metadata constructor requires its exact qualified opaque receiver assignment")
    flag = _guard(analysis, routine, assignment, opaque.binding)
    resolution = resolve_original_function(analysis, analysis.source_scope_for(expression, routine.scope), expression)
    callee = resolution.callee
    for scope in (routine.scope, callee.scope):
        if analysis._unknown_exports(scope) or analysis._candidates(scope, "assignment(=)"):
            raise CompilationError("metadata constructor requires proved intrinsic assignment without unknown exports")
    header = next(node for node in _children(callee.scope.node) if _kind(node) == "Function_Stmt")
    prefix = {str(node).lower() for node in _children(header.items[0])}
    if "pure" not in prefix or prefix & {"impure", "elemental", "recursive"}:
        raise CompilationError("metadata constructor requires an explicit nonrecursive PURE function")
    result_name = str(header.items[3].items[0] if header.items[3] is not None else header.items[1]).lower()
    result = callee.scope.bindings.get(result_name)
    schema = next(item.actual_type.schema for item in opaque._resolved.associations if item.binding is opaque.binding)
    from compiler.frontend.component_bindings import _schema, _typename
    if (result is None or result.rank or result.attributes & _UNSAFE or schema.reason
            or _schema(analysis, callee.scope, _typename(result.dtype)) is not schema):
        raise CompilationError("metadata constructor requires the exact fixed result type")
    if any(_kind(node) in {"Final_Binding", "Type_Bound_Procedure_Part"} for node in walk(schema.node)):
        raise CompilationError("metadata constructor cannot run finalizers or bound methods")
    if _part(callee.scope.node, "Internal_Subprogram_Part") is not None:
        raise CompilationError("metadata constructor cannot have contained host association")
    if any(_kind(node) == "Namelist_Stmt"
           for node in walk(_part(callee.scope.node, "Specification_Part"))):
        raise CompilationError("metadata constructor cannot have NAMELIST association")
    if set(callee.scope.bindings) != set(callee.arguments) | {result_name}:
        raise CompilationError("metadata constructor permits only formals and its private result")
    fields, defaulted = {}, set()
    for name, (dtype, attrs, entity) in schema.members.items():
        flags = {str(attr).split("(")[0].lower() for attr in attrs}
        init = entity.items[3]
        if flags & {"pointer", "volatile", "asynchronous"}:
            raise CompilationError("metadata constructor has uncertain result component lifetime")
        if "allocatable" in flags:
            if init is not None:
                raise CompilationError("metadata constructor allocatable result components must start unallocated")
            continue
        if _kind(dtype) != "Intrinsic_Type_Spec" or str(dtype.items[0]).lower() not in _NUMERIC:
            raise CompilationError("metadata constructor requires fixed numeric fields or untouched allocatables")
        if init is not None and (str(init.items[0]) != "=" or not _literal(init.items[1])):
            raise CompilationError("metadata constructor field initializer must be a literal")
        if init is not None:
            defaulted.add(name)
        field_binding, _ = _field_binding(analysis, callee.scope, result, name, lhs)
        # Component bindings are not top-level declarations; read fixed extents
        # directly from the original schema without evaluating caller values.
        extents = []
        for axis in field_binding.shape_nodes:
            if _kind(axis) != "Explicit_Shape_Spec":
                raise CompilationError("metadata constructor field requires fixed explicit shape")
            from compiler.ir import SourceLocation
            lo = 1 if axis.items[0] is None else schema.scope.kinds.integer(axis.items[0], SourceLocation(str(schema.scope.path)))
            hi = schema.scope.kinds.integer(axis.items[1], SourceLocation(str(schema.scope.path)))
            if not 0 < hi - lo + 1 <= 256:
                raise CompilationError("metadata constructor field shape exceeds fixed metadata bounds")
            extents.append(hi - lo + 1)
        fields[name] = field_binding, tuple(extents)
    formals, reads = {}, []
    for association in resolution.associations:
        formal, actual, typed = association.formal_binding, association.actual, association.actual_type
        if (actual is None or typed is None or formal.intent != "in" or formal.dtype not in _NUMERIC
                or formal.attributes & _UNSAFE or formal.kind not in {1, 4, 8}):
            raise CompilationError("metadata constructor requires fixed required INTENT(IN) numeric metadata")
        shape = _shape(analysis, formal)
        if formal.rank > 1:
            raise CompilationError("metadata constructor supports only fixed scalar/vector metadata")
        actual_shape = ()
        if typed.rank:
            if _kind(actual) == "Array_Constructor" and _kind(actual.items[1]) == "Ac_Value_List":
                actual_shape = (len(_children(actual.items[1])),)
            elif _kind(actual) == "Name" and typed.binding is not None:
                actual_shape = _shape(analysis, typed.binding)
            else:
                raise CompilationError("metadata constructor array actual requires an exact fixed vector")
        if shape != actual_shape:
            raise CompilationError("metadata constructor actual/formal fixed extents differ")
        for node in walk(actual):
            if _kind(node) != "Name":
                continue
            read = analysis._binding(analysis.source_scope_for(node, routine.scope), node)
            if (read is None or read.dtype not in _NUMERIC or read.attributes & _UNSAFE
                    or read.rank > 1 or read.rank and _kind(actual) != "Name"):
                raise CompilationError("metadata constructor actual requires independent fixed host metadata")
            reads.append({"resource": read.root, "descriptor": read.public(),
                          "bounds": [list(axis) for axis in _fixed_array_bounds(analysis, read)] if read.rank else [],
                          "access": "read at unchanged original constructor point"})
        formals[association.formal] = formal, shape
    writes, used_formals = [], set()
    for statement in _children(callee.execution):
        if _kind(statement) == "Comment":
            continue
        if _kind(statement) != "Assignment_Stmt":
            raise CompilationError("metadata constructor body must contain only direct fixed-field assignments")
        target, _, value = statement.items
        parts = _children(target)
        if (_kind(target) != "Data_Ref" or len(parts) != 2 or any(_kind(node) != "Name" for node in parts)
                or str(parts[0]).lower() != result_name or str(parts[1]).lower() not in fields):
            raise CompilationError("metadata constructor may write only its fixed numeric result fields")
        name = str(parts[1]).lower()
        field_binding, shape = fields[name]
        if _kind(value) == "Name" and str(value).lower() in formals:
            used_formals.add(str(value).lower())
            formal, formal_shape = formals[str(value).lower()]
            if formal.signature() != field_binding.signature() or shape != formal_shape:
                raise CompilationError("metadata constructor result/formal field types or shapes differ")
        elif not _literal(value) or shape:
            raise CompilationError("metadata constructor value must be a formal or scalar literal")
        writes.append(name)
    if (not writes or len(set(writes)) != len(writes) or set(writes) | defaulted != set(fields)
            or used_formals != set(formals)):
        raise CompilationError("metadata constructor must define all fields and unconditionally read every formal")
    graph, selected = analysis._selected_source(routine.qualified, (assignment,))
    return {"version": 1, "procedure": routine.qualified, "source_graph": graph.identity,
            "source_inputs": analysis.inputs.identity(), "selected_node_ids": list(selected),
            "opaque_state": opaque.identity, "constructor": resolution.public(), "once_flag": flag.public(),
            "metadata_reads": reads, "result_fields": writes,
            "execution": "unchanged original guarded assignment; no eager metadata evaluation",
            "capture_authorized": False, "definitions_authorized": False, "payload_access_authorized": False,
            "lifetime_authorized": False, "placement_authorized": False}


@dataclass(frozen=True, slots=True)
class OriginalMetadataConstructorProof:
    identity: str
    _analysis: object = field(repr=False, compare=False)
    _routine: object = field(repr=False, compare=False)
    _assignment: object = field(repr=False, compare=False)
    _opaque: object = field(repr=False, compare=False)
    _record: str = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)

    def validate(self, analysis, routine, assignment):
        if (self._issuer is not _ISSUER or analysis is not self._analysis or routine is not self._routine
                or assignment is not self._assignment
                or getattr(analysis, "_metadata_constructor_proofs", {}).get(self.identity) is not self):
            raise CompilationError("metadata constructor requires its exact compiler-issued original proof")
        if _encoded(_derive(analysis, routine, assignment, self._opaque)) != self._record:
            raise CompilationError("metadata constructor original source, guard or effects changed")
        return self

    def public(self):
        return {"identity": self.identity, **json.loads(self._record)}


def original_metadata_constructor(analysis, routine, assignment, opaque):
    encoded = _encoded(_derive(analysis, routine, assignment, opaque))
    registry = getattr(analysis, "_metadata_constructor_proofs", None)
    if registry is None:
        registry = analysis._metadata_constructor_proofs = {}
    for proof in registry.values():
        if proof._assignment is assignment and proof._opaque is opaque:
            return proof.validate(analysis, routine, assignment)
    proof = OriginalMetadataConstructorProof(sha256(encoded.encode()).hexdigest(), analysis, routine,
                                             assignment, opaque, encoded, _ISSUER)
    registry[proof.identity] = proof
    return proof.validate(analysis, routine, assignment)
