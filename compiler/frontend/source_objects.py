"""Bounded original-source type identity for fixed scalar object arguments.

This proof describes an unchanged Fortran call, not a numeric ABI, an object
layout for copying, or permission to execute a native continuation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256

from fparser.two.utils import Base

from compiler.frontend.component_bindings import _field_binding, _typename
from compiler.ir import CompilationError

SOURCE_OBJECT_VERSION = 1
_UNCERTAIN = {"pointer", "allocatable", "optional", "value", "volatile", "asynchronous",
              "parameter", "external", "intrinsic"}


def _kind(node):
    return type(node).__name__


def _children(node):
    if isinstance(node, (tuple, list)):
        return node
    return getattr(node, "content", getattr(node, "items", ())) or ()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _resolution(scope):
    return _canonical({"imports": scope.imports, "wildcards": scope.wildcards,
                       "exclusions": {name: sorted(values) for name, values in scope.wildcard_exclusions.items()},
                       "ambiguous": sorted(scope.ambiguous_imports), "access": scope.access,
                       "default_public": scope.default_public})


def _kind_state(scope):
    result, seen, kinds = [], set(), scope.kinds
    while kinds is not None and id(kinds) not in seen:
        seen.add(id(kinds))
        result.append((id(kinds), tuple((name, str(value)) for name, value in sorted(kinds.values.items()))))
        kinds = kinds.parent
    return tuple(result)


def _members(schema):
    return tuple((name, id(dtype), tuple(map(id, attrs)), id(entity), str(dtype), str(attrs), str(entity))
                 for name, (dtype, attrs, entity) in schema.members.items())


def _specification(scope):
    return str(next((part for part in _children(scope.node) if _kind(part) == "Specification_Part"), None))


def register_source_object_authority(analysis):
    """Snapshot existing declaration identities; projections borrow these only."""
    scopes = {id(scope): scope for scope in analysis.modules.values()}
    scopes.update((id(routine.scope), routine.scope) for routine in analysis.numerical_helpers.values())
    analysis._source_object_scopes = {
        key: (scope, scope.node, _specification(scope), _resolution(scope),
              {name: (binding, _canonical(binding.public())) for name, binding in scope.bindings.items()
               if _typename(binding.dtype) is not None},
              {name: (schema, schema.node, str(schema.node), _members(schema))
               for name, schema in getattr(scope, "component_types", {}).items()}, _kind_state(scope))
        for key, scope in scopes.items()}


class _TypeBuilder:
    def __init__(self, analysis):
        self.analysis = analysis
        self.visits = 0
        self.nodes = 0
        self.fields = 0
        self.checked = set()

    def scope(self, scope):
        record = self.analysis._source_object_scopes.get(id(scope))
        if record is None:
            raise CompilationError("source object requires original declaration scope authority")
        if id(scope) in self.checked:
            return record
        self.checked.add(id(scope))
        if (record[0] is not scope or record[1] is not scope.node or record[2] != _specification(scope)
                or record[3] != _resolution(scope) or record[6] != _kind_state(scope)):
            raise CompilationError("source object declaration scope changed")
        return record

    def resolve(self, scope, name, active=()):
        if scope is None or (id(scope), name) in active:
            return None
        self.visits += 1
        if self.visits > self.analysis.procedure_limit or len(active) >= self.analysis.depth_limit:
            raise CompilationError("source object type resolution budget exhausted")
        record = self.scope(scope)
        local = getattr(scope, "component_types", {}).get(name)
        if local is not None:
            original = record[5].get(name)
            if (original is None or original[0] is not local or original[1] is not local.node
                    or original[2] != str(local.node)
                    or original[3] != _members(local)):
                raise CompilationError("source object type schema changed")
            return local
        sources = self.analysis.use_sources(scope, name)
        if sources is None:
            return None
        found = {}
        for module, remote in sources:
            owner = self.analysis.modules.get(module)
            if owner is None:
                return None
            self.scope(owner)
            if not self.analysis._exported(owner, remote):
                if scope.imports.get(name) == (module, remote):
                    return None
                continue
            result = self.resolve(owner, remote, (*active, (id(scope), name)))
            if result is not None:
                found[id(result.node)] = result
        if found:
            return next(iter(found.values())) if len(found) == 1 else None
        return self.resolve(scope.parent, name, (*active, (id(scope), name))) if scope.parent else None

    def schema(self, schema, active=()):
        if id(schema.node) in active or len(active) >= self.analysis.depth_limit:
            raise CompilationError("source object nested type depth or recursion requires another proof")
        # Count every parser/traversal entry before inspecting complete type parts.
        pending = [schema.node]
        while pending:
            node = pending.pop()
            self.nodes += 1
            if self.nodes > 4*self.analysis.operation_limit+2:
                raise CompilationError("source object type node budget exhausted")
            if isinstance(node, (Base, tuple, list)):
                pending.extend(_children(node))
        parts = _children(schema.node)
        allowed = {"Derived_Type_Stmt", "Component_Part", "End_Type_Stmt", "Comment",
                   "Private_Components_Stmt", "Sequence_Stmt"}
        if schema.reason or any(_kind(part) not in allowed for part in parts):
            raise CompilationError("source object requires fixed types without extension, parameters or type-bound procedures")
        header = next(part for part in parts if _kind(part) == "Derived_Type_Stmt")
        attributes, _name, parameters = header.items
        if parameters is not None or any(str(flag).lower() not in {"public", "private"} for flag in _children(attributes)):
            raise CompilationError("source object type attributes require another proof")
        from compiler.frontend.source_effects import Binding
        parent = Binding("fort_source_object", "fort_source_object", "type(" + schema.name + ")", None, 0)
        fields = []
        for part in parts:
            if _kind(part) != "Component_Part":
                continue
            for declaration in _children(part):
                if _kind(declaration) in {"Private_Components_Stmt", "Comment"}:
                    continue
                if _kind(declaration) != "Data_Component_Def_Stmt":
                    raise CompilationError("source object contains unsupported component declarations")
                dtype, attrs, entities = declaration.items
                if any(_kind(attr) != "Dimension_Component_Attr_Spec" and str(attr).lower() not in {"public", "private"}
                       for attr in _children(attrs)):
                    raise CompilationError("source object contains descriptor-bearing or unsupported components")
                for entity in _children(entities):
                    self.fields += 1
                    if self.fields > self.analysis.operation_limit:
                        raise CompilationError("source object component budget exhausted")
                    name = str(entity.items[0]).lower()
                    binding, _owner = _field_binding(self.analysis, schema.scope, parent, name, entity,
                                                     consumer_scope=schema.scope)
                    member = {"name": name, "type": binding.dtype, "kind": binding.kind,
                              "shape": [{"lower": lower, "upper": upper} for lower, upper in
                                        self.bounds(schema.scope, binding)]}
                    nested = _typename(dtype)
                    if nested is not None:
                        if binding.rank:
                            raise CompilationError("source object arrays of derived components require another proof")
                        child = self.resolve(schema.scope, nested)
                        if child is None:
                            raise CompilationError("source object nested type is unavailable or ambiguous")
                        member["type"] = self.schema(child, (*active, id(schema.node)))
                    fields.append(member)
        return {"declaration": getattr(schema.scope, "qualified", schema.scope.module) + "::" + schema.name,
                "source": str(schema.scope.path), "definition": str(schema.node), "fields": fields}

    @staticmethod
    def bounds(scope, binding):
        from compiler.ir import SourceLocation
        result = []
        for axis in binding.shape_nodes:
            lower, upper = axis.items
            result.append((1 if lower is None else scope.kinds.integer(lower, SourceLocation(str(scope.path))),
                           scope.kinds.integer(upper, SourceLocation(str(scope.path)))))
        return result

    def binding(self, binding, *, projected_actual=False):
        if binding is None or binding.rank or binding.attributes & _UNCERTAIN:
            raise CompilationError("source object requires fixed nonoptional scalar original storage")
        scope = binding.declaring_scope
        record = self.scope(scope)
        original = record[4].get(binding.name)
        if original is not None and original[0] is not binding and projected_actual:
            projected = binding.public()
            # SourceEffects' private reached projection clears entry INTENT;
            # it does not create storage or change the original type/shape.
            projected["intent"] = original[0].intent
            if (getattr(self.analysis, "_descriptor_source", None) is not None and binding.intent is None
                    and _canonical(projected) == original[1]
                    and all(left is right for left, right in zip(binding.shape_nodes, original[0].shape_nodes, strict=True))):
                binding = original[0]
        if original is None or original[0] is not binding or original[1] != _canonical(binding.public()):
            raise CompilationError("source object requires unchanged original storage declaration")
        name = _typename(binding.dtype)
        schema = self.resolve(scope, name) if name else None
        if schema is None:
            raise CompilationError("source object declaring type is unavailable or ambiguous")
        return schema


@dataclass(frozen=True)
class SourceObjectMapping:
    identity: str
    _record: str = field(repr=False)

    def public(self):
        return json.loads(self._record)


def source_object_mapping(analysis, formal, actual, node, call, scope):
    """Match exact original scalar storage using canonical declaring type nodes."""
    analysis.inputs.verify()
    if _kind(node) != "Name" or formal.intent not in {"in", "inout"}:
        raise CompilationError("source object calls require whole scalar names and IN or INOUT formals")
    builder = _TypeBuilder(analysis)
    original = analysis._source_object_calls.get(id(call))
    if original is None or original[0] is not call or original[1] != str(call):
        raise CompilationError("source object mapping requires an exact unchanged original call")
    builder.scope(original[2])
    if _resolution(scope) != _resolution(original[2]):
        raise CompilationError("source object original call resolution changed")
    left, right = builder.binding(formal), builder.binding(actual, projected_actual=True)
    if left.node is not right.node or left.scope is not right.scope:
        raise CompilationError("source-call canonical declaring type mismatch: " + formal.name)
    definition = builder.schema(left)
    item = getattr(call, "item", None)
    span = getattr(item, "fort_original_span", None) or getattr(item, "span", None)
    record = {"schema_version": SOURCE_OBJECT_VERSION, "kind": "fixed_scalar_source_object",
              "type_identity": sha256(_canonical({"sources": analysis.inputs.identity(), "type": definition}).encode()).hexdigest(),
              "declaring_type": definition["declaration"], "type_definition": definition,
              "formal_resource": formal.root, "actual_resource": actual.root,
              "source_call": {"procedure": getattr(original[2], "qualified", original[2].module),
                              "source_access": str(call),
                              "ordinal": original[3], "span": list(span) if span is not None else None},
              "source_only": True, "numerical_abi": False, "whole_object_copy": False,
              "native_continuation_authority": False}
    identity = sha256(_canonical(record).encode()).hexdigest()
    return SourceObjectMapping(identity, _canonical({**record, "identity": identity}))
