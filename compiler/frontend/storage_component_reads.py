"""Declaration-only numeric component reads for the storage-effects scanner.

This does not describe a layout or authorize payload access. The unchanged
native expression retains allocation checks, value reads and exception behavior.
Read occurrences and original numeric CALL actuals use separate entry points.
Neither grants component writes, aggregate copies or payload access.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256

from fparser.two.utils import Base

from compiler.frontend.call_bindings import OriginalActualType, _OriginalTypes
from compiler.frontend.component_bindings import _typename
from compiler.frontend.source_functions import _original, _scope_roles, _validate_scope_roles
from compiler.ir import CompilationError
from compiler.ir.intrinsics import ARRAY_INQUIRIES, INTRINSICS, MODEL_INQUIRIES

_UNSAFE = {"pointer", "target", "volatile", "asynchronous", "optional", "value"}
_ALIASES = {"Common_Stmt", "Equivalence_Stmt", "Namelist_Stmt", "Cray_Pointer_Stmt", "Bind_Stmt",
            "Procedure_Declaration_Stmt", "Proc_Declaration_Stmt"}
_READ_INTRINSICS = set(INTRINSICS) | ARRAY_INQUIRIES | MODEL_INQUIRIES | {
    "allocated", "shape", "sum", "product", "any", "all", "count", "minval", "maxval",
    "dot_product", "matmul", "transpose", "minloc", "maxloc", "norm2",
}
STORAGE_COMPONENT_READ_POSITION_VERSION = 1


def _kind(node):
    return type(node).__name__


def _children(node):
    return node if isinstance(node, (tuple, list)) else getattr(node, "content", getattr(node, "items", ())) or ()


def _original_position(analysis, routine, expression, tick):
    """Find one exact executable occurrence in its original routine."""
    if _original(analysis, routine.qualified) is not routine:
        raise CompilationError("storage component read requires the exact original routine")
    pending, positions = [(routine.execution, ())], []
    while pending:
        node, parents = pending.pop()
        if not isinstance(node, (Base, tuple, list)):
            continue
        tick()
        if node is expression:
            positions.append(parents)
        pending.extend((child, (*parents, (node, index)))
                       for index, child in reversed(tuple(enumerate(_children(node)))))
    if len(positions) != 1:
        raise CompilationError("storage component read requires one exact original executable occurrence")
    return positions[0]


def _read_position(analysis, routine, expression, tick):
    """Authenticate the exact occurrence and reject write/address contexts."""
    position = _original_position(analysis, routine, expression, tick)
    for offset, (parent, branch) in enumerate(position):
        kind = _kind(parent)
        if kind in {"Io_Control_Spec", "Connect_Spec", "Close_Spec", "Wait_Spec", "Position_Spec", "Flush_Spec"}:
            keyword = str(parent.items[0]).lower() if parent.items[0] is not None else ""
            if keyword in {"iostat", "iomsg", "newunit", "size", "id"}:
                raise CompilationError("storage component I/O output/borrowed designator is not a read occurrence")
        if kind in {"Inquire_Stmt", "Inquire_Spec"}:
            raise CompilationError("storage component I/O inquiry designator requires separate write-context authority")
        # A field used to select an element of an original array is evaluated,
        # even when that array element is the assignment target. The selected
        # component itself remains a write; only an enclosing array subscript
        # separates this occurrence from the target designator.
        lhs_index = kind == "Assignment_Stmt" and branch == 0 and any(
            _kind(array) == "Part_Ref" and child == 1
            and (binding := analysis._binding(analysis.source_scope_for(array, routine.scope), array.items[0])) is not None
            and binding.rank
            for array, child in position[offset + 1:])
        if ((kind == "Assignment_Stmt" and branch == 0 and not lhs_index)
                or kind in {"Pointer_Assignment_Stmt", "Allocate_Stmt", "Deallocate_Stmt", "Nullify_Stmt", "Read_Stmt"}):
            raise CompilationError("storage component writes require a separate descriptor-mutation classification")
        if (kind in {"Call_Stmt", "Proc_Component_Ref", "Coindexed_Named_Object", "Level_1_Expr"}
                or kind.startswith("Defined_")
                or any(_kind(child).startswith("Defined_") for child in _children(parent))):
            raise CompilationError("storage component address/defined-operation escape requires separate authority")
        if kind in {"Part_Ref", "Function_Reference", "Intrinsic_Function_Reference", "Structure_Constructor"}:
            name = parent.items[0]
            scope = analysis.source_scope_for(parent, routine.scope)
            binding = analysis._binding(scope, name)
            if kind == "Part_Ref" and binding is not None and binding.rank:
                continue
            if (kind != "Intrinsic_Function_Reference" or binding is not None
                    or analysis._candidates(scope, name) or analysis._unknown_exports(scope)
                    or str(name).lower() not in _READ_INTRINSICS):
                raise CompilationError("storage component procedure/address escape requires separate authority")
    return position


def _actual_position(analysis, routine, node, resolved, association, *, tick):
    """An exact scalar field actual, independently of the called procedure's effects."""
    from compiler.frontend.call_bindings import ResolvedOriginalCall

    if (type(resolved) is not ResolvedOriginalCall or resolved._caller is not routine
            or _kind(resolved._original) != "Call_Stmt"
            or not any(item is association for item in resolved.associations)
            or association.actual is not node):
        raise CompilationError("storage component actual requires its exact original CALL association")
    resolved.validate(analysis, resolved._scope, resolved._original)
    formal = association.formal_binding
    if (formal.dtype not in {"real", "integer", "logical"} or formal.kind not in {1, 2, 4, 8, 16}
            or formal.rank or formal.intent not in {"in", "out", "inout"}
            or formal.attributes & (_UNSAFE | {"allocatable"})
            or _kind(node) != "Data_Ref" or _kind(node.items[0]) != "Part_Ref"):
        raise CompilationError("storage component actual requires a required numeric scalar formal and fixed array field")
    position = _original_position(analysis, routine, node, tick)
    if not any(parent is resolved._original for parent, _branch in position):
        raise CompilationError("storage component actual is outside its exact original CALL")
    return position


class _LexicalAliasTypes(_OriginalTypes):
    """One exact source-proved lexical declaration, never a general resolver."""
    def __init__(self, analysis, alias, declaration_scope):
        super().__init__(analysis)
        self.alias, self.declaration_scope = alias, declaration_scope

    def declared(self, binding):
        if binding is not self.alias:
            return super().declared(binding)
        self.scope(self.declaration_scope)
        return OriginalActualType(binding.dtype, binding.kind, binding.rank, binding,
                                  self.schema(self.declaration_scope, binding.dtype))


def readonly_record_declaration(analysis, binding, *, retain_scope, tick, formal=True):
    """Classify original record storage; never its value, layout or allocation.

    The only consumable dummy is a scalar nonoptional INTENT(IN) variable.
    Array roots are admitted solely as actual storage by the caller's existing
    exact scalar-element association. No pointer, finalizer or foreign layout
    enters this first read-only descriptor slice.
    """
    from compiler.frontend.descriptor_associations import _attached, _attachments
    from compiler.frontend.source_objects import _canonical, _kind_state, _members, _resolution, _specification

    if (binding is None or _typename(binding.dtype) is None or binding.attributes & _UNSAFE
            or formal and (binding.rank or binding.intent != "in" or "allocatable" in binding.attributes)):
        raise CompilationError("storage record requires an original scalar INTENT(IN) dummy and unassociated actual")
    attached = _attachments(binding)
    owner = binding.declaring_scope
    # Match the immutable original declaration even when the analysis is sealed;
    # namespace pointer equality alone cannot authenticate a changed Binding.
    original = analysis._source_object_scopes.get(id(owner))
    declared = None if original is None else original[4].get(binding.name)
    if declared is None or declared[0] is not binding or declared[1] != _canonical(binding.public()):
        raise CompilationError("storage record original declaration changed")
    def retain_original(scope):
        retain_scope(scope)
        saved = analysis._source_object_scopes.get(id(scope))
        if (saved is None or saved[0] is not scope or saved[1] is not scope.node
                or saved[2] != _specification(scope) or saved[3] != _resolution(scope)
                or saved[6] != _kind_state(scope)):
            raise CompilationError("storage record original declaring namespace changed")
        specification = next((part for part in _children(scope.node) if _kind(part) == "Specification_Part"), None)
        # Reuse original immutable expected parser rows, including scalar
        # leaves whose cached spelling alone cannot authenticate their items.
        # This shallow declaration subtree excludes unrelated procedure bodies.
        if specification is not None and not analysis._source_syntax.original_subtree_unchanged(specification):
            raise CompilationError("storage record original specification syntax changed")

    retain_original(owner)
    types = _OriginalTypes(analysis)
    schema = types.schema(owner, binding.dtype)
    visited = set()

    def inspect(schema, active=()):
        if id(schema) in active or len(active) >= analysis.depth_limit:
            raise CompilationError("storage record recursive type or depth requires separate authority")
        if id(schema) in visited:
            return
        visited.add(id(schema))
        retain_original(schema.scope)
        # Retain the complete existing declaring schema, not a new field/layout
        # model. The source authority snapshot exists before this service.
        original = analysis._source_object_scopes.get(id(schema.scope))
        saved = None if original is None else original[5].get(schema.name)
        if (saved is None or saved[0] is not schema or saved[1] is not schema.node
                or saved[2] != str(schema.node) or saved[3] != _members(schema)):
            raise CompilationError("storage record original component schema changed")
        parts = _children(schema.node)
        if schema.reason or any(_kind(part) not in {"Derived_Type_Stmt", "Component_Part", "End_Type_Stmt",
                "Comment", "Private_Components_Stmt"} for part in parts):
            raise CompilationError("storage record requires types without extension, finalization or bound procedures")
        header = next(part for part in parts if _kind(part) == "Derived_Type_Stmt")
        attributes, _name, parameters = header.items
        if parameters is not None or any(str(flag).lower() not in {"public", "private"} for flag in _children(attributes)):
            raise CompilationError("storage record foreign or parameterized type requires separate authority")
        for part in parts:
            if _kind(part) != "Component_Part":
                continue
            for declaration in _children(part):
                tick()
                if _kind(declaration) in {"Comment", "Private_Components_Stmt"}:
                    continue
                if _kind(declaration) != "Data_Component_Def_Stmt":
                    raise CompilationError("storage record component declaration requires separate authority")
                dtype, attributes, entities = declaration.items
                flags = {str(item).split("(")[0].lower() for item in _children(attributes)}
                if flags - {"public", "private", "allocatable", "dimension"}:
                    raise CompilationError("storage record pointer or associated component requires separate authority")
                if _typename(dtype) is not None:
                    inspect(types.schema(schema.scope, str(dtype).lower()), (*active, id(schema)))
                elif _kind(dtype) != "Intrinsic_Type_Spec" or str(dtype.items[0]).lower() not in {
                        "real", "integer", "logical", "double precision"}:
                    raise CompilationError("storage record requires primitive numeric or declared record components")
                for _entity in _children(entities):
                    tick()
        for scope in types.scopes.values():
            retain_original(scope)
        spec = next((part for part in _children(schema.scope.node) if _kind(part) == "Specification_Part"), None)
        pending = [spec]
        while pending:
            node = pending.pop()
            if not isinstance(node, (Base, tuple, list)):
                continue
            tick()
            if _kind(node) in _ALIASES or _kind(node) == "Comment" and str(node).lstrip().startswith("!$"):
                raise CompilationError("storage record declaring alias/concurrency requires separate authority")
            pending.extend(_children(node))

    inspect(schema)
    if not _attached(binding, attached) or binding.declaring_scope is not owner:
        raise CompilationError("storage record original declaration changed during traversal")
    for _ in range(types.visits + types.types.visits):
        tick()
    return schema


def record_array_declaration(analysis, binding, *, retain_scope, tick):
    """Authenticate original array storage, never record values or ownership.

    An inquiry of the root or read of one fixed numeric field does not touch
    unrelated allocatable components. Selected fields still need the separate
    path checks below; aggregate copies and calls receive no authority here.
    """
    if binding is None or not binding.rank:
        raise CompilationError("storage record array requires its original array declaration")
    return readonly_record_declaration(analysis, binding, retain_scope=retain_scope, tick=tick, formal=False)


def _component_scope_syntax(scope, tick):
    """Static declaration exclusions; a sealed scanner may retain this completed check."""
    spec = next((part for part in _children(scope.node) if _kind(part) == "Specification_Part"), None)
    pending = [spec]
    while pending:
        declaration = pending.pop()
        if not isinstance(declaration, (Base, tuple, list)):
            continue
        tick()
        if (_kind(declaration) in _ALIASES or _kind(declaration) == "Comment"
                and str(declaration).lstrip().startswith("!$")):
            raise CompilationError("storage component declaring alias/concurrency requires separate authority")
        pending.extend(_children(declaration))


def _describe(analysis, routine, node, tick, lexical_alias=None, readonly_record=None, scope_scan=None):
    if _kind(node) != "Data_Ref" or len(node.items) < 2 or _kind(node.items[0]) not in {"Name", "Part_Ref"}:
        raise CompilationError("storage component read requires a scalar original object and fixed field path")
    scope = analysis.source_scope_for(node, routine.scope)
    head = node.items[0]
    array_element = _kind(head) == "Part_Ref"
    root = analysis._binding(scope, head.items[0] if array_element else head)
    types = _OriginalTypes(analysis)
    if lexical_alias is not None:
        from compiler.frontend.descriptor_associations import FixedScalarAssociation

        if (type(lexical_alias) is not FixedScalarAssociation or lexical_alias._analysis is not analysis
                or lexical_alias._routine is not routine):
            raise CompilationError("storage component requires its exact original lexical association")
        lexical_alias.validate()
        declaration = lexical_alias._association
        matches = [access for name, access in zip(declaration.scope.bindings, declaration.selectors, strict=True)
                   if declaration.scope.bindings[name] is root]
        if len(matches) != 1:
            raise CompilationError("storage component alias is outside its original lexical declaration")
        types = _LexicalAliasTypes(analysis, root, matches[0].root_binding.declaring_scope)
    typed = types.actual(scope, node)
    if (root is None or _typename(root.dtype) is None or root.attributes & _UNSAFE
            or not array_element and (root.rank or "allocatable" in root.attributes)):
        raise CompilationError("storage component parent association requires separate authority")
    if array_element and (not root.rank or lexical_alias is not None
            or types.indices(scope, head, root.rank, 0)):
        raise CompilationError("storage component array root requires one original scalar element")
    parent_type = types.declared(root)
    schema, fields, identities, indices = parent_type.schema, [], [], []
    if array_element:
        indices.extend(_children(head.items[1]))
    for ordinal, part in enumerate(node.items[1:], 1):
        tick()
        if schema is None or _kind(part) not in {"Name", "Part_Ref"}:
            raise CompilationError("storage component requires an exact nonpolymorphic field declaration")
        member = str(part.items[0] if _kind(part) == "Part_Ref" else part).lower()
        declaration = schema.members.get(member)
        if declaration is None:
            raise CompilationError("storage component original field is unavailable")
        dtype, attributes, entity = declaration
        flags = frozenset(str(item).split("(")[0].lower() for item in attributes)
        if flags & _UNSAFE:
            raise CompilationError("storage component pointer/alias/concurrency attributes require separate authority")
        if array_element and "allocatable" in flags:
            raise CompilationError("storage record array selected component requires separate allocation ownership authority")
        if (schema.private or "private" in flags) and scope.module != schema.scope.module:
            raise CompilationError("storage component is private in its declaring module")
        dimension = next((attr.items[1] for attr in attributes if _kind(attr) == "Dimension_Component_Attr_Spec"), None)
        shape = entity.items[1] if entity.items[1] is not None else dimension
        rank = len(_children(shape))
        terminal = ordinal == len(node.items) - 1
        if not terminal:
            if _typename(dtype) is None:
                raise CompilationError("storage component intermediate object must be fixed scalar storage")
            if rank or "allocatable" in flags:
                if readonly_record is not root or _kind(part) != "Part_Ref" or not rank:
                    raise CompilationError("storage component intermediate object requires exact readonly record authority")
                # Use the same typed scalar-index rule as original call binding.
                # Every intermediate array must be reduced to one original
                # element; sections/vector subscripts never supply this proof.
                if types.indices(scope, part, rank, 0):
                    raise CompilationError("storage component intermediate array must select one scalar element")
        if _kind(part) == "Part_Ref":
            indices.extend(_children(part.items[1]))
        fields.append({"member": member, "declaring_module": schema.scope.module, "declaring_type": schema.name,
                       "declaration": str(entity), "attributes": sorted(flags), "declared_rank": rank,
                       "declaring_type_sha256": sha256(str(schema.node).encode()).hexdigest()})
        identities.append((schema, schema.node, dtype, tuple(attributes), entity))
        if not terminal:
            schema = types.schema(schema.scope, str(dtype).lower())
    if typed.dtype not in {"real", "integer", "logical"} or typed.kind not in {1, 2, 4, 8, 16}:
        raise CompilationError("storage component terminal must have a known original numeric kind")
    for consulted in types.scopes.values():
        if _kind(consulted.node) == "Module" and analysis.modules.get(consulted.module) is not consulted:
            raise CompilationError("storage component original declaring namespace authority changed")
        if scope_scan is None:
            _component_scope_syntax(consulted, tick)
        else:
            scope_scan(consulted, lambda scope=consulted: _component_scope_syntax(scope, tick))
    for _ in range(types.visits + types.types.visits):
        tick()
    return root, typed, fields, tuple(identities), tuple(indices), types.scopes


@dataclass(frozen=True)
class NumericComponentRead:
    root_binding: object = field(repr=False, compare=False)
    resource: str
    indices: tuple = field(repr=False, compare=False)
    _record: str = field(repr=False)

    def public(self):
        return json.loads(self._record)


def numeric_component_read(analysis, routine, node, *, visit_index, visit_scope, tick, lexical_alias=None,
                           readonly_record=None, _scope_scan=None):
    return _numeric_component(analysis, routine, node, visit_index=visit_index, visit_scope=visit_scope,
        tick=tick, lexical_alias=lexical_alias, readonly_record=readonly_record, scope_scan=_scope_scan)


def numeric_component_actual(analysis, routine, resolved, association, *, visit_index, visit_scope, tick,
                             _scope_scan=None):
    """Original fixed scalar field association; complete callee effects remain separate."""
    return _numeric_component(analysis, routine, association.actual, visit_index=visit_index,
        visit_scope=visit_scope, tick=tick, call_association=(resolved, association), scope_scan=_scope_scan)


def _numeric_component(analysis, routine, node, *, visit_index, visit_scope, tick, lexical_alias=None,
                       readonly_record=None, call_association=None, scope_scan=None):
    """Fresh scanner service, never a reusable proof/capture/layout token.

    ``visit_index`` must be the calling scanner's ordinary expression visitor;
    ``visit_scope`` joins its final namespace publication fence. Both run inside
    that scanner's existing transaction and source-sized traversal budget.
    """
    from compiler.frontend.descriptor_associations import _attached, _attachments
    from compiler.frontend.source_objects import _TypeBuilder

    original_root = None
    array_element = _kind(node) == "Data_Ref" and _kind(node.items[0]) == "Part_Ref"
    if lexical_alias is None:
        if _kind(node) != "Data_Ref" or len(node.items) < 2 or _kind(node.items[0]) not in {"Name", "Part_Ref"}:
            raise CompilationError("storage component read requires a scalar original object and fixed field path")
        original_scope = analysis.source_scope_for(node, routine.scope)
        head = node.items[0]
        original_root = analysis._binding(original_scope, head.items[0] if array_element else head)
        if original_root is None:
            raise CompilationError("storage component read requires its original object declaration")
        original_fields = _attachments(original_root)
        declaring_scope = original_root.declaring_scope
        # A sealed Binding still needs its immutable original public record.
        # Merely finding the same pointer in a namespace cannot authenticate a
        # canonical resource that changed before the first traversal callback.
        # binding() resolves the declaring type, without asking for its layout.
        if array_element:
            record_array_declaration(analysis, original_root, retain_scope=visit_scope, tick=tick)
        else:
            _TypeBuilder(analysis).binding(original_root)
        if not _attached(original_root, original_fields) or original_root.declaring_scope is not declaring_scope:
            raise CompilationError("storage component original object changed before read traversal")
    if lexical_alias is not None:
        from compiler.frontend.descriptor_associations import (
            FixedScalarAssociation,
            _same_syntax,
            _syntax_attachments,
        )

        if type(lexical_alias) is not FixedScalarAssociation:
            raise CompilationError("storage component requires its typed original lexical association")
        alias_fields = _attachments(lexical_alias)
        declaration = lexical_alias._association
        declaration_fields = _attachments(declaration)
        alias_map = declaration.scope.bindings
        aliases = tuple((name, binding, _attachments(binding), getattr(binding, "component_object", None),
                         getattr(binding, "associate_selector", None)) for name, binding in alias_map.items())
        header_syntax = _syntax_attachments(lexical_alias._original.content[0])
    if readonly_record is not None:
        if readonly_record is not original_root or lexical_alias is not None:
            raise CompilationError("storage component requires its exact original readonly record root")
        readonly_record_declaration(analysis, readonly_record, retain_scope=visit_scope, tick=tick)
    if array_element:
        if lexical_alias is not None or readonly_record is not None:
            raise CompilationError("storage array element requires its own original array declaration")
        record_array_declaration(analysis, original_root, retain_scope=visit_scope, tick=tick)
    roles = _scope_roles(analysis)
    def occurrence():
        if call_association is None:
            return _read_position(analysis, routine, node, tick)
        return _actual_position(analysis, routine, node, *call_association, tick=tick)

    position = occurrence()
    if lexical_alias is not None and not any(parent is lexical_alias._original for parent, _branch in position):
        raise CompilationError("storage component alias occurrence is outside its original complete construct")
    # The new record path scans every original intermediate/terminal index
    # through the caller before bounded type inspection can decline an index
    # form. Known child mutation must not disappear behind that frontier.
    if readonly_record is not None or array_element:
        for part in node.items:
            if _kind(part) == "Part_Ref":
                for index in _children(part.items[1]):
                    visit_index(index)
    root, typed, field_records, identities, indices, scopes = _describe(
        analysis, routine, node, tick, lexical_alias, readonly_record, scope_scan)
    if call_association is not None and (typed.rank or root is not call_association[1].binding):
        raise CompilationError("storage component actual requires its exact scalar field and parent binding")
    for scope in scopes.values():
        visit_scope(scope)
    if readonly_record is None and not array_element:
        for index in indices:
            visit_index(index)
    # Callback traversal cannot lend stale declaration/namespace authority.
    current = _describe(analysis, routine, node, tick, lexical_alias, readonly_record, scope_scan)
    if (current[0] is not root or (current[1].dtype, current[1].kind, current[1].rank) !=
            (typed.dtype, typed.kind, typed.rank) or current[2] != field_records or len(current[3]) != len(identities)
            or any(any(left is not right for left, right in zip(a[:3], b[:3], strict=True))
                   or len(a[3]) != len(b[3]) or any(x is not y for x, y in zip(a[3], b[3], strict=True))
                   or a[4] is not b[4] for a, b in zip(identities, current[3], strict=True))
            or len(current[4]) != len(indices) or any(a is not b for a, b in zip(indices, current[4], strict=True))):
        raise CompilationError("storage component original declaration authority changed during traversal")
    scopes.update(current[5])
    for scope in current[5].values():
        visit_scope(scope)
    _validate_scope_roles(analysis, roles, scopes)
    if _original(analysis, routine.qualified) is not routine:
        raise CompilationError("storage component original routine changed during traversal")
    final_position = occurrence()
    if (len(position) != len(final_position) or any(parent is not current or branch != current_branch
            for (parent, branch), (current, current_branch) in zip(position, final_position, strict=True))):
        raise CompilationError("storage component original read parent/branch authority changed during traversal")
    if lexical_alias is not None:
        lexical_alias.validate()
        if (not _attached(lexical_alias, alias_fields) or not _attached(declaration, declaration_fields)
                or declaration.scope.bindings is not alias_map or not _same_syntax(header_syntax)
                or len(alias_map) != len(aliases)
                or any(alias_map.get(name) is not binding or not _attached(binding, attached)
                       or getattr(binding, "component_object", None) is not root
                       or getattr(binding, "associate_selector", None) is not selector
                       for name, binding, attached, root, selector in aliases)):
            raise CompilationError("storage component original lexical association changed during traversal")
    if original_root is not None:
        if not array_element:
            _TypeBuilder(analysis).binding(original_root)
        if (root is not original_root or not _attached(original_root, original_fields)
                or original_root.declaring_scope is not declaring_scope):
            raise CompilationError("storage component original object changed during read traversal")
    if array_element:
        record_array_declaration(analysis, original_root, retain_scope=visit_scope, tick=tick)
    resource = root.root + "%" + "%".join(field["member"] for field in field_records)
    record = {"version": 3 if array_element else 2 if readonly_record is not None else 1,
              "kind": "original_numeric_component_read", "resource": resource,
              "object_resource": root.root, "source_expression": str(node), "fields": field_records,
              "terminal": {"type": typed.dtype, "kind": typed.kind, "rank": typed.rank},
              "allocation_or_definition_inferred": False, "descriptor_mutation_authorized": False,
              "payload_access_authorized": False, "layout_authorized": False, "capture_authorized": False,
              "execution_authorized": False, "motion_authorized": False, "alias_authority": False,
              "evaluation": "unchanged original reached numeric expression; no generated reevaluation"}
    if call_association is not None:
        resolved, association = call_association
        record.update(version=4, kind="original_numeric_component_actual", formal=association.formal,
            callee=resolved.procedure, original_call=str(resolved._original),
            evaluation="unchanged original scalar actual association; no generated address or value",
            callee_requirement="complete original storage and escape projection; no PURE or INTENT noescape inference")
    return NumericComponentRead(root, resource, indices, json.dumps(record, sort_keys=True, separators=(",", ":")))
