"""Original joined-team storage structure, without execution or payload facts.

This token checks only the original team, its join and data-sharing clauses.
The descriptor-window consumer must still rescan every body and direct callee;
the token cannot authorize native execution, initialization or a GPU task.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, fields
from hashlib import sha256

from fparser.two.utils import Base, walk

from compiler.frontend.native_group_structure import (
    NATIVE_GROUP_SCAN_LIMIT,
    NativeGroupCandidate,
    directive,
    grouped_original_sequence,
    original_sequence,
)
from compiler.ir import CompilationError

_ISSUER = object()
DESCRIPTOR_TEAM_VERSION = 2
DESCRIPTOR_TEAM_FIXED_ARRAY_REDUCTION_VERSION = 7
DESCRIPTOR_TEAM_FORMAL_ARRAY_REDUCTION_VERSION = 8
DESCRIPTOR_TEAM_MASTER_VERSION = 3
DESCRIPTOR_WINDOW_MASTER_VERSION = 19
DESCRIPTOR_TEAM_IMPLICIT_WORKSHARING_VERSION = 4
DESCRIPTOR_WINDOW_IMPLICIT_WORKSHARING_VERSION = 22
_ABSENT = object()
_UNSAFE = {"pointer", "target", "allocatable", "optional", "volatile", "asynchronous", "value", "save"}


class _FixedArrayAssociationRequired(CompilationError):
    """An incomplete storage scan, never a published team or permission."""

    def __init__(self, binding):
        super().__init__("native fixed-array reduction formal requires its exact original CALL association")
        self.binding = binding


class _ScalarReductionAssociationRequired(CompilationError):
    """A scalar formal needs the whole original native lifetime graph."""

    def __init__(self, binding):
        super().__init__("native scalar reduction formal requires whole-CALL lifetime association")
        self.binding = binding


def _require_fixed_array_associations(analysis, procedure, associations):
    from compiler.frontend.native_fixed_array_association import NativeFixedArrayAssociation

    if type(associations) is not tuple or len({id(item) for item in associations}) != len(associations):
        raise CompilationError("native array team requires distinct exact association receipts")
    attached = []
    for item in associations:
        if (type(item) is not NativeFixedArrayAssociation or item._resolved.procedure != procedure):
            raise CompilationError("native array team association belongs to another original callee")
        attached.extend((item, part.name, object.__getattribute__(item, part.name))
                        for part in fields(NativeFixedArrayAssociation))
    for item in associations:
        item.validate(analysis, item._caller.qualified, item._selection, item._formal_name)
    if any(object.__getattribute__(item, name) is not value for item, name, value in attached):
        raise CompilationError("native array team association changed during validation")


def _kind(node):
    return type(node).__name__


def _children(node):
    return getattr(node, "content", getattr(node, "items", ())) or ()


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _original_group(routine, originals, *, graph_selection=False):
    """Match complete original objects, including attached parser comments."""
    visits = 0

    def semantic(nodes):
        return tuple(node for node in nodes if _kind(node) != "Comment" or str(node).lstrip().startswith("!$"))

    requested = semantic(originals) if graph_selection else originals

    def find(items, peeled):
        nonlocal visits
        for item in grouped_original_sequence(items, peeled):
            visits += 1
            if visits > 65536:
                raise CompilationError("descriptor team original structure budget exhausted")
            if isinstance(item, NativeGroupCandidate):
                retained = semantic(item.original_nodes) if graph_selection else item.original_nodes
                if (len(retained) == len(requested)
                        and all(left is right for left, right in zip(retained, requested, strict=True))):
                    return item
            elif _kind(item) in {"Execution_Part", "If_Construct", "Block_Nonlabel_Do_Construct",
                                 "Associate_Construct"}:
                found = find(peeled.get(id(item), _children(item)), peeled)
                if found is not None:
                    return found
        return None

    found = find(_children(routine.execution), {})
    if found is None:
        raise CompilationError("descriptor team requires the exact complete original joined region")
    return found



@dataclass(frozen=True, slots=True)
class _ImplicitWorksharingEnd:
    directive: object
    loop: object
    header: object
    ending: object
    collapse: int

    def public(self):
        opening, header, ending = (getattr(node, "item", None)
                                   for node in (self.directive, self.header, self.ending))
        if any(item is None or not getattr(item, "span", None) for item in (opening, header, ending)):
            raise CompilationError("implicit worksharing requires exact original loop source positions")
        return {"original_directive": str(self.directive),
                "directive_source_span": list(opening.span),
                "associated_loop_source_span": [header.span[0], ending.span[1]],
                "associated_loop_sha256": sha256(str(self.loop).encode()).hexdigest(),
                "original_end_do": str(self.ending), "collapse": self.collapse,
                "implicit_default_barrier_retained": True,
                "original_native_source_unchanged": True}


def _worksharing_sequence(items, peeled):
    # A sentinel after a loop may be attached before the next BLOCK (or
    # another construct's header). Expose only its leading original comments;
    # comments following that header belong to its inner lexical sequence.
    result = []
    for node in original_sequence(items, peeled):
        if hasattr(node, "content") and id(node) not in peeled:
            children = list(_children(node))
            while children and _kind(children[0]) == "Comment":
                result.append(children.pop(0))
            peeled[id(node)] = tuple(children)
        result.append(node)
    return tuple(result)


def _counted_loop(loop, peeled):
    if id(loop) not in peeled:
        _worksharing_sequence((loop,), peeled)
    body = peeled.get(id(loop), _children(loop))
    if (not body or _kind(body[0]) != "Nonlabel_Do_Stmt" or _kind(body[-1]) != "End_Do_Stmt"
            or str(body[0].items[0]).lower() != "do" or body[0].items[1] is None):
        raise CompilationError("implicit worksharing requires its complete original ordinary counted DO")
    control = body[0].items[1]
    parts = _children(control)
    pair = parts[1] if len(parts) > 1 else None
    if (not parts or parts[0] is not None or any(item is not None for item in parts[2:])
            or not isinstance(pair, (tuple, list)) or len(pair) != 2 or _kind(pair[0]) != "Name"
            or not isinstance(pair[1], (tuple, list)) or len(pair[1]) not in {2, 3}):
        raise CompilationError("implicit worksharing requires its complete original ordinary counted DO")
    return body


def _implicit_worksharing_ends(group):
    """Associate a missing sentinel with one original lexical DO completion."""
    pending, peeled, result, seen, visits = [group.original_nodes], {}, {}, set(), 0
    while pending:
        sequence = _worksharing_sequence(pending.pop(), peeled)
        for index, original in enumerate(sequence):
            visits += 1
            if visits > NATIVE_GROUP_SCAN_LIMIT:
                raise CompilationError("implicit worksharing original-source scan budget exhausted")
            text = directive(original) or ""
            if re.fullmatch(r"do(?:\s+.*)?", text):
                loop_index = index + 1
                while (loop_index < len(sequence) and _kind(sequence[loop_index]) == "Comment"
                       and directive(sequence[loop_index]) is None):
                    loop_index += 1
                if loop_index < len(sequence) and _kind(sequence[loop_index]) == "Block_Nonlabel_Do_Construct":
                    loop = sequence[loop_index]
                    following = loop_index + 1
                    while (following < len(sequence) and _kind(sequence[following]) == "Comment"
                           and directive(sequence[following]) is None):
                        following += 1
                    ending = directive(sequence[following]) if following < len(sequence) else None
                    if ending not in {"end do", "end do nowait"}:
                        if ending is not None and ending.startswith("end do"):
                            raise CompilationError("implicit worksharing has an unsupported original END DO directive")
                        body = _counted_loop(loop, peeled)
                        collapse = re.findall(r"\bcollapse\s*\(\s*([1-4])\s*\)", text)
                        if len(collapse) > 1:
                            raise CompilationError("implicit worksharing requires one literal COLLAPSE")
                        depth = int(collapse[0]) if collapse else 1
                        nested = body
                        for _ in range(depth - 1):
                            children = [node for node in nested[1:-1] if _kind(node) != "Comment"]
                            if len(children) != 1 or _kind(children[0]) != "Block_Nonlabel_Do_Construct":
                                raise CompilationError("implicit worksharing COLLAPSE requires its original counted loop nest")
                            nested = _counted_loop(children[0], peeled)
                        if id(body[-1]) in result:
                            raise CompilationError("implicit worksharing has repeated original loop association")
                        result[id(body[-1])] = _ImplicitWorksharingEnd(original, loop, body[0], body[-1], depth)
            if hasattr(original, "content") and id(original) not in seen:
                seen.add(id(original))
                pending.append(peeled.get(id(original), _children(original)))
    return result


def _group_syntax_checkpoint(originals):
    from compiler.frontend.descriptor_associations import _syntax_attachments

    topology = tuple(row for root in originals for row in _syntax_attachments(root))
    pending, seen, nodes, containers = list(originals), set(), [], []
    item_fields = ("span", "fort_original_span", "label", "name", "line", "comment", "is_f2py_directive")
    while pending:
        node = pending.pop()
        if id(node) in seen or not isinstance(node, (Base, tuple, list)):
            continue
        seen.add(id(node))
        if len(seen) > NATIVE_GROUP_SCAN_LIMIT:
            raise CompilationError("implicit worksharing original syntax receipt budget exhausted")
        if isinstance(node, (tuple, list)):
            children = tuple(node)
            containers.append((node, children))
        else:
            item = getattr(node, "item", None)
            nodes.append((node, getattr(node, "string", _ABSENT), item,
                          getattr(node, "content", _ABSENT), getattr(node, "items", _ABSENT),
                          tuple((name, getattr(item, name, _ABSENT)) for name in item_fields)))
            children = _children(node)
        pending.extend(children)
    return topology, tuple(nodes), tuple(containers)


def _same_group_syntax(saved):
    from compiler.frontend.descriptor_associations import _same_syntax

    topology, nodes, containers = saved
    return (_same_syntax(topology)
            and all(getattr(node, "string", _ABSENT) is string and getattr(node, "item", None) is item
                    and getattr(node, "content", _ABSENT) is content and getattr(node, "items", _ABSENT) is items
                    and all(getattr(item, name, _ABSENT) is value for name, value in fields)
                    for node, string, item, content, items, fields in nodes)
            and all(len(node) == len(children)
                    and all(left is right for left, right in zip(node, children, strict=True))
                    for node, children in containers))


def _implicit_namespace_checkpoint(analysis, routine, syntax):
    """Retain consulted namespace objects before any delegated serializer.

    This is a stack-local attachment fence, not another source derivation.
    Its syntax receipt remains limited to the selected complete group.
    """
    from compiler.frontend import source_session
    from compiler.frontend.descriptor_intrinsics import intrinsic_source_attachments

    source = intrinsic_source_attachments(analysis)
    policy = source_session._ROOTS
    routine_fields = tuple((part.name, getattr(routine, part.name)) for part in fields(routine))
    analysis_fields = tuple((name, getattr(analysis, name, _ABSENT)) for name in
                            ("depth_limit", "procedure_limit", "operation_limit", "_descriptor_source"))
    originals = {id(node): node for node, *_fields in syntax[1]}
    pending = [routine.scope]
    for name in ("_block_scopes", "_associate_scopes"):
        for value in getattr(analysis, name, {}).values():
            scope = getattr(value, "scope", value)
            node = getattr(scope, "node", None)
            if node is not None and originals.get(id(node)) is node:
                pending.append(scope)
    scopes, containers, kinds, declarations, seen_containers, seen_kinds = {}, [], [], {}, set(), set()
    while pending:
        scope = pending.pop()
        if scope is None or id(scope) in scopes:
            continue
        if len(scopes) >= NATIVE_GROUP_SCAN_LIMIT:
            raise CompilationError("implicit worksharing namespace receipt budget exhausted")
        attached = tuple((part.name, getattr(scope, part.name)) for part in fields(scope))
        bindings = tuple((binding, tuple((part.name, getattr(binding, part.name)) for part in fields(binding)),
                          getattr(binding, "component_object", _ABSENT),
                          getattr(binding, "associate_selector", _ABSENT)) for binding in scope.bindings.values())
        for binding, _fields, _root, _selector in bindings:
            for node in (*binding.shape_nodes, *binding.lower_bound_nodes):
                if isinstance(node, Base):
                    declarations[id(node)] = node
        scopes[id(scope)] = (scope, attached, bindings)
        values = [value for _name, value in attached]
        values.extend(value for _binding, saved, _root, _selector in bindings for _name, value in saved)
        current = scope.kinds
        while current is not None and id(current) not in seen_kinds:
            seen_kinds.add(id(current))
            kinds.append((current, current.parent, current.values, current.intrinsic_kinds))
            values.extend((current.values, current.intrinsic_kinds))
            current = current.parent
        while values:
            value = values.pop()
            if id(value) in seen_containers or not isinstance(value, (dict, tuple, list, set, frozenset)):
                continue
            seen_containers.add(id(value))
            if len(seen_containers) > NATIVE_GROUP_SCAN_LIMIT:
                raise CompilationError("implicit worksharing namespace container budget exhausted")
            if isinstance(value, dict):
                items = tuple(value.items())
                containers.append((value, "mapping", items))
                values.extend(item for _name, item in items)
            elif isinstance(value, (set, frozenset)):
                containers.append((value, "set", frozenset(value)))
            else:
                items = tuple(value)
                containers.append((value, "sequence", items))
                values.extend(items)
        pending.append(scope.parent)
        pending.extend(analysis.modules[name] for name in
                       {*scope.wildcards, *(module for module, _name in scope.imports.values())}
                       if name in analysis.modules)
    declaration_syntax = _group_syntax_checkpoint(tuple(declarations.values()))
    return (source, policy, routine, routine_fields, analysis_fields, tuple(scopes.values()),
            tuple(containers), tuple(kinds), declaration_syntax)


def _same_implicit_namespace(analysis, saved):
    from compiler.frontend import source_session
    from compiler.frontend.descriptor_intrinsics import same_intrinsic_source_attachments

    source, policy, routine, routine_fields, analysis_fields, scopes, containers, kinds, declarations = saved
    if (source_session._ROOTS is not policy or not same_intrinsic_source_attachments(analysis, source)
            or any(getattr(routine, name) is not value for name, value in routine_fields)
            or any(getattr(analysis, name, _ABSENT) is not value for name, value in analysis_fields)
            or any(any(getattr(scope, name) is not value for name, value in attached)
                   or any(any(getattr(binding, name) is not value for name, value in binding_fields)
                          or getattr(binding, "component_object", _ABSENT) is not root
                          or getattr(binding, "associate_selector", _ABSENT) is not selector
                          for binding, binding_fields, root, selector in bindings)
                   for scope, attached, bindings in scopes)
            or not _same_group_syntax(declarations)
            or any(current.parent is not parent or current.values is not values
                   or current.intrinsic_kinds is not intrinsic for current, parent, values, intrinsic in kinds)):
        return False
    return all(len(value) == len(items) and (
        all(name in value and value[name] is item for name, item in items) if kind == "mapping"
        else value == items if kind == "set"
        else all(left is right for left, right in zip(value, items, strict=True)))
        for value, kind, items in containers)


def _derive(analysis, procedure, originals, record_region=None, *, native_graph=None, encoded_record=False,
            fixed_array_associations=()):
    _require_fixed_array_associations(analysis, procedure, fixed_array_associations)
    routine = analysis._require_original(procedure)
    group = _original_group(routine, originals)
    allowed_associations = ()
    record_checkpoint = None
    lifetime = None
    private_pointers = ()
    if native_graph is not None:
        from compiler.frontend.native_lifetime_context import NativeLifetimeContext
        if record_region is not None:
            raise CompilationError("native lifetime and owned numeric record teams are distinct authorities")
        lifetime = NativeLifetimeContext(analysis, native_graph)
        pairs = lifetime.team_bindings(procedure, group.original_nodes)
        allowed_associations = tuple(binding for pair in pairs for binding in pair)
        private_pointers = tuple(pointer for pointer, _target in pairs)
    if record_region is not None:
        from compiler.frontend.storage_record_regions import OwnedRecordRegion, RecordRegionCheckpoint
        if type(record_region) is not OwnedRecordRegion:
            raise CompilationError("descriptor record team requires its exact source region")
        record_checkpoint = RecordRegionCheckpoint(analysis, record_region)
        record_region.validate(analysis)
        if (record_region._roles[1] is not routine or len(record_region._roles[6]) != len(group.original_nodes)
                or any(a is not b for a, b in zip(record_region._roles[6], group.original_nodes, strict=True))):
            raise CompilationError("descriptor record team lost exact original region membership")
        allowed_associations = record_region._roles[3:5]
    implicit_ends = _implicit_worksharing_ends(group)
    syntax = _group_syntax_checkpoint(group.original_nodes) if implicit_ends else None
    namespace = _implicit_namespace_checkpoint(analysis, routine, syntax) if syntax is not None else None
    implicit_rows = {key: ending.public() for key, ending in implicit_ends.items()}
    graph, selected = analysis._selected_source(procedure, group.original_nodes)
    if syntax is not None and (not _same_group_syntax(syntax) or not _same_implicit_namespace(analysis, namespace)):
        raise CompilationError("implicit worksharing original loop association changed during source selection")
    specification = next((part for part in _children(routine.scope.node)
                          if _kind(part) == "Specification_Part"), None)
    if any(_kind(node) in {"Common_Stmt", "Equivalence_Stmt", "Namelist_Stmt"}
           or (directive(node) or "").startswith("threadprivate") for node in walk(specification)):
        raise CompilationError("descriptor team requires independent storage without association")
    if any(binding.attributes & {"pointer", "target", "volatile", "asynchronous"}
           for binding in routine.scope.bindings.values() if not any(binding is item for item in allowed_associations)):
        raise CompilationError("descriptor team requires independent storage without pointer or TARGET association")
    from compiler.frontend.optional_scalar_control import _same_scope_state, _scope_state
    from compiler.frontend.source_functions import _validate_scope

    state = _scope_state(routine.scope)
    binding_fields = tuple((binding, tuple((part.name, getattr(binding, part.name)) for part in fields(binding)))
                           for binding in routine.scope.bindings.values())
    texts, private, shared, stack, nowait = [], {}, {}, [], 0
    reductions, singles, masters = [], 0, 0
    scalar_formals = {}
    array_reductions = []
    used_associations = set()
    array_checkpoint = None
    worksharing, consumed_implicit = [], []
    reduction_scopes = {}

    def intrinsic_reduction_namespace(start):
        # A user combiner can call arbitrary source. Do not classify one as a
        # primitive operation merely because its identifier spells '+' or MAX.
        # Keep every original host/USE namespace relevant to that lookup.
        pending = [start]
        while pending:
            scope = pending.pop()
            if id(scope) in reduction_scopes:
                continue
            _validate_scope(analysis, scope)
            reduction_scopes[id(scope)] = _scope_state(scope)
            parts = []
            for part in _children(scope.node):
                if _kind(part) == "Specification_Part":
                    parts.append(part)
                elif _kind(part) in {"Execution_Part", "Module_Subprogram_Part", "Internal_Subprogram_Part"}:
                    # Parser comments just before the first executable/CONTAINS
                    # statement may belong to that part rather than the spec.
                    for prefix in _children(part):
                        if _kind(prefix) != "Comment":
                            break
                        parts.append(prefix)
            if any((directive(node) or "").startswith("declare reduction") for node in walk(parts)):
                raise CompilationError("descriptor team user reduction declaration requires separate authority")
            if scope.parent is not None:
                pending.append(scope.parent)
            for module in {*scope.wildcards, *(module for module, _name in scope.imports.values())}:
                target = analysis.modules.get(module)
                if target is None:
                    raise CompilationError("descriptor team intrinsic reduction namespace is unresolved")
                pending.append(target)

    def clauses(text, owner, original):
        nonlocal array_checkpoint
        remainder = text.strip()
        clause_scope = analysis.source_scope_for(original, routine.scope)
        reduced, explicit_private = set(), set()
        array_reduced, explicit_shared = set(), set()
        while remainder:
            match = re.match(r"(private|shared|default|collapse|schedule|reduction)\s*\(([^()]*)\)\s*", remainder)
            if match is None:
                raise CompilationError("descriptor team storage clause is unsupported: " + remainder)
            kind, values = match.group(1), match.group(2).strip()
            if kind == "default":
                if values not in {"shared", "none"}:
                    raise CompilationError("descriptor team DEFAULT changes storage association")
            elif kind == "collapse":
                if values not in {"1", "2", "3", "4"}:
                    raise CompilationError("descriptor team requires a literal bounded COLLAPSE")
            elif kind == "schedule":
                if not re.fullmatch(r"(?:static|runtime|dynamic|guided)(?:\s*,\s*[1-9][0-9]*)?", values):
                    raise CompilationError("descriptor team schedule metadata is unproved")
            elif kind == "reduction":
                intrinsic_reduction_namespace(clause_scope)
                if owner not in {"parallel", "parallel do", "do"} or values.count(":") != 1:
                    raise CompilationError("descriptor team requires an intrinsic scalar reduction clause")
                operator, names = (part.strip() for part in values.split(":"))
                numeric = {"+", "-", "*", "max", "min"}
                integer = {"iand", "ior", "ieor"}
                logical = {".and.", ".or.", ".eqv.", ".neqv."}
                if operator not in numeric | integer | logical:
                    raise CompilationError("descriptor team reduction identifier is not an intrinsic primitive operation")
                declarations, array_declarations = [], []
                for name in names.split(","):
                    name = name.strip()
                    binding = routine.scope.bindings.get(name)
                    allowed = {"real", "integer"} if operator in numeric else {"integer"} if operator in integer else {"logical"}
                    if (not re.fullmatch(r"[a-z][a-z0-9_]*", name) or binding is None
                            or analysis._binding(clause_scope, name) is not binding
                            or binding.declaring_scope is not routine.scope
                            or binding.dtype not in allowed or binding.kind not in {4, 8}
                            or binding.attributes & (_UNSAFE | {"parameter"}) or name in reduced):
                        raise CompilationError("descriptor team REDUCTION requires independent original primitive scalar locals")
                    reduced.add(name)
                    if binding.rank:
                        array_reduced.add(name)
                        from compiler.frontend.call_bindings import _fixed_array_bounds
                        from compiler.frontend.source_functions import original_name_absent
                        if binding.rank != 1:
                            raise CompilationError("descriptor reduction requires original fixed rank-one array storage")
                        if name in routine.arguments and binding.intent not in {"out", "inout"}:
                            raise CompilationError("native array reduction requires an original OUT or INOUT formal")
                        associated = tuple(item for item in fixed_array_associations if item.formal is binding)
                        if name in routine.arguments and not associated:
                            raise _FixedArrayAssociationRequired(binding)
                        if name not in routine.arguments and associated:
                            raise CompilationError("native fixed-array association cannot replace local storage")
                        if array_checkpoint is None:
                            array_syntax = _group_syntax_checkpoint(group.original_nodes)
                            array_checkpoint = array_syntax, _implicit_namespace_checkpoint(analysis, routine, array_syntax)
                        spelling = operator if operator[0].isalpha() else "operator(" + operator + ")"
                        if not original_name_absent(analysis, clause_scope, spelling):
                            raise CompilationError("fixed array reduction intrinsic operation is shadowed")
                        (lower, upper), = _fixed_array_bounds(analysis, binding, element_limit=None)
                        extent = max(0, upper - lower + 1)
                        byte_count = extent * binding.kind
                        if not extent or byte_count > (1 << 63) - 1:
                            raise CompilationError("fixed array reduction requires nonempty ABI-safe byte storage")
                        declaration = {"binding": binding.public(), "lower_bound": lower,
                            "upper_bound": upper, "element_count": extent, "byte_count": byte_count,
                            "storage": "original local fixed rank-one primitive array"}
                        if associated:
                            if any(item.formal_bounds != (lower, upper) or item.element_count != extent for item in associated):
                                raise CompilationError("native reduction formal association changed its original bounds")
                            declaration.update(storage="original fixed rank-one formal with exact native CALL associations",
                                original_call_associations=[item.public() for item in associated])
                            used_associations.update(id(item) for item in associated)
                        array_declarations.append(declaration)
                    else:
                        if name in routine.arguments:
                            if lifetime is None:
                                raise _ScalarReductionAssociationRequired(binding)
                            scalar_formals[name] = list(lifetime.scalar_reduction_associations(
                                procedure, group.original_nodes, binding))
                        declarations.append(binding.public())
                if declarations:
                    reductions.append({"original_directive": str(original), "operator": operator,
                                       "bindings": declarations, "descriptor_evidence_only": True})
                if array_declarations:
                    array_reductions.append({"version": 1, "original_directive": str(original),
                        "operator": operator, "bindings": array_declarations,
                        "descriptor_evidence_only": True, "native_only": True,
                        "payload_access_authorized": False, "entry_definition_authority": False,
                        "gpu_legality_established": False, "arithmetic_reassociation_authorized": False})
            else:
                for name in values.split(","):
                    name = name.strip()
                    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
                        raise CompilationError("descriptor team clauses require original binding names")
                    binding = analysis._binding(clause_scope, name)
                    if binding is None:
                        raise CompilationError("descriptor team clause binding is unresolved")
                    if kind == "private":
                        if ((record_region is not None and binding is record_region._roles[4])
                                or any(binding is pointer for pointer in private_pointers)):
                            private[binding.root] = binding.public()
                            explicit_private.add(name)
                            continue
                        if (binding.declaring_scope is not routine.scope or name in routine.arguments
                                or binding.dtype not in {"real", "integer", "logical"}
                                or binding.attributes & (_UNSAFE | {"parameter"})
                                or any(_kind(axis) != "Explicit_Shape_Spec" for axis in binding.shape_nodes)):
                            raise CompilationError("descriptor team PRIVATE requires independent original numeric locals")
                        private[binding.root] = binding.public()
                        explicit_private.add(name)
                    else:
                        if binding.attributes & {"pointer", "target", "volatile", "asynchronous"}:
                            raise CompilationError("descriptor team SHARED association is unproved")
                        shared[binding.root] = binding.public()
                        explicit_shared.add(name)
            remainder = remainder[match.end():].lstrip(", ")
        if array_reduced & explicit_shared:
            raise CompilationError("fixed array reduction cannot also be explicitly SHARED in the same clause list")
        if reduced & explicit_private:
            raise CompilationError("descriptor team reduction storage cannot also be explicitly PRIVATE")

    for root in group.original_nodes:
        for node in walk(root):
            completed = implicit_ends.get(id(node))
            if completed is not None:
                if (completed.ending is not node or not stack or stack[-1] != "do"
                        or not worksharing or worksharing[-1] is not completed.directive):
                    raise CompilationError("implicit worksharing lost its exact original counted loop completion")
                stack.pop()
                worksharing.pop()
                consumed_implicit.append(implicit_rows[id(node)])
            text = directive(node)
            if text is None:
                continue
            if len(texts) >= analysis.operation_limit:
                raise CompilationError("descriptor team directive budget exhausted")
            texts.append(text)
            opening = re.fullmatch(r"(parallel do|parallel|do)(?:\s+(.*))?", text)
            if opening:
                kind, suffix = opening.groups()
                if kind.startswith("parallel"):
                    if stack or len(texts) != 1:
                        raise CompilationError("descriptor team nested or repeated parallel regions need separate authority")
                elif stack != ["parallel"]:
                    raise CompilationError("descriptor team DO requires its original enclosing team")
                clauses(suffix or "", kind, node)
                stack.append(kind)
                if kind == "do":
                    worksharing.append(node)
            elif text == "single":
                if stack != ["parallel"]:
                    raise CompilationError("descriptor team SINGLE requires its original enclosing team")
                stack.append("single")
                singles += 1
            elif text == "master":
                if stack != ["parallel"]:
                    raise CompilationError("descriptor team MASTER requires its original enclosing team")
                stack.append("master")
                masters += 1
            elif text in {"end parallel", "end parallel do", "end do", "end do nowait", "end single", "end single nowait",
                          "end master"}:
                expected = ("do" if text.startswith("end do") else "single" if text.startswith("end single")
                            else text[4:])
                if not stack or stack[-1] != expected:
                    raise CompilationError("descriptor team changed its original synchronization nesting")
                if text.endswith("nowait"):
                    # The complete enclosing END PARALLEL remains mandatory;
                    # no access or lifetime cut inside this region is allowed.
                    nowait += 1
                stack.pop()
                if expected == "do":
                    if not worksharing:
                        raise CompilationError("descriptor team changed its original worksharing association")
                    worksharing.pop()
            elif text == "barrier":
                if stack != ["parallel"]:
                    raise CompilationError("descriptor team BARRIER is outside its original team")
            else:
                raise CompilationError("descriptor team asynchronous or unsupported directive: " + text)
    # END PARALLEL DO is optional for the exact original associated DO. Its
    # implicit team join is retained; a plain PARALLEL always needs its end.
    implicit = stack == ["parallel do"] and len(texts) == 1
    if (stack and not implicit) or not texts:
        raise CompilationError("descriptor team requires its complete original team join")
    record = {"version": 1, "procedure": procedure, "source_identity": analysis.inputs.identity(),
            "structured_identity": graph.identity, "selected_node_ids": list(selected),
            "original_directives": texts, "private_bindings": private, "shared_bindings": shared,
            "retained_internal_nowait": nowait, "implicit_parallel_do_join": implicit,
            "requires_complete_descriptor_body_and_call_rescan": True,
            "retains_original_team_and_directives": True, "internal_cuts_authorized": False,
            "execution_authorized": False, "entry_definition_authority": False,
            "payload_access_authorized": False, "gpu_legality_established": False}
    if reductions or singles:
        record.update(version=DESCRIPTOR_TEAM_VERSION, intrinsic_scalar_reductions=reductions,
                      retained_single_regions=singles, parent_parallel_join_required=True)
    if masters:
        # MASTER itself supplies no synchronization or completed-work fact.
        # Its unchanged body and any original explicit barriers remain inside
        # the complete enclosing team and its required END PARALLEL join.
        record.update(version=DESCRIPTOR_TEAM_MASTER_VERSION, intrinsic_scalar_reductions=reductions,
                      retained_single_regions=singles, parent_parallel_join_required=True,
                      retained_master_regions=masters, master_barrier_assumed=False,
                      master_completion_authorized=False)
    if implicit_ends:
        if len(consumed_implicit) != len(implicit_ends) or worksharing:
            raise CompilationError("implicit worksharing requires every original loop completion")
        record.update(version=DESCRIPTOR_TEAM_IMPLICIT_WORKSHARING_VERSION,
                      implicit_worksharing_endings=consumed_implicit,
                      implicit_worksharing_policy="close at exact original associated counted-loop end; default barrier retained")
    if record_region is not None:
        record.update(version=5, owned_record_region=record_region.public())
        record_region.validate(analysis)
    if lifetime is not None:
        if not private_pointers and not scalar_formals:
            raise CompilationError("native lifetime team requires its original pointer or scalar reduction role")
        if scalar_formals:
            record["native_lifetime_scalar_reductions"] = scalar_formals
        if any(pointer.root not in private for pointer in private_pointers):
            raise CompilationError("native lifetime pointer requires exact original PRIVATE participation")
        record.update(version=6, native_lifetime_context=lifetime.public())
        lifetime.checkpoint()
    if array_reductions:
        roots = {item["binding"]["resource"] for row in array_reductions for item in row["bindings"]}
        if roots & set(private):
            raise CompilationError("fixed array reduction cannot also use original PRIVATE storage")
        record.update(version=DESCRIPTOR_TEAM_FIXED_ARRAY_REDUCTION_VERSION,
                      intrinsic_fixed_array_reductions=array_reductions,
                      fixed_array_reduction_contract="whole original native team and every original join",
                      parent_parallel_join_required=True)
    if fixed_array_associations:
        if used_associations != {id(item) for item in fixed_array_associations}:
            raise CompilationError("native array team has an unused original CALL association")
        record.update(version=DESCRIPTOR_TEAM_FORMAL_ARRAY_REDUCTION_VERSION,
                      fixed_array_formal_associations=[item.public() for item in fixed_array_associations],
                      formal_association_authority="original whole CALL element mapping only")
    encoded = _encoded(record) if encoded_record else None
    analysis.verify_publication()
    if analysis._require_original(procedure) is not routine:
        raise CompilationError("descriptor team original routine attachment changed at publication")
    _validate_scope(analysis, routine.scope)
    if (not _same_scope_state(state) or any(not _same_scope_state(saved) for saved in reduction_scopes.values())
            or any(any(getattr(binding, name) is not value for name, value in saved)
                                          for binding, saved in binding_fields)):
        raise CompilationError("descriptor team consumed original scalar declaration changed at publication")
    if syntax is not None:
        # Scope resolution/serialization above may delegate to source hooks.
        # Finish with the actual live-file fence and callback-free retained
        # source/scalar checks, rather than publish an association captured
        # before a final delegated callback.
        analysis.verify_source()
        analysis.inputs.verify_live()
        if (not _same_group_syntax(syntax)
                or not _same_implicit_namespace(analysis, namespace)
                or any(any(getattr(binding, name) is not value for name, value in saved)
                       for binding, saved in binding_fields)):
            raise CompilationError("implicit worksharing original loop association changed at publication")
    if array_checkpoint is not None:
        analysis.verify_source()
        analysis.inputs.verify_live()
        if (not _same_group_syntax(array_checkpoint[0])
                or not _same_implicit_namespace(analysis, array_checkpoint[1])):
            raise CompilationError("fixed array reduction original storage or source changed at publication")
    if record_checkpoint is not None:
        analysis.inputs.verify_live()
        record_checkpoint.require()
    return (record, encoded) if encoded_record else record


@dataclass(frozen=True, slots=True)
class DescriptorTeamProof:
    identity: str
    procedure: str
    structured_identity: str
    selected_node_ids: tuple
    _analysis: object = field(repr=False, compare=False)
    _originals: tuple = field(repr=False, compare=False)
    _record: str = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)
    _record_region: object = field(default=None, repr=False, compare=False)
    _source_fact: object = field(default=None, repr=False, compare=False)
    _native_graph: object = field(default=None, repr=False, compare=False)
    _fixed_array_associations: tuple = field(default=(), repr=False, compare=False)

    def validate(self, analysis, procedure, selected):
        attached = _team_attachments(self)
        registry = getattr(analysis, "_descriptor_team_proofs", None)
        authority = (getattr(analysis, "_source_session", None),
                     getattr(analysis, "_source_authority_context", None),
                     getattr(analysis, "_source_derivation_arena", None), registry, self.identity)
        _require_fixed_array_associations(analysis, procedure, self._fixed_array_associations)
        associations = json.loads(self._record).get("fixed_array_formal_associations", [])
        if associations != [item.public() for item in self._fixed_array_associations]:
            raise CompilationError("native array team changed its original CALL association receipts")
        lifetime = json.loads(self._record).get("native_lifetime_context")
        if ((lifetime is None) != (self._native_graph is None)
                or (lifetime is not None and lifetime["native_graph_identity"] != self._native_graph.identity)
                or (self._source_fact is not None and self._source_fact.query[-1] != id(self._native_graph))):
            raise CompilationError("descriptor team changed its original native lifetime graph")
        if self._native_graph is not None:
            self._native_graph.validate(analysis, self._native_graph.procedure, self._native_graph._selection)
        scalar_rows = json.loads(self._record).get("native_lifetime_scalar_reductions", {})
        if scalar_rows:
            from compiler.frontend.native_lifetime_context import NativeLifetimeContext
            if self._native_graph is None:
                raise CompilationError("native scalar reduction lost its retirement-only graph")
            lifetime = NativeLifetimeContext(analysis, self._native_graph)
            routine = analysis._require_original(procedure)
            for name, rows in scalar_rows.items():
                binding = routine.scope.bindings.get(name)
                if binding is None or _encoded(lifetime.scalar_reduction_associations(
                        procedure, self._originals, binding)) != _encoded(rows):
                    raise CompilationError("native scalar reduction changed its original CALL storage associations")
        if self._source_fact is not None:
            if (analysis is not self._analysis or procedure != self.procedure
                    or tuple(selected) != self.selected_node_ids
                    or getattr(analysis, "_descriptor_team_proofs", {}).get(self.identity) is not self):
                raise CompilationError("descriptor team requires its exact original compiler-issued authority")
            result = self._source_fact.require(self)
            _require_fixed_array_associations(analysis, procedure, self._fixed_array_associations)
            _require_team_attachments(self, attached, analysis, authority)
            return result
        if getattr(analysis, "_source_session", None) is not None:
            raise CompilationError("mutable diagnostic proof cannot enter a sealed authority context")
        attached = tuple((part.name, getattr(self, part.name)) for part in fields(self))
        registry = getattr(analysis, "_descriptor_team_proofs", {})
        if (self._issuer is not _ISSUER or analysis is not self._analysis or procedure != self.procedure
                or tuple(selected) != self.selected_node_ids
                or registry.get(self.identity) is not self or sha256(self._record.encode()).hexdigest() != self.identity):
            raise CompilationError("descriptor team requires its exact original compiler-issued authority")
        derived = _derive(analysis, procedure, self._originals, self._record_region,
                          native_graph=self._native_graph, encoded_record=self._record_region is not None,
                          fixed_array_associations=self._fixed_array_associations)
        encoded = derived[1] if self._record_region is not None else _encoded(derived)
        if encoded != self._record:
            raise CompilationError("descriptor team original structure, storage or source changed")
        if (any(getattr(self, name) is not value for name, value in attached)
                or getattr(analysis, "_descriptor_team_proofs", None) is not registry or registry.get(self.identity) is not self):
            raise CompilationError("descriptor team issuer attachment changed at original publication")
        return self

    def public(self):
        return {"proof_identity": self.identity, **json.loads(self._record)}


def _team_attachments(proof):
    from compiler.frontend.native_fixed_array_association import NativeFixedArrayAssociation

    owners = [proof]
    if json.loads(proof._record).get("native_lifetime_scalar_reductions") and proof._native_graph is not None:
        owners.extend((proof._native_graph, *proof._native_graph.templates, *proof._native_graph.instances))
        for template in proof._native_graph.templates:
            for call in template.calls:
                owners.extend((call, call.edge, *_original_call_receipts(call.edge.resolved)))
                for association in call.edge.resolved.associations:
                    owners.append(association.formal_binding)
                    if association.binding is not None:
                        owners.append(association.binding)
    if type(proof._fixed_array_associations) is tuple:
        for item in proof._fixed_array_associations:
            if type(item) is NativeFixedArrayAssociation:
                owners.extend((item, *_original_call_receipts(item._resolved)))
    return tuple((owner, part.name, object.__getattribute__(owner, part.name))
                 for owner in owners for part in fields(type(owner)))


def _original_call_receipts(resolved):
    # These are the exact bounded receipt objects consumed by association
    # checks, not a recursive scan of source or arbitrary object state.
    return (resolved, *resolved.original_arguments, *resolved.associations,
            *(item.actual_type for item in resolved.associations if item.actual_type is not None))


def _require_team_attachments(proof, attached, analysis, authority):
    session, context, arena, registry, identity = authority
    if (any(object.__getattribute__(owner, name) is not value for owner, name, value in attached)
            or getattr(analysis, "_descriptor_team_proofs", None) is not registry
            or registry.get(identity) is not proof
            or getattr(analysis, "_source_session", None) is not session
            or getattr(analysis, "_source_authority_context", None) is not context
            or getattr(analysis, "_source_derivation_arena", None) is not arena):
        raise CompilationError("descriptor team attachments changed during validation")


def prove_descriptor_team(analysis, procedure, originals, *, record_region=None, native_graph=None,
                          fixed_array_associations=()):
    from compiler.frontend.source_facts import TEAM, issue, lookup, token_dependencies

    originals = tuple(originals)
    _require_fixed_array_associations(analysis, procedure, fixed_array_associations)
    context = analysis.source_authority_context() if getattr(analysis, "_source_session", None) is not None else None
    query = procedure, tuple(id(node) for node in originals), id(record_region), id(native_graph)
    if fixed_array_associations:
        query = (*query[:-1], tuple(id(item) for item in fixed_array_associations), query[-1])
    previous = lookup(analysis, TEAM, query)
    if previous is not None:
        return previous.validate(analysis, procedure, previous.selected_node_ids)
    if record_region is not None:
        record, encoded = _derive(analysis, procedure, originals, record_region, native_graph=native_graph,
            encoded_record=True, fixed_array_associations=fixed_array_associations)
    else:
        record = _derive(analysis, procedure, originals, native_graph=native_graph,
                         fixed_array_associations=fixed_array_associations)
        encoded = _encoded(record)
    identity = sha256(encoded.encode()).hexdigest()
    registry = getattr(analysis, "_descriptor_team_proofs", None)
    if registry is None:
        registry = analysis._descriptor_team_proofs = {}
    if identity not in registry:
        if len(registry) >= analysis.operation_limit:
            raise CompilationError("descriptor team proof registry budget exhausted")
        registry[identity] = DescriptorTeamProof(identity, procedure, record["structured_identity"],
            tuple(record["selected_node_ids"]), analysis, tuple(originals), encoded, _ISSUER, record_region,
            _native_graph=native_graph, _fixed_array_associations=fixed_array_associations)
    proof = registry[identity]
    if context is not None:
        context.require(analysis)
        proof = issue(analysis, TEAM, query, proof,
                      token_dependencies((record_region, native_graph, *fixed_array_associations)))
    return proof.validate(analysis, procedure, tuple(record["selected_node_ids"]))
