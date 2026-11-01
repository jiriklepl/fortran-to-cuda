"""Exact fixed scalar ASSOCIATE storage, independent of its body effects.

The existing original lexical scope and fixed declaration services establish
the association. Consumers must still scan every original body statement;
this receipt grants no payload access, array alias, lifetime or execution rule.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from hashlib import sha256

from compiler.frontend.component_bindings import AssociationScope, ComponentAccess, component_access
from compiler.frontend.immutable_graph import original_type
from compiler.frontend.source_effects import _children, _kind
from compiler.frontend.source_functions import _scope_roles, _validate_scope, _validate_scope_roles
from compiler.frontend.source_objects import _TypeBuilder
from compiler.ir import CompilationError

DESCRIPTOR_WINDOW_ASSOCIATION_VERSION = 15
_ISSUER = object()
_UNSAFE = {"pointer", "allocatable", "target", "optional", "value", "volatile", "asynchronous",
           "parameter", "external", "intrinsic"}


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _attachments(value):
    return tuple((item.name, getattr(value, item.name)) for item in fields(value))


def _attached(value, saved):
    return all(getattr(value, name) is item for name, item in saved)


def _syntax_attachments(root):
    """Retain each exact parser child consumed from the association header."""
    pending, saved = [root], []
    while pending:
        node = pending.pop()
        if not hasattr(node, "items") and not hasattr(node, "content") and not isinstance(node, (tuple, list)):
            continue
        children = tuple(_children(node))
        saved.append((node, children))
        pending.extend(reversed(children))
    return tuple(saved)


def _same_syntax(saved):
    return all(len(_children(node)) == len(children)
               and all(left is right for left, right in zip(_children(node), children, strict=True))
               for node, children in saved)


def _derive(analysis, routine, original, scope_validator=None, tick=None):
    if (analysis._require_original(routine.qualified) is not routine
            or _kind(original) != "Associate_Construct"):
        raise CompilationError("descriptor association requires its exact original routine and construct")
    graph, selected = analysis._selected_source(routine.qualified, (original,))
    if (len(selected) != 1 or graph.nodes[selected[0]].kind != "associate"
            or len(graph.source_nodes(selected[0])) != 1 or graph.source_nodes(selected[0])[0] is not original):
        raise CompilationError("descriptor association requires its complete original structured unit")
    association = getattr(analysis, "_associate_scopes", {}).get(id(original))
    session = getattr(analysis, "_source_session", None)
    if (original_type(association) is not AssociationScope or not association.available
            or type(association) is not AssociationScope and (session is None or not session.contains(association))):
        raise CompilationError("descriptor association requires its existing original lexical scope")
    scope, aliases = association.scope, association.scope.bindings
    children = tuple(_children(original))
    if (len(children) < 2 or _kind(children[0]) != "Associate_Stmt"
            or _kind(children[-1]) != "End_Associate_Stmt" or scope.node is not original
            or analysis.source_scope_for(children[0], routine.scope) is not scope.parent):
        raise CompilationError("descriptor association changed its exact original header or scope")
    originals = tuple(_children(children[0].items[1]))
    if not originals or len(originals) != len(association.selectors) or len(aliases) != len(originals):
        raise CompilationError("descriptor association lost its original complete selector list")
    header_syntax = _syntax_attachments(children[0])
    saved = _attachments(association)
    selectors = tuple((access, _attachments(access), _attachments(access.binding),
                       getattr(access.binding, "component_object", None)) for access in association.selectors)
    bindings = tuple((name, binding, _attachments(binding), getattr(binding, "component_object", None),
                      getattr(binding, "associate_selector", None)) for name, binding in aliases.items())
    roles, scopes = _scope_roles(analysis), {}

    def require_scope(current):
        if scope_validator is not None:
            scope_validator(current)
        _validate_scope(analysis, current)
        scopes[id(current)] = current

    require_scope(scope.parent)
    require_scope(scope)
    types = _TypeBuilder(analysis)
    records, roots = [], []
    for item, access in zip(originals, association.selectors, strict=True):
        if tick is not None:
            tick()
        alias, _arrow, selector = item.items
        spelling = str(alias).lower()
        if _kind(selector) == "Name":
            source = analysis._binding(scope.parent, selector)
            current = None if source is None else ComponentAccess(source, getattr(source, "component_object", source), selector)
        elif _kind(selector) == "Data_Ref":
            current = component_access(analysis, scope.parent, selector)
        else:
            current = None
        projected = aliases.get(spelling)
        if (current is None or current.indices or access.selector is not selector
                or current.root_binding is not access.root_binding or current.public() != access.public()
                or current.binding.rank or current.binding.attributes & _UNSAFE
                or current.root_binding.rank or current.root_binding.attributes & _UNSAFE
                or projected is None or getattr(projected, "associate_selector", None) is not selector
                or projected.public() != {**current.binding.public(), "name": spelling}
                or getattr(projected, "component_object", access.binding) is not current.root_binding
                or len(projected.shape_nodes) != len(current.binding.shape_nodes)
                or any(a is not b for a, b in zip(projected.shape_nodes, current.binding.shape_nodes, strict=True))):
            raise CompilationError("descriptor association requires exact independent fixed scalar variable selectors")
        root = current.root_binding
        require_scope(root.declaring_scope)
        if root.dtype.startswith("type("):
            schema = types.binding(root)
            declaration = types.schema(schema)
        elif root.dtype in {"real", "integer", "logical"} and root.kind in {4, 8}:
            if root.declaring_scope.bindings.get(root.name) is not root:
                raise CompilationError("descriptor association lost its original scalar declaration")
            declaration = {"original_binding": root.public()}
        else:
            raise CompilationError("descriptor association requires supported original intrinsic or fixed derived storage")
        records.append({"alias": spelling, "selector": current.public(), "fixed_root_declaration": declaration})
        roots.append((root, declaration))
    record = {"version": 1, "kind": "original_fixed_scalar_lexical_association",
        "procedure": routine.qualified, "source_graph": graph.identity, "selected_node_ids": list(selected),
        "original_header_sha256": sha256(str(children[0]).encode()).hexdigest(),
        "selectors": records, "requires_complete_original_body": True,
        "selector_evaluation": "unchanged original association entry and guards",
        "payload_access_authorized": False, "entry_definition_authority": False,
        "alias_separation_authorized": False, "descriptor_lifetime_authorized": False,
        "execution_authorized": False, "motion_authorized": False}
    analysis.verify_publication()
    syntax = analysis._require_original_global()
    if analysis._require_original_routine(routine.qualified, _syntax=syntax) is not routine:
        raise CompilationError("descriptor association changed its original routine at publication")
    _validate_scope_roles(analysis, roles, scopes)
    for current in scopes.values():
        _validate_scope(analysis, current)
    fixed = _TypeBuilder(analysis)
    for root, declaration in roots:
        current = fixed.schema(fixed.binding(root)) if root.dtype.startswith("type(") else {"original_binding": root.public()}
        if current != declaration:
            raise CompilationError("descriptor association changed its original fixed declaration at publication")
    if (getattr(analysis, "_associate_scopes", {}).get(id(original)) is not association
            or not _attached(association, saved) or scope.bindings is not aliases
            or not _same_syntax(header_syntax)
            or len(_children(original)) != len(children)
            or any(a is not b for a, b in zip(_children(original), children, strict=True))
            or any(not _attached(access, state) or not _attached(access.binding, binding)
                   or getattr(access.binding, "component_object", None) is not root
                   for access, state, binding, root in selectors)
            or any(aliases.get(name) is not binding or not _attached(binding, state)
                   or getattr(binding, "component_object", None) is not root
                   or getattr(binding, "associate_selector", None) is not selector
                   for name, binding, state, root, selector in bindings)):
        raise CompilationError("descriptor association attachment changed at source publication")
    return _encode(record), association


@dataclass(frozen=True, slots=True)
class FixedScalarAssociation:
    _analysis: object = field(repr=False)
    _routine: object = field(repr=False)
    _original: object = field(repr=False)
    _association: object = field(repr=False)
    _record: str = field(repr=False)
    _issuer: object = field(repr=False)

    def validate(self, *, scope_validator=None, tick=None):
        saved = _attachments(self)
        if type(self) is not FixedScalarAssociation or self._issuer is not _ISSUER:
            raise CompilationError("descriptor association requires its source-issued composition")
        record, association = _derive(self._analysis, self._routine, self._original, scope_validator, tick)
        if association is not self._association or record != self._record or not _attached(self, saved):
            raise CompilationError("descriptor association changed its exact original source composition")
        return self

    def public(self):
        return json.loads(self._record)


def original_fixed_association(analysis, routine, original, *, scope_validator=None, tick=None):
    record, association = _derive(analysis, routine, original, scope_validator, tick)
    return FixedScalarAssociation(analysis, routine, original, association, record, _ISSUER).validate(
        scope_validator=scope_validator, tick=tick)
