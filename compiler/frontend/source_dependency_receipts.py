"""Private exact source dependency receipts for immutable arena consumers.

Shared facts contain only encoded expected values and original object IDs.
Raw declaration/parser references remain private to each consuming handle.
"""
from dataclasses import dataclass

from compiler.frontend.source_effects import _children
from compiler.ir import CompilationError

_ABSENT = object()


def _kind(node):
    return type(node).__name__


def _part(node, kind):
    return next((item for item in getattr(node, "content", ()) if _kind(item) == kind), None)


@dataclass(frozen=True, slots=True)
class _SourceReceipt:
    analysis: object
    source: tuple
    limits: tuple
    scopes: tuple
    syntax: tuple
    declarations: tuple
    parents: tuple = ()
    layouts: tuple = ()
    components: tuple = ()
    routines: tuple = ()

    def dependencies(self):
        return self.scopes, self.syntax, self.declarations, self.parents, self.layouts, self.components, self.routines

    def require(self):
        from compiler.frontend.descriptor_intrinsics import same_intrinsic_source_attachments

        if (not same_intrinsic_source_attachments(self.analysis, self.source)
                or self.limits != (self.analysis.depth_limit, self.analysis.procedure_limit, self.analysis.operation_limit)):
            raise CompilationError("source dependency attachment changed: consumed source or namespace at publication")
        _require_dependencies(self.dependencies())


def _scope_layouts(scope):
    # Scope membership lives in these shallow lexical containers. Their
    # ordered child identities authenticate routine ownership without walking
    # executable bodies of unrelated module/contained siblings.
    nodes = (scope.node, *(node for node in _children(scope.node)
                          if _kind(node) in {"Module_Subprogram_Part", "Internal_Subprogram_Part"}))
    return tuple((node, getattr(node, "parent", None), getattr(node, "content", None),
                  getattr(node, "items", None), tuple(_children(node))) for node in nodes)


def _component_scope(scope):
    from compiler.frontend.descriptor_associations import _attachments

    namespace = getattr(scope, "component_types", None)
    entries = () if namespace is None else tuple(namespace.items())
    return (scope, namespace, entries,
            tuple((schema, _attachments(schema), schema.members, tuple(schema.members.items()))
                  for _name, schema in entries))


def _same_components(saved):
    from compiler.frontend.descriptor_associations import _attached

    scope, namespace, entries, schemas = saved
    return (getattr(scope, "component_types", None) is namespace
            and (namespace is None or len(namespace) == len(entries))
            and all(namespace.get(name) is schema for name, schema in entries)
            and all(_attached(schema, fields_) and schema.members is members
                    and len(members) == len(values) and all(members.get(name) is value for name, value in values)
                    for schema, fields_, members, values in schemas))


def _require_dependencies(dependencies):
    """Pure retained declarations, namespaces and original lexical structure."""
    from compiler.frontend.descriptor_associations import _attached
    from compiler.frontend.descriptor_intrinsics import same_intrinsic_scope_attachments
    from compiler.frontend.descriptor_teams import _same_group_syntax

    scopes, syntax, declarations, parents, layouts, components, routines = dependencies
    if (any(not same_intrinsic_scope_attachments(saved) for saved in scopes)
            or any(not _same_group_syntax(saved) for saved in syntax)
            or any(not _attached(binding, saved) for binding, saved in declarations)
            or any(getattr(node, "parent", None) is not parent for node, parent in parents)
            or any(getattr(node, "parent", None) is not parent
                   or getattr(node, "content", None) is not content or getattr(node, "items", None) is not items
                   or len(_children(node)) != len(children)
                   or any(left is not right for left, right in zip(_children(node), children, strict=True))
                   for node, parent, content, items, children in layouts)
            or any(not _same_components(saved) for saved in components)
            or any(not _attached(routine, saved) for routine, saved in routines)):
        raise CompilationError("source dependency attachment changed: consumed source or namespace at publication")


def _dependency_snapshot(dependencies):
    """Internal expected IDs only; never raw parser objects in shared facts."""
    scopes, syntax, declarations, parents, layouts, components, routines = dependencies
    def refs(values):
        return [id(value) for value in values]
    def attached(values):
        return [[name, id(value)] for name, value in values]
    namespace = []
    for scope, saved, containers, kinds, _syntax in scopes:
        nested = []
        for value, kind, entries in containers:
            content = ([[id(key), id(item)] for key, item in entries] if kind == "mapping"
                       else sorted(refs(entries)) if kind == "set" else refs(entries))
            nested.append([id(value), kind, content])
        namespace.append([id(scope), attached(saved), nested, [refs(row) for row in kinds]])
    trees = []
    for topology, nodes, containers in syntax:
        trees.append([[[id(node), refs(children)] for node, children in topology],
            [[id(node), id(string), id(item), id(content), id(items), attached(saved)]
             for node, string, item, content, items, saved in nodes],
            [[id(node), refs(children)] for node, children in containers]])
    return [namespace, trees, [[id(binding), attached(saved)] for binding, saved in declarations],
            [[id(node), id(parent)] for node, parent in parents],
            [[id(node), id(parent), id(content), id(items), refs(children)]
             for node, parent, content, items, children in layouts],
            [[id(scope), id(namespace), [[id(name), id(schema)] for name, schema in entries],
              [[id(schema), attached(saved), id(members), [[id(name), id(value)] for name, value in values]]
               for schema, saved, members, values in schemas]]
             for scope, namespace, entries, schemas in components],
            [[id(routine), attached(saved)] for routine, saved in routines]]


class DependencyCapture:
    def __init__(self, analysis, *, components=False, inherited=None):
        from compiler.frontend.descriptor_intrinsics import intrinsic_source_attachments

        self.analysis = analysis
        self.scope_receipts, self.syntax = {}, {}
        self.syntax_roots, self.parents, self.layouts = {}, {}, {}
        self.limits = (analysis.depth_limit, analysis.procedure_limit, analysis.operation_limit)
        self.declarations, self.routine_receipts = {}, {}
        self.component_scopes, self._components = {}, components
        self._inherited, self._ownership = inherited is not None, None
        self._covered_syntax = set()
        if inherited is None:
            self.source = intrinsic_source_attachments(analysis)
        else:
            # This imports expected private attachments, never a successful
            # validation result. The consuming issuer authenticates the
            # source handle; every imported row is checked again at its final
            # publication fence, together with newly consulted dependencies.
            if type(inherited) is _SourceReceipt and inherited.analysis is analysis:
                self.source, self.limits = inherited.source, inherited.limits
                dependencies = inherited.dependencies()
            elif (type(inherited) is _PrivateDependencies and type(inherited.ownership) is _SourceOwnership
                  and inherited.ownership.analysis is analysis):
                self._ownership = inherited.ownership
                self.source = self._ownership.source
                policy = dict(self._ownership.policy)
                self.limits = tuple(policy[name] for name in ("depth_limit", "procedure_limit", "operation_limit"))
                dependencies = inherited.saved
            else:
                raise CompilationError("source dependency composition requires the same analysis private receipt")
            scopes, syntax, declarations, parents, layouts, schemas, routines = dependencies
            self.scope_receipts.update((id(saved[0]), saved) for saved in scopes)
            self.syntax.update((("inherited", index), saved) for index, saved in enumerate(syntax))
            self.declarations.update((id(row[0]), row) for row in declarations)
            self.parents.update((id(row[0]), row) for row in parents)
            self.layouts.update((id(row[0]), row) for row in layouts)
            self.component_scopes.update((id(row[0]), row) for row in schemas)
            self.routine_receipts.update((id(row[0]), row) for row in routines)
            self._covered_syntax.update(id(row[0]) for saved in syntax for row in saved[1])
            if components and any(identity not in self.component_scopes for identity in self.scope_receipts):
                raise CompilationError("source dependency composition lacks original component namespace receipts")

    def retain_routine(self, routine):
        from compiler.frontend.descriptor_associations import _attachments

        if id(routine) in self.routine_receipts:
            return False
        self.routine_receipts[id(routine)] = routine, _attachments(routine)
        self.retain_syntax(id(routine), (routine.scope.node,))
        for binding in routine.scope.bindings.values():
            if id(binding) not in self.declarations:
                self.declarations[id(binding)] = binding, _attachments(binding)
        return True

    def retain_scope(self, scope):
        if scope is not None and id(scope) not in self.scope_receipts:
            from compiler.frontend.descriptor_associations import _attachments
            from compiler.frontend.descriptor_intrinsics import intrinsic_scope_attachments

            # Imported/host declarations participate in the same original
            # signature lookup as own formals. Keep their fields, not merely
            # namespace membership; do not reseed an already consumed binding.
            for binding in scope.bindings.values():
                if id(binding) not in self.declarations:
                    self.declarations[id(binding)] = binding, _attachments(binding)
            if self._components:
                component = _component_scope(scope)
                self.component_scopes[id(scope)] = component
            saved = intrinsic_scope_attachments(scope, syntax=False)
            for layout in _scope_layouts(scope):
                self.layouts.setdefault(id(layout[0]), layout)
            self.scope_receipts[id(scope)] = saved
            # Kind environments may borrow original PARAMETER expressions
            # from an imported scope. Reuse the existing syntax receipt for
            # those exact dependency roots as well as this specification.
            constants = tuple(value for _owner, _parent, values, _intrinsic in saved[3]
                              for value in values.values())
            specification = _part(scope.node, "Specification_Part")
            headers = tuple(node for node in _children(scope.node)
                            if _kind(node) not in {"Specification_Part", "Execution_Part", "Module_Subprogram_Part",
                                                   "Internal_Subprogram_Part"})
            self.retain_syntax(("scope", id(scope)), (*headers, specification, *constants))
            # Resolved widths are stored on declarations, but their original
            # kind expressions can name imported PARAMETERs. Follow exactly
            # those original selectors/constants through the existing source
            # resolver before any delegated validation, retaining each reached
            # namespace once. Module executable siblings are never traversed.
            from fparser.two.utils import walk

            from compiler.frontend.source_functions import _validate_resolution

            selectors = tuple(node for node in walk(specification) if _kind(node) == "Kind_Selector")
            for root in (*selectors, *constants):
                _validate_resolution(self.analysis, scope, tuple(walk(root)), _scope_validator=self.retain_scope)

    def retain_syntax(self, key, roots):
        from fparser.two.utils import Base

        from compiler.frontend.descriptor_teams import _group_syntax_checkpoint

        roots = tuple(root for root in roots if isinstance(root, Base)
                      and (not self._inherited or id(root) not in self._covered_syntax))
        if self._inherited and not roots:
            return
        checkpoint = _group_syntax_checkpoint(roots)
        self.syntax[key], self.syntax_roots[key] = checkpoint, roots
        if self._inherited:
            self._covered_syntax.update(id(row[0]) for row in checkpoint[1])
        for node, *_fields in checkpoint[1]:
            self.parents.setdefault(id(node), (node, getattr(node, "parent", None)))

    def receipt(self):
        syntax = tuple(self.syntax.values())
        if self._inherited:
            # Shared roots may appear in a routine body, its specification and
            # an imported kind expression. Keep the first expected row for
            # each exact object; a later capture must never reseed that row.
            rows = ({}, {}, {})
            for checkpoint in syntax:
                for collected, group in zip(rows, checkpoint, strict=True):
                    for row in group:
                        collected.setdefault(id(row[0]), row)
            syntax = (tuple(tuple(group.values()) for group in rows),)
        dependencies = (tuple(self.scope_receipts.values()), syntax, tuple(self.declarations.values()),
                        tuple(self.parents.values()), tuple(self.layouts.values()),
                        tuple(self.component_scopes.values()), tuple(self.routine_receipts.values()))
        if self._ownership is not None:
            return _PrivateDependencies(dependencies, self._ownership)
        return _SourceReceipt(self.analysis, self.source, self.limits, *dependencies)

    def recipe(self):
        if self._inherited:
            raise CompilationError("a private consuming dependency composition is not a shared arena fact")
        return {"scopes": list(self.scope_receipts), "declarations": list(self.declarations),
                "components": list(self.component_scopes), "routines": list(self.routine_receipts),
                "syntax_roots": [[id(node) for node in roots] for roots in self.syntax_roots.values()],
                "expected": _dependency_snapshot(self.receipt().dependencies())}


@dataclass(frozen=True, slots=True)
class _SourceOwnership:
    analysis: object
    source: tuple
    policy: tuple

    @classmethod
    def capture(cls, analysis):
        from compiler.frontend.authority_context import CONTEXT_FIELDS
        from compiler.frontend.descriptor_intrinsics import intrinsic_source_attachments

        return cls(analysis, intrinsic_source_attachments(analysis),
                   tuple((name, getattr(analysis, name, _ABSENT)) for name in sorted(CONTEXT_FIELDS)))

    def require(self):
        from compiler.frontend.descriptor_intrinsics import same_intrinsic_source_attachments

        if (not same_intrinsic_source_attachments(self.analysis, self.source)
                or any(getattr(self.analysis, name, _ABSENT) is not value for name, value in self.policy)):
            raise CompilationError("source dependency retained source or policy changed")


@dataclass(frozen=True, slots=True)
class _PrivateDependencies:
    saved: tuple
    ownership: object

    def require(self):
        self.ownership.require()
        _require_dependencies(self.saved)


def _resolve_dependencies(context, recipe, ownership):
    from compiler.frontend.descriptor_associations import _attachments
    from compiler.frontend.descriptor_intrinsics import intrinsic_scope_attachments
    from compiler.frontend.descriptor_teams import _group_syntax_checkpoint

    session = context.session
    def original(identity):
        value = session.objects.get(identity)
        if value is None or id(value) != identity:
            raise CompilationError("terminating arena dependency is not original session storage")
        return value
    scopes = tuple(intrinsic_scope_attachments(original(identity), syntax=False) for identity in recipe["scopes"])
    # Some imported kind environments hold compiler-owned expression copies,
    # rather than nodes in the lexical source index. Accept only exact values
    # reached through the already retained consulted kind maps; their map,
    # entry and syntax identities are all compared with the immutable recipe.
    kind_roots = {id(value): value for saved in scopes for _owner, _parent, values, _intrinsic in saved[3]
                  for value in values.values()}
    def syntax_root(identity):
        try:
            return session.original_node(identity)
        except CompilationError:
            value = kind_roots.get(identity)
            if value is None:
                raise
            return value
    syntax = tuple(_group_syntax_checkpoint(tuple(syntax_root(identity) for identity in roots))
                   for roots in recipe["syntax_roots"])
    declarations = tuple((original(identity), _attachments(original(identity))) for identity in recipe["declarations"])
    parents = {}
    for checkpoint in syntax:
        for node, *_fields in checkpoint[1]:
            parents.setdefault(id(node), (node, getattr(node, "parent", None)))
    layouts = {}
    for scope, *_fields in scopes:
        for layout in _scope_layouts(scope):
            layouts.setdefault(id(layout[0]), layout)
    components = tuple(_component_scope(original(identity)) for identity in recipe["components"])
    routines = tuple((original(identity), _attachments(original(identity))) for identity in recipe["routines"])
    dependencies = scopes, syntax, declarations, tuple(parents.values()), tuple(layouts.values()), components, routines
    if _dependency_snapshot(dependencies) != recipe["expected"]:
        raise CompilationError("terminating arena original dependency changed before consumption")
    return _PrivateDependencies(dependencies, ownership)


