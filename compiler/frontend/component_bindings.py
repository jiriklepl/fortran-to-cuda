"""Original-source identities for fixed fields and lexical ASSOCIATE aliases.

This provider never replaces source declarations or evaluates selectors.  A
numeric leaf identifies storage in its original object; unsupported siblings do
not change the proof for that leaf.  Array-valued objects, dynamic components
and expression selectors deliberately retain an explicit source boundary.
"""

from __future__ import annotations

import re
from copy import copy
from dataclasses import dataclass, field

from fparser.two.utils import Base

from compiler.frontend.syntax import children_of, declared_type, kind_of
from compiler.ir import CompilationError, SourceLocation
from compiler.ir.integers import fits_integer


class ComponentAccessUnsupported(CompilationError):
    """A selector syntax support limit, never source or budget authority."""


class ComponentStorageUnsupported(CompilationError):
    """Fixed payload layout is unavailable; no storage-read proof is implied."""


@dataclass(frozen=True)
class ComponentSchema:
    name: str
    scope: object = field(compare=False, repr=False)
    node: object = field(compare=False, repr=False)
    members: dict = field(compare=False, repr=False)
    private: bool = False
    reason: str | None = None


@dataclass(frozen=True)
class ComponentAccess:
    binding: object = field(compare=False, repr=False)
    root_binding: object = field(compare=False, repr=False)
    selector: object = field(compare=False, repr=False)
    path: tuple[str, ...] = ()
    indices: tuple[object, ...] = field(default=(), compare=False, repr=False)

    def public(self):
        return {"resource": self.binding.root, "object_resource": self.root_binding.root,
                "component_path": list(self.path), "source_selector": str(self.selector),
                "descriptor": self.binding.public(),
                "evaluation": "original reached source point; no selector evaluation during analysis"}


def register_types(analysis, scope, spec):
    """Retain schemas, including unsupported fields, without global rejection."""
    schemas = {}
    for node in children_of(spec):
        if kind_of(node) != "Derived_Type_Def":
            continue
        header = next((item for item in children_of(node) if kind_of(item) == "Derived_Type_Stmt"), None)
        if header is None:
            continue
        attributes, name, parameters = header.items
        name = str(name).lower()
        flags = tuple(str(item).lower() for item in children_of(attributes))
        if "public" in flags or "private" in flags:
            scope.access.setdefault(name, "public" in flags)
        reason = ("parameterized or extended derived types require a complete field-layout proof"
                  if parameters is not None or any(flag.startswith("extends") for flag in flags) else None)
        members, private = {}, False
        for part in children_of(node):
            if kind_of(part) in {"Private_Components_Stmt", "Private_Stmt"}:
                private = True
            if kind_of(part) != "Component_Part":
                continue
            for declaration in children_of(part):
                if kind_of(declaration) in {"Private_Components_Stmt", "Private_Stmt"}:
                    private = True
                elif kind_of(declaration) == "Data_Component_Def_Stmt":
                    dtype, attrs, entities = declaration.items
                    for entity in children_of(entities):
                        member = str(entity.items[0]).lower()
                        if member in members:
                            reason = "duplicate derived component declaration: " + member
                        members[member] = (dtype, tuple(children_of(attrs)), entity)
        schemas[name] = ComponentSchema(name, scope, node, members, private, reason)
    scope.component_types = schemas


def _schema(analysis, scope, name, visited=frozenset()):
    name = str(name).lower()
    key = id(scope), name
    if scope is None or key in visited:
        return None
    local = getattr(scope, "component_types", {}).get(name)
    if local is not None:
        return local
    found = []
    sources = analysis.use_sources(scope, name)
    if sources is None:
        return None
    for module, remote in sources:
        owner = analysis.modules.get(module)
        if owner is None:
            return None
        if not analysis._exported(owner, remote):
            if name in scope.imports and scope.imports[name] == (module, remote):
                return None
            continue
        result = _schema(analysis, owner, remote, visited | {key})
        if result is not None:
            found.append(result)
    choices = {id(item.node): item for item in found}
    if choices:
        return next(iter(choices.values())) if len(choices) == 1 else None
    return _schema(analysis, scope.parent, name, visited | {key}) if scope.parent else None


def _typename(dtype):
    match = re.fullmatch(r"type\s*\(\s*([a-z][a-z0-9_]*)\s*\)", str(dtype).lower())
    return match.group(1) if match else None


def _field_binding(analysis, scope, parent, member, node, *, consumer_scope=None):
    from compiler.frontend.source_effects import Binding

    typename = _typename(parent.dtype)
    schema = _schema(analysis, scope, typename) if typename else None
    if schema is None:
        raise CompilationError("source component type is unavailable or ambiguous: " + str(parent.dtype))
    if schema.reason:
        raise CompilationError(schema.reason)
    declaration = schema.members.get(member)
    if declaration is None:
        raise CompilationError("source component declaration is unavailable: " + member)
    dtype, attributes, entity = declaration
    flags = frozenset(str(item).split("(")[0].lower() for item in attributes)
    if (schema.private or "private" in flags) and (consumer_scope or scope).module != schema.scope.module:
        raise CompilationError("source component is private in its defining module: " + member)
    if flags & {"pointer", "allocatable", "volatile", "asynchronous"}:
        raise ComponentStorageUnsupported("source component association or lifetime is uncertain: " + member)
    if kind_of(dtype) == "Intrinsic_Type_Spec":
        if str(dtype.items[0]).lower() not in {"real", "integer", "logical", "double precision"}:
            raise CompilationError("source component requires fixed numeric storage: " + member)
        try:
            base, width = declared_type(dtype, schema.scope.kinds, SourceLocation(str(scope.path)))
        except CompilationError as error:
            raise CompilationError("source component kind is not fixed: " + member) from error
        if (base, width) not in {("real", 4), ("real", 8), ("integer", 4), ("integer", 8),
                                 ("logical", 1), ("logical", 4)}:
            raise CompilationError("unsupported source component numeric kind: " + member)
    elif _typename(dtype):
        base, width = str(dtype).lower(), None
    else:
        raise CompilationError("source component requires nonpolymorphic fixed storage: " + member)
    dimension = next((item.items[1] for item in attributes if kind_of(item) == "Dimension_Component_Attr_Spec"), None)
    shape = entity.items[1] if entity.items[1] is not None else dimension
    axes = tuple(children_of(shape))
    lowers = []
    for axis in axes:
        if kind_of(axis) != "Explicit_Shape_Spec":
            raise CompilationError("source component requires fixed explicit extents: " + member)
        lower, upper = axis.items
        try:
            lo = 1 if lower is None else schema.scope.constant(lower)
            hi = schema.scope.constant(upper)
        except CompilationError as error:
            raise CompilationError("source component extents are not fixed: " + member) from error
        if not fits_integer(lo) or not fits_integer(hi):
            raise CompilationError("source component bounds exceed supported INTEGER ABI: " + member)
        lowers.append(str(lo))
    if len(axes) > 4:
        raise CompilationError("source component rank exceeds supported layout: " + member)
    inherited = parent.attributes & {"save", "target", "contiguous", "intent"}
    binding = Binding(str(node).lower(), parent.root + "%" + member, base, width, len(axes), parent.intent,
                      frozenset(flags | inherited), tuple(lowers), tuple(axis.items[0] for axis in axes), axes)
    binding.component_object = getattr(parent, "component_object", parent)
    return binding, schema.scope


def _element_field(analysis, scope, node):
    """``objects(i)%field``: one numeric field of the elements of a rank-one array of derived type.

    The field of all elements is one array resource, ``objects%field``, with the
    shape and bounds of ``objects``. Its storage is strided through the array of
    structures, which the ``element_field`` attribute tells consumers.
    """
    from dataclasses import replace

    reference, member = node.items
    root = analysis._binding(scope, reference.items[0])
    if root is None:
        raise CompilationError("source component object is unresolved: " + str(reference.items[0]))
    indices = tuple(children_of(reference.items[1]))
    if root.rank != 1 or len(indices) != 1 or kind_of(indices[0]) == "Subscript_Triplet":
        raise ComponentAccessUnsupported("source component needs one element of a rank-one array of derived type: "
                                         + str(node))
    if root.attributes & {"pointer", "optional", "volatile", "asynchronous"}:
        raise CompilationError("source component object association or lifetime is uncertain: " + root.root)
    field_binding, _ = _field_binding(analysis, root.declaring_scope or scope, root, str(member).lower(), node,
                                      consumer_scope=scope)
    if field_binding.rank or (field_binding.dtype, field_binding.kind) not in {("real", 4), ("real", 8),
                                                                                ("integer", 4), ("logical", 4)}:
        raise ComponentStorageUnsupported("an element field needs a scalar REAL, default INTEGER or default LOGICAL "
                                          "component: " + str(node))
    binding = replace(field_binding, name=root.name + "%" + str(member).lower(), rank=root.rank,
                      attributes=frozenset((field_binding.attributes | {"element_field"}) - {"allocatable"}),
                      lower_bounds=root.lower_bounds, lower_bound_nodes=root.lower_bound_nodes,
                      shape_nodes=root.shape_nodes)
    binding.component_object = root
    return ComponentAccess(binding, root, node, (str(member).lower(),), indices)


def component_access(analysis, scope, node):
    if kind_of(node) != "Data_Ref":
        return None
    parts = tuple(node.items)
    if (getattr(analysis, '_native_metadata', False) and parts and kind_of(parts[0]) == 'Part_Ref'):
        from compiler.frontend.native_metadata import access
        return access(analysis, scope, node)
    # Resident entries bind a field of every element as one array; other consumers keep the boundary.
    if (getattr(analysis, "element_fields", False) and len(parts) == 2 and kind_of(parts[0]) == "Part_Ref"
            and kind_of(parts[1]) == "Name"):
        return _element_field(analysis, scope, node)
    if len(parts) < 2 or kind_of(parts[0]) != "Name":
        raise ComponentAccessUnsupported("source component needs one scalar variable object: " + str(node))
    root = analysis._binding(scope, parts[0])
    if root is None:
        raise CompilationError("source component object is unresolved: " + str(parts[0]))
    if root.rank or root.attributes & {"pointer", "allocatable", "optional", "volatile", "asynchronous"}:
        raise CompilationError("source component object association or lifetime is uncertain: " + root.root)
    # USE may import only the object, with several renamed re-exports. Its
    # declared type need not be visible (and can be shadowed) at the use site.
    # Resolve the type in the original declaration's scope, while retaining
    # the consumer scope for private-component access checks.
    current, defining_scope, path, indices = root, root.declaring_scope or scope, [], ()
    for index, part in enumerate(parts[1:], 1):
        if current.rank:
            raise CompilationError("array-valued derived component needs an explicit element mapping")
        if kind_of(part) == "Name":
            member = str(part).lower()
        elif kind_of(part) == "Part_Ref" and index == len(parts) - 1:
            member = str(part.items[0]).lower()
            indices = tuple(children_of(part.items[1]))
        else:
            raise CompilationError("source component selector needs a fixed field path: " + str(node))
        current, defining_scope = _field_binding(analysis, defining_scope, current, member, node,
                                                consumer_scope=scope)
        path.append(member)
    if indices and (not current.rank or len(indices) != current.rank):
        raise CompilationError("source component subscript rank differs from its declaration")
    current.name = "%".join(str(part.items[0] if kind_of(part) == "Part_Ref" else part).lower()
                            for part in parts)
    return ComponentAccess(current, getattr(root, "component_object", root), node, tuple(path), indices)


def resolve_binding(analysis, scope, node):
    access = component_access(analysis, scope, node)
    return access.binding if access is not None else None


def source_scope_for(analysis, node, default=None):
    registered = getattr(analysis, "_source_scopes", {}).get(id(node))
    return registered if registered is not None else default


def register_associates(analysis, routine):
    """Index lexical scopes; an ASSOCIATE with a selector that is not a plain scalar variable gets none."""
    from compiler.frontend.source_effects import Scope

    def visit(node, scope, path):
        analysis._source_scopes[id(node)] = scope
        if kind_of(node) == "Block_Construct":
            child = Scope(scope.module, scope.path, node, scope)
            child.qualified = routine.qualified + "$block:" + ".".join(map(str, path))
            start = len(analysis.kind_expressions)
            child.issues = analysis._specification(child, qualified=child.qualified)
            analysis._import_kinds(child)
            for binding, owner, selector in analysis.kind_expressions[start:]:
                try:
                    binding.kind = owner.constant(selector)
                except CompilationError:
                    binding.kind = None
            entered = False
            for index, item in enumerate(children_of(node)):
                if kind_of(item) == "Block_Stmt":
                    visit(item, scope, (*path, index))
                    entered = True
                else:
                    visit(item, child if entered else scope, (*path, index))
            return
        if kind_of(node) == "Associate_Construct":
            header = next((item for item in children_of(node) if kind_of(item) == "Associate_Stmt"), None)
            child = Scope(scope.module, scope.path, node, scope, kinds=scope.kinds)
            child.qualified = routine.qualified + "$associate:" + ".".join(map(str, path))
            child.component_types = {}
            try:
                if header is None:
                    raise CompilationError("source ASSOCIATE has no original selector list")
                for association in children_of(header.items[1]):
                    alias, _, selector = association.items
                    alias = str(alias).lower()
                    if kind_of(selector) == "Name":
                        binding = analysis._binding(scope, selector)
                        if binding is None:
                            raise CompilationError("source ASSOCIATE selector is unresolved")
                        access = ComponentAccess(binding, getattr(binding, "component_object", binding), selector)
                    elif kind_of(selector) == "Data_Ref":
                        access = component_access(analysis, scope, selector)
                    else:
                        raise CompilationError("source ASSOCIATE requires an original scalar variable selector")
                    binding = access.binding
                    if access.indices or binding.rank or binding.attributes & {
                            "pointer", "allocatable", "optional", "volatile", "asynchronous"}:
                        raise CompilationError("source ASSOCIATE selector association or bounds are unsupported")
                    if alias in child.bindings:
                        raise CompilationError("duplicate source ASSOCIATE alias: " + alias)
                    projected = copy(binding)
                    projected.name = alias
                    child.bindings[alias] = projected
                available = True
            except CompilationError:
                available = False
            if available:
                for index, item in enumerate(children_of(node)):
                    # Selector expressions execute in the enclosing scope.
                    visit(item, scope if item is header else child, (*path, index))
                return
            if header is not None:
                for index, item in enumerate(children_of(node)):
                    if item is header:
                        visit(item, scope, (*path, index))
            return
        for index, child in enumerate(children_of(node)):
            if isinstance(child, (Base, tuple, list)) or hasattr(child, "content"):
                visit(child, scope, (*path, index))

    if routine.execution is not None:
        visit(routine.execution, routine.scope, ())


def references(analysis, scope, node):
    """Yield whole resolved references; field names are never variable names."""
    scope = source_scope_for(analysis, node, scope)
    kind = kind_of(node)
    if kind == "Data_Ref":
        access = component_access(analysis, scope, node)
        yield access.binding, node
        for index in access.indices:
            yield from references(analysis, scope, index)
        return
    if kind in {"Name", "Part_Ref"}:
        base = node.items[0] if kind == "Part_Ref" else node
        binding = analysis._binding(scope, base)
        if binding is not None:
            yield binding, node
        if kind == "Part_Ref":
            for index in children_of(node.items[1]):
                yield from references(analysis, scope, index)
        return
    if kind == "Actual_Arg_Spec":
        yield from references(analysis, scope, node.items[1])
        return
    for child in children_of(node):
        if isinstance(child, (Base, tuple, list)) or hasattr(child, "content"):
            yield from references(analysis, scope, child)
