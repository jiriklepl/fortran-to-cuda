"""Fixed scalar fields in original native arrays of derived objects.

These objects never become numerical GPU captures. A consumer must reserve
their original storage range with the scoped runtime before native execution,
so an alias of an already managed numerical buffer closes ownership safely.
"""

from copy import copy

from compiler.ir import CompilationError


def access(analysis, scope, node):
    from compiler.frontend.component_bindings import ComponentAccess, _children, _field_binding, _kind

    parts = tuple(node.items)
    if len(parts) != 2 or _kind(parts[0]) != 'Part_Ref' or _kind(parts[1]) != 'Name':
        raise CompilationError('native metadata requires an array element and one fixed scalar field')
    base, subscripts = parts[0].items
    root = analysis._binding(scope, base)
    if (root is None or not root.rank or not root.dtype.startswith('type(')
            or root.attributes & {'pointer', 'optional', 'volatile', 'asynchronous', 'value'}):
        raise CompilationError('native metadata requires an original nonpolymorphic array descriptor')
    indices = tuple(_children(subscripts))
    if len(indices) != root.rank or any(_kind(index) == 'Subscript_Triplet' for index in indices):
        raise CompilationError('native metadata requires complete scalar element coordinates')
    # The selected scalar retains its declaring type and visibility checks.
    # Allocation lifetime belongs to the original array object, never to a
    # synthetic scalar field or a packed numerical capture.
    element = copy(root)
    element.rank, element.shape_nodes, element.lower_bounds, element.lower_bound_nodes = 0, (), (), ()
    member = str(parts[1]).lower()
    binding, _ = _field_binding(analysis, root.declaring_scope or scope, element, member, node,
                                consumer_scope=scope)
    if binding.rank or binding.dtype not in {'integer', 'real', 'logical'}:
        raise CompilationError('native metadata field requires fixed scalar numeric storage')
    binding.native_metadata_object = root
    binding.component_object = root
    binding.name = str(node).lower()
    return ComponentAccess(binding, root, node, (member,), indices)
