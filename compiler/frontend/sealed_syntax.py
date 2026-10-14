"""Read-only original parser syntax for one compilation session.

Generated projections use ordinary mutable parser classes. Copying an original
node explicitly returns such a projection; it does not inherit source authority.
Parser reader caches and parent backreferences are not source syntax.
"""

from copy import deepcopy
from types import MappingProxyType

from fparser.common.readfortran import FortranReaderBase
from fparser.two.utils import Base

from compiler.ir import CompilationError

_TYPES = {}
_ORIGINAL_TYPES = {}
_ITEM_FIELDS = frozenset({"span", "fort_original_span", "label", "name", "line", "comment",
                          "is_f2py_directive"})


class FrozenList(tuple):
    """A parser list whose order cannot change after source authentication."""

    def _reject(self, *args, **kwargs):
        raise CompilationError("original compilation-session syntax is immutable")

    __setitem__ = __delitem__ = append = extend = insert = pop = remove = clear = sort = reverse = _reject
    __iadd__ = __imul__ = _reject

    def __copy__(self):
        return list(self)

    def __deepcopy__(self, memo):
        result = []
        memo[id(self)] = result
        result.extend(deepcopy(value, memo) for value in self)
        return result


def _thaw(value):
    if isinstance(value, FrozenList):
        return [_thaw(item) for item in value]
    if type(value) is tuple:
        return tuple(_thaw(item) for item in value)
    return value


def mutable_copy(value):
    """Copy parser state into a mutable projection, including parser comments.

    Comments lack the ``string`` needed by fparser's default copy protocol.
    This preserves their provenance without copying or changing their reader.
    """
    result = object.__new__(_ORIGINAL_TYPES.get(type(value), type(value)))
    vars(result).update({key: _thaw(item) for key, item in object.__getattribute__(value, "__dict__").items()})
    return result


def _deepcopy(self, memo):
    result = object.__new__(_ORIGINAL_TYPES[type(self)])
    memo[id(self)] = result
    # Reader streams are external parser state, not owned syntax. Parent links
    # into the copied subtree use its memo; a root's outside parent stays shared.
    for name, value in object.__getattribute__(self, "__dict__").items():
        vars(result)[name] = (memo.get(id(value), value) if name == "parent"
                             or isinstance(value, FortranReaderBase) else deepcopy(value, memo))
    return result


def _getattribute(self, name):
    value = object.__getattribute__(self, name)
    return MappingProxyType(value) if name == "__dict__" else value


def _getstate(self):
    # The default object protocol exposes the backing dictionary even when
    # __dict__ itself is read-only. Serialization gets detached state too.
    return dict(object.__getattribute__(self, "__dict__"))


def _node_setattr(self, name, value):
    if name == "parent":
        object.__setattr__(self, name, value)
        return
    raise CompilationError("original compilation-session syntax is immutable: " + name)


def _item_setattr(self, name, value):
    if name not in _ITEM_FIELDS and name not in {"__dict__", "__class__"}:
        object.__setattr__(self, name, value)
        return
    raise CompilationError("original compilation-session provenance is immutable: " + name)


def _delattr(self, name):
    raise CompilationError("original compilation-session syntax is immutable: " + name)


def _sealed_type(original, *, item=False):
    key = original, item
    if key not in _TYPES:
        sealed = type(original.__name__, (original,), {
            "__module__": original.__module__,
            "__setattr__": _item_setattr if item else _node_setattr,
            "__delattr__": _delattr,
            "__getattribute__": _getattribute,
            "__getstate__": _getstate,
            "__copy__": mutable_copy,
            "__deepcopy__": _deepcopy,
        })
        _TYPES[key] = sealed
        _ORIGINAL_TYPES[sealed] = original
    return _TYPES[key]


def sealed(value):
    return type(value) in _ORIGINAL_TYPES


def seal(roots):
    """Seal already-authenticated original nodes in place, preserving identity."""
    visited = set()

    def freeze_value(value):
        if isinstance(value, Base):
            freeze_node(value)
            return value
        if type(value) is list:
            return FrozenList(freeze_value(item) for item in value)
        if type(value) is tuple:
            return tuple(freeze_value(item) for item in value)
        return value

    def freeze_node(node):
        if id(node) in visited or sealed(node):
            return
        visited.add(id(node))
        # A caller may already hold vars(node). Detach that mutable dictionary,
        # just as child lists are replaced, before granting immutable authority.
        object.__setattr__(node, "__dict__", dict(vars(node)))
        for name, value in tuple(vars(node).items()):
            if name not in {"parent", "item"}:
                object.__setattr__(node, name, freeze_value(value))
        item = getattr(node, "item", None)
        if item is not None and not sealed(item):
            object.__setattr__(item, "__dict__", dict(vars(item)))
            for name in _ITEM_FIELDS:
                if hasattr(item, name):
                    object.__setattr__(item, name, freeze_value(getattr(item, name)))
            object.__setattr__(item, "__class__", _sealed_type(type(item), item=True))
        object.__setattr__(node, "__class__", _sealed_type(type(node)))

    for root in roots:
        if root is not None:
            freeze_node(root)
