"""Seal explicitly owned object graphs without changing their global classes.

Callers select owned types/objects and borrowed authority explicitly. Unknown
objects fail closed. Mutable containers are detached from retained aliases;
private copies thaw owned state without inheriting immutable-object identity.
"""
from copy import deepcopy
from enum import Enum
from pathlib import PurePath
from types import MappingProxyType

from compiler.ir import CompilationError

_TYPES, _ORIGINAL_TYPES = {}, {}


class ImmutableGraphMutation(CompilationError):
    """An attempted authority mutation, never a numerical support decline."""


class FrozenDict(dict):
    """Read-only JSON-compatible map with no shadowable instance methods."""
    __slots__ = ()

    def _reject(self, *args, **kwargs):
        raise ImmutableGraphMutation("owned compilation graph is immutable")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _reject

    def __copy__(self):
        return dict(self)

    def __deepcopy__(self, memo):
        result = {}
        memo[id(self)] = result
        result.update((deepcopy(key, memo), deepcopy(value, memo)) for key, value in self.items())
        return result


class FrozenList(list):
    """A detached read-only sequence; construction supports graph backedges."""
    __slots__ = ()

    def _reject(self, *args, **kwargs):
        raise ImmutableGraphMutation("owned compilation graph is immutable")

    __setitem__ = __delitem__ = append = extend = insert = pop = remove = clear = sort = reverse = _reject
    __iadd__ = __imul__ = _reject

    def __copy__(self):
        return list(self)

    def __deepcopy__(self, memo):
        result = []
        memo[id(self)] = result
        result.extend(deepcopy(value, memo) for value in self)
        return result


def sealed(value):
    return type(value) in _ORIGINAL_TYPES


def original_type(value):
    return _ORIGINAL_TYPES.get(type(value), type(value))


def _state(value):
    state = dict(getattr(value, "__dict__", {}))
    for kind in type(value).__mro__:
        slots = kind.__dict__.get("__slots__", ())
        for name in (slots,) if isinstance(slots, str) else slots:
            if name not in {"__dict__", "__weakref__"} and hasattr(value, name):
                state[name] = getattr(value, name)
    return state


def _thaw(value, memo=None):
    memo = {} if memo is None else memo
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value, FrozenDict):
        result = {}
        memo[id(value)] = result
        result.update((key, _thaw(item, memo)) for key, item in value.items())
        return result
    if isinstance(value, FrozenList):
        result = []
        memo[id(value)] = result
        result.extend(_thaw(item, memo) for item in value)
        return result
    if type(value) is tuple:
        return tuple(_thaw(item, memo) for item in value)
    return value


def mutable_copy(value):
    """Private original-class copy with detached containers and borrowed edges."""
    result = object.__new__(original_type(value))
    memo = {id(value): result}
    for name, item in _state(value).items():
        object.__setattr__(result, name, _thaw(item, memo))
    return result


def _deepcopy(value, memo):
    result = object.__new__(original_type(value))
    memo[id(value)] = result
    for name, item in _state(value).items():
        object.__setattr__(result, name, deepcopy(item, memo))
    return result


def _reject(self, name, value=None):
    raise ImmutableGraphMutation("owned compilation graph is immutable: " + name)


def _getattribute(self, name):
    value = object.__getattribute__(self, name)
    return MappingProxyType(value) if name == "__dict__" else value


def _getstate(self):
    return _state(self)


def _sealed_type(original):
    if original not in _TYPES:
        kind = type(original.__name__, (original,), {
            "__module__": original.__module__, "__slots__": (),
            "__setattr__": _reject, "__delattr__": _reject,
            "__getattribute__": _getattribute, "__getstate__": _getstate,
            "__copy__": mutable_copy, "__deepcopy__": _deepcopy,
        })
        _TYPES[original], _ORIGINAL_TYPES[kind] = kind, original
    return _TYPES[original]


class GraphFreezer:
    """One explicit ownership boundary and memo for a connected graph.

    Borrowing wins over ownership. ``borrow_if`` should recognize independently
    authenticated immutable authorities or deliberately external services; it is
    never an implicit exemption for arbitrary mutable objects. ``objects`` is
    the identity registry of owned objects, excluding containers and borrowers.
    """
    def __init__(self, *, owned_types=(), own_if=None, borrowed=(), borrowed_types=(), borrow_if=None):
        self.owned_types, self.own_if = tuple(owned_types), own_if
        self.borrowed = {id(value): value for value in borrowed}
        self.borrowed_types, self.borrow_if = tuple(borrowed_types), borrow_if
        self.memo, self.objects = {}, {}
        self._active_tuples = set()

    def freeze(self, value):
        if id(value) in self.memo:
            return self.memo[id(value)]
        if (self.borrowed.get(id(value)) is value or isinstance(value, self.borrowed_types)
                or self.borrow_if is not None and self.borrow_if(value)):
            return value
        if value is None or isinstance(value, (bool, int, float, complex, str, bytes, Enum, PurePath, type)):
            return value
        if isinstance(value, dict):
            result = FrozenDict()
            self.memo[id(value)] = result
            dict.update(result, ((self.freeze(key), self.freeze(item)) for key, item in value.items()))
            return result
        if isinstance(value, list):
            result = FrozenList()
            self.memo[id(value)] = result
            list.extend(result, (self.freeze(item) for item in value))
            return result
        if isinstance(value, tuple):
            if id(value) in self._active_tuples:
                raise CompilationError("immutable graph has an unsupported tuple backedge")
            self._active_tuples.add(id(value))
            result = tuple(self.freeze(item) for item in value)
            self._active_tuples.remove(id(value))
            self.memo[id(value)] = result
            return result
        if isinstance(value, (set, frozenset)):
            result = frozenset(self.freeze(item) for item in value)
            self.memo[id(value)] = result
            return result
        if isinstance(value, MappingProxyType):
            result = MappingProxyType(self.freeze(dict(value)))
            self.memo[id(value)] = result
            return result
        if isinstance(value, self.owned_types) or self.own_if is not None and self.own_if(value):
            if sealed(value):
                self.memo[id(value)] = value
                self.objects[id(value)] = value
                return value
            self.memo[id(value)] = value
            self.objects[id(value)] = value
            state = _state(value)
            if hasattr(value, "__dict__"):
                object.__setattr__(value, "__dict__", dict(state))
            for name, item in state.items():
                object.__setattr__(value, name, self.freeze(item))
            object.__setattr__(value, "__class__", _sealed_type(type(value)))
            return value
        raise CompilationError("immutable graph ownership is unspecified for " + type(value).__name__)


def freeze_graph(value, **ownership):
    return GraphFreezer(**ownership).freeze(value)
