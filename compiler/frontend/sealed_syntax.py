"""Read-only original parser syntax for one compilation session.

Generated projections use ordinary mutable parser classes. Copying an original
node explicitly returns such a projection; it does not inherit source authority.
Lexical parents are sealed source edges. Reader caches remain external to
semantic authority; source provenance and the reader attachment are immutable.
"""

from copy import deepcopy

from compiler.frontend.immutable_graph import ImmutableGraphMutation


class FrozenList(tuple):
    """A parser list whose order cannot change after source authentication."""

    __slots__ = ()

    def _reject(self, *args, **kwargs):
        raise ImmutableGraphMutation("original compilation-session syntax is immutable")

    __setitem__ = __delitem__ = append = extend = insert = pop = remove = clear = sort = reverse = _reject
    __iadd__ = __imul__ = _reject

    def __copy__(self):
        return list(self)

    def __deepcopy__(self, memo):
        result = []
        memo[id(self)] = result
        result.extend(deepcopy(value, memo) for value in self)
        return result






















