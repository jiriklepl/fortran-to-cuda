"""One authenticated, immutable source graph shared by compilation projections.

Original declaration/syntax state and persistent projection values are sealed.
Effect caches, context interning and generated handles are separate work state.
A session query authenticates graph identities; publication authenticates files.
"""
from collections.abc import Mapping
from types import MappingProxyType

from compiler.ir import CompilationError


class ProjectedRoutines(Mapping):
    """Persistent private routine selection over one immutable original registry."""
    __slots__ = ("base", "_overrides")

    def __init__(self, base, overrides=()):
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "_overrides", MappingProxyType(dict(overrides)))

    def __setattr__(self, name, value):
        raise CompilationError("private projection registry identity is immutable")

    @property
    def overrides(self):
        return self._overrides

    def __getitem__(self, key):
        return self._overrides[key] if key in self._overrides else self.base[key]

    def __setitem__(self, key, value):
        raise CompilationError("routine projection is immutable; use project_routine")


    def __delitem__(self, key):
        raise CompilationError("private projections cannot remove original source roles")

    def __iter__(self):
        yield from self.base
        yield from (key for key in self._overrides if key not in self.base)

    def __len__(self):
        return len(self.base) + len(self._overrides.keys() - self.base.keys())

    def __copy__(self):
        return self






