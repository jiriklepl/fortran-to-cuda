"""Issuer-owned immutable derivations under one closed authority context.

The append-only index is implementation state, not an analysis registry. A
published result never points at a mutable parser projection or generated IR.
Failed and incomplete derivations are not installed.
"""

from compiler.ir import CompilationError


class DerivationArena:
    __slots__ = ("_context", "_parent", "_records", "_handles", "_active", "_derivations")

    def __init__(self, context, parent=None):
        if parent is not None and (type(parent) is not DerivationArena or parent._context is not context):
            raise CompilationError("derivation fork requires the exact immutable context")
        object.__setattr__(self, "_context", context)
        object.__setattr__(self, "_parent", parent)
        object.__setattr__(self, "_records", parent._records if parent is not None else {})
        object.__setattr__(self, "_handles", {})
        object.__setattr__(self, "_active", parent._active if parent is not None else set())
        object.__setattr__(self, "_derivations", parent._derivations if parent is not None else {})

    def __setattr__(self, name, value):
        raise CompilationError("derivation arena ownership is immutable")







    def fork(self):
        return DerivationArena(self._context, self)


