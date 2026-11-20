"""Seal explicitly owned object graphs without changing their global classes.

Callers select owned types/objects and borrowed authority explicitly. Unknown
objects fail closed. Mutable containers are detached from retained aliases;
private copies thaw owned state without inheriting immutable-object identity.
"""

from compiler.ir import CompilationError

_TYPES, _ORIGINAL_TYPES = {}, {}


class ImmutableGraphMutation(CompilationError):
    """An attempted authority mutation, never a numerical support decline."""


























