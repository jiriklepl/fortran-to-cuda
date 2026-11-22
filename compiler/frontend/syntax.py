"""The compiler's view of fparser trees: one parser, node kinds and parts, declared types.

Every pass reads Fortran text through ``parse_fortran``, which keeps the trees
of the texts it parsed last. The passes of a preparation each walk all the
sources of an application and most leave most sources as they are, so a pass
finds most of its trees already parsed. A tree is shared with other callers
and must not be changed.
"""

from __future__ import annotations

from functools import lru_cache

from fparser.common.readfortran import FortranStringReader
from fparser.two.parser import ParserFactory

# More trees than an application has sources, so that a pass over all of them does not evict the trees the next
# pass reads. A tree takes some tens of times the memory of its text.
_KEPT_TREES = 256

_PARSER = None


def parser():
    """The Fortran 2008 parser, created once: creating one registers all its syntax classes."""
    global _PARSER
    if _PARSER is None:
        _PARSER = ParserFactory().create(std="f2008")
    return _PARSER


@lru_cache(maxsize=_KEPT_TREES)
def parse_fortran(text: str):
    """The tree of free-form Fortran text, comments kept; shared, so it must not be changed."""
    return parser()(FortranStringReader(text, ignore_comments=False))


def kind_of(node):
    """The name of a node's syntax class, such as ``Assignment_Stmt``."""
    return type(node).__name__


def children_of(node):
    """A construct's statements, a statement's items, or the members of a tuple or list; empty for anything else."""
    if isinstance(node, (tuple, list)):
        return node
    return getattr(node, "content", getattr(node, "items", ())) or ()


def part_of(node, name):
    """The first child of the given kind, or None."""
    return next((child for child in children_of(node) if kind_of(child) == name), None)


def declared_type(spec, kinds, location):
    """The type and kind that an ``Intrinsic_Type_Spec`` declares, DOUBLE PRECISION as REAL(8).

    The kind selector is evaluated by ``kinds`` (a scope's ``_KindScope``); one
    that cannot be evaluated raises ``CompilationError`` at ``location``.
    Without a selector the kind is 4.
    """
    base, selector = spec.items
    base = str(base).lower()
    if base == "double precision":
        return "real", 8
    if selector is None:
        return base, 4
    return base, kinds.integer(selector.items[1], location)
