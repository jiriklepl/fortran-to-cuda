"""Lower compiler-owned source text with deterministic diagnostic identity."""

from fparser.common.readfortran import FortranStringReader
from fparser.two.parser import ParserFactory
from fparser.two.utils import FortranSyntaxError

from compiler.frontend.lowering import _lower_tree
from compiler.ir import CompilationError, SourceLocation


def lower_source(source, entry, *, source_name):
    """Use the ordinary frontend without publishing a temporary source file."""
    if not isinstance(source, str) or not isinstance(source_name, str) or not source_name:
        raise CompilationError("compiler-owned source requires text and a stable source identity")
    try:
        tree = ParserFactory().create(std="f2008")(FortranStringReader(source, ignore_comments=False))
    except FortranSyntaxError as error:
        raise CompilationError("invalid compiler-owned Fortran syntax: " + str(error), SourceLocation(source_name)) from error
    return _lower_tree(tree, source_name, entry)
