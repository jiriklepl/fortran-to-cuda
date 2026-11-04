"""Owned text inputs for rendering one already checked lexical unit.

Preparation retains all source, storage and execution authority. This value is
only lowering data: ordered original edit spans, complete Fortran fragments and
detached diagnostic JSON. Rendering cannot acquire more compiler authority.
"""

from dataclasses import dataclass

from compiler.ir import CompilationError
from compiler.scopes.segments import fortran_lines


@dataclass(frozen=True, slots=True)
class PreparedUnitEmission:
    edits: tuple
    diagnostics: bytes
    finished_edits: tuple = ()

    def __post_init__(self):
        _require_owned(self)


def _require_owned(image):
    # Exact builtin shapes keep construction and consumption free from custom
    # iteration, conversion, equality and string callbacks. No owner, binding,
    # provider, proof or original parser object can cross this boundary.
    if type(image) is not PreparedUnitEmission or type(image.edits) is not tuple or not image.edits:
        raise CompilationError("prepared unit emission requires its owned nonempty edit tuple")
    if type(image.diagnostics) is not bytes:
        raise CompilationError("prepared unit diagnostics require detached encoded JSON bytes")
    for edit in image.edits:
        if (type(edit) is not tuple or len(edit) != 3
                or type(edit[0]) is not int or type(edit[1]) is not int
                or edit[0] < 1 or edit[1] < edit[0]-1
                or type(edit[2]) is not tuple or any(type(line) is not str for line in edit[2])):
            raise CompilationError("prepared unit emission requires owned original spans and text fragments")
    if type(image.finished_edits) is not tuple:
        raise CompilationError("prepared unit finished edits require an owned tuple")
    for edit in image.finished_edits:
        if (type(edit) is not tuple or len(edit) != 3
                or type(edit[0]) is not int or type(edit[1]) is not int
                or edit[0] < 1 or edit[1] < edit[0]-1 or type(edit[2]) is not str):
            raise CompilationError("prepared unit finished edits require owned original spans and exact text")


def render_unit(image):
    """Render detached text only; never inspect live preparation inputs."""
    _require_owned(image)
    return [(first, last, "\n".join(fortran_lines(lines)) + "\n")
            for first, last, lines in image.edits] + list(image.finished_edits)
