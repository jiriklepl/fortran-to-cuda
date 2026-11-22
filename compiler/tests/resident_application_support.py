"""The small application the resident build is tested on: a periodic diffusion box."""
import shutil
from pathlib import Path

from compiler.resident.project import ResidentProject, Selection

BOX = Path(__file__).resolve().parent / "apps/box"
UNITS = ("params", "fields", "boundaries", "sources", "stepper", "diagnostics", "main")


def box_project(sources=BOX, *, defines=("DOUBLE",)):
    """The whole description the box needs: its sources, four selections and a precision."""
    return ResidentProject(
        sources=tuple(Path(sources) / (name + ".f90") for name in UNITS),
        selections=(Selection("step", ("stepper::diffuse", "stepper::store"), minimum_nests=2),
                    # The copies of a periodic direction do not meet from two cells on: the compiler finds
                    # that condition and the entry checks it.
                    Selection("halo", ("boundaries::*",), minimum_nests=3),
                    # Each accumulating loop becomes three nests: partial results of a short loop, and
                    # partial results and their fold for a long one.
                    Selection("sources", ("sources::inject", "sources::sense", "sources::range"), minimum_nests=5),
                    Selection("diagnostics", ("diagnostics::summary", "diagnostics::peak"), minimum_nests=6)),
        defines=defines)


def fortran_compiler():
    return shutil.which("gfortran-15") or shutil.which("gfortran")


RELAX = Path(__file__).resolve().parent / "apps/relax/relax.f90"


def relax_project():
    """A solver written as one main program and one external procedure, both selected."""
    return ResidentProject(sources=(RELAX,), selections=(
        Selection("all", ("$program::relax", "$external::damp"), minimum_nests=12),))
