"""The small application the resident build is tested on: a periodic diffusion box."""
import shutil
from pathlib import Path

from compiler.resident.project import ResidentProject, Selection

BOX = Path(__file__).resolve().parent / "apps/box"
UNITS = ("params", "fields", "boundaries", "sources", "stepper", "diagnostics", "main")
MINIMUM_EXTENT = "nx >= 2 .and. ny >= 2 .and. nz >= 2"


def box_project(sources=BOX, *, defines=("DOUBLE",)):
    """The whole description the box needs: its sources, three selections and a precision."""
    return ResidentProject(
        sources=tuple(Path(sources) / (name + ".f90") for name in UNITS),
        selections=(Selection("step", ("stepper::diffuse", "stepper::store"), minimum_nests=2),
                    Selection("halo", ("boundaries::*",), guard=MINIMUM_EXTENT, minimum_nests=3),
                    Selection("sources", ("sources::inject", "sources::sense"), minimum_nests=2)),
        defines=defines)


def fortran_compiler():
    return shutil.which("gfortran-15") or shutil.which("gfortran")
