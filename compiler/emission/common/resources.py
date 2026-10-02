"""Load the support header packaged with the emitters."""

from importlib.resources import files


def read_common_header() -> str:
    """Read the shared C++/CUDA support header as a package resource."""
    return files(__package__).joinpath("templates").joinpath("common_functions.cuh").read_text(encoding="utf-8")
