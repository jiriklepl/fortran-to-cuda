"""Assemble packaged runtime units into the existing shared-header artifact."""

from importlib.resources import files


def read_common_header() -> str:
    """Keep a single distributable support header with separately maintained units."""
    base = files(__package__).joinpath("templates").joinpath("common_functions.cuh").read_text(encoding="utf-8")
    runtime = files("compiler.runtime")
    units = "\n".join(
        runtime.joinpath(name).read_text(encoding="utf-8") for name in ("numeric.hpp", "timing.hpp", "storage.hpp")
    )
    return base.replace("// FORT_RUNTIME_UNITS", units)
