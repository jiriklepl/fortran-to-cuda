"""Assemble packaged runtime units into the existing shared-header artifact."""

from importlib.resources import files


def read_common_header() -> str:
    """Keep a single distributable support header with separately maintained units."""
    base = files(__package__).joinpath("templates").joinpath("common_functions.cuh").read_text(encoding="utf-8")
    runtime = files("compiler.runtime")
    storage = runtime.joinpath("storage.hpp").read_text(encoding="utf-8")
    allocation = runtime.joinpath("allocation.hpp").read_text(encoding="utf-8")
    storage = storage.replace('#include "allocation.hpp"', allocation)
    units = "\n".join(
        [runtime.joinpath(name).read_text(encoding="utf-8") for name in ("numeric.hpp", "timing.hpp")] + [storage]
    )
    experimental = [runtime.joinpath(name).read_text(encoding="utf-8")
                    for name in ("offload.hpp", "hybrid.hpp") if runtime.joinpath(name).is_file()]
    units += "\n#ifdef FORT_OFFLOAD_ENABLED\n" + "\n".join(experimental) + "\n#endif\n"
    return base.replace("// FORT_RUNTIME_UNITS", units)
