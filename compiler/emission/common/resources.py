"""Assemble packaged runtime units into the existing shared-header artifact."""

from hashlib import sha256
from importlib.resources import files

from compiler.numerical_contract import numerical_build_contract, numerical_source_prologue


def read_common_header() -> str:
    """Keep a single distributable support header with separately maintained units."""
    base = files(__package__).joinpath("templates").joinpath("common_functions.cuh").read_text(encoding="utf-8")
    runtime = files("compiler.runtime")
    units = runtime.joinpath("numeric.hpp").read_text(encoding="utf-8")
    units += "\n" + runtime.joinpath("floating_environment.hpp").read_text(encoding="utf-8")
    offload = runtime.joinpath("offload.hpp").read_text(encoding="utf-8")
    units += "\n#ifdef FORT_OFFLOAD_ENABLED\n" + offload + "\n#endif\n"
    return numerical_source_prologue() + base.replace("// FORT_RUNTIME_UNITS", units)


def read_scoped_runtime() -> tuple[dict[str, str], dict]:
    """Publish the common compiled runtime independently of numerical entries."""
    runtime = files("compiler.runtime")
    outputs = {
        "scoped_runtime.h": runtime.joinpath("scoped_runtime.h").read_text(encoding="utf-8"),
        "floating_environment.hpp": runtime.joinpath("floating_environment.hpp").read_text(encoding="utf-8"),
        "scoped_entry.hpp": runtime.joinpath("scoped_entry.hpp").read_text(encoding="utf-8"),
        "section_copy.hpp": runtime.joinpath("section_copy.hpp").read_text(encoding="utf-8"),
        "scoped_cuda_resources.hpp": runtime.joinpath("scoped_cuda_resources.hpp").read_text(encoding="utf-8"),
        "scoped_graphs.hpp": runtime.joinpath("scoped_graphs.hpp").read_text(encoding="utf-8"),
        "scoped_regions.hpp": runtime.joinpath("scoped_regions.hpp").read_text(encoding="utf-8"),
        "scoped_runtime.cu": numerical_source_prologue() + runtime.joinpath("scoped_runtime.cu").read_text(encoding="utf-8"),
        "fort_scoped_memory.f90": runtime.joinpath("scoped_memory.f90").read_text(encoding="utf-8"),
    }
    hashes = {name: sha256(content.encode()).hexdigest() for name, content in outputs.items()}
    identity = sha256("\n".join(f"{name}:{digest}" for name, digest in sorted(hashes.items())).encode()).hexdigest()
    manifest = {
        "schema_version": 1,
        "abi_version": 1,
        "runtime_id": identity,
        "link_once": True,
        "numerical_contract": numerical_build_contract(),
        "headers": sorted(name for name in outputs if name.endswith((".h", ".hpp", ".cuh"))),
        "sources": [
            {"path": "scoped_runtime.cu", "language": "cuda", "standard": "c++17", "host_openmp": True,
             "numerical_contract": numerical_build_contract()},
            {"path": "fort_scoped_memory.f90", "language": "fortran", "module": "fort_scoped_memory"},
        ],
        "source_sha256": hashes,
        "rectangle_limit": 128,
        "intersection_limit": 8192,
    }
    return outputs, manifest
