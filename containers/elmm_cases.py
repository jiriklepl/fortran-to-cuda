"""The maintained ELMM comparison cases and the outputs each run is compared on."""
from __future__ import annotations

import json
from pathlib import Path

from application_basis import digest

HERE = Path(__file__).resolve().parent
CASE_NAMES = ("channel-simple", "channel")


def select_configuration(originals, case, config=None):
    """The configuration directory of a maintained case, or of an explicit custom input."""
    if case not in CASE_NAMES:
        raise ValueError("unsupported ELMM case: " + case)
    if config is None:
        if case == "channel-simple":
            config = HERE / "cases/channel-simple"
            origin = json.loads((config / "origin.json").read_text())
            for name, expected in origin["prepared_files_sha256"].items():
                if digest(config / name) != expected:
                    raise ValueError("reviewed channel-simple configuration changed: " + name)
        else:
            config = originals["elmm"] / "examples/channel"
    source = Path(config).resolve(strict=True)
    if source.is_file():
        source = source.parent
    if not (source / "main.conf").is_file():
        raise ValueError("Configuration needs main.conf: " + str(source))
    return source


def read_outputs(directory, reader):
    """Retain final fields and averaged fields when the application emits them."""
    directory = Path(directory)
    paths = [directory / "out.vtk"]
    if (directory / "avg.vtk").is_file():
        paths.append(directory / "avg.vtk")
    return {path.name: reader(path) for path in paths}


def compare_outputs(reference, observed, compare):
    if reference.keys() != observed.keys():
        raise ValueError("Final/averaged VTK output files differ from the native reference")
    files = {name: compare(expected, observed[name], atol=1e-6, rtol=1e-5)
             for name, expected in reference.items()}
    return {"schema_version": 2, "passed": all(row["passed"] for row in files.values()),
            "atol": 1e-6, "rtol": 1e-5, "files": files,
            "note": "Final and available averaged VTK fields; float32 output, no float64 halo claim."}
