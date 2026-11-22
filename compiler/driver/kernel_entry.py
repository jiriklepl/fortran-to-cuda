"""Compile one kernel entry to its shared sources and public report."""

from pathlib import Path

from compiler.analysis import format_plan
from compiler.driver.options import CompilerOptions
from compiler.driver.pipeline import prepare_function
from compiler.emission import generate_sources, read_common_header
from compiler.frontend import lower_file
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig


def compile_kernel_entry(source_file, kernel, *, options: CompilerOptions, offload_config: OffloadConfig,
                         common_header="common_functions.cuh"):
    """Lower, schedule and emit one entry. Nothing is written."""
    function = lower_file(source_file, kernel)
    function, plan = prepare_function(function, options=options)
    sources = generate_sources(function, plan, common_header=common_header, offload_config=offload_config)
    if common_header in sources.artifacts:
        raise CompilationError("common header filename conflicts with a shared runtime/entry artifact")
    return function, plan, sources


def kernel_entry_outputs(sources, *, common_header="common_functions.cuh"):
    """The files of a compiled entry, by name."""
    outputs = {common_header: read_common_header()}
    outputs.update(sources.artifacts)
    return outputs


def kernel_entry_report(kernel, plan, sources, outputs):
    """What a consumer needs to build and call the entry."""
    return {
        "kernel": kernel,
        "artifact_schema_version": 2,
        "supported": True,
        "reason": None,
        "parallel_regions": len(plan.regions),
        "execution_plan": format_plan(plan),
        "outputs": list(outputs),
        "numerical_contract": sources.numerical_contract,
        "build_sources": [
            *[{**item, "role": "common_runtime"} for item in sources.scoped["runtime"]["sources"]],
            {"path": "shared_entry.cu", "language": "cuda", "role": "shared_entry",
             "numerical_contract": sources.numerical_contract},
            {"path": "shared_interface.f90", "language": "fortran", "role": "shared_entry"},
        ],
        "scoped": sources.scoped,
    }


def write_kernel_entry(outputs, directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    for filename, code in outputs.items():
        (directory / filename).write_text(code, encoding="utf-8")
    return directory
