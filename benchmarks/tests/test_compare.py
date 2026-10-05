"""Require actual GPU execution, including in resident-data validation."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.harness.compare import Comparison, Variant, cuda_trace, gpu_trace


def test_gpu_trace_rejects_silent_cpu_execution():
    with pytest.raises(ValueError, match="no GPU kernel"):
        gpu_trace("device is available, but the implementation ran on CPU\n")


def test_gpu_trace_records_multiple_calls_but_single_transfer_round():
    trace = (
        "upload CUDA data variable=u bytes=100\n"
        "upload CUDA data variable=v bytes=100\n"
        "launch CUDA kernel function=cdu\n"
        "launch CUDA kernel function=cdu\n"
        "download CUDA data variable=u2 bytes=100\n"
    )
    assert gpu_trace(trace) == {"kernel_launches": 2, "upload_bytes": 200, "download_bytes": 100}


def test_local_cuda_trace_requires_kernels_and_counts_lifetime():
    with pytest.raises(ValueError, match="no GPU kernel"):
        cuda_trace("FORT_RUNTIME upload bytes=100\n")
    trace = "\n".join(
        [
            "FORT_RUNTIME alloc bytes=100",
            "FORT_RUNTIME upload bytes=100",
            "FORT_RUNTIME kernel",
            "FORT_RUNTIME kernel",
            "FORT_RUNTIME download bytes=100",
            "FORT_RUNTIME free",
        ]
    )
    assert cuda_trace(trace) == {
        "kernel_launches": 2,
        "allocations": 1,
        "frees": 1,
        "upload_bytes": 100,
        "download_bytes": 100,
    }


@pytest.mark.parametrize(
    "variant",
    [
        Variant("Loki-OpenACC", Path("kernel.f90"), "openacc"),
        Variant("Local-CUDA", Path("kernel.f90"), "cuda", Path("kernel.cu"), 1),
    ],
)
def test_compile_only_honors_requested_resident_drivers(variant):
    comparison = Comparison.__new__(Comparison)
    comparison.args = SimpleNamespace(grids=[(1, 1, 1), (5, 4, 3)], resident=True, gpu="compile")
    comparison.report = {"compilation": []}
    built = []
    comparison.save = lambda: None

    def build(case, variant, shape, *, resident):
        built.append(resident)
        return Path("/tmp/compiled-benchmark")

    comparison.build = build
    comparison.validate("CDU", [variant])
    assert built == [False, True]
    assert [record["resident"] for record in comparison.report["compilation"]] == [False, True]
    assert all(not record["executed"] for record in comparison.report["compilation"])
