"""Shared locations and source names for the three benchmark cases."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "benchmarks"
CASES = BENCHMARKS / "cases"
GENERATED = BENCHMARKS / "generated"
RESULTS = BENCHMARKS / "results"
SOURCES = {"CDU": "cdu.f90", "CDV": "cvd.f90", "CDW": "cdw.f90"}


def variant_dir(case: str, variant: str) -> Path:
    if variant == "CUDA-pinned":
        variant = "CUDA"
    root = GENERATED if variant in {"CUDA", "CPP", "CPP-OMP"} else CASES
    return root / case / variant
