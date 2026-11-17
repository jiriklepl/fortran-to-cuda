"""Shared locations and source names for the three benchmark cases."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "benchmarks"
CASES = BENCHMARKS / "cases"
GENERATED = BENCHMARKS / "generated"
RESULTS = BENCHMARKS / "results"
SOURCES = {"CDU": "cdu.f90", "CDV": "cvd.f90", "CDW": "cdw.f90"}
