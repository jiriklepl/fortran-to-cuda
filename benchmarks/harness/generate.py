#!/usr/bin/env python3
"""
Runs the fort_to_cuda compiler for every benchmark case and distributes the
generated files under benchmarks/generated/<case>/<variant>/.

For each case:
  - CUDA/          ← generated_code.cu  +  <case_src>.f90 (Fortran interface)
  - CPP-OMP/       ← generated_cpp_impl.cpp (with #pragma omp)  +  <case_src>.f90
  - CPP/           ← generated_cpp_impl.cpp (#pragma omp lines stripped)  +  <case_src>.f90

The Fortran/  and Fortran-OMP/  variants are untouched (no generated code).

Usage:
    python -m benchmarks.harness.generate [CASE ...]

    If no CASEs are given, all cases are processed.
"""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from .paths import ROOT, SOURCES, variant_dir


def _strip_omp(text: str) -> str:
    """Remove lines that contain a #pragma omp directive."""
    return "\n".join(line for line in text.splitlines() if not re.search(r"^\s*#\s*pragma\s+omp\b", line)) + "\n"


def generate_case(case: str, verbose: bool) -> bool:
    fortran_src = variant_dir(case, "Fortran") / SOURCES[case]

    if not fortran_src.exists():
        print(f"[{case}] ERROR: Fortran source not found: {fortran_src}", file=sys.stderr)
        return False

    with tempfile.TemporaryDirectory(prefix=f"fort2cuda_{case}_") as tmp:
        tmp_dir = Path(tmp)

        # ── run compiler ──────────────────────────────────────────────────────
        cmd = [
            sys.executable,
            "-m",
            "compiler",
            "--input",
            str(fortran_src),
            "--kernel",
            case,
            "--output-dir",
            str(tmp_dir),
            "--no-common-header",
        ]
        if verbose:
            cmd.append("--verbose")

        print(f"[{case}] Running compiler …")
        result = subprocess.run(cmd, cwd=ROOT, capture_output=not verbose)
        if result.returncode != 0:
            print(f"[{case}] ERROR: compiler failed", file=sys.stderr)
            if not verbose:
                sys.stderr.buffer.write(result.stderr)
            return False

        cu_file = tmp_dir / "generated_code.cu"
        cpp_file = tmp_dir / "generated_cpp_impl.cpp"
        iface_file = tmp_dir / "generated_interface.f90"

        for f in (cu_file, cpp_file, iface_file):
            if not f.exists():
                print(f"[{case}] ERROR: expected output not found: {f.name}", file=sys.stderr)
                return False

        cpp_with_omp = cpp_file.read_text()
        cpp_without_omp = _strip_omp(cpp_with_omp)
        iface_text = iface_file.read_text()

        for variant, filename, source in (
            ("CUDA", "generated_code.cu", cu_file.read_text()),
            ("CPP-OMP", "generated_cpp_impl.cpp", cpp_with_omp),
            ("CPP", "generated_cpp_impl.cpp", cpp_without_omp),
        ):
            destination = variant_dir(case, variant)
            destination.mkdir(parents=True, exist_ok=True)
            (destination / filename).write_text(source)
            (destination / SOURCES[case]).write_text(iface_text)
            print(f"[{case}] {variant:7} → {filename}, {SOURCES[case]}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Generate CUDA / C++ / C++-OMP sources for all benchmark cases.")
    parser.add_argument(
        "cases",
        nargs="*",
        metavar="CASE",
        help=f"Cases to generate (default: all). Known cases: {', '.join(SOURCES)}",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Pass --verbose to the compiler and show its output.",
    )
    args = parser.parse_args()

    selected = args.cases if args.cases else list(SOURCES)
    unknown = [c for c in selected if c not in SOURCES]
    if unknown:
        parser.error(f"Unknown case(s): {', '.join(unknown)}. Known: {', '.join(SOURCES)}")

    failures = []
    for case in selected:
        ok = generate_case(case, args.verbose)
        if not ok:
            failures.append(case)

    if failures:
        print(f"\nFailed: {', '.join(failures)}", file=sys.stderr)
        sys.exit(1)
    else:
        print("\nAll done.")


if __name__ == "__main__":
    main()
