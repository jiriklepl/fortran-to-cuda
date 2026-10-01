#!/usr/bin/env python3
"""
generate_all_sources.py
-----------------------
Runs the fort_to_cuda compiler for every benchmark case and distributes the
generated files to the correct variant directories.

For each case:
  - CUDA/          ← generated_code.cu  +  <case_src>.f90 (Fortran interface)
  - CPP-OMP/       ← generated_cpp_impl.cpp (with #pragma omp)  +  <case_src>.f90
  - CPP/           ← generated_cpp_impl.cpp (#pragma omp lines stripped)  +  <case_src>.f90

The Fortran/  and Fortran-OMP/  variants are untouched (no generated code).

Usage:
    python benchmarks/generate_all_sources.py [CASE ...]

    If no CASEs are given, all cases are processed.
"""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

BENCHMARKS_DIR = Path(__file__).resolve().parent
REPO_ROOT      = BENCHMARKS_DIR.parent

# ── case definitions ──────────────────────────────────────────────────────────
# key          : folder name under benchmarks/
# fortran_src  : source file under <case>/Fortran/
# kernel       : --kernel argument passed to the compiler
# interface_dst: filename used for the Fortran interface in each variant dir
CASES = {
    "CDU": dict(fortran_src="cdu.f90", kernel="CDU", interface_dst="cdu.f90"),
    "CDW": dict(fortran_src="cdw.f90", kernel="CDW", interface_dst="cdw.f90"),
    "CDV": dict(fortran_src="cvd.f90", kernel="CDV", interface_dst="cvd.f90"),
}


def _strip_omp(text: str) -> str:
    """Remove lines that contain a #pragma omp directive."""
    return "\n".join(
        line for line in text.splitlines()
        if not re.search(r"^\s*#\s*pragma\s+omp\b", line)
    ) + "\n"


def generate_case(case: str, cfg: dict, verbose: bool) -> bool:
    case_dir    = BENCHMARKS_DIR / case
    fortran_src = case_dir / "Fortran" / cfg["fortran_src"]

    if not fortran_src.exists():
        print(f"[{case}] ERROR: Fortran source not found: {fortran_src}", file=sys.stderr)
        return False

    with tempfile.TemporaryDirectory(prefix=f"fort2cuda_{case}_") as tmp:
        tmp_dir = Path(tmp)

        # ── run compiler ──────────────────────────────────────────────────────
        cmd = [
            sys.executable, "-m", "compiler",
            "--input",  str(fortran_src),
            "--kernel", cfg["kernel"],
            "--output-dir", str(tmp_dir),
            "--no-common-header",
        ]
        if verbose:
            cmd.append("--verbose")

        print(f"[{case}] Running compiler …")
        result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=not verbose)
        if result.returncode != 0:
            print(f"[{case}] ERROR: compiler failed", file=sys.stderr)
            if not verbose:
                sys.stderr.buffer.write(result.stderr)
            return False

        cu_file      = tmp_dir / "generated_code.cu"
        cpp_file     = tmp_dir / "generated_cpp_impl.cpp"
        iface_file   = tmp_dir / "generated_interface.f90"

        for f in (cu_file, cpp_file, iface_file):
            if not f.exists():
                print(f"[{case}] ERROR: expected output not found: {f.name}", file=sys.stderr)
                return False

        cpp_with_omp    = cpp_file.read_text()
        cpp_without_omp = _strip_omp(cpp_with_omp)
        iface_text      = iface_file.read_text()

        dst = cfg["interface_dst"]

        # ── CUDA variant ─────────────────────────────────────────────────────
        cuda_dir = case_dir / "CUDA"
        (cuda_dir / "generated_code.cu").write_text(cu_file.read_text())
        (cuda_dir / dst).write_text(iface_text)
        print(f"[{case}] CUDA    → generated_code.cu, {dst}")

        # ── CPP-OMP variant ──────────────────────────────────────────────────
        cpp_omp_dir = case_dir / "CPP-OMP"
        (cpp_omp_dir / "generated_cpp_impl.cpp").write_text(cpp_with_omp)
        (cpp_omp_dir / dst).write_text(iface_text)
        print(f"[{case}] CPP-OMP → generated_cpp_impl.cpp (#pragma omp kept), {dst}")

        # ── CPP variant (no OMP) ─────────────────────────────────────────────
        cpp_dir = case_dir / "CPP"
        (cpp_dir / "generated_cpp_impl.cpp").write_text(cpp_without_omp)
        (cpp_dir / dst).write_text(iface_text)
        print(f"[{case}] CPP     → generated_cpp_impl.cpp (#pragma omp stripped), {dst}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate CUDA / C++ / C++-OMP sources for all benchmark cases."
    )
    parser.add_argument(
        "cases",
        nargs="*",
        metavar="CASE",
        help=f"Cases to generate (default: all). Known cases: {', '.join(CASES)}",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Pass --verbose to the compiler and show its output.",
    )
    args = parser.parse_args()

    selected = args.cases if args.cases else list(CASES)
    unknown  = [c for c in selected if c not in CASES]
    if unknown:
        parser.error(f"Unknown case(s): {', '.join(unknown)}. Known: {', '.join(CASES)}")

    failures = []
    for case in selected:
        ok = generate_case(case, CASES[case], args.verbose)
        if not ok:
            failures.append(case)

    if failures:
        print(f"\nFailed: {', '.join(failures)}", file=sys.stderr)
        sys.exit(1)
    else:
        print("\nAll done.")


if __name__ == "__main__":
    main()
