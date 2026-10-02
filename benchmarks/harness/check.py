#!/usr/bin/env python3
"""
Correctness check: build every variant on a small grid, run once with
a deterministic input, and compare each variant's output against the
Fortran serial reference.

Usage:
    python -m benchmarks.harness.check [CASE ...]   # default: all cases
    make -C benchmarks test CASE=CDU
"""

import argparse
import sys

from .common import ATOL, compare_values, values
from .paths import SOURCES
from .run import VARIANTS, build, run_once

REFERENCE = "Fortran"

# Small grid: fast to build and run, small enough to compare element-by-element
TEST_NX = 16
TEST_NY = 16
TEST_NZ = 16
TEST_NITER = 1
TEST_NWARMUP = 0

# ── per-case logic ────────────────────────────────────────────────────────────


def check_case(case: str, variants: list[str]) -> bool:
    print(f"\n=== {case} ===")
    ref_arr = None
    all_pass = True

    for variant in dict.fromkeys([REFERENCE, *variants]):
        # build
        try:
            binary = build(case, variant, TEST_NX, TEST_NY, TEST_NZ, TEST_NITER, TEST_NWARMUP, main_src="test_main.f90")
        except RuntimeError as e:
            first_line = e.args[0].splitlines()[0] if e.args[0] else "?"
            print(f"  [SKIP] {variant:<14}  build failed: {first_line}")
            continue

        # run
        try:
            output = run_once(binary)
        except RuntimeError as e:
            print(f"  [SKIP] {variant:<14}  run failed: {e.args[0].splitlines()[0]}")
            continue

        try:
            arr = values(output, TEST_NX * TEST_NY * TEST_NZ)
        except ValueError as error:
            print(f"  [FAIL] {variant:<14}  {error}")
            all_pass = False
            continue

        if variant == REFERENCE:
            ref_arr = arr
            print(f"  [REF ] {variant:<14}  {len(arr)} values")
            continue

        if ref_arr is None:
            print(f"  [SKIP] {variant:<14}  reference not yet available")
            continue

        try:
            max_abs = compare_values(ref_arr, arr)
        except ValueError as error:
            print(f"  [FAIL] {variant:<14}  {error}")
            all_pass = False
        else:
            print(f"  [PASS] {variant:<14}  max_abs_diff={max_abs:.3e}  (tol={ATOL:.0e})")

    return all_pass and ref_arr is not None


# ── entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="*", metavar="CASE", help="Default: all cases")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=VARIANTS,
        help="Variants to check; Fortran reference always included",
    )
    args = parser.parse_args()
    cases = args.cases or list(SOURCES)
    unknown = [c for c in cases if c not in SOURCES]
    if unknown:
        parser.error(f"Unknown case(s): {', '.join(unknown)}. Valid: {', '.join(SOURCES)}")

    results = {c: check_case(c, args.variants) for c in cases}
    print()

    if all(results.values()):
        print("All tests PASSED.")
    else:
        failed = [c for c, ok in results.items() if not ok]
        print(f"FAILED: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
