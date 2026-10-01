#!/usr/bin/env python3
"""
Correctness check: build every variant on a small grid, run once with
a deterministic input, and compare each variant's output against the
Fortran serial reference.

Usage:
    python run_tests.py [CASE ...]   # default: all cases
    make test CASE=CDU               # via Makefile
"""

import os
import sys
import subprocess

BENCHMARKS_DIR = os.path.dirname(os.path.abspath(__file__))

CASES     = ['CDU', 'CDW', 'CDV']
VARIANTS  = ['Fortran', 'Fortran-OMP', 'Fortran-ACC', 'CPP', 'CPP-OMP', 'CUDA', 'CUDA-pinned']
REFERENCE = 'Fortran'

# Small grid: fast to build and run, small enough to compare element-by-element
TEST_NX      = 16
TEST_NY      = 16
TEST_NZ      = 16
TEST_NITER   = 1
TEST_NWARMUP = 0

# Absolute tolerance — serial variants should be bit-exact with the reference;
# parallel/CUDA variants may reorder FP operations, so allow a small slack.
ATOL = 1e-10


# ── helpers ───────────────────────────────────────────────────────────────────

def _binary_path(case: str, variant: str) -> str:
    build_id = (f'{case}_{variant}'
                f'_NX{TEST_NX}_NY{TEST_NY}_NZ{TEST_NZ}'
                f'_NITER{TEST_NITER}_NWARMUP{TEST_NWARMUP}')
    return os.path.join(BENCHMARKS_DIR, 'bin', build_id, 'benchmark')


def _build(case: str, variant: str) -> str:
    """Build the test binary (using test_main.f90) and return its path."""
    command = [
        'make',
        f'CASE={case}', f'VARIANT={variant}',
        f'NX={TEST_NX}', f'NY={TEST_NY}', f'NZ={TEST_NZ}',
        f'NITER={TEST_NITER}', f'NWARMUP={TEST_NWARMUP}',
        'MAIN_SRC=test_main.f90'
    ]

    # Inject the macro flag if building the pinned variant
    if variant == 'CUDA-pinned':
        command.append('USE_PINNED_MEMORY=1')

    result = subprocess.run(
        command,
        cwd=BENCHMARKS_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return _binary_path(case, variant)


def _run(binary: str) -> list[float]:
    """Run the test binary and parse its output as a list of floats."""
    result = subprocess.run([binary], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return [float(x) for x in result.stdout.split()]


def _max_abs_diff(ref: list[float], other: list[float]) -> float:
    return max(abs(r - o) for r, o in zip(ref, other))


# ── per-case logic ────────────────────────────────────────────────────────────

def check_case(case: str) -> bool:
    print(f'\n=== {case} ===')
    ref_arr = None
    all_pass = True

    for variant in VARIANTS:
        # build
        try:
            binary = _build(case, variant)
        except RuntimeError as e:
            first_line = e.args[0].splitlines()[0] if e.args[0] else '?'
            print(f'  [SKIP] {variant:<14}  build failed: {first_line}')
            continue

        # run
        try:
            arr = _run(binary)
        except RuntimeError as e:
            print(f'  [SKIP] {variant:<14}  run failed: {e.args[0].splitlines()[0]}')
            continue

        expected = TEST_NX * TEST_NY * TEST_NZ
        if len(arr) != expected:
            print(f'  [FAIL] {variant:<14}  expected {expected} values, got {len(arr)}')
            all_pass = False
            continue

        if variant == REFERENCE:
            ref_arr = arr
            print(f'  [REF ] {variant:<14}  {len(arr)} values')
            continue

        if ref_arr is None:
            print(f'  [SKIP] {variant:<14}  reference not yet available')
            continue

        max_abs = _max_abs_diff(ref_arr, arr)
        ok = max_abs <= ATOL
        all_pass = all_pass and ok
        status = 'PASS' if ok else 'FAIL'
        print(f'  [{status}] {variant:<14}  max_abs_diff={max_abs:.3e}  (tol={ATOL:.0e})')

    return all_pass


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    cases = sys.argv[1:] if len(sys.argv) > 1 else CASES
    unknown = [c for c in cases if c not in CASES]
    if unknown:
        sys.exit(f'Unknown case(s): {", ".join(unknown)}. Valid: {", ".join(CASES)}')

    results = {c: check_case(c) for c in cases}
    print()

    if all(results.values()):
        print('All tests PASSED.')
    else:
        failed = [c for c, ok in results.items() if not ok]
        print(f'FAILED: {", ".join(failed)}')
        sys.exit(1)


if __name__ == '__main__':
    main()
