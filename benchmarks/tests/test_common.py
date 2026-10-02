"""Guard against false passes from malformed or numerically invalid outputs."""

import argparse
import math

import pytest

from benchmarks.harness.common import compare_values, grid, timing_values, values


def test_rejects_grids_that_overflow_array_indices():
    with pytest.raises(argparse.ArgumentTypeError, match="signed 32-bit"):
        grid("2048x2048x2048")


@pytest.mark.parametrize("output", ["1", "1 2 3", "1 nan", "1 inf", "1 -inf"])
def test_rejects_truncated_extra_and_nonfinite_values(output):
    with pytest.raises(ValueError, match="expected 2 output values|non-finite"):
        values(output, expected=2)


@pytest.mark.parametrize("actual", [[], [1.0], [1.0, 2.01], [1.0, math.nan]])
def test_comparison_cannot_silently_truncate_or_ignore_nan(actual):
    with pytest.raises(ValueError, match="equal, nonempty|exceeds|non-finite"):
        compare_values([1.0, 2.0], actual)


@pytest.mark.parametrize(
    "output",
    ["", "total_ms 1\n", "total_ms nan\nchecksum 1", "total_ms 0\nchecksum 1", "total_ms 1\nchecksum 1\nchecksum 2"],
)
def test_invalid_timings_are_failures(output):
    with pytest.raises(ValueError, match="missing timing|invalid timing|duplicate timing"):
        timing_values(output)


def test_timing_parser_allows_runtime_diagnostics_without_changing_measurement():
    assert timing_values("runtime diagnostic\ntotal_ms 1.25E+1\nchecksum 2.5E+2\n") == (12.5, 250.0)
