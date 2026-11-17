"""Guard against false passes from malformed or numerically invalid outputs."""

import math

import pytest

from benchmarks.harness.common import compare_values, values


@pytest.mark.parametrize("output", ["1", "1 2 3", "1 nan", "1 inf", "1 -inf"])
def test_rejects_truncated_extra_and_nonfinite_values(output):
    with pytest.raises(ValueError, match="expected 2 output values|non-finite"):
        values(output, expected=2)


@pytest.mark.parametrize("actual", [[], [1.0], [1.0, 2.01], [1.0, math.nan]])
def test_comparison_cannot_silently_truncate_or_ignore_nan(actual):
    with pytest.raises(ValueError, match="equal, nonempty|exceeds|non-finite"):
        compare_values([1.0, 2.0], actual)
