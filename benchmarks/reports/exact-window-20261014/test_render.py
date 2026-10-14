"""Test reporting rules without creating or presenting measurement evidence."""
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("exact_fixture_report", Path(__file__).with_name("render.py"))
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def test_conservative_advantage_is_strict_and_includes_holdouts():
    assert report.conservative_advantage({"native": [10] * 9, "automatic": [9] * 9}, "automatic")
    assert not report.conservative_advantage({"native": [10] * 9, "automatic": [9] * 8 + [9.5]}, "automatic")
    assert not report.conservative_advantage({"native": [10] * 8 + [9], "automatic": [9] * 9}, "automatic")


def test_summary_median_uses_seven_samples_and_range_includes_holdouts():
    summary = report.summarize({"native": [1, 2, 3, 4, 5, 6, 7, .5, 20]})["native"]
    assert summary["median"] == 4 and summary["min"] == .5 and summary["max"] == 20
    assert summary["holdout"] == [.5, 20]


def test_location_option_normalization_preserves_semantic_flags():
    assert report.semantic_fortran_options("-O3 -I '/path with spaces' -J/tmp/modules -ffp-contract=off -o object.o") == (
        "-O3\x1f-ffp-contract=off")
    assert report.semantic_fortran_options("-O3 -foffload-options=default -openmp -opt=1") == (
        "-O3\x1f-foffload-options=default\x1f-openmp\x1f-opt=1")
    with pytest.raises(ValueError, match="incomplete"):
        report.semantic_fortran_options("-O3 -I")


def test_incomplete_or_misordered_observations_are_not_plot_data():
    with pytest.raises(ValueError, match="seven measured"):
        report.samples([], "automatic")
    with pytest.raises(ValueError, match="rotated observation order"):
        report.samples([{}] * 18, "automatic")
