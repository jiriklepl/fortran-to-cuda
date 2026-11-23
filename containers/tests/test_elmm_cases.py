"""Check case selection and the outputs every run is compared on."""
import json
from pathlib import Path
import sys

import pytest

CONTAINERS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(CONTAINERS)]
import elmm_cases as cases


def test_the_channel_case_preserves_duration_and_original_inputs(tmp_path):
    from configuration import prepare_config
    upstream = tmp_path / "elmm/examples/channel"
    upstream.mkdir(parents=True)
    (upstream / "main.conf").write_text("! upstream method\n4\n")
    (upstream / "time_stepping.conf").write_text("time_stepping(\n end_time = 50,\n)\n")
    (upstream / "profile.dat").write_bytes(b"unchanged auxiliary input\n")
    before = {path.name: path.read_bytes() for path in upstream.iterdir()}
    source = cases.select_configuration({"elmm": tmp_path / "elmm"}, "channel")
    configuration = prepare_config(source, tmp_path / "snapshot")
    assert source == upstream
    assert configuration["advection_method"] == 4
    assert configuration["expected_max_steps"] is None
    assert "max_number_of_time_steps" not in configuration["overrides"]
    assert (tmp_path / "snapshot/time_stepping.conf").read_bytes() == before["time_stepping.conf"]
    assert (tmp_path / "snapshot/profile.dat").read_bytes() == before["profile.dat"]
    assert {path.name: path.read_bytes() for path in upstream.iterdir()} == before
    capped = prepare_config(source, tmp_path / "capped", steps=50)
    assert capped["expected_max_steps"] == 50
    assert capped["overrides"]["max_number_of_time_steps"] == 50


@pytest.mark.parametrize("case", cases.CASE_NAMES)
def test_a_custom_configuration_is_used_as_given(tmp_path, case):
    source = tmp_path / "my custom & configuration"
    source.mkdir()
    (source / "main.conf").write_text("6\n")
    assert cases.select_configuration({}, case, source) == source
    assert cases.select_configuration({}, case, source / "main.conf") == source


def test_prepared_channel_simple_preset_retains_reviewed_hash_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(cases, "HERE", tmp_path)
    source = tmp_path / "cases/channel-simple"
    source.mkdir(parents=True)
    (source / "main.conf").write_text("2\n")
    (source / "origin.json").write_text(json.dumps({"prepared_files_sha256": {
        "main.conf": cases.digest(source / "main.conf")}}))
    assert cases.select_configuration({}, "channel-simple") == source
    (source / "main.conf").write_text("4\n")
    with pytest.raises(ValueError, match="reviewed channel-simple configuration changed"):
        cases.select_configuration({}, "channel-simple")


def test_reviewed_channel_simple_fixture_matches_its_recorded_hashes():
    assert cases.select_configuration({}, "channel-simple") == CONTAINERS / "cases/channel-simple"


def test_unknown_case_and_missing_configuration_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="unsupported ELMM case"):
        cases.select_configuration({}, "unknown")
    source = tmp_path / "not-a-configuration"
    source.mkdir()
    with pytest.raises(ValueError, match="main.conf"):
        cases.select_configuration({}, "channel-simple", source)


def test_averaged_fields_are_required_when_native_emits_them(tmp_path):
    (tmp_path / "out.vtk").touch()
    (tmp_path / "avg.vtk").touch()
    reference = cases.read_outputs(tmp_path, lambda path: path.name)
    assert set(reference) == {"out.vtk", "avg.vtk"}
    (tmp_path / "avg.vtk").unlink()
    observed = cases.read_outputs(tmp_path, lambda path: path.name)
    with pytest.raises(ValueError, match="output files differ"):
        cases.compare_outputs(reference, observed, lambda *a, **kw: {"passed": True})


def test_averaged_only_mismatch_fails_complete_output_validation():
    reference = {"out.vtk": 1., "avg.vtk": 2.}
    observed = {"out.vtk": 1., "avg.vtk": 3.}

    def compare(expected, actual, *, atol, rtol):
        assert (atol, rtol) == (1e-6, 1e-5)
        return {"passed": expected == actual}

    result = cases.compare_outputs(reference, observed, compare)
    assert not result["passed"]
    assert result["files"]["out.vtk"]["passed"]
    assert not result["files"]["avg.vtk"]["passed"]
