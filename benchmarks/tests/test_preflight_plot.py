import json
from copy import deepcopy

import pytest

from benchmarks.harness.plot_preflight import (
    PHASES,
    checked_document,
    phase_state,
    render,
)


def document():
    return {
        "schema_version": 1,
        "execution_authorized": False,
        "snapshot_identity": "fixture",
        "truncated": True,
        "operations": [
            {
                "ordinal": 1,
                "procedure": "module::advance",
                "kind": "loop",
                "span": [4, 8],
                "guard": ["n > 0"],
                "resources": [{"resource": "argument::a"}],
                **{phase: {"status": "available", "reasons": []} for phase in PHASES},
                "cost": {"status": "unavailable", "reasons": ["calibration missing"]},
                "ownership": {"status": "not_requested", "reasons": []},
                "transfers": {
                    "status": "unknown",
                    "upload_bytes": None,
                    "download_bytes": None,
                },
            }
        ],
    }


def test_unknown_is_never_colored_as_admitted():
    row = document()["operations"][0]
    assert phase_state(row, "ownership") == "unknown"
    assert phase_state(row, "cost") == "unavailable"


@pytest.mark.parametrize("defect", ["authority", "bytes", "duplicate", "phase"])
def test_rejects_misleading_evidence(tmp_path, defect):
    value = document()
    if defect == "authority":
        value["execution_authorized"] = True
    elif defect == "bytes":
        value["operations"][0]["transfers"]["upload_bytes"] = 0
    elif defect == "duplicate":
        value["operations"].append(deepcopy(value["operations"][0]))
    else:
        del value["operations"][0]["ownership"]
    path = tmp_path / "input.json"
    path.write_text(json.dumps(value))
    with pytest.raises((ValueError, TypeError)):
        checked_document(path)


def test_renders_immutable_evidence_with_unknowns_and_escaped_text(tmp_path):
    value = document()
    value["operations"][0]["cost"]["reasons"] = ["<script>no calibration</script>"]
    source, output = tmp_path / "input.json", tmp_path / "figures"
    source.write_text(json.dumps(value))
    receipt = render(source, output)
    report = (output / "index.html").read_text()
    assert "inventory is incomplete" in report
    assert "&lt;script&gt;" in report and "<script>" not in report
    assert receipt["execution_authorized"] is False
    assert set(receipt["artifacts_sha256"]) == {
        "scope-01-01.svg",
        "scope-01-01.png",
        "index.html",
    }
    with pytest.raises(ValueError, match="new output"):
        render(source, output)
