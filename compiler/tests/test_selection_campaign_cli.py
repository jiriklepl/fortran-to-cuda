"""Public verifier integration using explicit synthetic build receipt fixtures.

The existing fixtures contain real hashed numerical payloads; their build JSON
does not claim real compiler execution or GPU performance.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from compiler.offload.selection_cost import evaluate_capture_evidence, verify_capture_builds
from compiler.tests.test_application_build import prepared_receipts
from compiler.tests.test_selection_validation import _campaign, reference


@pytest.fixture
def inputs(tmp_path):
    value = _campaign(prepared_receipts(tmp_path))
    root = tmp_path / "collector"
    root.mkdir()
    for name in ("contract", "evidence"):
        (root / (name + ".json")).write_text(json.dumps(value[name]))
    campaign = value["evidence"]["campaigns"][0]
    request = {"contract": reference(root / "contract.json"),
               "builds": {row["mode"]: row["build"] for row in campaign["observations"]},
               "diagnostics": {mode: variant["diagnostic_build"] for mode, variant in value["receipt"]["variants"].items()}}
    return value, root, request


def test_collector_build_preflight_returns_only_verified_original_binary_paths(inputs):
    value, _root, request = inputs
    response = verify_capture_builds(request)
    assert response["status"] == "verified"
    assert response["placement_authority"] is False
    for mode in ("native", "resident"):
        application = json.loads(Path(request["builds"][mode]["attestation"]).read_text())
        diagnostic = json.loads(Path(value["receipt"]["variants"][mode]["diagnostic_build"]["build"]["path"]).read_text())
        assert response["variants"][mode]["application"]["binary"] == application["binary"]
        assert response["variants"][mode]["diagnostic"]["binary"] == diagnostic["binary"]
    Path(response["variants"]["resident"]["application"]["binary"]).write_text("changed")
    with pytest.raises(ValueError, match="changed|differs"):
        verify_capture_builds(request)


def test_collector_canonicalizes_cpu_options_without_mutating_contract_or_raw_evidence(inputs):
    value, root, request = inputs
    raw = value["evidence"]
    for row in raw["campaigns"][0]["observations"]:
        row["observed_cpu_identity"]["fortran_options"] = "-O3 -fopenmp -ffp-contract=off -I /different/build"
    path = root / "raw.json"
    path.write_text(json.dumps(raw))
    before = path.read_bytes()
    response = evaluate_capture_evidence({"contract": request["contract"], "evidence": reference(path)})
    assert response["status"] == "verified", response["decision"]
    assert response["decision"]["selection"] == "resident"
    assert response["placement_authority"] is False
    assert path.read_bytes() == before
    assert response["evidence"]["campaigns"][0]["contract"] == value["contract"]
    assert response["evidence"]["campaigns"][0]["identity"] == raw["campaigns"][0]["identity"]


@pytest.mark.parametrize("payload", ["halos", "final", "average"])
def test_public_cli_verifies_actual_payloads_and_rejects_mutation(inputs, payload):
    value, root, request = inputs
    source = Path(__file__).resolve().parents[2]
    environment = {**os.environ, "PYTHONPATH": str(source)}
    for phase, requested in (("builds", request), ("evaluate", {"contract": request["contract"],
                                                          "evidence": reference(root / "evidence.json")})):
        path, output = root / (phase + "-request.json"), root / (phase + "-result.json")
        path.write_text(json.dumps(requested))
        argv = [sys.executable, "-m", "compiler.offload.selection_cost", phase,
                "--request", str(path), "--output", str(output)]
        result = subprocess.run(argv, cwd=source, env=environment, capture_output=True, text=True, timeout=180, check=False)
        assert result.returncode == 0, result.stderr
        assert json.loads(output.read_text())["status"] == "verified"
    field = (value["receipt"]["variants"]["resident"]["fields"]["state"]["payload"]["path"] if payload == "halos" else
             value["receipt"]["pairs"][0]["resident"]["outputs"]["output/" + payload + ".vtk"]["path"])
    Path(field).write_bytes(b"changed halo payload")
    result = subprocess.run(argv, cwd=source, env=environment, capture_output=True, text=True, timeout=180, check=False)
    assert result.returncode == 1
    assert json.loads(output.read_text())["status"] == "rejected"
