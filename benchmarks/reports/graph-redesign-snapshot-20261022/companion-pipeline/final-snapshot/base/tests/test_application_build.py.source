"""Transport build authority through the explicit compiler snapshot only."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from elmm_pipeline.application_build import selection_builds, verify
from elmm_pipeline.build import build
from elmm_pipeline.local_adapter import LocalAdapterError, transform_scopes


def test_public_verifier_transport_preserves_paths_and_blocks_failure(tmp_path):
    compiler = tmp_path / "compiler with spaces"
    (compiler / "compiler/offload").mkdir(parents=True)
    (compiler / "compiler/offload/application_build.py").touch()
    transport = {"compiler_root": str(compiler), "python": sys.executable}
    calls = []

    def invoke(argv, **kwargs):
        calls.append((argv, kwargs))
        assert kwargs["cwd"] == compiler and kwargs["env"]["PYTHONPATH"] == str(compiler)
        assert kwargs["timeout"] == 240
        request = json.loads(Path(argv[argv.index("--request") + 1]).read_text())
        assert request == {"manifest": "literal $(unchanged) path"}
        Path(argv[argv.index("--output") + 1]).write_text(json.dumps({"identity": "verified"}))
        return subprocess.CompletedProcess(argv, 0, json.dumps({"status": "verified", "identity": "verified"}), "")

    with patch("elmm_pipeline.application_build.subprocess.run", side_effect=invoke):
        assert verify(transport, "preflight", {"manifest": "literal $(unchanged) path"}, tmp_path / "evidence")["identity"] == "verified"
    assert calls[0][0][:4] == [sys.executable, "-m", "compiler.offload.application_build", "preflight"]
    with patch("elmm_pipeline.application_build.subprocess.run", return_value=subprocess.CompletedProcess([], 1, "", "actual argv differs")), \
            pytest.raises(ValueError, match="binary delivery blocked"):
        verify(transport, "attest", {}, tmp_path / "evidence")


@pytest.mark.parametrize("phase", ["preflight", "attest"])
@pytest.mark.parametrize("stdout,stderr", [(b"partial output\n", b"partial error\n"), (None, "still checking\n")])
def test_public_verifier_timeout_blocks_delivery_and_preserves_evidence(tmp_path, phase, stdout, stderr):
    compiler = tmp_path / "compiler with spaces"
    (compiler / "compiler/offload").mkdir(parents=True)
    (compiler / "compiler/offload/application_build.py").touch()
    transport = {"compiler_root": str(compiler), "python": sys.executable}
    evidence = tmp_path / "evidence"

    def timeout(argv, **kwargs):
        assert kwargs["timeout"] == 240
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"], output=stdout, stderr=stderr)

    with patch("elmm_pipeline.application_build.subprocess.run", side_effect=timeout), \
            pytest.raises(ValueError, match="timed out after 240 seconds; binary delivery blocked") as caught:
        verify(transport, phase, {"literal": "request"}, evidence)
    assert isinstance(caught.value.__cause__, subprocess.TimeoutExpired)
    command = json.loads((evidence / (phase + "-command.json")).read_text())
    assert command["argv"][:4] == [sys.executable, "-m", "compiler.offload.application_build", phase]
    assert command["cwd"] == str(compiler)
    assert command["timed_out"] is True
    assert command["returncode"] is None
    assert command["timeout_seconds"] == 240
    assert json.loads((evidence / (phase + "-request.json")).read_text()) == {"literal": "request"}
    expected = "partial output\npartial error\n" if isinstance(stdout, bytes) else "still checking\n"
    assert (evidence / (phase + ".log")).read_text() == expected
    assert not (evidence / (phase + ".json")).exists()


@pytest.mark.parametrize("protocol", ["complete-original-window-cold-v4", "checked-selection-process-v1",
                                      "checked-selection-process-v2", "checked-interval-process-v3"])
@pytest.mark.parametrize("capture", [False, True])
def test_old_cost_protocols_remain_rejected(protocol, capture):
    with pytest.raises(ValueError, match="supported executed application"):
        selection_builds({"scopes": [{"exact_window": {
            "capture": capture, "selection": "native",
            "contract": {"protocol": protocol,
                         "build_contract": {"protocol": "executed-application-build-v1"}}}}]})


@pytest.mark.parametrize("capture,selection", [(True, "native"), (True, "resident"), (False, "resident")])
def test_interval_build_transport_preserves_compiler_validation_policy(capture, selection):
    # Numerical interpretation belongs to the selected compiler. Transport the
    # exact policy and evidence reference without synthesizing a pass receipt.
    record = {"capture": capture, "selection": selection,
              "contract": {"protocol": "checked-interval-process-v4",
                           "acceptance": {"criterion": "paired_non_regression"},
                           "build_contract": {"protocol": "executed-application-build-v1"},
                           "validation_policy": {"identity": "compiler-issued-policy",
                                                 "atol": 1e-6, "rtol": 1e-5,
                                                 "outputs": {"files": {"final.vtk": ["velocity"]}}}},
              "numerical_validation": {"path": "/literal evidence/numerics.json", "sha256": "a" * 64}}
    before = copy.deepcopy(record)
    assert selection_builds({"scopes": [{"exact_window": record}]}) == [before]
    assert record == before


def test_static_automatic_native_diagnostic_requires_no_selector_build():
    record = {"selection": "native", "capture": False, "reason": "unavailable calibration",
              "contract": {"protocol": "checked-interval-process-v4",
                           "build_contract": {"protocol": "executed-application-build-v1"}}}
    assert selection_builds({"scopes": [{"exact_window": record}]}) == []
    record["capture"] = True
    record["capture_variant"] = "native"
    assert selection_builds({"scopes": [{"exact_window": record}]}) == [record]


def test_scope_cost_transport_is_hashed_and_not_reinterpreted(tmp_path):
    source = tmp_path / "source.f90"
    source.write_text("subroutine step\nend\n")
    cost = tmp_path / "cost.json"
    requested = {"coverage": "checked_interval", "acceptance": {"criterion": "paired_non_regression"}, "validation_policy": {
        "driver": {"original_source": str(source), "replacement_sha256": "a" * 64},
        "outputs": {"files": {"final.vtk": ["velocity"], "average.vtk": ["velocity"]}}}}
    encoded = json.dumps(requested, indent=3).encode()
    cost.write_bytes(encoded)

    def transport(*args):
        arguments = args[-2]
        assert Path(arguments[arguments.index("--scope-cost") + 1]).read_bytes() == encoded
        raise LocalAdapterError("transport sentinel")

    with patch("elmm_pipeline.local_adapter._invoke", side_effect=transport) as invoke, \
            pytest.raises(LocalAdapterError, match="transport sentinel"):
        transform_scopes(source, tmp_path, Path(sys.executable), tmp_path / "artifacts", entry="step", capture_facts={},
                         gpu_options={"gpu_policy": "auto", "scope_cost": str(cost)})
    arguments = invoke.call_args.args[-2]
    assert arguments[arguments.index("--scope-cost") + 1] == str(cost)

    def change_request(*_args):
        cost.write_text("{}")
        return {}, []

    with patch("elmm_pipeline.local_adapter._invoke", side_effect=change_request), pytest.raises(LocalAdapterError, match="cost request changed"):
        transform_scopes(source, tmp_path, Path(sys.executable), tmp_path / "other", entry="step", capture_facts={},
                         gpu_options={"gpu_policy": "auto", "scope_cost": str(cost)})


def test_post_build_failure_prevents_returning_binary(tmp_path):
    application, fft, directory = tmp_path / "ELMM", tmp_path / "PoisFFT", tmp_path / "build"
    (application / "src").mkdir(parents=True)
    (fft / "src").mkdir(parents=True)
    (application / "src/SConstruct").write_text("t = env.Program('program', [])\n")
    (fft / "src/poisfft.f90").touch()

    def execute(argv, **_kwargs):
        (directory / "ELMM").write_text("compiled binary exists")
        return subprocess.CompletedProcess(argv, 0)

    with patch("elmm_pipeline.application_build.preflight", return_value={"macros": {}}) as before, \
            patch("elmm_pipeline.build.subprocess.run", side_effect=execute), \
            patch("elmm_pipeline.build.summarize_build", return_value={"status": "complete", "identity": "receipt"}), \
            patch("elmm_pipeline.application_build.attest", side_effect=ValueError("actual link changed")) as after, \
            pytest.raises(ValueError, match="actual link changed"):
        build(application, fft, directory, Path(sys.executable), tmp_path, 1, 4, [], application_build={})
    assert before.call_count == after.call_count == 1
    assert (directory / "ELMM").is_file()
    assert "application_build" not in json.loads((directory / "build-command.json").read_text())
