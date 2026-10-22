"""Actual-device CPU-hint reuse and later profitable GPU geometry.

The inherited fixture uses deliberately synthetic, hardware-bound costs. This
checks control/coherence and changed data, never measured application speedup.
"""

import json
import os

import pytest

from compiler.tests.test_graph_native_choice_cuda import native_choice_binaries as native_choice_fixture
from compiler.tests.test_source_scopes import run

native_choice_binaries = native_choice_fixture


@pytest.mark.cuda
@pytest.mark.parametrize("style", ["combined", "joined"])
@pytest.mark.parametrize("policy", ["cpu", "mixed"])
def test_native_owner_hint_reuses_original_data_and_preserves_later_gpu(native_choice_binaries, style, policy):
    case, native, candidate, _manifest = native_choice_binaries[style, policy]
    env = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "FORT_RUNTIME_TRACE": "1"}
    run([str(native)], cwd=case, env=env)
    expected = (case / "fields.bin").read_bytes()
    result = run([str(candidate)], cwd=case, env=env)
    (case / "native-owner-hint.log").write_text(result.stderr)
    assert (case / "fields.bin").read_bytes() == expected
    assert "array temporary" not in result.stderr.lower()
    events = [json.loads(line.removeprefix("FORT_SCOPED evidence "))
              for line in result.stderr.splitlines() if line.startswith("FORT_SCOPED evidence ")]
    hints = [json.loads(line.removeprefix("FORT_SCOPED native_hint "))
             for line in result.stderr.splitlines() if line.startswith("FORT_SCOPED native_hint ")]
    hits = [event for event in hints if event["phase"] == "lookup" and event["hit"]]
    assert hits
    assert any(event["phase"] == "finish" and event["learned"] for event in hints)
    assert all(event["reason"] == "conservative_native_owner_hint" for event in hints)
    assert all(event["estimates_available"] is False for event in hints)
    assert all(not event["observation_eligible"] for event in hits)
    assert all("context" not in event and "plan" not in event for event in hits)
    decisions = [event for event in events if event["event"] == "decision"]
    assert decisions  # The first ordinary invocation always executes validation.
    assert all(event["available"] == 1 for event in decisions)
    if policy == "cpu":
        assert not any(event["gpu_units"] for event in decisions)
        assert not any(line.startswith("FORT_SCOPED " + operation + " ")
                       for line in result.stderr.splitlines()
                       for operation in ("initialize", "allocate", "upload", "download", "graph_submit", "launch"))
    else:
        # Empty/tiny original invocations may learn a CPU hint. The later
        # different array geometry cannot reuse it and still reaches CUDA.
        assert any(event["gpu_units"] for event in decisions)
        assert "FORT_SCOPED upload " in result.stderr
        assert "FORT_SCOPED download " in result.stderr
        assert "FORT_SCOPED launch " in result.stderr or "FORT_SCOPED graph_submit " in result.stderr
