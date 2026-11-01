"""Selected physical prices stay modeled even before failed or absent execution."""
# ruff: noqa: F811 - imported maintained pytest fixture.
import json
import os
import subprocess

import pytest
from elmm_pipeline.gpu_validation import gpu_activity, scoped_modeled_evidence

from compiler.tests.test_copy_startup import startup_runtime_program  # noqa: F401


def pricing_lines(stderr):
    return [line for line in stderr.splitlines() if '"event":"copy_pricing"' in line]


@pytest.mark.parametrize("mode", ["sequence", "seed_poison", "allocation_miss"])
def test_real_reference_prices_do_not_establish_cuda_execution(startup_runtime_program, mode, tmp_path):
    result = subprocess.run([str(startup_runtime_program), mode], text=True, capture_output=True,
                            timeout=15, env={**os.environ, "FORT_RUNTIME_TRACE": "1"})
    (tmp_path / "stdout.txt").write_text(result.stdout)
    (tmp_path / "stderr.txt").write_text(result.stderr)
    assert result.returncode == 0, result.stderr
    lines = pricing_lines(result.stderr)
    rows = scoped_modeled_evidence("\n".join(lines))
    assert [row["decision"] for row in rows] == (["credit_funded_cold", "reuse"] if mode == "sequence"
                                                 else ["credit_funded_cold"])
    first = rows[0]
    assert first["direct_seconds"] == 10
    assert first["cold_seconds"] == first["selected_seconds"] == 9
    assert first["reuse_seconds"] == 1
    assert first["prior_credit_seconds"] == first["startup_premium_seconds"] == 8
    assert first["confirms_execution"] is False
    assert first["evidence"] == "MODELED"
    activity = gpu_activity("\n".join(lines), "local", [])
    assert not activity["verified"]
    assert activity["kernel_launches"] == 0
    assert activity["upload_bytes"] == activity["download_bytes"] == activity["allocations"] == 0
    assert activity.get("copy_dma_calls", 0) == activity.get("copy_kernel_launches", 0) == 0
    assert activity["scoped_runtime"]["source_provenance"]["totals"]["events"] == 0
    completed = [json.loads(line.split("evidence ", 1)[1]) for line in result.stderr.splitlines()
                 if '"event":"compact_transfer"' in line]
    assert len(completed) == (2 if mode == "sequence" else 1 if mode == "allocation_miss" else 0)


def test_disabled_trace_serializes_no_startup_prices(startup_runtime_program):
    env = dict(os.environ)
    env.pop("FORT_RUNTIME_TRACE", None)
    result = subprocess.run([str(startup_runtime_program), "sequence"], text=True, capture_output=True,
                            timeout=15, env=env)
    assert result.returncode == 0, result.stderr
    assert not pricing_lines(result.stderr)
