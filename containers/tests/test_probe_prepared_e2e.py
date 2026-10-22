"""Development probes must not inherit routing overrides or imply qualification."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parent / "elmm-pipeline")]
import probe_prepared_e2e as probe


def test_probe_environment_drops_inherited_application_and_routing_overrides(monkeypatch):
    for key, value in {"FORT_GPU_MODE": "native", "FORT_SCOPE_CUDA_GRAPHS": "0",
                       "FORT_RUNTIME_TRACE": "1", "ELMM_TIME_LIMIT": "0.01",
                       "OMP_NUM_THREADS": "1", "GOMP_CPU_AFFINITY": "0"}.items():
        monkeypatch.setenv(key, value)
    environment = probe.probe_environment(4)
    assert not any(key.startswith(("FORT_", "ELMM_", "GOMP_")) for key in environment)
    assert environment["OMP_NUM_THREADS"] == "4"
    assert environment["OMP_DYNAMIC"] == "FALSE"
    assert environment["OMP_PROC_BIND"] == "false"


def test_explicit_single_sample_diagnostic_cannot_claim_an_iqr():
    rows = [{"mode": mode, "warmup": False, "traced": False, "full_storage": False,
             "wall_seconds": seconds} for mode, seconds in zip(probe.runner.MODES, (12., 10., 20.))]
    summary = probe.timing_summary(rows)
    assert summary["reworked"]["repetitions"] == 1
    assert summary["reworked"]["median_seconds"] == 20.
    assert summary["reworked"]["iqr_seconds"] is None
