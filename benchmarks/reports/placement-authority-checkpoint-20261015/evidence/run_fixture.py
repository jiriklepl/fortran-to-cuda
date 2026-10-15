"""One bounded, fresh generic fixture campaign; never change its decision."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
destination = HERE / "fixture"
if destination.exists():
    raise SystemExit("Refusing to overwrite an existing campaign")
command = ["taskset", "-c", "4,5,10,11", str(ROOT / ".venv/bin/python"),
           "-m", "compiler.offload.exact_window_calibration", "--output", str(destination),
           "--sizes", "16,192", "--architecture", "sm_86", "--original-teams"]
environment = dict(os.environ)
for name in ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "FORT_RUNTIME_TRACE", "FORT_EXACT_CALIBRATION_MODE"):
    environment.pop(name, None)
settings = {"CUDA_HOME": "/usr/local/cuda", "CUDA_VISIBLE_DEVICES": "GPU-d16b0d42-ee00-3af3-0c7e-d757b0cf9515",
            "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "OMP_PROC_BIND": "FALSE", "OMP_SCHEDULE": "static"}
environment.update(settings)
commits = {name: subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
           for name, path in (("compiler", ROOT), ("pipeline", ROOT / "elmm-pipeline"))}
started = time.monotonic()
timed_out = False
with (HERE / "fixture-driver.log").open("wb") as log:
    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    try:
        code = process.wait(timeout=295)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGTERM)
        try:
            code = process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            code = process.wait()
receipt = {"schema_version": 1, "repositories": commits, "command": command,
           "environment_overrides": settings, "timeout_seconds": 295, "termination_grace_seconds": 5,
           "timed_out": timed_out, "exit_code": code, "wall_seconds": time.monotonic() - started,
           "scope": "generic original-team cold invocation; not channel qualification"}
(HERE / "fixture-run.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt))
raise SystemExit(code if not timed_out else 124)
