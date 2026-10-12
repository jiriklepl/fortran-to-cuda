"""Keep container result locations visible without requiring a container runtime."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest


@pytest.mark.parametrize("engine", ["apptainer", "singularity"])
@pytest.mark.parametrize("exit_code", [0, 37])
def test_apptainer_announces_paths_and_preserves_command_status(tmp_path, engine, exit_code):
    commands = tmp_path / "commands"
    commands.mkdir()
    for name in ("basename", "dirname", "mkdir"):
        (commands / name).symlink_to(shutil.which(name))
    capture = tmp_path / "argv.json"
    runtime = commands / engine
    runtime.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['CAPTURE']).write_text(json.dumps(sys.argv[1:]))\n"
        f"sys.exit({exit_code})\n"
    )
    runtime.chmod(0o755)
    image = tmp_path / "image with spaces.sif"
    image.touch()
    output = tmp_path / "results with spaces"
    marker = tmp_path / "must-not-run"
    argv = ["run-elmm", "--label", f"$(touch {marker}) & literal"]
    env = {key: value for key, value in os.environ.items()
           if key not in {"WORKSPACE_ROOT", "PIPELINE_ROOT", "COMPILER_ROOT", "GPU_DEVICES",
                          "CUDA_VISIBLE_DEVICES"}
           and not key.startswith(("APPTAINER", "SINGULARITY"))}
    env.update(PATH=str(commands), SIF_FILE=str(image), OUTPUT_DIR=str(output), CAPTURE=str(capture))
    script = Path(__file__).resolve().parents[1] / "run-apptainer.sh"
    result = subprocess.run(["/bin/bash", str(script), *argv], env=env, text=True,
                            capture_output=True, timeout=10)
    assert result.returncode == exit_code, result.stderr
    assert f"Container runtime: {engine}\n" in result.stdout
    assert f"SIF image: {image}\n" in result.stdout
    assert f"Output directory: {output} -> /output\n" in result.stdout
    announced = next(line.removeprefix("Command:") for line in result.stdout.splitlines()
                     if line.startswith("Command:"))
    assert shlex.split(announced) == argv
    actual = json.loads(capture.read_text())
    assert actual[-len(argv):] == argv
    assert f"{output}:/output" in actual
    assert not marker.exists()
