"""Bounded complete-process checkpoint: one warmup, three rotated rounds."""
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MODES = {
    "native": Path("/tmp/fortran-residency-e2e-20261014/repaired-native-checkpoint-gfortran14/build/ELMM"),
    "sgs-direct": HERE / "sgs-control/build/ELMM",
    "sgs-plus-diffusion": HERE / "combined/build/ELMM",
}
rows = []
for round_number in range(4):
    order = list(MODES)
    if round_number:
        offset = round_number - 1
        order = order[offset:] + order[:offset]
    for mode in order:
        output = HERE / "runs/untraced/channel-simple" / f"round-{round_number}" / mode
        argv = [sys.executable, str(HERE / "run_case.py"), "--case", "channel-simple",
                "--binary", str(MODES[mode]), "--mode", mode, "--output", str(output),
                "--reference", str(HERE / "runs/fields/channel-simple/native"),
                "--gpu", "GPU-d16b0d42-ee00-3af3-0c7e-d757b0cf9515"]
        subprocess.run(argv, cwd=ROOT, check=True)
        row = json.loads((output / "run.json").read_text())
        row.update(round=round_number, excluded_warmup=round_number == 0,
                   receipt=str(output / "run.json"))
        rows.append(row)
        (HERE / "untraced-comparison.json").write_text(json.dumps({
            "schema_version": 1, "qualification": False,
            "reason": "Three diagnostic rounds of forced placement; not seven automatic qualification rounds",
            "runs": rows, "complete": len(rows) == 12}, indent=2) + "\n")
