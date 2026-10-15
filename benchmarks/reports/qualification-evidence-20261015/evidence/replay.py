"""Replay synthetic qualification evidence, never run an application."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "elmm-pipeline"))
from elmm_pipeline.campaigns.run import summaries
from elmm_pipeline.campaigns.targets import evaluate_target

state = {"candidate_mode": "auto-resident", "modes": [{"name": mode} for mode in
         ("native", "sgs-direct", "auto-resident")]}
case = {"performance_target": {"schema_version": 1, "metric": "ratio_to_native", "limit": .744,
                              "comparison": "le", "control_mode": "sgs-direct"}}
before = json.loads((HERE / "before.json").read_text())
rows = []
for scenario in before["scenarios"]:
    runs = [{**row, "validation": {"passed": True}} for row in scenario["runs"]]
    result = evaluate_target(state, case, summaries(runs, state["modes"]), runs)
    expected = scenario["scenario"] == "complete"
    assert result["comfortable"]["met"] == expected, scenario["scenario"]
    rows.append({"scenario": scenario["scenario"], "runs": runs, "result": result})
source = ROOT / "elmm-pipeline/elmm_pipeline/campaigns/targets.py"
report = {"schema_version": 1, "kind": "synthetic qualification regression probe; no application timings",
          "validation_facts": "Synthetic passed checks supplied to every run for the stricter gate; no real workload claim",
          "before_sha256": hashlib.sha256((HERE / "before.json").read_bytes()).hexdigest(),
          "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
          "pipeline_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                       cwd=ROOT / "elmm-pipeline", text=True).strip(),
          "scenarios": rows}
output = HERE / "after.json"
if output.exists():
    raise ValueError("Existing probe evidence is immutable")
output.write_text(json.dumps(report, indent=2) + "\n")
print([(row["scenario"], row["result"]["comfortable"]["met"]) for row in rows])
