"""Freeze the final bounded checkpoint, excluding large field/object payloads."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEST = ROOT / "benchmarks/reports/placement-authority-checkpoint-20261015"
FIXTURE = HERE / "fixture"
receipt = json.loads((HERE / "fixture-run.json").read_text())
if receipt["exit_code"] or receipt["timed_out"]:
    raise SystemExit("Campaign did not complete; retain raw diagnostics instead of publishing success")
DEST.mkdir(exist_ok=False)
inventory = []


def archive(source, name):
    raw = source.read_bytes()
    target = DEST / "evidence" / name
    if len(raw) > 200_000:
        target = target.with_name(target.name + ".gz")
        payload = gzip.compress(raw, mtime=0)
    else:
        payload = raw
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    inventory.append({"source": str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
                      "archive": str(target.relative_to(DEST)), "source_bytes": len(raw),
                      "source_sha256": hashlib.sha256(raw).hexdigest(),
                      "archive_sha256": hashlib.sha256(payload).hexdigest()})


for name in ("host-initial.log", "host-final.log", "pipeline.log", "cuda.log", "fixture-driver.log",
             "fixture-run.json", "run_fixture.py", "freeze_report.py"):
    archive(HERE / name, name)
for name in ("exact-window-placement-authority-final-20261015.log", "exact-window-placement-lint-final-20261015.log"):
    archive(Path("/tmp") / name, "tests/" + name)
for path in sorted(FIXTURE.rglob("*")):
    relative = path.relative_to(FIXTURE)
    if path.is_file() and (path.suffix in {".stdout", ".stderr"}
                          or path.name in {"report.json", "evidence.json", "production-observations.json",
                                           "build-receipt.json", "scope-manifest.json", "original.f90"}
                          or path.name.endswith("ledger.json")):
        archive(path, "fixture/" + str(relative))
shutil.copyfile("/tmp/render-placement-checkpoint.py", DEST / "render.py")
subprocess.run([str(ROOT / ".venv/bin/python"), str(DEST / "render.py"),
                "--report", str(FIXTURE / "report.json"), "--output", str(DEST)], check=True)
(DEST / "artifact-inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
metadata = {"schema_version": 1, "repositories": receipt["repositories"],
            "application_qualification": False, "automatic_application_window": False,
            "tests": {"compiler_host_distinct": 173, "compiler_cuda": 3, "pipeline": 111,
                      "pipeline_opt_in_skipped": 2, "post_style_recheck": 4},
            "fixture_wall_seconds": receipt["wall_seconds"],
            "source_inventory": json.loads((FIXTURE / "calibration/scope-manifest.json").read_text())[
                "scopes"][0]["exact_window"]["contract"]["implementation"],
            "prior_application_evidence": {"report": "../indirect-reuse-checkpoint-20261015/README.md",
                "compiler": "10ae18baf5097e6414a9b0698ee382c92dc757b9",
                "pipeline": receipt["repositories"]["pipeline"], "rerun_for_this_slice": False},
            "archived_inputs": len(inventory)}
(DEST / "report.json").write_text(json.dumps(metadata, indent=2) + "\n")
(DEST / ".gitattributes").write_text("evidence/** -whitespace\n*.svg -whitespace\n")
print(json.dumps({"report": str(DEST), "inputs": len(inventory)}))
