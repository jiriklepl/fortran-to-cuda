"""Freeze the bounded checkpoint; do not execute or tune applications."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "benchmarks/reports/diffusion-application-checkpoint-20261015"
EVIDENCE = OUT / "evidence"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(relative, compress=False, source=None):
    source = source or HERE / relative
    target = EVIDENCE / (relative + (".gz" if compress else ""))
    target.parent.mkdir(parents=True, exist_ok=True)
    assert not target.exists(), target
    if compress:
        target.write_bytes(gzip.compress(source.read_bytes(), mtime=0))
    else:
        shutil.copyfile(source, target)
    return {"original": str(source), "sha256": digest(source),
            "archived": str(target.relative_to(OUT)), "compressed": compress}


campaign = read(HERE / "untraced-comparison.json")
assert campaign["complete"] and len(campaign["runs"]) == 12
assert all(r["completed"] and not r["traced"] and r["binary_unchanged"]
           and r["configuration_unchanged"] and r["completion"]["completed_steps"] == 109
           for r in campaign["runs"])
inventory = []
for relative in ("request.json", "capture-facts.json", "command.json", "completion.json",
                 "worker-result.json", "readiness-audit.json", "source-before.json", "source-after.json",
                 "run.py", "build_candidate.py", "run_case.py", "compare_complete.py",
                 "full_field_diagnostic.py", "transfer_ledger.py", "freeze_report.py",
                 "channel-simple-sgs-fields.json", "channel-simple-combined-fields.json",
                 "untraced-comparison.json", "generation.log"):
    inventory.append(freeze(relative))
for relative in ("artifacts/scope-manifest.json", "sgs-control/artifacts/scope-manifest.json",
                 "channel-simple-sgs-ledger.json", "channel-simple-combined-ledger.json"):
    inventory.append(freeze(relative, True))
for mode in ("sgs-control", "combined"):
    for name in ("build-generation.json", "native-command-comparison.json"):
        inventory.append(freeze(f"{mode}/{name}"))
    inventory.append(freeze(f"{mode}/composition.json", True))
    inventory.append(freeze(f"{mode}/build/build-receipts.json", True))
for name in ("request.json", "capture-facts.json", "completion.json", "command.json", "run.py"):
    inventory.append(freeze(f"sgs-control/{name}"))
native = Path("/tmp/fortran-residency-e2e-20261014/repaired-native-checkpoint-gfortran14/build/build-receipts.json")
inventory.append(freeze("native/build-receipts.json", True, native))
for mode in ("native", "sgs-direct", "sgs-plus-diffusion"):
    for name in ("plan.json", "build.json", "dump-main.f90"):
        inventory.append(freeze(f"full-fields/{mode}/{name}", name.endswith(".f90")))
for path in sorted((HERE / "runs").rglob("run.json")):
    inventory.append(freeze(str(path.relative_to(HERE))))
    if "fields" in path.parts:
        for name in ("stdout.log", "stderr.log"):
            inventory.append(freeze(str(path.with_name(name).relative_to(HERE)), True))

performance = {}
for mode in ("native", "sgs-direct", "sgs-plus-diffusion"):
    samples = [r["diagnostic_complete_process_seconds"] for r in campaign["runs"]
               if r["mode"] == mode and not r["excluded_warmup"]]
    assert len(samples) == 3
    performance[mode] = {"samples_seconds": samples, "median_seconds": statistics.median(samples),
                         "minimum_seconds": min(samples), "maximum_seconds": max(samples)}
for row in performance.values():
    row["native_relative_median"] = row["median_seconds"] / performance["native"]["median_seconds"]

transfers = {}
for mode, name in (("sgs-direct", "sgs"), ("sgs-plus-diffusion", "combined")):
    ledger = read(HERE / f"channel-simple-{name}-ledger.json")
    assert ledger["observed_submissions"] == ledger["successful_close_submissions"]
    assert all(c["successful_close_observed"] for c in ledger["contexts"])
    issues = Counter(row["reason"] for row in ledger["issues"])
    assert issues == {"copy lacks unambiguous public canonical resource mapping": 1312}
    transfers[mode] = {"observed": ledger["observed_submissions"], "ledger_status": ledger["status"],
        "contexts_closed": len(ledger["contexts"]), "phase_submissions": ledger["phase_submissions"],
        "attribution_issues": dict(issues), "predicted_bytes": None,
        "prediction_unavailable_reason": "No compatible automatic application estimate",
        "canonical_resources": ledger["canonical_resources"]}
delta = {k: transfers["sgs-plus-diffusion"]["observed"][k] - transfers["sgs-direct"]["observed"][k]
         for k in transfers["sgs-direct"]["observed"]}
correctness = {}
for name in ("sgs", "combined"):
    check = read(HERE / f"channel-simple-{name}-fields.json")
    assert check["status"] == "passed" and len(check["comparisons"]) == 20
    correctness[name] = {"status": check["status"], "fields": 20,
        "nonempty_fields": sum(c["values"] > 0 for c in check["comparisons"].values()),
        "max_absolute_error": max(c["max_absolute_error"] for c in check["comparisons"].values()),
        "atol": 1e-6, "rtol": 1e-5, "includes_bounds_and_halos": True,
        "final_and_averaged_vtk_passed": check["vtk"]["candidate"]["passed"]}

report = {"schema_version": 1, "status": "bounded_diagnostic_complete_targets_unmet",
    "commits": read(HERE / "request.json")["commits"],
    "automatic_placement_available": False, "qualification": False,
    "targets": {"channel-simple": {"seconds_below": 9.0, "status": "unmet"},
                "channel": {"native_fraction_at_most": .744, "status": "unqualified_candidate_not_run"}},
    "protocol": {"case": "channel-simple", "steps": 109, "configuration_changes": [],
        "threads": 4, "affinity": [4, 5, 10, 11], "excluded_warmups_per_mode": 1,
        "measured_rounds": 3, "rotated": True, "tracing_separate": True,
        "timing": "Complete process from Popen through wait, including startup/output/teardown; validation excluded",
        "placement": "forced sections policy; SGS and diffusion have separate owners",
        "limitations": "Diagnostic only; seven-round automatic qualification not attempted"},
    "performance": performance, "transfers": transfers, "diffusion_increment": delta,
    "diffusion_invocations": 327, "correctness": correctness,
    "channel": {"native_completed_steps": 2458, "native_natural_end": True,
                "candidate_executed": False, "qualification": False,
                "reason": "Bounded stop after channel-simple regression; user requested wrap-up"},
    "decision": "Stop this isolated diffusion placement; do not add automatic pricing or expand its window on this evidence",
    "outstanding": ["Compiler-owned automatic placement for actual application intervals",
        "Shared producer through diffusion and accumulation ownership",
        "Both primary application qualification campaigns", "Full channel candidate numerical validation"],
    "evidence": inventory}
(OUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps({"performance": performance, "diffusion_increment": delta, "correctness": correctness}, indent=2))
