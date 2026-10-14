"""Archive intermediate correction checks without relabelling final validation."""

import json
import re
import subprocess
import tarfile
from hashlib import sha256
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
DESTINATION = Path(__file__).resolve().parent
BASE = "c16d01a0e2028c0f610f1291277f37df413344c9"
CORRECTED = "ff9565f8ac941f9d9fad19e20027840e83f953b5"
OVERLAY = Path("/tmp/checked-plan-correction-overlay")
FOCUSED = Path("/tmp/checked-plan-corrected-original-validation")
CROSS_PROCESS = Path("/tmp/checked-plan-cross-process-review")


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def write_once(relative, payload):
    path = DESTINATION / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError("refusing to replace archived evidence: " + str(path))
    else:
        path.write_bytes(payload)


def revision_files(revision):
    content = subprocess.check_output(["git", "archive", revision, "compiler"], cwd=ROOT)
    with tarfile.open(fileobj=BytesIO(content)) as archive:
        return {item.name: archive.extractfile(item).read() for item in archive if item.isfile()}


def fingerprint(payload):
    return sha256(payload).hexdigest()


def main():
    copies = {
        "baseline/receipt.json": OVERLAY / "baseline-receipt.json",
        "baseline/pytest.log": OVERLAY / "baseline.log",
        "baseline/test_corrections.py.txt": OVERLAY / "test_corrections.py",
        "overlay/receipt.json": OVERLAY / "overlay-consumers-receipt.json",
        "overlay/pytest.log": OVERLAY / "overlay-consumers.log",
        "overlay/modules/execution_plan.py.txt": OVERLAY / "execution_plan.py",
        "overlay/modules/regions.py.txt": OVERLAY / "regions.py",
        "overlay/modules/sitecustomize.py.txt": OVERLAY / "sitecustomize.py",
        "focused/receipt.json": FOCUSED / "focused-receipt.json",
        "focused/pytest.log": FOCUSED / "focused.log",
        "focused/authored-correction.patch": FOCUSED / "correction.patch",
    }
    copied = {}
    for relative, original in copies.items():
        payload = original.read_bytes()
        write_once(relative, payload)
        copied[relative] = {"original_path": str(original), "sha256": fingerprint(payload)}

    receipts = {name: json.loads((DESTINATION / name / "receipt.json").read_text())
                for name in ("baseline", "overlay", "focused")}
    base_files, corrected_files = revision_files(BASE), revision_files(CORRECTED)
    base_hashes = {name: fingerprint(data) for name, data in base_files.items()}
    corrected_hashes = {name: fingerprint(data) for name, data in corrected_files.items()}
    runs = {}
    for name, receipt in receipts.items():
        assert receipt["before"] == receipt["after"], name
        log = (DESTINATION / name / "pytest.log").read_text()
        expected = {"baseline": ("failed", 3), "overlay": ("passed", 120), "focused": ("passed", 90)}[name]
        match = re.search(rf"{expected[1]} {expected[0]}(?:, \d+ warnings)? in ([0-9.]+)s", log)
        assert match is not None, name
        relative_inventory = {path: digest for path, digest in receipt["before"].items()
                              if path.startswith("compiler/")}
        runs[name] = {
            "classification": "intermediate correction review; not final release validation",
            "result": {expected[0]: expected[1]},
            "pytest_seconds": float(match.group(1)),
            "runner_seconds": receipt["seconds"],
            "exit_code": receipt["exit_code"],
            "receipt": name + "/receipt.json", "log": name + "/pytest.log",
            "recorded_source_inventory_unchanged": True,
            "recorded_file_count": len(receipt["before"]),
            "repository_paths_differing_from_base": [path for path, digest in relative_inventory.items()
                                                     if base_hashes.get(path) != digest],
        }
    assert runs["baseline"]["exit_code"] == 1
    assert runs["overlay"]["exit_code"] == runs["focused"]["exit_code"] == 0
    runs["baseline"]["coverage_limit"] = (
        "Pre/post inventory records four repository files. The external regression script is archived "
        "and hashed at curation time; its hash was not recorded before/after execution.")
    runs["overlay"]["coverage_limit"] = (
        "Pre/post inventory records two repository implementation files, six test files and three "
        "replacement modules. It is not a complete compiler source inventory. The overlay predates "
        "the absent-preparation-registry fix and later captured-lowering corrections.")
    for relative, original in copies.items():
        if relative.startswith("overlay/modules/"):
            assert copied[relative]["sha256"] == receipts["overlay"]["before"][str(original)], relative
    runs["overlay"]["archived_replacement_modules_match_recorded_hashes"] = True
    runs["focused"]["coverage_limit"] = (
        "Pre/post inventory records every compiler Python implementation file and the three invoked "
        "test modules. This snapshot predates the five later source changes listed in revision-binding.json.")

    inventory = receipts["focused"]["before"]
    changed_from_base = sorted(path for path, digest in inventory.items() if base_hashes.get(path) != digest)
    for path in changed_from_base:
        assert corrected_hashes[path] == inventory[path], path
    reconstruction = subprocess.check_output(
        ["git", "diff", "--binary", BASE, CORRECTED, "--", *changed_from_base], cwd=ROOT)
    write_once("focused/intermediate-snapshot.patch", reconstruction)
    rebuilt = dict(base_hashes)
    rebuilt.update((path, corrected_hashes[path]) for path in changed_from_base)
    assert all(rebuilt.get(path) == digest for path, digest in inventory.items())
    later_changes = [{"path": path, "tested_sha256": digest, "corrected_commit_sha256": corrected_hashes[path]}
                     for path, digest in inventory.items() if corrected_hashes.get(path) != digest]
    write_once("revision-binding.json", encoded({
        "schema_version": 1,
        "base_commit": BASE, "later_corrected_commit": CORRECTED,
        "focused_snapshot": {
            "description": "base commit plus intermediate-snapshot.patch; an intermediate working revision",
            "patch": "focused/intermediate-snapshot.patch",
            "patched_paths": changed_from_base,
            "all_recorded_file_hashes_match_reconstruction": True,
            "recorded_file_count": len(inventory),
            "later_corrected_commit_differences": later_changes,
        },
        "final_corrected_release_validation": False,
    }))

    observations = {}
    for seed in (1, 91237):
        path = CROSS_PROCESS / f"records-{seed}.json"
        raw = path.read_bytes()
        records = json.loads(raw)
        observations[str(seed)] = {
            "original_diagnostic_record_path": str(path),
            "original_diagnostic_record_sha256": fingerprint(raw),
            "full_diagnostic_record_archived": False,
            "fixtures": {},
        }
        for mode, record in records.items():
            contract = record["contract"]
            private = [item["value"]["extraction"]["private_scalars"] for item in record["payloads"]
                       if isinstance(item["value"], dict)
                       and item["value"].get("portable_numerical_preparation_version") == 1]
            observations[str(seed)]["fixtures"][mode] = {
                "execution_identity": record["execution_identity"],
                "contract_sha256": fingerprint(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()),
                "source_hashes": contract["sources"], "worker_hashes": contract["workers"],
                "private_scalar_inventories": private,
            }
    write_once("cross-process-before.json", encoded({
        "schema_version": 1,
        "status": "read-only diagnostic before correction; derived extract of larger temporary records",
        "observations": observations,
        "cause": "RegionExtraction PRIVATE inventories inherited hash-seed-dependent set iteration order.",
        "correction": "Sort private resource roots before constructing the scalar and array inventories.",
        "after_evidence": "focused/pytest.log and its source-bound subprocess regression test",
        "application_performance_or_measured_calibration": False,
    }))

    write_once("evidence.json", encoded({
        "schema_version": 1,
        "claim": "Narrow correctness evidence for cross-process fingerprint reuse and explicit registered borrowing.",
        "base_commit": BASE, "later_corrected_commit": CORRECTED,
        "runs": runs, "copied_artifacts": copied,
        "before_after_regressions": [
            {"before": "Unknown frontend dataclass with validate/public retained mutable semantic children.",
             "after": "Unregistered metadata freezes normally, detaches retained aliases and contributes semantic fields to identity."},
            {"before": "A copied, unregistered PreparedOriginalRegion was treated as borrowed authority.",
             "after": "Known proof classes require exact membership in their existing issuing registries."},
            {"before": "Serial and original-team fixtures changed executable and contract hashes across Python hash seeds.",
             "after": "Same-path fixture compilation under seeds 1 and 91237 yields equal executable and contract hashes."},
        ],
        "limitations": [
            "All tests here cover intermediate source revisions; final full host/CUDA validation is separate.",
            "Outside the admitted complete-original-window class, graph-sharing differences may change diagnostic fingerprints despite equal emitted artifacts.",
            "Immutability follows ordinary supported APIs and existing borrowed frozen-dataclass contracts; reflective mutation bypasses are not hardened here.",
            "Synthetic profitable observations used by unit tests are not measured calibration or application performance.",
        ],
        "final_corrected_release_validation": False,
        "application_generation_or_performance_qualification": False,
    }))
    files = {str(path.relative_to(DESTINATION)): fingerprint(path.read_bytes())
             for path in sorted(DESTINATION.rglob("*"))
             if path.is_file() and path.name != "artifact-inventory.json"}
    write_once("artifact-inventory.json", encoded({"schema_version": 1, "sha256": files}))
    print(json.dumps({"artifacts": len(files), "focused_tested_files": len(inventory),
                      "later_source_differences": len(later_changes)}))


if __name__ == "__main__":
    main()
