"""Read one completed process trace; never run an application or infer placement.

All upload/download events are counted, including native interruptions and
publication before any close. Direct CUDA copies are submissions until the
context closes successfully. Missing closes preserve partial counts and fail
the complete-ledger check. Logical trace counts are not CUDA copy-call counts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EVENT = re.compile(r"^FORT_SCOPED ([a-z_]+) (.*)$")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def counters():
    return {"upload_events": 0, "upload_bytes": 0, "download_events": 0,
            "download_bytes": 0, "launch_events": 0, "wait_events": 0}


def count(row, event, amount):
    key = event + "_events"
    if key in row:
        row[key] += 1
    if event in {"upload", "download"}:
        row[event + "_bytes"] += amount


def add(target, source):
    for key in target:
        target[key] += source[key]


def extract(args):
    sys.path.insert(0, str(args.pipeline_root.resolve(strict=True)))
    from elmm_pipeline import gpu_validation, scoped_provenance
    from elmm_pipeline.campaigns import transfers

    manifests = [json.loads(path.read_text()) for path in args.manifest]
    # Existing parser validates public identity schema and conflicting packages.
    scoped_provenance.scoped_provenance("", manifests)
    records = {row["id"]: row for manifest in manifests
               for row in manifest.get("runtime_provenance", {}).get("records", [])}
    owners = {}
    for manifest in manifests:
        for scope in manifest.get("scopes", []):
            entry = scope.get("checked_execution_plan", {}).get("entry", manifest.get("entry"))
            candidates = [row for row in manifest.get("runtime_provenance", {}).get("records", [])
                          if row["kind"] == "owner" and row.get("procedure") == entry
                          and row.get("source") == scope.get("path")]
            if len(candidates) != 1:
                continue
            owner = candidates[0]["id"]
            mapping = {row["registration_identity"]: row["resource"]
                       for row in scope.get("resource_bindings", {}).get("resources", [])}
            value = {"procedure": entry, "resources": mapping,
                     "requested_transfers": scope.get("transfer_configuration", {}).get("requested")}
            if owner in owners and owners[owner] != value:
                raise ValueError("Ambiguous canonical-resource mapping for a traced owner")
            owners[owner] = value
    contexts, stats, source_groups = {}, {}, {}
    totals, parser_totals = counters(), Counter()
    event_counts, offload_modes, offload_reasons = Counter(), Counter(), Counter()
    model_reasons, close_reasons, transfer_reasons = Counter(), Counter(), Counter()
    model_decisions, model_incomplete, definition_failures = 0, 0, []
    legacy, unrecognized, issues = Counter(), Counter(), []

    def context(number):
        return contexts.setdefault(number, {"context": number, "created": 0, "closed": 0,
            "events_after_close": 0, "owners": set(), "observed_submissions": counters(),
            "event_counts": Counter(), "resources": {}, "phases": {}, "close_boundaries": [],
            "decision_reasons": Counter(), "requested_transfer_modes": set()})

    def consume(lines):
        nonlocal model_decisions, model_incomplete
        text = "".join(lines)
        # Keep the existing strict evidence/provenance parsers authoritative.
        activity = gpu_validation.scoped_runtime_activity(text)
        provenance = scoped_provenance.scoped_provenance(text, manifests)
        parser_totals.update({"upload_bytes": activity["upload_bytes"],
                              "download_bytes": activity["download_bytes"],
                              "launch_events": activity["kernel_launches"]})
        legacy["unversioned_scoped_events"] += provenance["legacy_events"]
        policies = gpu_validation.offload_decisions(text, [],
            scope_packages=[{"scope_manifest": manifest} for manifest in manifests])
        offload_modes.update(policies["modes"])
        offload_reasons.update(policies["native_reasons"])
        for row in activity["modeled_decisions"]:
            model_decisions += 1
            model_reasons[row["reason"]] += 1
        for row in activity["modeled_evidence"]:
            item = context(row["context"])
            if row["event"] == "transfer_statistics":
                old = stats.get(row["context"])
                if old is not None:
                    raise ValueError("Repeated completed transfer statistics for one context")
                stats[row["context"]] = row
                transfer_reasons[row["reason"]] += 1
            elif row["event"] == "decision":
                item["decision_reasons"][row.get("reason", "unknown")] += 1
                if "transfer_requested_mode" in row:
                    item["requested_transfer_modes"].add(row["transfer_requested_mode"])
            elif row["event"] == "end" and (not row["complete"] or row["truncated"]):
                model_incomplete += 1
            elif row["event"] == "definition_validation" and row["status"] != 0:
                definition_failures.append(row)
        for line in lines:
            if "FORT_RUNTIME " in line:
                for direction, value in re.findall(r"FORT_RUNTIME (upload|download) bytes=(\d+)", line):
                    legacy[direction + "_bytes"] += int(value)
                    legacy[direction + "_events"] += 1
                if "FORT_RUNTIME launch" in line:
                    legacy["launch_events"] += 1
            match = EVENT.match(line.rstrip("\n"))
            if not match:
                continue
            event, encoded = match.groups()
            if event in {"evidence", "decision", "planning_timing"}:
                continue
            if event not in scoped_provenance.OBSERVED:
                # Coalesced diagnostics duplicate the following upload; never add them.
                if event not in {"position", "restore", "initialize_reuse", "upload_coalesced", "upload_requested"}:
                    unrecognized[event] += 1
                continue
            fields = dict(token.split("=", 1) for token in encoded.split())
            if fields.get("provenance_version") != "1":
                continue  # Preserved by the pipeline aggregate and marked incomplete.
            number, amount = int(fields["context"]), int(fields["bytes"])
            item = context(number)
            if item["closed"]:
                item["events_after_close"] += 1
            if event == "create":
                item["created"] += 1
            if event == "close":
                item["closed"] += 1
                boundary = fields["boundary"]
                reason = records.get(boundary, {}).get("reason", "unattributed successful close")
                close_reasons[reason] += 1
                item["close_boundaries"].append({"identity": boundary, "reason": reason})
            owner = fields["owner"]
            if owner != "unknown":
                item["owners"].add(owner)
            count(totals, event, amount)
            count(item["observed_submissions"], event, amount)
            event_counts[event] += 1
            item["event_counts"][event] += 1
            implementation = records.get(fields["implementation"], {})
            phase = ("publication_or_cleanup" if fields["boundary"] != "unknown" else
                     "original_native_point" if implementation.get("backend") == "original_native" else
                     "cuda_point" if implementation.get("backend") == "cuda" else "other_or_unknown_point")
            count(item["phases"].setdefault(phase, counters()), event, amount)
            if "buffer" in fields:
                registration = int(fields["buffer"])
                resource = owners.get(owner, {}).get("resources", {}).get(registration)
                key = (owner, registration, int(fields["buffer_handle"]), int(fields["generation"]))
                observed = item["resources"].setdefault(key, {"owner": owner,
                    "canonical_resource": resource, "registration_identity": registration,
                    "buffer_handle": int(fields["buffer_handle"]), "generation": int(fields["generation"]),
                    "observed_submissions": counters(), "phases": {}})
                if observed["canonical_resource"] != resource:
                    raise ValueError("Canonical runtime registration mapping changed")
                count(observed["observed_submissions"], event, amount)
                count(observed["phases"].setdefault(phase, counters()), event, amount)
            if event in {"upload", "download", "launch", "close"}:
                key = tuple(fields[field] for field in scoped_provenance.FIELDS)
                group = source_groups.setdefault(key, {**dict(zip(scoped_provenance.FIELDS, key)),
                    "observed_submissions": counters(), "contexts": set()})
                group["contexts"].add(number)
                count(group["observed_submissions"], event, amount)

    before = {str(path.resolve()): digest(path) for path in
              [args.log, *args.manifest, args.runtime_source, Path(__file__),
               Path(gpu_validation.__file__), Path(scoped_provenance.__file__), Path(transfers.__file__)]}
    with args.log.open() as stream:
        batch = []
        for line in stream:
            batch.append(line)
            if len(batch) >= 4096:
                consume(batch)
                batch.clear()
        consume(batch)
    if any(digest(path) != value for path, value in before.items()):
        raise ValueError("Trace/manifest/parser input changed while extracting")
    pinned = transfers.aggregate_final_statistics(list(stats.values()))
    completed, incomplete = counters(), counters()
    phase_totals, completed_phases = {}, {}
    by_resource = {}
    for number, item in sorted(contexts.items()):
        valid_close = item["created"] == item["closed"] == 1 and not item["events_after_close"]
        item["successful_close_observed"] = valid_close
        add(completed if valid_close else incomplete, item["observed_submissions"])
        for phase, amounts in item["phases"].items():
            add(phase_totals.setdefault(phase, counters()), amounts)
            if valid_close:
                add(completed_phases.setdefault(phase, counters()), amounts)
        if not valid_close:
            issues.append({"context": number, "reason": "missing/repeated create or close, or activity after close"})
        row = stats.get(number)
        non_direct = any(owners.get(owner, {}).get("requested_transfers") in {"pinned", "auto", "pipelined", "coalesced"}
                         for owner in item["owners"]) or any(mode != 0 for mode in item["requested_transfer_modes"])
        if row is not None and not valid_close:
            issues.append({"context": number, "reason": "completed transfer statistics lack successful close event"})
        if valid_close and non_direct and row is None:
            issues.append({"context": number, "reason": "non-direct context lacks completed transfer statistics"})
        item["pinned_completed_statistics"] = row
        item["non_pinned_bytes"] = None
        if valid_close and (row is not None or not non_direct):
            remaining = {direction + "_bytes": item["observed_submissions"][direction + "_bytes"]
                         - (row["pinned_" + direction + "_bytes"] if row else 0)
                         for direction in ("upload", "download")}
            if min(remaining.values()) < 0:
                issues.append({"context": number, "reason": "pinned bytes exceed observed total bytes"})
            else:
                item["non_pinned_bytes"] = remaining
        for resource in item["resources"].values():
            if resource["canonical_resource"] is None and any(resource["observed_submissions"][key]
                    for key in ("upload_events", "download_events")):
                issues.append({"context": number, "registration": resource["registration_identity"],
                               "reason": "copy lacks unambiguous public canonical resource mapping"})
            key = (resource["owner"], resource["canonical_resource"])
            group = by_resource.setdefault(key, {"owner": key[0], "canonical_resource": key[1],
                "owner_procedure": owners.get(key[0], {}).get("procedure"), "observed_submissions": counters(),
                "successful_close_submissions": counters(), "contexts": set()})
            add(group["observed_submissions"], resource["observed_submissions"])
            if valid_close:
                add(group["successful_close_submissions"], resource["observed_submissions"])
            group["contexts"].add(number)
        item["resources"] = list(item["resources"].values())
        item["owners"] = sorted(item["owners"])
        item["requested_transfer_modes"] = sorted(item["requested_transfer_modes"])
    for key in ("upload_bytes", "download_bytes", "launch_events"):
        if parser_totals[key] != totals[key]:
            issues.append({"reason": "pipeline aggregate differs from versioned event ledger", "counter": key,
                           "pipeline": parser_totals[key], "versioned": totals[key]})
    for group in [*by_resource.values(), *source_groups.values()]:
        group["contexts"] = sorted(group["contexts"])
    if args.returncode != 0:
        issues.append({"reason": "application return code is not successful", "returncode": args.returncode})
    if not contexts:
        issues.append({"reason": "no contexts observed; trace activation/completeness is unproved"})
    if any(legacy.values()) or unrecognized:
        issues.append({"reason": "legacy or unrecognized runtime events remain unattributed"})
    if definition_failures:
        issues.append({"reason": "definition preflight failures observed", "count": len(definition_failures)})
    admitted = sum(manifest.get("scope_count", len(manifest.get("scopes", []))) for manifest in manifests)
    return {"schema_version": 1, "status": "complete_observed_trace" if not issues else "incomplete_observed_trace",
        "performance_measured": False, "qualification": False, "process_returncode": args.returncode,
        "input_sha256": before, "observed_submissions": totals, "pipeline_aggregate_crosscheck": parser_totals,
        "successful_close_submissions": completed, "incomplete_context_submissions": incomplete,
        "phase_submissions": phase_totals, "successful_close_phase_submissions": completed_phases,
        "event_counts": event_counts, "contexts": list(contexts.values()),
        "canonical_resources": list(by_resource.values()), "source_groups": list(source_groups.values()),
        "source_records": records, "pinned_successful_close_statistics": pinned,
        "close_reasons": close_reasons, "transfer_fallback_reasons": transfer_reasons,
        "offload_policy_modes": offload_modes, "offload_native_reasons": offload_reasons,
        "modeled_decisions": {"count": model_decisions, "reasons": model_reasons,
                              "incomplete_or_truncated_evidence_groups": model_incomplete},
        "definition_failures": definition_failures, "legacy_unattributed": legacy,
        "unrecognized_events": unrecognized, "issues": issues,
        "admission_vs_execution": {"manifest_scope_count": admitted,
            "observed_launch_events": totals["launch_events"],
            "admitted_without_observed_gpu_work": bool(admitted and not totals["launch_events"]),
            "contexts_without_launches": [row["context"] for row in contexts.values()
                                         if not row["observed_submissions"]["launch_events"]]},
        "semantics": [
            "upload/download totals include every matching event, without filtering source point or close reason",
            "direct and batch events record successful API submission; successful close confirms stream completion and publication",
            "failed close/abandon has no successful-close trace or complete statistics; partial traffic is retained without completion claims",
            "logical transfer event counts are not CUDA copy-call counts; pinned counters count physical copy tiles",
            "pinned statistics are a subset, never added to total transfer bytes; direct mode emits no transfer_statistics row",
            "upload_coalesced/upload_requested are supplementary and never double-counted",
            "native/publication phases use exact emitted provenance; unknown attribution remains explicit",
            "trace success is best-effort and has no sequence checksum; this report cannot prove omitted events never occurred",
            "modeled copy/export/decision records are not observed traffic; launch events are submissions, not performance measurements"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True, help="one process's complete stderr/combined trace, never concatenated runs")
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--runtime-source", type=Path, default=Path(__file__).parent / "artifacts/scoped_runtime.cu")
    parser.add_argument("--pipeline-root", type=Path, default=ROOT / "elmm-pipeline")
    parser.add_argument("--returncode", type=int, required=True, help="actual recorded process return code")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Existing ledger evidence is immutable")
    report = extract(args)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], **report["observed_submissions"],
                      "issues": len(report["issues"])}))
    if report["issues"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
