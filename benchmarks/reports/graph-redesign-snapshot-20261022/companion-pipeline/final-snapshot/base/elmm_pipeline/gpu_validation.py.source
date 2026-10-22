"""Read actual GPU activity from separate, numerically checked trace runs."""
from __future__ import annotations

import json
import math
import re

from .scoped_provenance import scoped_provenance

TRACE_VARIABLES = ("FORT_RUNTIME_TRACE", "FORT_OFFLOAD_TRACE", "NVCOMPILER_ACC_NOTIFY", "NVCOMPILER_ACC_TIME",
                   "PGI_ACC_NOTIFY", "PGI_ACC_TIME", "CUDA_LAUNCH_BLOCKING")


def trace_environment(environment, tool, *, enabled=False):
    result = dict(environment)
    for name in TRACE_VARIABLES:
        result.pop(name, None)
    if enabled and tool == "local":
        result["FORT_RUNTIME_TRACE"] = "1"
        result["FORT_OFFLOAD_TRACE"] = "1"
    return result


def _scope_manifests(scope_manifest, scope_packages):
    """Transport separate compiler manifests without synthesizing decisions."""
    manifests = []
    if scope_manifest is not None:
        if not isinstance(scope_manifest, dict):
            raise ValueError("Scoped diagnostics require a public source-scope manifest")
        manifests.append(scope_manifest)
    if scope_packages is not None:
        if not isinstance(scope_packages, (list, tuple)):
            raise ValueError("Scoped diagnostics require a list of compiler packages")
        for package in scope_packages:
            if not isinstance(package, dict) or not isinstance(package.get("scope_manifest"), dict):
                raise ValueError("Scoped diagnostics package lacks its public source-scope manifest")
            manifests.append(package["scope_manifest"])
    return manifests


def offload_decisions(log, regions, *, scope_manifest=None, scope_packages=None):
    """Separate policy decisions from evidence that a CUDA kernel ran."""
    entries = {region["entry"].lower() for region in regions}
    manifests = _scope_manifests(scope_manifest, scope_packages)
    for manifest in manifests:
        entries.update(item["entry"].lower().split("::")[-1] for item in manifest.get("numerical_sources", []))
        entries.update(leaf.lower().split("::")[-1] for scope in manifest.get("scopes", [])
                       for leaf in scope.get("gpu_leaves", []))
    pattern = (r"(?<![A-Za-z0-9_])FORT_OFFLOAD[ \t]+entry=([A-Za-z][A-Za-z0-9_]*)"
               r"[ \t]+mode=(native(?:-scoped-[a-z-]+)?|scoped-scheduled|gpu|mixed|hybrid|chunked)[ \t]+gpu_units=(\d+)"
               r"[ \t]+cpu_units=(\d+)(?![A-Za-z0-9_])"
               r"(?:[ \t]+unit_kind=(regions|slabs)(?![A-Za-z0-9_]))?")
    counts = {mode: 0 for mode in ("native", "gpu", "mixed", "hybrid", "chunked", "scoped-scheduled")}
    per_entry = {}
    unit_counts = {}
    reasons = {}
    for entry, mode, gpu, cpu, unit_kind in re.findall(pattern, log):
        entry = entry.lower()
        unit_kind = unit_kind or "regions"
        if entry not in entries:
            raise ValueError("Offload policy trace contains an entry absent from the build manifest")
        if (mode.startswith("native-scoped-") or mode == "scoped-scheduled") and not manifests:
            raise ValueError("Scoped offload trace requires its public source-scope manifest")
        if mode.startswith("native-scoped-"):
            reasons[mode] = reasons.get(mode, 0) + 1
            mode = "native"
        gpu, cpu = int(gpu), int(cpu)
        if mode == "native" and gpu:
            raise ValueError("Native offload policy decision claims GPU work")
        counts[mode] += 1
        record = per_entry.setdefault(entry, {"calls": 0, "gpu_units": 0, "cpu_units": 0, "modes": {},
                                               "unit_kind": unit_kind})
        if record["unit_kind"] != unit_kind:
            raise ValueError("Offload policy trace changes unit kind for one entry")
        units = unit_counts.setdefault(unit_kind, {"gpu_units": 0, "cpu_units": 0})
        units["gpu_units"] += gpu
        units["cpu_units"] += cpu
        record["calls"] += 1
        record["gpu_units"] += gpu
        record["cpu_units"] += cpu
        record["modes"][mode] = record["modes"].get(mode, 0) + 1
    return {"calls": sum(counts.values()), "modes": counts, "entries": per_entry, "unit_counts": unit_counts,
            "native_reasons": reasons,
            "gpu_units": sum(value["gpu_units"] for value in per_entry.values()),
            "cpu_units": sum(value["cpu_units"] for value in per_entry.values())}


def native_policy_exercised(activity, regions):
    """Intentional all-native policy execution is valid without a GPU claim."""
    policies = [region.get("transformation", {}).get("compiler_report", {}).get("offload") or {}
                for region in regions]
    scoped = activity.get("scope_manifest_count", 0) > 0 or any(
        region.get("transformation", {}).get("memory_model") == "scoped" for region in regions)
    decisions = activity.get("offload_decisions", {})
    return activity.get("compiler_native_selection", False) or ((scoped or any(policy.get("policy", "always") != "always" for policy in policies))
            and decisions.get("modes", {}).get("native", 0) > 0 and decisions.get("gpu_units", 0) == 0)


def scoped_runtime_activity(log):
    """Count common-runtime events without inferring identities from CUDA source."""
    events = re.findall(r"(?<![A-Za-z0-9_])FORT_SCOPED[ \t]+([a-z_]+)"
                        r"(?:[ \t]+buffer=(\d+)[ \t]+generation=(\d+))?[ \t]+bytes=(\d+)(?![A-Za-z0-9_])", log)
    counts = {}
    totals = {}
    for operation, _buffer, _generation, amount in events:
        counts[operation] = counts.get(operation, 0) + 1
        totals[operation] = totals.get(operation, 0) + int(amount)
    decisions = scoped_modeled_decisions(log)
    modeled_totals = {"calls": len(decisions), "available_calls": sum(record["available"] for record in decisions)}
    for name in ("gpu_units", "cpu_units", "candidates", "simulated_operations", "upload_bytes", "download_bytes", "launches", "waits",
                 "estimated_seconds", "native_seconds"):
        modeled_totals[name] = sum(record[name] for record in decisions)
    modeled_totals["peak_device_bytes"] = max((record["peak_device_bytes"] for record in decisions), default=0)
    return {"events": counts, "kernel_launches": counts.get("launch", 0),
            "upload_bytes": totals.get("upload", 0), "download_bytes": totals.get("download", 0),
            "allocations": counts.get("allocate", 0), "waits": counts.get("wait", 0),
            "modeled_decisions": decisions, "modeled_totals": modeled_totals,
            "modeled_evidence": scoped_modeled_evidence(log)}


def scoped_modeled_evidence(log):
    """Preserve public JSON evidence without interpreting compiler decisions.

    Resource sections, copies, exports and gates describe the modeled final
    schedule. Their records are distinct from observed CUDA execution events.
    The end record's completeness/truncation fields remain visible to readers.
    Versioned transfer_statistics records describe completed execution at close;
    preserving them does not itself verify a kernel launch.
    """
    events = {"decision", "snapshot", "operation", "requirement", "copy", "export", "gate", "end",
              "definition_validation", "transfer_statistics", "batch_statistics"}

    def object_fields(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Scoped evidence JSON contains a duplicate field")
            result[key] = value
        return result

    def constant(value):
        raise ValueError("Scoped evidence JSON must contain finite numbers: " + value)

    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Scoped evidence JSON must contain finite numbers")
        if isinstance(value, dict):
            for child in value.values():
                finite(child)
        elif isinstance(value, list):
            for child in value:
                finite(child)

    records = []
    for match in re.finditer(r"(?<![A-Za-z0-9_])FORT_SCOPED[ \t]+evidence[ \t]+([^\r\n]*)", log):
        try:
            record = json.loads(match.group(1), object_pairs_hook=object_fields, parse_constant=constant)
        except ValueError as error:
            raise ValueError("Malformed scoped evidence JSON: " + str(error)) from error
        if not isinstance(record, dict) or type(record.get("schema_version")) is not int or record["schema_version"] != 1:
            raise ValueError("Scoped evidence requires public schema_version 1")
        context = record.get("context")
        event = record.get("event")
        if type(context) is not int or not 0 <= context < 2**64 or not isinstance(event, str) or event not in events:
            raise ValueError("Scoped evidence requires a valid context and public event")
        if event == "end" and (type(record.get("rows")) is not int or record["rows"] < 0
                               or type(record.get("complete")) is not bool or type(record.get("truncated")) is not bool):
            raise ValueError("Scoped evidence end requires row and completeness fields")
        if event == "definition_validation" and (
                type(record.get("status")) is not int or not 0 <= record["status"] <= 8
                or not isinstance(record.get("reason"), str) or not record["reason"]
                or record.get("evidence") != "ordered_definition_preflight"
                or record.get("mutates_live_state") is not False):
            raise ValueError("Scoped definition validation requires status, reason and a nonmutating proof")
        if event == "transfer_statistics":
            counts = ("fallbacks", "pinned_uploads", "pinned_downloads", "pinned_upload_bytes", "pinned_download_bytes",
                      "packed_bytes", "unpacked_bytes", "tiles", "events", "event_waits", "staging_allocations",
                      "staging_reuses", "slot_capacity", "staging_device_bytes", "process_reserved_bytes", "process_peak_bytes")
            durations = ("packing_seconds", "unpacking_seconds", "event_wait_seconds")
            if (type(record.get("stats_version")) is not int or record["stats_version"] != 1
                    or record.get("complete") is not True
                    or any(type(record.get(field)) is not int or not 0 <= record[field] <= 4
                           for field in ("requested_mode", "effective_mode"))
                    or type(record.get("fallback_reason")) is not int or not 0 <= record["fallback_reason"] <= 5
                    or not isinstance(record.get("reason"), str) or not record["reason"]
                    or any(type(record.get(field)) is not int or not 0 <= record[field] < 2**64 for field in counts)
                    or any(type(record.get(field)) not in (int, float) or record[field] < 0 for field in durations)):
                raise ValueError("Scoped transfer statistics require versioned complete public counters")
        if event == "batch_statistics":
            counts = ("preparation_operations", "batches", "chunk_iterations", "slot_bytes", "upload_bytes",
                      "download_bytes", "prefix_upload_bytes", "prefix_uploads", "launches", "completed_batches",
                      "actual_upload_bytes", "actual_download_bytes", "actual_launches")
            durations = ("baseline_seconds", "pinned_seconds", "pipelined_seconds", "execution_seconds")
            signed_costs = ("estimated_seconds", "terminal_delta_seconds")
            if (type(record.get("stats_version")) is not int or record["stats_version"] != 1
                    or any(type(record.get(field)) is not int or record[field] not in (0, 1)
                           for field in ("available", "applied", "owner_cost_available"))
                    or type(record.get("selected_transfers")) is not int or record["selected_transfers"] not in (0, 1, 2)
                    or any(not isinstance(record.get(field), str) or not record[field]
                           for field in ("reason", "owner_cost_reason"))
                    or any(type(record.get(field)) is not int or not 0 <= record[field] < 2**64 for field in counts)
                    or any(type(record.get(field)) not in (int, float) or record[field] < 0 for field in durations)
                    or any(type(record.get(field)) not in (int, float) for field in signed_costs)
                    or record["completed_batches"] > record["batches"]
                    or (record["applied"] and (not record["available"] or not record["batches"]
                                              or record["completed_batches"] != record["batches"]))):
                raise ValueError("Scoped batch statistics require finalized versioned public counters and costs")
        finite(record)
        records.append(record)
    return records


def scoped_modeled_decisions(log):
    """Read public common-runtime estimates separately from execution evidence.

    These fields correspond to the public planning-decision ABI. Modeled
    durations exclude common native helper compute; they are not wall time.
    A GPU estimate alone never verifies an actual CUDA launch or transfer.
    """
    integer_fields = {"available": 32, "gpu_units": 32, "cpu_units": 32, "candidates": 32,
                      "simulated_operations": 64, "upload_bytes": 64, "download_bytes": 64,
                      "uploads": 64, "downloads": 64, "launches": 64, "waits": 64,
                      "allocations": 64, "peak_device_bytes": 64}
    duration_fields = {"estimated_seconds", "native_seconds"}
    required = (set(integer_fields) - {"uploads", "downloads", "allocations"}) | duration_fields
    records = []
    for match in re.finditer(r"(?<![A-Za-z0-9_])FORT_SCOPED[ \t]+decision[ \t]+([^\r\n]*)", log):
        parts = re.split(r"[ \t]+reason=", match.group(1), maxsplit=1)
        if len(parts) != 2 or not parts[1].strip():
            raise ValueError("Scoped decision trace requires a reason")
        record = {"reason": parts[1].strip()}
        for token in parts[0].split():
            field, separator, raw = token.partition("=")
            if not separator or field in record or field not in integer_fields.keys() | duration_fields:
                raise ValueError("Scoped decision trace contains an unknown or duplicate field")
            if field in integer_fields:
                if not re.fullmatch(r"\d+", raw) or int(raw) >= 2**integer_fields[field]:
                    raise ValueError("Scoped decision count must fit its unsigned public ABI field")
                record[field] = int(raw)
            else:
                try:
                    value = float(raw)
                except ValueError as error:
                    raise ValueError("Scoped decision duration must be finite and nonnegative") from error
                if not math.isfinite(value) or value < 0:
                    raise ValueError("Scoped decision duration must be finite and nonnegative")
                record[field] = value
        if not required <= record.keys() or record["available"] not in (0, 1):
            raise ValueError("Scoped decision trace lacks valid public decision fields")
        record["available"] = bool(record["available"])
        if not record["available"] and record["gpu_units"]:
            raise ValueError("Unavailable scoped estimate claims a GPU schedule")
        records.append(record)
    return records


def gpu_activity(log, tool, regions, *, scope_manifest=None, scope_packages=None):
    if tool == "local":
        kernels = len(re.findall(r"(?<![A-Za-z0-9_])FORT_RUNTIME kernel\b", log))
        upload = sum(map(int, re.findall(r"(?<![A-Za-z0-9_])FORT_RUNTIME upload bytes=(\d+)", log)))
        download = sum(map(int, re.findall(r"(?<![A-Za-z0-9_])FORT_RUNTIME download bytes=(\d+)", log)))
        scoped = scoped_runtime_activity(log)
        manifests = _scope_manifests(scope_manifest, scope_packages)
        scoped["source_provenance"] = scoped_provenance(log, manifests)
        native = bool(manifests)
        for manifest in manifests:
            scopes = manifest.get("scopes", [])
            native = native and (manifest.get("schema_version") == 1
                      and (manifest.get("scope_count") == 0 or
                           manifest.get("automatic_estimate_available") is False and bool(scopes)
                           and all(scope.get("mode") == "auto" and scope.get("estimate_available") is False
                                   for scope in scopes)))
        if native and (kernels or scoped["kernel_launches"]):
            raise ValueError("CUDA launch contradicts the compiler's all-native source-scope contract")
        return {"verified": kernels + scoped["kernel_launches"] > 0,
                "kernel_launches": kernels + scoped["kernel_launches"], "probe_kernel_launches": 0,
                "upload_bytes": upload + scoped["upload_bytes"], "download_bytes": download + scoped["download_bytes"],
                "allocations": len(re.findall(r"(?<![A-Za-z0-9_])FORT_RUNTIME alloc\b", log)) + scoped["allocations"],
                "scoped_runtime": scoped, "compiler_native_selection": native,
                "scope_manifest_count": len(manifests),
                "native_selection_evidence": ("versioned compiler manifest; transformed-call coverage unverified"
                                               if native else None),
                "source": "successful local CUDA runtime events",
                "offload_decisions": offload_decisions(log, regions, scope_manifest=scope_manifest,
                                                      scope_packages=scope_packages)}
    raise ValueError("GPU trace tool must be local")
