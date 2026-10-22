#!/usr/bin/env python3
"""Fresh native ELMM / committed compiler / reworked compiler comparison.

Every mode is built from the pinned approved common application sources. Compiler snapshots,
build receipts, binaries, configuration, logs and validation outputs are hashed.
Untraced complete-process timings and traced/full-storage diagnostics are separate.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import io
import json
import math
import os
import re
import shutil
import sqlite3
import statistics
import subprocess
import sys
import tarfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKSPACE = Path(os.environ.get("WORKSPACE_ROOT", HERE.parent)).resolve()
PIPELINE = Path(os.environ.get("PIPELINE_ROOT", WORKSPACE / "elmm-pipeline")).resolve()
COMPILER = Path(os.environ.get("COMPILER_ROOT", WORKSPACE)).resolve()
sys.path.insert(0, str(HERE))
from application_basis import prepare_application_basis
from sgs_experiment import (
    CASE_NAMES,
    DEFAULT_CONTRACT,
    application_paths,
    compare_outputs,
    contract_path,
    digest,
    read,
    read_outputs,
    repository_provenance,
    select_configuration,
    verify_contract,
    write,
)

MODES = ("native", "committed", "reworked")
REQUIRED_FIELDS = ("U", "V", "W", "Pr", "Viscosity", "U2", "V2", "W2", "Ustar", "Vstar", "Wstar")
OPTIONAL_FIELDS = ("Temperature", "Moisture", "Scalar", "TDiff", "U_avg", "V_avg", "W_avg", "Pr_avg", "Scalar_avg")
FIELD_POLICY = {"dtype": "float64", "atol": 1e-6, "rtol": 1e-5,
                "metadata_format": "storage-shape-bounds-v1"}


def compiler_code_equivalence(root):
    """Separate source identity from historical, immutable snapshot identities."""
    root = Path(root)
    hashes = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if (relative.parts[0] == "_local" or path.suffix == ".pyc"
                or any(part in {".git", "__pycache__", ".pytest_cache", ".ruff_cache"}
                       for part in relative.parts)):
            continue
        if path.is_file():
            hashes[str(relative)] = digest(path)
    return {"schema_version": 1,
            "excluded": ["_local/**", "**/.git/**", "**/__pycache__/**", "**/*.pyc",
                         "**/.pytest_cache/**", "**/.ruff_cache/**"],
            "sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "files_sha256": hashes}


def snapshot_compiler(root, destination, *, revision=None):
    """Archive the requested commit exactly; freeze candidate files separately."""
    from elmm_pipeline.common import digest_tree
    root, destination = Path(root).resolve(strict=True), Path(destination)
    if destination.exists():
        raise ValueError("compiler snapshot already exists: " + str(destination))
    if revision is not None:
        commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "--verify", revision + "^{commit}"], text=True).strip()
        archive = subprocess.check_output(["git", "-C", str(root), "archive", commit, "compiler"])
        destination.mkdir(parents=True)
        with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
            stream.extractall(destination, filter="data")
    else:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip() or None
        destination.mkdir(parents=True)
        def ignore(directory, names):
            excluded = {name for name in names if name in {"__pycache__", ".pytest_cache", ".ruff_cache"}
                        or name.endswith(".pyc")}
            if Path(directory) == root / "compiler" and "_local" in names:
                excluded.add("_local")
            return excluded
        shutil.copytree(root / "compiler", destination / "compiler", ignore=ignore)
    if not (destination / "compiler/__main__.py").is_file():
        raise ValueError("snapshot contains no compiler CLI")
    return {"root": str(destination.resolve()), "source_root": str(root), "base_commit": commit,
            "exact_committed_snapshot": revision is not None,
            "compiler_tree_sha256": digest_tree(destination / "compiler"),
            "code_equivalence": compiler_code_equivalence(destination / "compiler")}


def rotated_order(round_number):
    offset = round_number % len(MODES)
    return MODES[offset:] + MODES[:offset]


def run_directory(configuration, destination):
    """Remove old output copied with a custom configuration before executing."""
    shutil.copytree(configuration, destination)
    if (destination / "output").exists():
        shutil.rmtree(destination / "output")
    return destination


def prepare_workloads(source, output, *, advection=None, steps=None, timing_steps=None):
    """Performance caps never replace the natural correctness configuration."""
    from elmm_pipeline.configuration import prepare_config
    if steps is not None and timing_steps is not None:
        raise ValueError("choose an all-phase smoke cap or a performance-only cap")
    correctness = prepare_config(source, output / "config", steps=steps, advection=advection)
    timing_dir = output / "timing-config" if timing_steps is not None else output / "config"
    timing = (prepare_config(source, timing_dir, steps=timing_steps, advection=advection)
              if timing_steps is not None else correctness)
    return correctness, timing, timing_dir


def cuda_api_activity(path):
    """Keep CPU CUDA API durations distinct from overlapping device activity."""
    result = {}
    with sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "StringIds" not in tables:
            return result
        for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
            if table not in tables:
                continue
            columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
            if not {"nameId", "start", "end"}.issubset(columns):
                continue
            result[table] = [{"name": name, "calls": count, "cpu_api_seconds": duration / 1e9}
                for name, count, duration in connection.execute(
                    f'SELECT s.value, COUNT(*), SUM(a."end"-a.start) FROM {table} a '
                    'LEFT JOIN StringIds s ON s.id=a.nameId GROUP BY a.nameId '
                    'ORDER BY SUM(a."end"-a.start) DESC')]
    return result


def node_profile_transport(nsys, directory):
    """Ask Nsight for actual graph node activity, independently of wall timing."""
    profiler = shutil.which(str(nsys or "nsys"))
    if profiler is None and nsys is None and Path("/usr/local/cuda/bin/nsys").is_file():
        profiler = "/usr/local/cuda/bin/nsys"
    if profiler is None:
        return nsys, None
    profiler = str(Path(profiler).resolve(strict=True))
    directory.mkdir(parents=True)
    wrapper = directory / "nsys-node-profile"
    wrapper.write_text("#!/usr/bin/env python3\nimport os, sys\n"
        f"profiler = {profiler!r}\n"
        "arguments = sys.argv[1:]\n"
        "if arguments[:1] == ['profile']:\n"
        "    arguments.insert(1, '--cuda-graph-trace=node')\n"
        "os.execv(profiler, [profiler, *arguments])\n")
    wrapper.chmod(0o755)
    return str(wrapper), {"profiler": profiler, "profiler_sha256": digest(profiler),
                         "wrapper": str(wrapper), "wrapper_sha256": digest(wrapper), "cuda_graph_trace": "node"}


def policy_trace_aliases(log, scope_packages):
    """Authenticate compiler-local policy labels without inventing source IDs."""
    if "FORT_OFFLOAD " not in log:
        return {}
    aliases = {}
    for package in scope_packages or []:
        if not package.get("artifact_root"):
            continue
        for name, expected in package["scope_manifest"].get("artifacts_sha256", {}).items():
            if Path(name).suffix != ".cu":
                continue
            path = package_artifact(package["artifact_root"], name)
            if digest(path) != expected:
                raise ValueError("policy trace artifact changed: " + name)
            for alias in set(re.findall(r'\boffload::decision_trace\(\s*"([A-Za-z][A-Za-z0-9_]*)"', path.read_text())):
                aliases.setdefault(alias.lower(), []).append({"path": str(path), "sha256": expected})
    return aliases


def observed_gpu_activity(log, *, scope_manifest=None, scope_packages=None):
    """Transport completion, graph and queue counters beside pipeline evidence.

    The independent pipeline currently accepts a closed set of schema-1 events.
    Validate these additive runtime events explicitly rather than silently losing
    its statistics or weakening validation of the existing event set.
    """
    from elmm_pipeline.gpu_validation import gpu_activity
    counters = {
        "completion_statistics": ("event_records", "event_queries", "event_waits", "stream_waits", "created_events",
                                  "reused_events", "independent_host_tasks"),
        "graph_statistics": ("creations", "instantiations", "rebuilds", "submissions", "replays", "parameter_updates",
                             "logical_kernels", "live_graphs", "destructions"),
        "command_queue_statistics": ("accepted_tasks", "submissions", "conflict_flushes", "explicit_flushes",
                                     "pending_tasks", "peak_pending_tasks", "copied_argument_bytes", "direct_submissions",
                                     "direct_kernels", "graph_resource_fallbacks"),
        "residency_statistics": ("device_budget", "reserve_batches", "requested_resources", "reuse_hits", "evictions",
                                 "evicted_bytes", "dirty_evictions", "dirty_download_bytes", "pressure_flushes"),
    }
    retained = {event: [] for event in counters}
    compact_transfers = []
    native_hints = []

    def object_fields(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("runtime evidence JSON contains a duplicate field")
            result[key] = value
        return result

    def constant(_value):
        raise ValueError("runtime evidence JSON must contain finite numbers")

    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("runtime evidence JSON must contain finite numbers")
        if isinstance(value, dict):
            for child in value.values():
                finite(child)
        elif isinstance(value, list):
            for child in value:
                finite(child)

    def retain_native_hint(match):
        if not match[1] or match[1][0] not in " \t":
            raise ValueError("malformed native hint record separator")
        record = json.loads(match[1], object_pairs_hook=object_fields, parse_constant=constant)
        finite(record)
        required = {"schema_version", "event", "phase", "reason", "hit", "observation_eligible",
                    "learned", "estimates_available", "key_words", "key_hash"}
        if (not isinstance(record, dict) or set(record) != required
                or type(record.get("schema_version")) is not int or record["schema_version"] != 1
                or record.get("event") != "native_hint"
                or record.get("reason") != "conservative_native_owner_hint"
                or record.get("phase") not in ("lookup", "finish")
                or any(type(record.get(name)) is not bool
                       for name in ("hit", "observation_eligible", "learned", "estimates_available"))
                or record["estimates_available"] is not False
                or type(record.get("key_words")) is not int or not 1 <= record["key_words"] <= 512
                or type(record.get("key_hash")) is not int or not 0 <= record["key_hash"] < 2**64
                or (record["phase"] == "lookup"
                    and (record["learned"] or record["observation_eligible"] == record["hit"]))
                or (record["phase"] == "finish" and (record["hit"] or record["observation_eligible"]))):
            raise ValueError("versioned contextless native hint evidence required")
        native_hints.append(record)
        return ""

    def retain(match):
        record = json.loads(match[1], object_pairs_hook=object_fields, parse_constant=constant)
        finite(record)
        event = record.get("event") if isinstance(record, dict) else None
        if event == "compact_transfer":
            numbers = ("context", "identity", "bytes", "operations", "kernels", "word_bytes", "scratch_capacity")
            if (type(record.get("schema_version")) is not int or record["schema_version"] != 1
                    or record.get("complete") is not True
                    or any(type(record.get(name)) is not int or not 0 < record[name] < 2**64 for name in numbers)
                    or record.get("direction") not in ("h2d", "d2h")
                    or record["word_bytes"] not in (1, 2, 4, 8)
                    or record["bytes"] % record["word_bytes"] != 0
                    or record["bytes"] > record["scratch_capacity"]
                    or record["kernels"] != (record["operations"] + 7) // 8):
                raise ValueError("versioned complete compact transfer evidence required")
            compact_transfers.append(record)
            return ""
        if event not in counters:
            return match[0]
        if (type(record.get("schema_version")) is not int or record["schema_version"] != 1
                or type(record.get("stats_version")) is not int or record["stats_version"] != 1
                or record.get("complete") is not True
                or any(type(record.get(name)) is not int or not 0 <= record[name] < 2**64
                       for name in ("context", *counters[event]))
                or (event == "command_queue_statistics" and any(type(record.get(name)) is not int
                    or record[name] not in (0, 1) for name in ("queue_enabled", "graphs_enabled")))):
            raise ValueError("versioned complete runtime counters required: " + event)
        retained[event].append(record)
        return ""

    compatible_log = re.sub(r"(?<![A-Za-z0-9_])FORT_SCOPED[ \t]+native_hint\b([^\r\n]*)",
                            retain_native_hint, log)
    compatible_log = re.sub(r"FORT_SCOPED evidence ([^\r\n]*)", retain, compatible_log)
    aliases = policy_trace_aliases(log, scope_packages)
    activity = gpu_activity(compatible_log, "local", [{"entry": alias} for alias in aliases], scope_manifest=scope_manifest,
                            **({"scope_packages": scope_packages} if scope_packages is not None else {}))
    if aliases:
        activity["policy_trace_aliases"] = {"semantics": "Literal policy labels in authenticated generated CUDA; source attribution uses versioned positions.",
                                             "artifacts": aliases}
    activity.update(retained)
    activity["compact_transfer_events"] = compact_transfers
    activity["native_hint_events"] = native_hints
    activity["native_hint_counts"] = {
        "events": len(native_hints),
        "lookups": sum(row["phase"] == "lookup" for row in native_hints),
        "lookup_hits": sum(row["phase"] == "lookup" and row["hit"] for row in native_hints),
        "observation_eligible": sum(row["observation_eligible"] for row in native_hints),
        "finishes": sum(row["phase"] == "finish" for row in native_hints),
        "learned": sum(row["learned"] for row in native_hints),
    }
    manifests = ([scope_manifest] if scope_manifest is not None else []) + [
        package["scope_manifest"] for package in scope_packages or []]
    graph_events = graph_submission_evidence(log, manifests, retained)
    activity["graph_submission_evidence"] = graph_events
    activity["runtime_cuda_submissions"] = activity.get("kernel_launches", 0) + len(graph_events)
    activity["runtime_cuda_numerical_kernels"] = activity.get("kernel_launches", 0) + sum(
        event["logical_kernels"] for event in graph_events)
    activity["runtime_cuda_carrier_kernels"] = sum(event["kernels"] for event in compact_transfers)
    activity["runtime_cuda_total_kernels"] = activity["runtime_cuda_numerical_kernels"] + activity["runtime_cuda_carrier_kernels"]
    activity["runtime_cuda_total_submissions"] = activity["runtime_cuda_submissions"] + activity["runtime_cuda_carrier_kernels"]
    if graph_events and activity.get("compiler_native_selection"):
        raise ValueError("CUDA graph submission contradicts an all-native compiler manifest")
    activity["verified"] = activity.get("verified", False) or bool(graph_events)
    return activity


def runtime_activity_summary(activity, reference):
    """Keep observed counters separate from concise modeled decision summaries."""
    scoped = activity.get("scoped_runtime", {})
    decisions = scoped.get("modeled_decisions", [])
    evidence = scoped.get("modeled_evidence", [])
    counts = Counter((row.get("reason"), row.get("available"), row.get("gpu_units"), row.get("cpu_units"))
                     for row in decisions)
    decision_summary = {"calls": len(decisions), "units_are_modeled": True,
        "selections": [{"reason": key[0], "available": key[1], "gpu_units": key[2], "cpu_units": key[3], "occurrences": count}
                       for key, count in counts.items()]}
    evidence_summary = {"records": len(evidence), "events": dict(Counter(row["event"] for row in evidence)),
        "incomplete_end_records": sum(row["event"] == "end" and row.get("complete") is not True for row in evidence),
        "definition_declines": [{"status": key[0], "reason": key[1], "occurrences": count}
            for key, count in Counter((row["status"], row["reason"]) for row in evidence
                if row["event"] == "definition_validation" and row["status"] != 0).items()]}
    result = {**activity, "runtime_activity_reference": reference,
        "scoped_runtime": {**{key: value for key, value in scoped.items() if key not in ("modeled_decisions", "modeled_evidence")},
                           "modeled_decision_summary": decision_summary, "modeled_evidence_summary": evidence_summary}}
    if "compact_transfer_events" in activity:
        transfers = result.pop("compact_transfer_events")
        result["compact_transfer_summary"] = {"events": len(transfers), "directions": {
            direction: {"events": sum(row["direction"] == direction for row in transfers),
                        **{name: sum(row[name] for row in transfers if row["direction"] == direction)
                           for name in ("bytes", "operations", "kernels")}}
            for direction in ("h2d", "d2h")},
            "peak_scratch_capacity": max((row["scratch_capacity"] for row in transfers), default=0),
            "semantics": "Completed physical carrier kernels, separate from numerical graph/direct kernels; full events retained in authenticated sidecar."}
    if "native_hint_events" in activity:
        result.pop("native_hint_events")
        result["native_hint_semantics"] = (
            "Contextless native-owner hint metadata only; no estimated costs, placement decisions, "
            "native execution, CUDA activity or performance improvement inferred. Full events retained in authenticated sidecar.")
    if activity.get("native_auto_selection_evidence"):
        native = activity["native_auto_selection_evidence"]
        result["native_auto_selection_evidence"] = {
            **{key: value for key, value in native.items() if key not in ("modeled_native_decisions", "preexecution_declines")},
            "modeled_native_decision_count": len(native.get("modeled_native_decisions", [])),
            "preexecution_decline_count": len(native.get("preexecution_declines", []))}
    return result


def runtime_activity_reference(activity, directory):
    """Publish complete parsed evidence once; avoid duplicating it each round."""
    path = Path(directory) / "gpu-activity.json"
    if path.exists():
        raise ValueError("runtime activity sidecar already exists: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    write(path, activity)
    reference = {"schema_version": 1, "kind": "complete-runtime-gpu-activity",
                 "path": str(path.resolve()), "sha256": digest(path)}
    return runtime_activity_summary(activity, reference)


def load_runtime_activity(activity):
    """Authenticate both complete evidence and its derived report summary."""
    reference = activity.get("runtime_activity_reference")
    if reference is None:
        return activity
    if (not isinstance(reference, dict) or reference.get("schema_version") != 1 or reference.get("kind") != "complete-runtime-gpu-activity"
            or digest(reference["path"]) != reference["sha256"]):
        raise ValueError("runtime activity sidecar changed")
    complete = read(reference["path"])
    if not isinstance(complete, dict) or runtime_activity_summary(complete, reference) != activity:
        raise ValueError("runtime activity summary differs from its authenticated sidecar")
    return complete


def graph_submission_evidence(log, manifests, counters):
    """Validate successful graph submissions without inventing direct launches."""
    fields = ("owner", "procedure", "segment", "operation", "implementation", "boundary")
    operation_kinds = {"source_call", "borrowed_call", "numerical_segment", "native_operation"}
    registry = {record["id"]: record for manifest in manifests
                for record in manifest.get("runtime_provenance", {}).get("records", [])}
    graph_events, direct_counts = [], {}
    for line in log.splitlines():
        match = re.match(r"^FORT_SCOPED (graph_submit|launch) (.*)$", line)
        if not match:
            continue
        encoded = {}
        for token in match[2].split():
            key, separator, value = token.partition("=")
            if not separator or key in encoded:
                raise ValueError("malformed CUDA submission evidence")
            encoded[key] = value
        if match[1] == "launch":
            if "context" in encoded:
                context = int(encoded["context"])
                direct_counts[context] = direct_counts.get(context, 0) + 1
            continue
        if (encoded.get("provenance_version") != "1"
                or not re.fullmatch(r"[1-9][0-9]*", encoded.get("context", ""))
                or not re.fullmatch(r"[1-9][0-9]*", encoded.get("bytes", ""))):
            raise ValueError("versioned positive CUDA graph submission evidence required")
        for field in fields:
            identity = encoded.get(field)
            if identity == "unknown":
                continue
            record = registry.get(identity)
            if (record is None or (record["kind"] not in operation_kinds if field == "operation"
                                   else record["kind"] != field)):
                raise ValueError("CUDA graph submission identity absent from source manifest: " + field)
        graph_events.append({"context": int(encoded["context"]), "logical_kernels": int(encoded["bytes"]),
                             **{field: encoded[field] for field in fields}})
    by_context = {}
    for event in graph_events:
        context = by_context.setdefault(event["context"], {"submissions": 0, "logical_kernels": 0})
        context["submissions"] += 1
        context["logical_kernels"] += event["logical_kernels"]
    seen = set()
    for record in counters["graph_statistics"]:
        context = record["context"]
        if context in seen:
            raise ValueError("duplicate completed graph statistics for one context")
        seen.add(context)
        observed = by_context.get(context, {"submissions": 0, "logical_kernels": 0})
        if any(record[name] != observed[name] for name in observed):
            raise ValueError("graph statistics disagree with successful graph submission events")
    if set(by_context) - seen:
        raise ValueError("CUDA graph submissions lack completed graph statistics")
    seen.clear()
    for record in counters["command_queue_statistics"]:
        context = record["context"]
        if context in seen or record["pending_tasks"]:
            raise ValueError("command queue statistics must describe one completed queue")
        seen.add(context)
        if record["direct_kernels"] != direct_counts.get(context, 0):
            raise ValueError("queue direct kernels disagree with successful direct launch events")
    return graph_events


def summarize(runs):
    result = {}
    for mode in MODES:
        rows = [row for row in runs if row["mode"] == mode and not row["warmup"]]
        if not rows or any(row["traced"] or row.get("full_storage") for row in rows):
            raise ValueError("only untraced production binaries may enter timing statistics")
        samples = [row["wall_seconds"] for row in rows]
        if any(not math.isfinite(value) or value <= 0 for value in samples):
            raise ValueError("positive finite complete wall measurements required")
        result[mode] = {"median_seconds": statistics.median(samples), "minimum_seconds": min(samples),
                        "maximum_seconds": max(samples), "samples_seconds": samples, "repetitions": len(samples)}
        for source, target in (("time_steps_seconds", "median_loop_seconds"), ("poisson_seconds", "median_poisson_seconds")):
            values = [row["timing"][source] for row in rows if source in row.get("timing", {})]
            result[mode][target] = statistics.median(values) if values else None
    for row in result.values():
        row["ratio_to_native"] = row["median_seconds"] / result["native"]["median_seconds"]
        row["speedup_over_native"] = 1 / row["ratio_to_native"]
        row["wall_reduction_percent"] = 100 * (1 - row["ratio_to_native"])
        row["speedup_over_committed"] = result["committed"]["median_seconds"] / row["median_seconds"]
    return result


def dump_block(specifications, imports=()):
    lines = ["  block", *["    " + line for line in imports], "    integer :: comparison_unit"]
    for label, expression, required in specifications:
        if required:
            lines.append(f'    if (.not.allocated({expression})) error stop "missing comparison field {label}"')
        lines.extend([f"    if (allocated({expression})) then",
            f"      open(newunit=comparison_unit,file='fields-{label}.bin', &",
            "           access='stream',form='unformatted',status='new')",
            f"      write(comparison_unit) {expression}", "      close(comparison_unit)",
            f"      open(newunit=comparison_unit,file='fields-{label}.shape',status='new')",
            f"      write(comparison_unit,*) storage_size({expression}), shape({expression}), &",
            f"           lbound({expression}), ubound({expression})", "      close(comparison_unit)", "    end if"])
    return "\n".join([*lines, "  end block", ""])


def diagnostic_main(text):
    """Observe native precision and original bounds immediately before retirement."""
    first, second = "  call Dynamics_Deallocate\n", "  call DeallocateGlobals\n"
    if text.count(first) != 1 or text.count(second) != 1:
        raise ValueError("unique original application lifetime boundaries required")
    intermediate = REQUIRED_FIELDS[5:]
    before = [(name, name, True) for name in REQUIRED_FIELDS[:5]]
    before += [(name, "comparison_" + name, True) for name in intermediate]
    before += [(name, name, False) for name in OPTIONAL_FIELDS[:4]]
    imports = [f"use Dynamics, only: comparison_{name} => {name}" for name in intermediate]
    averages = OPTIONAL_FIELDS[4:]
    text = text.replace(first, dump_block(before, imports) + first)
    return text.replace(second, dump_block([(name, name, False) for name in averages],
                        ["use OutputData, only: " + ", ".join(averages)]) + second)


def relink_diagnostic(binary, source, directory):
    """Replace only the main object in the exact freshly executed build recipes."""
    from elmm_pipeline.build_receipts import (
        CommandRecorder,
        read_receipts,
        require_build_receipts,
    )
    from elmm_pipeline.common import COMPILER_PHASE_TIMEOUT_SECONDS
    directory.mkdir(parents=True)
    receipt_path = binary.parent / "build-receipts.json"
    receipt = require_build_receipts(receipt_path)
    units = [row for row in receipt["units"] if row["source"] == str(source)]
    if len(units) != 1:
        raise ValueError("diagnostic main must select exactly one fresh main object")
    unit = units[0]
    main, obj, executable = directory / "main.f90", directory / "main.o", directory / "ELMM"
    main.write_text(diagnostic_main(source.read_text()))
    recorder = CommandRecorder(directory / "commands")
    plans = []
    for original, old, new in ((unit["execution"], source, main), (receipt["link"], Path(unit["object"]), obj)):
        argv = list(original["argv"])
        cwd = Path(original["cwd"])
        indices = [i for i, value in enumerate(argv) if not value.startswith("-") and (cwd / value).resolve() == old.resolve()]
        if len(indices) != 1 or argv.count("-o") != 1:
            raise ValueError("unambiguous original source/object and output operands required")
        argv[indices[0]] = str(new)
        argv[argv.index("-o") + 1] = str(obj if original is unit["execution"] else executable)
        env = dict(os.environ)
        for name, value in original["environment"].items():
            if value is None:
                env.pop(name, None)
            else:
                env[name] = value
        with (directory / ("compile.log" if original is unit["execution"] else "link.log")).open("w") as log:
            result = recorder.run(argv, cwd=cwd, env=env, stdout=log, stderr=log,
                                  additional_inputs=sorted(binary.parent.rglob("*.mod")),
                                  timeout=COMPILER_PHASE_TIMEOUT_SECONDS)
        plans.append({"argv": argv, "cwd": str(cwd), "returncode": result.returncode})
        write(directory / "plan.json", plans)
        if result.returncode:
            raise RuntimeError("full-storage diagnostic build failed: " + str(directory))
    read_receipts(directory / "commands")
    require_build_receipts(receipt_path)
    return {"binary": str(executable), "sha256": digest(executable), "source": str(main),
            "source_sha256": digest(main), "original_build_receipt": str(receipt_path),
            "original_build_receipt_sha256": digest(receipt_path), "production_binary_modified": False}


def selection_request(contract, custom=None, *, scope_execution="bounded"):
    request = {"source": contract["source"], "entry": contract["entry"],
               "source_files": [name for name in contract["source_sha256"] if name != contract["source"]],
               "capture_facts": contract["capture_facts"],
               "options": {"gpu_policy": "sections", "scope_transfers": "direct", "scope_execution": scope_execution}}
    if custom is not None:
        if not isinstance(custom, dict) or custom.get("schema_version") != 1 or not all(name in custom for name in ("source", "entry", "capture_facts")):
            raise ValueError("candidate request requires schema_version 1, source, entry and capture_facts")
        request = {**request, **custom, "source_files": custom.get("source_files", []),
                   "options": {**request["options"], **custom.get("options", {})}}
    return request


def selection_requests(contract, custom=None, *, scope_execution="bounded"):
    """Normalize one root or several independently reviewed pristine roots."""
    candidates = custom if isinstance(custom, list) else [custom]
    if not candidates:
        raise ValueError("candidate request array cannot be empty")
    requests = [selection_request(contract, item, scope_execution=scope_execution) for item in candidates]
    identities = [(item["source"], item["entry"]) for item in requests]
    if len(set(identities)) != len(identities):
        raise ValueError("candidate request roots cannot repeat")
    return requests


def candidate_calibration(requests, profile_path, *, threads):
    """Bind one external hardware profile to candidate roots only."""
    profiles = []
    for request in requests:
        selected = request["options"].get("calibration_profile")
        if selected is not None:
            if not isinstance(selected, (str, os.PathLike)):
                raise ValueError("candidate root calibration_profile must be a file path")
            profiles.append(Path(selected).resolve(strict=True))
    if profile_path is not None:
        profiles.append(Path(profile_path).resolve(strict=True))
    if not profiles:
        return requests, None
    if len(set(profiles)) != 1:
        raise ValueError("candidate roots have conflicting calibration profiles")
    path = profiles[0]
    if not path.is_file():
        raise ValueError("candidate calibration must be a JSON file")
    document = read(path)
    scoped = document.get("scoped", {}) if isinstance(document, dict) else {}
    if (not isinstance(document, dict) or document.get("schema_version") != 1 or document.get("cpu_threads") != threads
            or document.get("precision_bits") != 64 or not isinstance(document.get("hardware"), dict)
            or not isinstance(document.get("toolchain"), dict) or not isinstance(scoped, dict) or scoped.get("schema_version") != 1
            or not re.fullmatch(r"[0-9a-f]{64}", str(scoped.get("runtime_id", "")))
            or not isinstance(document.get("numerical_contract"), dict)):
        raise ValueError("candidate calibration requires matching float64/thread, hardware and scoped runtime bindings")
    requests = [{**request, "options": {**request["options"], "calibration_profile": str(path)}} for request in requests]
    identity = {"path": str(path), "sha256": digest(path), "candidate_only": True,
                **{key: document[key] for key in ("hardware", "toolchain", "precision_bits", "cpu_threads", "numerical_contract")},
                "scoped_runtime_id": scoped["runtime_id"], "scoped_schema_version": scoped["schema_version"]}
    return requests, identity


def calibration_hardware_binding(identity, toolchain):
    """Check the calibrated GPU UUID against a fresh read-only device query."""
    executable = toolchain["device"]["argv"][0]
    argv = [executable, "--query-gpu=uuid,name,compute_cap", "--format=csv,noheader,nounits"]
    observed = subprocess.check_output(argv, text=True, timeout=10)
    devices = [{"gpu_uuid": row[0].strip(), "gpu_name": row[1].strip(), "compute_capability": row[2].strip()}
               for row in csv.reader(observed.splitlines()) if len(row) == 3]
    wanted = identity["hardware"]
    if not any(all(device[key] == wanted.get(key) for key in device) for device in devices):
        raise ValueError("candidate calibration GPU UUID/name/capability differs from current hardware")
    cpuinfo = Path("/proc/cpuinfo")
    cpu_name = next((line.partition(":")[2].strip() for line in cpuinfo.read_text().splitlines()
                     if line.startswith("model name")), None) if cpuinfo.is_file() else None
    machine = os.uname().machine
    if cpu_name != wanted.get("cpu_name") or machine != wanted.get("machine"):
        raise ValueError("candidate calibration CPU/machine differs from current hardware")
    return {"argv": argv, "devices": devices, "matched_profile_hardware": wanted,
            "cpu_name": cpu_name, "machine": machine,
            "profile_sha256": identity["sha256"], "scoped_runtime_id": identity["scoped_runtime_id"]}


def native_auto_selection(activity, requests):
    """Accept demonstrated all-native AUTO decisions without claiming GPU work."""
    if (activity.get("runtime_cuda_submissions", 0)
            or not requests or any(request["options"].get("gpu_policy") != "auto" for request in requests)):
        return None
    scoped = activity.get("scoped_runtime", {})
    decisions = [row for row in scoped.get("modeled_decisions", [])
                 if row.get("gpu_units") == 0 and row.get("cpu_units", 0) > 0]
    declines = [row for row in scoped.get("modeled_evidence", [])
                if row.get("event") == "definition_validation" and row.get("status", 0) != 0]
    policy = activity.get("offload_decisions", {})
    native_calls = policy.get("modes", {}).get("native", 0) if policy.get("gpu_units", 0) == 0 else 0
    if not decisions and not declines and not native_calls:
        return None
    return {"policy": "explicit generic AUTO", "modeled_native_decisions": decisions,
            "preexecution_declines": declines, "native_policy_calls": native_calls,
            "gpu_execution_observed": False, "offloading_performance_achievement": False}


def package_artifact(root, name):
    root = Path(root).resolve(strict=True)
    path = (root / name).resolve(strict=True)
    if Path(name).is_absolute() or not path.is_relative_to(root) or not path.is_file():
        raise ValueError("compiler artifact must stay inside its package: " + str(name))
    return path


def verify_composed_sources(packages, root, destinations):
    """Authenticate the compiler's edit union before changing any source.

    Component execution proofs remain attached to their original manifests.
    The aggregate authorizes only merged source bytes and shared build inputs.
    """
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans
    root = Path(root)
    manifest = read(root / "scope-manifest.json")
    components = [item["scope_manifest"] for item in packages]
    if manifest.get("schema_version") != 1 or manifest.get("component_manifests") != components:
        raise ValueError("composition must preserve every original component manifest")
    source_artifacts = {item["replacement"] for component in components for item in component["sources"].values()}
    inputs, artifacts, units, replacements = {}, {}, [], {}
    for package, component in zip(packages, components):
        validate_checked_execution_plans(component)
        component_root = Path(package["artifact_root"])
        if read(component_root / "scope-manifest.json") != component:
            raise ValueError("saved component manifest changed")
        for name, expected in component["artifacts_sha256"].items():
            if digest(package_artifact(component_root, name)) != expected:
                raise ValueError("component artifact changed: " + name)
            # Complete original-source replacements may differ; the compiler
            # alone merges their authenticated edit ledgers.
            if name in artifacts and artifacts[name] != expected and name not in source_artifacts:
                raise ValueError("component generated artifact collision: " + name)
            artifacts[name] = expected
        for name, expected in component["source_inputs"].items():
            if name in inputs and inputs[name] != expected:
                raise ValueError("component original input identities disagree")
            inputs[name] = expected
        dependencies = (component.get("analysis_sources") or {}).get("dependencies", {})
        for name, expected in {**component["source_inputs"], **dependencies}.items():
            if digest(destinations.get(name, name)) != expected:
                raise ValueError("original component source/dependency changed: " + name)
        for name, item in component["sources"].items():
            if name in replacements and replacements[name] != item:
                raise ValueError("component replacement identities disagree")
            replacements[name] = item
        for item in component["build_sources"]:
            if item not in units:
                units.append(item)
    if (manifest.get("source_inputs") != inputs or manifest.get("sources") != replacements
            or manifest.get("build_sources") != units
            or any(component.get("runtime") != manifest.get("runtime") for component in components)
            or set(manifest.get("artifacts_sha256", {})) != set(artifacts)):
        raise ValueError("aggregate source/build identities differ from its checked components")
    replacement_paths = {item["replacement"] for item in replacements.values()}
    for name, expected in manifest["artifacts_sha256"].items():
        if digest(package_artifact(root, name)) != expected:
            raise ValueError("composed artifact changed: " + name)
        if name not in replacement_paths and artifacts[name] != expected:
            raise ValueError("composition changed a component generated artifact: " + name)
    prepared = []
    for name, item in replacements.items():
        if name not in destinations:
            raise ValueError("no application destination for composed source: " + name)
        prepared.append((Path(destinations[name]), package_artifact(root, item["replacement"]).read_bytes()))
    return manifest, prepared


def compose_generated_roots(packages, compiler_root, python, output, destinations):
    from elmm_pipeline.common import command, python_environment
    output.mkdir(parents=True)
    argv = [str(python), "-m", "compiler.scopes.compose", *[item["artifact_root"] for item in packages],
            "--output-dir", str(output)]
    command(argv, cwd=compiler_root, env=python_environment(), log=output / "compose.log", timeout=240)
    manifest, prepared = verify_composed_sources(packages, output, destinations)
    for destination, content in prepared:
        destination.write_bytes(content)
    units = manifest["build_sources"]
    return {"command": argv, "artifact_root": str(output), "scope_manifest": manifest,
            "scope_packages": packages, "changed_sources": [str(path) for path, _ in prepared],
            "extra_sources": [str(package_artifact(output, item["path"])) for item in units
                              if item["language"] == "fortran" and item["role"] != "original_source"],
            "gpu_sources": [{"source": str(package_artifact(output, item["path"])), "language": "cuda", "tool": "local",
                             **({"numerical_contract": item["numerical_contract"]} if "numerical_contract" in item else {})}
                            for item in units if item["language"] == "cuda"]}


def generation_references(generation):
    """Transport proof packages by authenticated sidecar without duplicating them."""
    def reference(package):
        result = {key: value for key, value in package.items() if key not in ("scope_manifest", "scope_packages")}
        manifest = Path(package["artifact_root"]) / "scope-manifest.json"
        identity = {"path": str(manifest), "sha256": digest(manifest)}
        previous = package.get("scope_manifest_reference")
        if previous is not None and previous != identity:
            raise ValueError("scope manifest sidecar changed")
        result["scope_manifest_reference"] = identity
        if "scope_packages" in package:
            result["scope_packages"] = [reference(component) for component in package["scope_packages"]]
        return result
    return {mode: reference(package) for mode, package in generation.items()}


def load_generation(generation):
    """Restore original authorities only after their sidecar identities match."""
    def load_package(package):
        result = dict(package)
        if "scope_manifest_reference" in package:
            identity = package["scope_manifest_reference"]
            manifest = Path(package["artifact_root"]) / "scope-manifest.json"
            if (Path(identity["path"]).resolve() != manifest.resolve()
                    or digest(manifest) != identity["sha256"]):
                raise ValueError("scope manifest sidecar identity differs")
            authenticated = read(manifest)
            if "scope_manifest" in package and package["scope_manifest"] != authenticated:
                raise ValueError("embedded scope manifest differs from its authenticated sidecar")
            result["scope_manifest"] = authenticated
        elif "scope_manifest" not in package:
            raise ValueError("missing scope manifest authority")
        if "scope_packages" in package:
            result["scope_packages"] = [load_package(component) for component in package["scope_packages"]]
        return result
    return {mode: load_package(package) for mode, package in generation.items()}


def validate_prepared_builds(prepared, expected, *, threads, build_args):
    """Reuse checked build artifacts across cases without reusing any run."""
    from elmm_pipeline.build_receipts import read_receipts, require_build_receipts
    from elmm_pipeline.common import digest_tree
    if (prepared.get("schema_version") != 1 or prepared.get("kind") != "prepared-elmm-comparison-builds"
            or prepared.get("contract_sha256") != expected["contract_sha256"]
            or prepared.get("original_application") != expected["original_application"]
            or prepared.get("selection_requests") != expected["selection_requests"]
            or prepared.get("calibration_profile") != expected.get("calibration_profile")
            or prepared.get("field_policy") != FIELD_POLICY
            or prepared.get("threads") != threads or prepared.get("build_args") != build_args):
        raise ValueError("prepared builds differ from the requested original sources, roots or build configuration")
    prepared = {**prepared, "generation": load_generation(prepared["generation"])}
    if prepared.get("calibration_profile"):
        identity = prepared["calibration_profile"]
        runtime = prepared["generation"]["reworked"]["scope_manifest"]["runtime"]
        if (digest(identity["path"]) != identity["sha256"]
                or runtime["runtime_id"] != identity["scoped_runtime_id"]
                or runtime["numerical_contract"] != identity["numerical_contract"]):
            raise ValueError("prepared candidate calibration/runtime binding changed")
    identities = {}
    for mode in ("committed", "reworked"):
        compiler = prepared["compilers"][mode]
        wanted = expected["compilers"][mode]
        if (compiler["compiler_tree_sha256"] != wanted["compiler_tree_sha256"]
                or compiler.get("exact_committed_snapshot") != wanted.get("exact_committed_snapshot")
                or compiler.get("base_commit") != wanted.get("base_commit")
                or digest_tree(Path(compiler["root"]) / "compiler") != compiler["compiler_tree_sha256"]):
            raise ValueError("prepared compiler snapshot differs: " + mode)
        for path in (Path(compiler["root"]) / "compiler").rglob("*"):
            if path.is_file() and path.suffix != ".pyc" and "__pycache__" not in path.parts:
                identities[str(path)] = digest(path)
    for mode in MODES:
        built, diagnostic = prepared["builds"][mode], prepared["diagnostic_builds"][mode]
        if (digest(built["binary"]) != built["sha256"]
                or digest(built["build_receipt"]) != built["build_receipt_sha256"]
                or digest(diagnostic["binary"]) != diagnostic["sha256"]
                or digest(diagnostic["source"]) != diagnostic["source_sha256"]
                or diagnostic["original_build_receipt"] != built["build_receipt"]
                or diagnostic["original_build_receipt_sha256"] != built["build_receipt_sha256"]):
            raise ValueError("prepared executable or diagnostic changed: " + mode)
        receipt = require_build_receipts(built["build_receipt"])
        if receipt["binary"] != built["binary"]:
            raise ValueError("prepared build receipt belongs to another executable")
        if mode == "native":
            original_hashes = {value for repository in expected["original_application"]["repositories"].values()
                               for value in repository["tracked_sha256"].values()}
            if any(receipt["source_inputs"].get(unit["source"]) not in original_hashes for unit in receipt["units"]):
                raise ValueError("prepared native executable contains a non-original compilation unit")
        else:
            generated = prepared["generation"][mode]
            for package in [generated, *generated["scope_packages"]]:
                root = Path(package["artifact_root"])
                manifest = package["scope_manifest"]
                if read(root / "scope-manifest.json") != manifest:
                    raise ValueError("prepared compiler manifest changed")
                for name, value in manifest["artifacts_sha256"].items():
                    artifact = package_artifact(root, name)
                    if digest(artifact) != value:
                        raise ValueError("prepared compiler artifact changed: " + name)
                    identities[str(artifact)] = value
                identities[str(root / "scope-manifest.json")] = digest(root / "scope-manifest.json")
            for unit in generated["gpu_sources"]:
                if receipt["source_inputs"].get(str(Path(unit["source"]).resolve())) != digest(unit["source"]):
                    raise ValueError("prepared compiler CUDA source was not compiled in this executable")
        mains = [Path(unit["source"]) for unit in receipt["units"]
                 if Path(unit["source"]).suffix.lower() == ".f90"
                 and "  call Dynamics_Deallocate\n" in Path(unit["source"]).read_text()
                 and "  call DeallocateGlobals\n" in Path(unit["source"]).read_text()]
        if len(mains) != 1 or Path(diagnostic["source"]).read_text() != diagnostic_main(mains[0].read_text()):
            raise ValueError("prepared full-storage diagnostic differs from the current field observer")
        identities.update(receipt["files_sha256"])
        for row in read_receipts(Path(diagnostic["binary"]).parent / "commands"):
            for item in (row["receipt"], row["execution"]["compiler"], *row["execution"]["inputs"], *row["execution"]["outputs"]):
                identities[item["path"]] = item["sha256"]
    return {key: prepared[key] for key in ("builds", "diagnostic_builds", "generation")}, identities


def validate_control_evidence(evidence, expected, *, case, configuration, configuration_dir, threads):
    """Authenticate this session's natural control fields without importing timings."""
    from elmm_pipeline.campaigns.halos import compare_fields, fields
    from elmm_pipeline.validation import compare_vtk, read_vtk, validate_log
    if (evidence.get("schema_version") != 1 or evidence.get("kind") != "fresh-session-natural-controls"
            or evidence.get("field_policy") != FIELD_POLICY or evidence.get("threads") != threads
            or evidence.get("original_application") != expected["original_application"]
            or evidence.get("contract_sha256") != expected["contract_sha256"]
            or configuration["expected_max_steps"] is not None):
        raise ValueError("natural control evidence differs from the requested source/configuration contract")
    source_report = evidence["source_report"]
    if digest(source_report["path"]) != source_report["sha256"]:
        raise ValueError("natural control source report changed")
    original = read(source_report["path"])
    selected = evidence["cases"][case]
    original_case = original["cases"][case]
    identities = {source_report["path"]: source_report["sha256"]}
    for name, value in selected["config_sha256"].items():
        source = Path(selected["config_dir"]) / name
        if digest(source) != value:
            raise ValueError("natural control configuration changed: " + name)
        identities[str(source)] = value
    requested_inputs = {str(path.relative_to(configuration_dir)): digest(path) for path in configuration_dir.rglob("*")
                        if path.is_file() and path.name != "origin.json"}
    control_inputs = {name: value for name, value in selected["config_sha256"].items() if Path(name).name != "origin.json"}
    if requested_inputs != control_inputs or selected["config_sha256"] != original_case["config_sha256"]:
        raise ValueError("natural control numerical inputs differ from this case")
    primary = Path(selected["runs"]["native"]["directory"])
    field_reference = fields(primary, {**FIELD_POLICY, "field_count": 20})
    vtk_reference = read_outputs(primary / "output", read_vtk)
    completed_steps = original_case["runs"]["native"]["timing"]["completed_steps"]
    imported = []
    for mode in ("native", "native-repeat", "committed"):
        control = selected["runs"][mode]
        directory, row = Path(control["directory"]), control["run"]
        if row != original_case["runs"][mode] or read(directory / "run.json") != row:
            raise ValueError("natural control run differs from its original record")
        build_mode = "native" if mode == "native-repeat" else mode
        built, diagnostic = expected["builds"][build_mode], expected["diagnostic_builds"][build_mode]
        environment = row["environment"]
        if (row.get("natural_duration") is not True or row.get("completed_original_configuration") is not True
                or row["binary_sha256"] != diagnostic["sha256"] or row["production_binary_sha256"] != built["sha256"]
                or row["build_receipt_sha256"] != built["build_receipt_sha256"]
                or row["argv"] != [diagnostic["binary"], *configuration["runtime_arguments"]]
                or environment.get("OMP_NUM_THREADS") != str(threads)
                or environment.get("OMP_DYNAMIC") != "FALSE" or environment.get("OMP_SCHEDULE") != "static"
                or environment.get("OMP_PROC_BIND") != "false"
                or environment.get("CUDA_VISIBLE_DEVICES") != ("" if mode.startswith("native") else None)):
            raise ValueError("natural control executable, invocation or thread environment differs")
        names = {str(path.relative_to(directory)) for path in directory.glob("fields-*")}
        names.update(str(path.relative_to(directory)) for path in (directory / "output").glob("*.vtk"))
        names.update(("run.log", "run.json"))
        if set(control["artifacts_sha256"]) != names:
            raise ValueError("natural controls must bind every full-field, VTK and log artifact")
        if any(control["artifacts_sha256"].get(name) != value for name, value in row["field_sha256"].items()):
            raise ValueError("natural control field bytes differ from their original recorded hashes")
        for name, value in control["artifacts_sha256"].items():
            path = package_artifact(directory, name)
            if digest(path) != value:
                raise ValueError("natural control artifact changed: " + str(path))
            identities[str(path)] = value
        timing = validate_log((directory / "run.log").read_text(), None, expected_completed_steps=completed_steps)
        observed = fields(directory, {**FIELD_POLICY, "field_count": 20})
        comparison = compare_fields(observed, field_reference, FIELD_POLICY)
        vtk = compare_outputs(vtk_reference, read_outputs(directory / "output", read_vtk), compare_vtk)
        passed = all(value["passed"] for value in comparison.values()) and vtk["passed"]
        if (row["timing"] != timing or row["field_count"] != 20 or comparison != row["fields"]
                or vtk != row["vtk"] or passed != row["passed"]):
            raise ValueError("natural control strict comparison changed")
        # Preserve strict errors and repeatability failures verbatim. Original
        # diagnostic durations remain in the source report, outside all samples.
        imported.append({"mode": build_mode, "label": mode + "-natural-control-evidence", "passed": passed,
            "vtk_validation": row["vtk"], "float64_storage_validation": {"passed": all(value["passed"] for value in comparison.values()),
                "fields": row["fields"], "coverage": "every allocated element including original halos and bounds"},
            "timing": {"completed_steps": completed_steps, "natural_end": True}, "reused_control_evidence": True,
            "source_report": source_report, "source_run": mode, "cwd": str(directory), "performance_sample": False})
    return {"vtk": vtk_reference, "steps": completed_steps}, field_reference, imported, identities


def markdown_report(receipt):
    lines = ["# Fresh ELMM compiler comparison", "", f"Status: **{receipt['status']}**.", "",
             "All timing samples use fresh production executables and include startup, computation, transfers, solver, output and teardown.",
             "Float64 storage diagnostics and runtime tracing execute separately from timing.", ""]
    if "performance_duration" in receipt:
        lines += [f"Case: **{receipt['case']}**. Performance duration: **{receipt['performance_duration']}**, identical in all modes.",
                  f"Correctness preserves the original configured duration: **{receipt['correctness_natural_duration']}**.", ""]
    if receipt.get("prepared_builds"):
        lines += ["Prepared executables were authenticated against compiler snapshots and executed build receipts; every production timing sample was collected afresh.", ""]
    if receipt.get("control_evidence"):
        evidence = receipt["control_evidence"]
        lines += [f"Natural native/committed control evidence was reused from `{evidence['source_report']['path']}` after authenticating all fields, bounds, VTK bytes, logs, binaries and recorded inputs.",
                  "Original strict failures remain unchanged. The candidate's natural-duration correctness run is fresh; no timing sample was imported.", ""]
    if "summary" in receipt:
        lines += ["| Mode | Median wall (s) | Range (s) | Speedup vs native | Wall reduction | Speedup vs committed |",
                  "|---|---:|---:|---:|---:|---:|"]
        for mode in MODES:
            row = receipt["summary"][mode]
            lines.append(f"| {mode} | {row['median_seconds']:.6f} | {row['minimum_seconds']:.6f}–{row['maximum_seconds']:.6f} | {row['speedup_over_native']:.3f}× | {row['wall_reduction_percent']:.2f}% | {row['speedup_over_committed']:.3f}× |")
        lines += ["", f"Correctness qualified: **{receipt.get('correctness_qualified', False)}**.",
                  f"Repeated performance protocol: **{receipt.get('performance_eligible', False)}**.", ""]
    lines += ["| Mode | Direct kernel launches | CUDA graph submissions | Upload bytes | Download bytes | Allocations | Waits |", "|---|---:|---:|---:|---:|---:|---:|"]
    for mode, row in receipt.get("gpu_evidence", {}).items():
        counts = row["scoped_runtime"]
        lines.append(f"| {mode} | {counts['kernel_launches']} | {len(row.get('graph_submission_evidence', []))} | {counts['upload_bytes']} | {counts['download_bytes']} | {counts['allocations']} | {counts['waits']} |")
    lines += ["", "Counts above are observed runtime events in a separate trace using the same duration as the timing workload. They are not inferred from procedure counts.",
              "See `report.json`, per-run `run.json`, compiler manifests and fresh build receipts for complete provenance, field errors, bounds, outputs and sample values.", ""]
    if any(row.get("command_queue_statistics") for row in receipt.get("gpu_evidence", {}).values()):
        lines += ["| Mode | Accepted tasks | Queue submissions | Graph submissions | Graph replays | Direct kernels | Peak pending tasks |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for mode, row in receipt["gpu_evidence"].items():
            queue, graph = row.get("command_queue_statistics", []), row.get("graph_statistics", [])
            if not queue:
                continue
            total = lambda records, name: sum(item[name] for item in records)
            lines.append(f"| {mode} | {total(queue, 'accepted_tasks')} | {total(queue, 'submissions')} | "
                         f"{total(graph, 'submissions')} | {total(graph, 'replays')} | {total(queue, 'direct_kernels')} | "
                         f"{max(item['peak_pending_tasks'] for item in queue)} |")
        lines += ["", "Queue and graph counters are completed runtime measurements. Logical kernels, CUDA graph submissions and replay counts are separate quantities.", ""]
    if receipt.get("cuda_profiles"):
        lines += ["| Mode | CUPTI kernels | Kernel activity (s) | H2D bytes | D2H bytes | Copy activity (s) |",
                  "|---|---:|---:|---:|---:|---:|"]
        for mode, profile in receipt["cuda_profiles"].items():
            metrics = profile["metrics"]
            if not metrics["available"]:
                lines.append(f"| {mode} | unavailable | unavailable | unavailable | unavailable | unavailable |")
                continue
            lines.append(f"| {mode} | {metrics['kernel_count']} | {metrics['kernel_seconds']:.6f} | {metrics['h2d_bytes']} | {metrics['d2h_bytes']} | {metrics['transfer_seconds']} |")
        lines += ["", "CUPTI counts and activity sums come from additional Nsight Systems runs. Overlapping device/API durations are not added to process wall time. CUDA API calls and CPU durations remain separate in `report.json`.", ""]
    if receipt.get("compilers"):
        lines += ["| Compiler | Base commit | Exact committed bytes | Compiler tree SHA-256 |", "|---|---|---|---|"]
        for mode, compiler in receipt["compilers"].items():
            lines.append(f"| {mode} | `{compiler['base_commit']}` | {compiler['exact_committed_snapshot']} | `{compiler['compiler_tree_sha256']}` |")
        lines += [""]
    if "error" in receipt:
        lines += ["Failure: " + receipt["error"]["reason"], ""]
    return "\n".join(lines)


def run_comparison(args):
    # Match the established native/container workflow's supported GNU choice.
    # Newer system defaults can reject the pinned upstream Fortran sources.
    os.environ.setdefault("FC", shutil.which("gfortran-15") or "gfortran")
    sys.path.insert(0, str(args.pipeline_root.resolve(strict=True)))
    import elmm_pipeline.local_adapter as scope_adapter
    from elmm_pipeline._legacy_scopes import transform_scopes as transform_legacy_scopes
    from elmm_pipeline.build import build, preprocessing_defines
    from elmm_pipeline.campaigns.halos import compare_fields, fields
    from elmm_pipeline.common import clean_environment, command
    from elmm_pipeline.gpu_profiling import profile_command
    from elmm_pipeline.gpu_toolchain import discover_toolchain
    from elmm_pipeline.gpu_validation import trace_environment
    from elmm_pipeline.local_adapter import apply_scope_sources, transform_scopes
    from elmm_pipeline.scope_extraction import export_analysis_sources
    from elmm_pipeline.structural import Project
    from elmm_pipeline.validation import compare_vtk, read_vtk, validate_log
    # This application comparison keeps the documented bounded generation
    # budget even if a separately maintained pipeline uses a longer default.
    scope_adapter.COMPILER_PHASE_TIMEOUT_SECONDS = 240
    output = args.output_dir.resolve()
    work = args.work_dir.resolve() if args.work_dir else output / "work"
    if output.exists() or work.exists():
        raise ValueError("comparison output/work must be new; existing evidence is immutable")
    if min(args.threads, args.jobs, args.repeats, args.timeout) < 1 or args.warmups < 0:
        raise ValueError("positive jobs, threads, repeats and timeout required; warmups cannot be negative")
    if args.steps is not None and args.steps < 1:
        raise ValueError("steps must be positive")
    if args.control_evidence and (not args.prepared_builds or args.steps is not None):
        raise ValueError("natural control evidence requires prepared builds and original-duration correctness")
    contract = read(DEFAULT_CONTRACT)
    if preprocessing_defines(args.build_arg) != contract["defines"]:
        raise ValueError("comparison requires the pinned DPREC preprocessing configuration")
    application_root = (args.app_root if args.app_root is not None else
                        prepare_application_basis(args.pipeline_root / ".cache/repos",
                                                  output.with_name(output.name + "-application-basis")))
    originals, _source_hashes = verify_contract(application_root, contract)
    if any(path == source or path.is_relative_to(source)
           for path in (output, work) for source in originals.values()):
        raise ValueError("comparison output/work must stay outside original application trees")
    original_provenance = repository_provenance(originals, contract, command)
    output.mkdir(parents=True)
    work.mkdir(parents=True)
    receipt = {"schema_version": 2, "status": "in_progress", "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "case": args.case, "protocol": {"warmups": args.warmups, "repetitions": args.repeats,
                   "threads": args.threads, "jobs": args.jobs, "rotate": True, "explicit_steps": args.steps,
                   "timing_steps": args.timing_steps,
                   "timeout_seconds": args.timeout, "diagnostics_excluded_from_timings": True},
               "performance_eligible": args.steps is None and args.repeats >= 3 and args.warmups >= 1,
               "full_duration_performance": args.steps is None and args.timing_steps is None,
               "correctness_natural_duration": args.steps is None,
               "performance_duration": f"matched {args.timing_steps}-step cap" if args.timing_steps else "original configured duration" if args.steps is None else f"{args.steps}-step smoke cap",
               "timings_reused": False, "correctness_qualified": False, "full_array_halos_validated": False,
               "field_policy": FIELD_POLICY, "required_fields": REQUIRED_FIELDS, "optional_fields": OPTIONAL_FIELDS,
               "original_application": original_provenance, "script_sha256": digest(__file__),
               "contract_sha256": digest(DEFAULT_CONTRACT), "compilers": {}, "builds": {}, "generation": {},
               "diagnostic_builds": {}, "correctness": [], "runs": [], "gpu_evidence": {}, "cuda_profiles": {}}
    receipt["protocol"]["compiler_timeout_seconds"] = 240
    report = output / "report.json"

    def save():
        write(report, {**receipt, "generation": generation_references(receipt["generation"])})
        (output / "report.md").write_text(markdown_report(receipt))

    save()
    try:
        options = {"nvcc": args.nvcc, "cuda_host_cxx": args.cuda_host, "architecture": args.arch}
        options = {key: value for key, value in options.items() if value is not None}
        toolchain = discover_toolchain(options=options, languages=("cuda",))
        if not toolchain["device"]["available"]:
            raise RuntimeError("visible GPU required for comparison: " + str(toolchain["device"]["error"]))
        receipt["gpu_toolchain"] = toolchain
        profile_tool, profile_transport = (node_profile_transport(args.nsys, work / "profiling")
                                           if not args.skip_cuda_profile else (args.nsys, None))
        if profile_transport:
            receipt["profiling_transport"] = profile_transport
        receipt["cpu_affinity"] = sorted(os.sched_getaffinity(0))
        receipt["compilers"]["committed"] = snapshot_compiler(args.committed_root, work / "compilers/committed", revision=args.committed_ref)
        receipt["compilers"]["reworked"] = snapshot_compiler(args.candidate_root, work / "compilers/reworked")
        selected, advection = select_configuration(originals, args.case, args.config)
        configuration, timing_configuration, timing_config_dir = prepare_workloads(
            selected, output, steps=args.steps, timing_steps=args.timing_steps, advection=advection)
        receipt["configuration"] = configuration
        receipt["configuration_sha256"] = {str(path.relative_to(output / "config")): digest(path) for path in (output / "config").rglob("*") if path.is_file()}
        receipt["timing_configuration"] = timing_configuration
        receipt["timing_configuration_sha256"] = {str(path.relative_to(timing_config_dir)): digest(path)
                                                 for path in timing_config_dir.rglob("*") if path.is_file()}
        requests = {"committed": selection_requests(contract),
                    "reworked": selection_requests(contract, read(args.candidate_request) if args.candidate_request else None,
                                                    scope_execution="graph")}
        requests["reworked"], calibration = candidate_calibration(requests["reworked"], args.calibration_profile, threads=args.threads)
        if calibration:
            receipt["calibration_profile"] = calibration
            receipt["calibration_hardware_binding"] = calibration_hardware_binding(calibration, toolchain)
        receipt["selection_requests"] = requests
        prepared_identities = {}
        if args.prepared_builds:
            selected_builds, prepared_identities = validate_prepared_builds(
                read(args.prepared_builds), receipt, threads=args.threads, build_args=args.build_arg)
            receipt.update(selected_builds)
            receipt["prepared_builds"] = {"path": str(args.prepared_builds.resolve()), "sha256": digest(args.prepared_builds),
                                          "timings_reused": False, "artifacts_authenticated": True}
            prepared_identities[str(args.prepared_builds.resolve())] = digest(args.prepared_builds)
            save()
        else:
            for mode in MODES:
                print("BUILD", mode, flush=True)
                app = work / mode / "application"
                for name, source in originals.items():
                    shutil.copytree(source, app / name, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", "*.o", "*.mod", ".sconsign*", "SConstruct.pipeline"))
                verify_contract(app, contract)
                paths = application_paths(app)
                metadata, changed = None, []
                if mode != "native":
                    packages, destinations = [], {}
                    for number, request in enumerate(requests[mode]):
                        source = contract_path(paths, request["source"])
                        sources = [source, *[contract_path(paths, name) for name in request["source_files"]]]
                        destinations.update({str(path): path for path in sources})
                        suffix = str(number) if len(requests[mode]) > 1 else ""
                        analysis = export_analysis_sources(Project(app, ["DPREC"]), work / mode / "analysis" / suffix, source_files=sources)
                        gpu_options = {**toolchain, **request["options"], "host_threads": args.threads}
                        scope_transform = transform_legacy_scopes if mode == "committed" else transform_scopes
                        component = scope_transform(source, Path(receipt["compilers"][mode]["root"]), args.python,
                            work / mode / "generated" / suffix, entry=request["entry"], source_files=sources[1:],
                            capture_facts=request["capture_facts"], analysis_sources=analysis, gpu_options=gpu_options)
                        if component["scope_manifest"]["scope_count"] < 1 or not component["gpu_sources"]:
                            raise RuntimeError(mode + " compiler emitted no device work for " + request["entry"])
                        packages.append(component)
                    if len(packages) == 1:
                        metadata = packages[0]
                        changed = apply_scope_sources(metadata, destinations)
                    else:
                        metadata = compose_generated_roots(packages, Path(receipt["compilers"][mode]["root"]), args.python,
                            work / mode / "generated/composed", destinations)
                        changed = [Path(name) for name in metadata["changed_sources"]]
                    manifest = metadata["scope_manifest"]
                    if mode == "reworked" and calibration and (
                            manifest["runtime"]["runtime_id"] != calibration["scoped_runtime_id"]
                            or manifest["runtime"]["numerical_contract"] != calibration["numerical_contract"]):
                        raise ValueError("candidate calibration differs from its compiled runtime/numerical contract")
                    receipt["generation"][mode] = {"command": metadata["command"], "scope_manifest": manifest,
                        "artifact_root": metadata["artifact_root"], "gpu_sources": metadata["gpu_sources"],
                        "changed_sources": [str(path) for path in changed],
                        "scope_packages": [{key: package[key] for key in ("entry", "command", "artifact_root", "scope_manifest")}
                                           for package in packages]}
                else:
                    gpu_options = {}
                binary = build(paths["elmm"], paths["PoisFFT"], work / mode / "build", args.scons_python,
                    args.native_library_dir, jobs=args.jobs, threads=args.threads, extra_args=args.build_arg,
                    extra_sources=[Path(path) for path in metadata["extra_sources"]] if metadata else None,
                    patched_sources=changed, gpu_sources=metadata["gpu_sources"] if metadata else None, gpu_options=gpu_options)
                receipt["builds"][mode] = {"binary": str(binary), "sha256": digest(binary),
                    "build_receipt": str(binary.parent / "build-receipts.json"),
                    "build_receipt_sha256": digest(binary.parent / "build-receipts.json")}
                receipt["diagnostic_builds"][mode] = relink_diagnostic(binary, paths["elmm"] / "src/main.f90", work / mode / "diagnostic")
                save()
        # Bind every compiler/generated/application build input before any run.
        bound = {str(path): digest(path) for tree in (work / "compilers",) for path in tree.rglob("*") if path.is_file() and path.suffix != ".pyc" and "__pycache__" not in path.parts}
        bound.update(prepared_identities)
        if calibration:
            bound[calibration["path"]] = calibration["sha256"]
        if profile_transport:
            bound[profile_transport["profiler"]] = profile_transport["profiler_sha256"]
            bound[profile_transport["wrapper"]] = profile_transport["wrapper_sha256"]
        for mode in MODES:
            for key in ("binary", "build_receipt"):
                path = receipt["builds"][mode][key]
                bound[path] = digest(path)
            for key in ("binary", "source"):
                path = receipt["diagnostic_builds"][mode][key]
                bound[path] = digest(path)
            if mode != "native":
                generated = receipt["generation"][mode]
                for package in [generated, *generated["scope_packages"]]:
                    for relative, sha in package["scope_manifest"]["artifacts_sha256"].items():
                        bound[str(Path(package["artifact_root"]) / relative)] = sha
                    manifest_path = Path(package["artifact_root"]) / "scope-manifest.json"
                    bound[str(manifest_path)] = digest(manifest_path)
        # Runtime and build helpers are inputs too; their roots may be separate checkouts.
        for tree in (args.pipeline_root / "elmm_pipeline", args.native_library_dir):
            for path in tree.rglob("*"):
                if path.is_file() and path.suffix != ".pyc" and "__pycache__" not in path.parts:
                    bound[str(path)] = digest(path)
        bound[str(Path(__file__))], bound[str(DEFAULT_CONTRACT)] = receipt["script_sha256"], receipt["contract_sha256"]
        helper = HERE / "sgs_experiment.py"
        bound[str(helper)] = digest(helper)
        prepared_path = output / "prepared-builds.json"
        write(prepared_path, {"schema_version": 1, "kind": "prepared-elmm-comparison-builds",
              "field_policy": FIELD_POLICY, "threads": args.threads, "build_args": args.build_arg,
              **{key: receipt[key] for key in ("contract_sha256", "original_application", "selection_requests",
                                              "compilers", "builds", "diagnostic_builds")},
              "generation": generation_references(receipt["generation"]),
              **({"calibration_profile": calibration} if calibration else {})})
        bound[str(prepared_path)] = digest(prepared_path)
        receipt["bound_input_sha256"] = bound
        save()
        base = clean_environment(args.threads)
        for name in list(base):
            if name.startswith(("FORT_", "OMP_", "GOMP_", "NVCOMPILER_", "PGI_")):
                base.pop(name)
        base.update(OMP_NUM_THREADS=str(args.threads), OMP_DYNAMIC="FALSE", OMP_SCHEDULE="static", OMP_PROC_BIND="false")
        field_reference = None
        references = {"correctness": {"vtk": None, "steps": None}, "timing": {"vtk": None, "steps": None}}
        if args.timing_steps is None:
            references["timing"] = references["correctness"]
        failures = []
        if args.control_evidence:
            natural, field_reference, imported, identities = validate_control_evidence(
                read(args.control_evidence), receipt, case=args.case, configuration=configuration,
                configuration_dir=output / "config", threads=args.threads)
            references["correctness"] = natural
            if args.timing_steps is None:
                references["timing"] = natural
            receipt["correctness"].extend(imported)
            failures.extend({"mode": row["mode"], "label": row["label"], "run": str(Path(row["cwd"]) / "run.json")}
                            for row in imported if not row["passed"])
            bound.update(identities)
            bound[str(args.control_evidence.resolve())] = digest(args.control_evidence)
            receipt["control_evidence"] = {"path": str(args.control_evidence.resolve()), "sha256": digest(args.control_evidence),
                "source_report": read(args.control_evidence)["source_report"], "imported_runs": len(imported),
                "timing_samples_imported": 0, "candidate_natural_correctness_reused": False,
                "strict_failures_preserved": True}
            save()

        def check_inputs():
            if repository_provenance(originals, contract, command) != original_provenance:
                raise ValueError("original application checkout changed during comparison")
            for path, sha in bound.items():
                if digest(path) != sha:
                    raise ValueError("bound comparison input changed: " + path)
            for name, sha in receipt["configuration_sha256"].items():
                if digest(output / "config" / name) != sha:
                    raise ValueError("comparison configuration changed: " + name)
            for name, sha in receipt["timing_configuration_sha256"].items():
                if digest(timing_config_dir / name) != sha:
                    raise ValueError("timing configuration changed: " + name)

        def execute(mode, label, *, traced=False, full_storage=False, warmup=False, round_number=None, phase="correctness"):
            nonlocal field_reference
            check_inputs()
            directory = output / "runs" / mode / label
            run_directory(timing_config_dir if phase == "timing" else output / "config", directory)
            selected_configuration = timing_configuration if phase == "timing" else configuration
            reference = references[phase]
            binary = receipt["diagnostic_builds" if full_storage else "builds"][mode]["binary"]
            env = trace_environment(base, "local", enabled=traced)
            argv = [binary, *selected_configuration["runtime_arguments"]]
            print("TIMING" if round_number is not None else "VERIFY", mode, label, flush=True)
            log_path = directory / "run.log"
            log, elapsed = command(argv, cwd=directory, env=env, log=log_path, timeout=args.timeout)
            timing = validate_log(log, selected_configuration["expected_max_steps"], expected_completed_steps=reference["steps"])
            observed = read_outputs(directory / "output", read_vtk)
            if reference["vtk"] is None:
                if mode != "native" or full_storage or traced:
                    raise RuntimeError("original production native reference must execute first")
                reference["vtk"], reference["steps"] = observed, timing["completed_steps"]
            validation = compare_outputs(reference["vtk"], observed, compare_vtk)
            row = {"mode": mode, "label": label, "argv": argv, "cwd": str(directory), "timing": timing,
                "workload_phase": phase, "configuration": selected_configuration,
                "traced": traced, "full_storage": full_storage, "warmup": warmup, "round": round_number,
                "vtk_validation": validation, "log": str(log_path), "log_sha256": digest(log_path),
                "environment": {key: env.get(key) for key in ("OMP_NUM_THREADS", "OMP_DYNAMIC", "OMP_SCHEDULE", "OMP_PROC_BIND",
                    "OMP_PLACES", "GOMP_CPU_AFFINITY", "CUDA_VISIBLE_DEVICES", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "LD_LIBRARY_PATH", "FORT_RUNTIME_TRACE", "FORT_OFFLOAD_TRACE")},
                "wall_seconds" if round_number is not None else "diagnostic_wall_seconds": elapsed}
            if full_storage:
                count = len(list(directory.glob("fields-*.bin")))
                observed_fields = fields(directory, {**FIELD_POLICY, "field_count": count})
                required = {"fields-" + name + ".bin" for name in REQUIRED_FIELDS}
                if not required.issubset(observed_fields):
                    raise RuntimeError("missing required full-storage fields")
                if field_reference is None:
                    if mode != "native":
                        raise RuntimeError("native full-storage reference must execute first")
                    field_reference = observed_fields
                comparison = compare_fields(observed_fields, field_reference, FIELD_POLICY)
                row["float64_storage_validation"] = {"passed": all(item["passed"] for item in comparison.values()),
                    "fields": comparison, "coverage": "every allocated element including halos; original bounds and shape retained"}
                row["float64_storage_sha256"] = {str(path.relative_to(directory)): digest(path) for path in directory.glob("fields-*")}
            if traced:
                packages = receipt["generation"][mode]["scope_packages"]
                activity = observed_gpu_activity(log, scope_packages=packages)
                row["gpu_activity"] = activity
                receipt["gpu_evidence"][mode] = activity
                row["cuda_execution_observed"] = activity["verified"] and activity["runtime_cuda_submissions"] > 0
                native_selection = native_auto_selection(activity, requests[mode])
                if native_selection:
                    row["native_auto_selection_evidence"] = native_selection
                    activity["native_auto_selection_evidence"] = native_selection
                # A historical control may leave this case's original branch
                # entirely native. Preserve its actual zero runtime count;
                # only the candidate must establish real offloaded execution.
                if mode == "reworked" and not row["cuda_execution_observed"] and not native_selection:
                    raise RuntimeError(mode + " CUDA execution is not verified")
                # A checked rejection before a worker starts is a valid native
                # task in a mixed graph. Preserve its reason/position so its
                # frequency remains visible beside numerical correctness.
                row["gpu_preflight_declines"] = [item for item in activity["scoped_runtime"].get("modeled_evidence", [])
                                                  if item["event"] == "definition_validation" and item["status"] != 0]
                activity = runtime_activity_reference(activity, directory)
                row["gpu_activity"] = activity
                receipt["gpu_evidence"][mode] = activity
                if native_selection:
                    row["native_auto_selection_evidence"] = activity["native_auto_selection_evidence"]
            elif "FORT_SCOPED " in log or "FORT_OFFLOAD " in log:
                raise RuntimeError("untraced execution unexpectedly emitted runtime diagnostics")
            row["passed"] = validation["passed"] and row.get("float64_storage_validation", {"passed": True})["passed"]
            if not row["passed"]:
                failures.append({"mode": mode, "label": label, "run": str(directory / "run.json")})
            write(directory / "run.json", row)
            return row

        correctness_sequence = (
            ("native", "reference", False, False), ("native", "repeatability", False, False),
            ("native", "float64-reference", False, True), ("native", "float64-repeatability", False, True),
            ("committed", "gpu-validation", True, False), ("reworked", "gpu-validation", True, False),
            ("committed", "float64-validation", False, True), ("reworked", "float64-validation", False, True))
        for mode, label, traced, full_storage in correctness_sequence:
            if args.control_evidence and (mode == "native" or mode == "committed" and full_storage):
                continue
            if traced and args.timing_steps is not None and references["timing"]["vtk"] is None:
                receipt["correctness"].append(execute("native", "timing-reference", phase="timing"))
                save()
            receipt["correctness"].append(execute(mode, label, traced=traced, full_storage=full_storage,
                                                  phase="timing" if traced else "correctness"))
            save()
        receipt["validation_failures"] = failures
        if failures and not args.allow_validation_failure:
            raise RuntimeError("correctness/repeatability failed; see retained per-run field comparisons")
        if not args.skip_cuda_profile:
            for mode in ("committed", "reworked"):
                check_inputs()
                print("PROFILE", mode, flush=True)
                directory = run_directory(timing_config_dir, output / "runs" / mode / "cuda-profile")
                argv = [receipt["builds"][mode]["binary"], *timing_configuration["runtime_arguments"]]
                log, elapsed, metrics = profile_command(argv, directory / "profile", base, directory,
                                                        timeout=args.timeout, nsys=profile_tool)
                if profile_transport:
                    metrics["underlying_argv"] = [profile_transport["profiler"], "profile", "--cuda-graph-trace=node",
                                                  *metrics.get("argv", [])[2:]]
                    metrics["cuda_graph_trace"] = "node"
                profile = {"metrics": metrics, "diagnostic_wall_seconds": elapsed, "performance_sample": False,
                           "configuration": timing_configuration, "workload_phase": "timing"}
                receipt["cuda_profiles"][mode] = profile
                if metrics["available"]:
                    profile["cuda_api_activity"] = cuda_api_activity(metrics["sqlite_path"])
                if metrics["available"] or metrics.get("process_returncode") == 0:
                    profile["timing"] = validate_log(log, timing_configuration["expected_max_steps"], expected_completed_steps=references["timing"]["steps"])
                    profile["vtk_validation"] = compare_outputs(references["timing"]["vtk"], read_outputs(directory / "output", read_vtk), compare_vtk)
                    profile["application_verified"] = profile["vtk_validation"]["passed"]
                    if not profile["vtk_validation"]["passed"]:
                        failures.append({"mode": mode, "label": "cuda-profile", "run": str(directory)})
                        if not args.allow_validation_failure:
                            save()
                            raise RuntimeError("profiled application output differs from native reference")
                needs_cuda_profile = (receipt["gpu_evidence"][mode]["verified"]
                                      or mode == "reworked" and not receipt["gpu_evidence"][mode].get("native_auto_selection_evidence"))
                if not metrics["available"] and args.require_cuda_profile and needs_cuda_profile:
                    save()
                    raise RuntimeError(mode + " CUDA profile is unavailable: " + str(metrics["reason"]))
                save()
        for number in range(args.warmups + args.repeats):
            for mode in rotated_order(number):
                receipt["runs"].append(execute(mode, f"round-{number:02d}", warmup=number < args.warmups, round_number=number, phase="timing"))
                save()
        check_inputs()
        receipt["validation_failures"] = failures
        receipt["correctness_qualified"] = not failures
        receipt["full_array_halos_validated"] = not failures
        receipt["summary"] = summarize(receipt["runs"])
        receipt["status"] = "passed" if not failures else "completed_unvalidated"
    except BaseException as error:
        receipt.update(status="failed", error={"type": type(error).__name__, "reason": str(error)})
        raise
    finally:
        receipt["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-root", type=Path,
                        help="Explicit clean approved ELMM/PoisFFT siblings; default reconstructs the common basis from the pipeline cache")
    parser.add_argument("--pipeline-root", type=Path, default=PIPELINE)
    parser.add_argument("--committed-root", type=Path, default=COMPILER)
    parser.add_argument("--committed-ref", default="be58ab809296b97ea39d701828a2621adfc6b3dc",
                        help="Historical SGS/direct compiler revision; an explicit value overrides the pinned default")
    parser.add_argument("--candidate-root", type=Path, default=COMPILER)
    parser.add_argument("--prepared-builds", type=Path, help="Reuse an authenticated prepared build set across cases; collect every run afresh")
    parser.add_argument("--control-evidence", type=Path, help="Reuse authenticated natural native/committed fields from this session; candidate correctness and every timing run stay fresh")
    parser.add_argument("--candidate-request", type=Path,
                        help="schema-1 source/entry/source_files/capture_facts/options object or array of roots; SGS-only control if omitted")
    parser.add_argument("--calibration-profile", type=Path,
                        help="Candidate-only offline hardware/scoped-runtime calibration; committed control is unchanged")
    parser.add_argument("--python", type=Path, default=Path(sys.executable).absolute())
    parser.add_argument("--scons-python", type=Path, default=PIPELINE / ".cache/envs/build/bin/python")
    parser.add_argument("--native-library-dir", type=Path, default=PIPELINE / ".cache/native/lib")
    parser.add_argument("--case", choices=CASE_NAMES, default="channel-simple")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--timing-steps", type=int, help="Cap only performance/profile runs; full-storage correctness preserves natural duration")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--nvcc")
    parser.add_argument("--cuda-host")
    parser.add_argument("--arch")
    parser.add_argument("--nsys", help="Explicit Nsight Systems executable for separate CUDA activity diagnostics")
    parser.add_argument("--skip-cuda-profile", action="store_true", help="Skip additional Nsight diagnostics; runtime counts remain available")
    parser.add_argument("--require-cuda-profile", action="store_true", help="Fail if actual CUDA activity tracing is unavailable")
    parser.add_argument("--build-arg", action="append", default=[])
    parser.add_argument("--allow-validation-failure", action="store_true",
                        help="retain failed comparisons and collect explicitly unqualified timings")
    parser.add_argument("--output-dir", "--output", "-o", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args(argv)
    if args.skip_cuda_profile and args.require_cuda_profile:
        parser.error("--skip-cuda-profile conflicts with --require-cuda-profile")
    if args.steps is not None and args.timing_steps is not None:
        parser.error("--steps smoke cap conflicts with --timing-steps performance-only cap")
    if args.timing_steps is not None and args.timing_steps < 1:
        parser.error("--timing-steps must be positive")
    try:
        run_comparison(args)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, "error: " + str(error) + "\n")


if __name__ == "__main__":
    main()
