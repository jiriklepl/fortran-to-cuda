"""Read actual GPU activity from separate, numerically checked trace runs."""
from __future__ import annotations

import json
import math
import re
from hashlib import sha256

from .scoped_compact_trace import (
    COUNTERS,
    validate_compact_transfer,
    validate_modeled_compact_copy,
)
from .scoped_copy_pricing import validate_copy_pricing
from .scoped_domain_trace import DomainTrace
from .scoped_provenance import FIELDS, HASH, _identities, _registry, scoped_provenance

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


class _InlineNativeFallbacks:
    """Correlate inline fallback and aggregate schedule labels separately.

    Fallbacks need the exact reached source slot. Schedule summaries retain
    only their active domain and modeled counts, with ambiguous source labels
    explicit. Neither establishes a submitted kernel or payload operation.
    """

    def __init__(self, manifests):
        self.records, _supported = _registry(manifests)
        self.used = set()
        self.validated_sibling_components = set()
        self.domains = DomainTrace(lambda fields: _identities(fields, self.records, self.used))
        self.components = [component for manifest in manifests
                           for component in manifest.get("component_manifests", [manifest])]
        self.scheduled_entries = {}
        self.pending_schedule = None
        for component in self.components:
            groups = [component.get("inline_numerical_regions", {}), *component.get("child_numerical_regions", ())]
            for group in groups:
                for region in group.get("regions", ()):
                    entry, source = region.get("entry"), region.get("source_identity")
                    if not isinstance(entry, str) or not HASH.fullmatch(str(source)):
                        continue
                    if any(row.get("kind") == "procedure" and row.get("procedure") == entry
                           and row.get("source") == "inline:" + source
                           and HASH.fullmatch(str(row.get("numerical_identity", "")))
                           for row in self.records.values()):
                        self.scheduled_entries.setdefault(entry.rsplit("::", 1)[-1].lower(), set()).add(entry)

    def lifecycle(self, line):
        if line.startswith("FORT_SCOPED domain "):
            self.domains.lifecycle(line.removeprefix("FORT_SCOPED domain "))
        if line.startswith("FORT_SCOPED decision "):
            suffix = re.search(r" context=([1-9][0-9]*) domain_version=1 domain=([1-9][0-9]*) domain_role=execution$", line)
            self.pending_schedule = None
            if suffix:
                decision, = scoped_modeled_decisions(line[:suffix.start()])
                context, domain = suffix.groups()
                if max(int(context), int(domain)) >= 2**64:
                    raise ValueError("Inline schedule domain identity exceeds its ABI")
                fields = {"context": context, "domain": domain, "domain_version": "1", "domain_role": "execution"}
                self.domains.operation(fields, "position", 0, tagged=True)
                self.pending_schedule = (decision, fields)
        elif line.startswith("FORT_SCOPED planning_timing ") and self.pending_schedule is not None:
            decision, fields = self.pending_schedule
            timing = re.fullmatch(r"FORT_SCOPED planning_timing context=([1-9][0-9]*) domain_version=1 "
                r"domain=([1-9][0-9]*) domain_role=execution phase=selection seconds=([0-9.eE+-]+) "
                r"cache_hit=[01] query_generation=[0-9]+ state_generation=[0-9]+", line)
            if (timing is None or timing.group(1, 2) != (fields["context"], fields["domain"])
                    or not math.isfinite(float(timing[3])) or float(timing[3]) < 0):
                raise ValueError("Inline schedule selection timing changed its domain or format")
        elif not line.startswith("FORT_OFFLOAD "):
            self.pending_schedule = None

    def scheduled_summary(self, raw_entry, mode, gpu, cpu, unit_kind, line_number):
        """Match an aggregate label; it cannot identify the executed source unit."""
        pending, self.pending_schedule = self.pending_schedule, None
        entries = self.scheduled_entries.get(raw_entry)
        if not entries:
            raise ValueError("Offload policy trace entry is absent from its inline source manifest")
        if pending is None or unit_kind != "regions":
            raise ValueError("Inline schedule summary lacks a preceding checked domain decision")
        decision, fields = pending
        expected_mode = "scoped-scheduled" if decision["gpu_units"] else "native-scoped-scheduled"
        if (mode != expected_mode
                or (int(gpu), int(cpu)) != (decision["gpu_units"], decision["cpu_units"])):
            raise ValueError("Inline schedule summary disagrees with its preceding decision")
        return {"kind": "checked_inline_schedule_summary", "entry": raw_entry, "log_line": line_number,
                "context": int(fields["context"]), "domain": int(fields["domain"]),
                "mode": mode, "gpu_units": int(gpu), "cpu_units": int(cpu), "unit_kind": unit_kind,
                "estimate_available": decision["available"],
                "candidate_entries": sorted(entries), "source_attribution_available": False,
                "aggregate_only": True, "gpu_execution_confirmed": False}

    def borrowed_owner_routes(self, scope, plan, owner, caller, component):
        """Use the existing checked companion CALL and coordinator receipts."""
        procedures = {row["procedure"]: row for row in plan.get("procedures", ())}
        name = owner.get("procedure")
        member = procedures.get(name)
        called = procedures.get(caller.get("procedure"))
        sources = plan.get("sources", {})
        if (owner.get("lifetime") != "borrowed helper storage epoch" or member is None
                or called is None or len(procedures) != len(plan.get("procedures", ()))
                or any(owner.get(key) != member.get(key) for key in ("source", "source_sha256"))
                or any(caller.get(key) != called.get(key) for key in ("source", "source_sha256"))
                or any(not HASH.fullmatch(str(row.get("structured_summary_identity", "")))
                       or sources.get(row.get("source")) != row.get("source_sha256") for row in (member, called))):
            return None

        def digest(row):
            return sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

        def route(start, target):
            pending, seen = [(start, [])], set()
            while pending:
                current, path = pending.pop(0)
                if current == target:
                    return path
                if current in seen or current not in procedures:
                    continue
                seen.add(current)
                parent = procedures[current]
                if (not HASH.fullmatch(str(parent.get("structured_summary_identity", "")))
                        or sources.get(parent.get("source")) != parent.get("source_sha256")):
                    raise ValueError("Borrowed inline owner lost its checked caller graph")
                for child_name, child in procedures.items():
                    coordinators = [row for row in scope.get("module_coordinators", ())
                                    if row.get("procedure") == child_name]
                    if len(coordinators) != 1 or coordinators[0].get("parameterized_handles") is not True:
                        continue
                    coordinator = coordinators[0]
                    for call in child.get("calls", ()):
                        if call.get("caller") != current or call not in coordinator.get("calls", ()):
                            continue
                        first, last = call.get("first_line"), call.get("last_line")
                        handles, mappings = call.get("handle_bindings"), call.get("resource_mappings")
                        if (type(first) is not int or type(last) is not int or not 1 <= first <= last
                                or not HASH.fullmatch(str(child.get("structured_summary_identity", "")))
                                or sources.get(child.get("source")) != child.get("source_sha256")
                                or not isinstance(handles, dict) or not handles
                                or any(not isinstance(key, str) or not re.fullmatch(r"[1-9][0-9]*", key)
                                       or type(value) is not int or value <= 0 for key, value in handles.items())
                                or not isinstance(mappings, dict)
                                or any(not isinstance(key, str) or not key or not isinstance(value, str) or not value
                                       for key, value in mappings.items())):
                            continue
                        edge = {"caller": current, "callee": child_name, "first_line": first, "last_line": last,
                                "caller_source": parent["source"], "caller_source_sha256": parent["source_sha256"],
                                "caller_graph": parent["structured_summary_identity"],
                                "callee_graph": child["structured_summary_identity"],
                                "checked_call_identity": digest(call), "coordinator_identity": digest(coordinator),
                                "handle_bindings": handles, "resource_mappings": mappings}
                        pending.append((child_name, [*path, edge]))
            return None

        root_path = route(plan.get("entry"), name)
        worker_path = route(name, caller.get("procedure"))
        if root_path is None:
            return None
        if worker_path is None:
            # A domain may be opened lazily in one child and then borrowed by
            # a sibling after returning to their checked caller. The origin
            # tag does not assert a dynamic CALL ancestry for later helpers.
            caller_path = route(plan.get("entry"), caller.get("procedure"))
            contract = plan.get("storage_epoch", {}).get("return_contract")
            if caller_path is None or not isinstance(contract, dict):
                return None
            if id(component) not in self.validated_sibling_components:
                from .checked_execution_plan import validate_checked_execution_plans

                validate_checked_execution_plans(component)
                self.validated_sibling_components.add(id(component))
            return {"kind": "checked sibling helper storage epoch",
                    "owner_graph": member["structured_summary_identity"],
                    "return_contract_identity": contract["identity"],
                    "root_to_owner": root_path, "root_to_caller": caller_path,
                    "dynamic_call_ancestry_inferred": False}
        return {"kind": "checked borrowed helper storage epoch", "owner_graph": member["structured_summary_identity"],
                "root_to_owner": root_path, "owner_to_caller": worker_path}

    def attribution(self, previous, raw_entry, mode, gpu, cpu, unit_kind, line_number):
        if mode in {"native-scoped-scheduled", "scoped-scheduled"}:
            return self.scheduled_summary(raw_entry, mode, gpu, cpu, unit_kind, line_number)
        if (mode not in {"native-scoped-resource", "native-scoped-boundary"}
                or (gpu, cpu, unit_kind) != ("0", "1", "regions")
                or not previous.startswith("FORT_SCOPED position ")):
            raise ValueError("Offload policy trace entry is absent from the build manifest and lacks an exact preceding inline source position")
        fields = {}
        for token in previous.removeprefix("FORT_SCOPED position ").split():
            key, separator, value = token.partition("=")
            if not separator or key in fields:
                raise ValueError("Malformed inline fallback source position")
            fields[key] = value
        if (set(fields) != {"bytes", "provenance_version", "context", "domain_version", "domain", "domain_role", *FIELDS}
                or fields["bytes"] != "0" or fields["provenance_version"] != "1"
                or fields["domain_version"] != "1" or fields["domain_role"] != "execution"
                or any(not re.fullmatch(r"[1-9][0-9]*", fields[key]) or int(fields[key]) >= 2**64
                       for key in ("context", "domain"))
                or any(not HASH.fullmatch(fields[key]) for key in ("owner", "procedure", "segment", "implementation"))):
            raise ValueError("Inline fallback requires exact versioned source/domain attribution")
        _identities(fields, self.records, self.used)
        self.domains.operation(fields, "position", 0, tagged=True)
        owner, caller, segment, implementation = (self.records[fields[key]]
                                                 for key in ("owner", "procedure", "segment", "implementation"))
        worker = self.records.get(implementation.get("procedure_id"), {})
        if (worker.get("kind") != "procedure" or implementation.get("backend") != "cuda"
                or type(implementation.get("region")) is not int or implementation["region"] < 0
                or any(worker.get(key) != implementation.get(key) for key in ("procedure", "source", "numerical_identity"))
                or not HASH.fullmatch(str(worker.get("numerical_identity", "")))
                or not isinstance(worker.get("procedure"), str)
                or worker["procedure"].rsplit("::", 1)[-1].lower() != raw_entry
                or segment.get("procedure") != caller.get("procedure")
                or any(segment.get(key) != caller.get(key) for key in ("source", "source_sha256"))):
            raise ValueError("Inline fallback worker/source provenance relationship differs")
        matches = {}
        for component in self.components:
            local = {record["id"] for record in component.get("runtime_provenance", {}).get("records", ())}
            if not {fields[key] for key in ("owner", "procedure", "segment", "implementation")} <= local:
                continue
            # Catalog ownership authenticates serial-loop completions, whose
            # existing record deliberately has no procedure field. Keep that
            # source relationship instead of flattening it out of the reader.
            catalogs = [(component.get("entry"), component.get("inline_numerical_regions", {}))]
            catalogs += [(child.get("procedure"), child)
                         for child in component.get("child_numerical_regions", ())]
            for scope in component.get("scopes", ()):
                plan = scope.get("checked_execution_plan", {})
                if caller.get("procedure") not in {p.get("procedure") for p in plan.get("procedures", ())}:
                    continue
                owner_route = None
                if plan.get("entry") != owner.get("procedure"):
                    owner_route = self.borrowed_owner_routes(scope, plan, owner, caller, component)
                    if owner_route is None:
                        continue
                regions = [(catalog_owner, region) for catalog_owner, catalog in catalogs
                           for region in catalog.get("regions", ())]
                for catalog_owner, region in regions:
                    completion = region.get("completion", {})
                    declared = completion.get("procedure")
                    owner_matches = catalog_owner is None or catalog_owner == caller.get("procedure")
                    if "procedure" in completion:
                        completion_matches = owner_matches and declared == caller.get("procedure")
                    else:
                        # Current graphs may group consecutive sibling loops.
                        # Frozen controls retain the earlier single-loop record;
                        # both require the same exact source/catalog ownership.
                        reason = completion.get("reason")
                        serial = {"available": True, "caller_contract": "serial_source_scope", "reason": reason}
                        completion_matches = (catalog_owner == caller.get("procedure")
                                              and completion.get("available") is True and completion == serial
                                              and reason in (
                                                  "original serial counted loop completes before continuation",
                                                  "original serial counted loops complete in source order before continuation"))
                    if (region.get("entry") == worker["procedure"]
                            and worker.get("source") == "inline:" + str(region.get("source_identity"))
                            and completion_matches
                            and all(type(region.get(key)) is int and type(segment.get(key)) is int
                                    for key in ("first_line", "last_line"))
                            and segment["first_line"] <= region["first_line"] <= region["last_line"] <= segment["last_line"]):
                        match = {"entry": worker["procedure"], "raw_entry": raw_entry,
                                 "owner": fields["owner"], "procedure": fields["procedure"],
                                 "segment": fields["segment"], "implementation": fields["implementation"],
                                 "worker_procedure": implementation["procedure_id"],
                                 "numerical_identity": worker["numerical_identity"],
                                 "source_identity": region["source_identity"],
                                 "first_line": region["first_line"], "last_line": region["last_line"]}
                        if owner_route is not None:
                            match["borrowed_owner_route"] = owner_route
                        matches[json.dumps(match, sort_keys=True)] = match
        if len(matches) != 1:
            raise ValueError("Inline fallback is absent from its exact owning source component")
        return {**next(iter(matches.values())), "log_line": line_number, "context": int(fields["context"]),
                "domain": int(fields["domain"]), "mode": mode, "gpu_units": 0, "cpu_units": 1,
                "unit_kind": "regions", "gpu_execution_confirmed": False}


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
    prefix = r"(?<![A-Za-z0-9_])FORT_OFFLOAD[ \t]+entry=([A-Za-z][A-Za-z0-9_]*)"
    inline = (_InlineNativeFallbacks(manifests) if any(match[1].lower() not in entries
              for match in re.finditer(prefix, log)) else None)
    attributed, scheduled, previous = [], [], ""
    observed = []
    for number, line in enumerate(log.splitlines(), 1):
        if inline is not None:
            inline.lifecycle(line)
        matches = list(re.finditer(pattern, line))
        for claim in re.finditer(prefix, line):
            if claim[1].lower() not in entries and not any(match.start() == claim.start() for match in matches):
                raise ValueError("Malformed unknown inline offload policy row")
        for match in matches:
            entry, mode, gpu, cpu, unit_kind = match.groups()
            unit_kind = unit_kind or "regions"
            if entry.lower() not in entries:
                if match.start() != 0 or match.end() != len(line):
                    raise ValueError("Inline fallback requires one complete policy trace row")
                attribution = inline.attribution(previous, entry.lower(), mode, gpu, cpu, unit_kind, number)
                if attribution.get("kind") == "checked_inline_schedule_summary":
                    scheduled.append(attribution)
                else:
                    attributed.append(attribution)
                entry = attribution["entry"]
            observed.append((entry, mode, gpu, cpu, unit_kind))
        if inline is not None and line.startswith("FORT_OFFLOAD "):
            inline.pending_schedule = None
        previous = line
    for entry, mode, gpu, cpu, unit_kind in observed:
        entry = entry.lower()
        unit_kind = unit_kind or "regions"
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
            **({"authenticated_inline_fallbacks": attributed} if attributed else {}),
            **({"matched_inline_schedule_summaries": scheduled} if scheduled else {}),
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
    Versioned runtime statistics preserve both complete close records and
    incomplete abandonment records; preserving them never verifies a launch.
    """
    statistics = {
        "completion_statistics": (
            ("event_records", "event_queries", "event_waits", "stream_waits", "created_events",
             "reused_events", "independent_host_tasks"), ()),
        "command_queue_statistics": (
            ("accepted_tasks", "submissions", "conflict_flushes", "explicit_flushes", "pending_tasks",
             "peak_pending_tasks", "copied_argument_bytes", "direct_submissions", "direct_kernels",
             "graph_resource_fallbacks"), ("queue_enabled", "graphs_enabled")),
        "graph_statistics": (
            ("creations", "instantiations", "rebuilds", "submissions", "replays", "parameter_updates",
             "logical_kernels", "live_graphs", "destructions"), ()),
    }
    events = {"decision", "snapshot", "operation", "requirement", "copy", "export", "gate", "end",
              "definition_validation", "transfer_statistics", "batch_statistics", "compact_transfer", "copy_pricing"} | statistics.keys()

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
        domain_fields = {"domain_version", "domain", "domain_role"}
        has_domain = bool(domain_fields & record.keys())
        if has_domain:
            domain, role = record.get("domain"), record.get("domain_role")
            if (not domain_fields <= record.keys()
                    or type(record.get("domain_version")) is not int or record["domain_version"] != 1
                    or type(domain) is not int or not 0 < domain < 2**64
                    or role not in ("storage", "execution") or not context
                    or (context == domain) != (role == "storage")):
                raise ValueError("Scoped evidence requires complete versioned residency domain attribution")
        if event == "end" and (type(record.get("rows")) is not int or record["rows"] < 0
                               or type(record.get("complete")) is not bool or type(record.get("truncated")) is not bool):
            raise ValueError("Scoped evidence end requires row and completeness fields")
        if event == "definition_validation" and (
                type(record.get("status")) is not int or not 0 <= record["status"] <= 8
                or not isinstance(record.get("reason"), str) or not record["reason"]
                or record.get("evidence") != "ordered_definition_preflight"
                or record.get("mutates_live_state") is not False):
            raise ValueError("Scoped definition validation requires status, reason and a nonmutating proof")
        if event in statistics:
            counts, flags = statistics[event]
            fields = {"schema_version", "context", "event", "stats_version", "complete", *counts, *flags}
            if has_domain:
                fields |= domain_fields
            if (record.keys() != fields
                    or type(record.get("stats_version")) is not int or record["stats_version"] != 1
                    or type(record.get("complete")) is not bool
                    or any(type(record.get(field)) is not int or not 0 <= record[field] < 2**64
                           for field in counts)
                    or any(type(record.get(field)) is not int or record[field] not in (0, 1)
                           for field in flags)):
                raise ValueError("Scoped runtime statistics require exact versioned public counters and flags")
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
        if event == "compact_transfer":
            validate_compact_transfer(record)
        elif event == "copy_pricing":
            validate_copy_pricing(record)
        else:
            validate_modeled_compact_copy(record)
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
        observed = scoped["source_provenance"]["totals"]
        graph_counts = {name: observed[name] for name in ("graph_submissions", "graph_kernel_submissions")
                        if name in observed}
        scoped.update(graph_counts)
        copy_counts = {name: observed[name] for name in COUNTERS if name in observed}
        scoped.update(copy_counts)
        graph_kernels = graph_counts.get("graph_kernel_submissions", 0)
        native = bool(manifests)
        for manifest in manifests:
            scopes = manifest.get("scopes", [])
            native = native and (manifest.get("schema_version") == 1
                      and (manifest.get("scope_count") == 0 and not scopes or
                           bool(scopes) and all(_native_source_preflight(scope) for scope in scopes)))
        if native and (kernels or scoped["kernel_launches"] or graph_kernels):
            raise ValueError("CUDA launch contradicts the compiler's all-native source-scope contract")
        return {**graph_counts, **copy_counts, "verified": kernels + scoped["kernel_launches"] + graph_kernels > 0,
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


def _native_source_preflight(scope):
    """Unavailable costs do not settle placement of a reached GPU query."""
    preflight = scope.get("automatic_preflight")
    return (scope.get("mode") == "auto" and isinstance(preflight, dict)
            and preflight.get("selection") == "native" and preflight.get("successful") is True
            and preflight.get("caller_source_unchanged") is True
            and preflight.get("generated_owner_invoked") is False
            and type(preflight.get("contexts_created")) is int and preflight["contexts_created"] == 0)
