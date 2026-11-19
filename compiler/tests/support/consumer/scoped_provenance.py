"""Attribute observed scope operations through compiler-published identities.

These counters distinguish API submissions, submitted logical graph members
and coherence commits. They do not measure device duration, prove overlap, or
reconstruct an execution policy.
"""
from __future__ import annotations

import re

from .scoped_compact_trace import compact_from_line, count_compact, provenance_fields
from .scoped_copy_pricing import pricing_from_line
from .scoped_copy_pricing import provenance_fields as pricing_provenance_fields
from .scoped_domain_trace import DomainTrace

FIELDS = ("owner", "procedure", "segment", "operation", "implementation", "boundary")
OBSERVED = frozenset({"create", "initialize", "allocate", "upload", "download", "launch", "wait",
    "release", "close", "device_commit", "host_commit", "reconcile_device", "reconcile_host",
    "forget_definition", "forget_sections", "scratch_allocate", "scratch_free", "scratch_acquire",
    "scratch_release", "external_commit", "source_native_prepare", "source_native_start",
    "source_native_commit", "source_native_cancel", "graph_submit", "graph_kernel"})
HASH = re.compile(r"[0-9a-f]{64}\Z")
OPERATION_KINDS = frozenset({"source_call", "borrowed_call", "numerical_segment", "native_operation",
                             "external_device_task", "source_device_task"})


def _registry(manifests):
    records, supported = {}, False
    for manifest in manifests:
        mapping = manifest.get("runtime_provenance")
        if mapping is None:
            continue
        if not isinstance(mapping, dict) or type(mapping.get("schema_version")) is not int or mapping["schema_version"] != 1:
            raise ValueError("unsupported public runtime provenance version")
        if not isinstance(mapping.get("records"), list):
            raise TypeError("public runtime provenance requires identity records")
        supported = True
        local = set()
        for record in mapping["records"]:
            if not isinstance(record, dict) or not HASH.fullmatch(str(record.get("id", ""))) or not isinstance(record.get("kind"), str):
                raise ValueError("invalid public runtime provenance identity")
            identity = record["id"]
            if identity in local:
                raise ValueError("duplicate public runtime provenance identity")
            local.add(identity)
            if identity in records and records[identity] != record:
                raise ValueError("conflicting public runtime provenance identities across packages")
            records[identity] = record
    return records, supported


def _integer(fields, name, *, positive=False):
    value = fields.get(name, "")
    if not re.fullmatch(r"[0-9]+", value) or (positive and int(value) == 0):
        raise ValueError("invalid observed scoped provenance " + name)
    return int(value)


def _counters():
    return {"events": 0, "upload_bytes": 0, "download_bytes": 0, "kernel_launches": 0}


def _count(target, event, amount, compact=None):
    target["events"] += 1
    if event in {"upload", "download"}:
        target[event + "_bytes"] += amount
    elif event == "launch":
        target["kernel_launches"] += 1
    elif event in {"graph_submit", "graph_kernel"}:
        field = "graph_submissions" if event == "graph_submit" else "graph_kernel_submissions"
        target[field] = target.get(field, 0) + 1
    elif compact is not None:
        count_compact(target, compact)


class _GraphMembers:
    """Check observed member coverage without inventing historical members.

    Older runtimes emitted only aggregate submissions. Once member rows are
    present, that context must report the entire submitted block. Different
    contexts may interleave; the aggregate's source position is not a member's.
    """

    def __init__(self):
        self.pending = {}

    def _finish(self, context):
        block = self.pending.pop(context, None)
        if block is not None and block[1] not in (0, block[0]):
            raise ValueError("incomplete observed graph kernel coverage")

    def operation(self, context, event, amount, *, tagged):
        if event == "graph_kernel":
            if not tagged:
                raise ValueError("graph kernel requires runtime provenance version 1")
            if amount != 0:
                raise ValueError("graph kernel bytes must be zero")
            block = self.pending.get(context)
            if block is None or block[1] >= block[0]:
                raise ValueError("graph kernel lacks a matching observed submission")
            block[1] += 1
            return
        if context is not None:
            self._finish(context)
        if event == "graph_submit":
            if amount == 0:
                raise ValueError("graph submission requires positive member count")
            if tagged:
                self.pending[context] = [amount, 0]

    def finish(self):
        for context in tuple(self.pending):
            self._finish(context)

    def interruption(self, encoded):
        # Position/restore and other diagnostic rows are not counted, but a
        # same-context row still ends the immediately emitted member block.
        match = re.search(r"(?:^|\s)context=([1-9][0-9]*)(?:\s|$)", encoded)
        if match:
            self._finish(int(match[1]))


def _identities(fields, records, used):
    for field in FIELDS:
        identity = fields.get(field)
        if identity == "unknown":
            continue
        if not isinstance(identity, str) or not HASH.fullmatch(identity) or identity not in records:
            raise ValueError("observed scoped provenance identity absent from public manifest: " + field)
        kind = records[identity]["kind"]
        if (field == "operation" and kind not in OPERATION_KINDS) or (field != "operation" and kind != field):
            raise ValueError("observed scoped provenance identity has incorrect kind: " + field)
        used.add(identity)


def scoped_provenance(log, manifests):
    """Keep unknown/legacy events visible; never guess their source identity."""
    records, supported = _registry(manifests)
    groups, used, total = {}, set(), _counters()
    graph_members = _GraphMembers()
    domains = DomainTrace(lambda fields: _identities(fields, records, used))
    unknown = {field: _counters() for field in FIELDS}
    versioned, legacy = 0, 0
    pricing = []
    for line in log.splitlines():
        if line.startswith("FORT_SCOPED domain "):
            domains.lifecycle(line.removeprefix("FORT_SCOPED domain "))
            continue
        modeled = pricing_from_line(line)
        if modeled is not None:
            fields = pricing_provenance_fields(modeled)
            _identities(fields, records, used)
            domains.modeled(fields)
            pricing.append(modeled)
            continue
        compact = compact_from_line(line)
        if compact is not None:
            event, fields = "compact_transfer", provenance_fields(compact)
        else:
            match = re.match(r"^FORT_SCOPED ([a-z_]+) (.*)$", line)
            if not match:
                continue
            if match[1] not in OBSERVED:
                graph_members.interruption(match[2])
                continue
            event, encoded = match.groups()
            fields = {}
            for token in encoded.split():
                key, separator, value = token.partition("=")
                if not separator or key in fields:
                    raise ValueError("malformed observed scoped provenance fields")
                fields[key] = value
        amount = _integer(fields, "bytes")
        tagged = "provenance_version" in fields
        context = None
        if tagged:
            if fields["provenance_version"] != "1":
                raise ValueError("unsupported observed runtime provenance version")
            context = _integer(fields, "context", positive=True)
            if "buffer_handle" in fields:
                _integer(fields, "buffer_handle", positive=True)
            _identities(fields, records, used)
            if event in {"upload", "download"}:
                _integer(fields, "buffer_handle", positive=True)
                _integer(fields, "buffer", positive=True)
                _integer(fields, "generation")
            versioned += 1
        else:
            if any(field in fields for field in FIELDS) or "context" in fields:
                raise ValueError("observed source attribution lacks runtime provenance version")
            legacy += 1
        graph_members.operation(context, event, amount, tagged=tagged)
        domains.operation(fields, event, amount, tagged=tagged, compact=compact)
        identities = tuple(fields[field] if tagged else "unknown" for field in FIELDS)
        key = (tagged, *identities)
        if key not in groups:
            groups[key] = {"versioned": tagged, **dict(zip(FIELDS, identities, strict=True)),
                           **_counters(), "event_counts": {}, "contexts": set()}
        group = groups[key]
        if context is not None:
            group["contexts"].add(context)
        group["event_counts"][event] = group["event_counts"].get(event, 0) + 1
        _count(group, event, amount, compact)
        _count(total, event, amount, compact)
        for field, identity in zip(FIELDS, identities, strict=True):
            if identity == "unknown":
                _count(unknown[field], event, amount, compact)
    graph_members.finish()
    for group in groups.values():
        group["contexts"] = sorted(group["contexts"])
    result = {"schema_version": 1, "available": supported and versioned > 0,
            "public_mapping_available": supported, "versioned_events": versioned, "legacy_events": legacy,
            "totals": total, "unattributed_fields": unknown,
            "groups": [groups[key] for key in sorted(groups)],
            "records": [records[identity] for identity in sorted(used)],
            "timing_sample": False,
            "semantics": ("actual API submissions, submitted logical graph members and coherence commits; no device durations or policy inference"
                          if "graph_kernel_submissions" in total else
                          "actual API submissions and coherence commits; no device durations or policy inference")}
    if domains.available:
        result["residency_domains"] = domains.report()
    if pricing:
        result["modeled_copy_pricing"] = pricing
    if "copy_dma_calls" in total:
        result["physical_copy_semantics"] = (
            "compact_physical_operations counts original physical rectangles before batching; "
            "copy_dma_calls and copy_kernel_launches count transport work; "
            "payload bytes remain in upload/download events and numerical launch counts exclude copy kernels")
    return result
