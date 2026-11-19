"""Exact physical compact-copy evidence, distinct from numerical kernel work."""
from __future__ import annotations

import json
import re

SOURCE_FIELDS = ("owner", "procedure", "segment", "operation", "implementation", "boundary")
COUNTERS = {
    "compact_batches": None,
    "copy_dma_calls": "dma_calls",
    "copy_kernel_launches": "copy_kernel_launches",
    "compact_logical_rectangles": "logical_rectangles",
    "compact_physical_operations": "physical_operations",
}
DOMAIN_FIELDS = {"domain_version", "domain", "domain_role"}
FIELDS = {"schema_version", "context", "event", "version", "buffer", "identity", "generation",
          "upload", "bytes", "logical_rectangles", "physical_operations", "dma_calls",
          "copy_kernel_launches", "word_bytes", "workspace_bytes", "reused", *SOURCE_FIELDS}


def _integer(record, name, *, positive=True):
    value = record.get(name)
    if type(value) is not int or not int(positive) <= value < 2**64:
        raise ValueError("compact transfer requires a bounded integer " + name)
    return value


def validate_compact_transfer(record):
    expected = FIELDS | (DOMAIN_FIELDS if DOMAIN_FIELDS & record.keys() else set())
    if (record.keys() != expected or type(record.get("schema_version")) is not int
            or record["schema_version"] != 1 or record.get("event") != "compact_transfer"
            or type(record.get("version")) is not int or record["version"] != 1):
        raise ValueError("compact transfer requires exact version1 public fields")
    for name in ("context", "buffer", "identity", "generation", "bytes", "logical_rectangles",
                 "physical_operations", "dma_calls", "copy_kernel_launches", "word_bytes", "workspace_bytes"):
        _integer(record, name)
    if type(record["upload"]) is not bool or type(record["reused"]) is not bool:
        raise ValueError("compact transfer direction and reuse must be boolean")
    operations = record["physical_operations"]
    if (not 2 <= operations <= 256 or not 1 <= record["logical_rectangles"] <= operations
            or record["dma_calls"] != 1 or record["copy_kernel_launches"] != (operations + 47) // 48
            or record["word_bytes"] not in (1, 4, 8)
            or record["bytes"] % record["word_bytes"] or record["bytes"] < operations * record["word_bytes"]
            or record["workspace_bytes"] not in (512 * 1024, 2 * 1024**2, 8 * 1024**2, 32 * 1024**2)
            or record["bytes"] > record["workspace_bytes"] // 2):
        raise ValueError("compact transfer has inconsistent physical copy accounting")
    if DOMAIN_FIELDS & record.keys():
        _integer(record, "domain")
        if (type(record["domain_version"]) is not int or record["domain_version"] != 1
                or record["domain_role"] not in ("storage", "execution")
                or (record["context"] == record["domain"]) != (record["domain_role"] == "storage")):
            raise ValueError("compact transfer requires consistent residency domain attribution")
    if any(not isinstance(record[name], str)
           or record[name] != "unknown" and not re.fullmatch(r"[0-9a-f]{64}", record[name]) for name in SOURCE_FIELDS):
        raise ValueError("compact transfer requires exact source identity fields")
    return record


def compact_from_line(line):
    """Recognize this JSON event without changing legacy handling of other rows."""
    prefix = "FORT_SCOPED evidence "
    if not line.startswith(prefix):
        return None

    def fields(pairs):
        result = {}
        for name, value in pairs:
            if name in result:
                raise ValueError("duplicate compact transfer field")
            result[name] = value
        return result

    try:
        record = json.loads(line[len(prefix):], object_pairs_hook=fields)
    except ValueError:
        if re.search(r'"event"\s*:\s*"compact_transfer"', line):
            raise
        return None
    if not isinstance(record, dict) or record.get("event") != "compact_transfer":
        return None
    return validate_compact_transfer(record)


def provenance_fields(record):
    """Convert exact JSON attribution into the existing checked trace protocol."""
    fields = {name: record[name] for name in SOURCE_FIELDS}
    fields.update(provenance_version="1", context=str(record["context"]), bytes=str(record["bytes"]),
                  buffer=str(record["identity"]), buffer_handle=str(record["buffer"]), generation=str(record["generation"]))
    if DOMAIN_FIELDS & record.keys():
        fields.update({name: str(record[name]) for name in DOMAIN_FIELDS})
    return fields


def count_compact(target, record):
    for output, field in COUNTERS.items():
        target[output] = target.get(output, 0) + (record[field] if field else 1)


def validate_modeled_compact_copy(record):
    fields = {"transport", "physical_operations", "copy_kernel_launches"}
    if not fields & record.keys():
        return
    if (record.get("event") != "copy" or not fields <= record.keys() or record["transport"] != "compact"
            or type(record.get("copy_calls")) is not int or record["copy_calls"] != 1):
        raise ValueError("modeled compact copy requires complete transport fields")
    operations = _integer(record, "physical_operations")
    kernels = _integer(record, "copy_kernel_launches")
    _integer(record, "bytes")
    if not 2 <= operations <= 256 or kernels != (operations + 47) // 48:
        raise ValueError("modeled compact copy has inconsistent physical copy accounting")
