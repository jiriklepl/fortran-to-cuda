"""Selected modeled copy prices; never evidence of submitted transport work."""
from __future__ import annotations

import json
import math
import re

from .scoped_compact_trace import DOMAIN_FIELDS, SOURCE_FIELDS

FIELDS = {"schema_version", "context", "event", "version", "evidence", "phase", "confirms_execution",
          "buffer", "identity", "generation", "device", "upload", "decision", "capacity_bytes",
          "direct_seconds", "selected_seconds", "cold_seconds", "reuse_seconds", "prior_credit_seconds",
          "startup_premium_seconds", "state_revision", "state_generation", "model_key", *SOURCE_FIELDS}


def _uint(value, *, positive=False):
    return type(value) is int and int(positive) <= value < 2**64


def _seconds(value, *, nullable=False, positive=True):
    if nullable and value is None:
        return True
    try:
        return (type(value) in (int, float) and math.isfinite(value)
                and (value > 0 if positive else value >= 0))
    except OverflowError:
        return False


def validate_copy_pricing(record):
    expected = FIELDS | (DOMAIN_FIELDS if DOMAIN_FIELDS & record.keys() else set())
    if (record.keys() != expected or type(record.get("schema_version")) is not int
            or record["schema_version"] != 1 or record.get("event") != "copy_pricing"
            or type(record.get("version")) is not int or record["version"] != 1
            or record["evidence"] != "MODELED" or record["phase"] != "before_submission"
            or record["confirms_execution"] is not False):
        raise ValueError("copy pricing requires exact modeled version1 fields")
    if (any(not _uint(record[name], positive=True) for name in ("context", "buffer", "identity", "generation"))
            or any(not _uint(record[name]) for name in ("state_revision", "state_generation"))
            or type(record["device"]) is not int or not 0 <= record["device"] < 2**31
            or type(record["upload"]) is not bool
            or type(record["capacity_bytes"]) is not int
            or record["capacity_bytes"] not in (256 * 1024, 1024**2, 4 * 1024**2, 16 * 1024**2)):
        raise ValueError("copy pricing requires bounded identities, capacity and direction")
    model = record["model_key"]
    if (type(model) is not list or len(model) != 155 or any(not _uint(word) for word in model)
            or model[:4] != [record["device"], 3, 1, 4602678819172646912]):
        raise ValueError("copy pricing requires the exact complete V3 model key; rebuild older traces")
    if (any(not _seconds(record[name]) for name in ("direct_seconds", "selected_seconds"))
            or any(not _seconds(record[name], nullable=True) for name in ("cold_seconds", "reuse_seconds"))
            or any(not _seconds(record[name], nullable=True, positive=False)
                   for name in ("prior_credit_seconds", "startup_premium_seconds"))):
        raise ValueError("copy pricing requires finite positive prices and nonnegative credit")
    cold, reuse, prior, premium = (record[name] for name in
                                  ("cold_seconds", "reuse_seconds", "prior_credit_seconds", "startup_premium_seconds"))
    expected_premium = max(0, cold - reuse) if cold is not None and reuse is not None else None
    if premium != expected_premium:
        raise ValueError("copy pricing has inconsistent startup premium")
    choice, selected, threshold = record["decision"], record["selected_seconds"], record["direct_seconds"] * .5
    if choice == "credit_funded_cold":
        valid = (cold is not None and reuse is not None and selected == cold and prior is not None
                 and prior >= premium and cold >= threshold and reuse < threshold)
    elif choice == "locally_profitable_cold":
        valid = cold is not None and selected == cold and cold < threshold
    elif choice == "reuse":
        valid = reuse is not None and selected == reuse and reuse < threshold and prior is None
    else:
        valid = False
    if not valid:
        raise ValueError("copy pricing has inconsistent selected decision")
    if any(type(record[name]) is not str or record[name] != "unknown"
           and not re.fullmatch(r"[0-9a-f]{64}", record[name]) for name in SOURCE_FIELDS):
        raise ValueError("copy pricing requires exact source identity fields")
    if DOMAIN_FIELDS & record.keys() and (not _uint(record["domain"], positive=True) or type(record["domain_version"]) is not int
                or record["domain_version"] != 1 or record["domain_role"] not in ("storage", "execution")
                or (record["context"] == record["domain"]) != (record["domain_role"] == "storage")):
        raise ValueError("copy pricing requires complete residency domain attribution")
    return record


def pricing_from_line(line):
    prefix = "FORT_SCOPED evidence "
    if not line.startswith(prefix):
        return None

    def fields(pairs):
        result = {}
        for name, value in pairs:
            if name in result:
                raise ValueError("duplicate copy pricing field")
            result[name] = value
        return result

    try:
        record = json.loads(line[len(prefix):], object_pairs_hook=fields)
    except ValueError:
        if re.search(r'"event"\s*:\s*"copy_pricing"', line):
            raise
        return None
    if not isinstance(record, dict) or record.get("event") != "copy_pricing":
        return None
    return validate_copy_pricing(record)


def provenance_fields(record):
    """Adapt attribution only, without adding physical byte or count fields."""
    fields = {name: record[name] for name in SOURCE_FIELDS}
    fields.update(context=str(record["context"]), device=str(record["device"]))
    if DOMAIN_FIELDS & record.keys():
        fields.update({name: str(record[name]) for name in DOMAIN_FIELDS})
    return fields
