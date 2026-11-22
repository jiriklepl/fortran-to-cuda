"""Small compiler-owned identities for optional actual submission diagnostics."""

import json
from hashlib import sha256


class Provenance:
    def __init__(self):
        self.records = {}

    def add(self, kind, **facts):
        record = {"kind": kind, **facts}
        identity = sha256(json.dumps({"schema_version": 1, **record}, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()
        self.records[identity] = {"id": identity, **record}
        return identity

    def public(self, *, available=True, additional=()):
        records = {**self.records}
        for mapping in additional:
            for record in mapping.get("records", ()):
                records[record["id"]] = record
        return {"schema_version": 1, "available": available, "activation": "FORT_RUNTIME_TRACE=1",
                "runtime_api": ["fort_scope_trace_set_v1", "fort_scope_trace_restore_v1"],
                "records": [records[key] for key in sorted(records)], "unknown": "unknown",
                "semantics": "actual API submission and completed coherence commits; no physical execution timestamps",
                "batch_worker_attribution": "unavailable unless the individual implementation is explicitly tagged",
                "initial_context_creation": "unknown until the original source coordinator binds its owner",
                "execution_error_attribution": "unavailable; existing numerical error reporting is unchanged"}
