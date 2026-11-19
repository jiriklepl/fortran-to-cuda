"""Validate explicit storage ownership without inferring placement or timing."""
from __future__ import annotations

import json
import re

from .scoped_compact_trace import count_compact

_SOURCE = ("owner", "procedure", "segment", "operation", "implementation", "boundary")
_FIELDS = {"schema_version", "event", "domain", "context", "physical_context", "last_context", "device", "failed", *_SOURCE}


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate residency domain trace field")
        result[key] = value
    return result


def _number(row, field, *, positive=False):
    value = row.get(field)
    if type(value) is not int or not (int(positive) <= value < 2**64):
        raise ValueError("invalid residency domain " + field)
    return value


def _counts():
    return {"events": 0, "upload_bytes": 0, "download_bytes": 0, "kernel_launches": 0}


class DomainTrace:
    """One active borrower per domain, as required by residency ABI v1.

    An unfinished trace remains explicitly incomplete. Contradictory identities
    or lifecycle transitions reject the diagnostic instead of silently merging
    unrelated execution and storage contexts.
    """

    def __init__(self, validate_identities):
        self._validate_identities = validate_identities
        self._domains = {}
        self._contexts = {}

    @property
    def available(self):
        return bool(self._domains)

    def lifecycle(self, encoded):
        try:
            row = json.loads(encoded, object_pairs_hook=_object)
        except (json.JSONDecodeError, TypeError) as error:
            raise ValueError("malformed residency domain lifecycle") from error
        if (not isinstance(row, dict) or set(row) != _FIELDS
                or type(row["schema_version"]) is not int or row["schema_version"] != 1
                or type(row["failed"]) is not bool):
            raise ValueError("unsupported residency domain lifecycle record")
        domain = _number(row, "domain", positive=True)
        context = _number(row, "context")
        physical = _number(row, "physical_context", positive=True)
        last = _number(row, "last_context")
        device = _number(row, "device")
        if physical != domain or context == domain:
            raise ValueError("inconsistent physical residency domain identity")
        self._validate_identities(row)
        event = row["event"]
        if event == "domain_create":
            if domain in self._domains or context or last or row["failed"]:
                raise ValueError("invalid residency domain creation")
            self._domains[domain] = {
                "domain": domain, "physical_context": physical, "device": device,
                "contexts": [], "active_context": 0, "last_context": 0,
                "lifecycle": [], "terminal": None, "failed": False,
                "execution": _counts(), "storage": _counts(), "pending_create": 0,
            }
        state = self._domains.get(domain)
        if state is None or state["terminal"] is not None or device != state["device"]:
            raise ValueError("unknown, retired or inconsistent residency domain")
        if state["failed"] and not row["failed"]:
            raise ValueError("residency domain failure state cannot reset")
        if event == "domain_attach":
            if (not context or state["active_context"] or context in self._contexts
                    or last != state["last_context"] or row["failed"]
                    or state["pending_create"] not in (0, context)):
                raise ValueError("invalid residency domain attachment")
            state["pending_create"] = 0
            state["active_context"] = context
            state["contexts"].append(context)
            self._contexts[context] = domain
        elif event == "domain_detach":
            if context != state["active_context"] or not context or last != context:
                raise ValueError("invalid residency domain detachment")
            state["active_context"] = 0
            state["last_context"] = context
        elif event in {"domain_retire", "domain_abandon"}:
            if (context or state["active_context"] or state["pending_create"]
                    or last != state["last_context"]
                    or row["failed"] != (event == "domain_abandon")):
                raise ValueError("invalid residency domain retirement")
            state["terminal"] = event
        elif event != "domain_create":
            raise ValueError("unsupported residency domain lifecycle event")
        state["failed"] = row["failed"]
        state["lifecycle"].append(row)

    def _attribution(self, fields, event, *, tagged):
        keys = {"domain_version", "domain", "domain_role"}
        if not keys & fields.keys():
            return
        if not tagged or not keys <= fields.keys() or fields["domain_version"] != "1":
            raise ValueError("unsupported observed residency domain attribution")
        if not re.fullmatch(r"[1-9][0-9]*", fields["domain"]):
            raise ValueError("invalid observed residency domain identity")
        domain = int(fields["domain"])
        state = self._domains.get(domain)
        role = fields["domain_role"]
        context = int(fields["context"])
        if state is None or state["terminal"] is not None or role not in {"execution", "storage"}:
            raise ValueError("unknown or retired observed residency domain")
        if role == "storage":
            if context != domain or state["active_context"] or state["pending_create"]:
                raise ValueError("storage publication conflicts with a residency borrower")
        elif event == "create":
            # Runtime emits the context create immediately before its attach.
            if state["active_context"] or state["pending_create"] or context in self._contexts or context == domain:
                raise ValueError("invalid residency borrower creation")
            state["pending_create"] = context
        elif context != state["active_context"]:
            raise ValueError("operation belongs to a detached residency borrower")
        return state, role

    def modeled(self, fields):
        """Check a decision's active attachment without counting an operation."""
        target = self._attribution(fields, "copy_pricing", tagged=True)
        if target is not None and int(fields["device"]) != target[0]["device"]:
            raise ValueError("modeled copy pricing has inconsistent residency device")

    def operation(self, fields, event, amount, *, tagged, compact=None):
        target = self._attribution(fields, event, tagged=tagged)
        if target is None:
            return
        state, role = target
        counts = state[role]
        counts["events"] += 1
        if event in {"upload", "download"}:
            counts[event + "_bytes"] += amount
        elif event == "launch":
            counts["kernel_launches"] += 1
        elif event in {"graph_submit", "graph_kernel"}:
            field = "graph_submissions" if event == "graph_submit" else "graph_kernel_submissions"
            counts[field] = counts.get(field, 0) + 1
        elif compact is not None:
            count_compact(counts, compact)

    def report(self):
        domains = []
        for state in self._domains.values():
            domains.append({key: value for key, value in state.items() if key != "pending_create"})
        return {"schema_version": 1, "complete": all(row["terminal"] is not None for row in domains),
                "domains": domains, "timing_sample": False,
                "semantics": "explicit storage/borrower identities and physical copy submissions; no placement or lifetime authority"}
