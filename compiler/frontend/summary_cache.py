"""Bounded JSON storage for complete source-effect closures.

The caller owns the source, configuration, contract and capture-proof identity,
and must re-admit its analysis budgets before using a cached closure. This cache
stores no parser objects and grants no authority to an imported summary.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections import OrderedDict
from contextlib import suppress
from hashlib import sha256
from pathlib import Path
from threading import RLock

_SCHEMA_VERSION = 1
_FILE_PREFIX = "fort_source_summary_v1_"
_MAX_BYTES = 8 * 1024 * 1024
_MAX_DEPTH = 128
_MAX_NODES = 1_000_000
_CACHE_ERRORS = (OSError, ValueError, TypeError, RecursionError, OverflowError, UnicodeError, MemoryError)


def _plain_json(value, depth=0, active=None, budget=None):
    """Validate without coercing parser objects or custom container classes."""
    if active is None:
        active, budget = set(), [_MAX_NODES]
    budget[0] -= 1
    if depth > _MAX_DEPTH or budget[0] < 0:
        raise ValueError("summary cache JSON exceeds structural bounds")
    kind = type(value)
    if kind is str and len(value) > _MAX_BYTES:
        raise ValueError("summary cache JSON exceeds byte limit")
    if value is None or kind in (str, bool, int):
        return
    if kind is float:
        if not math.isfinite(value):
            raise ValueError("summary cache JSON requires finite numbers")
        return
    if kind not in (dict, list, tuple):
        raise TypeError("summary cache accepts JSON values only")
    marker = id(value)
    if marker in active:
        raise ValueError("summary cache JSON is cyclic")
    active.add(marker)
    try:
        if kind is dict:
            for key, item in value.items():
                if type(key) is not str:
                    raise TypeError("summary cache JSON keys must be strings")
                if len(key) > _MAX_BYTES:
                    raise ValueError("summary cache JSON exceeds byte limit")
                _plain_json(item, depth + 1, active, budget)
        else:
            for item in value:
                _plain_json(item, depth + 1, active, budget)
    finally:
        active.remove(marker)


def _canonical(value):
    _plain_json(value)
    encoder = json.JSONEncoder(ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))
    chunks, size = [], 0
    for chunk in encoder.iterencode(value):
        size += len(chunk)
        if size > _MAX_BYTES:
            raise ValueError("summary cache JSON exceeds byte limit")
        chunks.append(chunk)
    return "".join(chunks).encode("ascii")


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("summary cache JSON contains duplicate keys")
        result[key] = value
    return result


def _decode(data):
    result = json.loads(data, object_pairs_hook=_object)
    _plain_json(result)
    return result


def _payload(payload, requested):
    if type(payload) is not dict or set(payload) not in ({"summaries", "operations"}, {"summaries", "operations", "order"}):
        raise ValueError("summary cache requires a closure payload")
    summaries, operations = payload["summaries"], payload["operations"]
    if type(summaries) is not dict or requested not in summaries:
        raise ValueError("summary cache requires its requested procedure")
    if type(operations) is not int or operations < 0:
        raise ValueError("summary cache requires a nonnegative operation count")
    if "order" in payload:
        order = payload["order"]
        if (type(order) is not list or any(type(procedure) is not str for procedure in order)
                or len(order) != len(summaries) or len(set(order)) != len(order) or set(order) != set(summaries)):
            raise ValueError("summary cache requires the original distinct procedure order")
    for procedure, summary in summaries.items():
        if type(procedure) is not str or not procedure or type(summary) is not dict:
            raise ValueError("summary cache requires procedure summaries")
        if summary.get("procedure") != procedure or summary.get("complete") is not True:
            raise ValueError("summary cache stores complete closures only")
        if type(summary.get("operations")) not in (list, tuple):
            raise ValueError("summary cache requires ordered summary operations")
        if any(type(operation) is not dict or type(operation.get("kind")) is not str or not operation["kind"]
               for operation in summary["operations"]):
            raise ValueError("summary cache requires structured summary operations")
        if type(summary.get("closure_depth")) is not int or summary["closure_depth"] < 1:
            raise ValueError("summary cache requires a positive closure depth")


class SummaryCache:
    """A local LRU, optionally backed by atomic files in an explicit directory.

    Reuse an instance across analyses for a shared memory-only cache. Files are
    read by exact key; the cache never scans or removes other directory entries.
    Every file and memory payload is bounded by eight MiB. ``max_entries`` bounds
    memory entries; an explicit disk directory retains its keyed files until its
    owner removes them. Returned closures and statistics are independent copies.
    """

    def __init__(self, directory=None, max_entries=32):
        self.max_entries = max_entries if type(max_entries) is int and max_entries >= 0 else 0
        self.directory = None
        self._memory = OrderedDict()
        self._lock = RLock()
        self._stats = dict.fromkeys(("hits", "misses", "rejected", "stores", "memory_hits", "disk_hits", "disk_writes"), 0)
        if directory is not None:
            try:
                self.directory = Path(directory)
            except _CACHE_ERRORS:
                self._stats["rejected"] += 1

    @property
    def stats(self):
        with self._lock:
            return dict(self._stats)

    @staticmethod
    def _identity(identity, requested):
        if type(identity) is not dict or type(requested) is not str or not requested:
            raise ValueError("summary cache requires an identity object and requested procedure")
        identity_digest = sha256(_canonical(identity)).hexdigest()
        key = sha256(_canonical({"schema_version": _SCHEMA_VERSION, "identity_sha256": identity_digest,
                                 "requested": requested})).hexdigest()
        return identity_digest, key

    def _remember(self, key, encoded):
        if not self.max_entries:
            return
        self._memory[key] = encoded
        self._memory.move_to_end(key)
        while len(self._memory) > self.max_entries:
            self._memory.popitem(last=False)

    def _filename(self, key):
        return self.directory / (_FILE_PREFIX + key + ".json")

    def lookup(self, identity, requested):
        with self._lock:
            try:
                identity_digest, key = self._identity(identity, requested)
                encoded = self._memory.get(key)
                if encoded is not None:
                    result = _decode(encoded)
                    self._memory.move_to_end(key)
                    self._stats["hits"] += 1
                    self._stats["memory_hits"] += 1
                    return result
                if self.directory is not None:
                    try:
                        with self._filename(key).open("rb") as handle:
                            data = handle.read(_MAX_BYTES + 1)
                    except FileNotFoundError:
                        data = None
                    if data is not None:
                        if len(data) > _MAX_BYTES:
                            raise ValueError("summary cache file exceeds byte limit")
                        record = _decode(data)
                        required = {"schema_version", "identity_sha256", "requested", "key", "payload_sha256", "payload"}
                        if type(record) is not dict or set(record) != required:
                            raise ValueError("summary cache record schema is invalid")
                        if (type(record["schema_version"]) is not int or record["schema_version"] != _SCHEMA_VERSION
                                or record["identity_sha256"] != identity_digest or record["requested"] != requested
                                or record["key"] != key):
                            raise ValueError("summary cache record identity is stale")
                        _payload(record["payload"], requested)
                        encoded = _canonical(record["payload"])
                        if record["payload_sha256"] != sha256(encoded).hexdigest():
                            raise ValueError("summary cache payload digest is invalid")
                        self._remember(key, encoded)
                        self._stats["hits"] += 1
                        self._stats["disk_hits"] += 1
                        return _decode(encoded)
            except _CACHE_ERRORS:
                self._stats["rejected"] += 1
            self._stats["misses"] += 1
            return None

    def store(self, identity, requested, payload):
        with self._lock:
            try:
                identity_digest, key = self._identity(identity, requested)
                encoded = _canonical(payload)
                _payload(payload, requested)
                record = _canonical({"schema_version": _SCHEMA_VERSION, "identity_sha256": identity_digest,
                                     "requested": requested, "key": key, "payload_sha256": sha256(encoded).hexdigest(),
                                     "payload": _decode(encoded)})
                self._remember(key, encoded)
                self._stats["stores"] += 1
                if self.directory is not None:
                    temporary = None
                    try:
                        self.directory.mkdir(parents=True, exist_ok=True)
                        with tempfile.NamedTemporaryFile(mode="wb", dir=self.directory, prefix=_FILE_PREFIX,
                                                         suffix=".tmp", delete=False) as handle:
                            temporary = Path(handle.name)
                            handle.write(record)
                        os.replace(temporary, self._filename(key))
                        temporary = None
                        self._stats["disk_writes"] += 1
                    finally:
                        if temporary is not None:
                            with suppress(OSError):
                                temporary.unlink()
            except _CACHE_ERRORS:
                self._stats["rejected"] += 1
