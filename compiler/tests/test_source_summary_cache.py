"""JSON cache primitives remain independent of parser trees and source proofs."""

import copy
import json
from hashlib import sha256

import pytest

from compiler.frontend.summary_cache import SummaryCache


def identity():
    return {"analysis_version": 2, "sources": {"source.f90": "original"},
            "configuration": {"precision": "real64"}, "contracts": {"library": {"identity": "version1"}},
            "budgets": {"depth": 8, "procedures": 64, "operations": 256},
            "capture_authority": ["module::allocation"], "role": "original"}


def payload(requested="module::root"):
    return {"summaries": {
        requested: {"procedure": requested, "complete": True, "cloneable": True,
                    "operations": [{"kind": "call", "procedure": "module::leaf", "guard": ("valid",)}],
                    "closure_depth": 2},
        "module::leaf": {"procedure": "module::leaf", "complete": True, "cloneable": True,
                         "operations": [{"kind": "read", "resource": "argument::a", "guard": ()}],
                         "closure_depth": 1}},
        "operations": 2, "order": [requested, "module::leaf"]}


def disk_record(directory):
    files = list(directory.glob("fort_source_summary_v2_*.json"))
    assert len(files) == 1
    return files[0]


def write_record(path, record):
    path.write_text(json.dumps(record))


def refresh_payload_hash(record):
    canonical = json.dumps(record["payload"], ensure_ascii=True, allow_nan=False,
                           sort_keys=True, separators=(",", ":")).encode("ascii")
    record["payload_sha256"] = sha256(canonical).hexdigest()


def test_memory_cache_returns_independent_json_copies():
    cache = SummaryCache()
    original, key = payload(), identity()
    cache.store(key, "module::root", original)
    original["summaries"]["module::root"]["operations"].clear()
    first = cache.lookup(key, "module::root")
    assert first["summaries"]["module::root"]["operations"][0]["guard"] == ["valid"]
    assert first["order"] == ["module::root", "module::leaf"]
    first["summaries"]["module::leaf"]["complete"] = False
    first["order"].reverse()
    second = cache.lookup(key, "module::root")
    assert second["summaries"]["module::leaf"]["complete"]
    assert second["order"] == ["module::root", "module::leaf"]
    snapshot = cache.stats
    snapshot["hits"] = -1
    assert cache.stats["hits"] == 2
    assert cache.stats["memory_hits"] == 2


@pytest.mark.parametrize("field", ["analysis_version", "sources", "configuration", "contracts", "budgets",
                                   "capture_authority", "role"])
def test_changed_source_config_contract_budget_capture_or_role_misses(field):
    cache = SummaryCache()
    original = identity()
    cache.store(original, "module::root", payload())
    changed = copy.deepcopy(original)
    changed[field] = {"changed": True}
    assert cache.lookup(changed, "module::root") is None
    assert cache.lookup(original, "module::root") is not None
    assert cache.stats["misses"] == 1


def test_identity_and_requested_name_are_immutable_key_inputs():
    cache = SummaryCache()
    key = identity()
    original = copy.deepcopy(key)
    cache.store(key, "module::root", payload())
    key["sources"]["source.f90"] = "modified"
    assert cache.lookup(key, "module::root") is None
    assert cache.lookup(original, "module::other") is None
    assert cache.lookup(original, "module::root") is not None


def test_lru_evicts_only_oldest_memory_entry():
    cache = SummaryCache(max_entries=2)
    for name in ("first", "second"):
        cache.store({"source": name}, "module::root", payload())
    assert cache.lookup({"source": "first"}, "module::root") is not None
    cache.store({"source": "third"}, "module::root", payload())
    assert cache.lookup({"source": "second"}, "module::root") is None
    assert cache.lookup({"source": "first"}, "module::root") is not None
    assert cache.lookup({"source": "third"}, "module::root") is not None


def test_disk_cache_survives_instance_and_canonical_identity_order(tmp_path):
    unrelated = tmp_path / "keep.txt"
    unrelated.write_text("unrelated")
    cache = SummaryCache(tmp_path)
    key = identity()
    cache.store(key, "module::root", payload())
    assert cache.stats["disk_writes"] == 1
    saved = disk_record(tmp_path)
    suffix = saved.stem.removeprefix("fort_source_summary_v2_")
    assert len(suffix) == 64
    assert set(suffix) <= set("0123456789abcdef")
    imported = SummaryCache(tmp_path)
    result = imported.lookup(dict(reversed(list(key.items()))), "module::root")
    assert result["order"] == ["module::root", "module::leaf"]
    assert imported.stats["disk_hits"] == 1
    result["summaries"].clear()
    assert imported.lookup(key, "module::root")["summaries"]
    assert unrelated.read_text() == "unrelated"
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("change", ["schema", "identity", "requested", "key", "digest", "payload", "incomplete",
                                    "wrong_procedure", "operation_type", "order", "extra_field"])
def test_corrupt_stale_or_invalid_disk_records_are_misses(tmp_path, change):
    cache = SummaryCache(tmp_path)
    cache.store(identity(), "module::root", payload())
    path = disk_record(tmp_path)
    record = json.loads(path.read_bytes())
    if change == "schema":
        record["schema_version"] = True
    elif change == "identity":
        record["identity_sha256"] = "different"
    elif change == "requested":
        record["requested"] = "module::other"
    elif change == "key":
        record["key"] = "different"
    elif change == "digest":
        record["payload_sha256"] = "different"
    elif change == "payload":
        record["payload"]["operations"] += 1
    elif change == "incomplete":
        record["payload"]["summaries"]["module::leaf"]["complete"] = False
        refresh_payload_hash(record)
    elif change == "wrong_procedure":
        record["payload"]["summaries"]["module::leaf"]["procedure"] = "module::other"
        refresh_payload_hash(record)
    elif change == "operation_type":
        record["payload"]["operations"] = True
        refresh_payload_hash(record)
    elif change == "order":
        record["payload"]["order"] = ["module::root", "module::root"]
        refresh_payload_hash(record)
    else:
        record["extra"] = "unknown schema field"
    write_record(path, record)
    imported = SummaryCache(tmp_path)
    assert imported.lookup(identity(), "module::root") is None
    assert imported.stats["rejected"] == 1
    assert imported.stats["misses"] == 1


@pytest.mark.parametrize("data", [b"not json", b'{"schema_version":1,"schema_version":1}', b'{}', b'[]', b'null'])
def test_malformed_disk_json_is_a_miss(tmp_path, data):
    cache = SummaryCache(tmp_path)
    cache.store(identity(), "module::root", payload())
    disk_record(tmp_path).write_bytes(data)
    imported = SummaryCache(tmp_path)
    assert imported.lookup(identity(), "module::root") is None
    assert imported.stats["rejected"] == 1


def test_oversized_disk_read_and_payload_store_are_bounded(tmp_path):
    cache = SummaryCache(tmp_path)
    cache.store(identity(), "module::root", payload())
    path = disk_record(tmp_path)
    path.write_bytes(b" " * (8 * 1024 * 1024 + 1))
    imported = SummaryCache(tmp_path)
    assert imported.lookup(identity(), "module::root") is None
    assert imported.stats["rejected"] == 1
    large = payload()
    large["summaries"]["module::root"]["extra"] = "x" * (8 * 1024 * 1024)
    cache.store({"source": "large"}, "module::root", large)
    assert cache.lookup({"source": "large"}, "module::root") is None
    assert len(list(tmp_path.glob("fort_source_summary_v2_*.json"))) == 1


@pytest.mark.parametrize("invalid", ["incomplete", "operations", "operation_structure", "missing_requested", "order", "depth", "ast",
                                     "nonfinite", "non_string_key", "cyclic"])
def test_invalid_or_non_json_payload_is_never_stored(invalid):
    cache = SummaryCache()
    value = payload()
    if invalid == "incomplete":
        value["summaries"]["module::leaf"]["complete"] = False
    elif invalid == "operations":
        value["operations"] = -1
    elif invalid == "operation_structure":
        value["summaries"]["module::root"]["operations"] = ["not an operation"]
    elif invalid == "missing_requested":
        del value["summaries"]["module::root"]
    elif invalid == "order":
        value["order"] = ["module::root", "module::other"]
    elif invalid == "depth":
        value["summaries"]["module::root"]["closure_depth"] = False
    elif invalid == "ast":
        value["summaries"]["module::root"]["extra"] = object()
    elif invalid == "nonfinite":
        value["summaries"]["module::root"]["extra"] = float("inf")
    elif invalid == "non_string_key":
        value["summaries"]["module::root"]["extra"] = {1: "value"}
    else:
        value["summaries"]["module::root"]["extra"] = value
    cache.store(identity(), "module::root", value)
    assert cache.lookup(identity(), "module::root") is None
    assert cache.stats["rejected"] == 1
    assert cache.stats["stores"] == 0


def test_failed_disk_write_preserves_memory_and_removes_only_owned_temp(tmp_path, monkeypatch):
    unrelated = tmp_path / "not_a_cache.tmp"
    unrelated.write_text("keep")

    def fail_replace(*_args):
        raise OSError("simulated atomic replacement failure")

    monkeypatch.setattr("compiler.frontend.summary_cache.os.replace", fail_replace)
    cache = SummaryCache(tmp_path)
    cache.store(identity(), "module::root", payload())
    assert cache.lookup(identity(), "module::root") is not None
    assert cache.stats["rejected"] == 1
    assert cache.stats["disk_writes"] == 0
    assert list(tmp_path.iterdir()) == [unrelated]


def test_disk_directory_failure_is_nonfatal_and_memory_remains_usable(tmp_path):
    path = tmp_path / "regular_file"
    path.write_text("keep")
    cache = SummaryCache(path)
    cache.store(identity(), "module::root", payload())
    assert cache.lookup(identity(), "module::root") is not None
    assert path.read_text() == "keep"
    assert cache.stats["rejected"] == 1


def test_zero_memory_budget_supports_explicit_disk_only_cache(tmp_path):
    cache = SummaryCache(tmp_path, max_entries=0)
    cache.store(identity(), "module::root", payload())
    assert cache.lookup(identity(), "module::root") is not None
    assert cache.stats["memory_hits"] == 0
    assert cache.stats["disk_hits"] == 1


def test_disk_miss_does_not_create_directory(tmp_path):
    path = tmp_path / "not_created"
    cache = SummaryCache(path)
    assert cache.lookup(identity(), "module::root") is None
    assert not path.exists()
    assert cache.stats["rejected"] == 0


def test_order_is_optional_for_clients_without_ordered_public_procedures():
    cache = SummaryCache()
    value = payload()
    del value["order"]
    cache.store(identity(), "module::root", value)
    assert cache.lookup(identity(), "module::root")["operations"] == 2


@pytest.mark.parametrize("invalid", [None, [], {"parser": object()}, {"nonfinite": float("nan")}, {1: "invalid"}])
def test_invalid_identity_is_a_nonfatal_miss(invalid):
    cache = SummaryCache()
    cache.store(invalid, "module::root", payload())
    assert cache.lookup(invalid, "module::root") is None
    assert cache.stats["stores"] == 0
    assert cache.stats["rejected"] == 2
