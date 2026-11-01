"""Registry fences retain exact objects without per-entry Python comparisons."""
from copy import copy
from types import SimpleNamespace

import pytest

from compiler.frontend.descriptor_intrinsics import (
    _MAPPING_COLUMNS,
    intrinsic_source_attachments,
    same_intrinsic_source_attachments,
)
from compiler.frontend.immutable_graph import FrozenDict


class Equal:
    def __eq__(self, other):
        raise AssertionError("source ownership must not call value equality")


def receipt(mapping):
    analysis = SimpleNamespace(modules=mapping)
    return analysis, intrinsic_source_attachments(analysis)


@pytest.mark.parametrize("mapping_type", [dict, FrozenDict])
def test_compact_receipt_keeps_objects_and_exact_identity(mapping_type):
    original = Equal()
    mapping = mapping_type({"entry": original, "none": None, 7: object()})
    analysis, state = receipt(mapping)
    saved = next(contents for name, _root, contents in state[1] if name == "modules")
    assert type(saved) is tuple
    assert saved[0] is _MAPPING_COLUMNS
    assert saved[2][0] is original
    assert same_intrinsic_source_attachments(analysis, state)
    dict.__setitem__(mapping, "entry", Equal())
    assert not same_intrinsic_source_attachments(analysis, state)


@pytest.mark.parametrize("mutation", ["missing_none", "added", "removed", "root", "session"])
def test_registry_changes_do_not_reuse_a_previous_validation(mutation):
    mapping = {"entry": object(), "none": None}
    analysis, state = receipt(mapping)
    assert same_intrinsic_source_attachments(analysis, state)
    if mutation == "missing_none":
        del mapping["none"]
        mapping["other"] = None
    elif mutation == "added":
        mapping["added"] = object()
    elif mutation == "removed":
        del mapping["entry"]
    elif mutation == "root":
        analysis.modules = copy(mapping)
    else:
        analysis._source_session = object()
    assert not same_intrinsic_source_attachments(analysis, state)


def test_mapping_order_is_not_a_new_authority_requirement():
    mapping = {"first": object(), "second": object()}
    analysis, state = receipt(mapping)
    mapping["first"] = mapping.pop("first")
    assert same_intrinsic_source_attachments(analysis, state)


def test_inert_identity_columns_are_immutable():
    analysis, state = receipt({"entry": object()})
    saved = next(contents for name, _root, contents in state[1] if name == "modules")
    with pytest.raises(AttributeError):
        object.__setattr__(saved, "identities", ())
    assert same_intrinsic_source_attachments(analysis, state)


def test_custom_dictionary_callbacks_keep_original_path():
    class Custom(dict):
        def items(self):
            self.items_calls += 1
            return super().items()
    mapping = Custom(entry=object())
    mapping.items_calls = 0
    analysis, state = receipt(mapping)
    assert mapping.items_calls == 1
    saved = next(contents for name, _root, contents in state[1] if name == "modules")
    assert type(saved) is tuple
    assert saved[0] is not _MAPPING_COLUMNS
    assert same_intrinsic_source_attachments(analysis, state)
    mapping["entry"] = object()
    assert not same_intrinsic_source_attachments(analysis, state)


def test_nonprimitive_keys_keep_original_mapping_path():
    key = object()
    analysis, state = receipt({key: object()})
    saved = next(contents for name, _root, contents in state[1] if name == "modules")
    assert type(saved) is tuple
    assert saved[0] is not _MAPPING_COLUMNS
    assert same_intrinsic_source_attachments(analysis, state)


def test_changed_dictionary_class_cannot_install_a_lookup_callback():
    class Foreign(dict):
        __slots__ = ()

        def __getitem__(self, key):
            raise AssertionError("changed source class must be rejected before lookup")

    mapping = FrozenDict({"entry": object()})
    analysis, state = receipt(mapping)
    mapping.__class__ = Foreign
    assert not same_intrinsic_source_attachments(analysis, state)


def test_foreign_equivalent_key_is_rejected_before_equality_callback():
    class Foreign:
        def __hash__(self):
            return hash("entry")

        def __eq__(self, other):
            raise AssertionError("foreign key equality must not run at the source fence")

    mapping = {"entry": object()}
    analysis, state = receipt(mapping)
    original = mapping.pop("entry")
    mapping[Foreign()] = original
    assert not same_intrinsic_source_attachments(analysis, state)


def test_compact_receipt_uses_builtin_dictionary_length(monkeypatch):
    mapping = FrozenDict({"entry": object()})
    analysis, state = receipt(mapping)

    def foreign_length(self):
        raise AssertionError("source fence must not invoke replaced dictionary length")

    monkeypatch.setattr(FrozenDict, "__len__", foreign_length)
    assert same_intrinsic_source_attachments(analysis, state)
