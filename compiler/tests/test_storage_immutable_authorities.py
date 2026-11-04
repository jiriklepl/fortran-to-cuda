"""Immutable proof containers still fence every mutable child and owner edge."""
import gc
import weakref
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from compiler.frontend.immutable_graph import FrozenDict
from compiler.ir import CompilationError
from compiler.scopes.storage_admission import StorageAdmission


@dataclass
class Envelope:
    payload: object


@dataclass(eq=False)
class Leaf:
    payload: object


def inspection(payload):
    """Exercise only the private attachment fence; issue no storage permission."""
    envelope = Envelope(payload)
    owner = SimpleNamespace(builder=SimpleNamespace(), emission=SimpleNamespace(selection_prefix=envelope))
    plan = SimpleNamespace(_contexts=(), _procedures=(), _numerical_authorities=(),
                           _proof_handoffs=(), _selection_prefixes=(), members=())
    admission = StorageAdmission.__new__(StorageAdmission)
    admission._owner, admission._plan = owner, plan
    admission._dependencies = ()
    admission._retained = admission._authorities()
    admission._registrations = admission._registry_state()
    admission._check_retained()
    return admission, envelope


def test_mutable_child_below_nested_immutable_containers_stays_fenced():
    child = ["original"]
    admission, _envelope = inspection(((child,), frozenset({Leaf(child)})))
    child.append("changed")
    with pytest.raises(CompilationError, match="retained exact proof attachment"):
        admission._check_retained()


def test_frozen_mapping_mutation_below_frozenset_stays_fenced():
    child = FrozenDict(value=object())
    admission, _envelope = inspection(frozenset({Leaf((child,))}))
    dict.__setitem__(child, "value", object())
    with pytest.raises(CompilationError, match="retained exact proof attachment"):
        admission._check_retained()


@pytest.mark.parametrize("container", [tuple, frozenset])
def test_equal_immutable_replacement_still_changes_owning_field(container):
    original = container([object()])
    admission, envelope = inspection(original)
    envelope.payload = container(list(original))
    assert envelope.payload is not original
    with pytest.raises(CompilationError, match="retained exact proof attachment"):
        admission._check_retained()


def test_immutable_parent_keeps_original_child_alive_until_fence_release():
    class Object:
        pass

    child = Object()
    observed = weakref.ref(child)
    original = ((child,),)
    admission, envelope = inspection(original)
    envelope.payload = None
    del child, original
    gc.collect()
    assert observed() is not None
    del admission
    gc.collect()
    assert observed() is None


@pytest.mark.parametrize("container", [tuple, frozenset])
def test_subclass_cannot_spoof_builtin_type_and_hide_changed_iteration(container):
    callbacks, contents = [], [object()]

    class SpoofBuiltin(type):
        def __hash__(cls):
            callbacks.append("hash")
            return hash(container)

        def __eq__(cls, other):
            callbacks.append("equal")
            return other is container

    class Dynamic(container, metaclass=SpoofBuiltin):
        def __iter__(self):
            return iter(contents)

    admission, _envelope = inspection(Dynamic(contents))
    assert callbacks == []
    contents[0] = object()
    with pytest.raises(CompilationError, match="retained exact proof attachment"):
        admission._check_retained()
    assert callbacks == []
