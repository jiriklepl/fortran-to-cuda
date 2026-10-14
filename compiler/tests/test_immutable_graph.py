"""Explicit graph ownership, retained aliases, cycles and private copies."""
from copy import copy, deepcopy
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from compiler.frontend.immutable_graph import FrozenDict, FrozenList, GraphFreezer, sealed
from compiler.ir import CompilationError


@dataclass
class Owned:
    values: object = field(default_factory=dict)
    parent: object = None


@dataclass(slots=True)
class Slotted:
    value: int


def test_owned_graph_seals_mutations_but_borrows_explicit_external_services():
    external = SimpleNamespace(counter=0)
    owner = Owned({"values": [1, 2]}, external)
    old_state, old_values, old_list = vars(owner), owner.values, owner.values["values"]
    freezer = GraphFreezer(owned_types=(Owned,), borrowed=(external,))
    assert freezer.freeze(owner) is owner
    old_state["parent"] = None
    old_values.clear()
    old_list.clear()
    assert owner.values["values"] == [1, 2]
    assert owner.parent is external
    external.counter += 1
    assert owner.parent.counter == 1
    with pytest.raises(CompilationError, match="immutable"):
        owner.values["values"].append(3)
    with pytest.raises(CompilationError, match="immutable"):
        owner.parent = None
    with pytest.raises(TypeError):
        vars(owner)["parent"] = None
    owner.__getstate__()["parent"] = None
    assert owner.parent is external


def test_object_and_container_backedges_retain_identity():
    owner = Owned()
    values = [owner]
    owner.values["back"] = values
    frozen = GraphFreezer(owned_types=(Owned,)).freeze(values)
    assert frozen[0] is owner
    assert owner.values["back"] is frozen
    assert isinstance(frozen, FrozenList)
    assert isinstance(owner.values, FrozenDict)


@pytest.mark.parametrize("copier", [copy, deepcopy])
def test_private_copies_thaw_owned_state_and_slots(copier):
    owner = Owned({"values": [1, 2]})
    slotted = Slotted(3)
    freezer = GraphFreezer(owned_types=(Owned, Slotted))
    freezer.freeze(owner)
    freezer.freeze(slotted)
    cloned = copier(owner)
    assert not sealed(cloned)
    cloned.values["values"].append(4)
    assert owner.values["values"] == [1, 2]
    cloned_slot = copier(slotted)
    cloned_slot.value = 5
    assert slotted.value == 3
    with pytest.raises(CompilationError, match="immutable"):
        slotted.value = 6


def test_unknown_mutable_edges_fail_closed():
    with pytest.raises(CompilationError, match="ownership is unspecified"):
        GraphFreezer(owned_types=(Owned,)).freeze(Owned(parent=SimpleNamespace(state=1)))
