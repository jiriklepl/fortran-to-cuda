"""Explicit graph ownership, retained aliases, cycles and private copies."""
import gc
import weakref
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


def test_second_owner_adopts_transitive_completion_and_precondition_aliases():
    completion = {"available": True, "steps": [1, 2]}
    extraction = Owned({"completion": completion, "preconditions": (Slotted(3),)})
    first = GraphFreezer(owned_types=(Owned, Slotted))
    first.freeze(extraction)
    completion, conditions = extraction.values["completion"], extraction.values["preconditions"]
    second = GraphFreezer(owned_types=(Owned, Slotted))
    second.adopt_sealed(extraction)
    frozen = second.freeze({"completion": completion, "conditions": conditions, "extraction": extraction})
    assert frozen["extraction"] is extraction
    assert frozen["completion"] is completion
    assert frozen["conditions"] is conditions
    assert frozen["completion"]["steps"] is completion["steps"]
    with pytest.raises(CompilationError, match="immutable"):
        frozen["completion"]["steps"].append(4)


def test_automatic_adoption_retains_container_and_object_backedges():
    owner = Owned()
    owner.values["self"] = owner
    owner.values["cycle"] = [owner.values]
    GraphFreezer(owned_types=(Owned,)).freeze(owner)
    second = GraphFreezer(owned_types=(Owned,))
    assert second.freeze(owner) is owner
    assert second.freeze(owner.values) is owner.values
    assert second.freeze(owner.values["cycle"]) is owner.values["cycle"]
    assert owner.values["cycle"][0] is owner.values


def test_adoption_does_not_expand_the_new_owners_borrowing_policy():
    external = SimpleNamespace(mutable=[])
    owner = Owned(parent=external)
    GraphFreezer(owned_types=(Owned,), borrowed=(external,)).freeze(owner)
    rejected = GraphFreezer(owned_types=(Owned,))
    with pytest.raises(CompilationError, match="unowned mutable descendant"):
        rejected.adopt_sealed(owner)
    assert rejected.memo == {}
    assert rejected.objects == {}
    second = GraphFreezer(owned_types=(Owned,), borrowed=(external,))
    second.adopt_sealed(owner)
    assert second.freeze(owner).parent is external
    assert id(external) not in second.objects
    external.mutable.append(1)
    assert owner.parent.mutable == [1]


@pytest.mark.parametrize("replacement", [{"unowned": []}, FrozenDict({"mutable": []})])
def test_adoption_rejects_mutable_edges_injected_into_an_old_sealed_owner(replacement):
    owner = Owned()
    GraphFreezer(owned_types=(Owned,)).freeze(owner)
    object.__setattr__(owner, "values", replacement)
    second = GraphFreezer(owned_types=(Owned,))
    with pytest.raises(CompilationError, match="unowned mutable descendant"):
        second.adopt_sealed(owner)
    assert second.memo == {}


def test_adoption_cannot_rebind_an_alias_that_was_already_detached():
    owner = Owned({"completed": True})
    GraphFreezer(owned_types=(Owned,)).freeze(owner)
    second = GraphFreezer(owned_types=(Owned,))
    detached = second.freeze(owner.values)
    assert detached is not owner.values
    with pytest.raises(CompilationError, match="already detached alias"):
        second.adopt_sealed(owner)
    assert second.memo[id(owner.values)] is detached


def test_immutable_tuple_roots_preserve_a_previous_proofs_exact_field_attachments():
    child = Slotted(2)
    GraphFreezer(owned_types=(Slotted,)).freeze(child)
    selection, changes = (child,), ("original::scalar",)
    second = GraphFreezer(owned_types=(Slotted,))
    second.adopt_sealed(selection)
    second.adopt_sealed(changes)
    assert second.freeze(selection) is selection
    assert second.freeze(changes) is changes
    with pytest.raises(CompilationError, match="unowned mutable descendant"):
        GraphFreezer().adopt_sealed(([],))


def test_detached_container_memo_retains_originals_only_for_its_own_lifetime():
    class Original(dict):
        pass

    original = Original(old=1)
    retained = weakref.ref(original)
    freezer = GraphFreezer()
    frozen = freezer.freeze(original)
    assert frozen == {"old": 1}
    del original
    gc.collect()
    assert retained() is not None
    for index in range(100):
        current = {"different": index}
        assert freezer.freeze(current) == current
    del freezer
    gc.collect()
    assert retained() is None


def test_a_detached_memo_entry_cannot_be_attached_to_another_original():
    original, foreign = {"original": 1}, {"foreign": 2}
    freezer = GraphFreezer()
    frozen = freezer.freeze(original)
    freezer.memo[id(foreign)] = frozen
    with pytest.raises(CompilationError, match="exact original object"):
        freezer.freeze(foreign)


def test_tuple_subclass_does_not_retain_unauthenticated_mutable_attributes():
    class MutableTuple(tuple):
        pass

    original = MutableTuple((1, "constant"))
    original.payload = []
    frozen = GraphFreezer().freeze(original)
    assert type(frozen) is tuple
    assert frozen == original
    assert frozen is not original
    original.payload.append("changed")
    assert not hasattr(frozen, "payload")
