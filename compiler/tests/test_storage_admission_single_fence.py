"""One complete fence covers iterable callbacks and immutable membership reads."""
from copy import copy

import pytest

from compiler.ir import CompilationError
from compiler.scopes import storage_admission as module
from compiler.scopes.storage_admission import StorageAdmission, active_admission
from compiler.tests.test_storage_emission_admission import final_plan as final_plan
from compiler.tests.test_storage_emission_admission import query


@pytest.mark.parametrize("count", [1, 2048])
def test_one_full_check_after_materializing_ordinary_binding_values(final_plan, monkeypatch, count):
    _path, _owner, plan = final_plan
    member, bindings = query(plan)
    proof = plan.storage_epoch()
    expected = proof.admits_many(plan, member, bindings)
    values = {index: bindings[index % len(bindings)] for index in range(count)}
    checks = []
    original = StorageAdmission._check

    def checked(self):
        checks.append(self)
        return original(self)

    with StorageAdmission(plan) as admission:
        monkeypatch.setattr(StorageAdmission, "_check", checked)
        assert admission.admits_many(proof, plan, member, values.values()) == tuple(
            expected[index % len(bindings)] for index in range(count))
        assert checks == [admission]


@pytest.mark.parametrize("fault", ["plan", "proof", "released"])
def test_wrong_or_released_authority_rejects_before_iterating(final_plan, fault):
    _path, _owner, plan = final_plan
    member, bindings = query(plan)
    proof = plan.storage_epoch()
    reached = []

    class Untrusted:
        def __iter__(self):
            reached.append("iterated")
            return iter(bindings)

    admission = StorageAdmission(plan)
    if fault == "released":
        with admission:
            pass
        with pytest.raises(CompilationError):
            admission.admits_many(proof, plan, member, Untrusted())
    else:
        with admission, pytest.raises(CompilationError):
            admission.admits_many(copy(proof) if fault == "proof" else proof,
                                  copy(plan) if fault == "plan" else plan, member, Untrusted())
    assert not reached
    assert active_admission() is None


@pytest.mark.parametrize("fault", ["fact", "table"])
@pytest.mark.parametrize("callback", ["iteration", "length_hint"])
def test_iterable_mutation_is_fenced_before_membership_returns(final_plan, fault, callback):
    _path, owner, plan = final_plan
    member, bindings = query(plan)
    reached = []

    def inspect():
        with StorageAdmission(plan) as admission:
            def changed():
                reached.append(callback)
                if fault == "fact":
                    owner.builder.facts["captures"]["argument::a"]["initialized"] = "none"
                else:
                    dict.__setitem__(admission._admitted, (id(member), id(bindings[0])),
                                     (member, bindings[0], "argument::forged"))

            class Untrusted:
                def __iter__(self):
                    if callback == "iteration":
                        changed()
                    return iter(bindings)

                def __len__(self):
                    if callback == "length_hint":
                        changed()
                    return len(bindings)

            admission.admits_many(plan.storage_epoch(), plan, member, Untrusted())
            pytest.fail("an iterable substitution must not return canonical membership")
    with pytest.raises(CompilationError):
        inspect()
    assert reached == [callback]
    assert active_admission() is None


@pytest.mark.parametrize("fault", ["entry", "root_subclass", "table", "anchors", "active", "equal_key"])
def test_last_source_callback_cannot_forge_postfence_membership(final_plan, monkeypatch, fault):
    _path, owner, plan = final_plan
    member, bindings = query(plan)
    inputs = owner.builder.analysis.inputs
    original = type(inputs).verify_live
    calls, callbacks = [], []

    def inspect():
        with StorageAdmission(plan) as admission:
            key = id(member), id(bindings[0])

            class EqualKey:
                def __hash__(self):
                    return hash(key)

                def __eq__(self, other):
                    callbacks.append("equal")
                    return other == key

            class Root(str):
                def __str__(self):
                    callbacks.append("str")
                    return super().__str__()

            def changed(self):
                result = original(self)
                if self is inputs:
                    calls.append(self)
                    if len(calls) == 1:
                        if fault == "entry":
                            dict.__setitem__(admission._admitted, key, (member, bindings[0], "argument::forged"))
                        elif fault == "root_subclass":
                            dict.__setitem__(admission._admitted, key, (member, bindings[0], Root("argument::forged")))
                        elif fault == "table":
                            admission._admitted = dict(admission._admitted)
                        elif fault == "anchors":
                            admission._record = "{}"
                        elif fault == "active":
                            module._ACTIVE.set(None)
                        else:
                            entry = dict.pop(admission._admitted, key)
                            dict.__setitem__(admission._admitted, EqualKey(), entry)
                return result

            monkeypatch.setattr(type(inputs), "verify_live", changed)
            admission.admits_many(plan.storage_epoch(), plan, member, bindings)
            pytest.fail("the final source callback must not forge detached membership")

    with pytest.raises(CompilationError):
        inspect()
    assert len(calls) == 1
    assert callbacks == []
    assert active_admission() is None
