"""Complete CALL inventory projects physical roles without a future-work scan."""
from copy import copy

import pytest

from compiler.frontend.native_call_effects import prove_native_call_effects
from compiler.frontend.resource_reachability import prove_resource_reachability
from compiler.ir import CompilationError
from compiler.tests.test_native_graph_associations import SOURCE, setup


def issue(tmp_path, source=SOURCE):
    _, analysis, selection, _, _ = setup(tmp_path, source)
    return analysis, selection, prove_native_call_effects(analysis, "work::owner", selection)


def test_complete_call_projects_all_storage_without_future_demands(tmp_path):
    analysis, selection, proof = issue(tmp_path)
    assert proof.validate(analysis, "work::owner", selection) is proof
    assert prove_native_call_effects(analysis, "work::owner", selection) is proof
    roles = {binding.root: (role, actions) for binding, role, actions in proof.resources}
    assert roles == {"work::points": ("reservation", frozenset({"association", "read", "write", "undefine", "descriptor"})),
                     "work::old": ("managed", frozenset({"association", "read"}))}
    assert not proof.reachability.demanded
    assert not proof.reachability.obligations
    assert {item.storage.resource for item in proof.reachability.accesses} >= {"work::points%flux", "work::points%aux"}
    assert all(not value for key, value in proof.public().items() if key.endswith("authorized"))


def test_empty_demands_keep_complete_accesses_and_completion(tmp_path):
    _, analysis, selection, demanded, required = setup(tmp_path)
    inventory = prove_resource_reachability(analysis, "work::owner", selection, ())
    assert not inventory.barriers
    assert [item.public() for item in inventory.accesses] == [item.public() for item in required.accesses]
    assert inventory._completion == required._completion
    assert not inventory.obligations
    assert required.obligations
    assert inventory.validate(analysis, "work::owner", selection, ()) is inventory
    with pytest.raises(CompilationError):
        inventory.validate(analysis, "work::owner", selection, demanded)


def test_numeric_partial_write_retains_read_and_write_without_definition_credit(tmp_path):
    text = SOURCE
    text = text.replace("real(8),intent(in)::input(:)", "real(8),intent(inout)::input(:)")
    text = text.replace("p=>items(i)", "p=>items(i)\ninput(i)=2*input(i)")
    _, _, proof = issue(tmp_path, text)
    binding, role, actions = next(row for row in proof.resources if row[0].root == "work::old")
    assert role == "managed"
    assert {"read", "write"} <= actions
    assert not proof.public()["definitions_authorized"]


def test_scalar_state_requires_a_physical_reservation(tmp_path):
    source = SOURCE.replace("real(8)::old(4),future(4)", "real(8)::old(4),future(4)\ninteger::visits=0")
    source = source.replace("enddo\n!$omp end parallel do", "enddo\n!$omp end parallel do\nvisits=visits+1")
    _, _, proof = issue(tmp_path, source)
    row, = [row for row in proof.resources if row[0].root == "work::visits"]
    assert row[1:] == ("reservation", frozenset({"read", "write"}))


@pytest.mark.parametrize(("before", "after"), [
    ("call helper(p%flux,input(i),aux=p%aux)", "call unknown(p%flux)"),
    ("private(i,p)", "private(i)"),
    ("real(8)::old(4),future(4)", "real(8),pointer::old(:)\nreal(8)::future(4)"),
    ("real(8)::flux,aux", "real(8)::flux\nreal(8),allocatable::aux"),
    ("real(8),intent(in)::input(:)", "real(8),intent(out)::input(:)"),
])
def test_unknown_effects_storage_or_managed_undefine_are_not_admitted(tmp_path, before, after):
    with pytest.raises(CompilationError):
        issue(tmp_path, SOURCE.replace(before, after))


def test_foreign_or_changed_call_evidence_cannot_be_reused(tmp_path):
    analysis, selection, proof = issue(tmp_path)
    with pytest.raises(CompilationError):
        copy(proof).validate(analysis, "work::owner", selection)
    with pytest.raises(CompilationError):
        proof.validate(analysis, "work::update", selection)
    with pytest.raises(CompilationError):
        proof.validate(analysis, "work::owner", (copy(selection[0]),))
    object.__setattr__(proof, "resources", ())
    with pytest.raises(CompilationError, match="projection changed"):
        proof.validate(analysis, "work::owner", selection)


@pytest.mark.parametrize("mutation", ["access", "path", "binding", "storage_graph", "registry"])
def test_transitive_original_effect_evidence_is_retained(tmp_path, mutation):
    analysis, selection, proof = issue(tmp_path)
    access = next(item for item in proof.reachability.accesses if item.storage.binding.root == "work::old")
    if mutation == "access":
        object.__setattr__(access, "kind", "write")
    elif mutation == "path":
        object.__setattr__(access.storage, "binding", analysis.modules["work"].bindings["future"])
    elif mutation == "binding":
        object.__setattr__(access.storage.binding, "attributes", frozenset({"pointer"}))
    elif mutation == "storage_graph":
        object.__setattr__(proof.storage, "_native_graph", copy(proof.reachability.native_graph))
    else:
        analysis._resource_reachability_proofs.clear()
    with pytest.raises(CompilationError):
        proof.validate(analysis, "work::owner", selection)
