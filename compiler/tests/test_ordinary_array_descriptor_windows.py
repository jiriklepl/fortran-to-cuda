"""Borrowed ordinary descriptors establish no caller storage or payload facts."""
from copy import copy

import pytest

from compiler.frontend.descriptor_windows import prove_descriptor_window
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError

ENTRY = "borrowed::run"
ROOT = "argument::a"


def prepare(tmp_path, *, declaration="real(8),contiguous,intent(inout)::a(:)",
            selected="call work(a)", child="a(:)=a(:)+1.0_8"):
    path = tmp_path / "borrowed.f90"
    path.write_text(f"""module borrowed
contains
subroutine work(a)
real(8),intent(inout)::a(:)
{child}
end subroutine
subroutine run(a)
{declaration}
call before_native()
{selected}
call after_native()
end subroutine
end module
""")
    analysis = SourceEffects([path])
    routine = analysis.routines[ENTRY]
    graph = analysis.structure(ENTRY)
    node = routine.execution.content[1]
    ids = (graph.node_id(node),)
    return path, analysis, ids


@pytest.mark.parametrize("declaration", [
    "real(8),contiguous,intent(inout)::a(:)", "real(8),intent(in)::a(-2:)",
    "real(8),intent(out)::a(-2:4)", "integer(4),intent(inout)::a(:,:)",
])
def test_exact_original_dummy_descriptor_does_not_claim_an_allocation_or_initial_values(tmp_path, declaration):
    path, analysis, ids = prepare(tmp_path, declaration=declaration, selected="continue")
    original = path.read_bytes()
    proof = prove_descriptor_window(analysis, ENTRY, ids, (ROOT,))
    assert proof.validate(analysis, ENTRY, ids) is proof
    record = proof.public()
    assert record["schema_version"] == 10
    assert record["prefix_effects_proved"] is False
    assert record["entry_definition_authority"] is False
    assert record["execution_authorized"] is False
    state = proof.resource_state(ROOT)
    assert state["resource_kind"] == "original_ordinary_array_formal"
    assert state["borrowed_descriptor"] is True
    for flag in ("caller_storage_lifetime_authority", "physical_allocation_origin_authority",
                 "payload_access_authorized", "alias_authority", "entry_definition_authority"):
        assert state[flag] is False
    assert state["requires_prior_noescape_and_current_allocation_proof"] is True
    assert state["requires_original_allocation_presence_and_alias_guards"] is True
    assert path.read_bytes() == original


def test_full_original_child_effects_still_run_through_the_existing_descriptor_scanner(tmp_path):
    _path, analysis, ids = prepare(tmp_path)
    proof = prove_descriptor_window(analysis, ENTRY, ids, (ROOT,))
    assert "borrowed::work" in proof.public()["source_call_closure"]
    assert proof.resource_state(ROOT)["descriptor_changes_within_selection"] is False


@pytest.mark.parametrize("attributes", ["pointer", "target", "optional", "volatile", "asynchronous", "value", "save"])
def test_unproved_array_associations_cannot_be_borrowed(tmp_path, attributes):
    _path, analysis, ids = prepare(tmp_path, declaration=f"real(8),{attributes},intent(inout)::a(:)", selected="continue")
    with pytest.raises(CompilationError):
        prove_descriptor_window(analysis, ENTRY, ids, (ROOT,))


@pytest.mark.parametrize("selected", ["call unknown(a)", "call unknown()", "deallocate(a)", "allocate(a(4))"])
def test_ordinary_descriptor_support_does_not_bypass_unknown_or_mutating_selected_work(tmp_path, selected):
    _path, analysis, ids = prepare(tmp_path, selected=selected)
    with pytest.raises(CompilationError):
        prove_descriptor_window(analysis, ENTRY, ids, (ROOT,))


@pytest.mark.parametrize("fault", ["copy", "registry", "source", "binding"])
def test_new_descriptor_contract_keeps_original_registered_source_authority(tmp_path, fault):
    path, analysis, ids = prepare(tmp_path)
    proof = prove_descriptor_window(analysis, ENTRY, ids, (ROOT,))
    if fault == "copy":
        proof = copy(proof)
    elif fault == "registry":
        analysis._descriptor_window_proofs.pop(proof.identity)
    elif fault == "source":
        path.write_text(path.read_text()+"! changed\n")
    else:
        analysis.routines[ENTRY].scope.bindings["a"].rank = 2
    with pytest.raises(CompilationError):
        proof.validate(analysis, ENTRY, ids)
