"""FORALL storage projection scans control and payload through one engine."""
import pytest

from compiler.frontend.storage_mutation_effects import prove_storage_mutation_effects
from compiler.tests.test_storage_mutation_effects import analysis


def projected(tmp_path, body, *, helpers="", declarations="integer::i,j,n", argument=None):
    options = {} if argument is None else {"argument": argument}
    _path, source = analysis(tmp_path, body, declarations=declarations, helpers=helpers,
        module_declarations="real(8),allocatable::hidden(:)", **options)
    before = source.summarize("storage::advance")
    proof = prove_storage_mutation_effects(source, "storage::advance")
    assert source.summarize("storage::advance") == before
    return source, proof


@pytest.mark.parametrize("body", ["forall(i=1:n) a(i)=2.0_8*a(i)",
    "forall(i=1:n:2,i/=2) a(i)=a(i-1)",
    "forall(i=1:n)\na(i)=a(i)+1.0_8\na(i)=a(i)*2.0_8\nend forall"])
def test_complete_original_forall_payload_writes_are_lifetime_inert_without_native_credit(tmp_path, body):
    _source, proof = projected(tmp_path, body)
    assert proof.public()["complete"], proof.public()["frontiers"]
    assert proof.descriptor_unchanged("argument::a")
    for flag in ("payload_access_authorized", "entry_definition_authority", "capture_authorized",
                 "execution_authorized", "motion_authorized", "gpu_legality_established", "alias_authority"):
        assert proof.public()[flag] is False


def test_nested_original_forall_retains_each_header_mask_and_assignment(tmp_path):
    _source, proof = projected(tmp_path, "forall(i=1:n)\nforall(j=1:n,j/=i) a(i,j)=a(i,j)+1.0_8\nend forall",
        argument="real(8),allocatable,intent(inout)::a(:,:)")
    assert proof.public()["complete"], proof.public()["frontiers"]
    assert proof.descriptor_unchanged("argument::a")


@pytest.mark.parametrize("body", ["forall(i=lower():n) a(i)=1.0_8", "forall(i=1:lower()) a(i)=1.0_8",
    "forall(i=1:n:lower()) a(i)=1.0_8", "forall(i=1:n,lower()>0) a(i)=1.0_8",
    "forall(i=1:n) a(lower())=1.0_8", "forall(i=1:n) a(i)=lower()"])
def test_every_bound_stride_mask_and_assignment_expression_uses_full_original_function_closure(tmp_path, body):
    _source, proof = projected(tmp_path, body, helpers="""integer function lower()
deallocate(hidden)
lower=1
end function""")
    assert proof.public()["complete"], proof.public()["frontiers"]
    assert any(event["resource"] == "storage::hidden" for event in proof.public()["mutations"])
    assert {member["procedure"] for member in proof.public()["closure"]} == {"storage::advance", "storage::lower"}
    assert not proof.descriptor_unchanged("argument::a")


def test_multidimensional_saved_coefficient_payloads_match_original_pressure_shape(tmp_path):
    _source, proof = projected(tmp_path, """forall(i=1:n)
ae(i)=1.0_8/(a(i,1)*a(i+1,1))
aw(i)=1.0_8/(a(i-1,1)*a(i,1))
end forall
forall(i=1:n,j=1:n,i/=j) a(i,j)=a(i,j)+ae(i)*aw(j)""",
        declarations="integer::i,j,n\nreal(8),allocatable,save::ae(:),aw(:)",
        argument="real(8),allocatable,intent(inout)::a(:,:)")
    assert proof.public()["complete"], proof.public()["frontiers"]
    assert proof.preserves_descriptors()
    assert not proof.public()["mutations"]
    assert not proof.public()["entry_definition_authority"]


def test_selected_forall_does_not_hide_prior_allocation_or_create_entry_definition_credit(tmp_path):
    source, whole = projected(tmp_path, "allocate(a(n))\nforall(i=1:n)a(i)=1.0_8")
    routine = source.routines["storage::advance"]
    selected = prove_storage_mutation_effects(source, routine.qualified, selection=(routine.execution.content[-1],))
    assert whole.public()["mutations"]
    assert not whole.preserves_descriptors()
    assert selected.public()["version"] == 3
    assert selected.preserves_descriptors()
    assert selected.public()["entry_exit_lifecycle_authorized"] is False
    assert selected.public()["entry_definition_authority"] is False


@pytest.mark.parametrize("body", ["forall(i=1:n)\na=1.0_8\nend forall",
    "forall(i=1:n)\na(i)=1.0_8\nend forall\ndeallocate(a)"])
def test_whole_assignment_and_original_tail_mutation_remain_visible(tmp_path, body):
    _source, proof = projected(tmp_path, body)
    assert proof.public()["mutations"]
    assert not proof.preserves_descriptors()


def test_forall_pointer_assignment_is_still_an_escape_frontier(tmp_path):
    _source, proof = projected(tmp_path, "forall(i=1:n) p=>hidden", declarations="integer::i,j,n\nreal(8),pointer::p(:)")
    assert not proof.preserves_descriptors()
    assert any(event["kind"] == "pointer association" for event in proof.public()["escapes"])


def test_unknown_header_does_not_omit_known_body_reallocation(tmp_path):
    _source, proof = projected(tmp_path, "forall(i=1:unknown())\na=1.0_8\nend forall")
    assert not proof.public()["complete"]
    assert any("unknown function" in event["kind"] for event in proof.public()["frontiers"])
    assert any("whole allocatable" in event["kind"] for event in proof.public()["mutations"])


@pytest.mark.parametrize("declaration", ["real(8)::i", "integer,pointer::i", "integer,optional::i"])
def test_original_index_declaration_must_be_independent_scalar_integer(tmp_path, declaration):
    _source, proof = projected(tmp_path, "forall(i=1:n)a(i)=1.0_8", declarations=declaration+"\ninteger::j,n")
    assert not proof.public()["complete"]
    assert any("FORALL index" in event["kind"] for event in proof.public()["frontiers"])


def test_zero_trip_header_is_not_evaluated_and_does_not_mask_original_source_effects(tmp_path):
    _source, proof = projected(tmp_path, "forall(i=1:0,unknown()>0)a(i)=1.0_8")
    assert not proof.public()["complete"]
    assert not proof.public()["execution_authorized"]
