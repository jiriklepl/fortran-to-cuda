"""Optional native payloads do not imply descriptor mutation or presence."""
import pytest

from compiler.frontend.storage_mutation_effects import prove_storage_mutation_effects
from compiler.tests.test_storage_mutation_effects import analysis


def projected(tmp_path, body="call change(a)", *, helper_body="if(present(x)) x(:)=x(:)+1.0_8",
              declaration="real(8),optional,intent(inout)::x(:)", extra="", argument=None):
    options = {} if argument is None else {"argument": argument}
    path, source = analysis(tmp_path, body, module_declarations="real(8),allocatable::hidden(:)",
        helpers=f"""subroutine change(x)
{declaration}
{helper_body}
end subroutine
{extra}""", **options)
    before = source.summarize("storage::advance")
    proof = prove_storage_mutation_effects(source, "storage::advance")
    # The lifetime projection cannot upgrade ordinary native/access admission.
    assert source.summarize("storage::advance") == before
    assert path.read_text() == source.source_text(path)
    return source, proof


@pytest.mark.parametrize("helper_body", ["if(present(x)) x(:)=x(:)+1.0_8",
    "if(present(x)) x(1)=sum(x)", "if(present(x)) write(*,*)size(x)"])
def test_exact_optional_numeric_payload_touches_are_descriptor_inert(tmp_path, helper_body):
    _source, proof = projected(tmp_path, helper_body=helper_body)
    assert proof.public()["complete"], proof.public()["frontiers"]
    assert proof.descriptor_unchanged("argument::a")
    for flag in ("payload_access_authorized", "entry_definition_authority", "capture_authorized",
                 "execution_authorized", "motion_authorized", "gpu_legality_established", "alias_authority"):
        assert proof.public()[flag] is False


@pytest.mark.parametrize("allocatable", [False, True])
def test_omitted_optional_never_acquires_a_caller_descriptor_or_definition(tmp_path, allocatable):
    attrs = ",allocatable" if allocatable else ""
    _source, proof = projected(tmp_path, "call change()", declaration=f"real(8),optional{attrs},intent(in)::x(:)",
        helper_body="if(present(x)) write(*,*)size(x)")
    record = proof.public()
    assert record["complete"], record["frontiers"]
    assert record["definition_kills"] == []
    root, = [member for member in record["closure"] if member["procedure"] == "storage::advance"]
    association, = root["calls"][0]["association"]["associations"]
    assert association["actual"] is None
    assert association["actual_type"] is None
    assert root["calls"][0]["canonical_resources"] == {"storage::change::argument::x": None}
    assert not record["mutations"]
    assert record["entry_definition_authority"] is False


def test_original_optional_forwarding_preserves_absence_without_payload_read_authority(tmp_path):
    source, proof = projected(tmp_path, argument="real(8),allocatable,optional,intent(inout)::a(:)",
        helper_body="call leaf(x)", extra="""subroutine leaf(x)
real(8),optional,intent(inout)::x(:)
if(present(x)) x(:)=2.0_8*x(:)
end subroutine""")
    assert proof.descriptor_unchanged("argument::a")
    for member in proof.public()["closure"]:
        for call in member["calls"]:
            association, = call["association"]["associations"]
            assert "optional" in association["formal_descriptor"]["attributes"]
            assert association["actual"] in {"a", "x"}
    for call in source._original_call_associations.values():
        association, = call.associations
        assert "optional" in association.binding.attributes


@pytest.mark.parametrize("body", ["x=1.0_8", "deallocate(x)", "allocate(x(3))", "call move_alloc(x,hidden)"])
def test_optional_allocation_replacement_remains_a_mutation(tmp_path, body):
    _source, proof = projected(tmp_path, declaration="real(8),allocatable,optional,intent(inout)::x(:)",
        helper_body="if(present(x)) then\n"+body+"\nendif")
    assert proof.public()["complete"], proof.public()["frontiers"]
    assert any(event["resource"] == "argument::a" for event in proof.public()["mutations"])
    assert not proof.preserves_descriptors()


def test_optional_allocatable_out_entry_kill_is_not_a_presence_or_definition_grant(tmp_path):
    _source, proof = projected(tmp_path, declaration="real(8),allocatable,optional,intent(out)::x(:)",
        helper_body="if(present(x)) allocate(x(3))")
    assert any(event["resource"] == "argument::a" and "entry deallocation" in event["kind"]
               for event in proof.public()["mutations"])
    assert any(event["resource"] == "argument::a" for event in proof.public()["definition_kills"])
    assert not proof.descriptor_unchanged("argument::a")


@pytest.mark.parametrize("body", ["call unknown(x)", "write(*,*)loc(x)", "write(*,*)c_loc(x)"])
def test_optional_address_or_unknown_call_cannot_borrow_numeric_touch_classification(tmp_path, body):
    _source, proof = projected(tmp_path, helper_body="if(present(x)) then\n"+body+"\nendif")
    assert not proof.public()["complete"]
    assert not proof.preserves_descriptors()


@pytest.mark.parametrize("attribute", ["target", "pointer", "volatile", "asynchronous", "value"])
def test_optional_does_not_hide_an_independently_unsafe_association(tmp_path, attribute):
    _source, proof = projected(tmp_path, declaration=f"real(8),optional,{attribute},intent(inout)::x(:)")
    assert not proof.public()["complete"]
    assert not proof.preserves_descriptors()


def test_hidden_module_mutation_still_interferes_with_optional_actual_without_alias_proof(tmp_path):
    _source, proof = projected(tmp_path, helper_body="deallocate(hidden)\nif(present(x)) x(:)=1.0_8")
    assert proof.public()["complete"]
    assert any(event["resource"] == "storage::hidden" for event in proof.public()["mutations"])
    assert not proof.descriptor_unchanged("argument::a")


def test_optional_attribute_on_non_dummy_does_not_authorize_original_storage(tmp_path):
    _path, source = analysis(tmp_path, "a(1)=hidden(1)", module_declarations="real(8),optional::hidden(3)")
    proof = prove_storage_mutation_effects(source, "storage::advance")
    assert not proof.public()["complete"]
    assert not proof.preserves_descriptors()
