"""Presence forwarding is descriptor evidence, never an optional value read."""
from copy import copy

import pytest

from compiler.frontend.descriptor_windows import prove_descriptor_window
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError

SOURCE = """module optional_descriptors
contains
subroutine leaf(a,n,regime)
real(8),intent(inout)::a(:)
integer,intent(in)::n
integer,optional,intent(in)::regime
integer::i,factor
if(present(regime))then
factor=regime
else
factor=1
endif
do i=1,n
a(i)=a(i)*factor
enddo
end subroutine
subroutine wrapper(a,n,regime)
real(8),intent(inout)::a(:)
integer,intent(in)::n
integer,optional,intent(in)::regime
call leaf(a,n,regime)
end subroutine
subroutine advance(a,n,regime)
real(8),allocatable,intent(inout)::a(:)
integer,intent(in)::n
integer,optional,intent(in)::regime
call wrapper(a,n,regime)
deallocate(a)
end subroutine
end module
"""


def fixture(tmp_path, source=SOURCE):
    path = tmp_path / "source.f90"
    path.write_text(source)
    analysis = SourceEffects([path])
    entry = "optional_descriptors::advance"
    routine = analysis.routines[entry]
    selection = (analysis.structure(entry).node_id(routine.execution.content[0]),)
    return path, analysis, entry, selection


@pytest.mark.parametrize(("actual", "presence"), [("regime", "forwarded_optional"), ("4", "supplied"),
                                                    (None, "omitted")])
def test_original_optional_scalar_in_presence_preserved_through_complete_callees(tmp_path, actual, presence):
    call = "call wrapper(a,n" + ("," + actual if actual is not None else "") + ")"
    source = SOURCE.replace("call wrapper(a,n,regime)", call)
    path, analysis, entry, selection = fixture(tmp_path, source)
    before = path.read_bytes()
    proof = prove_descriptor_window(analysis, entry, selection, ("argument::a",))
    public = proof.public()
    records = public["native_optional_scalar_requirements"]
    assert {record["callee"] for record in records} == {"optional_descriptors::wrapper", "optional_descriptors::leaf"}
    outer, = [record for record in records if record["caller"] == entry]
    assert outer["presence"] == presence
    inner, = [record for record in records if record["caller"] == "optional_descriptors::wrapper"]
    assert inner["presence"] == "forwarded_optional"
    for record in records:
        assert record["formal"] == "regime"
        assert record["original_presence_and_evaluation_required"] is True
        assert record["value_evaluation_authorized"] is False
        assert record["payload_access_authorized"] is False
        assert record["entry_definition_authority"] is False
        assert record["execution_authorized"] is False
    assert public["entry_definition_authority"] is False
    assert public["execution_authorized"] is False
    assert path.read_bytes() == before
    records[0]["value_evaluation_authorized"] = True
    assert proof.public()["native_optional_scalar_requirements"][0]["value_evaluation_authorized"] is False


@pytest.mark.parametrize("attributes", ["intent(inout)", "target,intent(in)", "pointer,intent(in)",
                                      "allocatable,intent(in)", "volatile,intent(in)", "asynchronous,intent(in)",
                                      "value,intent(in)"])
def test_optional_mutation_alias_dynamic_or_by_value_formals_still_reject(tmp_path, attributes):
    source = SOURCE.replace("optional,intent(in)::regime", "optional," + attributes + "::regime")
    _path, analysis, entry, selection = fixture(tmp_path, source)
    with pytest.raises(CompilationError):
        prove_descriptor_window(analysis, entry, selection, ("argument::a",))


def test_optional_array_never_gains_scalar_presence_exemption(tmp_path):
    source = SOURCE.replace("::regime", "::regime(:)").replace("factor=regime", "factor=regime(1)")
    _path, analysis, entry, selection = fixture(tmp_path, source)
    with pytest.raises(CompilationError, match="association|formal storage"):
        prove_descriptor_window(analysis, entry, selection, ("argument::a",))


@pytest.mark.parametrize("attributes", ["allocatable", "pointer"])
def test_runtime_allocation_or_association_dependent_presence_is_not_forwarding(tmp_path, attributes):
    source = SOURCE.replace("integer,optional,intent(in)::regime\ncall wrapper", "integer," + attributes
                            + "::regime\ncall wrapper")
    _path, analysis, entry, selection = fixture(tmp_path, source)
    with pytest.raises(CompilationError, match="association/escape"):
        prove_descriptor_window(analysis, entry, selection, ("argument::a",))


def test_hidden_descriptor_mutation_still_rejects_on_every_optional_branch(tmp_path):
    source = SOURCE.replace("real(8),intent(inout)::a(:)", "real(8),allocatable,intent(inout)::a(:)")
    source = source.replace("factor=regime", "deallocate(a)\nfactor=regime")
    _path, analysis, entry, selection = fixture(tmp_path, source)
    with pytest.raises(CompilationError, match="unknown selected storage effects"):
        prove_descriptor_window(analysis, entry, selection, ("argument::a",))


def test_optional_presence_records_do_not_survive_binding_or_token_substitution(tmp_path):
    _path, analysis, entry, selection = fixture(tmp_path)
    proof = prove_descriptor_window(analysis, entry, selection, ("argument::a",))
    with pytest.raises(CompilationError, match="registered original interval"):
        copy(proof).validate(analysis, entry, selection)
    binding = analysis.routines["optional_descriptors::leaf"].scope.bindings["regime"]
    binding.attributes |= {"target"}
    with pytest.raises(CompilationError):
        proof.validate(analysis, entry, selection)
