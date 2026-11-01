"""Original scalar clocks preserve descriptors without granting host execution."""
import pytest
from fparser.two.utils import walk

from compiler.frontend.descriptor_windows import prove_descriptor_window
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError

SOURCE = """module descriptor_clocks
implicit none
contains
subroutine advance(a,n)
real(8),allocatable,intent(inout)::a(:)
integer,intent(in)::n
integer(8)::ticks
if(n>0)call system_clock(count=ticks)
end subroutine
end module
"""


def fixture(tmp_path, source=SOURCE):
    path = tmp_path / "source.f90"
    path.write_text(source)
    analysis = SourceEffects([path])
    analysis.seal_session()
    routine = analysis.routines["descriptor_clocks::advance"]
    selection = tuple(routine.execution.content)
    return analysis, routine, selection


def issue(parts):
    analysis, routine, selection = parts
    _graph, selected = analysis._selected_source(routine.qualified, selection)
    return prove_descriptor_window(analysis, routine.qualified, selected, ("argument::a",))


@pytest.mark.parametrize("conditional", [False, True])
def test_only_scalar_clock_source_effects_are_added_to_descriptor_evidence(tmp_path, conditional):
    source = SOURCE if conditional else SOURCE.replace("if(n>0)", "")
    analysis, routine, selection = parts = fixture(tmp_path, source)
    before = dict(analysis._scalar_clock_proofs) if hasattr(analysis, "_scalar_clock_proofs") else {}
    proof = issue(parts)
    record = proof.public()
    assert record["schema_version"] == 12
    clock, = record["native_scalar_clock_descriptor_requirements"]
    assert clock["intrinsic"] == "SYSTEM_CLOCK"
    assert clock["descriptor_evidence_only"] is True
    assert clock["payload_access_authorized"] is False
    assert clock["definitions_authorized"] is False
    assert clock["execution_authorized"] is False
    assert clock["placement_authorized"] is False
    assert clock["outputs"]["count"]["resource"] == "descriptor_clocks::advance::ticks"
    # The SAME scanner owns its ephemeral clock-token registry; it does not
    # publish native execution authority into the caller analysis.
    assert getattr(analysis, "_scalar_clock_proofs", {}) == before
    assert proof.validate(analysis, routine.qualified, analysis._selected_source(routine.qualified, selection)[1]) is proof


@pytest.mark.parametrize("declaration", ["integer(8),target::ticks", "integer(8),pointer::ticks",
    "integer(8)::ticks(2)"])
def test_clock_outputs_cannot_be_arrays_or_associated_storage(tmp_path, declaration):
    with pytest.raises(CompilationError):
        issue(fixture(tmp_path, SOURCE.replace("integer(8)::ticks", declaration)))


def test_original_same_named_source_procedure_keeps_full_closure_and_no_intrinsic_authority(tmp_path):
    source = SOURCE.replace("end module", "subroutine system_clock(count)\ninteger(8),intent(out)::count\ncount=3\nend subroutine\nend module")
    record = issue(fixture(tmp_path, source)).public()
    assert "native_scalar_clock_descriptor_requirements" not in record
    assert "descriptor_clocks::system_clock" in record["source_call_closure"]


def test_standard_clock_does_not_hide_an_unsafe_original_suffix(tmp_path):
    source = SOURCE.replace("if(n>0)call system_clock(count=ticks)",
                            "if(n>0)call system_clock(count=ticks)\ndeallocate(a)")
    with pytest.raises(CompilationError):
        issue(fixture(tmp_path, source))


def test_clock_content_identity_is_checked_at_descriptor_publication(tmp_path, monkeypatch):
    from compiler.frontend import descriptor_windows as service

    parts = fixture(tmp_path)
    original = service._Check.scalar_clock
    fired = False

    def alter(check, routine, call):
        nonlocal fired
        accepted = original(check, routine, call)
        if accepted and not fired:
            fired = True
            _routine, _call, proof = check.scalar_clock_tokens[id(call)]
            object.__setattr__(proof, "identity", "0"*64)
            check.join_analysis._scalar_clock_proofs[proof.identity] = proof
        return accepted

    monkeypatch.setattr(service._Check, "scalar_clock", alter)
    with pytest.raises(CompilationError, match="scalar clock"):
        issue(parts)
    assert fired


def test_clock_receipt_is_bound_to_the_exact_original_output_name(tmp_path):
    analysis, routine, selection = parts = fixture(tmp_path)
    proof = issue(parts)
    call = next(node for node in walk(routine.execution) if type(node).__name__ == "Call_Stmt")
    with pytest.raises(CompilationError, match="immutable"):
        call.items = (call.items[0], None)
    assert proof.validate(analysis, routine.qualified, analysis._selected_source(routine.qualified, selection)[1]) is proof
