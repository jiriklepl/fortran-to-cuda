"""Private library scratch requires exact original receiver/observer identity."""
import json
from copy import copy
from hashlib import sha256

import pytest
from fparser.two import Fortran2003 as F
from fparser.two.utils import walk

from compiler.frontend.call_bindings import resolve_original_call
from compiler.frontend.opaque_library_state import original_opaque_state
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError

SOURCE = """module opaque_fixture
implicit none
type state_type
real(8)::scratch(2)
end type
contains
function make_state() result(value)
type(state_type)::value
value%scratch=0.0_8
end function
subroutine operation(d,a)
type(state_type),intent(inout)::d
real(8),intent(inout)::a(:)
d%scratch(1)=3.0_8
a=a+d%scratch(1)
end subroutine
subroutine advance(a)
real(8),intent(inout)::a(:)
type(state_type),save::cache
logical,save::called=.false.
if(.not.called)then
cache=make_state()
called=.true.
endif
call operation(cache,a)
! A literal containing the variable spelling is not a typed observer.
print *, 'cache is private library scratch'
end subroutine
end module
"""


def fixture(tmp_path, source=SOURCE):
    path = tmp_path / "source.f90"
    path.write_text(source)
    analysis = SourceEffects([path])
    routine = analysis.routines["opaque_fixture::advance"]
    node = next(item for item in walk(routine.execution)
                if type(item).__name__ == "Call_Stmt" and str(item.items[0]).lower() == "operation")
    constructor = next(item for item in walk(routine.execution)
                       if type(item).__name__ == "Assignment_Stmt" and str(item.items[0]).lower() == "cache")
    resolved = resolve_original_call(analysis, routine.scope, node)
    qualification = tmp_path / "qualification.json"
    qualification.write_text(json.dumps({"fixture_only": True, "not_execution_evidence": True,
                                         "scratch_reuse": "explicit supplied library contract"}))
    contract = {"formal": "d", "constructor": {"span": list(constructor.item.span),
                "statement_sha256": sha256(str(constructor).encode()).hexdigest()},
                "qualification": {"path": str(qualification), "sha256": sha256(qualification.read_bytes()).hexdigest()}}
    return path, analysis, routine, node, resolved, constructor, contract, qualification


def issue(parts):
    _, analysis, routine, node, resolved, _, contract, _ = parts
    return original_opaque_state(analysis, routine, node, resolved, "d", contract)


def test_exact_private_save_receiver_preserves_original_state_without_capture_authority(tmp_path):
    parts = fixture(tmp_path)
    _, analysis, routine, node, resolved, _, _, _ = parts
    proof = issue(parts)
    assert proof.validate(analysis, routine, node, resolved) is proof
    assert issue(parts) is proof
    assert proof.binding is routine.scope.bindings["cache"]
    public = proof.public()
    assert public["storage"] == "original SAVE declaration; no copy or C address export"
    assert public["constructor"]["execution"] == "unchanged original source and guards"
    assert "never a readonly object claim" in public["workspace_effects"]
    for key in ("capture_authorized", "definitions_authorized", "gpu_access_authorized", "metadata_reads_authorized"):
        assert public[key] is False
    public["capture_authorized"] = True
    assert proof.public()["capture_authorized"] is False


@pytest.mark.parametrize("extra", [
    "print *, cache%scratch(1)",
    "call escape(cache)",
    "call escape(cache%scratch)",
    "call operation(cache,a)",
    "cache%scratch(1)=9.0_8",
    "call finalize(cache)",
])
def test_any_other_typed_receiver_use_rejects_including_second_operation(tmp_path, extra):
    source = SOURCE.replace("call operation(cache,a)\n", "call operation(cache,a)\n" + extra + "\n", 1)
    with pytest.raises(CompilationError, match="observer, mutation or escape"):
        issue(fixture(tmp_path, source))


@pytest.mark.parametrize("declaration", [
    "type(state_type)::cache", "type(state_type),save,target::cache", "type(state_type),save,pointer::cache",
    "type(state_type),save,allocatable::cache", "type(state_type),save,volatile::cache",
])
def test_unproved_storage_controls_never_gain_private_receiver_exemption(tmp_path, declaration):
    with pytest.raises(CompilationError, match="local SAVE derived-type"):
        issue(fixture(tmp_path, SOURCE.replace("type(state_type),save::cache", declaration)))


def test_copy_assignment_cannot_be_treated_as_original_constructor(tmp_path):
    source = SOURCE.replace("logical,save::called", "type(state_type)::other\nlogical,save::called")
    source = source.replace("cache=make_state()", "cache=other")
    with pytest.raises(CompilationError, match="constructor changed or copies"):
        issue(fixture(tmp_path, source))


def test_contained_host_association_rejects_even_without_a_reached_call(tmp_path):
    source = SOURCE.replace("end subroutine\nend module", "contains\nsubroutine observer()\nprint *,cache%scratch\n"
                             "end subroutine\nend subroutine\nend module")
    with pytest.raises(CompilationError, match="contained host association"):
        issue(fixture(tmp_path, source))


def test_type_finalization_is_an_observable_receiver_effect(tmp_path):
    source = SOURCE.replace("real(8)::scratch(2)\nend type", "real(8)::scratch(2)\ncontains\nfinal::finish_state\nend type")
    source = source.replace("function make_state()", "subroutine finish_state(value)\ntype(state_type)::value\n"
                            "print *,value%scratch\nend subroutine\nfunction make_state()", 1)
    with pytest.raises(CompilationError, match="finalization or bound methods"):
        issue(fixture(tmp_path, source))


def test_namelist_can_observe_or_mutate_receiver_without_an_execution_name(tmp_path):
    source = SOURCE.replace("logical,save::called", "namelist /private_state/ cache\nlogical,save::called")
    source = source.replace("call operation(cache,a)\n", "call operation(cache,a)\nread(*,nml=private_state)\n", 1)
    # The READ contains only the group name; the specification exposes cache.
    with pytest.raises(CompilationError, match="NAMELIST exposure"):
        issue(fixture(tmp_path, source))


def test_constructor_and_qualification_changes_invalidate_existing_proof(tmp_path):
    parts = fixture(tmp_path)
    _, analysis, routine, node, resolved, constructor, _, qualification = parts
    proof = issue(parts)
    original = constructor.items
    constructor.items = (original[0], original[1], F.Part_Ref("different_factory(1)"))
    # The original call/source authority may reject the AST mutation before
    # the opaque receiver's narrower constructor check runs.
    with pytest.raises(CompilationError, match="constructor changed|unchanged original source-backed"):
        proof.validate(analysis, routine, node, resolved)
    constructor.items = original
    qualification.write_text('{"changed_after_issuance":true}\n')
    with pytest.raises(CompilationError, match="artifact changed"):
        proof.validate(analysis, routine, node, resolved)


def test_copied_foreign_or_unregistered_proofs_cannot_authorize_receiver(tmp_path):
    parts = fixture(tmp_path)
    _, analysis, routine, node, resolved, _, _, _ = parts
    proof = issue(parts)
    for other, call, association in ((copy(proof), node, resolved), (proof, copy(node), resolved),
                                      (proof, node, copy(resolved))):
        with pytest.raises(CompilationError, match="exact compiler-issued"):
            other.validate(analysis, routine, call, association)
    del analysis._opaque_library_state_proofs[proof.identity]
    with pytest.raises(CompilationError, match="exact compiler-issued"):
        proof.validate(analysis, routine, node, resolved)
