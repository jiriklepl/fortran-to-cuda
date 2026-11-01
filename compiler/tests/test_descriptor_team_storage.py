"""Joined-team lifetime evidence does not require numerical execution proof."""
from copy import copy

import pytest

from compiler.frontend.descriptor_teams import prove_descriptor_team
from compiler.frontend.descriptor_windows import prove_descriptor_window
from compiler.frontend.native_group_structure import NativeGroupCandidate, grouped_original_sequence
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError

ENTRY = "storage_team::advance"
CHILD = "storage_team::update"


def fixture(tmp_path, *, opening="parallel private(i,scratch)", ending="end parallel",
            inner_open="!$omp do schedule(dynamic,1)", inner_end="!$omp end do nowait", action="", specification=""):
    path = tmp_path / "teams.f90"
    path.write_text(f"""module storage_team
contains
subroutine update(a,n)
real(8),contiguous,intent(inout)::a(:)
integer,intent(in)::n
integer::i
real(8)::scratch(n)
{specification}
!$omp {opening}
{inner_open}
do i=1,n
scratch(i)=a(i)
a(i)=scratch(i)+1.0_8
{action}
end do
{inner_end}
!$omp {ending}
end subroutine
subroutine advance(a,n)
real(8),allocatable,intent(inout)::a(:)
integer,intent(in)::n
real(8),allocatable::temporary(:)
call update(a,n)
call move_alloc(a,temporary)
end subroutine
end module
""")
    analysis = SourceEffects([path])
    routine = analysis.routines[CHILD]
    groups = [item for item in grouped_original_sequence(routine.execution.content, {})
              if isinstance(item, NativeGroupCandidate)]
    if len(groups) != 1:
        raise CompilationError("descriptor team requires its complete original source group")
    group, = groups
    owner = analysis.routines[ENTRY]
    selected = (analysis.structure(ENTRY).node_id(owner.execution.content[0]),)
    return path, analysis, routine, group, selected


def test_dynamic_private_storage_has_lifetime_evidence_without_payload_or_execution_authority(tmp_path):
    path, analysis, _routine, _group, selected = fixture(tmp_path)
    before = path.read_bytes()
    proof = prove_descriptor_window(analysis, ENTRY, selected, ("argument::a",))
    public = proof.public()
    join, = public["native_joined_descriptor_requirements"]
    assert join["proof_role"] == "original_joined_storage_structure"
    storage = join["storage_structure"]
    assert storage["retained_internal_nowait"] == 1
    assert storage["requires_complete_descriptor_body_and_call_rescan"]
    assert set(storage["private_bindings"]) == {CHILD + "::i", CHILD + "::scratch"}
    assert not storage["entry_definition_authority"]
    assert not storage["payload_access_authorized"]
    assert not storage["execution_authorized"]
    assert not storage["gpu_legality_established"]
    assert not analysis._joined_completions
    assert not analysis._numerical_completions
    assert not getattr(analysis, "_descriptor_team_proofs", {})
    assert public["resources"][0]["requires_prior_noescape_and_current_allocation_proof"]
    assert not public["entry_definition_authority"]
    assert path.read_bytes() == before


def test_combined_team_retains_its_original_implicit_join(tmp_path):
    _path, analysis, _routine, _group, selected = fixture(
        tmp_path, opening="parallel do private(i,scratch)", ending="end parallel do",
        inner_open="", inner_end="")
    public = prove_descriptor_window(analysis, ENTRY, selected, ("argument::a",)).public()
    assert public["native_joined_descriptor_requirements"]


@pytest.mark.parametrize("opening", [
    "parallel private(a)", "parallel firstprivate(a)", "parallel copyin(a)",
    "parallel default(private)", "parallel private(scratch) num_threads(unknown())",
])
def test_uncertain_data_sharing_cannot_authorize_original_allocation_lifetime(tmp_path, opening):
    _path, analysis, routine, group, _selected = fixture(tmp_path, opening=opening)
    with pytest.raises(CompilationError, match="PRIVATE|clause|DEFAULT"):
        prove_descriptor_team(analysis, routine.qualified, group.original_nodes)


@pytest.mark.parametrize("directive", ["!$omp task", "!$omp target", "!$omp flush(a)",
                                        "!$omp atomic", "!$omp cancel parallel"])
def test_async_task_and_unproved_synchronization_are_not_storage_authority(tmp_path, directive):
    _path, analysis, routine, group, _selected = fixture(tmp_path, action=directive)
    with pytest.raises(CompilationError, match="unsupported directive"):
        prove_descriptor_team(analysis, routine.qualified, group.original_nodes)


@pytest.mark.parametrize("specification", ["common /hidden/ i", "namelist /hidden/ i",
                                             "!$omp threadprivate(i)"])
def test_declaration_aliases_are_not_hidden_by_a_complete_team(tmp_path, specification):
    def issue_original():
        _path, analysis, routine, group, _selected = fixture(tmp_path, specification=specification)
        return prove_descriptor_team(analysis, routine.qualified, group.original_nodes)

    with pytest.raises(CompilationError, match="independent storage|complete original"):
        issue_original()


def test_storage_team_does_not_hide_unknown_calls_or_callee_storage_effects(tmp_path):
    _path, analysis, _routine, _group, selected = fixture(tmp_path, action="call absent(a)")
    with pytest.raises(CompilationError, match="unknown selected storage|unresolved|original direct"):
        prove_descriptor_window(analysis, ENTRY, selected, ("argument::a",))
    assert not getattr(analysis, "_descriptor_window_proofs", {})


def test_registry_original_region_and_source_are_revalidated(tmp_path):
    path, analysis, routine, group, _selected = fixture(tmp_path)
    proof = prove_descriptor_team(analysis, routine.qualified, group.original_nodes)
    assert proof.validate(analysis, routine.qualified, proof.selected_node_ids) is proof
    with pytest.raises(CompilationError, match="exact original compiler-issued"):
        copy(proof).validate(analysis, routine.qualified, proof.selected_node_ids)
    with pytest.raises(CompilationError, match="exact original compiler-issued"):
        proof.validate(SourceEffects([path]), routine.qualified, proof.selected_node_ids)
    with pytest.raises(CompilationError, match="complete original joined region"):
        prove_descriptor_team(analysis, routine.qualified, group.original_nodes[:-1])
    public = proof.public()
    public["entry_definition_authority"] = True
    assert not proof.public()["entry_definition_authority"]
    path.write_text(path.read_text().replace("scratch(i)=a(i)", "scratch(i)=a(i)+2.0_8"))
    with pytest.raises(CompilationError, match="source changed"):
        proof.validate(analysis, routine.qualified, proof.selected_node_ids)
