"""Implicit original loop ends retain every independent window obligation."""

import pytest

from compiler.frontend.descriptor_windows import prove_descriptor_window
from compiler.frontend.native_group_structure import NativeGroupCandidate, grouped_original_sequence
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError
from compiler.tests.test_descriptor_private_allocations import ROOT, SOURCE
from compiler.tests.test_descriptor_private_allocations import fixture as private_fixture
from compiler.tests.test_descriptor_private_allocations import issue as private_issue
from compiler.tests.test_descriptor_team_storage import ENTRY
from compiler.tests.test_descriptor_team_storage import fixture as call_fixture


def no_grants(record):
    for name in ("entry_definition_authority", "payload_access_authorized", "execution_authorized",
                 "gpu_legality_established"):
        if name in record:
            assert record[name] is False


def feature_fixture(tmp_path, region, *, sealed):
    # Three original loops exercise the feature composition on the ordinary
    # source graph. This fixture makes no deferred-group or 76-loop claim.
    loop = """!$omp do reduction(+:total)
do i=1,n
scratch(i)=a(i)
total=total+scratch(i)
end do
"""
    text = """module implicit_features
contains
subroutine update(a,n)
real(8),intent(inout)::a(:)
integer,intent(in)::n
integer::i
real(8)::scratch(n),total
total=0
!$omp parallel private(i,scratch)
""" + loop * 3 + "!$omp " + region + """
block
use,intrinsic::iso_fortran_env,only:real64
real(real64)::value
value=total
a(1)=value
end block
!$omp end """ + region + """
!$omp end parallel
end subroutine
subroutine advance(a,n)
real(8),allocatable,intent(inout)::a(:)
integer,intent(in)::n
call update(a,n)
end subroutine
end module
"""
    path = tmp_path / "features.f90"
    path.write_text(text)
    analysis = SourceEffects([path])
    if sealed:
        analysis.seal_session()
    child = analysis.routines["implicit_features::update"]
    group, = (item for item in grouped_original_sequence(child.execution.content, {})
              if isinstance(item, NativeGroupCandidate))
    owner = analysis.routines["implicit_features::advance"]
    graph, selected = analysis._selected_source(owner.qualified, tuple(owner.execution.content))
    return path, analysis, child, group, owner, graph, selected


def test_native_child_join_retains_exact_implicit_loop_end_in_window22(tmp_path):
    path, analysis, _routine, _group, selected = call_fixture(tmp_path, inner_end="")
    before = path.read_bytes()
    proof = prove_descriptor_window(analysis, ENTRY, selected, ("argument::a",))
    public = proof.public()
    assert public["schema_version"] == 22
    join, = public["native_joined_descriptor_requirements"]
    team = join["storage_structure"]
    assert team["version"] == 4
    ending, = team["implicit_worksharing_endings"]
    assert ending["implicit_default_barrier_retained"] is True
    assert team["original_directives"][-1] == "end parallel"
    assert "end do" not in team["original_directives"]
    for record in (team, public):
        no_grants(record)
    assert proof.validate(analysis, ENTRY, selected) is proof
    assert path.read_bytes() == before


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("region", ["single", "master"])
def test_implicit_windows_keep_master_single_reduction_and_block_obligations(tmp_path, sealed, region):
    path, analysis, child, original_group, routine, _graph, selected = feature_fixture(
        tmp_path, region, sealed=sealed)
    before = path.read_bytes()
    proof = prove_descriptor_window(analysis, routine.qualified, selected, ("argument::a",))
    public = proof.public()
    assert public["schema_version"] == 22
    group, = public["native_joined_block_requirements"]
    team = group["team_storage"]
    assert team["version"] == 4
    assert len(team["implicit_worksharing_endings"]) == 3
    assert len(team["intrinsic_scalar_reductions"]) == 3
    assert set(team["private_bindings"]) == {child.qualified + "::i", child.qualified + "::scratch"}
    assert team["retained_single_regions"] == int(region == "single")
    assert team["parent_parallel_join_required"] is True
    assert len(group["storage_mutation"]["fixed_scalar_blocks"]) == 1
    assert group["storage_mutation"]["complete"] is True
    assert any("scalar reduction, SINGLE" in item for item in public["limits"])
    if region == "master":
        assert team["retained_master_regions"] == 1
        assert team["master_barrier_assumed"] is False
        assert team["master_completion_authorized"] is False
        assert any("END MASTER implies no barrier" in item for item in public["limits"])
    for record in (team, group, group["storage_mutation"], public):
        no_grants(record)
    assert proof.validate(analysis, routine.qualified, selected) is proof
    for completion in (analysis.joined_completion, analysis.numerical_joined_completion):
        with pytest.raises(CompilationError):
            completion(child.qualified, original_group.original_nodes)
    assert path.read_bytes() == before
    authority = analysis._summary_authority()
    assert authority["descriptor_team_implicit_worksharing_version"] == 4
    assert authority["descriptor_window_implicit_worksharing_version"] == 22


@pytest.mark.parametrize("sealed", [False, True])
def test_implicit_team_keeps_all_private_allocation_exclusion_requirements(tmp_path, sealed):
    text = SOURCE.replace("integer::i", "integer::i\nreal(8)::temporary(size(a))")
    text = text.replace("a(1)=a(1)+sum(scratch)", """!$omp parallel private(i,temporary)
!$omp do schedule(dynamic,1)
do i=1,size(a)
temporary(i)=scratch(i)
a(i)=a(i)+temporary(i)
end do
!$omp end parallel""")
    parts = private_fixture(tmp_path, source=text, sealed=sealed)
    before = parts[0].read_bytes()
    proof = private_issue(parts)
    public = proof.public()
    assert public["schema_version"] == 22
    assert public["requires_unregistered_private_allocations"] == [ROOT]
    assert public["native_private_allocation_requirements"][0]["joined_storage"]
    assert not public["native_private_allocation_requirements"][0]["allocation_presence_authority"]
    join, = public["native_joined_descriptor_requirements"]
    assert join["proof_role"] == "original_joined_storage_structure"
    assert join["storage_structure"]["version"] == 4
    assert len(join["storage_structure"]["implicit_worksharing_endings"]) == 1
    child = parts[1].routines["private_scratch::work"]
    group, = (item for item in grouped_original_sequence(child.execution.content, {})
              if isinstance(item, NativeGroupCandidate))
    for completion in (parts[1].joined_completion, parts[1].numerical_joined_completion):
        with pytest.raises(CompilationError):
            completion(child.qualified, group.original_nodes)
    assert proof.validate(parts[1], parts[2].qualified, parts[4]) is proof
    no_grants(public)
    assert parts[0].read_bytes() == before
