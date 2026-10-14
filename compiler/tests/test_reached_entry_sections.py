"""Physical entry contracts stay partial in original logical coordinates."""

from copy import deepcopy
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.frontend.reached_windows import plan_reached_window, prove_reached_window
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.reached_owner import ReachedWindowOwner
from compiler.scopes.source import ScopeBuilder

ENTRY = "frost_fields::advance"
ROOTS = ("frost_fields::a", "frost_fields::b")
SOURCE = """module frost_fields
implicit none
real(8),allocatable::a(:),b(:)
integer::prefix_visits=0,tail_visits=0
contains
subroutine advance(iterations)
integer,intent(in)::iterations
integer::stage,i
prefix_visits=prefix_visits+1
do stage=1,iterations
do i=-3,5
b(i)=a(i)+real(i,8)
enddo
do i=-3,5
a(i)=2.0_8*b(i)
enddo
tail_visits=tail_visits+1
enddo
end subroutine
end module
"""


def entry_sections_fixture(directory, *, source=SOURCE, precision=8, policy="sections", device_budget=None):
    directory.mkdir(parents=True, exist_ok=True)
    source = source.replace("real(8)", f"real({precision})").replace(",8)", f",{precision})").replace(
        "_8", f"_{precision}")
    path = directory / "fields.f90"
    path.write_text(source)
    captures = {root: {"storage": "stable", "initialized": "none", "escapes": False,
                       "allocation_changes": False, "lifetime": "selected_window"} for root in ROOTS}
    captures[ROOTS[0]].update(initialized="sections", sections=[{"lower": [1], "upper": [10]}],
                             expected_descriptor={"lower_bounds": [-4], "extents": [12]})
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()}, "captures": captures}
    if device_budget is not None:
        facts["device_budget_bytes"] = device_budget
    builder = ScopeBuilder([path], ENTRY, facts=facts, options=CompilerOptions(),
                           config=OffloadConfig(policy=policy, scope_execution="reached"))
    outer = [node for node in builder.entry.execution.content
             if type(node).__name__ == "Block_Nonlabel_Do_Construct"][-1]
    originals = tuple(node for node in outer.content[1:-1]
                      if type(node).__name__ == "Block_Nonlabel_Do_Construct")
    return path, builder, originals


@pytest.mark.parametrize("planned", [False, True])
def test_entry_physical_sections_are_guarded_partial_logical_definitions(tmp_path, planned):
    _path, builder, originals = entry_sections_fixture(tmp_path)
    original_facts = deepcopy(builder.facts)
    prove = plan_reached_window if planned else prove_reached_window
    proof = prove(builder.analysis, ENTRY, originals, ROOTS, builder.facts)
    state = proof.capture_state(ROOTS[0])
    assert state["initialized"] == "sections"
    resource = proof.activation_sections(ROOTS[0])
    assert resource.lower_bounds == (-4,)
    box, = resource.overwrites
    axis, = box.axes
    assert (axis.lower.value, axis.upper.value) == (-3, 5)
    assert not resource.reads
    assert not resource.writes
    entry, = proof.public()["entry_section_facts"]
    assert entry["physical_sections"] == [{"lower": [1], "upper": [10]}]
    assert not entry["source_overwrite_proven"]
    assert not entry["bounding_hull_inferred"]
    assert not proof.public()["execution_authorized"]
    assert builder.facts == original_facts


def test_partial_entry_emits_original_descriptor_guard_before_section_registration(tmp_path):
    path, builder, originals = entry_sections_fixture(tmp_path)
    proof = plan_reached_window(builder.analysis, ENTRY, originals, ROOTS, builder.facts)
    owner = ReachedWindowOwner(builder, proof, originals)
    assert owner.run()
    outputs, report = owner.builder.finish()
    text = outputs[report["sources"][str(path)]["replacement"]]
    activation = text.index(f"if (.not. {owner.spent}) then")
    allocation = text.index("allocated(a)", activation)
    descriptor = text.index("lbound(a, 1, kind=c_int64_t)", allocation)
    registration = text.index("fort_scope_register_sections(", descriptor)
    assert activation < allocation < descriptor < registration
    assert "size(a, 1, kind=c_int64_t) == 12_c_int64_t" in text
    assert len(report["scopes"][0]["gpu_leaves"]) == 2
    assert path.read_text() == SOURCE


@pytest.mark.parametrize("change", [
    {"sections": [{"lower": [1], "upper": [13]}]},
    {"sections": [{"lower": [-1], "upper": [10]}]},
    {"sections": [{"lower": [True], "upper": [10]}]},
    {"sections": [{"lower": [10], "upper": [1]}]},
    {"sections": [{"lower": [], "upper": []}]},
    {"sections": []},
    {"sections": [{"lower": [2], "upper": [2]}]},
    {"expected_descriptor": {"lower_bounds": [2**31], "extents": [12]}},
])
def test_malformed_empty_or_uncovered_entry_never_becomes_whole(tmp_path, change):
    _path, builder, originals = entry_sections_fixture(tmp_path)
    builder.facts["captures"][ROOTS[0]].update(change)
    with pytest.raises(CompilationError, match="capture section|incomplete definition|source INTEGER ABI"):
        prove_reached_window(builder.analysis, ENTRY, originals, ROOTS, builder.facts)


def test_entry_section_proof_requires_original_descriptor_contract(tmp_path):
    _path, builder, originals = entry_sections_fixture(tmp_path)
    del builder.facts["captures"][ROOTS[0]]["expected_descriptor"]
    with pytest.raises(CompilationError, match="expected original descriptor"):
        prove_reached_window(builder.analysis, ENTRY, originals, ROOTS, builder.facts)


def test_separate_faces_do_not_authorize_an_uncovered_middle(tmp_path):
    _path, builder, originals = entry_sections_fixture(tmp_path)
    builder.facts["captures"][ROOTS[0]]["sections"] = [
        {"lower": [0], "upper": [1]}, {"lower": [11], "upper": [12]}]
    with pytest.raises(CompilationError, match="incomplete definition"):
        prove_reached_window(builder.analysis, ENTRY, originals, ROOTS, builder.facts)


def test_opposite_faces_keep_two_rectangles_without_defining_the_volume(tmp_path):
    source = SOURCE.replace("do i=-3,5\nb(i)=a(i)+real(i,8)", "do i=-4,-4\nb(i)=a(i)").replace(
        "do i=-3,5\na(i)=2.0_8*b(i)", "do i=7,7\nb(i)=a(i)")
    _path, builder, originals = entry_sections_fixture(tmp_path, source=source)
    builder.facts["captures"][ROOTS[0]]["sections"] = [
        {"lower": [0], "upper": [1]}, {"lower": [11], "upper": [12]}]
    proof = prove_reached_window(builder.analysis, ENTRY, originals, ROOTS, builder.facts)
    resource = proof.activation_sections(ROOTS[0])
    assert [(box.axes[0].lower.value, box.axes[0].upper.value) for box in resource.overwrites] == [(-4, -4), (7, 7)]
    assert proof.capture_state(ROOTS[0])["initialized"] == "sections"
    assert proof.capture_state(ROOTS[1])["coverage_on_return"] == "sections"


def test_halo_read_beyond_asserted_sections_is_rejected(tmp_path):
    source = SOURCE.replace("b(i)=a(i)", "b(i)=a(i+1)")
    _path, builder, originals = entry_sections_fixture(tmp_path, source=source)
    with pytest.raises(CompilationError, match="incomplete definition"):
        prove_reached_window(builder.analysis, ENTRY, originals, ROOTS, builder.facts)


def test_native_prefix_out_event_kills_asserted_entry_sections(tmp_path):
    source = SOURCE.replace("subroutine advance(iterations)", """subroutine erase(values)
real(8),intent(out)::values(:)
values(1)=0.0_8
end subroutine
subroutine advance(iterations)""").replace("prefix_visits=prefix_visits+1", "call erase(a)")
    _path, builder, originals = entry_sections_fixture(tmp_path, source=source)
    with pytest.raises(CompilationError, match="incomplete definition"):
        prove_reached_window(builder.analysis, ENTRY, originals, ROOTS, builder.facts)


def test_selected_position_sections_do_not_authorize_earlier_joined_prefix_reads(tmp_path):
    source = SOURCE.replace("prefix_visits=prefix_visits+1", """!$omp parallel private(i)
!$omp do
do i=-3,5
a(i)=a(i)+1.0_8
enddo
!$omp end do
!$omp end parallel""")
    _path, builder, originals = entry_sections_fixture(tmp_path, source=source)
    proof = plan_reached_window(builder.analysis, ENTRY, originals, ROOTS, builder.facts)
    eligibility = proof.public()["prefix_eligibility"]
    assert not eligibility["activation_path_possible"]
    assert any("selected-window whole or partial assertion" in point["reason"]
               for point in eligibility["disable_points"])


def test_changed_source_invalidates_partial_entry_proof(tmp_path):
    path, builder, originals = entry_sections_fixture(tmp_path)
    proof = prove_reached_window(builder.analysis, ENTRY, originals, ROOTS, builder.facts)
    sections = proof.activation_sections(ROOTS[0])
    path.write_text(path.read_text().replace("i=-3,5", "i=-3,6"))
    assert proof.activation_sections(ROOTS[0]) == sections
    with pytest.raises(CompilationError, match="source changed"):
        builder.finish()
    assert not builder.outputs
    assert not builder.edits
