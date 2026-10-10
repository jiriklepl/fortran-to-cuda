"""Incomplete child lifetimes retain typed original-call provenance."""

from copy import deepcopy

from compiler.frontend.source_effects import SourceEffects


def hidden_fixture(tmp_path):
    source = tmp_path / "renamed.f90"
    source.write_text("""module mesh
real(8),allocatable::spacing(:)
contains
subroutine measure(a,total)
real(8),intent(in)::a(:)
real(8),intent(out)::total
total=sum(a)/spacing(1)
end subroutine
end module
module renamed
use mesh,only:sample=>measure
contains
subroutine middle(field,total,active)
real(8),intent(in)::field(:)
real(8),intent(out)::total
logical,intent(in)::active
if(active) call sample(field,total)
end subroutine
subroutine advance(values,total,active)
real(8),intent(in)::values(:)
real(8),intent(out)::total
logical,intent(in)::active
call middle(values,total,active)
end subroutine
end module
""")
    return SourceEffects([source])


def test_hidden_incomplete_lifetime_obligation_survives_multiple_calls(tmp_path):
    analysis = hidden_fixture(tmp_path)
    summary = analysis.summarize("renamed::advance")
    assert not summary["complete"]
    assert not summary["ordered_effects"]
    assert not summary["effect_composition"]["available"]
    requirement, = summary["capture_lifetime_requirements"]
    assert requirement == {
        "resource": "mesh::spacing", "authorized": False,
        "source_procedure": "mesh::measure",
        "guard_frames": [{"procedure": "renamed::middle", "condition": "active"}]}
    assert not analysis.stable_module_allocatables
    assert analysis.summarize("mesh::measure")["capture_lifetime_requirements"] == [
        {"resource": "mesh::spacing", "authorized": False}]


def test_original_formal_descriptor_requirement_maps_to_actual_with_every_view(tmp_path):
    source = tmp_path / "views.f90"
    source.write_text("""module views
contains
subroutine leaf(a,total)
real(8),allocatable,intent(in)::a(:)
real(8),intent(out)::total
total=sum(a)
end subroutine
subroutine middle(b,total)
real(8),allocatable,intent(in)::b(:)
real(8),intent(out)::total
call leaf(a=b,total=total)
end subroutine
subroutine advance(values,total)
real(8),allocatable,intent(in)::values(:)
real(8),intent(out)::total
call middle(values,total)
end subroutine
end module
""")
    analysis = SourceEffects([source])
    summary = analysis.summarize("views::advance")
    assert summary["complete"], summary["reasons"]
    requirement, = summary["capture_lifetime_requirements"]
    assert requirement["resource"] == "argument::values"
    assert requirement["authorized"] is True
    assert requirement["requirement"] == "original_readonly_allocatable_descriptor"
    assert requirement["execution_requires_runtime_guard"]
    assert requirement["source_procedure"] == "views::leaf"
    outer, inner = requirement["view_chain"]
    assert (outer["caller"], outer["callee"], outer["formal_resource"], outer["resource"]) == (
        "views::advance", "views::middle", "argument::b", "argument::values")
    assert (inner["caller"], inner["callee"], inner["formal_resource"], inner["resource"]) == (
        "views::middle", "views::leaf", "argument::a", "argument::b")
    assert outer["actual_descriptor"]["attributes"] == ["allocatable", "intent"]
    assert outer["actual_descriptor"]["intent"] == "in"
    assert requirement["guard_frames"] == []
    saved = deepcopy(summary)
    requirement["resource"] = "forged::binding"
    assert analysis.summarize("views::advance") == saved


def test_repeated_views_and_guards_do_not_collapse_into_one_root_fact(tmp_path):
    analysis = hidden_fixture(tmp_path)
    routine = analysis.routines["renamed::advance"]
    # Use genuine source changes and rebuild, not synthetic authority.
    path = routine.scope.path
    path.write_text(path.read_text().replace("call middle(values,total,active)",
        "call middle(values,total,active)\nif(.not.active) call sample(values,total)"))
    analysis = SourceEffects([path])
    summary = analysis.summarize("renamed::advance")
    assert not summary["complete"]
    requirements = summary["capture_lifetime_requirements"]
    assert len(requirements) == 2
    assert {item["resource"] for item in requirements} == {"mesh::spacing"}
    assert {tuple((frame["procedure"], frame["condition"]) for frame in item["guard_frames"])
            for item in requirements} == {
                (("renamed::middle", "active"),), (("renamed::advance", ".NOT. active"),)}
    assert not any(item["authorized"] for item in requirements)


def test_reached_segment_retains_lifetime_obligation_without_entry_events(tmp_path):
    analysis = hidden_fixture(tmp_path)
    routine = analysis.routines["renamed::advance"]
    selected, = routine.execution.content
    summary = analysis.segment_summary(routine.qualified, (selected,), capture_locals=True)
    assert not summary["complete"]
    assert summary["includes_entry"] is False
    assert summary["capture_lifetime_requirements"][0]["resource"] == "mesh::spacing"
    assert summary["capture_lifetime_requirements"][0]["authorized"] is False
    assert summary["definition_changes"] == []


def test_proved_hidden_storage_preserves_authority_without_blessing_others(tmp_path):
    original = hidden_fixture(tmp_path)
    analysis = SourceEffects(list(original.sources))
    analysis.authorize_stable_module_allocatables({"mesh::spacing"})
    summary = analysis.summarize("renamed::advance")
    assert summary["complete"], summary["reasons"]
    requirement, = summary["capture_lifetime_requirements"]
    assert requirement["authorized"]
    assert requirement["resource"] == "mesh::spacing"
    assert original.summarize("renamed::advance")["capture_lifetime_requirements"][0]["authorized"] is False
