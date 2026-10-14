"""Unused original array actuals do not acquire managed storage authority."""

import re

import pytest

from compiler.ir import CompilationError
from compiler.tests.test_mapped_child_reached_plans import (
    CHILD_SOURCE,
    PARENT_SOURCE,
    emit_mapped_child_window,
    mapped_child_fixture,
)


def projected_sources(*, intent="in", body="", actual="c"):
    child = CHILD_SOURCE.replace("bridge(x,y,n,bad)", "bridge(x,y,z,n,bad)").replace(
        "integer,intent(in)::n", f"real(8),intent({intent})::z(-7:)\ninteger,intent(in)::n")
    if body:
        child = child.replace("call native_workspace(x)", body)
    parent = PARENT_SOURCE.replace("advance(a,b,n,iterations,bad)", "advance(a,b,c,n,iterations,bad)").replace(
        "integer,intent(in)::n,iterations", "real(8),allocatable,intent(inout)::c(:)\ninteger,intent(in)::n,iterations")
    parent = parent.replace("call renamed_bridge(b,a,n,bad)", f"call renamed_bridge(b,a,{actual},n,bad)")
    return {"parent_source": parent, "child_source": child}


@pytest.mark.parametrize("intent", ["in", "inout"])
def test_unused_actual_keeps_original_association_without_managed_state(tmp_path, intent):
    paths, outputs, report, owner = emit_mapped_child_window(tmp_path, **projected_sources(intent=intent))
    mapped, = owner.window_proof.public()["planned_closures"]["mapped_child_plans"]
    projection = mapped["managed_actual_projection"]
    assert projection["managed_associations"] == {"argument::x": "argument::b", "argument::y": "argument::a"}
    assert projection["unmanaged_associations"] == {"argument::z": "argument::c"}
    assert projection["original_associations_preserved"]
    assert not projection["unmanaged_storage_authority"]
    assert "argument::c" not in owner.resources
    child = owner.members["inner_fields::bridge"]
    assert "argument::z" not in child.resources
    assert "argument::z" not in {state["resource"] for state in mapped["plan"]["states"]}
    association = next(item for item in mapped["coordinate_proof"]["coordinate_associations"]
                       if item["formal_resource"] == "argument::z")
    assert association["formal_lower_origins"][0]["value"] == -7
    assert association["actual_lower_origins"][0]["resource"] == "argument::c"
    with pytest.raises(CompilationError, match="exceeds its reached managed projection"):
        child.builder.capture(child.routine.scope.bindings["z"])
    outer = outputs[report["sources"][str(paths[1])]["replacement"]]
    inner = outputs[report["sources"][str(paths[0])]["replacement"]]
    assert "is_contiguous(c)" in outer
    assert "allocated(c)" in outer
    assert "lbound(c," in outer
    assert "c_loc(c)" not in outer
    assert re.search(r"call " + child.entry + r"\(\s*b,\s*a,\s*c,\s*n,\s*bad,", outer.replace("&", ""))
    assert f"real(8),intent({intent})::z(-7:)" in inner
    assert owner.builder.facts["captures"].keys() == {"argument::a", "argument::b"}


def test_guarded_first_use_closes_inside_child_without_prefix_replay(tmp_path):
    paths, outputs, report, owner = emit_mapped_child_window(tmp_path, **projected_sources(body="x(-3)=z(-7)"))
    child = owner.members["inner_fields::bridge"]
    assert not owner.boundaries
    point, = child.boundaries
    assert "untracked array" in point["reason"]
    assert "argument::z" in point["reason"]
    assert child.units
    inner = outputs[report["sources"][str(paths[0])]["replacement"]]
    assert inner.index("fort_scope_close(") < inner.index("x(-3)=z(-7)")
    assert inner.count("x(-3)=z(-7)") == 1
    assert inner.count("child_visits=child_visits+1") == 1
    assert inner.count("condition_visits=condition_visits+1") == 1
    assert "argument::c" not in owner.resources
    mapped, = owner.window_proof.public()["planned_closures"]["mapped_child_plans"]
    assert mapped["plan"]["planned_closures"]["active_path_may_return"]
    assert mapped["plan"]["planned_closures"]["closed_path_possible"]


def test_duplicate_actual_still_closes_at_original_call(tmp_path):
    _paths, _builder, proof, _originals = mapped_child_fixture(tmp_path, **projected_sources(actual="b"))
    assert not proof.public()["planned_closures"]["mapped_child_plans"]
    point, = proof.public()["planned_closures"]["close_points"]
    assert point["original_source_kind"] == "Call_Stmt"
    assert "distinct original array actual roots" in point["reason"]


def test_unmanaged_out_closes_before_its_original_entry_event(tmp_path):
    _paths, _builder, proof, _originals = mapped_child_fixture(tmp_path, **projected_sources(intent="out"))
    assert not proof.public()["planned_closures"]["mapped_child_plans"]
    point, = proof.public()["planned_closures"]["close_points"]
    assert point["original_source_kind"] == "Call_Stmt"
    assert "unmanaged OUT actual" in point["reason"]
    assert {state["resource"] for state in proof.public()["states"]} == {"argument::a", "argument::b"}


def test_empty_managed_projection_closes_at_original_call(tmp_path):
    sources = projected_sources()
    sources["parent_source"] = sources["parent_source"].replace(
        "call renamed_bridge(b,a,c,n,bad)", "call renamed_bridge(c,d,e,n,bad)").replace(
        "real(8),allocatable,intent(inout)::c(:)", "real(8),allocatable,intent(inout)::c(:),d(:),e(:)")
    _paths, _builder, proof, _originals = mapped_child_fixture(tmp_path, **sources)
    assert not proof.public()["planned_closures"]["mapped_child_plans"]
    point, = proof.public()["planned_closures"]["close_points"]
    assert "no independently managed resource" in point["reason"]


def test_nested_managed_out_still_kills_coverage_with_unused_actual(tmp_path):
    sources = projected_sources(body="call replace_one(x,z)")
    sources["child_source"] = sources["child_source"].removesuffix("end module\n") + """subroutine replace_one(p,q)
real(8),intent(out)::p(-3:)
real(8),intent(in)::q(-7:)
p(-2)=2.0_8
end subroutine
end module
"""
    _paths, _builder, proof, _originals = mapped_child_fixture(tmp_path, **sources)
    bridge, = proof.public()["planned_closures"]["mapped_child_plans"]
    nested, = bridge["plan"]["planned_closures"]["mapped_child_plans"]
    assert nested["coordinate_proof"]["entry_definition_changes"] == ["argument::x"]
    assert nested["managed_actual_projection"]["unmanaged_associations"] == {"argument::q": "argument::z"}
    state, = nested["plan"]["states"]
    assert state["resource"] == "argument::p"
    assert state["initialized"] == "none"
    assert state["coverage_on_return"] == "sections"
    assert any("incomplete definition" in point["reason"]
               for point in bridge["plan"]["planned_closures"]["close_points"])


def test_repeated_projected_child_reuses_one_original_body_variant(tmp_path):
    sources = projected_sources()
    sources["parent_source"] = sources["parent_source"].replace(
        "call renamed_bridge(b,a,c,n,bad)", "call renamed_bridge(b,a,c,n,bad)\ncall renamed_bridge(b,a,c,n,bad)")
    _paths, _outputs, report, owner = emit_mapped_child_window(tmp_path, **sources)
    child = owner.members["inner_fields::bridge"]
    assert len(child.call_sites) == 2
    assert len(owner.members) == 2
    mapped = owner.window_proof.public()["planned_closures"]["mapped_child_plans"]
    assert len(mapped) == 2
    assert mapped[0]["managed_actual_projection"] == mapped[1]["managed_actual_projection"]
    assert "argument::c" not in owner.resources
    assert len(report["scopes"][0]["gpu_leaves"]) == 3


def test_changed_unused_original_association_invalidates_projection_proof(tmp_path):
    sources = projected_sources()
    paths, builder, proof, originals = mapped_child_fixture(tmp_path, **sources)
    paths[0].write_text(paths[0].read_text().replace("z(-7:)", "z(-9:)"))
    assert proof.validate(builder.analysis, builder.entry.qualified, originals) is proof
    with pytest.raises(CompilationError, match="source|original|changed"):
        builder.finish()
    assert not builder.outputs
    assert not builder.edits


def test_changed_managed_subset_cannot_reuse_an_incompatible_companion(tmp_path):
    sources = projected_sources()
    sources["parent_source"] = sources["parent_source"].replace(
        "call renamed_bridge(b,a,c,n,bad)", "call renamed_bridge(b,a,c,n,bad)\ncall renamed_bridge(b,c,a,n,bad)")
    _paths, _outputs, _report, owner = emit_mapped_child_window(tmp_path, **sources)
    point, = owner.boundaries
    assert "one canonical mapping" in point["reason"]
    assert point["execution"] == "publish and close when reached; original continuation once"
    assert not point["reopen"]
    assert "argument::c" not in owner.resources


def test_native_leaf_also_defers_an_unused_forwarded_actual(tmp_path):
    sources = projected_sources()
    sources["child_source"] = sources["child_source"].replace(
        "condition_visits=condition_visits+1", "call touch_scalar(x,z)\ncondition_visits=condition_visits+1")
    sources["child_source"] = sources["child_source"].removesuffix("end module\n") + """subroutine touch_scalar(p,q)
real(8),intent(inout)::p(-3:)
real(8),intent(in)::q(-7:)
p(-3)=p(-3)+2.0_8
end subroutine
end module
"""
    _paths, _outputs, _report, owner = emit_mapped_child_window(tmp_path, **sources)
    bridge, = owner.window_proof.public()["planned_closures"]["mapped_child_plans"]
    nested, = bridge["plan"]["planned_closures"]["mapped_child_plans"]
    assert nested["procedure"] == "inner_fields::touch_scalar"
    assert nested["managed_actual_projection"]["unmanaged_associations"] == {"argument::q": "argument::z"}
    assert "argument::q" not in owner.members["inner_fields::touch_scalar"].resources
    assert "argument::c" not in owner.resources
