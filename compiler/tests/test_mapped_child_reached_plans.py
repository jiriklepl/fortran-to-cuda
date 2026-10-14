"""Registered child plans share one original owner, including native closes."""

from copy import deepcopy
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.frontend.reached_windows import plan_reached_window
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.reached_owner import ReachedWindowOwner
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_source_scopes import FACT

ENTRY = "outer_fields::advance"
ROOTS = ("argument::a", "argument::b")
CHILD_SOURCE = """module inner_fields
implicit none
integer::child_visits=0,bad_visits=0,good_visits=0,condition_visits=0
real(8),allocatable::workspace(:)
contains
subroutine native_workspace(x)
real(8),intent(inout)::x(-3:)
real(8),allocatable::temporary(:)
allocate(temporary(2))
temporary=1.0_8
call move_alloc(temporary,workspace)
x(-3)=x(-3)+10.0_8
bad_visits=bad_visits+1
end subroutine
subroutine bridge(x,y,n,bad)
real(8),intent(inout)::x(-3:),y(-3:)
integer,intent(in)::n
logical,intent(in)::bad
integer::i
child_visits=child_visits+1
do i=-3,n-4
x(i)=x(i)+real(i,8)
enddo
condition_visits=condition_visits+1
if(bad) then
call native_workspace(x)
else
x(-3)=x(-3)+3.0_8
good_visits=good_visits+1
endif
do i=-3,n-4
y(i)=2.0_8*x(i)+real(i,8)
enddo
end subroutine
end module
"""
PARENT_SOURCE = """module outer_fields
use inner_fields,only:renamed_bridge=>bridge
implicit none
integer::prefix_visits=0,tail_visits=0
contains
subroutine advance(a,b,n,iterations,bad)
real(8),allocatable,intent(inout)::a(:),b(:)
integer,intent(in)::n,iterations
logical,intent(in)::bad
integer::i,stage
real(8),allocatable::temporary(:)
prefix_visits=prefix_visits+1
do stage=1,iterations
do i=lbound(a,1),ubound(a,1)
b(i)=a(i)+real(i,8)
enddo
call renamed_bridge(b,a,n,bad)
do i=lbound(a,1),ubound(a,1)
b(i)=a(i)+2.0_8
enddo
call move_alloc(a,temporary)
call move_alloc(temporary,a)
tail_visits=tail_visits+1
enddo
end subroutine
end module
"""


def mapped_child_fixture(directory, *, precision=8, device_budget=None,
                         parent_source=PARENT_SOURCE, child_source=CHILD_SOURCE, roots=ROOTS):
    directory.mkdir(parents=True, exist_ok=True)
    paths = (directory / "inner.f90", directory / "outer.f90")
    for path, source in zip(paths, (child_source, parent_source), strict=True):
        path.write_text(source.replace("real(8)", f"real({precision})")
                        .replace("_8", f"_{precision}").replace(",8)", f",{precision})"))
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest() for path in paths},
             "captures": {root: deepcopy(FACT) for root in roots}}
    if device_budget is not None:
        facts["device_budget_bytes"] = device_budget
    builder = ScopeBuilder(paths, ENTRY, facts=facts, options=CompilerOptions(),
                           config=OffloadConfig(policy="sections", scope_execution="reached"))
    outer = next(node for node in builder.entry.execution.content
                 if type(node).__name__ == "Block_Nonlabel_Do_Construct")
    originals = tuple(outer.content[1:4])
    proof = plan_reached_window(builder.analysis, ENTRY, originals, roots, facts)
    return paths, builder, proof, originals


def emit_mapped_child_window(directory, *, precision=8, device_budget=None, **sources):
    paths, builder, proof, originals = mapped_child_fixture(directory, precision=precision,
                                                          device_budget=device_budget, **sources)
    owner = ReachedWindowOwner(builder, proof, originals)
    assert owner.run()
    outputs, report = owner.builder.finish()
    return paths, outputs, report, owner


def test_mapped_child_retains_active_return_and_absorbing_close(tmp_path):
    _paths, builder, proof, originals = mapped_child_fixture(tmp_path)
    mapped, = proof.public()["planned_closures"]["mapped_child_plans"]
    child = mapped["plan"]
    assert mapped["procedure"] == "inner_fields::bridge"
    assert child["planned_closures"]["closed_path_possible"]
    assert len(child["planned_closures"]["close_points"]) == 1
    assert all(state["coverage_on_return"] == "unknown_after_close" for state in child["states"])
    assert all(state["active_path_coverage_on_return"] == "whole" for state in child["states"])
    assert proof.public()["mapped_work"]["calls"] >= 1
    assert proof.public()["mapped_work"]["coordinates"] > 0
    assert proof.validate(builder.analysis, ENTRY, originals) is proof


def test_emitted_child_shares_control_and_original_caller_descriptors(tmp_path):
    paths, outputs, report, owner = emit_mapped_child_window(tmp_path)
    child = owner.members["inner_fields::bridge"]
    outer = outputs[report["sources"][str(paths[1])]["replacement"]]
    inner = outputs[report["sources"][str(paths[0])]["replacement"]]
    assert len(owner.units) == 2
    assert len(child.units) >= 2
    assert len(report["scopes"][0]["gpu_leaves"]) == 4
    assert not owner.boundaries
    assert len(child.boundaries) == 1
    assert outer.index("fort_actual_lowers_") < outer.index("call " + child.entry)
    assert "lbound(b, kind=c_int64_t)" in outer
    assert "lbound(a, kind=c_int64_t)" in outer
    assert f"{child.enabled} = {owner.enabled}" in outer
    assert inner.index("fort_scope_close(") < inner.index("call native_workspace(x)")
    assert inner.count("call native_workspace(x)") == 1
    assert outer.count("prefix_visits=prefix_visits+1") == 1
    assert outer.count("call move_alloc(a,temporary)") == 1
    assert not report["scopes"][0]["reopen_after_boundary"]


def test_child_record_substitution_and_foreign_call_are_rejected(tmp_path):
    _paths, builder, proof, originals = mapped_child_fixture(tmp_path)
    owner = ReachedWindowOwner(builder, proof, originals)
    owner.scan(owner.builder.inline.prepare(originals))
    child = owner.members["inner_fields::bridge"]
    child.builder._mapped_reached_path = ("foreign",)
    with pytest.raises(CompilationError, match="registered reached authority"):
        child.builder.reached_plan_record()


def test_allocatable_child_actual_replacement_closes_at_original_caller(tmp_path):
    source = CHILD_SOURCE.replace("real(8),intent(inout)::x(-3:),y(-3:)",
                                  "real(8),allocatable,intent(inout)::x(:),y(:)")
    _paths, _builder, proof, _originals = mapped_child_fixture(tmp_path, child_source=source)
    assert not proof.public()["planned_closures"]["mapped_child_plans"]
    point, = proof.public()["planned_closures"]["close_points"]
    assert point["original_source_kind"] == "Call_Stmt"
    assert "ordinary array associations" in point["reason"]


PARTIAL_CHILD = """module inner_fields
contains
subroutine bridge(x,y,n,bad)
real(8),intent(in)::x(-3:)
real(8),intent(out)::y(-3:)
integer,intent(in)::n
logical,intent(in)::bad
y(-2:1)=x(-2:1)+4.0_8
end subroutine
end module
"""


def test_child_out_partial_sections_feed_exact_parent_consumer(tmp_path):
    source = PARENT_SOURCE.replace("do i=lbound(a,1),ubound(a,1)\nb(i)=a(i)+2.0_8\nenddo",
        "b(lbound(b,1)+1:lbound(b,1)+4)=a(lbound(a,1)+1:lbound(a,1)+4)+2.0_8")
    source = source.replace("call move_alloc(a,temporary)",
        "a(lbound(a,1))=-123.0_8\na(lbound(a,1)+5:ubound(a,1))=-124.0_8\ncall move_alloc(a,temporary)")
    paths, outputs, report, owner = emit_mapped_child_window(tmp_path, parent_source=source,
                                                          child_source=PARTIAL_CHILD)
    record = owner.window_proof.public()
    assert not record["planned_closures"]["closed_path_possible"]
    assert len(report["scopes"][0]["gpu_leaves"]) == 3
    state = next(item for item in record["states"] if item["resource"] == "argument::a")
    assert state["coverage_on_return"] == "sections"
    assert len(state["defined_sections_on_return"]) == 1
    child = owner.members["inner_fields::bridge"]
    text = outputs[report["sources"][str(paths[0])]["replacement"]]
    assert "fort_scope_forget_definition" in text
    assert child.units
    assert not child.boundaries


def test_child_out_missing_halo_closes_before_original_parent_read(tmp_path):
    _paths, _builder, proof, _originals = mapped_child_fixture(tmp_path, child_source=PARTIAL_CHILD)
    point, = proof.public()["planned_closures"]["close_points"]
    assert point["original_source_kind"] == "Block_Nonlabel_Do_Construct"
    assert "incomplete definition" in point["reason"]


def test_mapped_original_joined_group_retains_whole_completion(tmp_path):
    source = CHILD_SOURCE.replace("do i=-3,n-4\nx(i)=x(i)+real(i,8)\nenddo",
        "!$omp parallel private(i)\n!$omp do\ndo i=-3,n-4\nx(i)=x(i)+real(i,8)\nenddo\n"
        "!$omp end do\n!$omp end parallel")
    _paths, _outputs, report, owner = emit_mapped_child_window(tmp_path, child_source=source)
    assert len(report["scopes"][0]["gpu_leaves"]) == 4
    child = owner.members["inner_fields::bridge"]
    assert len(child.boundaries) == 1
    assert all("joined" not in boundary["reason"] for boundary in child.boundaries)


def test_repeated_identical_child_reuses_original_body_variant(tmp_path):
    source = PARENT_SOURCE.replace("call renamed_bridge(b,a,n,bad)",
                                  "call renamed_bridge(b,a,n,bad)\ncall renamed_bridge(b,a,n,bad)")
    paths, outputs, _report, owner = emit_mapped_child_window(tmp_path, parent_source=source)
    child = owner.members["inner_fields::bridge"]
    assert len(child.call_sites) == 2
    assert len(owner.members) == 2
    assert len(owner.window_proof.public()["planned_closures"]["mapped_child_plans"]) == 2
    assert owner.execution_plan.numerical_invocations == 5  # one parent kernel and two two-kernel child calls
    assert sum(text.count("entry " + child.entry) for name, text in outputs.items()
               if name.endswith(".f90")) == 1
    assert paths[0].read_text() == CHILD_SOURCE


def test_original_child_source_mutation_invalidates_root_plan(tmp_path):
    paths, builder, proof, originals = mapped_child_fixture(tmp_path)
    paths[0].write_text(CHILD_SOURCE.replace("2.0_8*x(i)", "3.0_8*x(i)"))
    assert proof.validate(builder.analysis, ENTRY, originals) is proof
    with pytest.raises(CompilationError, match="source|original|changed"):
        builder.finish()
    assert not builder.outputs
    assert not builder.edits


def test_nested_unused_out_kills_caller_definition_before_read(tmp_path):
    child_source = """module inner_fields
contains
subroutine leaf(x,y,z)
real(8),intent(in)::x(-2:)
real(8),intent(out)::y(-2:),z(-2:)
y(-1:2)=x(-1:2)+4.0_8
end subroutine
subroutine bridge(x,y,z,n,bad)
real(8),intent(in)::x(-3:)
real(8),intent(out)::y(-3:),z(-3:)
integer,intent(in)::n
logical,intent(in)::bad
call leaf(x,y,z)
end subroutine
end module
"""
    parent = (PARENT_SOURCE.replace("advance(a,b,n,iterations,bad)", "advance(a,b,c,n,iterations,bad)")
              .replace("a(:),b(:)", "a(:),b(:),c(:)")
              .replace("call renamed_bridge(b,a,n,bad)", "call renamed_bridge(b,a,c,n,bad)")
              .replace("b(i)=a(i)+2.0_8", "b(i)=c(i)+2.0_8"))
    _paths, builder, proof, originals = mapped_child_fixture(tmp_path, parent_source=parent,
        child_source=child_source, roots=(*ROOTS, "argument::c"))
    child, = proof.public()["planned_closures"]["mapped_child_plans"]
    nested, = child["plan"]["planned_closures"]["mapped_child_plans"]
    assert nested["coordinate_proof"]["entry_definition_changes"] == ["argument::y", "argument::z"]
    state = next(item for item in child["plan"]["states"] if item["resource"] == "argument::z")
    assert state["coverage_on_return"] == "none"
    point, = proof.public()["planned_closures"]["close_points"]
    assert "argument::c" in point["reason"]
    owner = ReachedWindowOwner(builder, proof, originals)
    assert owner.run()
    assert "inner_fields::leaf" in owner.members


def test_imported_compiler_private_actual_alias_declines_before_registration(tmp_path):
    # Declared fort_* names are already rejected by LexicalOwner; imported
    # aliases need the same protection inside the generated registration BLOCK.
    source = PARENT_SOURCE.replace("use inner_fields,only:renamed_bridge=>bridge",
                                  "use inner_fields,only:renamed_bridge=>bridge,fort_actual_status=>data")
    source = source.replace("call renamed_bridge(b,a,n,bad)", "call renamed_bridge(fort_actual_status,a,n,bad)")
    child_source = CHILD_SOURCE.replace("real(8),allocatable::workspace(:)",
                                       "real(8),allocatable::workspace(:),data(:)")
    paths, builder, proof, originals = mapped_child_fixture(tmp_path, parent_source=source, child_source=child_source,
        roots=(*ROOTS, "inner_fields::data"))
    owner = ReachedWindowOwner(builder, proof, originals)
    assert owner.run()
    outputs, report = owner.builder.finish()
    point, = owner.boundaries
    assert "compiler-private registration" in point["reason"]
    text = outputs[report["sources"][str(paths[1])]["replacement"]]
    assert "call renamed_bridge(fort_actual_status,a,n,bad)" in text
    assert "integer(c_int) :: fort_actual_status" not in text
