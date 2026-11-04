"""Graph execution retains a complete native CALL without an interval prepass."""
import json
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder

SOURCE = """module native_graph
implicit none
type record
real(8)::flux,aux
end type
type(record),target::points(-1:2)
integer::visits=0
real(8)::a(-1:6,-1:6,-1:6),b(-1:6,-1:6,-1:6),c(-1:6,-1:6,-1:6)
contains
subroutine advance()
integer::i,j,k
do k=-1,6
do j=-1,6
do i=-1,6
b(i,j,k)=2*a(i,j,k)
enddo
enddo
enddo
call update(points,a)
do k=-1,6
do j=-1,6
do i=-1,6
c(i,j,k)=b(i,j,k)+a(i,j,k)
enddo
enddo
enddo
end subroutine
subroutine update(items,input)
type(record),target,intent(inout)::items(:)
real(8),intent(in)::input(-1:,-1:,-1:)
type(record),pointer::p
integer::i
!$omp parallel do private(i,p) schedule(guided,5)
do i=1,size(items)
p=>items(i)
p%flux=input(1,1,1)+i
p%aux=2*p%flux
enddo
!$omp end parallel do
visits=visits+1
end subroutine
end module
"""


def generate(tmp_path, text=SOURCE, *, initialized="whole", capture_updates=None):
    path = tmp_path / "source.f90"
    path.write_text(text)
    fact = {"storage": "stable", "initialized": initialized, "escapes": False, "allocation_changes": False}
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"native_graph::" + name: dict(fact) for name in ("a", "b", "c")}}
    facts["captures"].update(capture_updates or {})
    builder = ScopeBuilder([path], "native_graph::advance", facts=facts, options=CompilerOptions(),
                           config=OffloadConfig(policy="sections", scope_execution="graph"))
    outputs, report = builder.run()
    (tmp_path / "report.json").write_text(json.dumps(report, indent=2))
    return path, builder, outputs, report


def test_complete_original_call_keeps_graph_domain_and_host_reservations(tmp_path):
    path, builder, outputs, report = generate(tmp_path)
    owner, = report["scopes"]
    assert not owner["boundaries"], owner["boundaries"]
    operations = [op for unit in owner["planning_segments"] for op in unit["operations"]["native_operations"]
                  if op["kind"] == "complete original native CALL"]
    operation, = operations
    assert operation["resource_effects"] == {"native_graph::a": ["read"]}
    assert set(operation["host_metadata"]) == {"native_graph::points", "native_graph::visits"}
    assert operation["preserves_original_source"]
    assert operation["completion"]["available"]
    assert not operation["sections"]["available"]
    assert not any(demand._plan is None for demand in builder._resource_demands.values())
    changed = outputs[report["sources"][str(path)]["replacement"]]
    assert changed.count("call update(points,a)") == 1
    assert "fort_scope_access_begin_v1" in changed
    assert "storage_size(visits" in changed.lower()
    assert "shape(visits" not in changed.lower()
    assert "is_contiguous(visits" not in changed.lower()


def module_helper_source():
    # Ordinary numeric formals allow companion discovery. Its record-valued
    # joined loop still has no numerical implementation, so the discovered
    # native-only subtree must not impose artificial publication boundaries.
    return (SOURCE.replace("call update(points,a)", "call update(a)")
            .replace("subroutine update(items,input)", "subroutine update(input)")
            .replace("type(record),target,intent(inout)::items(:)\n", "")
            .replace("do i=1,size(items)", "do i=-1,2")
            .replace("p=>items(i)", "p=>points(i)")
            .replace("p%flux=input(1,1,1)+i", "p%flux=input(1,1,1)+i+2"))


def test_discovered_native_only_companion_uses_complete_caller_transaction(tmp_path, monkeypatch):
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans

    from compiler.scopes import module_owner

    original = module_owner.ModuleOwner.__init__
    discoveries = []

    def construct(self, parent, routine, *args, **kwargs):
        result = original(self, parent, routine, *args, **kwargs)
        if routine.qualified == "native_graph::update":
            discoveries.append(self)
            assert self.boundaries
            assert not any(unit.leaves for unit in self.units)
        return result

    monkeypatch.setattr(module_owner.ModuleOwner, "__init__", construct)
    path, builder, outputs, report = generate(tmp_path, module_helper_source())
    owner, = report["scopes"]
    assert discoveries
    assert not owner["boundaries"]
    assert "native_graph::update" not in {row["procedure"] for row in owner["checked_execution_plan"]["procedures"]}
    operation, = [op for unit in owner["planning_segments"] for op in unit["operations"]["native_operations"]
                   if op["kind"] == "complete original native CALL"]
    assert operation["resource_effects"] == {"native_graph::a": ["read"]}
    assert set(operation["host_metadata"]) == {"native_graph::points", "native_graph::visits"}
    assert outputs[report["sources"][str(path)]["replacement"]].count("call update(a)") == 1
    validate_checked_execution_plans(report, capture_facts=builder.facts)
    assert builder.finish()


def test_discovered_companion_with_gpu_descendant_keeps_checked_child(tmp_path):
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans

    worker = """subroutine numerical(input)
real(8),intent(in)::input(-1:,-1:,-1:)
integer::i,j,k
do k=-1,6
do j=-1,6
do i=-1,6
b(i,j,k)=3*input(i,j,k)
enddo
enddo
enddo
end subroutine
"""
    text = module_helper_source().replace("!$omp parallel do", "call numerical(input)\n!$omp parallel do", 1)
    text = text.replace("end module", worker + "end module")
    _path, builder, _outputs, report = generate(tmp_path, text)
    owner, = report["scopes"]
    members = {row["procedure"]: row for row in owner["checked_execution_plan"]["procedures"]}
    assert members["native_graph::update"]["boundaries"]
    assert members["native_graph::numerical"]["units"]
    assert not any(op["kind"] == "complete original native CALL" for unit in owner["planning_segments"]
                   for op in unit["operations"]["native_operations"])
    validate_checked_execution_plans(report, capture_facts=builder.facts)
    assert builder.finish()


def test_discovered_companion_reusing_an_existing_child_keeps_shared_calls(tmp_path):
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans

    text = module_helper_source().replace("call update(a)", "call shared()\ncall update(a)")
    text = text.replace("!$omp parallel do", "call shared()\n!$omp parallel do", 1)
    text = text.replace("end module", "subroutine shared()\nvisits=visits+1\nend subroutine\nend module")
    _path, builder, _outputs, report = generate(tmp_path, text)
    owner, = report["scopes"]
    members = {row["procedure"]: row for row in owner["checked_execution_plan"]["procedures"]}
    assert members["native_graph::update"]["boundaries"]
    assert {call["caller"] for call in members["native_graph::shared"]["calls"]} == {
        "native_graph::advance", "native_graph::update"}
    validate_checked_execution_plans(report, capture_facts=builder.facts)
    assert builder.finish()


@pytest.mark.parametrize("change", ["unknown", "pointer_field", "reallocate"])
def test_unproved_call_retains_a_boundary_and_original_execution(tmp_path, change):
    text = SOURCE
    initialized = "whole"
    if change == "unknown":
        text = text.replace("p%aux=2*p%flux", "call unknown(p%aux)")
    elif change == "pointer_field":
        text = text.replace("real(8)::flux,aux", "real(8)::flux\nreal(8),pointer::aux")
    elif change == "reallocate":
        text = text.replace("type(record),target::points(-1:2)", "type(record),allocatable,target::points(:)")
        text = text.replace("visits=visits+1", "visits=visits+1\ndeallocate(points)")
    path, _, outputs, report = generate(tmp_path, text, initialized=initialized)
    assert any(scope["boundaries"] for scope in report["scopes"]) or report["boundaries"]
    assert not any(op["kind"] == "complete original native CALL" for scope in report["scopes"]
                   for unit in scope["planning_segments"] for op in unit["operations"]["native_operations"])
    assert outputs[report["sources"][str(path)]["replacement"]].count("call update(points,a)") == 1


def test_call_reservations_remain_inside_original_condition(tmp_path):
    from fparser.common.readfortran import FortranStringReader
    from fparser.two.parser import ParserFactory
    from fparser.two.utils import walk

    text = SOURCE.replace("subroutine advance()", "subroutine advance(run)\nlogical,intent(in)::run")
    text = text.replace("call update(points,a)", "if(run)then\ncall update(points,a)\nendif")
    path, _, outputs, report = generate(tmp_path, text)
    owner, = report["scopes"]
    assert not owner["boundaries"]
    changed = outputs[report["sources"][str(path)]["replacement"]]
    parsed = ParserFactory().create(std="f2008")(FortranStringReader(changed))
    guarded = [node for node in walk(parsed) if type(node).__name__ == "If_Construct"
               and str(node.content[0]).lower() == "if (run) then"]
    guard, = guarded
    original_calls = [node for node in walk(guard) if type(node).__name__ == "Call_Stmt"
                      and str(node.items[0]).lower() == "update"]
    assert len(original_calls) == 1
    reservations = [node for node in walk(parsed) if type(node).__name__ == "Assignment_Stmt"
                    and "c_loc(points)" in str(node).lower()]
    assert reservations
    inside = {id(node) for node in walk(guard)}
    assert all(id(node) in inside for node in reservations)


@pytest.mark.parametrize("produced_whole", [False, True])
def test_call_whole_definition_requirement_is_checked_when_reached(tmp_path, produced_whole):
    text = SOURCE.replace("call update(points,a)", "call update(points,b)")
    if not produced_whole:
        for axis in ("i", "j", "k"):
            text = text.replace(f"do {axis}=-1,6", f"do {axis}=0,5", 1)
    fact = {"storage": "stable", "initialized": "none", "escapes": False, "allocation_changes": False}
    path, builder, outputs, report = generate(tmp_path, text, capture_updates={"native_graph::b": fact})
    owner, = report["scopes"]
    assert not owner["boundaries"], owner["boundaries"]
    unit, = [unit for unit in owner["planning_segments"] if any(
        op["kind"] == "complete original native CALL" for op in unit["operations"]["native_operations"])]
    operation, = unit["operations"]["native_operations"]
    assert operation["resource_effects"] == {"native_graph::b": ["read"]}
    plan_unit, = [unit for proc in owner["checked_execution_plan"]["procedures"] for unit in proc["units"]
                  if any(op["kind"] == "complete original native CALL" for op in unit["operations"]["native_operations"])]
    assert plan_unit["ordered_definitions"]["required_whole"] == ["native_graph::b"]
    assert not plan_unit["ordered_definitions"]["possible_definition_changes"]
    assert builder.facts["captures"]["native_graph::b"]["initialized"] == "none"
    changed = outputs[report["sources"][str(path)]["replacement"]]
    call = changed.index("call update(points,b)")
    preflight = changed.rfind("fort_scope_plan_validate(fort_context)", 0, call)
    acquisition = changed.rfind("fort_scope_access_begin_v1", 0, call)
    assert 0 <= preflight < acquisition < call
    assert "FORT_SCOPE_UNINITIALIZED" in changed[:call]
    assert "fort_native_ready = .true." in changed[preflight:call]


def test_unknown_entry_definitions_do_not_create_native_definition_credit(tmp_path):
    _, builder, _, report = generate(tmp_path, initialized="none")
    owner, = report["scopes"]
    assert not owner["boundaries"], owner["boundaries"]
    assert all(fact["initialized"] == "none" for fact in builder.facts["captures"].values())
    unit, = [unit for proc in owner["checked_execution_plan"]["procedures"] for unit in proc["units"]
             if any(op["kind"] == "complete original native CALL" for op in unit["operations"]["native_operations"])]
    assert unit["ordered_definitions"]["required_whole"] == ["native_graph::a"]
    assert not unit["ordered_definitions"]["possible_definition_changes"]


def mesh_source():
    return SOURCE.replace("integer::visits=0", "integer::visits=0\nreal(8),allocatable::mesh(:)").replace(
        "call update(points,a)", "call update(points,a,mesh)").replace(
        "subroutine update(items,input)", "subroutine update(items,input,coords)\nreal(8),intent(in)::coords(-1:)").replace(
        "p%flux=input(1,1,1)+i", "p%flux=input(1,1,1)+i+coords(i-2)")


def test_complete_call_reserves_readonly_mesh_without_gpu_capture_fact(tmp_path):
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans

    path, builder, outputs, report = generate(tmp_path, mesh_source())
    owner, = report["scopes"]
    assert not owner["boundaries"], owner["boundaries"]
    operation, = [op for unit in owner["planning_segments"] for op in unit["operations"]["native_operations"]
                   if op["kind"] == "complete original native CALL"]
    assert operation["managed_resources"] == ["native_graph::a"]
    access, = operation["native_opaque_accesses"]["accesses"]
    assert access["resource"] == "native_graph::mesh"
    assert access["mode"] == "R"
    assert access["permit_kind"] == "original_readonly"
    assert access["permit"]["device_capture"] is False
    assert "native_graph::mesh" not in builder.facts["captures"]
    assert not any(demand._plan is None for demand in builder._resource_demands.values())
    assert outputs[report["sources"][str(path)]["replacement"]].count("call update(points,a,mesh)") == 1
    validate_checked_execution_plans(report, capture_facts=builder.facts)


@pytest.mark.parametrize("mutation", ["write", "out", "pointer", "allocation"])
def test_readonly_mesh_contract_cannot_hide_writes_or_descriptor_changes(tmp_path, mutation):
    text = mesh_source()
    if mutation == "write":
        text = text.replace("intent(in)::coords", "intent(inout)::coords").replace(
            "visits=visits+1", "coords(-1)=coords(-1)+1\nvisits=visits+1")
    elif mutation == "out":
        text = text.replace("intent(in)::coords", "intent(out)::coords")
    elif mutation == "pointer":
        text = text.replace("allocatable::mesh", "pointer::mesh")
    else:
        text = text.replace("visits=visits+1", "deallocate(mesh)\nvisits=visits+1")
    path, _, outputs, report = generate(tmp_path, text)
    assert any(scope["boundaries"] for scope in report["scopes"]) or report["boundaries"]
    assert not any(op["kind"] == "complete original native CALL" for scope in report["scopes"]
                   for unit in scope["planning_segments"] for op in unit["operations"]["native_operations"])
    assert outputs[report["sources"][str(path)]["replacement"]].count("call update(points,a,mesh)") == 1
