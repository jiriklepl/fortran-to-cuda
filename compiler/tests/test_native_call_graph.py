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


def generate(tmp_path, text=SOURCE, *, initialized="whole"):
    path = tmp_path / "source.f90"
    path.write_text(text)
    fact = {"storage": "stable", "initialized": initialized, "escapes": False, "allocation_changes": False}
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"native_graph::" + name: dict(fact) for name in ("a", "b", "c")}}
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


@pytest.mark.parametrize("change", ["unknown", "pointer_field", "reallocate", "partial_definitions"])
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
    else:
        initialized = "none"
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
