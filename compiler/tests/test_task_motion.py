"""Graph scheduling restores fusion while retaining numerical/control semantics."""

from pathlib import Path

import pytest

from compiler.driver.options import CompilerOptions
from compiler.driver.pipeline import prepare_function
from compiler.frontend import lower_file
from compiler.tests.test_language import run_reference_and_cpp

ROOT = Path(__file__).resolve().parents[2]
GRAPH = CompilerOptions(gpu_policy="sections", memory_model="scoped", scope_execution="graph")


@pytest.mark.parametrize(("case", "filename"), [("CDU", "cdu.f90"), ("CDV", "cvd.f90"), ("CDW", "cdw.f90")])
def test_graph_placement_retains_checked_stencil_fusion(case, filename):
    function = lower_file(ROOT / "benchmarks" / "cases" / case / "Fortran" / filename, case)
    _, old = prepare_function(function, options=CompilerOptions(gpu_policy="sections"))
    _, graph = prepare_function(function, options=GRAPH)
    assert len(old.regions) == 4
    assert len(graph.regions) == 1
    assert any("task graph:" in report for report in graph.reports)


SOURCE = """module graph_motion_case
contains
subroutine entry(a,b,out,n,enabled)
integer,intent(inout)::a(:),b(:),out(:)
integer,intent(in)::n
logical,intent(in)::enabled
integer::i,j,k,factor
do i=1,n
  b(i)=a(i)*2
enddo
factor=7
if(enabled)then
  do j=2,n
    out(j)=b(j-1)+factor
  enddo
else
  do j=1,n
    out(j)=b(j)+factor
  enddo
endif
do k=1,n
  a(k)=b(k)+out(k)
enddo
end subroutine
end module
"""

DRIVER = """program main
use graph_motion_case
implicit none
integer::a(9),b(9),out(9),i
a=[(i*11,i=1,9)]
b=-3
out=-5
call entry(a,b,out,9,.true.)
print *,a,b,out
call entry(a,b,out,0,.false.)
print *,a,b,out
call entry(a,b,out,1,.true.)
print *,a,b,out
call entry(a,b,out,9,.false.)
print *,a,b,out
end program
"""


@pytest.mark.native
@pytest.mark.parametrize("openmp", [False, True])
def test_graph_motion_preserves_branches_empty_domains_and_neighbor_versions(tmp_path, openmp):
    source = tmp_path / "original.f90"
    source.write_text(SOURCE)
    function, plan = prepare_function(lower_file(source, "entry"), options=GRAPH)
    run_reference_and_cpp(tmp_path, function, plan, DRIVER, openmp=openmp)


def test_graph_motion_preserves_cross_pass_global_dependence(tmp_path):
    source = tmp_path / "original.f90"
    source.write_text("""module snapshot_case
contains
subroutine entry(a,b,n)
integer,intent(inout)::a(:),b(:)
integer,intent(in)::n
integer::i,j
do i=1,n
b(i)=a(i)
enddo
do j=2,n
a(j)=b(j-1)+b(j)
enddo
end subroutine
end module
""")
    _, plan = prepare_function(lower_file(source, "entry"), options=GRAPH)
    assert len(plan.regions) == 2
    assert any("skipped" in report for report in plan.reports)


RECURRENCE_SOURCE = """module regroup_native
contains
subroutine entry(a,b,out,x,n)
integer,intent(in)::a(:)
integer,intent(inout)::b(:),out(:),x(:)
integer,intent(in)::n
integer::i,j,k
do i=1,n
b(i)=2*a(i)
enddo
do k=2,n
x(k)=x(k-1)+x(k)
enddo
do j=1,n
out(j)=b(j)+3
enddo
end subroutine
end module
"""

RECURRENCE_DRIVER = """program main
use regroup_native
implicit none
integer::a(9),b(9),out(9),x(9),i
a=[(11*i,i=1,9)]
b=-7
out=-5
x=[(i,i=1,9)]
call entry(a,b,out,x,9)
print *,a,b,out,x
call entry(a,b,out,x,0)
print *,a,b,out,x
call entry(a,b,out,x,1)
print *,a,b,out,x
end program
"""


@pytest.mark.native
def test_cpu_only_integer_recurrence_moves_outside_device_group(tmp_path):
    from dataclasses import replace

    from compiler.ir import SequentialRegion

    source = tmp_path / "original.f90"
    source.write_text(RECURRENCE_SOURCE)
    function = lower_file(source, "entry")
    _, baseline = prepare_function(function, options=CompilerOptions(gpu_policy="sections", fallback="host"))
    assert len(baseline.regions) == 2
    function, graph = prepare_function(function, options=replace(GRAPH, fallback="host"))
    assert isinstance(graph.steps[0], SequentialRegion)
    assert len(graph.regions) == 1
    run_reference_and_cpp(tmp_path, function, graph, RECURRENCE_DRIVER, openmp=True)
