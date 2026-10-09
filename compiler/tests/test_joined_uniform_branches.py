"""Original teams may choose shared, unchanged branches before worksharing."""

from types import SimpleNamespace

import pytest

from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError
from compiler.scopes.segments import grouped_nodes, joined_group_completion


def completion(tmp_path, condition="flag", clause="private(i)", update=""):
    path = tmp_path / "uniform.f90"
    path.write_text(f"""module teams
contains
subroutine apply(a,flag,n)
real(8)::a(:)
logical::flag
integer::i,n
continue
!$omp parallel {clause}
if({condition}) then
!$omp do
do i=1,n
a(i)=2*a(i)
{update}
enddo
!$omp end do nowait
else
!$omp do
do i=1,n
a(i)=3*a(i)
enddo
!$omp end do
endif
!$omp end parallel
end subroutine
end module
""")
    analysis = SourceEffects([path])
    entry = analysis.routines["teams::apply"]
    originals = tuple(node for node in entry.execution.content if type(node).__name__ != "Continue_Stmt")
    return joined_group_completion(SimpleNamespace(analysis=analysis, entry=entry), originals)


def test_uniform_branch_keeps_original_join_and_thread_budget(tmp_path):
    result = completion(tmp_path)
    assert result["available"] and result["has_openmp_in_closure"]
    assert result["requires_serial_caller"]
    assert "PARALLEL" in result["reason"] and "join" in result["reason"]


@pytest.mark.parametrize("condition,clause,update,reason", [
    ("flag", "private(i,flag)", "", "PRIVATE requires"),
    ("flag", "private(i)", "flag=.false.", "private or changes"),
    ("omp_get_thread_num()==0", "private(i)", "", "unproved uniform"),
    ("a(1)>0", "private(i)", "", "private or changes"),
])
def test_nonuniform_or_mutable_conditions_remain_boundaries(tmp_path, condition, clause, update, reason):
    with pytest.raises(CompilationError, match=reason):
        completion(tmp_path, condition, clause, update)
