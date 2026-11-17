"""Only proved full-team callers acquire additive coherent source workers."""


from compiler.driver.options import CompilerOptions
from compiler.offload.config import OffloadConfig

SOURCE = """module operators
implicit none
contains
subroutine produce(a,b,n)
real(8),intent(in)::a(:)
real(8),intent(inout)::b(:)
integer,intent(in)::n
integer::i
!$omp do
do i=1,n
b(i)=2*a(i)+real(i,8)
enddo
!$omp end do
end subroutine
subroutine consume(a,b,c,n)
real(8),intent(in)::a(:),b(:)
real(8),intent(inout)::c(:)
integer,intent(in)::n
integer::i
!$omp do
do i=1,n
c(i)=a(i)+b(i)
enddo
!$omp end do
end subroutine
subroutine step(a,b,c,n)
real(8),intent(in)::a(:)
real(8),intent(inout)::b(:),c(:)
integer,intent(in)::n
call produce(a,b,n)
call consume(a,b,c,n)
end subroutine
end module
module callers
use operators,only:renamed_step=>step
implicit none
contains
subroutine qualified(a,b,c,n)
real(8),intent(in)::a(:)
real(8),intent(inout)::b(:),c(:)
integer,intent(in)::n
!$omp parallel default(none) shared(a,b,c,n) num_threads(4)
call renamed_step(a,b,c,n)
!$omp end parallel
end subroutine
subroutine unknown(a,b,c,n)
real(8),intent(in)::a(:)
real(8),intent(inout)::b(:),c(:)
integer,intent(in)::n
call renamed_step(a,b,c,n)
end subroutine
end module
"""


def test_collective_entry_preparation_uses_effect_records_without_native_call_markers(tmp_path):
    from compiler.driver.pipeline import prepare_function
    from compiler.emission.cuda.scoped import generate_scoped
    from compiler.frontend import lower_file

    source = SOURCE.replace("integer::i\n!$omp do\ndo i=1,n\nb(i)",
                            "integer::i,limit\nlimit=n\n!$omp do\ndo i=1,limit\nb(i)")
    path = tmp_path / "preparation.f90"
    path.write_text(source)
    function, plan = prepare_function(lower_file(path, "operators::produce"),
                                      options=CompilerOptions(gpu_policy="sections"))
    emitted = generate_scoped(function, plan, OffloadConfig("sections", None, 4, True), "common_functions.cuh")
    assert ".record(FORT_SCOPE_PLAN_NATIVE," in emitted.cuda
    assert "fort_scope_plan_team_native_call_v1(" not in emitted.cuda
