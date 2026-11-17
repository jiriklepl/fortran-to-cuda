"""One conditional owner: close before a native replacement, never replay."""



from compiler.frontend.descriptor_windows import DescriptorWindowUnsupported
from compiler.ir import CompilationError

ENTRY = "renamed_stage::step"
ROOTS = ("argument::a", "argument::b")

PLANNED_SOURCE = """module renamed_stage
implicit none
integer::prefix_visits=0,tail_visits=0,bad_visits=0,good_visits=0,condition_visits=0
real(8)::prefix_total=0.0_8
contains
subroutine native_prefix(a)
real(8),intent(in)::a(:)
prefix_visits=prefix_visits+1
prefix_total=sum(a)
end subroutine
subroutine replace_storage(a)
real(8),allocatable,intent(inout)::a(:)
real(8),allocatable::temporary(:)
allocate(temporary(lbound(a,1):ubound(a,1)))
temporary=a+10.0_8
call move_alloc(temporary,a)
bad_visits=bad_visits+1
end subroutine
subroutine face_update(a)
real(8),intent(inout)::a(:)
a(1)=a(1)+3.0_8
good_visits=good_visits+1
end subroutine
subroutine step(a,b,n,iterations,bad)
real(8),allocatable,intent(inout)::a(:),b(:)
integer,intent(in)::n,iterations
logical,intent(in)::bad
real(8),allocatable::temporary(:)
integer::stage,i
call native_prefix(a)
do stage=1,iterations
do i=-2,n-3
b(i)=a(i)+real(i+4,8)
enddo
condition_visits=condition_visits+1
if(bad) then
call replace_storage(b)
else
call face_update(b)
endif
do i=-2,n-3
a(i)=2.0_8*b(i)
enddo
call move_alloc(a,temporary)
call move_alloc(temporary,a)
tail_visits=tail_visits+1
enddo
end subroutine
end module
"""


def test_plannable_failure_has_explicit_type_without_new_execution_authority():
    assert isinstance(DescriptorWindowUnsupported("local unsupported source"), CompilationError)
