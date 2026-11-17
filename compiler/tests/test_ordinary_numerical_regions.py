"""Ordinary lexical owners consume source-issued numerical pairs, without a plan."""



ORDINARY_CHILD_SOURCE = """module quartz_numerics
use,intrinsic::ieee_exceptions,only:ieee_set_halting_mode,ieee_invalid
implicit none
integer::bridge_visits=0,middle_visits=0
real(8)::coefficient(-1:0)=[2.0_8,3.0_8]
contains
subroutine bridge(a,b,c,n,tile)
real(8),intent(in)::a(-4:)
real(8),intent(inout)::b(-4:),c(-4:)
integer,intent(in)::n,tile
integer::i
bridge_visits=bridge_visits+1
i=0
!$omp parallel
call ieee_set_halting_mode(ieee_invalid,.false.)
!$omp end parallel
!$omp parallel private(i)
!$omp do schedule(static)
do i=-3,n
b(i)=b(i)+a(i)*coefficient(-1)
enddo
!$omp end do
!$omp do schedule(static)
do i=-2,n
b(i)=b(i)+a(i)*coefficient(0)
enddo
!$omp end do nowait
!$omp end parallel
if(n>=-3) then
call touch_face(b)
endif
!$omp parallel private(i)
!$omp do schedule(static)
do i=-3,n
c(i)=2.0_8*b(i)+real(i+1,8)
enddo
!$omp end do
!$omp end parallel
end subroutine
subroutine touch_face(b)
real(8),intent(inout)::b(-4:)
middle_visits=middle_visits+1
b(-3)=b(-3)+0.5_8
end subroutine
end module
"""
