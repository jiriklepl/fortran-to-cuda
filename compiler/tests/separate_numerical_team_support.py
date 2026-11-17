"""Whole separate numerical teams retain their source worksharing sequence."""




ENTRY = "bounded_team::step"
BODY = """!$omp parallel private(i)
!$omp do schedule(static)
do i=-3,n
b(i)=a(i)+real(i,8)
enddo
!$omp end do nowait
!$omp do schedule(static)
do i=-3,n
c(i)=b(i)+a(i)
enddo
!$omp end do nowait
!$omp end parallel
"""
SOURCE = """module bounded_team
implicit none
contains
subroutine step(a,b,c,n)
real(8),intent(in)::a(-4:)
real(8),intent(inout)::b(-4:),c(-4:)
integer,intent(in)::n
integer::i
real(8)::s
""" + BODY + "end subroutine\nend module\n"
