module demand_owner
implicit none
type profile
real(8),pointer::flux(:)
integer::visits=0
end type
type(profile)::metadata
real(8)::a(-1:6,-1:6,-1:6),b(-1:6,-1:6,-1:6)
real(8)::star(-1:6,-1:6,-1:6),out(-1:6,-1:6,-1:6)
contains
subroutine advance()
integer::i,j,k
do k=-1,6
do j=-1,6
do i=-1,6
b(i,j,k)=b(i,j,k)+2*a(i,j,k)
enddo
enddo
enddo
call correct(-0.5d0*a(0,0,0))
do k=0,5
do j=0,5
do i=0,5
star(i,j,k)=b(i,j,k)+0.25d0*(a(i-1,j,k)+a(i+1,j,k)-2*a(i,j,k))
enddo
enddo
enddo
do k=0,5
do j=0,5
do i=0,5
out(i,j,k)=out(i,j,k)+0.5d0*star(i,j,k)
enddo
enddo
enddo
end subroutine
subroutine correct(scale)
real(8),intent(in)::scale
if(scale<0.d0)then
metadata%flux(1)=metadata%flux(1)+scale+a(1,1,1)
else
metadata%flux(2)=metadata%flux(2)-scale+a(2,2,2)
endif
metadata%visits=metadata%visits+1
end subroutine
end module
