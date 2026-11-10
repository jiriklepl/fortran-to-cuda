module resident_cfl_source
implicit none
contains
subroutine values(u,v,w,cfl,nx,ny,nz,dx,dy,dz)
real(8), intent(in) :: u(:,:,:),v(:,:,:),w(:,:,:)
real(8), intent(out) :: cfl(:,:,:)
integer, value, intent(in) :: nx,ny,nz
real(8), value, intent(in) :: dx,dy,dz
integer :: i,j,k
real(8) :: p
do k=1,nz
 do j=1,ny
  do i=1,nx
   p = max(abs(u(i+3,j+3,k+3)),abs(u(i+2,j+3,k+3))) / dx
   p = p + max(abs(v(i+3,j+3,k+3)),abs(v(i+3,j+2,k+3))) / dy
   p = p + max(abs(w(i+3,j+3,k+3)),abs(w(i+3,j+3,k+2))) / dz
   cfl(i,j,k)=p
  enddo
 enddo
enddo
end subroutine
end module
