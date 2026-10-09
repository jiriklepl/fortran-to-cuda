module painted_faces
  implicit none
contains
  subroutine paint_faces(source, destination, nx, ny, nz)
    integer, intent(in) :: nx, ny, nz
    real(8), intent(in) :: source(:,:,:)
    real(8), intent(inout) :: destination(:,:,:)
    integer :: i, j
    !$omp parallel do collapse(2) private(i,j)
    do j=2,ny+1
      do i=2,nx+1
        destination(i,j,1)=0.75d0*source(i,j,2)
        destination(i,j,nz+2)=1.25d0*source(i,j,nz+1)
      end do
    end do
    !$omp end parallel do
  end subroutine
end module
