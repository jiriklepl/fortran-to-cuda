module tidal_fields
  implicit none
contains
  subroutine tidal_update(source, destination, nx, ny, nz)
    integer, intent(in) :: nx, ny, nz
    real(8), intent(in) :: source(:,:,:)
    real(8), intent(inout) :: destination(:,:,:)
    integer :: i, j, k
    !$omp parallel do collapse(3) private(i,j,k)
    do k=2,nz+1
      do j=2,ny+1
        do i=2,nx+1
          destination(i,j,k)=1.25d0*source(i,j,k)+0.125d0
        end do
      end do
    end do
    !$omp end parallel do
  end subroutine
end module
