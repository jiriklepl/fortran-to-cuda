! kernels
module scale_array_module
  implicit none
  integer, parameter :: knd = kind(1.0d0)
contains
  ! kernel
  subroutine scale_array(arr, factor, nx, ny, nz)
    real(knd), contiguous, intent(inout) :: arr(:,:,:)
    real(knd), intent(in) :: factor
    integer, intent(in) :: nx, ny, nz
    integer :: i, j, k

    do k = 1, nz
      do j = 1, ny
        do i = 1, nx
          arr(i,j,k) = arr(i,j,k) * factor
        end do
      end do
    end do
  end subroutine scale_array
end module scale_array_module
