! kernels
module fill_array_module
  implicit none
  integer, parameter :: knd = kind(1.0d0)
contains
  ! kernel
  subroutine fill_array(arr, value, nx, ny, nz)
    real(knd), contiguous, intent(out) :: arr(:,:,:)
    real(knd), intent(in) :: value
    integer, intent(in) :: nx, ny, nz
    integer :: i, j, k

    do k = 1, nz
      do j = 1, ny
        do i = 1, nx
          arr(i,j,k) = value
        end do
      end do
    end do
  end subroutine fill_array
end module fill_array_module
