! kernels
module native_literals_module
  implicit none
  integer, parameter :: knd = kind(1.0d0)
contains
  ! kernel
  subroutine native_literals(arr, value, nx, ny, nz)
    real(knd), contiguous, intent(inout) :: arr(:,:,:)
    real(knd), intent(in) :: value
    integer, intent(in) :: nx, ny, nz
    integer :: i, j, k

    do k = 010 - 9, nz
      do j = 1, ny
        do i = 1, nx
          arr(i,j,k) = arr(i,j,k) + value * (0.1 + 0.2) + 0.1 + 0.1d0 + 0.2_knd + 010 + 09
        end do
      end do
    end do
  end subroutine native_literals
end module native_literals_module
