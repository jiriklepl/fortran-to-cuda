! kernels
module native_half_bound_module
  implicit none
  integer, parameter :: knd = kind(1.0d0)
contains
  ! kernel
  subroutine native_half_bound(arr, value, nx, ny, nz)
    real(knd), contiguous, intent(inout) :: arr(:,:,:)
    real(knd), intent(in) :: value
    integer, intent(in) :: nx, ny, nz
    integer :: i, j, k, stop

    stop = nx / 2
    do k = 1, nz
      do j = 1, ny
        do i = 1, stop
          arr(i,j,k) = arr(i,j,k) * value
        end do
      end do
    end do
  end subroutine native_half_bound
end module native_half_bound_module
