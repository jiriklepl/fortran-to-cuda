! kernels
module native_recurrence_module
  implicit none
  integer, parameter :: knd = kind(1.0d0)
contains
  ! kernel
  subroutine native_recurrence(arr, nx, ny, nz)
    real(knd), contiguous, intent(inout) :: arr(:,:,:)
    integer, intent(in) :: nx, ny, nz
    integer :: i, j, k

    do k = 1, nz
      do j = 1, ny
        do i = 2, nx
          arr(i,j,k) = arr(i-1,j,k) + 1.0_knd
        end do
      end do
    end do
  end subroutine native_recurrence
end module native_recurrence_module
