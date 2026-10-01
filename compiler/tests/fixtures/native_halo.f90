! kernels
module native_halo_module
  implicit none
  integer, parameter :: knd = kind(1.0d0)
contains
  ! kernel
  subroutine native_halo(arr, factor, nx, ny, nz)
    real(knd), contiguous, intent(inout) :: arr(:,:,:)
    real(knd), intent(in) :: factor
    integer, intent(in) :: nx, ny, nz
    integer :: i, j, k

    do k = 2, nz + 1
      do j = 2, ny + 1
        do i = 2, nx + 1
          arr(i,j,k) = arr(i,j,k) * factor
        end do
      end do
    end do
  end subroutine native_halo
end module native_halo_module
