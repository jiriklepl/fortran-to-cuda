! kernels
module native_shadow_module
  implicit none
  integer, parameter :: knd = kind(1.0d0)
contains
  ! kernel
  subroutine native_shadow(arr, value, size, int, real)
    real(knd), contiguous, intent(inout) :: arr(:,:,:)
    real(knd), intent(in) :: value
    integer, intent(in) :: size, int, real
    integer :: i, j, k

    do k = 1, real
      do j = 1, int
        do i = 1, size
          arr(i,j,k) = arr(i,j,k) * value
        end do
      end do
    end do
  end subroutine native_shadow
end module native_shadow_module
