! kernels
module native_device_bounds_module
  implicit none
  integer, parameter :: knd = kind(1.0d0)
contains
  ! kernel
  subroutine native_device_bounds(bounds, arr, n)
    integer, intent(inout) :: bounds(:)
    real(knd), intent(inout) :: arr(:)
    integer, intent(in) :: n
    integer :: i
    do i = 1, n
      bounds(i) = i + 1
    end do
    do i = bounds(1), bounds(4), bounds(2)
      arr(i) = arr(i) + bounds(3)
    end do
  end subroutine native_device_bounds
end module native_device_bounds_module
