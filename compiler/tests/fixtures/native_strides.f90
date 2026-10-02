! kernels
module native_strides_module
  implicit none
  integer, parameter :: knd = kind(1.0d0)
contains
  ! kernel
  subroutine native_strides(arr, lower, upper, stride)
    real(knd), intent(inout) :: arr(:,:)
    integer, intent(in) :: lower, upper, stride
    integer :: i, j
    do j = 1, size(arr,2)
      do i = lower, upper, stride
        arr(i,j) = arr(i,j) + 0.125_knd*i - 0.25_knd*j
      end do
    end do
    do i = 2, size(arr,1)-1, 2
      do j = 1, size(arr,2), 2
        arr(i,j) = arr(i,j) * 1.5_knd
      end do
    end do
    do j = size(arr,2), 1, -2
      do i = size(arr,1)-1, 2, -3
        arr(i,j) = arr(i,j) - 0.375_knd
      end do
    end do
  end subroutine native_strides
end module native_strides_module
