! kernels
module native_single_module
  implicit none
contains
  ! kernel
  subroutine native_single_precision(arr, factor, n)
    real :: arr(:), factor
    integer :: n, i
    do i = 1, n
      arr(i) = arr(i)*factor + 0.1
    end do
  end subroutine native_single_precision
end module native_single_module
