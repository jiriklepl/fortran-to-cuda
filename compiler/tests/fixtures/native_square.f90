! kernels
module native_square_module
  implicit none
  integer, parameter :: knd = kind(1.0d0)
contains
  ! kernel
  subroutine native_square(arr, n)
    real(knd), intent(inout) :: arr(:)
    integer, intent(in) :: n
    integer :: i
    do i = 1, n
      arr(i*i) = arr(i*i) + 0.125_knd*i
    end do
    do i = -n, -1
      arr(i*i) = arr(i*i)*1.5_knd + 0.25_knd*i
    end do
  end subroutine native_square
end module native_square_module
