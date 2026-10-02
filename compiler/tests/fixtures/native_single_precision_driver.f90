program native_test
  use native_single_module
  implicit none
  real :: arr(9)
  integer :: i
  do i = 1, size(arr,1)
    arr(i) = 0.125*i
  end do
  call native_single_precision(arr, 1.25, 9)
  do i = 1, size(arr,1)
    write(*,'(g0.17)') arr(i)
  end do
end program native_test
