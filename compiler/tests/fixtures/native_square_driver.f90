program native_test
  use native_square_module
  implicit none
  real(knd) :: arr(16)
  integer :: i
  do i = 1, size(arr,1)
    arr(i) = 0.125_knd*i - 0.375_knd
  end do
  call native_square(arr, 4)
  do i = 1, size(arr,1)
    write(*,'(g0.17)') arr(i)
  end do
end program native_test
