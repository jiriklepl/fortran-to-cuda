program native_test
  use native_device_bounds_module
  implicit none
  integer :: bounds(4), i
  real(knd) :: arr(9)
  bounds = [8, 1, 9, 9]
  do i = 1, size(arr,1)
    arr(i) = 0.125_knd*i - 0.375_knd
  end do
  call native_device_bounds(bounds, arr, 4)
  do i = 1, size(arr,1)
    write(*,'(g0.17)') arr(i)
  end do
  do i = 1, size(bounds,1)
    write(*,'(i0)') bounds(i)
  end do
end program native_test
