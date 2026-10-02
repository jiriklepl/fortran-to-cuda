program native_test
  use native_strides_module
  implicit none
  real(knd) :: arr(10,3)
  integer :: i, j
  do j = 1, size(arr,2)
    do i = 1, size(arr,1)
      arr(i,j) = 0.125_knd*i + 0.5_knd*j
    end do
  end do
  call native_strides(arr, VAR_NX, VAR_NY, VAR_NZ)
  do j = 1, size(arr,2)
    do i = 1, size(arr,1)
      write(*,'(g0.17)') arr(i,j)
    end do
  end do
end program native_test
