program native_test
  use native_nonaffine_module
  implicit none
  real(knd) :: src(9), dst(3,9)
  integer :: indices(3), i, j
  indices = [9, 1, 4]
  do i = 1, size(src,1)
    src(i) = 0.125_knd*i - 0.5_knd
  end do
  do j = 1, size(dst,2)
    do i = 1, size(dst,1)
      dst(i,j) = -10.0_knd*i - 0.25_knd*j
    end do
  end do
  call native_nonaffine(src, indices, dst, 3)
  do j = 1, size(dst,2)
    do i = 1, size(dst,1)
      write(*,'(g0.17)') dst(i,j)
    end do
  end do
end program native_test
