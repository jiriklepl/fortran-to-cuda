program native_test
  use native_integer_module
  implicit none
  integer :: arr(3,2,1,2,2)
  integer :: i, j, k, l, m
  do m = 1, size(arr,5)
    do l = 1, size(arr,4)
      do k = 1, size(arr,3)
        do j = 1, size(arr,2)
          do i = 1, size(arr,1)
            arr(i,j,k,l,m) = -10*i - 100*j - 1000*l - 10000*m
          end do
        end do
      end do
    end do
  end do
  call native_integer_rank5(arr, 17)
  do m = 1, size(arr,5)
    do l = 1, size(arr,4)
      do k = 1, size(arr,3)
        do j = 1, size(arr,2)
          do i = 1, size(arr,1)
            write(*,'(i0)') arr(i,j,k,l,m)
          end do
        end do
      end do
    end do
  end do
end program native_test
