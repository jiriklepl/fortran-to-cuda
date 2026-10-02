! kernels
module native_integer_module
  implicit none
contains
  ! kernel
  subroutine native_integer_rank5(arr, bias)
    integer, intent(inout) :: arr(:,:,:,:,:)
    integer, intent(in) :: bias
    integer :: i, j, k, l, m
    do m = 1, size(arr,5)
      do l = 1, size(arr,4)
        do k = 1, size(arr,3)
          do j = 1, size(arr,2)
            do i = 1, size(arr,1)
              arr(i,j,k,l,m) = arr(i,j,k,l,m) + bias + i + 3*j + 7*k + 11*l + 13*m
            end do
          end do
        end do
      end do
    end do
  end subroutine native_integer_rank5
end module native_integer_module
