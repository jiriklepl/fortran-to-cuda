! kernels
module native_nonaffine_module
  implicit none
  integer, parameter :: knd = kind(1.0d0)
contains
  ! kernel
  subroutine native_nonaffine(src, indices, dst, n)
    real(knd), intent(in) :: src(:)
    integer, intent(in) :: indices(:)
    real(knd), intent(inout) :: dst(:,:)
    integer, intent(in) :: n
    integer :: i, j
    do i = 1, n
      dst(i,1) = src(indices(i))
      do j = 2, n
        dst(i,j*j) = dst(i,1) + src(j*j) + 0.25_knd*i
      end do
    end do
  end subroutine native_nonaffine
end module native_nonaffine_module
