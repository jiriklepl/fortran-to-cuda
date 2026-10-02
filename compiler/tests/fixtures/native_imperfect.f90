! kernels
module native_imperfect_module
  implicit none
  integer, parameter :: knd = kind(1.0d0)
contains
  ! kernel
  subroutine native_imperfect(arr, value, nx, ny, nz)
    real(knd), intent(inout) :: arr(:,:,:)
    real(knd), intent(in) :: value
    integer, intent(in) :: nx, ny, nz
    real(knd) :: coefficient, before
    integer :: i, j, k
    do k = 1, nz
      coefficient = value + 0.25_knd*k
      do j = 1, ny
        before = arr(1,j,k)
        arr(1,j,k) = before + coefficient
        do i = 2, nx
          arr(i,j,k) = arr(i-1,j,k) + coefficient
        end do
        arr(nx,j,k) = arr(nx,j,k) + before + 0.125_knd*i
      end do
    end do
  end subroutine native_imperfect
end module native_imperfect_module
