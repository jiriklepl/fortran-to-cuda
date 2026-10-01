! kernels
module native_order_module
  implicit none
  integer, parameter :: knd = kind(1.0d0)
contains
  ! kernel
  subroutine adjust(arr, factor, nx, ny, nz)
    real(knd), contiguous, intent(inout) :: arr(:,:,:)
    real(knd), intent(in) :: factor
    integer, intent(in) :: nx, ny, nz
    real(knd) :: coefficient
    integer :: i, j, k

    coefficient = 2.0_knd * factor
    do k = 1, nz
      do j = 1, ny
        do i = 1, nx
          arr(i,j,k) = arr(i,j,k) + coefficient
        end do
      end do
    end do
  end subroutine adjust

  ! kernel
  subroutine native_order(arr, value, nx, ny, nz)
    integer, intent(in) :: nz, nx, ny
    real(knd), intent(in) :: value
    real(knd), contiguous, intent(inout) :: arr(:,:,:)
    real(knd) :: factor
    integer :: i, j, k

    do k = 1, nz
      do j = 1, ny
        do i = 1, nx
          arr(i,j,k) = arr(i,j,k) * value
        end do
      end do
    end do

    factor = value + 1.0_knd
    call adjust(arr, factor, nx, ny, nz)
    factor = value + 2.0_knd
    call adjust(arr, factor, nx, ny, nz)
  end subroutine native_order
end module native_order_module
