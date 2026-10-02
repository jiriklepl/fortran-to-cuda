! kernels
module native_host_module
  implicit none
  integer, parameter :: knd = kind(1.0d0)
contains
  ! kernel
  subroutine native_host_coherence(arr, value, nx, ny, nz)
    real(knd), intent(inout) :: arr(:,:,:)
    real(knd), intent(in) :: value
    integer, intent(in) :: nx, ny, nz
    real(knd) :: before, between, after
    integer :: i, j, k
    before = arr(1,1,1)
    arr(1,1,1) = before + value
    do k = 2, nz-1
      do j = 2, ny-1
        do i = 2, nx-1
          arr(i,j,k) = arr(i,j,k) + arr(1,1,1)
        end do
      end do
    end do
    between = arr(2,2,2)
    arr(nx,ny,nz) = between + before
    do k = 2, nz-1
      do j = 2, ny-1
        do i = 2, nx-1
          arr(i,j,k) = arr(i,j,k) + arr(nx,ny,nz)
        end do
      end do
    end do
    after = arr(3,2,2)
    arr(1,2,1) = after + between
  end subroutine native_host_coherence
end module native_host_module
