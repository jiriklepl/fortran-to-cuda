! The field and its update buffer, with two halo cells on every side.
module fields
  use params
  implicit none
  real(rp), allocatable :: u(:,:,:), unew(:,:,:)
contains
  subroutine create()
    integer :: i, j, k
    real(rp), parameter :: pi = 3.14159265358979323846_rp
    allocate(u(-1:nx+2,-1:ny+2,-1:nz+2), unew(-1:nx+2,-1:ny+2,-1:nz+2))
    u = 0
    unew = 0
    do k = 1, nz
      do j = 1, ny
        do i = 1, nx
          u(i,j,k) = sin(2*pi*(i-0.5_rp)/nx) * cos(2*pi*(j-0.5_rp)/ny) + 0.25_rp*sin(4*pi*(k-0.5_rp)/nz)
        end do
      end do
    end do
  end subroutine create
end module fields
