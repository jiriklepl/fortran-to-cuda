! Periodic halos. The copies do not overlap from two cells per direction up.
module boundaries
  use params
  implicit none
contains
  subroutine periodic(a)
    real(rp), contiguous, intent(inout) :: a(-1:,-1:,-1:)
    integer :: i, j, k
    !$omp parallel private(i,j,k)
    !$omp do
    do k = 1, nz
      do j = 1, ny
        do i = 1, 2
          a(nx+i,j,k) = a(i,j,k)
          a(i-2,j,k) = a(nx+i-2,j,k)
        end do
      end do
    end do
    !$omp end do
    !$omp do
    do k = 1, nz
      do j = 1, 2
        do i = -1, nx+2
          a(i,ny+j,k) = a(i,j,k)
          a(i,j-2,k) = a(i,ny+j-2,k)
        end do
      end do
    end do
    !$omp end do
    !$omp do
    do k = 1, 2
      do j = -1, ny+2
        do i = -1, nx+2
          a(i,j,nz+k) = a(i,j,k)
          a(i,j,k-2) = a(i,j,nz+k-2)
        end do
      end do
    end do
    !$omp end do
    !$omp end parallel
  end subroutine periodic
end module boundaries
