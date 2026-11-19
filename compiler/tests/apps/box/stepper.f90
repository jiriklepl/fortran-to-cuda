! One explicit diffusion step.
module stepper
  use params
  use boundaries, only: periodic
  implicit none
contains
  pure function laplacian(i, j, k, field, spacing) result(value)
    integer, intent(in) :: i, j, k
    real(rp), contiguous, intent(in) :: field(-1:,-1:,-1:)
    real(rp), intent(in) :: spacing
    real(rp) :: value
    value = (field(i-1,j,k) + field(i+1,j,k) + field(i,j-1,k) + field(i,j+1,k) + field(i,j,k-1) + field(i,j,k+1) &
             - 6*field(i,j,k)) / (spacing*spacing)
  end function laplacian

  subroutine diffuse(a, b)
    real(rp), contiguous, intent(in) :: a(-1:,-1:,-1:)
    real(rp), contiguous, intent(inout) :: b(-1:,-1:,-1:)
    integer :: i, j, k
    !$omp parallel do private(i,j)
    do k = 1, nz
      do j = 1, ny
        do i = 1, nx
          b(i,j,k) = a(i,j,k) + dt*nu*laplacian(i, j, k, a, dx)
        end do
      end do
    end do
    !$omp end parallel do
  end subroutine diffuse

  subroutine store(a, b)
    real(rp), contiguous, intent(inout) :: a(-1:,-1:,-1:)
    real(rp), contiguous, intent(in) :: b(-1:,-1:,-1:)
    integer :: i, j, k
    !$omp parallel do private(i,j)
    do k = 1, nz
      do j = 1, ny
        do i = 1, nx
          a(i,j,k) = b(i,j,k)
        end do
      end do
    end do
    !$omp end parallel do
  end subroutine store

  subroutine advance(a, b)
    real(rp), contiguous, intent(inout) :: a(-1:,-1:,-1:), b(-1:,-1:,-1:)
    call periodic(a)
    call diffuse(a, b)
    call store(a, b)
  end subroutine advance
end module stepper
