module polynomial_fields
  implicit none
contains
  subroutine polynomial_steps(source, destination, nx, ny, nz)
    integer, intent(in) :: nx, ny, nz
    real(8), intent(in) :: source(:,:,:)
    real(8), intent(inout) :: destination(:,:,:)
    integer :: i, j, k
    real(8) :: term
    !$omp parallel do collapse(3) private(i,j,k,term)
    do k=2,nz+1
      do j=2,ny+1
        do i=2,nx+1
          term=source(i,j,k)
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          destination(i,j,k)=term
        end do
      end do
    end do
    !$omp end parallel do
    !$omp parallel do collapse(3) private(i,j,k,term)
    do k=2,nz+1
      do j=2,ny+1
        do i=2,nx+1
          term=destination(i,j,k)
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          destination(i,j,k)=term
        end do
      end do
    end do
    !$omp end parallel do
    !$omp parallel do collapse(3) private(i,j,k,term)
    do k=2,nz+1
      do j=2,ny+1
        do i=2,nx+1
          term=destination(i,j,k)
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          term=term*0.9999d0+source(i,j,k)*0.0001d0
          term=term*0.99995d0+source(i,j,k)*0.0001d0
          destination(i,j,k)=term
        end do
      end do
    end do
    !$omp end parallel do
  end subroutine
end module
