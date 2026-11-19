! Host-side diagnostics. This code is not compiled for the device: it reads the field on the host.
module diagnostics
  use params
  implicit none
contains
  subroutine report(step, a)
    integer, intent(in) :: step
    real(rp), contiguous, intent(in) :: a(-1:,-1:,-1:)
    real(rp) :: energy
    integer :: i, j, k
    !$fort observe(a)
    energy = 0
    do k = 1, nz
      do j = 1, ny
        do i = 1, nx
          energy = energy + a(i,j,k)*a(i,j,k)
        end do
      end do
    end do
    print '(a,i0,a,es24.16)', 'step ', step, ' energy ', energy
  end subroutine report
end module diagnostics
