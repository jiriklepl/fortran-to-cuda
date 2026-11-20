! Diagnostics. report is host code: it reads the field on the host. summary and peak are compiled for
! the device: their loops leave only a few scalars, so the field stays where it is.
module diagnostics
  use params
  use sources, only: points, range
  implicit none
contains
  subroutine report(step, a)
    integer, intent(in) :: step
    real(rp), contiguous, intent(in) :: a(-1:,-1:,-1:)
    real(rp) :: energy
    integer :: i, j, k
    !$fort observe(a, points(:)%seen)
    energy = 0
    do k = 1, nz
      do j = 1, ny
        do i = 1, nx
          energy = energy + a(i,j,k)*a(i,j,k)
        end do
      end do
    end do
    print '(a,i0,a,es24.16,a,es24.16)', 'step ', step, ' energy ', energy, ' seen ', sum(points%seen)
  end subroutine report

  subroutine summary(step, a)
    integer, intent(in) :: step
    real(rp), contiguous, intent(in) :: a(-1:,-1:,-1:)
    real(rp) :: weight, lowest
    integer :: i, j, k, warm
    logical :: broken, every
    weight = 0
    warm = 0
    broken = .false.
    !$omp parallel do private(i, j, k) reduction(+: weight, warm) &
    !$omp& reduction(.or.: broken)
    do k = 1, nz
      do j = 1, ny
        do i = 1, nx
          if (a(i,j,k) > 0) then
            warm = warm + 1
            ! Quarters add up to the same sum in any order.
            weight = weight + 0.25_rp
          end if
          if (a(i,j,k) /= a(i,j,k)) broken = .true.
        end do
      end do
    end do
    !$omp end parallel do
    call range(lowest, every)
    print '(a,i0,a,i0,a,es24.16,a,es24.16,a,l1,a,es24.16,a,l1)', 'step ', step, ' warm ', warm, &
      ' weight ', weight, ' peak ', peak(a), ' broken ', broken, ' lowest ', lowest, ' every ', every
  end subroutine summary

  ! The largest magnitude: a maximum is the same in any order, so this loop needs no directive.
  real(rp) function peak(a)
    real(rp), contiguous, intent(in) :: a(-1:,-1:,-1:)
    integer :: i, j, k
    peak = 0
    do k = 1, nz
      do j = 1, ny
        do i = 1, nx
          peak = max(peak, abs(a(i,j,k)))
        end do
      end do
    end do
  end function peak
end module diagnostics
