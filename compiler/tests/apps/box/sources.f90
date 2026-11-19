! Point sources: a list of cells that receive heat each step and record what they see.
module sources
  use params
  implicit none
  type source
    integer :: xi, yj, zk
    real(rp) :: strength
    real(rp) :: seen = 0
    logical :: marked = .false.
  end type source
  type(source), allocatable :: points(:)
contains
  ! count > 0 places that many sources in different cells; count < 0 places -count sources that share
  ! cells in pairs; count == 0 leaves the list empty.
  subroutine place(count)
    integer, intent(in) :: count
    integer :: q, cell
    allocate(points(abs(count)))
    do q = 1, abs(count)
      cell = (q - 1) * 7
      if (count < 0) cell = ((q - 1) / 2) * 7
      points(q)%xi = 1 + mod(cell, nx)
      points(q)%yj = 1 + mod(cell / nx, ny)
      points(q)%zk = 1 + mod(cell / (nx * ny), nz)
      points(q)%strength = 0.5_rp + 0.25_rp * q
    end do
  end subroutine place

  subroutine inject(a)
    real(rp), contiguous, intent(inout) :: a(-1:,-1:,-1:)
    integer :: q
    do q = 1, size(points)
      a(points(q)%xi, points(q)%yj, points(q)%zk) = a(points(q)%xi, points(q)%yj, points(q)%zk) &
                                                     + dt * points(q)%strength
    end do
  end subroutine inject

  subroutine sense(a)
    real(rp), contiguous, intent(in) :: a(-1:,-1:,-1:)
    integer :: q
    do q = 1, size(points)
      points(q)%seen = points(q)%seen + a(points(q)%xi, points(q)%yj, points(q)%zk)
    end do
  end subroutine sense
end module sources
