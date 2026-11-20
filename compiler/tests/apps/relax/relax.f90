! relax N SWEEPS: Jacobi sweeps on a square whose west edge is held at one, written the way a small
! solver often is: everything in the main program, array assignments for the copies, and one
! procedure outside any module. The field goes to field.bin.
program relax
  implicit none
  integer, parameter :: rp = kind(1d0)
  real(rp), allocatable :: u(:,:), v(:,:)
  real(rp) :: change, top
  integer :: n, sweeps, sweep, i, j, unit
  character(32) :: argument

  n = 64
  sweeps = 40
  if (command_argument_count() >= 1) then
    call get_command_argument(1, argument)
    read(argument, *) n
  end if
  if (command_argument_count() >= 2) then
    call get_command_argument(2, argument)
    read(argument, *) sweeps
  end if
  allocate(u(0:n+1, 0:n+1), v(0:n+1, 0:n+1))
  u = 0
  u(0, :) = 1
  v(:, :) = u(:, :)

  !$fort enter
  do sweep = 1, sweeps
    !$omp parallel do private(i)
    do j = 1, n
      do i = 1, n
        v(i,j) = 0.25_rp * (u(i-1,j) + u(i+1,j) + u(i,j-1) + u(i,j+1))
      end do
    end do
    !$omp end parallel do
    change = 0
    !$omp parallel do private(i) reduction(max: change)
    do j = 1, n
      do i = 1, n
        change = max(change, abs(v(i,j) - u(i,j)))
      end do
    end do
    !$omp end parallel do
    u(1:n, 1:n) = v(1:n, 1:n)
    if (mod(sweep, 10) == 0) then
      call damp(u, n, top)
      print '(a,i0,a,es24.16,a,es24.16)', 'sweep ', sweep, ' change ', change, ' top ', top
    end if
  end do

  open(newunit=unit, file='field.bin', access='stream', form='unformatted', status='replace')
  !$fort leave
  write(unit) u
  close(unit)
  !$fort close
  deallocate(u, v)
end program relax

! Scales the interior a little and reports its largest value.
subroutine damp(a, m, top)
  implicit none
  integer, intent(in) :: m
  real(kind(1d0)), intent(inout) :: a(0:m+1, 0:m+1)
  real(kind(1d0)), intent(out) :: top
  integer :: i, j
  a(1:m, 1:m) = a(1:m, 1:m) * 0.999d0
  top = 0
  !$omp parallel do reduction(max: top)
  do j = 1, m
    do i = 1, m
      top = max(top, a(i,j))
    end do
  end do
  !$omp end parallel do
end subroutine damp
