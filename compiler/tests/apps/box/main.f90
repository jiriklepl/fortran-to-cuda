! box NX NY NZ STEPS REPORT_EVERY [SOURCES]: diffuse a periodic field with point sources and write it to
! field.bin. SOURCES > 0 places that many sources in different cells, < 0 lets them share cells, 0 none.
program box
  use params
  use fields
  use stepper, only: advance
  use diagnostics, only: report
  use sources, only: place, points
  implicit none
  integer :: step, unit
  character(32) :: argument
  integer :: values(6), index

  values(6) = 5
  do index = 1, min(6, command_argument_count())
    call get_command_argument(index, argument)
    read(argument, *) values(index)
  end do
  nx = values(1)
  ny = values(2)
  nz = values(3)
  steps = values(4)
  report_every = values(5)
  dx = 1._rp / nx
  nu = 0.1_rp
  dt = 0.1_rp * dx * dx / nu
  call create()
  call place(values(6))

  !$fort enter if(min(nx, ny, nz) >= 2)
  do step = 1, steps
    call advance(u, unew)
    if (mod(step, report_every) == 0) call report(step, u)
  end do
  !$fort leave

  open(newunit=unit, file='field.bin', access='stream', form='unformatted', status='replace')
  write(unit) u
  write(unit) points%seen
  close(unit)
  !$fort close
  deallocate(u, unew)
end program box
