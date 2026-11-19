! box NX NY NZ STEPS REPORT_EVERY: diffuse a periodic field and write it to field.bin.
program box
  use params
  use fields
  use stepper, only: advance
  use diagnostics, only: report
  implicit none
  integer :: step, unit
  character(32) :: argument
  integer :: values(5), index

  do index = 1, 5
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

  !$fort enter budget(1073741824) if(min(nx, ny, nz) >= 2)
  do step = 1, steps
    call advance(u, unew)
    if (mod(step, report_every) == 0) call report(step, u)
  end do
  !$fort leave

  open(newunit=unit, file='field.bin', access='stream', form='unformatted', status='replace')
  write(unit) u
  close(unit)
  !$fort close
  deallocate(u, unew)
end program box
