! Run parameters of the box. The precision is chosen when the sources are preprocessed.
module params
  implicit none
#ifdef DOUBLE
  integer, parameter :: rp = kind(1.d0)
#else
  integer, parameter :: rp = kind(1.0)
#endif
  integer :: nx = 0, ny = 0, nz = 0
  integer :: steps = 0, report_every = 1
  real(rp) :: dx, dt, nu
end module params
