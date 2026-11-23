! The RK owner and pressure child share one caller-owned domain. Replica
! validity, dependencies and allocations remain in the scoped runtime.
module elmm_resident_pressure_state
  use, intrinsic :: iso_c_binding
  use fort_scoped_memory
  use fort_resident_domain, only: resident_domain, pressure_require => resident_require
  use elmm_gpu_pressure
  implicit none
  private
  public :: resident_pressure_state, pressure_require, resident_flow

  type, extends(resident_domain) :: resident_pressure_state
    integer(c_int64_t) :: phi=0, rhs=0
    type(c_ptr) :: solver=c_null_ptr
    type(elmm_gpu_pressure_config_v1) :: config
    logical :: joined=.false.
  contains
    procedure :: start => pressure_start
    procedure :: pressure_bind3_f32
    procedure :: pressure_bind3_f64
    generic :: bind3 => pressure_bind3_f32, pressure_bind3_f64
    procedure :: solve => pressure_solve
    procedure :: observe_scratch => pressure_observe_scratch
    procedure :: finish => pressure_finish
    procedure :: end_epoch => pressure_end_epoch
  end type
  type(resident_pressure_state), save :: resident_flow
contains
  subroutine pressure_start(self, config)
    class(resident_pressure_state), intent(inout) :: self
    type(elmm_gpu_pressure_config_v1), intent(in) :: config
    if (c_associated(self%solver)) then
      if (any(self%config%solver%sizes /= config%solver%sizes) .or. &
          any(self%config%solver%lengths /= config%solver%lengths) .or. &
          any(self%config%solver%boundary_conditions /= config%solver%boundary_conditions) .or. &
          self%config%solver%approximation /= config%solver%approximation .or. &
          self%config%solver%flags /= config%solver%flags) then
        if (self%joined) error stop 'pressure configuration changed inside an RK epoch'
        call self%finish()
      endif
    endif
    ! An epoch that only the pressure solve opens leaves the time step to the host's OpenMP code.
    if (.not. self%active) call self%begin_epoch(0_c_int,keep_team=.true.)
    if (.not. c_associated(self%solver)) then
      self%config=config
      ! The solver sizes its own workspace within what the domain has left.
      self%config%solver%device_budget_bytes=int(self%unclaimed(),c_int64_t)
      call pressure_require(elmm_gpu_pressure_create_v1(self%domain,self%config,self%solver))
    endif
  end subroutine

  function pressure_bind3_f32(self, values, lower, initialized, persistent, origin) result(handle)
    class(resident_pressure_state), intent(inout) :: self
    real(c_float), contiguous, target, intent(in) :: values(:,:,:)
    integer, intent(in) :: lower(3)
    logical, intent(in) :: initialized
    logical, optional, intent(in) :: persistent
    character(*), optional, intent(in) :: origin
    integer(c_int64_t) :: handle
    handle=self%bind_real32(values,lower,initialized,persistent,origin)
  end function

  function pressure_bind3_f64(self, values, lower, initialized, persistent, origin) result(handle)
    class(resident_pressure_state), intent(inout) :: self
    real(c_double), contiguous, target, intent(in) :: values(:,:,:)
    integer, intent(in) :: lower(3)
    logical, intent(in) :: initialized
    logical, optional, intent(in) :: persistent
    character(*), optional, intent(in) :: origin
    integer(c_int64_t) :: handle
    handle=self%bind_real64(values,lower,initialized,persistent,origin)
  end function

  subroutine pressure_solve(self)
    class(resident_pressure_state), intent(inout) :: self
    type(fort_scope_external_report_v1) :: report
    call pressure_require(elmm_gpu_pressure_solve_v1(self%context,self%solver,self%phi,self%rhs,report))
  end subroutine

  subroutine pressure_observe_scratch(self)
    class(resident_pressure_state), intent(inout) :: self
    call self%observe([self%phi,self%rhs])
  end subroutine

  ! Whoever ends the stay on the device, this adapter or code the compiler generated before host
  ! statements, the time step is no longer part of it.
  subroutine pressure_end_epoch(self)
    class(resident_pressure_state), intent(inout) :: self
    call self%resident_domain%end_epoch()
    self%joined=.false.
  end subroutine

  subroutine pressure_finish(self)
    class(resident_pressure_state), intent(inout) :: self
    if (self%domain == 0) return
    call self%publish()
    call pressure_require(fort_scope_wait(self%context))
    if (c_associated(self%solver)) call pressure_require(elmm_gpu_pressure_destroy_v1(self%solver))
    call self%close()
    self%phi=0; self%rhs=0; self%joined=.false.
  end subroutine
end module
