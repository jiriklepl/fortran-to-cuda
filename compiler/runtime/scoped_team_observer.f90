! Offline observers never calibrate an application invocation.
module fort_scoped_team_observer
  use iso_c_binding
  implicit none
  integer(c_int32_t), parameter :: FORT_SCOPE_TEAM_OBSERVE_OWNER=1, FORT_SCOPE_TEAM_OBSERVE_DESCRIPTOR=2
  integer(c_int32_t), parameter :: FORT_SCOPE_TEAM_OBSERVE_ENTRY=3, FORT_SCOPE_TEAM_OBSERVE_CPU_WORKER=4
  integer(c_int32_t), parameter :: FORT_SCOPE_TEAM_OBSERVE_GPU_WORKER=5, FORT_SCOPE_TEAM_OBSERVE_NATIVE_CALL=6
  integer(c_int32_t), parameter :: FORT_SCOPE_TEAM_OBSERVE_COMPUTE=7, FORT_SCOPE_TEAM_OBSERVE_NATIVE_WORKER=8
  interface
    function fort_scope_team_fortran_compatible_v1(version,options,expected,semantic) bind(C) result(compatible)
      import c_char, c_int
      character(c_char), intent(in) :: version(*),options(*),expected(*),semantic(*)
      integer(c_int) :: compatible
    end function
    function fort_scope_team_observer_enabled_v1() bind(C) result(enabled)
      import c_int
      integer(c_int) :: enabled
    end function
    subroutine fort_scope_team_observe_begin_v1(stage) bind(C)
      import c_int32_t
      integer(c_int32_t), value :: stage
    end subroutine
    subroutine fort_scope_team_observe_end_v1(stage) bind(C)
      import c_int32_t
      integer(c_int32_t), value :: stage
    end subroutine
    function fort_scope_team_observer_reset_v1(threads) bind(C) result(status)
      import c_int32_t, c_int
      integer(c_int32_t), value :: threads
      integer(c_int) :: status
    end function
    function fort_scope_team_observer_read_v1(stage,threads,seconds,calls) bind(C) result(status)
      import c_int32_t, c_int64_t, c_int, c_double
      integer(c_int32_t), value :: stage, threads
      real(c_double), intent(out) :: seconds
      integer(c_int64_t), intent(out) :: calls
      integer(c_int) :: status
    end function
  end interface
end module
