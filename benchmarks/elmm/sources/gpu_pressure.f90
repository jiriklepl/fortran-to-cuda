module elmm_gpu_pressure
  use, intrinsic :: iso_c_binding
  use poisson_solver_device, only: poisson_solver_config_v1
  use fort_scoped_memory, only: fort_scope_external_report_v1
  implicit none
  type, bind(C) :: elmm_gpu_pressure_config_v1
    integer(c_int32_t) :: version=1, reserved=0
    type(poisson_solver_config_v1) :: solver
    integer(c_int64_t) :: phi_lower_bounds(3)=-1, rhs_lower_bounds(3)=0
  end type
  interface
    function elmm_gpu_pressure_create_v1(domain, config, handle) bind(C) result(status)
      import :: c_int, c_int64_t, c_ptr, elmm_gpu_pressure_config_v1
      integer(c_int64_t), value :: domain
      type(elmm_gpu_pressure_config_v1), intent(in) :: config
      type(c_ptr), intent(out) :: handle
      integer(c_int) :: status
    end function
    function elmm_gpu_pressure_solve_v1(context, handle, phi, rhs, report) bind(C) result(status)
      import :: c_int, c_int64_t, c_ptr, fort_scope_external_report_v1
      integer(c_int64_t), value :: context, phi, rhs
      type(c_ptr), value :: handle
      type(fort_scope_external_report_v1), intent(out) :: report
      integer(c_int) :: status
    end function
    function elmm_gpu_pressure_destroy_v1(handle) bind(C) result(status)
      import :: c_int, c_ptr
      type(c_ptr), intent(inout) :: handle
      integer(c_int) :: status
    end function
    function elmm_gpu_pressure_error_v1() bind(C) result(message)
      import :: c_ptr
      type(c_ptr) :: message
    end function
  end interface
end module
