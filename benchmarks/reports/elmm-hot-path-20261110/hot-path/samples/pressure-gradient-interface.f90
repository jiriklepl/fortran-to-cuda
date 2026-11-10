module fort_shared_80aa32c5945f
  use iso_c_binding
  use fort_scoped_memory, only: fort_scope_plan_decision
  implicit none
  private
  public :: run, plan, choose, configure
  interface
    function run( &
        fort_context, &
        fort_mode, &
        fort_v0_u_handle, &
        fort_v1_fort_region_lb_u_1, &
        fort_v2_fort_region_lb_u_2, &
        fort_v3_fort_region_lb_u_3, &
        fort_v4_a, &
        fort_v5_pr_gradient_x, &
        fort_v6_unx, &
        fort_v7_uny, &
        fort_v8_unz &
    ) bind(C, name='cpp_fort_shared_80aa32c5945f') result(fort_status)
      import :: c_int, c_int64_t, c_double, c_float, c_bool
      integer(c_int) :: fort_status
      integer(c_int64_t), value :: fort_context
      integer(c_int), value :: fort_mode
      integer(c_int64_t), value :: fort_v0_u_handle
      integer(c_int), intent(in), value :: fort_v1_fort_region_lb_u_1
      integer(c_int), intent(in), value :: fort_v2_fort_region_lb_u_2
      integer(c_int), intent(in), value :: fort_v3_fort_region_lb_u_3
      real(c_double), intent(in) :: fort_v4_a
      real(c_double), intent(in) :: fort_v5_pr_gradient_x
      integer(c_int), intent(in), value :: fort_v6_unx
      integer(c_int), intent(in), value :: fort_v7_uny
      integer(c_int), intent(in), value :: fort_v8_unz
    end function
