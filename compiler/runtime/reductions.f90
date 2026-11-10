module fort_resident_reductions
  use, intrinsic :: iso_c_binding
  use fort_scoped_memory, only: fort_scope_external_report_v1
  implicit none
  private
  public :: fort_scope_max_real64_v1
  interface
    function fort_scope_max_real64_v1(context,input,rank,elements,initial,maximum,any_nan,report) bind(C) result(status)
      import :: c_int,c_int32_t,c_int64_t,c_size_t,c_double,fort_scope_external_report_v1
      integer(c_int64_t), value :: context,input
      integer(c_int32_t), value :: rank
      integer(c_size_t), value :: elements
      real(c_double), value :: initial
      real(c_double), intent(out) :: maximum
      integer(c_int32_t), intent(out) :: any_nan
      type(fort_scope_external_report_v1), intent(out) :: report
      integer(c_int) :: status
    end function
  end interface
end module
