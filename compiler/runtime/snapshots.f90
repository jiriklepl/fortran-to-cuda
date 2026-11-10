module fort_resident_snapshots
  use, intrinsic :: iso_c_binding
  implicit none
  private
  public :: fort_snapshot_create_v1, fort_snapshot_acquire_v1, fort_snapshot_cancel_v1
  public :: fort_snapshot_submit_v1, fort_snapshot_wait_v1, fort_snapshot_destroy_v1
  interface
    function fort_snapshot_create_v1(domain,bytes,pinned_budget,queue) bind(C) result(status)
      import
      integer(c_int64_t), value :: domain
      integer(c_size_t), value :: bytes,pinned_budget
      type(c_ptr), intent(out) :: queue
      integer(c_int) :: status
    end function
    function fort_snapshot_acquire_v1(queue,slot) bind(C) result(status)
      import
      type(c_ptr), value :: queue
      integer(c_int32_t), intent(out) :: slot
      integer(c_int) :: status
    end function
    function fort_snapshot_cancel_v1(queue,slot) bind(C) result(status)
      import
      type(c_ptr), value :: queue
      integer(c_int32_t), value :: slot
      integer(c_int) :: status
    end function
    function fort_snapshot_submit_v1(queue,slot,context,bindings,bytes,count,writer,user) bind(C) result(status)
      import
      type(c_ptr), value :: queue,bindings,bytes,user
      integer(c_int32_t), value :: slot
      integer(c_int64_t), value :: context
      integer(c_size_t), value :: count
      type(c_funptr), value :: writer
      integer(c_int) :: status
    end function
    function fort_snapshot_wait_v1(queue) bind(C) result(status)
      import
      type(c_ptr), value :: queue
      integer(c_int) :: status
    end function
    function fort_snapshot_destroy_v1(queue) bind(C) result(status)
      import
      type(c_ptr), value :: queue
      integer(c_int) :: status
    end function
  end interface
end module
