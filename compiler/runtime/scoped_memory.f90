! Common interoperable interface; compile once before generated scoped modules.
module fort_scoped_memory
  use iso_c_binding
  implicit none
  private
  integer(c_int), parameter, public :: FORT_SCOPE_OK=0, FORT_SCOPE_ARGUMENT=1, FORT_SCOPE_STALE=2
  integer(c_int), parameter, public :: FORT_SCOPE_ALIAS=3, FORT_SCOPE_RESOURCE=4, FORT_SCOPE_BOUNDARY=5
  integer(c_int), parameter, public :: FORT_SCOPE_EXECUTION=6, FORT_SCOPE_UNINITIALIZED=7, FORT_SCOPE_STATE=8
  ! An entry did nothing because its scalars have values it was not compiled for; the caller runs the original statements.
  integer(c_int), parameter, public :: FORT_SCOPE_DECLINED=9
  integer(c_int32_t), parameter, public :: FORT_SCOPE_READ_ALL=1, FORT_SCOPE_WRITE_ALL=2, FORT_SCOPE_OVERWRITE_ALL=4
  integer(c_int32_t), parameter, public :: FORT_SCOPE_BYTES=0, FORT_SCOPE_REAL32=1, FORT_SCOPE_REAL64=2
  integer(c_int32_t), parameter, public :: FORT_SCOPE_INTEGER32=3, FORT_SCOPE_LOGICAL=4
  integer(c_int), parameter, public :: FORT_SCOPE_NATIVE=0, FORT_SCOPE_GPU=1
  integer(c_int32_t), parameter, public :: FORT_SCOPE_SCRATCH_ABI_VERSION=1
  integer(c_int32_t), parameter, public :: FORT_SCOPE_TRACE_ABI_VERSION=1
  integer(c_int32_t), parameter, public :: FORT_SCOPE_TRACE_OWNER=1, FORT_SCOPE_TRACE_PROCEDURE=2
  integer(c_int32_t), parameter, public :: FORT_SCOPE_TRACE_SEGMENT=4, FORT_SCOPE_TRACE_OPERATION=8
  integer(c_int32_t), parameter, public :: FORT_SCOPE_TRACE_IMPLEMENTATION=16, FORT_SCOPE_TRACE_BOUNDARY=32
  integer(c_int32_t), parameter, public :: FORT_SCOPE_TRACE_ALL=63


  type, bind(C), public :: fort_scope_trace_state_v1
    integer(c_int32_t) :: version=0, reserved=0
    character(kind=c_char) :: owner(65), procedure(65), segment(65), operation(65), implementation(65), boundary(65)
  end type
  type, bind(C), public :: fort_scope_section
    type(c_ptr) :: lower=c_null_ptr, upper=c_null_ptr
  end type
  type, bind(C), public :: fort_scope_access
    integer(c_int32_t) :: flags=0
    integer(c_size_t) :: read_count=0
    type(c_ptr) :: reads=c_null_ptr
    integer(c_size_t) :: write_count=0
    type(c_ptr) :: writes=c_null_ptr
    integer(c_size_t) :: overwrite_count=0
    type(c_ptr) :: overwrites=c_null_ptr
  end type
  type, bind(C), public :: fort_scope_external_binding_v1
    integer(c_int64_t) :: buffer=0
    integer(c_int32_t) :: element_type=0, rank=0
    integer(c_size_t) :: element_bytes=0
    type(fort_scope_access) :: access
  end type
  type, bind(C), public :: fort_scope_external_view_v1
    integer(c_int64_t) :: buffer=0, generation=0
    integer(c_int32_t) :: element_type=0, rank=0
    integer(c_size_t) :: element_bytes=0, bytes=0
    type(c_ptr) :: extents=c_null_ptr, strides=c_null_ptr, lower_bounds=c_null_ptr, device=c_null_ptr
  end type
  type, bind(C), public :: fort_scope_external_call_v1
    integer(c_int32_t) :: version=1, reserved=0
    type(c_ptr) :: bindings=c_null_ptr
    integer(c_size_t) :: binding_count=0
    integer(c_int64_t) :: workspace_lease=0
    integer(c_size_t) :: workspace_bytes=0
  end type
  type, bind(C), public :: fort_scope_external_report_v1
    integer(c_int32_t) :: version=1, status=0, started=0, committed=0
  end type
  type, bind(C), public :: fort_scope_layout
    integer(c_int32_t) :: rank=0, element_type=0
    integer(c_size_t) :: element_bytes=0
    type(c_ptr) :: host=c_null_ptr, extents=c_null_ptr, lower_bounds=c_null_ptr
    integer(c_int64_t) :: generation=0
  end type
  type, bind(C), public :: fort_scope_stats
    integer(c_int64_t) :: uploads=0, downloads=0, upload_bytes=0, download_bytes=0
    integer(c_int64_t) :: allocations=0, allocated_bytes=0, peak_device_bytes=0
    integer(c_int64_t) :: launches=0, waits=0, reconciliations=0
  end type
  type, bind(C), public :: fort_scope_scratch_lease_v1
    integer(c_int32_t) :: version=0, reserved=0
    integer(c_int64_t) :: token=0
    type(c_ptr) :: device=c_null_ptr
    integer(c_size_t) :: bytes=0, capacity=0
  end type
  type, bind(C), public :: fort_scope_scratch_stats_v1
    integer(c_int32_t) :: version=1, active=0
    integer(c_int64_t) :: acquisitions=0, allocations=0, reuses=0, releases=0, grows=0
    integer(c_int64_t) :: capacity_bytes=0, active_bytes=0, peak_scratch_bytes=0, peak_total_device_bytes=0
  end type

  type, bind(C), public :: fort_scope_plan_binding
    integer(c_int64_t) :: buffer=0
    type(fort_scope_access) :: access
  end type

  public :: fort_scope_abi_version, fort_scope_error, fort_scope_create, fort_scope_register
  public :: fort_scope_domain_create_v1, fort_scope_create_in_domain_v1, fort_scope_domain_get_v1
  public :: fort_scope_domain_close_v1, fort_scope_domain_abandon_v1
  public :: fort_scope_domain_stats_get_v1
  public :: fort_scope_domain_device_get_v1, fort_scope_identity_swap_v1
  public :: fort_scope_domain_workspace_reserve_v1, fort_scope_domain_workspace_resize_v1
  public :: fort_scope_domain_workspace_release_v1, fort_scope_external_execute_v1
  public :: fort_scope_trace_set_v1, fort_scope_trace_restore_v1
  public :: fort_scope_register_sections, fort_scope_forget_definition, fort_scope_discard_v1
  public :: fort_scope_recycle_v1
  public :: fort_scope_retire_v1
  public :: fort_scope_admit_source_reads_v1
  public :: fort_scope_set_device_budget
  public :: fort_scope_device_budget_default_v1, fort_scope_domain_budget_get_v1
  public :: fort_scope_scratch_acquire_v1, fort_scope_scratch_release_v1, fort_scope_scratch_stats_get_v1
  public :: fort_scope_layout_get, fort_scope_host_begin, fort_scope_host_end
  public :: fort_scope_access_begin_v1
  public :: fort_scope_original_host_begin_v1
  public :: fort_scope_observe_v1, fort_scope_observe_all_v1
  public :: fort_scope_buffer_valid_v1
  public :: fort_scope_device_begin, fort_scope_device_end, fort_scope_cancel_access
  public :: fort_scope_gpu_enter, fort_scope_gpu_leave, fort_scope_note_launch, fort_scope_wait
  public :: fort_scope_execution_error, fort_scope_report_error
  public :: fort_scope_stats_get, fort_scope_unregister, fort_scope_close, fort_scope_abandon


  interface
    subroutine fort_scope_trace_set_v1(context, update_mask, owner, procedure, segment, operation, implementation, boundary, &
        previous) &
        bind(C)
      import c_int64_t, c_int32_t, c_char, c_ptr
      integer(c_int64_t), value :: context
      integer(c_int32_t), value :: update_mask
      character(kind=c_char), intent(in) :: owner(*), procedure(*), segment(*), operation(*), implementation(*), boundary(*)
      type(c_ptr), value :: previous
    end subroutine
    subroutine fort_scope_trace_restore_v1(context, previous) bind(C)
      import c_int64_t, fort_scope_trace_state_v1
      integer(c_int64_t), value :: context
      type(fort_scope_trace_state_v1), intent(in) :: previous
    end subroutine

    function fort_scope_scratch_acquire_v1(context, bytes, lease) bind(C) result(status)
      import c_int, c_int64_t, c_size_t, fort_scope_scratch_lease_v1
      integer(c_int64_t), value :: context
      integer(c_size_t), value :: bytes
      type(fort_scope_scratch_lease_v1), intent(out) :: lease
      integer(c_int) :: status
    end function
    function fort_scope_scratch_release_v1(context, token) bind(C) result(status)
      import c_int, c_int64_t
      integer(c_int64_t), value :: context, token
      integer(c_int) :: status
    end function
    function fort_scope_scratch_stats_get_v1(context, stats) bind(C) result(status)
      import c_int, c_int64_t, fort_scope_scratch_stats_v1
      integer(c_int64_t), value :: context
      type(fort_scope_scratch_stats_v1), intent(out) :: stats
      integer(c_int) :: status
    end function


    function fort_scope_admit_source_reads_v1(context, buffer, sections, count) bind(C) result(status)
      import c_int, c_int64_t, c_ptr, c_size_t
      integer(c_int64_t), value :: context, buffer
      type(c_ptr), value :: sections
      integer(c_size_t), value :: count
      integer(c_int) :: status
    end function


    function fort_scope_abi_version() bind(C) result(version)
      import c_int32_t
      integer(c_int32_t) :: version
    end function
    function fort_scope_error() bind(C) result(message)
      import c_ptr
      type(c_ptr) :: message
    end function
    function fort_scope_create(device, context) bind(C) result(status)
      import c_int, c_int64_t
      integer(c_int), value :: device
      integer(c_int64_t), intent(out) :: context
      integer(c_int) :: status
    end function
    function fort_scope_domain_create_v1(device, domain) bind(C) result(status)
      import c_int, c_int64_t
      integer(c_int), value :: device
      integer(c_int64_t), intent(out) :: domain
      integer(c_int) :: status
    end function
    function fort_scope_create_in_domain_v1(domain, context) bind(C) result(status)
      import c_int, c_int64_t
      integer(c_int64_t), value :: domain
      integer(c_int64_t), intent(out) :: context
      integer(c_int) :: status
    end function
    function fort_scope_domain_get_v1(context, domain) bind(C) result(status)
      import c_int, c_int64_t
      integer(c_int64_t), value :: context
      integer(c_int64_t), intent(out) :: domain
      integer(c_int) :: status
    end function
    function fort_scope_domain_close_v1(domain) bind(C) result(status)
      import c_int, c_int64_t
      integer(c_int64_t), value :: domain
      integer(c_int) :: status
    end function
    function fort_scope_domain_abandon_v1(domain) bind(C) result(status)
      import c_int, c_int64_t
      integer(c_int64_t), value :: domain
      integer(c_int) :: status
    end function
    function fort_scope_domain_stats_get_v1(domain, stats) bind(C) result(status)
      import c_int, c_int64_t, fort_scope_stats
      integer(c_int64_t), value :: domain
      type(fort_scope_stats), intent(out) :: stats
      integer(c_int) :: status
    end function
    function fort_scope_domain_device_get_v1(domain, device) bind(C) result(status)
      import c_int, c_int64_t
      integer(c_int64_t), value :: domain
      integer(c_int), intent(out) :: device
      integer(c_int) :: status
    end function
    function fort_scope_identity_swap_v1(context, first, second, first_after, second_after, &
                                         first_current, second_current) bind(C) result(status)
      import c_int, c_int64_t, fort_scope_layout
      integer(c_int64_t), value :: context, first, second
      type(fort_scope_layout), intent(in) :: first_after, second_after
      integer(c_int64_t), intent(out) :: first_current, second_current
      integer(c_int) :: status
    end function
    function fort_scope_domain_workspace_reserve_v1(domain, bytes, lease) bind(C) result(status)
      import c_int, c_int64_t, c_size_t
      integer(c_int64_t), value :: domain
      integer(c_size_t), value :: bytes
      integer(c_int64_t), intent(out) :: lease
      integer(c_int) :: status
    end function
    function fort_scope_domain_workspace_resize_v1(domain, lease, bytes) bind(C) result(status)
      import c_int, c_int64_t, c_size_t
      integer(c_int64_t), value :: domain, lease
      integer(c_size_t), value :: bytes
      integer(c_int) :: status
    end function
    function fort_scope_domain_workspace_release_v1(domain, lease) bind(C) result(status)
      import c_int, c_int64_t
      integer(c_int64_t), value :: domain, lease
      integer(c_int) :: status
    end function
    function fort_scope_external_execute_v1(context, call, worker, user, report) bind(C) result(status)
      import c_int, c_int64_t, c_funptr, c_ptr, fort_scope_external_call_v1, fort_scope_external_report_v1
      integer(c_int64_t), value :: context
      type(fort_scope_external_call_v1), intent(in) :: call
      type(c_funptr), value :: worker
      type(c_ptr), value :: user
      type(fort_scope_external_report_v1), intent(out) :: report
      integer(c_int) :: status
    end function
    function fort_scope_set_device_budget(context, bytes) bind(C) result(status)
      import c_int, c_int64_t, c_size_t
      integer(c_int64_t), value :: context
      integer(c_size_t), value :: bytes
      integer(c_int) :: status
    end function
    function fort_scope_device_budget_default_v1(device, bytes) bind(C) result(status)
      import c_int, c_size_t
      integer(c_int), value :: device
      integer(c_size_t), intent(out) :: bytes
      integer(c_int) :: status
    end function
    function fort_scope_domain_budget_get_v1(domain, budget, unclaimed) bind(C) result(status)
      import c_int, c_int64_t, c_size_t
      integer(c_int64_t), value :: domain
      integer(c_size_t), intent(out) :: budget, unclaimed
      integer(c_int) :: status
    end function
    function fort_scope_register(context, identity, generation, layout, host_initialized, buffer) bind(C) result(status)
      import c_int, c_int64_t, fort_scope_layout
      integer(c_int64_t), value :: context, identity, generation
      type(fort_scope_layout), intent(in) :: layout
      integer(c_int), value :: host_initialized
      integer(c_int64_t), intent(out) :: buffer
      integer(c_int) :: status
    end function
    function fort_scope_register_sections(context, identity, generation, layout, initialized, count, buffer) &
        bind(C) result(status)
      import c_int, c_int64_t, c_size_t, c_ptr, fort_scope_layout
      integer(c_int64_t), value :: context, identity, generation
      type(fort_scope_layout), intent(in) :: layout
      type(c_ptr), value :: initialized
      integer(c_size_t), value :: count
      integer(c_int64_t), intent(out) :: buffer
      integer(c_int) :: status
    end function
    function fort_scope_forget_definition(context, buffer) bind(C) result(status)
      import c_int, c_int64_t
      integer(c_int64_t), value :: context, buffer
      integer(c_int) :: status
    end function
    function fort_scope_discard_v1(context, buffer) bind(C) result(status)
      import c_int, c_int64_t
      integer(c_int64_t), value :: context, buffer
      integer(c_int) :: status
    end function
    function fort_scope_recycle_v1(context, buffer) bind(C) result(status)
      import c_int, c_int64_t
      integer(c_int64_t), value :: context, buffer
      integer(c_int) :: status
    end function

    function fort_scope_retire_v1(context, buffer) bind(C) result(status)
      import :: c_int, c_int64_t
      integer(c_int64_t), value :: context, buffer
      integer(c_int) :: status
    end function
    function fort_scope_layout_get(context, buffer, layout) bind(C) result(status)
      import c_int, c_int64_t, fort_scope_layout
      integer(c_int64_t), value :: context, buffer
      type(fort_scope_layout), intent(out) :: layout
      integer(c_int) :: status
    end function
    function fort_scope_buffer_valid_v1(context, buffer, valid) bind(C) result(status)
      import c_int, c_int64_t
      integer(c_int64_t), value :: context, buffer
      integer(c_int), intent(out) :: valid
      integer(c_int) :: status
    end function
    function fort_scope_host_begin(context, buffer, access) bind(C) result(status)
      import c_int, c_int64_t, fort_scope_access
      integer(c_int64_t), value :: context, buffer
      type(fort_scope_access), intent(in) :: access
      integer(c_int) :: status
    end function
    function fort_scope_access_begin_v1(context, bindings, count, device, devices) bind(C) result(status)
      import c_int, c_int64_t, c_ptr, c_size_t
      integer(c_int64_t), value :: context
      type(c_ptr), value :: bindings, devices
      integer(c_size_t), value :: count
      integer(c_int), value :: device
      integer(c_int) :: status
    end function
    function fort_scope_original_host_begin_v1(context, bindings, count) bind(C) result(status)
      import c_int, c_int64_t, c_ptr, c_size_t
      integer(c_int64_t), value :: context
      type(c_ptr), value :: bindings
      integer(c_size_t), value :: count
      integer(c_int) :: status
    end function
    function fort_scope_observe_v1(context, buffers, count) bind(C) result(status)
      import c_int, c_int64_t, c_ptr, c_size_t
      integer(c_int64_t), value :: context
      type(c_ptr), value :: buffers
      integer(c_size_t), value :: count
      integer(c_int) :: status
    end function
    function fort_scope_observe_all_v1(context) bind(C) result(status)
      import c_int, c_int64_t
      integer(c_int64_t), value :: context
      integer(c_int) :: status
    end function
    function fort_scope_host_end(context, buffer) bind(C) result(status)
      import c_int, c_int64_t
      integer(c_int64_t), value :: context, buffer
      integer(c_int) :: status
    end function
    function fort_scope_device_begin(context, buffer, access, device) bind(C) result(status)
      import c_int, c_int64_t, c_ptr, fort_scope_access
      integer(c_int64_t), value :: context, buffer
      type(fort_scope_access), intent(in) :: access
      type(c_ptr), intent(out) :: device
      integer(c_int) :: status
    end function
    function fort_scope_device_end(context, buffer) bind(C) result(status)
      import c_int, c_int64_t
      integer(c_int64_t), value :: context, buffer
      integer(c_int) :: status
    end function
    function fort_scope_cancel_access(context, buffer) bind(C) result(status)
      import c_int, c_int64_t
      integer(c_int64_t), value :: context, buffer
      integer(c_int) :: status
    end function
    function fort_scope_gpu_enter(context, previous_device, stream) bind(C) result(status)
      import c_int, c_int64_t, c_ptr
      integer(c_int64_t), value :: context
      integer(c_int), intent(out) :: previous_device
      type(c_ptr), intent(out) :: stream
      integer(c_int) :: status
    end function
    function fort_scope_gpu_leave(context, previous_device) bind(C) result(status)
      import c_int, c_int64_t
      integer(c_int64_t), value :: context
      integer(c_int), value :: previous_device
      integer(c_int) :: status
    end function
    function fort_scope_note_launch(context) bind(C) result(status)
      import c_int, c_int64_t
      integer(c_int64_t), value :: context
      integer(c_int) :: status
    end function
    function fort_scope_execution_error(context, message) bind(C) result(status)
      import c_int, c_int64_t, c_ptr
      integer(c_int64_t), value :: context
      type(c_ptr), value :: message
      integer(c_int) :: status
    end function
    function fort_scope_report_error(error, message) bind(C) result(status)
      import c_int, c_ptr
      integer(c_int), value :: error
      type(c_ptr), value :: message
      integer(c_int) :: status
    end function
    function fort_scope_wait(context) bind(C) result(status)
      import c_int, c_int64_t
      integer(c_int64_t), value :: context
      integer(c_int) :: status
    end function
    function fort_scope_stats_get(context, stats) bind(C) result(status)
      import c_int, c_int64_t, fort_scope_stats
      integer(c_int64_t), value :: context
      type(fort_scope_stats), intent(out) :: stats
      integer(c_int) :: status
    end function
    function fort_scope_unregister(context, buffer) bind(C) result(status)
      import c_int, c_int64_t
      integer(c_int64_t), value :: context, buffer
      integer(c_int) :: status
    end function
    function fort_scope_close(context) bind(C) result(status)
      import c_int, c_int64_t
      integer(c_int64_t), value :: context
      integer(c_int) :: status
    end function
    function fort_scope_abandon(context) bind(C) result(status)
      import c_int, c_int64_t
      integer(c_int64_t), value :: context
      integer(c_int) :: status
    end function
  end interface
end module fort_scoped_memory
