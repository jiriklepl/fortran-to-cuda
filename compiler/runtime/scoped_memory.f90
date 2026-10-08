! Common interoperable interface; compile once before generated scoped modules.
module fort_scoped_memory
  use iso_c_binding
  implicit none
  private
  integer(c_int), parameter, public :: FORT_SCOPE_OK=0, FORT_SCOPE_ARGUMENT=1, FORT_SCOPE_STALE=2
  integer(c_int), parameter, public :: FORT_SCOPE_ALIAS=3, FORT_SCOPE_RESOURCE=4, FORT_SCOPE_BOUNDARY=5
  integer(c_int), parameter, public :: FORT_SCOPE_EXECUTION=6, FORT_SCOPE_UNINITIALIZED=7, FORT_SCOPE_STATE=8
  integer(c_int32_t), parameter, public :: FORT_SCOPE_READ_ALL=1, FORT_SCOPE_WRITE_ALL=2, FORT_SCOPE_OVERWRITE_ALL=4
  integer(c_int32_t), parameter, public :: FORT_SCOPE_BYTES=0, FORT_SCOPE_REAL32=1, FORT_SCOPE_REAL64=2
  integer(c_int32_t), parameter, public :: FORT_SCOPE_INTEGER32=3, FORT_SCOPE_LOGICAL=4
  integer(c_int), parameter, public :: FORT_SCOPE_NATIVE=0, FORT_SCOPE_GPU=1, FORT_SCOPE_AUTO=2

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

  public :: fort_scope_abi_version, fort_scope_error, fort_scope_create, fort_scope_register
  public :: fort_scope_register_sections, fort_scope_forget_definition
  public :: fort_scope_set_device_budget
  public :: fort_scope_serial_caller
  public :: fort_scope_layout_get, fort_scope_host_begin, fort_scope_host_end
  public :: fort_scope_device_begin, fort_scope_device_end, fort_scope_cancel_access
  public :: fort_scope_gpu_enter, fort_scope_gpu_leave, fort_scope_note_launch, fort_scope_wait
  public :: fort_scope_execution_error, fort_scope_report_error
  public :: fort_scope_stats_get, fort_scope_unregister, fort_scope_close, fort_scope_abandon

  interface
    function fort_scope_serial_caller() bind(C) result(serial)
      import c_int
      integer(c_int) :: serial
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
    function fort_scope_set_device_budget(context, bytes) bind(C) result(status)
      import c_int, c_int64_t, c_size_t
      integer(c_int64_t), value :: context
      integer(c_size_t), value :: bytes
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
    function fort_scope_layout_get(context, buffer, layout) bind(C) result(status)
      import c_int, c_int64_t, fort_scope_layout
      integer(c_int64_t), value :: context, buffer
      type(fort_scope_layout), intent(out) :: layout
      integer(c_int) :: status
    end function
    function fort_scope_host_begin(context, buffer, access) bind(C) result(status)
      import c_int, c_int64_t, fort_scope_access
      integer(c_int64_t), value :: context, buffer
      type(fort_scope_access), intent(in) :: access
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
