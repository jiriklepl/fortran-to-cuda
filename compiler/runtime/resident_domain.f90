! Caller-owned storage epochs over the authoritative scoped runtime.
! This table caches allocation handles, never replica validity or completion.
module fort_resident_domain
  use, intrinsic :: iso_c_binding
  use fort_scoped_memory
  implicit none
  private
  public :: resident_domain, resident_require

  type :: registered_storage
    integer(c_int64_t) :: identity=0, handle=0
    integer(c_int) :: element_type=0
    integer(c_size_t) :: element_bytes=0
    integer(c_size_t), allocatable :: extents(:)
    logical :: persistent=.false.
  end type

  ! Compiler-owned temporaries. The domain owns their backing allocation until
  ! close; a source invocation borrows a slot, not its storage lifetime. Replica
  ! validity and queued users remain exclusively in the scoped runtime.
  type :: work_real64_3_slot
    integer(c_int64_t) :: site=0,handle=0
    real(c_double), pointer, contiguous :: values(:,:,:)=>null()
    logical :: borrowed=.false.
  end type

  type :: resident_domain
    integer(c_int64_t) :: domain=0, context=0, generation=0
    logical :: active=.false.
    type(registered_storage), allocatable, private :: storage(:)
    type(work_real64_3_slot), allocatable, private :: work3(:)
  contains
    procedure :: begin_epoch
    procedure :: register_storage
    procedure :: bind_real64
    procedure :: bind_integer32
    procedure :: observe
    procedure :: host_begin
    procedure :: host_end
    procedure :: host_box_real64
    procedure :: release_real64
    procedure :: acquire_work_real64_3
    procedure :: release_work_real64_3
    procedure :: publish
    procedure :: end_epoch
    procedure :: close
  end type
contains
  subroutine resident_require(status)
    integer(c_int), intent(in) :: status
    if (status /= FORT_SCOPE_OK) then
      write(*,*) 'Resident execution failed; source work will not be replayed. Status:', status
      error stop 'resident execution failure'
    endif
  end subroutine

  subroutine begin_epoch(self, device, budget)
    class(resident_domain), intent(inout) :: self
    integer(c_int), intent(in) :: device
    integer(c_size_t), intent(in) :: budget
    if (self%active) error stop 'nested storage epoch'
    if (self%domain == 0) then
      call resident_require(fort_scope_domain_create_v1(device,self%domain))
      call resident_require(fort_scope_create_in_domain_v1(self%domain,self%context))
      call resident_require(fort_scope_set_device_budget(self%context,budget))
      allocate(self%storage(0))
    endif
    if (self%generation == huge(self%generation)) error stop 'storage generation overflow'
    self%generation = self%generation + 1
    self%active = .true.
  end subroutine

  function register_storage(self, host, extents, lower, element_type, element_bytes, initialized, persistent) result(handle)
    class(resident_domain), intent(inout) :: self
    type(c_ptr), intent(in) :: host
    integer(c_size_t), target, intent(in) :: extents(:)
    integer(c_int64_t), target, intent(in) :: lower(:)
    integer(c_int), intent(in) :: element_type
    integer(c_size_t), intent(in) :: element_bytes
    logical, intent(in) :: initialized, persistent
    integer(c_int64_t) :: handle, allocation_id, generation
    type(fort_scope_layout) :: layout
    type(registered_storage) :: item
    integer :: i
    if (.not. self%active) error stop 'registration outside a storage epoch'
    if (size(extents) /= size(lower) .or. any(extents == 0) .or. .not. c_associated(host)) &
      error stop 'invalid or empty resident allocation'
    allocation_id = transfer(host,allocation_id)
    do i=1,size(self%storage)
      if (self%storage(i)%identity /= allocation_id) cycle
      if (self%storage(i)%element_type /= element_type .or. self%storage(i)%element_bytes /= element_bytes .or. &
          (self%storage(i)%persistent .neqv. persistent)) error stop 'storage contract changed inside an epoch'
      if (size(self%storage(i)%extents) /= size(extents)) error stop 'storage rank changed inside an epoch'
      if (any(self%storage(i)%extents /= extents)) error stop 'storage shape changed inside an epoch'
      ! Complete-array dummy rebasing does not change allocation identity.
      ! Numerical entries pass their own logical origins as scalar arguments;
      ! external tasks validate the canonical layout established by the owner.
      handle = self%storage(i)%handle
      return
    enddo
    layout%rank=size(extents); layout%element_type=element_type; layout%element_bytes=element_bytes
    layout%host=host; layout%extents=c_loc(extents); layout%lower_bounds=c_loc(lower)
    generation=self%generation
    if (persistent) generation=1
    call resident_require(fort_scope_register(self%context,allocation_id,generation,layout, &
                                             merge(1_c_int,0_c_int,initialized),handle))
    item%identity=allocation_id; item%handle=handle; item%element_type=element_type
    item%element_bytes=element_bytes; item%extents=extents; item%persistent=persistent
    self%storage=[self%storage,item]
  end function

  function bind_real64(self, values, lower, initialized, persistent) result(handle)
    class(resident_domain), intent(inout) :: self
    real(c_double), contiguous, target, intent(in) :: values(..)
    integer, intent(in) :: lower(:)
    logical, intent(in) :: initialized
    logical, optional, intent(in) :: persistent
    integer(c_int64_t) :: handle
    logical :: saved
    saved=.false.
    if (present(persistent)) saved=persistent
    handle=self%register_storage(c_loc(values),shape(values,kind=c_size_t),int(lower,c_int64_t), &
                                 FORT_SCOPE_REAL64,8_c_size_t,initialized,saved)
  end function

  function bind_integer32(self, values, lower, initialized, persistent) result(handle)
    class(resident_domain), intent(inout) :: self
    integer(c_int), contiguous, target, intent(in) :: values(..)
    integer, intent(in) :: lower(:)
    logical, intent(in) :: initialized
    logical, optional, intent(in) :: persistent
    integer(c_int64_t) :: handle
    logical :: saved
    saved=.false.
    if (present(persistent)) saved=persistent
    handle=self%register_storage(c_loc(values),shape(values,kind=c_size_t),int(lower,c_int64_t), &
                                 FORT_SCOPE_INTEGER32,4_c_size_t,initialized,saved)
  end function

  subroutine observe(self, handles)
    class(resident_domain), intent(inout) :: self
    integer(c_int64_t), target, intent(in) :: handles(:)
    if (size(handles)==0) return
    call resident_require(fort_scope_observe_v1(self%context,c_loc(handles),size(handles,kind=c_size_t)))
  end subroutine

  subroutine host_begin(self, handle, access)
    class(resident_domain), intent(inout) :: self
    integer(c_int64_t), intent(in) :: handle
    type(fort_scope_access), intent(in) :: access
    call resident_require(fort_scope_host_begin(self%context,handle,access))
  end subroutine

  subroutine host_end(self, handle)
    class(resident_domain), intent(inout) :: self
    integer(c_int64_t), intent(in) :: handle
    call resident_require(fort_scope_host_end(self%context,handle))
  end subroutine

  function host_box_real64(self, values, lower, first, last, writing) result(handle)
    class(resident_domain), intent(inout) :: self
    real(c_double), contiguous, target, intent(in) :: values(:,:,:)
    integer, intent(in) :: lower(3), first(3), last(3)
    logical, intent(in) :: writing
    integer(c_int64_t) :: handle
    integer(c_size_t), target :: lo(3), hi(3)
    type(fort_scope_section), target :: section
    type(fort_scope_access) :: access
    if (any(first>last) .or. any(last<lower) .or. any(first>lower+shape(values)-1)) &
      error stop 'invalid resident native section'
    handle=self%bind_real64(values,lower,.true.)
    lo=int(max(first,lower)-lower,c_size_t)
    hi=int(min(last,lower+shape(values)-1)-lower+1,c_size_t)
    section%lower=c_loc(lo); section%upper=c_loc(hi)
    access%read_count=1; access%reads=c_loc(section)
    if (writing) then
      access%write_count=1; access%writes=c_loc(section)
    endif
    call self%host_begin(handle,access)
  end function

  subroutine release_real64(self, values)
    class(resident_domain), intent(inout) :: self
    real(c_double), contiguous, target, intent(in) :: values(..)
    integer(c_int64_t) :: allocation_id
    logical, allocatable :: keep(:)
    integer :: i
    if (.not. self%active) return
    allocation_id=transfer(c_loc(values),allocation_id)
    do i=1,size(self%storage)
      if (self%storage(i)%identity /= allocation_id) cycle
      if (self%storage(i)%persistent) error stop 'persistent storage retired through a borrowed view'
      ! The owner proves the value dead (temporary storage, not a publication).
      ! Retire waits for queued users without downloading the dead result.
      call resident_require(fort_scope_retire_v1(self%context,self%storage(i)%handle))
      allocate(keep(size(self%storage))); keep=.true.; keep(i)=.false.
      self%storage=pack(self%storage,keep)
      return
    enddo
  end subroutine

  subroutine acquire_work_real64_3(self,site,lower,upper,values)
    class(resident_domain), intent(inout) :: self
    integer(c_int64_t), intent(in) :: site
    integer, intent(in) :: lower(3),upper(3)
    real(c_double), pointer, contiguous, intent(out) :: values(:,:,:)
    type(work_real64_3_slot) :: slot
    integer :: i
    if (.not.self%active.or.site==0) error stop 'workspace outside a checked epoch'
    if (any(upper<lower)) error stop 'empty workspace must be skipped by its reached caller'
    if (.not.allocated(self%work3)) allocate(self%work3(0))
    do i=1,size(self%work3)
      if (self%work3(i)%site/=site) cycle
      if (any(lbound(self%work3(i)%values)/=lower).or.any(ubound(self%work3(i)%values)/=upper)) cycle
      if (self%work3(i)%borrowed) error stop 'overlapping use of a compiler workspace slot'
      ! An earlier invocation may still read this allocation on the device.
      ! Discard submits those users before forgetting dead values. It neither
      ! waits for kernels nor frees their storage; later writes share the stream.
      call resident_require(fort_scope_discard_v1(self%context,self%work3(i)%handle))
      self%work3(i)%borrowed=.true.
      values=>self%work3(i)%values
      return
    enddo
    slot%site=site
    allocate(slot%values(lower(1):upper(1),lower(2):upper(2),lower(3):upper(3)))
    slot%handle=self%bind_real64(slot%values,lower,.false.,persistent=.true.)
    slot%borrowed=.true.
    self%work3=[self%work3,slot]
    values=>slot%values
  end subroutine

  subroutine release_work_real64_3(self,site,values)
    class(resident_domain), intent(inout) :: self
    integer(c_int64_t), intent(in) :: site
    real(c_double), pointer, contiguous, intent(inout) :: values(:,:,:)
    integer :: i
    if (.not.self%active.or..not.associated(values).or..not.allocated(self%work3)) &
      error stop 'invalid compiler workspace release'
    do i=1,size(self%work3)
      if (self%work3(i)%site/=site) cycle
      if (.not.associated(values,self%work3(i)%values)) cycle
      if (.not.self%work3(i)%borrowed) error stop 'compiler workspace released twice'
      self%work3(i)%borrowed=.false.
      nullify(values)
      return
    enddo
    error stop 'foreign compiler workspace release'
  end subroutine

  subroutine publish(self)
    class(resident_domain), intent(inout) :: self
    integer(c_int64_t), allocatable :: handles(:)
    integer :: i
    if (.not. allocated(self%storage)) return
    handles=pack(self%storage%handle,.not. self%storage%persistent)
    call self%observe(handles)
    do i=1,size(handles)
      call resident_require(fort_scope_unregister(self%context,handles(i)))
    enddo
    self%storage=pack(self%storage,self%storage%persistent)
  end subroutine

  subroutine end_epoch(self)
    class(resident_domain), intent(inout) :: self
    if (allocated(self%work3)) then
      if (any(self%work3%borrowed)) error stop 'unmanaged return with borrowed workspace'
    endif
    call self%publish()
    self%active=.false.
  end subroutine

  subroutine close(self)
    class(resident_domain), intent(inout) :: self
    integer :: i
    if (self%domain == 0) return
    call self%end_epoch()
    if (allocated(self%work3)) then
      do i=1,size(self%work3)
        ! Private work values are dead, including any pending final writes.
        ! Domain close then waits for their queued users before freeing storage.
        call resident_require(fort_scope_discard_v1(self%context,self%work3(i)%handle))
      enddo
    endif
    call resident_require(fort_scope_close(self%context))
    call resident_require(fort_scope_domain_close_v1(self%domain))
    if (allocated(self%work3)) then
      do i=1,size(self%work3)
        deallocate(self%work3(i)%values)
      enddo
      deallocate(self%work3)
    endif
    self%context=0; self%domain=0
    if (allocated(self%storage)) deallocate(self%storage)
  end subroutine
end module
