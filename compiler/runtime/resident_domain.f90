! Caller-owned storage epochs over the authoritative scoped runtime.
! This table caches allocation handles, never replica validity or completion.
module fort_resident_domain
  use, intrinsic :: iso_c_binding
  use, intrinsic :: iso_fortran_env, only: error_unit
  use fort_scoped_memory
  !$ use omp_lib, only: omp_get_max_threads, omp_set_num_threads
  implicit none
  private
  public :: resident_domain, resident_require

  type :: registered_storage
    integer(c_int64_t) :: identity=0, handle=0
    integer(c_int) :: element_type=0
    integer(c_size_t) :: element_bytes=0
    integer(c_size_t), allocatable :: extents(:)
    character(:), allocatable :: origin
    logical :: persistent=.false.
  end type

  ! Compiler-owned temporaries. The domain owns their backing allocation until
  ! close; a source invocation borrows a slot, not its storage lifetime. Replica
  ! validity and queued users remain exclusively in the scoped runtime.
  type :: work_real32_3_slot
    integer(c_int64_t) :: site=0,handle=0
    real(c_float), pointer, contiguous :: values(:,:,:)=>null()
    logical :: borrowed=.false.
  end type

  type :: work_real64_3_slot
    integer(c_int64_t) :: site=0,handle=0
    real(c_double), pointer, contiguous :: values(:,:,:)=>null()
    logical :: borrowed=.false.
  end type

  ! One numeric field of the elements of an array of derived type. Its values
  ! are strided through the structures, so the domain keeps a contiguous copy
  ! that the runtime manages like any bound array, and writes the copy back to
  ! the structures whenever the host needs them.
  type :: field_mirror
    integer(c_intptr_t) :: first=0
    integer(c_size_t) :: stride=0, count=0, bytes=0
    integer(c_int) :: element_type=0
    integer(c_int64_t) :: handle=0
    logical :: written=.false.
    integer(c_int32_t), pointer :: integers(:)=>null()
    real(c_float), pointer :: singles(:)=>null()
    real(c_double), pointer :: doubles(:)=>null()
  end type

  type :: resident_domain
    integer(c_int64_t) :: domain=0, context=0, generation=0
    ! Advances whenever host code may have changed bound values: at every
    ! publication and host access. A caller that checked something about index
    ! values can keep its answer while generation and stamp are unchanged.
    integer(c_int64_t) :: stamp=0
    logical :: active=.false.
    ! The team size to give back when the epoch ends, if the epoch narrowed it.
    integer, private :: outer_threads=0
    type(registered_storage), allocatable, private :: storage(:)
    type(field_mirror), allocatable, private :: mirrors(:)
    ! Blocks a field copy could not use because the memory is still registered
    ! for an array its owner has freed. They stay allocated until the next
    ! publication so that neither the allocator nor that registration's
    ! download can reach anything else through them.
    type(field_mirror), allocatable, private :: avoided(:)
    type(work_real32_3_slot), allocatable, private :: work3_f32(:)
    type(work_real64_3_slot), allocatable, private :: work3(:)
  contains
    procedure :: begin_epoch
    procedure :: register_storage
    procedure :: bind_real32
    procedure :: bind_real64
    generic :: bind_real => bind_real32, bind_real64
    procedure :: bind_integer32
    generic :: bind => bind_real32, bind_real64, bind_integer32
    procedure :: bind_field
    procedure :: observe_field
    procedure :: observe
    procedure :: host_begin
    procedure :: host_end
    procedure :: host_box_real32
    procedure :: host_box_real64
    generic :: host_box_real => host_box_real32, host_box_real64
    procedure :: release_real32
    procedure :: release_real64
    generic :: release_real => release_real32, release_real64
    procedure :: acquire_work_real32_3
    procedure :: acquire_work_real64_3
    generic :: acquire_work_real_3 => acquire_work_real32_3, acquire_work_real64_3
    procedure :: release_work_real32_3
    procedure :: release_work_real64_3
    generic :: release_work_real_3 => release_work_real32_3, release_work_real64_3
    procedure :: publish
    procedure :: end_epoch
    procedure :: close
  end type
contains
  subroutine resident_require(status)
    integer(c_int), intent(in) :: status
    character(len=16) :: name
    if (status /= FORT_SCOPE_OK) then
      select case(status)
      case(FORT_SCOPE_ARGUMENT)
        name='ARGUMENT'
      case(FORT_SCOPE_STALE)
        name='STALE'
      case(FORT_SCOPE_ALIAS)
        name='ALIAS'
      case(FORT_SCOPE_RESOURCE)
        name='RESOURCE'
      case(FORT_SCOPE_BOUNDARY)
        name='BOUNDARY'
      case(FORT_SCOPE_EXECUTION)
        name='EXECUTION'
      case(FORT_SCOPE_UNINITIALIZED)
        name='UNINITIALIZED'
      case(FORT_SCOPE_STATE)
        name='STATE'
      case(FORT_SCOPE_PLAN_MISMATCH)
        name='PLAN_MISMATCH'
      case default
        name='UNKNOWN'
      end select
      write(error_unit,'(a,a,a,i0,a)') 'Resident runtime failed: ',trim(name),' (status ',status, &
                                      '); source work will not be replayed.'
      error stop 'resident execution failure'
    endif
  end subroutine

  ! keep_team is for an epoch during which the host still does its parallel
  ! work: such an epoch leaves the OpenMP team as it is (see narrow_team).
  subroutine begin_epoch(self, device, budget, keep_team)
    class(resident_domain), intent(inout) :: self
    integer(c_int), intent(in) :: device
    integer(c_size_t), intent(in) :: budget
    logical, optional, intent(in) :: keep_team
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
    if (present(keep_team)) then
      if (keep_team) return
    endif
    call narrow_team(self)
  end subroutine

  ! While values stay on the device the host runs serial code between kernels,
  ! and each parallel region it still enters wakes the whole team. The epoch
  ! therefore narrows the team to FORT_RESIDENT_HOST_THREADS, when that is set,
  ! and gives the application its own count back when it ends. The count the
  ! application starts with is not the place to save: it can decide results,
  ! as a parallel random initialization does.
  subroutine narrow_team(self)
    class(resident_domain), intent(inout) :: self
    character(len=16) :: text
    integer :: wanted, status
    call get_environment_variable('FORT_RESIDENT_HOST_THREADS', text, status=status)
    if (status /= 0) return
    read(text, *, iostat=status) wanted
    if (status /= 0 .or. wanted < 1) error stop 'FORT_RESIDENT_HOST_THREADS must be a positive integer'
    !$ if (wanted < omp_get_max_threads()) then
    !$   self%outer_threads = omp_get_max_threads()
    !$   call omp_set_num_threads(wanted)
    !$ endif
  end subroutine

  function storage_type_name(element_type) result(name)
    integer(c_int), intent(in) :: element_type
    character(len=12) :: name
    select case(element_type)
    case(FORT_SCOPE_BYTES)
      name='bytes'
    case(FORT_SCOPE_REAL32)
      name='real32'
    case(FORT_SCOPE_REAL64)
      name='real64'
    case(FORT_SCOPE_INTEGER32)
      name='integer32'
    case(FORT_SCOPE_LOGICAL)
      name='logical'
    case default
      name='unknown'
    end select
  end function

  subroutine report_storage_conflict(self, item, extents, element_type, element_bytes, persistent, origin)
    class(resident_domain), intent(in) :: self
    type(registered_storage), intent(in) :: item
    integer(c_size_t), intent(in) :: extents(:), element_bytes
    integer(c_int), intent(in) :: element_type
    logical, intent(in) :: persistent
    character(*), optional, intent(in) :: origin
    character(:), allocatable :: requested_origin
    requested_origin='unknown'
    if (present(origin)) requested_origin=origin
    write(error_unit,'(a,i0,a,i0,a,i0)') 'Resident storage conflict: address=',item%identity, &
                                         ' domain=',self%domain,' epoch=',self%generation
    write(error_unit,*) 'Registered: origin=',item%origin,' type=',trim(storage_type_name(item%element_type)), &
                        ' bytes=',item%element_bytes,' persistent=',item%persistent,' extents=',item%extents
    write(error_unit,*) 'Requested: origin=',requested_origin,' type=',trim(storage_type_name(element_type)), &
                        ' bytes=',element_bytes,' persistent=',persistent,' extents=',extents
  end subroutine

  ! With collision present, host is a block the caller has just allocated for
  ! itself: storage already registered at or across it belongs to an array that
  ! no longer exists. The function then returns no handle and sets collision,
  ! so that the caller can take another block.
  function register_storage(self, host, extents, lower, element_type, element_bytes, initialized, persistent, origin, &
                            collision) result(handle)
    class(resident_domain), intent(inout) :: self
    type(c_ptr), intent(in) :: host
    integer(c_size_t), target, intent(in) :: extents(:)
    integer(c_int64_t), target, intent(in) :: lower(:)
    integer(c_int), intent(in) :: element_type
    integer(c_size_t), intent(in) :: element_bytes
    logical, intent(in) :: initialized, persistent
    character(*), optional, intent(in) :: origin
    logical, optional, intent(out) :: collision
    integer(c_int64_t) :: handle, allocation_id, generation
    type(fort_scope_layout) :: layout
    type(registered_storage) :: item
    integer(c_int) :: status
    integer :: i
    handle=0
    if (present(collision)) collision=.false.
    if (.not. self%active) error stop 'registration outside a storage epoch'
    if (size(extents) /= size(lower) .or. any(extents == 0) .or. .not. c_associated(host)) &
      error stop 'invalid or empty resident allocation'
    allocation_id = transfer(host,allocation_id)
    do i=1,size(self%storage)
      if (self%storage(i)%identity /= allocation_id) cycle
      if (present(collision)) then
        collision=.true.
        return
      endif
      if (self%storage(i)%element_type /= element_type .or. self%storage(i)%element_bytes /= element_bytes .or. &
          (self%storage(i)%persistent .neqv. persistent)) then
        call report_storage_conflict(self,self%storage(i),extents,element_type,element_bytes,persistent,origin)
        error stop 'storage contract changed inside an epoch'
      endif
      if (size(self%storage(i)%extents) /= size(extents)) then
        call report_storage_conflict(self,self%storage(i),extents,element_type,element_bytes,persistent,origin)
        error stop 'storage rank changed inside an epoch'
      endif
      if (any(self%storage(i)%extents /= extents)) then
        call report_storage_conflict(self,self%storage(i),extents,element_type,element_bytes,persistent,origin)
        error stop 'storage shape changed inside an epoch'
      endif
      ! Complete-array dummy rebasing does not change allocation identity.
      ! Numerical entries pass their own logical origins as scalar arguments;
      ! external tasks validate the canonical layout established by the owner.
      if (present(origin)) then
        if (self%storage(i)%origin=='unknown') self%storage(i)%origin=origin
      endif
      handle = self%storage(i)%handle
      return
    enddo
    layout%rank=size(extents); layout%element_type=element_type; layout%element_bytes=element_bytes
    layout%host=host; layout%extents=c_loc(extents); layout%lower_bounds=c_loc(lower)
    generation=self%generation
    if (persistent) generation=1
    status=fort_scope_register(self%context,allocation_id,generation,layout, &
                               merge(1_c_int,0_c_int,initialized),handle)
    if (status == FORT_SCOPE_ALIAS) then
      if (present(collision)) then
        collision=.true.
        handle=0
        return
      endif
      call report_overlap(self,allocation_id,product(extents)*element_bytes,origin)
    endif
    call resident_require(status)
    item%identity=allocation_id; item%handle=handle; item%element_type=element_type
    item%element_bytes=element_bytes; item%extents=extents; item%persistent=persistent
    item%origin='unknown'
    if (present(origin)) item%origin=origin
    self%storage=[self%storage,item]
  end function

  ! Name the registered storage that overlaps count bytes at address. The
  ! runtime refuses two registrations that share host memory; without the
  ! names the refusal cannot be traced to its array.
  subroutine report_overlap(self, address, count, origin)
    class(resident_domain), intent(in) :: self
    integer(c_int64_t), intent(in) :: address
    integer(c_size_t), intent(in) :: count
    character(*), optional, intent(in) :: origin
    character(:), allocatable :: requested
    integer(c_int64_t) :: other, bytes
    integer :: i, found
    requested='unknown'
    if (present(origin)) requested=origin
    write(error_unit,'(a,a,a,i0,a,i0,a,i0,a,i0)') 'Resident storage overlap: origin=',requested,' address=',address, &
      ' bytes=',count,' domain=',self%domain,' epoch=',self%generation
    found=0
    do i=1,size(self%storage)
      other=self%storage(i)%identity
      bytes=int(product(self%storage(i)%extents)*self%storage(i)%element_bytes,c_int64_t)
      if (address+int(count,c_int64_t) <= other .or. other+bytes <= address) cycle
      found=found+1
      write(error_unit,'(a,a,a,i0,a,i0,a,l1)') '  registered: origin=',self%storage(i)%origin,' address=',other, &
        ' bytes=',bytes,' persistent=',self%storage(i)%persistent
    enddo
    if (found==0) write(error_unit,'(a)') '  registered: by storage this domain has no record of'
  end subroutine

  function bind_real32(self, values, lower, initialized, persistent, origin) result(handle)
    class(resident_domain), intent(inout) :: self
    real(c_float), contiguous, target, intent(in) :: values(..)
    integer, intent(in) :: lower(:)
    logical, intent(in) :: initialized
    logical, optional, intent(in) :: persistent
    character(*), optional, intent(in) :: origin
    integer(c_int64_t) :: handle
    logical :: saved
    saved=.false.
    if (present(persistent)) saved=persistent
    handle=self%register_storage(c_loc(values),shape(values,kind=c_size_t),int(lower,c_int64_t), &
                                 FORT_SCOPE_REAL32,4_c_size_t,initialized,saved,origin)
  end function

  function bind_real64(self, values, lower, initialized, persistent, origin) result(handle)
    class(resident_domain), intent(inout) :: self
    real(c_double), contiguous, target, intent(in) :: values(..)
    integer, intent(in) :: lower(:)
    logical, intent(in) :: initialized
    logical, optional, intent(in) :: persistent
    character(*), optional, intent(in) :: origin
    integer(c_int64_t) :: handle
    logical :: saved
    saved=.false.
    if (present(persistent)) saved=persistent
    handle=self%register_storage(c_loc(values),shape(values,kind=c_size_t),int(lower,c_int64_t), &
                                 FORT_SCOPE_REAL64,8_c_size_t,initialized,saved,origin)
  end function

  function bind_integer32(self, values, lower, initialized, persistent, origin) result(handle)
    class(resident_domain), intent(inout) :: self
    integer(c_int), contiguous, target, intent(in) :: values(..)
    integer, intent(in) :: lower(:)
    logical, intent(in) :: initialized
    logical, optional, intent(in) :: persistent
    character(*), optional, intent(in) :: origin
    integer(c_int64_t) :: handle
    logical :: saved
    saved=.false.
    if (present(persistent)) saved=persistent
    handle=self%register_storage(c_loc(values),shape(values,kind=c_size_t),int(lower,c_int64_t), &
                                 FORT_SCOPE_INTEGER32,4_c_size_t,initialized,saved,origin)
  end function

  subroutine observe(self, handles)
    class(resident_domain), intent(inout) :: self
    integer(c_int64_t), target, intent(in) :: handles(:)
    integer :: i
    if (size(handles)==0) return
    call resident_require(fort_scope_observe_v1(self%context,c_loc(handles),size(handles,kind=c_size_t)))
    if (.not. allocated(self%mirrors)) return
    do i=1,size(self%mirrors)
      ! A field the device wrote reaches its structures through the copy.
      if (self%mirrors(i)%written .and. any(handles==self%mirrors(i)%handle)) &
        call copy_field(self%mirrors(i),.false.)
    enddo
  end subroutine

  ! Bring one field of an array of derived type back into its structures and
  ! keep its device copy. first is the address of the field in the first
  ! element. A field that is not bound is already current on the host.
  subroutine observe_field(self, first)
    class(resident_domain), intent(inout) :: self
    type(c_ptr), intent(in) :: first
    integer :: i
    if (.not. allocated(self%mirrors)) return
    do i=1,size(self%mirrors)
      if (self%mirrors(i)%first==transfer(first,self%mirrors(i)%first)) call self%observe([self%mirrors(i)%handle])
    enddo
  end subroutine

  function mirror_address(mirror) result(address)
    type(field_mirror), intent(in) :: mirror
    type(c_ptr) :: address
    select case(mirror%element_type)
    case(FORT_SCOPE_INTEGER32)
      address=c_loc(mirror%integers)
    case(FORT_SCOPE_REAL32)
      address=c_loc(mirror%singles)
    case default
      address=c_loc(mirror%doubles)
    end select
  end function

  subroutine copy_field(mirror, gather)
    type(field_mirror), intent(in) :: mirror
    logical, intent(in) :: gather
    integer(c_int8_t), pointer :: fields(:), values(:)
    integer(c_size_t) :: i
    call c_f_pointer(transfer(mirror%first,c_null_ptr),fields,[(mirror%count-1)*mirror%stride+mirror%bytes])
    call c_f_pointer(mirror_address(mirror),values,[mirror%count*mirror%bytes])
    do i=0,mirror%count-1
      if (gather) then
        values(i*mirror%bytes+1:(i+1)*mirror%bytes)=fields(i*mirror%stride+1:i*mirror%stride+mirror%bytes)
      else
        fields(i*mirror%stride+1:i*mirror%stride+mirror%bytes)=values(i*mirror%bytes+1:(i+1)*mirror%bytes)
      endif
    enddo
  end subroutine

  ! Bind one field of every element of a rank-one array of derived type.
  ! first is the address of the field in the first element, stride the size of
  ! one element in bytes. The values are copied from the structures when the
  ! field is first bound and stay bound until the next publication.
  function bind_field(self, first, stride, count, element_type, writing, origin) result(handle)
    class(resident_domain), intent(inout) :: self
    type(c_ptr), intent(in) :: first
    integer(c_size_t), intent(in) :: stride, count
    integer(c_int), intent(in) :: element_type
    logical, intent(in) :: writing
    character(*), optional, intent(in) :: origin
    integer(c_int64_t) :: handle
    type(field_mirror) :: item
    ! Memory one binding may set aside while it looks for a block no registration covers.
    integer(c_size_t), parameter :: avoided_limit=1073741824_c_size_t
    integer(c_size_t) :: set_aside
    logical :: collision
    integer :: i
    if (.not. self%active) error stop 'field binding outside a storage epoch'
    if (count==0 .or. .not. c_associated(first)) error stop 'invalid or empty resident field'
    if (.not. allocated(self%mirrors)) allocate(self%mirrors(0))
    item%first=transfer(first,item%first)
    do i=1,size(self%mirrors)
      if (self%mirrors(i)%first/=item%first) cycle
      if (self%mirrors(i)%stride/=stride .or. self%mirrors(i)%count/=count .or. &
          self%mirrors(i)%element_type/=element_type) error stop 'field layout changed inside an epoch'
      self%mirrors(i)%written=self%mirrors(i)%written .or. writing
      handle=self%mirrors(i)%handle
      return
    enddo
    item%stride=stride; item%count=count; item%element_type=element_type; item%written=writing
    select case(element_type)
    case(FORT_SCOPE_INTEGER32, FORT_SCOPE_REAL32)
      item%bytes=4
    case(FORT_SCOPE_REAL64)
      item%bytes=8
    case default
      error stop 'unsupported resident field type'
    end select
    if (stride<item%bytes) error stop 'invalid resident field stride'
    set_aside=0
    do
      select case(element_type)
      case(FORT_SCOPE_INTEGER32)
        allocate(item%integers(count))
      case(FORT_SCOPE_REAL32)
        allocate(item%singles(count))
      case default
        allocate(item%doubles(count))
      end select
      call copy_field(item,.true.)
      item%handle=self%register_storage(mirror_address(item),[count],[1_c_int64_t],element_type,item%bytes, &
                                        .true.,.false.,origin,collision)
      if (.not. collision) exit
      ! The allocator returned memory that is still registered for an array its
      ! owner has freed. The copy is the domain's own, so it can live anywhere:
      ! keep this block out of circulation and take another.
      if (.not. allocated(self%avoided)) allocate(self%avoided(0))
      if (size(self%avoided)==0) then
        call report_overlap(self,transfer(mirror_address(item),0_c_int64_t),count*item%bytes,origin)
        write(error_unit,'(a)') '  The field copy takes other memory; the registration above outlived its array.'
      endif
      self%avoided=[self%avoided,item]
      nullify(item%integers,item%singles,item%doubles)
      set_aside=set_aside+count*item%bytes
      if (set_aside>avoided_limit) error stop 'no host memory free of stale registrations for a resident field'
    enddo
    self%mirrors=[self%mirrors,item]
    handle=item%handle
  end function

  subroutine host_begin(self, handle, access)
    class(resident_domain), intent(inout) :: self
    integer(c_int64_t), intent(in) :: handle
    type(fort_scope_access), intent(in) :: access
    self%stamp=self%stamp+1
    call resident_require(fort_scope_host_begin(self%context,handle,access))
  end subroutine

  subroutine host_end(self, handle)
    class(resident_domain), intent(inout) :: self
    integer(c_int64_t), intent(in) :: handle
    call resident_require(fort_scope_host_end(self%context,handle))
  end subroutine

  function host_box_real32(self, values, lower, first, last, writing) result(handle)
    class(resident_domain), intent(inout) :: self
    real(c_float), contiguous, target, intent(in) :: values(:,:,:)
    integer, intent(in) :: lower(3), first(3), last(3)
    logical, intent(in) :: writing
    integer(c_int64_t) :: handle
    integer(c_size_t), target :: lo(3), hi(3)
    type(fort_scope_section), target :: section
    type(fort_scope_access) :: access
    if (any(first>last) .or. any(last<lower) .or. any(first>lower+shape(values)-1)) &
      error stop 'invalid resident native section'
    handle=self%bind_real32(values,lower,.true.)
    lo=int(max(first,lower)-lower,c_size_t)
    hi=int(min(last,lower+shape(values)-1)-lower+1,c_size_t)
    section%lower=c_loc(lo); section%upper=c_loc(hi)
    access%read_count=1; access%reads=c_loc(section)
    if (writing) then
      access%write_count=1; access%writes=c_loc(section)
    endif
    call self%host_begin(handle,access)
  end function

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

  subroutine release_real32(self, values)
    class(resident_domain), intent(inout) :: self
    real(c_float), contiguous, target, intent(in) :: values(..)
    integer(c_int64_t) :: allocation_id
    logical, allocatable :: keep(:)
    integer :: i
    if (.not. self%active) return
    allocation_id=transfer(c_loc(values),allocation_id)
    do i=1,size(self%storage)
      if (self%storage(i)%identity /= allocation_id) cycle
      if (self%storage(i)%persistent) error stop 'persistent storage retired through a borrowed view'
      call resident_require(fort_scope_retire_v1(self%context,self%storage(i)%handle))
      allocate(keep(size(self%storage))); keep=.true.; keep(i)=.false.
      self%storage=pack(self%storage,keep)
      return
    enddo
  end subroutine

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

  subroutine acquire_work_real32_3(self,site,lower,upper,values)
    class(resident_domain), intent(inout) :: self
    integer(c_int64_t), intent(in) :: site
    integer, intent(in) :: lower(3),upper(3)
    real(c_float), pointer, contiguous, intent(out) :: values(:,:,:)
    type(work_real32_3_slot) :: slot
    integer :: i
    if (.not.self%active.or.site==0) error stop 'workspace outside a checked epoch'
    if (any(upper<lower)) error stop 'empty workspace must be skipped by its reached caller'
    if (.not.allocated(self%work3_f32)) allocate(self%work3_f32(0))
    do i=1,size(self%work3_f32)
      if (self%work3_f32(i)%site/=site) cycle
      if (any(lbound(self%work3_f32(i)%values)/=lower).or.any(ubound(self%work3_f32(i)%values)/=upper)) cycle
      if (self%work3_f32(i)%borrowed) error stop 'overlapping use of a compiler workspace slot'
      ! An earlier invocation may still read this allocation on the device. Its
      ! values are dead; they stay until this invocation's writes replace them
      ! in queue order, so that reusing a slot does not submit the queue.
      call resident_require(fort_scope_recycle_v1(self%context,self%work3_f32(i)%handle))
      self%work3_f32(i)%borrowed=.true.
      values=>self%work3_f32(i)%values
      return
    enddo
    slot%site=site
    allocate(slot%values(lower(1):upper(1),lower(2):upper(2),lower(3):upper(3)))
    slot%handle=self%bind_real32(slot%values,lower,.false.,persistent=.true.)
    slot%borrowed=.true.
    self%work3_f32=[self%work3_f32,slot]
    values=>slot%values
  end subroutine

  subroutine release_work_real32_3(self,site,values)
    class(resident_domain), intent(inout) :: self
    integer(c_int64_t), intent(in) :: site
    real(c_float), pointer, contiguous, intent(inout) :: values(:,:,:)
    integer :: i
    if (.not.self%active.or..not.associated(values).or..not.allocated(self%work3_f32)) &
      error stop 'invalid compiler workspace release'
    do i=1,size(self%work3_f32)
      if (self%work3_f32(i)%site/=site) cycle
      if (.not.associated(values,self%work3_f32(i)%values)) cycle
      if (.not.self%work3_f32(i)%borrowed) error stop 'compiler workspace released twice'
      self%work3_f32(i)%borrowed=.false.
      nullify(values)
      return
    enddo
    error stop 'foreign compiler workspace release'
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
      ! An earlier invocation may still read this allocation on the device. Its
      ! values are dead; they stay until this invocation's writes replace them
      ! in queue order, so that reusing a slot does not submit the queue.
      call resident_require(fort_scope_recycle_v1(self%context,self%work3(i)%handle))
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
    self%stamp=self%stamp+1
    handles=pack(self%storage%handle,.not. self%storage%persistent)
    call self%observe(handles)
    do i=1,size(handles)
      call resident_require(fort_scope_unregister(self%context,handles(i)))
    enddo
    self%storage=pack(self%storage,self%storage%persistent)
    if (allocated(self%mirrors)) then
      ! The structures hold the fields again; host code may change them freely.
      call free_mirrors(self%mirrors)
    endif
    if (allocated(self%avoided)) call free_mirrors(self%avoided)
  end subroutine

  subroutine free_mirrors(mirrors)
    type(field_mirror), allocatable, intent(inout) :: mirrors(:)
    integer :: i
    do i=1,size(mirrors)
      if (associated(mirrors(i)%integers)) deallocate(mirrors(i)%integers)
      if (associated(mirrors(i)%singles)) deallocate(mirrors(i)%singles)
      if (associated(mirrors(i)%doubles)) deallocate(mirrors(i)%doubles)
    enddo
    deallocate(mirrors)
  end subroutine

  subroutine end_epoch(self)
    class(resident_domain), intent(inout) :: self
    if (allocated(self%work3_f32)) then
      if (any(self%work3_f32%borrowed)) error stop 'unmanaged return with borrowed workspace'
    endif
    if (allocated(self%work3)) then
      if (any(self%work3%borrowed)) error stop 'unmanaged return with borrowed workspace'
    endif
    call self%publish()
    self%active=.false.
    !$ if (self%outer_threads > 0) call omp_set_num_threads(self%outer_threads)
    self%outer_threads = 0
  end subroutine

  subroutine close(self)
    class(resident_domain), intent(inout) :: self
    integer :: i
    if (self%domain == 0) return
    call self%end_epoch()
    if (allocated(self%work3_f32)) then
      do i=1,size(self%work3_f32)
        call resident_require(fort_scope_discard_v1(self%context,self%work3_f32(i)%handle))
      enddo
    endif
    if (allocated(self%work3)) then
      do i=1,size(self%work3)
        ! Private work values are dead, including any pending final writes.
        ! Domain close then waits for their queued users before freeing storage.
        call resident_require(fort_scope_discard_v1(self%context,self%work3(i)%handle))
      enddo
    endif
    call resident_require(fort_scope_close(self%context))
    call resident_require(fort_scope_domain_close_v1(self%domain))
    if (allocated(self%work3_f32)) then
      do i=1,size(self%work3_f32)
        deallocate(self%work3_f32(i)%values)
      enddo
      deallocate(self%work3_f32)
    endif
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
