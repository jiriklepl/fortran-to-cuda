  subroutine ResidentCFL(U,V,W,nx,ny,nz,dx,dy,dz,maximum,nan)
    real(c_double), contiguous, target, intent(in) :: U(:,:,:),V(:,:,:),W(:,:,:)
    integer, intent(in) :: nx,ny,nz
    real(c_double), intent(in) :: dx,dy,dz
    real(c_double), intent(out) :: maximum
    logical, intent(out) :: nan
    integer(c_int64_t) :: handles(3),handle
    integer(c_int32_t) :: any_nan
    type(fort_scope_external_report_v1) :: report
    if (min(nx,ny,nz)<1) error stop 'invalid CFL grid'
    if (.not.allocated(cfl_values)) allocate(cfl_values(nx,ny,nz))
    if (any(shape(cfl_values)/=[nx,ny,nz])) error stop 'CFL storage epoch shape changed'
    if (any(shape(U)<[nx,ny,nz]+3) .or. any(shape(V)<[nx,ny,nz]+3) .or. &
        any(shape(W)<[nx,ny,nz]+3)) error stop 'CFL velocity geometry changed'
    handles(1)=resident_flow%bind_real64(U,[-2,-2,-2],.true.)
    handles(2)=resident_flow%bind_real64(V,[-2,-2,-2],.true.)
    handles(3)=resident_flow%bind_real64(W,[-2,-2,-2],.true.)
    handle=resident_flow%bind_real64(cfl_values,[1,1,1],.false.,persistent=.true.)
    call pressure_require(cfl_run( &
      resident_flow%context, &
      FORT_SCOPE_GPU, &
      handles(1), &
      handles(2), &
      handles(3), &
      handle, &
      nx, &
      ny, &
      nz, &
      dx, &
      dy, &
      dz))
    call pressure_require(fort_scope_max_real64_v1(resident_flow%context,handle,3_c_int32_t, &
      size(cfl_values,kind=c_size_t),0._c_double,maximum,any_nan,report))
    nan=any_nan/=0
  end subroutine
