
  subroutine StartResidentPressure(Phi,RHS)
    real(knd), contiguous, target, intent(in) :: Phi(-1:,-1:,-1:), RHS(0:,0:,0:)
    type(elmm_gpu_pressure_config_v1) :: config
    logical :: supported
    supported = gridtype == GRID_UNIFORM .and. explicit_diffusion .and. &
                pressure_solution%poisson_solver == POISSON_SOLVER_POISFFT .and. &
                (discretization_order == 2 .or. discretization_order == 4) .and. &
                min(Prnx,Prny,Prnz) >= 2
    supported = supported .and. all(PoissonBtype == 0 .or. PoissonBtype == 4)
    supported = supported .and. all(PoissonBtype(1:6:2) == PoissonBtype(2:6:2))
    if (.not. supported) then
      call resident_pressure%finish()
      return
    endif
    config%solver%sizes = [Prnx,Prny,Prnz]
    config%solver%lengths = [real(gxmax-gxmin,c_double),real(gymax-gymin,c_double),real(gzmax-gzmin,c_double)]
    config%solver%boundary_conditions = PoissonBtype
    config%solver%approximation = discretization_order
#ifdef DPREC
    config%solver%flags = 0
#else
    config%solver%flags = 1
#endif
    call resident_pressure%start(config)
    resident_pressure%phi = resident_pressure%bind3(Phi,lbound(Phi),.true.,.true.,origin='pressure::phi')
    resident_pressure%rhs = resident_pressure%bind3(RHS,lbound(RHS),.false.,.true.,origin='pressure::rhs')
  end subroutine

  subroutine FinalizeResidentPressure()
    call resident_pressure%finish()
  end subroutine
