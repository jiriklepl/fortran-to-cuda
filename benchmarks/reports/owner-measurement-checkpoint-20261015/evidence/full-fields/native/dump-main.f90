program CLMM

  use Parameters
  use Initial
  use TimeSteps,  only: TMarchRK3
  use Dynamics,   only: Dynamics_Deallocate
  use Scalars,    only: Scalars_Deallocate
  use Outputs,    only: AllocateOutputs, OutTStep, Output
  use Endianness, only: GetEndianness
  use custom_par


  implicit none




  real(knd), allocatable, dimension(:,:,:)   :: U           !Velocity component in x direction -- horizontal
  real(knd), allocatable, dimension(:,:,:)   :: V           !Velocity component in y direction -- horizontal
  real(knd), allocatable, dimension(:,:,:)   :: W           !Velocity component in z direction -- vertical
  real(knd), allocatable, dimension(:,:,:)   :: Pr          !Pressure (kinematic pressure with possible subgrid TKE part)
  real(knd), allocatable, dimension(:,:,:)   :: Temperature !If buoyancy enabled then liquid potential temperature, othrewise potential temperature.
  real(knd), allocatable, dimension(:,:,:)   :: Moisture    !Total specific humidity q_t.
  real(knd), allocatable, dimension(:,:,:,:) :: Scalar      !last index is a number of scalar (because of paging)

  real(knd) :: delta = 0

  real(tim) :: dt = 0 !time_step

  integer   :: time_step

  real(knd) :: time_step_time

  integer(dbl) :: time_steps_timer_count_start, &
                  time_steps_timer_count_1, &
                  time_steps_timer_count_2, &
                  timer_max_count
  
  integer(dbl) :: timer_count_time_limit

  logical :: error_exit = .false.


  call par_init

  call GetEndianness


  call system_clock(count = time_steps_timer_count_start)  
  
  
  call par_sync_out("Reading parameters...")
  call ReadConfiguration


  call par_sync_out("Setting up boundary conditions...")
  call InitBoundaryConditions


  call par_sync_out("Allocating arrays...")
  call AllocateGlobals

  call par_sync_out("Preparing output data and files...")
  call AllocateOutputs

  call par_sync_out("Setting up initial conditions...")
  call InitialConditions(U, V, W, Pr, &
                         Temperature ,Moisture, Scalar, &
                         time_stepping%dt)


  call InitTimer

  time_stepping%time = time_stepping%start_time
  time_step = 0

  call OutTStep(U, V, W, Pr, &
                Temperature, Moisture, Scalar, &
                time_stepping%dt, delta)
  
  init_phase = .false.
  run_phase = .true.

  if (time_stepping%end_time > time_stepping%start_time) then

    call par_sync_out("Computing...")

    time_step = 1

    do

      if (time_step > time_stepping%max_number_of_time_steps) exit

      call system_clock(count = time_steps_timer_count_1)

#ifdef PAR
      if (master) then
        if (enable_multiple_domains) then
          write (*,'(a,i0,a,i12,a,f12.6)') "domain: ", domain_index, &
                                           "   tstep: ", time_step, &
                                           "   time: ",  time_stepping%time
        else
          write (*,'(a,i12,a,f12.6)') "tstep: ", time_step, "   time: ", time_stepping%time
        end if
      end if
#else
      write (*,'(a,i12,a,f12.6)') "tstep: ", time_step, "   time: ", time_stepping%time
#endif



      call TMarchRK3(U, V, W, Pr, &
                     Temperature, Moisture, Scalar, &
                     delta)



      if (time_stepping%variable_time_steps) then
        time_stepping%time = time_stepping%time + time_stepping%dt
        if (abs(time_stepping%time - time_stepping%end_time)<5*spacing(time_stepping%end_time)) then
          time_stepping%time = time_stepping%end_time
        end if
      else
        time_stepping%time = time_stepping%start_time + time_step * time_stepping%dt
        if (abs(time_stepping%time - time_stepping%end_time)<time_stepping%dt/10) then
          time_stepping%time = time_stepping%end_time
        end if
      end if



      call OutTStep(U, V, W, Pr, &
                    Temperature, Moisture, Scalar, &
                    time_stepping%dt, delta)


      call system_clock(count = time_steps_timer_count_2)
      if (master) then
        time_step_time = real(time_steps_timer_count_2 - time_steps_timer_count_1, knd) / &
          real(timer_rate, knd)
        time_steps_time = time_steps_time + time_step_time
      end if

      if ((steady==1) .and. (delta<eps)) then
        if (master) write (*,*) "Steady state reached."
        exit
      endif

      if ((steady==0) .and. (time_stepping%time>=time_stepping%end_time)) then
        if (master) write (*,*) "Time limit reached."
        exit
      endif

      if (time_step>=3 .and. &
          time_stepping%variable_time_steps .and. &
          time_stepping%dt < time_stepping%dt_min) then
        if (master) write (*,*) "Solution diverged. 1"
        error_exit = .true.
        exit
      endif

      if (time_step>=3 .and. &
          .not.time_stepping%variable_time_steps .and. &
          time_stepping%enable_CFL_check .and. &
          time_stepping%CFL > time_stepping%CFL_max) then
        if (master) write (*,*) "Solution diverged. 2"
        error_exit = .true.
        exit
      endif
      
      !don't check that often to avoid synchronizing unnecesarilly
      if (mod(time_step,time_stepping%check_period)==0) then
        if (master) then
          error_exit = time_steps_timer_count_2 - time_steps_timer_count_start > timer_count_time_limit
          if (error_exit) write(*,*) "Maximum clock time exceeded."
        end if
#ifdef PAR        
        call par_co_broadcast(error_exit, master_im)
#endif        
        if (error_exit) exit
      end if

      time_step = time_step + 1
      
    enddo

  endif

  call par_sync_all
  if (master .and. time_steps_time>0) &
    write(*,'(1x,a,f0.2)') "Total wall clock time for time steps: ", time_steps_time
  if (master .and. poisson_solver_time>0) &
    write(*,'(1x,a,f0.2)') "Wall clock time for Poisson solver: ", poisson_solver_time
#ifdef PAR
  block
    use domains_bc_par
    if (master .and. enable_multiple_domains .and. time_communicating_domains>0) &
      write(*,'(1x,a,i0,a,f0.2)') "Wall clock time waiting and communicating with other domains on domain ", &
        domain_index,": ",time_communicating_domains
  end block
#endif
  call par_sync_all

  block
    use Dynamics, only: diagnostic_U2 => U2
    use Dynamics, only: diagnostic_V2 => V2
    use Dynamics, only: diagnostic_W2 => W2
    use Dynamics, only: diagnostic_Ustar => Ustar
    use Dynamics, only: diagnostic_Vstar => Vstar
    use Dynamics, only: diagnostic_Wstar => Wstar
    integer :: diagnostic_unit
    if (.not.allocated(U)) error stop "missing diagnostic field U"
    open(newunit=diagnostic_unit,file='fields-U.bin', &
         access='stream',form='unformatted',status='new')
    write(diagnostic_unit) U
    close(diagnostic_unit)
    open(newunit=diagnostic_unit,file='fields-U.shape',status='new')
    write(diagnostic_unit,*) storage_size(U), shape(U), &
         lbound(U), ubound(U)
    close(diagnostic_unit)
    if (.not.allocated(V)) error stop "missing diagnostic field V"
    open(newunit=diagnostic_unit,file='fields-V.bin', &
         access='stream',form='unformatted',status='new')
    write(diagnostic_unit) V
    close(diagnostic_unit)
    open(newunit=diagnostic_unit,file='fields-V.shape',status='new')
    write(diagnostic_unit,*) storage_size(V), shape(V), &
         lbound(V), ubound(V)
    close(diagnostic_unit)
    if (.not.allocated(W)) error stop "missing diagnostic field W"
    open(newunit=diagnostic_unit,file='fields-W.bin', &
         access='stream',form='unformatted',status='new')
    write(diagnostic_unit) W
    close(diagnostic_unit)
    open(newunit=diagnostic_unit,file='fields-W.shape',status='new')
    write(diagnostic_unit,*) storage_size(W), shape(W), &
         lbound(W), ubound(W)
    close(diagnostic_unit)
    if (.not.allocated(Pr)) error stop "missing diagnostic field Pr"
    open(newunit=diagnostic_unit,file='fields-Pr.bin', &
         access='stream',form='unformatted',status='new')
    write(diagnostic_unit) Pr
    close(diagnostic_unit)
    open(newunit=diagnostic_unit,file='fields-Pr.shape',status='new')
    write(diagnostic_unit,*) storage_size(Pr), shape(Pr), &
         lbound(Pr), ubound(Pr)
    close(diagnostic_unit)
    if (.not.allocated(Viscosity)) error stop "missing diagnostic field Viscosity"
    open(newunit=diagnostic_unit,file='fields-Viscosity.bin', &
         access='stream',form='unformatted',status='new')
    write(diagnostic_unit) Viscosity
    close(diagnostic_unit)
    open(newunit=diagnostic_unit,file='fields-Viscosity.shape',status='new')
    write(diagnostic_unit,*) storage_size(Viscosity), shape(Viscosity), &
         lbound(Viscosity), ubound(Viscosity)
    close(diagnostic_unit)
    if (.not.allocated(Temperature)) error stop "missing diagnostic field Temperature"
    open(newunit=diagnostic_unit,file='fields-Temperature.bin', &
         access='stream',form='unformatted',status='new')
    write(diagnostic_unit) Temperature
    close(diagnostic_unit)
    open(newunit=diagnostic_unit,file='fields-Temperature.shape',status='new')
    write(diagnostic_unit,*) storage_size(Temperature), shape(Temperature), &
         lbound(Temperature), ubound(Temperature)
    close(diagnostic_unit)
    if (.not.allocated(Moisture)) error stop "missing diagnostic field Moisture"
    open(newunit=diagnostic_unit,file='fields-Moisture.bin', &
         access='stream',form='unformatted',status='new')
    write(diagnostic_unit) Moisture
    close(diagnostic_unit)
    open(newunit=diagnostic_unit,file='fields-Moisture.shape',status='new')
    write(diagnostic_unit,*) storage_size(Moisture), shape(Moisture), &
         lbound(Moisture), ubound(Moisture)
    close(diagnostic_unit)
    if (.not.allocated(Scalar)) error stop "missing diagnostic field Scalar"
    open(newunit=diagnostic_unit,file='fields-Scalar.bin', &
         access='stream',form='unformatted',status='new')
    write(diagnostic_unit) Scalar
    close(diagnostic_unit)
    open(newunit=diagnostic_unit,file='fields-Scalar.shape',status='new')
    write(diagnostic_unit,*) storage_size(Scalar), shape(Scalar), &
         lbound(Scalar), ubound(Scalar)
    close(diagnostic_unit)
    if (.not.allocated(TDiff)) error stop "missing diagnostic field TDiff"
    open(newunit=diagnostic_unit,file='fields-TDiff.bin', &
         access='stream',form='unformatted',status='new')
    write(diagnostic_unit) TDiff
    close(diagnostic_unit)
    open(newunit=diagnostic_unit,file='fields-TDiff.shape',status='new')
    write(diagnostic_unit,*) storage_size(TDiff), shape(TDiff), &
         lbound(TDiff), ubound(TDiff)
    close(diagnostic_unit)
    if (.not.allocated(diagnostic_U2)) error stop "missing diagnostic field U2"
    open(newunit=diagnostic_unit,file='fields-U2.bin', &
         access='stream',form='unformatted',status='new')
    write(diagnostic_unit) diagnostic_U2
    close(diagnostic_unit)
    open(newunit=diagnostic_unit,file='fields-U2.shape',status='new')
    write(diagnostic_unit,*) storage_size(diagnostic_U2), shape(diagnostic_U2), &
         lbound(diagnostic_U2), ubound(diagnostic_U2)
    close(diagnostic_unit)
    if (.not.allocated(diagnostic_V2)) error stop "missing diagnostic field V2"
    open(newunit=diagnostic_unit,file='fields-V2.bin', &
         access='stream',form='unformatted',status='new')
    write(diagnostic_unit) diagnostic_V2
    close(diagnostic_unit)
    open(newunit=diagnostic_unit,file='fields-V2.shape',status='new')
    write(diagnostic_unit,*) storage_size(diagnostic_V2), shape(diagnostic_V2), &
         lbound(diagnostic_V2), ubound(diagnostic_V2)
    close(diagnostic_unit)
    if (.not.allocated(diagnostic_W2)) error stop "missing diagnostic field W2"
    open(newunit=diagnostic_unit,file='fields-W2.bin', &
         access='stream',form='unformatted',status='new')
    write(diagnostic_unit) diagnostic_W2
    close(diagnostic_unit)
    open(newunit=diagnostic_unit,file='fields-W2.shape',status='new')
    write(diagnostic_unit,*) storage_size(diagnostic_W2), shape(diagnostic_W2), &
         lbound(diagnostic_W2), ubound(diagnostic_W2)
    close(diagnostic_unit)
    if (.not.allocated(diagnostic_Ustar)) error stop "missing diagnostic field Ustar"
    open(newunit=diagnostic_unit,file='fields-Ustar.bin', &
         access='stream',form='unformatted',status='new')
    write(diagnostic_unit) diagnostic_Ustar
    close(diagnostic_unit)
    open(newunit=diagnostic_unit,file='fields-Ustar.shape',status='new')
    write(diagnostic_unit,*) storage_size(diagnostic_Ustar), shape(diagnostic_Ustar), &
         lbound(diagnostic_Ustar), ubound(diagnostic_Ustar)
    close(diagnostic_unit)
    if (.not.allocated(diagnostic_Vstar)) error stop "missing diagnostic field Vstar"
    open(newunit=diagnostic_unit,file='fields-Vstar.bin', &
         access='stream',form='unformatted',status='new')
    write(diagnostic_unit) diagnostic_Vstar
    close(diagnostic_unit)
    open(newunit=diagnostic_unit,file='fields-Vstar.shape',status='new')
    write(diagnostic_unit,*) storage_size(diagnostic_Vstar), shape(diagnostic_Vstar), &
         lbound(diagnostic_Vstar), ubound(diagnostic_Vstar)
    close(diagnostic_unit)
    if (.not.allocated(diagnostic_Wstar)) error stop "missing diagnostic field Wstar"
    open(newunit=diagnostic_unit,file='fields-Wstar.bin', &
         access='stream',form='unformatted',status='new')
    write(diagnostic_unit) diagnostic_Wstar
    close(diagnostic_unit)
    open(newunit=diagnostic_unit,file='fields-Wstar.shape',status='new')
    write(diagnostic_unit,*) storage_size(diagnostic_Wstar), shape(diagnostic_Wstar), &
         lbound(diagnostic_Wstar), ubound(diagnostic_Wstar)
    close(diagnostic_unit)
  end block
  call Dynamics_Deallocate
  call Scalars_Deallocate

  call par_sync_out("Saving results...")


  call Output(U,V,W,Pr,Temperature,Moisture,Scalar)


  call par_sync_out("saved.")

  block
    use OutputData, only: U_avg, V_avg, W_avg, Pr_avg, Scalar_avg
    integer :: diagnostic_unit
    if (.not.allocated(U_avg)) error stop "missing diagnostic field U_avg"
    open(newunit=diagnostic_unit,file='fields-U_avg.bin', &
         access='stream',form='unformatted',status='new')
    write(diagnostic_unit) U_avg
    close(diagnostic_unit)
    open(newunit=diagnostic_unit,file='fields-U_avg.shape',status='new')
    write(diagnostic_unit,*) storage_size(U_avg), shape(U_avg), &
         lbound(U_avg), ubound(U_avg)
    close(diagnostic_unit)
    if (.not.allocated(V_avg)) error stop "missing diagnostic field V_avg"
    open(newunit=diagnostic_unit,file='fields-V_avg.bin', &
         access='stream',form='unformatted',status='new')
    write(diagnostic_unit) V_avg
    close(diagnostic_unit)
    open(newunit=diagnostic_unit,file='fields-V_avg.shape',status='new')
    write(diagnostic_unit,*) storage_size(V_avg), shape(V_avg), &
         lbound(V_avg), ubound(V_avg)
    close(diagnostic_unit)
    if (.not.allocated(W_avg)) error stop "missing diagnostic field W_avg"
    open(newunit=diagnostic_unit,file='fields-W_avg.bin', &
         access='stream',form='unformatted',status='new')
    write(diagnostic_unit) W_avg
    close(diagnostic_unit)
    open(newunit=diagnostic_unit,file='fields-W_avg.shape',status='new')
    write(diagnostic_unit,*) storage_size(W_avg), shape(W_avg), &
         lbound(W_avg), ubound(W_avg)
    close(diagnostic_unit)
    if (.not.allocated(Pr_avg)) error stop "missing diagnostic field Pr_avg"
    open(newunit=diagnostic_unit,file='fields-Pr_avg.bin', &
         access='stream',form='unformatted',status='new')
    write(diagnostic_unit) Pr_avg
    close(diagnostic_unit)
    open(newunit=diagnostic_unit,file='fields-Pr_avg.shape',status='new')
    write(diagnostic_unit,*) storage_size(Pr_avg), shape(Pr_avg), &
         lbound(Pr_avg), ubound(Pr_avg)
    close(diagnostic_unit)
    if (.not.allocated(Scalar_avg)) error stop "missing diagnostic field Scalar_avg"
    open(newunit=diagnostic_unit,file='fields-Scalar_avg.bin', &
         access='stream',form='unformatted',status='new')
    write(diagnostic_unit) Scalar_avg
    close(diagnostic_unit)
    open(newunit=diagnostic_unit,file='fields-Scalar_avg.shape',status='new')
    write(diagnostic_unit,*) storage_size(Scalar_avg), shape(Scalar_avg), &
         lbound(Scalar_avg), ubound(Scalar_avg)
    close(diagnostic_unit)
  end block
  call DeallocateGlobals

#ifdef PAR
  if (error_exit .and. enable_multiple_domains .and. number_of_domains>1) then
    call error_stop("Error, one of the domains stopped.")
  else
    call par_finalize
  end if
#endif


contains


  subroutine AllocateGlobals
    use WaterThermodynamics
    allocate(U(-2:Unx+3,-2:Uny+3,-2:Unz+3))
    allocate(V(-2:Vnx+3,-2:Vny+3,-2:Vnz+3))
    allocate(W(-2:Wnx+3,-2:Wny+3,-2:Wnz+3))
    allocate(Pr(-1:Prnx+2,-1:Prny+2,-1:Prnz+2))

    U = 0
    V = 0
    W = 0
    Pr = 0


    if (enable_buoyancy) then
      allocate(Temperature(-2:Prnx+3,-2:Prny+3,-2:Prnz+3))
      Temperature = 0
    else
      allocate(Temperature(0,0,0))
    endif

    if (enable_moisture) then
      allocate(Moisture(-2:Prnx+3,-2:Prny+3,-2:Prnz+3))
      Moisture = 0      
    else
      allocate(Moisture(0,0,0))
    endif
    
    if (enable_liquid) then
      allocate(LiquidWater(1:Prnx,1:Prny,1:Wnz+1))
      LiquidWater = 0      
    else
      allocate(LiquidWater(0,0,0))
    endif


    if (num_of_scalars>0) then
      allocate(Scalar(-2:Prnx+3,-2:Prny+3,-2:Prnz+3,num_of_scalars))
      Scalar = 0
    else
      allocate(Scalar(0,0,0,0))
    endif


    allocate(Viscosity(-1:Prnx+2,-1:Prny+2,-1:Prnz+2))

    if (enable_buoyancy .or. &
        enable_moisture .or. &
        num_of_scalars>0)      then

      allocate(TDiff(-1:Prnx+2,-1:Prny+2,-1:Prnz+2))
      TDiff = 0

    else
      allocate(TDiff(0,0,0))
    endif

  end subroutine AllocateGlobals


  subroutine DeallocateGlobals
    !To assist memory leak checking because some compilers do not
    !deallocate main program and module variables automatically
    !(in F2008 they have the SAVE attribute).

    deallocate(U, V, W, Pr)

    deallocate(Temperature, Moisture, Scalar)

    deallocate(Viscosity, TDiff)

    deallocate(xU, yV, zW, dxU, dyV, dzW)
    deallocate(xPr, yPr, zPr, dxPr, dyPr, dzPr)

    deallocate(Utype, Vtype, Wtype, Prtype)

    deallocate(Uin, Vin, Win)

    if (allocated(TempIn)) deallocate(TempIn)
    if (allocated(MoistIn)) deallocate(MoistIn)

    if (allocated(BsideTArr)) deallocate(BsideTArr)
    if (allocated(BsideTFlArr)) deallocate(BsideTFlArr)

    if (allocated(BsideMArr)) deallocate(BsideMArr)
    if (allocated(BsideMFlArr)) deallocate(BsideMFlArr)


  end subroutine
  
  
  subroutine InitTimer
    use ieee_exceptions
    logical :: saved_fpe_mode(size(ieee_all))

    call system_clock(count_rate = timer_rate)
    call system_clock(count_max = timer_max_count)   

    call ieee_get_halting_mode(ieee_all, saved_fpe_mode)
    call ieee_set_halting_mode(ieee_all, .false.)

    timer_count_time_limit = int( min(time_stepping%clock_time_limit &
                                        * real(timer_rate, knd),  &
                                      real(timer_max_count-timer_max_count/1024, knd)) &
                                , dbl)

    call ieee_set_halting_mode(ieee_all, saved_fpe_mode)
  end subroutine


end program CLMM

