
    call PrePoisson(U,V,W,Q,RHS,dt2,mass_flux)


    if (debugparam>1 .and. called>1) call system_clock(count=time2)

    if (pressure_solution%report_mass_flux .and. pressure_solution%check_mass_flux) then
      if (master) write(*,*) "mass flux:", mass_flux
    end if

    if (pressure_solution%poisson_solver==POISSON_SOLVER_SOR) then

        call PoissSOR(Phi,RHS)

    else if (pressure_solution%poisson_solver==POISSON_SOLVER_POISFFT) then

        if (gridtype==GRID_VARIABLE_Z) then
          call Poiss_PoisFFT_variable_z(Phi,RHS)
        else
          if (resident_pressure%active) then
            call resident_pressure%solve()
          else
            call Poiss_PoisFFT(Phi,RHS)
          endif
        end if

    else if (pressure_solution%poisson_solver==POISSON_SOLVER_MULTIGRID) then

        if (Prny==1) then
          call PoissMG2d(Phi,RHS)
        else
          call PoissMG(Phi,RHS)
        end if

    end if


    if (resident_pressure%active) then
      call resident_pressure%halo()
    else
      call Bound_Phi(Phi)
    endif
    
    
