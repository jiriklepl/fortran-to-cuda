module Dynamics
  use Parameters
  use Boundaries, only: BoundUVW
  use ScalarBoundaries, only: BoundTemperature, BoundViscosity
  
  use MomentumDiffusion

  implicit none

  !module variables to allow their deallocation before the program end
  real(knd), dimension(:,:,:), allocatable:: U3,V3,W3

  real(knd), dimension(:,:,:), allocatable :: Q
  real(knd), dimension(:,:,:), allocatable :: U2,Ustar
  real(knd), dimension(:,:,:), allocatable :: V2,Vstar
  real(knd), dimension(:,:,:), allocatable :: W2,Wstar

  real(knd), dimension(:), allocatable :: Uwm, Vwm, Wwm

contains


  subroutine Dynamics_Deallocate
    !Deallocates the working arrays
    use Wallmodels
    if (allocated(U3)) deallocate(U3)
    if (allocated(V3)) deallocate(V3)
    if (allocated(W3)) deallocate(W3)
    if (allocated(Q)) deallocate(Q)
    if (allocated(U2)) deallocate(U2)
    if (allocated(V2)) deallocate(V2)
    if (allocated(W2)) deallocate(W2)
    if (allocated(Ustar)) deallocate(Ustar)
    if (allocated(Vstar)) deallocate(Vstar)
    if (allocated(Wstar)) deallocate(Wstar)
    if (allocated(ApU)) deallocate(ApU)
    if (allocated(ApV)) deallocate(ApV)
    if (allocated(ApW)) deallocate(ApW)
    if (allocated(work)) deallocate(work)

    if (allocated(Uwm)) deallocate(Uwm)
    if (allocated(Vwm)) deallocate(Vwm)
    if (allocated(Wwm)) deallocate(Wwm)

    !imported from Wallmodels
    if (allocated(Uflx_mask)) deallocate(Uflx_mask)
    if (allocated(Ufly_mask)) deallocate(Ufly_mask)
    if (allocated(Uflz_mask)) deallocate(Uflz_mask)
    if (allocated(Vflx_mask)) deallocate(Vflx_mask)
    if (allocated(Vfly_mask)) deallocate(Vfly_mask)
    if (allocated(Vflz_mask)) deallocate(Vflz_mask)
    if (allocated(Wflx_mask)) deallocate(Wflx_mask)
    if (allocated(Wfly_mask)) deallocate(Wfly_mask)
    if (allocated(Wflz_mask)) deallocate(Wflz_mask)
  end subroutine


  subroutine PressureGrad(Pr, U, V, W, coef)
    use Pressure, only: pressure_solution, PROJECTION_METHOD_PRESSURE
    real(knd), intent(inout), contiguous :: Pr(-1:,-1:,-1:)
    real(knd), intent(inout), contiguous, dimension(-2:,-2:,-2:) :: U,V,W
    real(knd), intent(in)    :: coef
    real(knd) :: A, Ax, Ay, Az
    integer :: i,j,k
    real(knd), parameter :: C1 = 9._knd / 8, C3 = 1._knd / (8*3)

    A = -coef
    Ax = - coef / dxmin
    Ay = - coef / dymin
    Az = - coef / dzmin

    !$omp parallel
    if (enable_pr_gradient_x_profile) then
      !$omp do
      do k = 1, Unz
        do j = 1, Uny
          do i = 1, Unx
            U(i,j,k) = U(i,j,k) + A * pr_gradient_profile_x(k)
          end do
        end do
      end do
      !$omp end do
    else if (enable_pr_gradient_x_uniform) then
      !$omp do
      do k = 1, Unz
        do j = 1, Uny
          do i = 1, Unx
            U(i,j,k) = U(i,j,k) + A * pr_gradient_x
          end do
        end do
      end do
      !$omp end do
    end if

    if (enable_pr_gradient_y_profile) then
      !$omp do
      do k = 1, Vnz
        do j = 1, Vny
          do i = 1, Vnx
            V(i,j,k) = V(i,j,k) + A * pr_gradient_profile_y(k)
          end do
        end do
      end do
      !$omp end do
    else if (enable_pr_gradient_y_uniform) then
      !$omp do
      do k = 1, Vnz
        do j = 1, Vny
          do i = 1, Vnx
            V(i,j,k) = V(i,j,k) + A * pr_gradient_y
          end do
        end do
      end do
      !$omp end do
    end if
    
    if (pressure_solution%projection_method/=PROJECTION_METHOD_PRESSURE) then
    
      if (discretization_order==4) then
        !grad p_i = 9/8 (p_i+1 - p_i) / dx  - 1/8 (p_i+2 - p_i-1) / 3*dx
      
        !$omp do
        do k = 1, Unz
          do j = 1, Uny
            do i = 1, Unx
              U(i,j,k) = U(i,j,k) + &
                    Ax * ( (Pr(i+1,j,k) - Pr(i  ,j,k)) * C1 - &
                            (Pr(i+2,j,k) - Pr(i-1,j,k)) * C3 )
            end do
          end do
        end do
        !$omp end do nowait
        !$omp do
        do k = 1, Vnz
          do j = 1, Vny
            do i = 1, Vnx
              V(i,j,k) = V(i,j,k) + &
                    Ay * ( (Pr(i,j+1,k) - Pr(i,j  ,k)) * C1 - &
                            (Pr(i,j+2,k) - Pr(i,j-1,k)) * C3 )
            end do
          end do
        end do
        !$omp end do nowait
        !$omp do
        do k = 1, Wnz
          do j = 1, Wny
            do i = 1, Wnx
              W(i,j,k) = W(i,j,k) + &
                    Az * ( (Pr(i,j,k+1) - Pr(i,j,k  )) * C1 - &
                            (Pr(i,j,k+2) - Pr(i,j,k-1)) * C3 )
            end do
          end do
        end do
        !$omp end do nowait
      else
        !$omp do
        do k = 1, Unz
          do j = 1, Uny
            do i = 1, Unx
              U(i,j,k) = U(i,j,k) + Ax * (Pr(i+1,j,k)-Pr(i,j,k))
            end do
          end do
        end do
        !$omp end do nowait
        !$omp do
        do k = 1, Vnz
          do j = 1, Vny
            do i = 1, Vnx
              V(i,j,k) = V(i,j,k) + Ay * (Pr(i,j+1,k)-Pr(i,j,k))
            end do
          end do
        end do
        !$omp end do nowait
        if (gridtype==GRID_VARIABLE_Z) then
          !$omp do
          do k = 1, Wnz
            do j = 1, Wny
              do i = 1, Wnx
                W(i,j,k) = W(i,j,k) - coef * (Pr(i,j,k+1)-Pr(i,j,k)) / dzW(k)
              end do
            end do
          end do
          !$omp end do nowait
        else
          !$omp do
          do k = 1, Wnz
            do j = 1, Wny
              do i = 1, Wnx
                W(i,j,k) = W(i,j,k) + Az * (Pr(i,j,k+1)-Pr(i,j,k))
              end do
            end do
          end do
          !$omp end do nowait
        end if
      end if
      
    end if
    !$omp end parallel
  end subroutine PressureGrad


  
  
  subroutine StressBoundaryFlux(U2, V2, dt)
    use ArrayUtilities,only: add
    use Outputs,only: current_profiles
    real(knd), dimension(-2:,-2:,-2:), contiguous, intent(inout) :: U2, V2
    real(knd), intent(in) :: dt
    real(knd) :: flux
    integer :: first,last
    
    if (Btype(To)==BC_AUTOMATIC_FLUX) then
      first = min(Prnz*5/6,Prnz-5)
      last = Prnz-5
      
      flux = sum(current_profiles%uw(first:last)) + sum(current_profiles%uwsgs(first:last))
      flux = flux / (last-first+1)
      call add(U2(:,:,Unz), -dt*flux/dzPr(Unz))
      
      flux = sum(current_profiles%vw(first:last)) + sum(current_profiles%vwsgs(first:last))
      flux = flux / (last-first+1)
      call add(V2(:,:,Vnz), -dt*flux/dzPr(Vnz))
    end if
  end subroutine





  subroutine Convection(U, V, W, U2, V2, W2, &
                        Ustar, Vstar, Wstar, &
                        Temperature, Moisture, Scalar, Pr, &
                        beta, rho, RK_stage, dt)
    use MomentumAdvection
    use MomentumAdvection_variable_z
    use VolumeSources, only: ResistanceForce
    use LinearForcing, only: linear_forcing
    use Sponge, only: rayleigh_damping_uvw
    real(knd), dimension(-2:,-2:,-2:), contiguous, intent(in)    :: U, V, W
    real(knd), dimension(-2:,-2:,-2:), contiguous, intent(out)   :: U2, V2, W2
    real(knd), dimension(-2:,-2:,-2:), contiguous, intent(inout) :: Ustar, Vstar ,Wstar
    real(knd), dimension(-2:,-2:,-2:), contiguous, intent(in)    :: Temperature
    real(knd), dimension(-2:,-2:,-2:), contiguous, intent(in)    :: Moisture
    real(knd), dimension(-2:,-2:,-2:,:), contiguous, intent(in)  :: Scalar
    real(knd), dimension(-1:,-1:,-1:), contiguous, intent(in)    :: Pr
    real(knd), dimension(1:3), intent(in) :: beta,rho
    integer,   intent(in) :: RK_stage
    real(knd), intent(in) :: dt
    integer :: i,j,k
    
#ifdef CUSTOM_FORCING
    interface
      subroutine CustomForcingProcedure(U2, V2, W2, U, V, W, &
                                     Temperature, Moisture, Scalar, Pr)
        use Parameters
        real(knd), dimension(-2:,-2:,-2:), contiguous, intent(in)    :: U, V, W
        real(knd), dimension(-2:,-2:,-2:), contiguous, intent(out)   :: U2, V2, W2
        real(knd), dimension(-2:,-2:,-2:), contiguous, intent(in)    :: Temperature
        real(knd), dimension(-2:,-2:,-2:), contiguous, intent(in)    :: Moisture
        real(knd), dimension(-2:,-2:,-2:,:), contiguous, intent(in)  :: Scalar
        real(knd), dimension(-1:,-1:,-1:), contiguous, intent(in)    :: Pr
      end subroutine
    end interface
#endif

    if (RK_stage>1) then
      !$omp parallel private(i,j,k)
      !$omp do
      do k = 1, Unz
        do j = 1, Uny
          do i = 1, Unx
            U2(i,j,k) = Ustar(i,j,k) * rho(RK_stage) * dt
          end do
        end do
      end do
      !$omp end do nowait
      !$omp do
      do k = 1, Vnz
        do j = 1, Vny
          do i = 1, Vnx
            V2(i,j,k) = Vstar(i,j,k) * rho(RK_stage) * dt
          end do
        end do
      end do
      !$omp end do nowait
      !$omp do
      do k = 1, Wnz
        do j = 1, Wny
          do i = 1, Wnx
            W2(i,j,k) = Wstar(i,j,k) * rho(RK_stage) * dt
          end do
        end do
      end do
      !$omp end do
      !$omp end parallel
    else
      !$omp parallel private(i,j,k)
      !$omp do
      do k = 1, Unz
        do j = 1, Uny
          do i = 1, Unx
            Ustar(i,j,k) = 0
          end do
        end do
      end do
      !$omp end do nowait
      !$omp do
      do k = 1, Vnz
        do j = 1, Vny
          do i = 1, Vnx
            Vstar(i,j,k) = 0
          end do
        end do
      end do
      !$omp end do nowait
      !$omp do
      do k = 1, Wnz
        do j = 1, Wny
          do i = 1, Wnx
            Wstar(i,j,k) = 0
          end do
        end do
      end do
      !$omp end do nowait

      !$omp do
      do k = 1, Unz
        do j = 1, Uny
          do i = 1, Unx
            U2(i,j,k) = 0
          end do
        end do
      end do
      !$omp end do nowait
      !$omp do
      do k = 1, Vnz
        do j = 1, Vny
          do i = 1, Vnx
            V2(i,j,k) = 0
          end do
        end do
      end do
      !$omp end do nowait
      !$omp do
      do k = 1, Wnz
        do j = 1, Wny
          do i = 1, Wnx
            W2(i,j,k) = 0
          end do
        end do
      end do
      !$omp end do
      !$omp end parallel
    end if

    if (advection_method>0) then

      if (advection_method==2) then
        if (gridtype==GRID_VARIABLE_Z) then
          call CDU_variable_z(Ustar,U,V,W)
          call CDV_variable_z(Vstar,U,V,W)
          call CDW_variable_z(Wstar,U,V,W)
        else
          call CDU(Ustar,U,V,W)
          call CDV(Vstar,U,V,W)
          call CDW(Wstar,U,V,W)
        end if
      else if (advection_method==4) then
        call CD4divU(Ustar,U,V,W)
        call CD4divV(Vstar,U,V,W)
        call CD4divW(Wstar,U,V,W)
      else if (advection_method==5) then
        call CDUdiv(Ustar,U,V,W)
        call CDVdiv(Vstar,U,V,W)
        call CDWdiv(Wstar,U,V,W)
      else if (advection_method==6) then
        call set(Ustar, 0)
        call set(Vstar, 0)
        call set(Wstar, 0)
        call CDUadv(Ustar,U,V,W)
        call CDVadv(Vstar,U,V,W)
        call CDWadv(Wstar,U,V,W)
      end if

    else

      !$omp parallel private(i,j,k)
      !$omp do
      do k = 1, Unz
        do j = 1, Uny
          do i = 1, Unx
            Ustar(i,j,k) = 0
          end do
        end do
      end do
      !$omp end do nowait
      !$omp do
      do k = 1, Vnz
        do j = 1, Vny
          do i = 1, Vnx
            Vstar(i,j,k) = 0
          end do
        end do
      end do
      !$omp end do nowait
      !$omp do
      do k = 1, Wnz
        do j = 1, Wny
          do i = 1, Wnx
            Wstar(i,j,k) = 0
          end do
        end do
      end do
      !$omp end do
      !$omp end parallel

    end if

    call FilterUstar    

    if (abs(Coriolis_parameter)>tiny(1._knd)) call CoriolisForce(Ustar, Vstar, U, V)

    call BuoyancyForce(Wstar, Temperature, Moisture, Scalar, Pr)

    call ResistanceForce(Ustar, Vstar, Wstar, U, V, W)
    
    call linear_forcing(Ustar, Vstar, Wstar, U, V, W)
    
    call rayleigh_damping_uvw(Ustar, Vstar, Wstar, U, V, W)

    call StressBoundaryFlux(Ustar, Vstar, dt)

    if (explicit_diffusion) then
      if (discretization_order>=3) then
        call MomentumDiffusion_4ord_5point(Ustar, Vstar, Wstar, U, V, W)
      else
        if (gridtype==GRID_VARIABLE_Z) then
          call MomentumDiffusion_variable_z_nobranch_2ord(Ustar, Vstar, Wstar, U, V, W)
        else
          call MomentumDiffusion_nobranch_2ord(Ustar, Vstar, Wstar, U, V, W)
        end if
      end if
    end if  

#ifdef CUSTOM_FORCING
    call CustomForcingProcedure(Ustar, Vstar, Wstar, U, V, W, &
                                Temperature, Moisture, Scalar, Pr)
#endif

    !$omp parallel private(i,j,k)
    !$omp do
    do k = 1, Unz
      do j = 1, Uny
        do i = 1, Unx
          U2(i,j,k) = U2(i,j,k)  +  Ustar(i,j,k) * beta(RK_stage) * dt
        end do
      end do
    end do
    !$omp end do nowait
    !$omp do
    do k = 1, Vnz
      do j = 1, Vny
        do i = 1, Vnx
          V2(i,j,k) = V2(i,j,k)  +  Vstar(i,j,k) * beta(RK_stage) * dt
        end do
      end do
    end do
    !$omp end do nowait
    !$omp do
    do k = 1, Wnz
      do j = 1, Wny
        do i = 1, Wnx
          W2(i,j,k) = W2(i,j,k)  +  Wstar(i,j,k) * beta(RK_stage) * dt
        end do
      end do
    end do
    !$omp end do
    !$omp end parallel

  contains

    subroutine FilterUstar

      use Filters, only: filtertype, Filter

      if (filtertype/=0) then
        call BoundUVW(Ustar, Vstar, Wstar, regime=2)

        call Filter(Ustar,Utype)

        call Filter(Vstar,Vtype)

        call Filter(Wstar,Wtype)

        call BoundUVW(Ustar, Vstar, Wstar, regime=2)
      end if

    end subroutine
  end subroutine Convection











  subroutine TimeStepLength(U, V, W, t_s)
    use ieee_arithmetic
#ifdef PAR
    use custom_par
#endif
    real(knd), dimension(-2:,-2:,-2:), contiguous, intent(in)  :: U,V,W
    type(time_step_control), intent(inout) :: t_s
    integer :: i,j,k
    real(knd) :: U_dx_max, p
    logical :: nan
    
    nan = .false.

    if (t_s%variable_time_steps) then

      call get_U_dx_max
 
      if (nan) then
        t_s%dt = tiny(t_s%dt)
      else if (U_dx_max > 0) then
        t_s%dt = min(t_s%CFL / U_dx_max, t_s%dt_max)
      else
        t_s%dt = t_s%dt_max
      end if

#ifdef PAR
      t_s%dt = par_co_min(t_s%dt)
#endif

    else

      if (t_s%enable_CFL_check) call get_U_dx_max

      if (nan) U_dx_max = huge(U_dx_max)

      t_s%dt = t_s%dt_constant

      t_s%CFL = U_dx_max * t_s%dt

#ifdef PAR
      if (t_s%enable_CFL_check) t_s%CFL = par_co_max(t_s%CFL)
#endif

    end if

    if (time_stepping%variable_time_steps .and. steady /= 1 .and. t_s%dt + t_s%time > t_s%end_time)  &
      t_s%dt = t_s%end_time - t_s%time
    

  contains

    subroutine get_U_dx_max
      U_dx_max = 0
      
      if (gridtype==GRID_VARIABLE_Z) then
        !$omp parallel do private(i,j,k,p) reduction(max:U_dx_max) reduction(.or.:nan)
        do k = 1, Prnz
          do j = 1, Prny
            do i = 1, Prnx
              !For scalar advection the sum proved to be necessary when the flow is not aligned to grid.
              p =     max( abs(U(i,j,k)), abs(U(i-1,j,k)) ) / dxPr(i)
              p = p + max( abs(V(i,j,k)), abs(V(i,j-1,k)) ) / dyPr(j)
              p = p + max( abs(W(i,j,k)), abs(W(i,j,k-1)) ) / dzPr(k)
              
              U_dx_max = max(U_dx_max, p)
              if (ieee_is_nan(p)) nan = .true.
            end do
          end do
        end do
        !$omp end parallel do
      else
        !$omp parallel do private(i,j,k,p) reduction(max:U_dx_max) reduction(.or.:nan)
        do k = 1, Prnz
          do j = 1, Prny
            do i = 1, Prnx
              !For scalar advection the sum proved to be necessary when the flow is not aligned to grid.
              p =     max( abs(U(i,j,k)), abs(U(i-1,j,k)) ) / dxmin
              p = p + max( abs(V(i,j,k)), abs(V(i,j-1,k)) ) / dymin
              p = p + max( abs(W(i,j,k)), abs(W(i,j,k-1)) ) / dzmin
              
              U_dx_max = max(U_dx_max, p)
              if (ieee_is_nan(p)) nan = .true.
            end do
          end do
        end do
        !$omp end parallel do
      end if
    end subroutine

  end subroutine TimeStepLength







  subroutine BuoyancyForce(W, Temperature, Moisture, Scalar, Pr)
    use WaterThermodynamics, only: LiquidWater, compute_liquid_water_content
    use BuoyantGases, only: enable_buoyant_scalars
    real(knd), dimension(-2:,-2:,-2:), contiguous, intent(inout) :: W
    real(knd), dimension(-2:,-2:,-2:), contiguous, intent(in) :: Temperature, Moisture
    real(knd), dimension(-2:,-2:,-2:,:), contiguous, intent(in) :: Scalar
    real(knd), contiguous, intent(in) :: Pr(-1:,-1:,-1:)
    real(knd) :: A, A2
    integer :: i, j, k
    ! Wolfram: InterpolatingPolynomial[{{-1.5,a},{-0.5,b},{0.5,c},{1.5,d}},x] /. x = 0
    real(knd), parameter :: C0 = 9._knd / 16, C1 = - 1._knd / 16


    if (enable_moisture) then
      if (enable_liquid) then
        call compute_liquid_water_content(Temperature, Moisture, Pr)
        A = grav_acc / temperature_ref
        A2 = A / 2._KND

        if (discretization_order==4) then
          call error_stop("not yet implemented")
        else
          call apply_liquid(1)
          call apply_liquid(2)
        end if
      else
        A = grav_acc / temperature_ref
        A2 = A / 2._KND

        if (discretization_order==4) then
          call error_stop("not yet implemented")
        else
          call apply_moist(1)
          call apply_moist(2)
        end if
      end if
    else if (enable_buoyant_scalars) then
        A = grav_acc / temperature_ref
        A2 = A / 2._KND
        
        if (discretization_order==4) then
          call error_stop("not yet implemented")
        else
          if (enable_buoyancy) then                    
            call apply_buoyant_scalars_temperature(1)
            call apply_buoyant_scalars_temperature(2)
          else
            call apply_buoyant_scalars(1)
            call apply_buoyant_scalars(2)
          end if
        end if
    else if (enable_buoyancy) then
      A = grav_acc / temperature_ref
      A2 = A / 2._KND
      if (discretization_order==4) then
        !$omp parallel do private(i,j,k)
        do k = 1, Wnz
          do j = 1, Wny
            do i = 1, Wnx
              !interpolation/deconvolution of Temperature: e.g. Morinishi et al. eq. 35 or in Hokpunna, Manhart, (2010), JCP 229
              W(i,j,k) = W(i,j,k) + &
                 A * ( C1 * Temperature(i,j,k+2) + &
                       C0 * Temperature(i,j,k+1) + &
                       C0 * Temperature(i,j,k)   + &
                       C1 * Temperature(i,j,k-1) ) - &
                A * temperature_ref
            end do
          end do
        end do
        !$omp end parallel do
      else
        !$omp parallel do private(i,j,k)
        do k = 1, Wnz
          do j = 1, Wny
            do i = 1, Wnx
              W(i,j,k) = W(i,j,k) + A2 * (Temperature(i,j,k+1)+Temperature(i,j,k)) - &
              A * temperature_ref
            end do
          end do
        end do
        !$omp end parallel do
      end if
    end if

    contains

      subroutine apply_moist(start)
        integer, intent(in) :: start
        integer :: i, j, k
        real(knd) :: temperature_virt

        !$omp parallel do private(i,j,k,temperature_virt)
        do k = start, Wnz+1,2
          do j = 1, Wny
            do i = 1, Wnx
              temperature_virt = theta_v(i,j,k)
              W(i,j,k)   = W(i,j,k)   + A2 * temperature_virt - A * temperature_ref
              W(i,j,k-1) = W(i,j,k-1) + A2 * temperature_virt
            end do
          end do
        end do
        !$omp end parallel do
      end subroutine

      subroutine apply_liquid(start)
        integer, intent(in) :: start
        integer :: i, j, k
        real(knd) :: temperature_virt

        !$omp parallel do private(i,j,k,temperature_virt)
        do k = start, Wnz+1,2
          do j = 1, Wny
            do i = 1, Wnx
              temperature_virt = theta_v_liquid(i,j,k)
              W(i,j,k)   = W(i,j,k)   + A2 * temperature_virt - A * temperature_ref
              W(i,j,k-1) = W(i,j,k-1) + A2 * temperature_virt
            end do
          end do
        end do
        !$omp end parallel do
      end subroutine

      subroutine apply_buoyant_scalars(start)
        integer, intent(in) :: start
        integer :: i, j, k
        real(knd) :: temperature_virt

        !$omp parallel do private(i,j,k,temperature_virt)
        do k = start, Wnz+1,2
          do j = 1, Wny
            do i = 1, Wnx
              temperature_virt = theta_v_scalars(i,j,k)
              W(i,j,k)   = W(i,j,k)   + A2 * temperature_virt - A * temperature_ref
              W(i,j,k-1) = W(i,j,k-1) + A2 * temperature_virt
            end do
          end do
        end do
        !$omp end parallel do
      end subroutine

      subroutine apply_buoyant_scalars_temperature(start)
        integer, intent(in) :: start
        integer :: i, j, k
        real(knd) :: temperature_virt

        !$omp parallel do private(i,j,k,temperature_virt)
        do k = start, Wnz+1,2
          do j = 1, Wny
            do i = 1, Wnx
              temperature_virt = theta_v_scalars_temperature(i,j,k)
              W(i,j,k)   = W(i,j,k)   + A2 * temperature_virt - A * temperature_ref
              W(i,j,k-1) = W(i,j,k-1) + A2 * temperature_virt
            end do
          end do
        end do
        !$omp end parallel do
      end subroutine

      pure real(knd) function theta_v(i, j, k)
        integer, intent(in) :: i, j, k

        theta_v = Temperature(i,j,k) * (1._knd + 0.61_knd * Moisture(i,j,k))
      end function

      real(knd) function theta_v_liquid(i, j, k)
        use PhysicalProperties
        integer, intent(in) :: i, j, k
        real(knd) :: theta, p, theta_t
        real(knd), parameter :: reference_pressure = 100000
        real(knd), parameter :: lw_min = 1e-8_knd
        
        if (LiquidWater(i,j,k) > lw_min) then
          !pressure
          p = reference_pressure_z(k) + Pr(i,j,k) * rho_air_ref
          
          !inv Exner
          theta_t = (reference_pressure / p)**(Rd_air_ref / Cp_air_ref)

          !pot. temperature from Betts
          theta = Temperature(i,j,k) + &
                  (Lv_water_ref / Cp_air_ref) * theta_t * LiquidWater(i,j,k)

          theta_v_liquid = theta * &
                    (1._knd + 0.61_knd * (Moisture(i,j,k) - LiquidWater(i,j,k)) -LiquidWater(i,j,k))
        else
          theta_v_liquid = theta_v(i, j, k)
        end if
                  
      end function
      
      real(knd) function theta_v_scalars(i,j,k)
        use BuoyantGases
        integer, intent(in) :: i, j, k
        real(knd) :: fact_sum
        integer :: iscal
        
        fact_sum = 0
        do iscal = 1, min(num_of_scalars, size(buoyant_scalars))
          fact_sum = fact_sum + &
              buoyant_scalars(iscal)%eps * mass_fraction(buoyant_scalars(iscal), Scalar(i,j,k,iscal))
        end do
        
        theta_v_scalars = temperature_ref * (1 + fact_sum)
      end function
      
      real(knd) function theta_v_scalars_temperature(i,j,k)
        use BuoyantGases
        integer, intent(in) :: i, j, k
        real(knd) :: fact_sum
        integer :: iscal
        
        fact_sum = 0
        do iscal = 1, min(num_of_scalars, size(buoyant_scalars))
          fact_sum = fact_sum + &
              buoyant_scalars(iscal)%eps * mass_fraction(buoyant_scalars(iscal), Scalar(i,j,k,iscal))
        end do
        
        theta_v_scalars_temperature = Temperature(i,j,k) * (1 + fact_sum)
      end function
  end subroutine BuoyancyForce





  subroutine CoriolisForce(U2, V2, U, V)
    real(knd), dimension(-2:,-2:,-2:), contiguous, intent(in)    :: U, V
    real(knd), dimension(-2:,-2:,-2:), contiguous, intent(inout) :: U2, V2
    integer :: i,j,k

    if (abs(Coriolis_parameter)>0) then
    !$omp parallel private(i,j,k)
    !$omp do
      do k = 1, Unz
        do j = 1, Uny
          do i = 1, Unx
            U2(i,j,k) = U2(i,j,k) + &
                  Coriolis_parameter * (V(i,j-1,k)+V(i+1,j-1,k)+V(i,j,k)+V(i+1,j,k))/4._knd
          end do
        end do
      end do
      !$omp end do nowait

      !$omp do
      do k = 1, Vnz
        do j = 1, Vny
          do i = 1, Vnx
            V2(i,j,k) = V2(i,j,k) - &
                  Coriolis_parameter * (U(i-1,j,k)+U(i-1,j+1,k)+U(i,j,k)+U(i,j+1,k))/4._knd
          end do
        end do
      end do
      !$omp end do
      !$omp end parallel
    end if
  end subroutine CoriolisForce





  subroutine SubgridStresses(U,V,W,Pr,Temperature,Moisture)
    use Subgrid, only: SubgridModel, sgstype, StabSubgridModel
    use ImmersedBoundary, only: ScalFlIBPoints, TIBPoint_Viscosity
    use Wallmodels
    use Scalars, only: ComputeTDiff

    real(knd), contiguous, intent(in) :: U(-2:,-2:,-2:),V(-2:,-2:,-2:),W(-2:,-2:,-2:)
    real(knd), contiguous, intent(in) :: Pr(-1:,-1:,-1:)
    real(knd), contiguous, intent(in) :: Temperature(-2:,-2:,-2:)
    real(knd), contiguous, intent(in) :: Moisture(-2:,-2:,-2:)
    integer :: i


    if (wallmodeltype>0) then
                      !resulting Viscosity intentionally overwritten
                      call ComputeViscsWM(U,V,W,Pr,Temperature,Moisture)
    else if (enable_buoyancy) then
                      call UpdateSurfaceTemperatures(U, V, W, Pr, Temperature, Moisture)
    end if
    

    call SubgridModel(U,V,W)


    if (wallmodeltype>0) then
                    call ComputeUVWFluxesWM(U, V, W, Pr, Temperature, Moisture)
    end if

    call BoundViscosity(Viscosity)

    do i = 1, size(ScalFlIBPoints)
      Viscosity(ScalFlIBPoints(i)%xi,ScalFlIBPoints(i)%yj,ScalFlIBPoints(i)%zk) =  &
                                              TIBPoint_Viscosity(ScalFlIBPoints(i),Viscosity)
    end do

    if (sgstype/=StabSubgridModel.and.enable_buoyancy)  call ComputeTDiff(U,V,W)

    if (size(TDiff)>0) call BoundViscosity(TDiff)

  end subroutine SubgridStresses






  subroutine CorrectFlowRate(U, V, W, dt)
    use custom_par

    real(knd), intent(inout), contiguous, dimension(-2:,-2:,-2:) :: U, V, W
    real(knd), intent(in) :: dt

    real(knd) :: rate_actual
    integer :: i, j, k

    if (flow_rate_x_fixed) then
      if (iim==nxims) then
        rate_actual = sum(U(Unx, 1:Uny, 1:Unz)) * dymin * dzmin
#ifdef PAR        
        rate_actual = par_co_sum_plane_yz(rate_actual)
#endif
      end if

#ifdef PAR        
      call par_broadcast_from_last_x(rate_actual)
#endif
      pr_gradient_x_dynamic = (rate_actual - flow_rate_x) / ( dymin * dzmin * gPrny * gPrnz)

      !$omp parallel do private(i,j,k)
      do k = 1, Unz
        do j = 1, Uny
          do i = 1, Unx
            U(i,j,k) = U(i,j,k) -  pr_gradient_x_dynamic
          end do
        end do
      end do
      !$omp end parallel do

      pr_gradient_x_dynamic = pr_gradient_x_dynamic / dt
    end if

      
    if (flow_rate_y_fixed) then
      if (jim==nyims) then
        rate_actual = sum(V(1:Vnx, Vny, 1:Vnz)) * dxmin * dzmin
#ifdef PAR        
        rate_actual = par_co_sum_plane_xz(rate_actual)
#endif
      end if

#ifdef PAR        
      call par_broadcast_from_last_y(rate_actual)
#endif
      pr_gradient_y_dynamic = (rate_actual - flow_rate_y) / (dxmin * dzmin * gPrnx * gPrnz)

      !$omp parallel do private(i,j,k)
      do k = 1, Vnz
        do j = 1, Vny
          do i = 1, Vnx
            V(i,j,k) = V(i,j,k) - pr_gradient_y_dynamic
          end do
        end do
      end do
      !$omp end parallel do

      pr_gradient_y_dynamic = pr_gradient_y_dynamic / dt
    end if

      
    if (flow_rate_z_fixed) then
      if (kim==nzims) then
        rate_actual = sum(W(1:Wnx, 1:Wny, Wnz)) * dxmin * dymin
#ifdef PAR        
        rate_actual = par_co_sum_plane_xy(rate_actual)
#endif
      end if

#ifdef PAR        
      call par_broadcast_from_last_z(rate_actual)
#endif
      pr_gradient_z_dynamic = (rate_actual - flow_rate_z) / (dxmin * dymin * gPrnx * gPrny)

      !$omp parallel do private(i,j,k)
      do k = 1, Wnz
        do j = 1, Wny
          do i = 1, Wnx
            W(i,j,k) = W(i,j,k) - pr_gradient_z_dynamic
          end do
        end do
      end do
      !$omp end parallel do

      pr_gradient_z_dynamic = pr_gradient_z_dynamic / dt
    end if
      
  end subroutine CorrectFlowRate



end module Dynamics
