module Initial

  use Parameters
  use ArrayUtilities, only: avg
  use Limiters, only: limiter_parameter, limiter_type
  use Multigrid, only: SetMGParams
  use Multigrid2d, only: SetMGParams2d
  use Pressure
  use Boundaries
  use ScalarBoundaries
  use Outputs, only: store, display, probes, scalar_probes, ReadProbes, &
                     ProfileSwitches, profiles_config, enable_profiles, &
                     CreateOutputDirectories
  use Scalars
  use Filters, only: filtertype, filter_ratios
  use Subgrid
  use Turbinlet, only: default_turbulence_generator, GetInletFromFile
  use SolarRadiation, only: InitSolarRadiation
  use SolidBodies, only: obstacles_file, &
                         InitSolidBodies
  use ImmersedBoundary, only: GetSolidBodiesBC, InitIBPFluxes!, SetIBPFluxes
  use VolumeSources!, only: InitVolumeSources, InitVolumeSourceBodies, ScalarFlVolume, ScalarFlVolumesContainer
  use AreaSources, only: InitAreaSources
  use LineSources, only: InitLineSources
  use PointSources, only: InitPointSources
  use Wallmodels
  use Tiling, only: tilesize,InitTiles
  use FreeUnit, only: newunit
  use Puffs, only: InitPuffSources
  use custom_par
#ifdef PAR
  use exchange_par
  use domains_bc_par
#endif

  implicit none

  private
  public  ReadConfiguration, InitialConditions, InitBoundaryConditions, probes_file, scalar_probes_file

  real(knd) :: x0,y0,z0 !domain boundaries, will become xU(0), yV(0), zW(0)
  real(knd) :: lx,ly,lz !domain extents

  character(80) :: probes_file = ""
  character(80) :: scalar_probes_file = ""

  type spline_coefs
    real(knd), allocatable :: z(:)
    real(knd), allocatable :: cu(:,:), cv(:,:)
  end type
    

contains


 subroutine ReadConfiguration
   use StaggeredFrames, only: rrange, TFrameTimes, TSaveFlags, &
                              TStaggeredFrameDomain,  AddDomain
   use PoisFFT, only: PoisFFT_NeumannStag, PoisFFT_DirichletStag, PoisFFT_Periodic
   use Sponge, only: enable_in_sponge_x, enable_out_sponge_x, enable_out_sponge_y, &
                     enable_top_sponge, enable_top_sponge_scalar
   integer ::  lmg,minmglevel,bnx,bny,bnz,mgncgc,mgnpre,mgnpost,mgmaxinnerGSiter
   real(knd) :: mgepsinnerGS
   integer ::  i, j, k, io, io2, itmp
   integer :: numframeslices
   real(knd) :: rtmp

   character(len = 1024) :: command_line, msg
   integer :: exenamelength
   integer :: unit

   type(rrange) :: range
   type(TFrameTimes) :: frame_times
   type(TSaveFlags) :: frame_save_flags
   character(10) :: domain_label
   integer :: num_staggered_domains
   integer :: number_of_probes, number_of_scalar_probes

   integer :: dimension,direction
   real(knd) :: position

   integer :: masssourc

   real(knd) :: Re = 70000, Prandtl = 0.7!1/molecular viscosity, viscosity/thermal diffusivity

   interface get
     procedure chget1
     procedure lget1, lget2, lget3
     procedure iget1, iget2, iget3
     procedure rget1, rget2, rget3
     procedure rgetv3
   end interface

   interface

     subroutine CustomConfiguration_First
     end subroutine

     subroutine CustomConfiguration_Last
     end subroutine

   end interface

   input_dir = "input/"
   output_dir = "output/"
   shared_output_dir = "output/"


   call newunit(unit)
  
   !try read input/output_dir from environment, command line has priority
   call get_io_dirs

!    the command line arguments can also be specified in cmd.conf
   call read_cmd_conf

   call parse_command_line


   !the actual command line has priority over cmd.conf
   call read_command_line
   !NOTE: it is parsed one more time lower in this subroutine
   call parse_command_line

#ifdef CUSTOM_CONFIG
   call CustomConfiguration_First
#endif

   open(unit,file="main.conf",status="old",action="read")
   call get(advection_method)
   
   if (discretization_order==4) advection_method=4
   
   call get(limiter_type)
   call get(limiter_parameter)

   call get(masssourc)
   masssourc = 1 !Seems to be necessary for stability, change with caution.
   enable_ibm_mass_sources = (masssourc>0)

   call get(steady)
   call get(task_type)
   call get(initcondsfromfile)
   call get(timeavg1)
   call get(timeavg2)
   call get(Re)
   if (master) write(*,*) "Re=",Re
   call get(eps)
   if (master) write(*,*) "eps=",eps
   call get(maxCNiter)
   if (master) write(*,*) "maxCNiter=",maxCNiter
   call get(epsCN)
   if (master) write(*,*) "epsCN=",epsCN
   call get(debugparam)
   if (master) write(*,*) "debug parameter=",debugparam
   close(unit)


   call get_les("les.conf")


   open(unit,file="grid.conf",status="old",action="read")
   call get(xgridfromfile)
   call get(ygridfromfile)
   call get(zgridfromfile)
   read(unit,fmt='(/)')

   read(unit,*) x0
   if (master) write(*,*) "x0=",x0
   call get(y0)
   if (master)write(*,*) "y0=",y0
   call get(z0)
   if (master) write(*,*) "z0=",z0
   read(unit,fmt='(/)')

   read(unit,*) lx
   if (lx>0) then
     if (master) write(*,*) "lx=",lx
   else
     if (master) write (*,*) "Domain length in x direction must be positive."
     call error_stop
   end if
   read(unit,fmt='(/)')

   read(unit,*) ly
   if (ly>0) then
     if (master) write(*,*) "ly=",ly
   else
     if (master) write (*,*) "Domain length in y direction must be positive."
     call error_stop
   end if
   read(unit,fmt='(/)')

   read(unit,*) lz
   if (lz>0) then
     if (master) write(*,*) "lz=",lz
   else
     if (master) write (*,*) "Domain length in z direction must be positive."
     call error_stop
   end if
   read(unit,fmt='(/)')

   read(unit,*) Prnx
   if (Prnx>0) then
     if (master) write(*,*) "nx=",Prnx
   else
     if (master) write (*,*) "Number of cells in x direction must be positive."
     call error_stop
   end if
   read(unit,fmt='(/)')

   read(unit,*) Prny
   if (Prny>0) then
     if (master) write(*,*) "ny=",Prny
   else
     if (master) write (*,*) "Number of cells in y direction must be positive."
     call error_stop
   end if
   read(unit,fmt='(/)')

   read(unit,*) Prnz
   if (Prnz>0) then
     if (master) write(*,*) "nz=",Prnz
   else
     if (master) write (*,*) "Number of cells in z direction must be positive."
     call error_stop
   end if
   close(unit)
   
   gxmin = x0
   gxmax = x0 + lx
   gymin = y0
   gymax = y0 + ly
   gzmin = z0
   gzmax = z0 + lz
   
   dxmin = lx/(Prnx)
   dymin = ly/(Prny)
   dzmin = lz/(Prnz)

   if (Prnx==1) then
     dxmin = sqrt(dymin*dzmin)
     lx = dxmin
   elseif (Prny==1) then
     dymin = sqrt(dxmin*dzmin)
     ly = dymin
   elseif (Prnz==1) then
     dzmin = sqrt(dxmin*dymin)
     lz = dzmin
   end if
   
   
    
   if (zgridfromfile) then           
      open(unit=unit,file="zgrid.txt", status="old", action="read")
      j = -1
      do
        read (unit,*,iostat = io) rtmp
        if (io==0) then
          j = j+1
        else
          exit
        end if
      end do
      rewind(unit)
      Prnz = j
      allocate(gzW(0:Prnz))
      do j = 0, Prnz
        read (unit,*) gzW(j)
      end do
      close(unit)
      
      z0 = gzW(0)
      lz = gzW(Prnz) - gzW(0)
      
      gzmin = gzW(0)
      gzmax = gzW(Prnz)
      
      dzmin = minval(gzW(1:Prnz)-gzW(0:Prnz-1))
   else
      allocate(gzW(0:Prnz))
      gzW(:) = [(k*dzmin + gzmin, k = 0, Prnz)]
   end if
   
   allocate(gxU(0:Prnx))
   allocate(gyV(0:Prny))
   gxU(:) = [(i*dxmin + gxmin, i = 0, Prnx)]
   gyV(:) = [(j*dymin + gymin, j = 0, Prny)]

   gxPr = [( (gxU(i)+gxU(i-1)) / 2, i = 1, Prnx)]
   gyPr = [( (gyV(j)+gyV(j-1)) / 2, j = 1, Prny)]
   gzPr = [( (gzW(k)+gzW(k-1)) / 2, k = 1, Prnz)]
  

   !can be overwritten after MPI decomposition
   gPrnx = Prnx
   gPrny = Prny
   gPrnz = Prnz   



   if (master) then
      write(*,*) "dxmin ",dxmin
      write(*,*) "dymin ",dymin
      write(*,*) "dzmin ",dzmin

      write(*,*) "lx:",lx
      write(*,*) "ly:",ly
      write(*,*) "lz:",lz
   end if

   

   call get_boundary_conditions_velocity("boundconds.conf")

   open(unit,file="large_scale.conf",status="old",action="read",iostat = io)
   if (io==0) then

     call get(Coriolis_parameter)
     if (master) write(*,*) "Coriolis_parameter=", Coriolis_parameter

     call get(pr_gradient_x)
     if (master) write(*,*) "pr_gradient_x=", pr_gradient_x
     enable_pr_gradient_x_uniform = pr_gradient_x /= 0

     call get(pr_gradient_y)
     if (master) write(*,*) "pr_gradient_y=", pr_gradient_y
     enable_pr_gradient_y_uniform = pr_gradient_y /= 0

     call get(SubsidenceGradient)
     if (master) write(*,*) "SubsidenceGradient=",SubsidenceGradient
     close(unit)

   else

     if (master) write(*,*) "Warning! Could not open file large_scale.conf. Using defaults."

   end if


   open(unit,file="thermal.conf",status="old",action="read",iostat = io)
   if (io==0) then
     call get(enable_buoyancy)
     call get(Prandtl)
     call get(grav_acc)
     call get(temperature_ref)
     call get(TempBtype(We))
     call get(TempBtype(Ea))
     call get(TempBtype(So))
     call get(TempBtype(No))
     call get(TempBtype(Bo))
     call get(TempBtype(To))
     call get(sideTemp(We))
     call get(sideTemp(Ea))
     call get(sideTemp(So))
     call get(sideTemp(No))
     call get(sideTemp(Bo))
     call get(sideTemp(To))
     close(unit)
   else
     enable_buoyancy = .false.
   end if

   if (enable_buoyancy) then

     open(unit,file="temp_profile.conf",status="old",action="read",iostat = io)

     if (io==0) then
       call get(TemperatureProfileObj%randomize)
       call get(TemperatureProfileObj%randomizeTop)
       call get(TemperatureProfileObj%randomizeAmplitude)
       call get(itmp)

       allocate(TemperatureProfileObj%Sections(max(itmp,0)))

       do i = 1, size(TemperatureProfileObj%Sections)
         call get(TemperatureProfileObj%Sections(i)%top)
         call get(TemperatureProfileObj%Sections(i)%jump)
         call get(TemperatureProfileObj%Sections(i)%gradient)
       end do

       close(unit)

     else

       if (master) write(*,*) "Warning! Could not open file temp_profile.conf. Using defaults."
       TemperatureProfileObj%randomize = .false.

       allocate(TemperatureProfileObj%Sections(0))

     end if

   else

     TempBtype = 0

   end if



   open(unit,file="moisture.conf",status="old",action="read",iostat = io)
   if (io==0) then
     call get(enable_moisture)
     call get(moisture_ref)
     call get(MoistBtype(We))
     call get(MoistBtype(Ea))
     call get(MoistBtype(So))
     call get(MoistBtype(No))
     call get(MoistBtype(Bo))
     call get(MoistBtype(To))
     call get(sideMoist(We))
     call get(sideMoist(Ea))
     call get(sideMoist(So))
     call get(sideMoist(No))
     call get(sideMoist(Bo))
     call get(sideMoist(To))
     close(unit)
   else
     enable_moisture = .false.
   end if

   if (enable_moisture) then

     open(unit,file="moist_profile.conf",status="old",action="read",iostat = io)

     if (io==0) then
       call get(MoistureProfileObj%randomize)
       call get(MoistureProfileObj%randomizeTop)
       call get(MoistureProfileObj%randomizeAmplitude)
       call get(itmp)

       allocate(MoistureProfileObj%Sections(max(itmp,0)))

       do i = 1, size(MoistureProfileObj%Sections)
         call get(MoistureProfileObj%Sections(i)%top)
         call get(MoistureProfileObj%Sections(i)%jump)
         call get(MoistureProfileObj%Sections(i)%gradient)
       end do

       close(unit)

     else

       if (master) write(*,*) "Warning! Could not open file moist_profile.conf. Using defaults."
       MoistureProfileObj%randomize = .false.

       allocate(MoistureProfileObj%Sections(0))

     end if

   else

     MoistBtype = 0

   end if
   
   
   ! Liquid water cannot be allowed if either of buoyancy and moisture is not.
   if (.not.(enable_buoyancy.and.enable_moisture)) enable_liquid = .false.

   
!    open(unit,file="inlet.conf",status="old",action="read")
!    call get(inlettype)
!    call get(profiletype)
!    call get(ShearInletTypeParameter)
!    if (master) write(*,*) "G=",ShearInletTypeParameter
!    call get(Uinlet_vec)
!    Uinlet = norm2(Uinlet_vec)
!    if (master) write(*,*) "Uinlet=",Uinlet_vec
!    call get(default_turbulence_generator%Ustar_surf_inlet)  !-<u'w'>
!    call get(default_turbulence_generator%stress_gradient_inlet) !in relative part per 1m
!    call get(default_turbulence_generator%z0_inlet)
!    call get(default_turbulence_generator%power_exponent_inlet)
!    call get(default_turbulence_generator%z_ref_inlet)
!    call get(default_turbulence_generator%U_ref_inlet)
!    call get(default_turbulence_generator%relative_stress(1,1))
!    call get(default_turbulence_generator%relative_stress(2,2))
!    call get(default_turbulence_generator%relative_stress(3,3))
!    call get(default_turbulence_generator%relative_stress(1,2))
!    call get(default_turbulence_generator%relative_stress(1,3))
!    call get(default_turbulence_generator%relative_stress(2,3))
!    call get(default_turbulence_generator%T_Lag)
!    call get(default_turbulence_generator%L_y)
!    call get(default_turbulence_generator%L_z)
!    close(unit)
!
!    default_turbulence_generator%relative_stress(2,1) = default_turbulence_generator%relative_stress(1,2)
!    default_turbulence_generator%relative_stress(3,1) = default_turbulence_generator%relative_stress(1,3)
!    default_turbulence_generator%relative_stress(3,2) = default_turbulence_generator%relative_stress(2,3)
!
!    if (Btype(We)==BC_TURBULENT_INLET .or. Btype(Ea)==BC_TURBULENT_INLET) then
!      default_turbulence_generator%direction = 1
!    else if (Btype(So)==BC_TURBULENT_INLET .or. Btype(No)==BC_TURBULENT_INLET) then
!      default_turbulence_generator%direction = 2
!    else
!      default_turbulence_generator%direction = 0
!    end if

   open(unit,file="scalars.conf",status="old",action="read",iostat = io)
   if (io==0) then
     call get(num_of_scalars)
     call get(computedeposition)
     call get(computegravsettling)
     call get(partdistrib)
     call get(totalscalsource)
     call get(scalsourcetype)

     call get(ScalBtype(We))
     call get(ScalBtype(Ea))
     call get(ScalBtype(So))
     call get(ScalBtype(No))
     call get(ScalBtype(Bo))
     call get(ScalBtype(To))
     call get(sideScal(We))
     call get(sideScal(Ea))
     call get(sideScal(So))
     call get(sideScal(No))
     call get(sideScal(Bo))
     call get(sideScal(To))

     if (scalsourcetype==1) then
       if (partdistrib>0) then

          allocate(partdiam(partdistrib),partrho(partdistrib),percdistrib(partdistrib))

          do i = 1, partdistrib
            call get(partdiam(i))
            call get(partrho(i))
            call get(percdistrib(i))
          end do

       else

          allocate(partdiam(num_of_scalars),partrho(num_of_scalars),percdistrib(num_of_scalars))

          do i = 1, num_of_scalars
            call get(partdiam(i))
            call get(partrho(i))
            call get(percdistrib(i))
          end do
       end if
     end if
     close(unit)
   else
     num_of_scalars = 0
     if (master) write (*,*) "scalars.conf not found, no passive scalars for computation."
   end if
   
   
   call get_buoyant_scalars("buoyant_scalars.conf")


   if (num_of_scalars>0) then
      open(unit, file="line_sources.conf",status="old",action="read",iostat=io)
      
      if (io==0) then
        call get_line_sources
        close(unit)
      end if
   end if

   call get_area_sources("area_sources.conf")

   if (num_of_scalars>0) then
      open(unit, file="point_sources.conf",status="old",action="read",iostat=io)
      
      if (io==0) then
        call get_point_sources
        close(unit)
      end if
   end if

   call get_scalar_zones("scalar_removal_zones.conf")

   if (pressure_solution%poisson_solver==POISSON_SOLVER_MULTIGRID) then
     open(unit,file="mgopts.conf",status="old",action="read")
     call get(lmg)
     call get(minmglevel)
     call get(bnx)
     call get(bny)
     call get(bnz)
     call get(mgncgc)
     call get(mgnpre)
     call get(mgnpost)
     call get(mgmaxinnerGSiter)
     call get(mgepsinnerGS)
     close(unit)

     if (Prny==1) then
      call SetMGParams2d(llmg = lmg,lminmglevel = minmglevel,lbnx = bnx,lbnz = bnz,&
                         lmgncgc = mgncgc,lmgnpre = mgnpre,lmgnpost = mgnpost,&
                         lmgmaxinnerGSiter = mgmaxinnerGSiter,lmgepsinnerGS = mgepsinnerGS)
     else
      call SetMGParams(llmg = lmg,lminmglevel = minmglevel,&
                         lbnx = bnx,lbny = bny,lbnz = bnz,&
                         lmgncgc = mgncgc,lmgnpre = mgnpre,lmgnpost = mgnpost,&
                         lmgmaxinnerGSiter = mgmaxinnerGSiter,lmgepsinnerGS = mgepsinnerGS)
     end if
   end if


  

   open(unit,file="output.conf",status="old",action="read",iostat = io)
   if (io==0) then
     call read_namelist_output

     if (.not.enable_buoyancy) then
       store%out_temperature = 0
       store%out_moisture = 0
     end if

     if (.not.enable_buoyancy) then
       store%avg_temperature = 0
       store%avg_moisture = 0
     end if

     if (num_of_scalars < 1) then
       store%scalars = 0
       store%scalars_avg = 0
       store%scalsum_time = 0
       store%scaltotsum_time = 0
     end if
     close(unit)
   else
     if (master) write(*,*) "No output.conf found, defaults will be used."
   end if
   
   
   call get_profiles("profiles.conf", profiles_config, enable_profiles)

   !probes_file and scalar_probes_file read from command line
   if (probes_file == "" .and. scalar_probes_file == "") then

     open(unit,file="probes.conf",status="old",action="read",iostat = io)
     if (io==0) then
       call get(number_of_probes)

       allocate(probes(number_of_probes))

       do i = 1, number_of_probes
         call get(probes(i)%x, probes(i)%y, probes(i)%z)
       end do

       scalar_probes = probes

       close(unit)

     else
       allocate(probes(0))
       allocate(scalar_probes(0))
     end if

   else

     if (len_trim(probes_file)>0) then
       call ReadProbes(probes,number_of_probes,probes_file)
     else
       allocate(probes(0))
     end if

     if (len_trim(scalar_probes_file)>0) then
       call ReadProbes(scalar_probes,number_of_scalar_probes,scalar_probes_file)
     else
       allocate(scalar_probes(0))
     end if
   end if


   if (master) write(*,*) "num_of_scalars", num_of_scalars

   call parse_command_line

   if (master) then

     write(*,*) "Boundaries:"

     write(*,'(a2)',advance='no') " W "
     select case (Btype(We))
       case (BC_NOSLIP)
         write(*,*) "noslip"
       case (BC_FREESLIP)
         write(*,*) "freeslip"
       case (BC_PERIODIC)
         write(*,*) "periodic"
       case (BC_DIRICHLET)
         write(*,*) "dirichlet"
       case (BC_NEUMANN)
         write(*,*) "neumann"
       case default
         write(*,*) "other", Btype(We)
     endselect

     write(*,'(a2)',advance='no') " E "
     select case (Btype(Ea))
       case (BC_NOSLIP)
         write(*,*) "noslip"
       case (BC_FREESLIP)
         write(*,*) "freeslip"
       case (BC_PERIODIC)
         write(*,*) "periodic"
       case (BC_DIRICHLET)
         write(*,*) "dirichlet"
       case (BC_NEUMANN)
         write(*,*) "neumann"
       case default
         write(*,*) "other", Btype(Ea)
     endselect

     write(*,'(a2)',advance='no') " S "
     select case (Btype(So))
       case (BC_NOSLIP)
         write(*,*) "noslip"
       case (BC_FREESLIP)
         write(*,*) "freeslip"
       case (BC_PERIODIC)
         write(*,*) "periodic"
       case (BC_DIRICHLET)
         write(*,*) "dirichlet"
       case (BC_NEUMANN)
         write(*,*) "neumann"
       case default
         write(*,*) "other", Btype(So)
     endselect

     write(*,'(a2)',advance='no') " N "
     select case (Btype(No))
       case (BC_NOSLIP)
         write(*,*) "noslip"
       case (BC_FREESLIP)
         write(*,*) "freeslip"
       case (BC_PERIODIC)
         write(*,*) "periodic"
       case (BC_DIRICHLET)
         write(*,*) "dirichlet"
       case (BC_NEUMANN)
         write(*,*) "neumann"
       case default
         write(*,*) "other", Btype(No)
     endselect

     write(*,'(a2)',advance='no') " B "
     select case (Btype(Bo))
       case (BC_NOSLIP)
         write(*,*) "noslip"
       case (BC_FREESLIP)
         write(*,*) "freeslip"
       case (BC_PERIODIC)
         write(*,*) "periodic"
       case (BC_DIRICHLET)
         write(*,*) "dirichlet"
       case (BC_NEUMANN)
         write(*,*) "neumann"
       case default
         write(*,*) "other", Btype(Bo)
     endselect

     write(*,'(a2)',advance='no') " T "
     select case (Btype(To))
       case (BC_NOSLIP)
         write(*,*) "noslip"
       case (BC_FREESLIP)
         write(*,*) "freeslip"
       case (BC_PERIODIC)
         write(*,*) "periodic"
       case (BC_DIRICHLET)
         write(*,*) "dirichlet"
       case (BC_NEUMANN)
         write(*,*) "neumann"
       case default
         write(*,*) "other", Btype(To)
     endselect
     
     if ((Btype(We)==BC_PERIODIC.or.Btype(Ea)==BC_PERIODIC).and.Btype(We)/=Btype(Ea)) &
       call error_stop("Error: Both X boundary conditions must be periodic or not periodic.")
     if ((Btype(So)==BC_PERIODIC.or.Btype(No)==BC_PERIODIC).and.Btype(So)/=Btype(No)) &
       call error_stop("Error: Both Y boundary conditions must be periodic or not periodic.")
     if ((Btype(Bo)==BC_PERIODIC.or.Btype(To)==BC_PERIODIC).and.Btype(Bo)/=Btype(To)) &
       call error_stop("Error: Both Z boundary conditions must be periodic or not periodic.")

   end if



   if (Re > 0) then
     molecular_viscosity = 1 / Re
   else
     molecular_viscosity = 0
   end if

   molecular_diffusivity = molecular_viscosity / Prandtl



   if (Btype(We)==BC_TURBULENT_INLET) inlettype = TurbulentInletType
   if (Btype(We)==BC_INLET_FROM_FILE) inlettype = FromFileInletType


   call get_time_stepping("time_stepping.conf", time_stepping)

   if (timeavg2>=timeavg1) then
     averaging = 1
   else
     averaging = 0
   end if

   if (.not.xgridfromfile.and..not.ygridfromfile.and..not.zgridfromfile) then
     gridtype = GRID_UNIFORM
     if (master) write(*,*) "Uniform grid"
   else if  (.not.xgridfromfile.and..not.ygridfromfile) then
     gridtype = GRID_VARIABLE_Z
     if (master) write(*,*) "Grid variable in z"
   else
     gridtype = GRID_GENERAL
     if (master) write(*,*) "General grid not supported."; call error_stop
   end if

   !Btype and PrBtype might get overwritten by MPI procedures
   do i = We, To
      if (.not.PrBtype_explicitly_set(i)) then
        if (Btype(i)==BC_PERIODIC) then
          PrBtype(i) = BC_PERIODIC
        else
          PrBtype(i) = BC_NEUMANN
        end if
      end if
   end do
   
   do i = We, To
      if (PrBtype(i)==BC_PERIODIC) then
        PoissonBtype(i) = PoisFFT_PERIODIC
      else if (PrBtype(i)==BC_DIRICHLET) then
        PoissonBtype(i) = PoisFFT_DirichletStag
      else
        PoissonBtype(i) = PoisFFT_NeumannStag
      end if
   end do
   

   im_xmin = gxmin
   im_ymin = gymin
   im_zmin = gzmin

   im_xmax = gxmax
   im_ymax = gymax
   im_zmax = gzmax

#ifdef PAR
   call get_domains("domains.conf")

   call par_init_grid
   
   call par_init_boundaries
   
   x0 = im_xmin
   y0 = im_ymin
   z0 = im_zmin
#endif


   call init_random_seed



   !both procedures below use the name of the output directory (affected by MPI)
   call get_frames("frames.conf")
   
   call read_staggered_frames


   if (Btype(Ea)==BC_PERIODIC .or. &
       (Btype(Ea)>=BC_MPI_BOUNDS_MIN.and.Btype(Ea)<=BC_MPI_BOUNDS_MAX)) then
                          Unx = Prnx
   else
                          Unx = Prnx-1
   end if
   Uny = Prny
   Unz = Prnz

   Vnx = Prnx
   if (Btype(No)==BC_PERIODIC .or. &
       (Btype(No)>=BC_MPI_BOUNDS_MIN.and.Btype(No)<=BC_MPI_BOUNDS_MAX)) then
                          Vny = Prny
   else
                          Vny = Prny-1
   end if
   Vnz = Prnz

   Wnx = Prnx
   Wny = Prny
   if (Btype(To)==BC_PERIODIC .or. &
       (Btype(To)>=BC_MPI_BOUNDS_MIN.and.Btype(To)<=BC_MPI_BOUNDS_MAX)) then
                          Wnz = Prnz
   else
                          Wnz = Prnz-1
   end if
   
   if (Unx<=0) then
#ifdef PAR
     if (enable_multiple_domains) &
       write(*,*) "Error on domain:",domain_index
     write(*,*) "Error on image:",iim,jim,kim
#endif
     call error_stop("Error: Unx must be larger than zero, &
       &check the number of grid cells and the boundary conditions in the x direction.")
   end if
   
   if (Vny<=0) then   
#ifdef PAR
     if (enable_multiple_domains) &
       write(*,*) "Error on domain:",domain_index
     write(*,*) "Error on image:",iim,jim,kim
#endif
     call error_stop("Error: Vny must be larger than zero, &
       &check the number of grid cells and the boundary conditions in the y direction.")
   end if
   
   if (Wnz<=0) then   
#ifdef PAR
     if (enable_multiple_domains) &
       write(*,*) "Error on domain:",domain_index
     write(*,*) "Error on image:",iim,jim,kim
#endif
     call error_stop("Error: Wnz must be larger than zero, &
       &check  the number of grid cells and the boundary conditions in the z direction.")
   end if
   
   
#ifdef PAR
   call par_init_exchange
      
   gUnx = par_co_sum(Unx, comm_row_x)
   gUny = par_co_sum(Uny, comm_row_y)
   gUnz = par_co_sum(Unz, comm_row_z)
   
   gVnx = par_co_sum(Vnx, comm_row_x)
   gVny = par_co_sum(Vny, comm_row_y)
   gVnz = par_co_sum(Vnz, comm_row_z)
   
   gWnx = par_co_sum(Wnx, comm_row_x)
   gWny = par_co_sum(Wny, comm_row_y)
   gWnz = par_co_sum(Wnz, comm_row_z)
#else
   gUnx = Unx
   gUny = Uny
   gUnz = Unz
   
   gVnx = Vnx
   gVny = Vny
   gVnz = Vnz
   
   gWnx = Wnx
   gWny = Wny
   gWnz = Wnz 
#endif

   call InitFlowRatesBC

#ifdef PAR
   call par_init_domain_boundary_conditions
#endif
  

#ifdef CUSTOM_CONFIG
   call CustomConfiguration_Last
#endif

   if (master) write(*,*) "set"
   
   

 contains
 
 
 

     subroutine chget1(x)
       character(*),intent(out) :: x
       read(unit,fmt='(/)')
       read(unit,*) x
     end subroutine
     subroutine lget1(x)
       logical,intent(out) :: x
       character(120) :: line, fname
       integer :: ierr, tmp
       read(unit,fmt='(/)')
       read(unit,'(a)') line
       read(line,*,iostat=ierr) x
       if (ierr/=0) then
         read(line,*,iostat=ierr) tmp
         if (ierr/=0) then
           inquire(unit,name=fname)
           if (master) write(*,*) "Stop expected a boolean flag in file "//trim(fname)
           if (master) write(*,*) "Received '"//trim(line)//"' instead."
           call error_stop
         end if
         x = tmp /=0
       end if
     end subroutine
     subroutine lget2(x,y)
       logical,intent(out) :: x,y
       read(unit,fmt='(/)')
       read(unit,*) x,y
     end subroutine
     subroutine lget3(x,y,z)
       logical,intent(out) :: x,y,z
       read(unit,fmt='(/)')
       read(unit,*) x,y,z
     end subroutine
     subroutine iget1(x)
       integer,intent(out) :: x
       read(unit,fmt='(/)')
       read(unit,*) x
     end subroutine
     subroutine iget2(x,y)
       integer,intent(out) :: x,y
       read(unit,fmt='(/)')
       read(unit,*) x,y
     end subroutine
     subroutine iget3(x,y,z)
       integer,intent(out) :: x,y,z
       read(unit,fmt='(/)')
       read(unit,*) x,y,z
     end subroutine
     subroutine rget1(x)
       real(knd),intent(out) :: x
       read(unit,fmt='(/)')
       read(unit,*) x
     end subroutine
     subroutine rget2(x,y)
       real(knd),intent(out) :: x,y
       read(unit,fmt='(/)')
       read(unit,*) x,y
     end subroutine
     subroutine rget3(x,y,z)
       real(knd),intent(out) :: x,y,z
       read(unit,fmt='(/)')
       read(unit,*) x,y,z
     end subroutine
     subroutine rgetv3(v)
       real(knd),intent(out) :: v(3)
       read(unit,fmt='(/)')
       read(unit,*) v
     end subroutine

     subroutine read_cmd_conf
        command_line = ""
        open(unit,file="cmd.conf",status="old",action="read",iostat = io)
        if (io==0) then
          read(unit,'(a)') command_line
        end if
        command_line = "&cmd "//adjustl(trim(command_line))//" /"
     end subroutine

     subroutine read_command_line
       command_line = ""
       call get_command(command = command_line,status = io)
       if (io==0) then
         call get_command_argument(0,length = exenamelength,status = io2)
         if (io2==0) then
           command_line = "&cmd "//adjustl(trim(command_line(exenamelength+1:)))//" /"
         else
           command_line = "&cmd "//adjustl(trim(command_line))//" /"
         end if
       else
         write(*,*) io,"Error getting command line."
       end if
     end subroutine

     subroutine parse_command_line
      use Strings, only: itoa
#ifdef PAR
       use custom_par
#endif

       use ScalarAdvection, only: enable_correct_divergence_scalar

       namelist /cmd/ tilesize, debugparam, debuglevel, windangle, &
                       Prnx, Prny, Prnz,&
#ifdef PAR
                       npxyz, domain_index, number_of_domains, &
#endif
                       obstacles_file, probes_file, scalar_probes_file, input_dir, output_dir, &
                       enable_fixed_flow_rate, set_U_bulk, U_bulk_vec, &
                       enable_in_sponge_x, enable_out_sponge_x, enable_out_sponge_y, &
                       enable_top_sponge, enable_top_sponge_scalar, &
                       enable_liquid, &
                       discretization_order, &
                       explicit_diffusion, explicit_scalar_diffusion, &
                       enable_correct_divergence_scalar

       if (len_trim(command_line)>0) then
         msg = ''
         read(command_line,nml = cmd,iostat = io,iomsg = msg)
         if (io/=0) then
           if (master) write(*,*) "Command line: ",trim(command_line)
           call error_stop("Error parsing the command line or cmd.conf, error: "// itoa(io) // &
                           " " // msg)
         end if
       else
         call error_stop("Error reading the command line or cmd.conf, error: "// itoa(io))
       end if
     end subroutine


     subroutine get_line_sources
        use Strings, only: itoa
        use LineSources, only: ScalarLineSource, ScalarLineSources
        type(ScalarLineSource) :: src
        integer :: n

        call get(n)
        allocate(ScalarLineSources(0))
        do i = 1, n
          read(unit,fmt=*)
          call get(src%scalar_number)
          if (src%scalar_number<0) call error_stop("Error: Scalar number of line source "//itoa(i)//" negative.")
          if (src%scalar_number>num_of_scalars) call error_stop("Error: Scalar number of line source "//itoa(i)//" too large.")
          
          call get(src%start)
          call get(src%end)
          call get(src%flux)
          src%number_of_points = 20 * max(Prnx,Prny,Prnz)
          ScalarLineSources = [ScalarLineSources, src]
        end do
     end subroutine

     subroutine get_point_sources
        use Strings, only: itoa
        use PointSources, only: ScalarPointSource, ScalarPointSources
        type(ScalarPointSource) :: src
        integer :: n

        call get(n)
        allocate(ScalarPointSources(0))
        do i = 1, n
          read(unit,fmt=*)
          call get(src%scalar_number)
          if (src%scalar_number<0) call error_stop("Error: Scalar number of point source "//itoa(i)//" negative.")
          if (src%scalar_number>num_of_scalars) call error_stop("Error: Scalar number of point source "//itoa(i)//" too large.")
          
          call get(src%position)
          call get(src%flux)
          ScalarPointSources = [ScalarPointSources, src]
        end do
     end subroutine

     subroutine read_namelist_output
       namelist /output/ store, display

       read(unit,nml = output,iostat = io,iomsg = msg)
       if (io /= 0) then
         write(*,*) "Error reading namelist from output.conf:"
         write(*,*) trim(msg)
         stop
       end if
     end subroutine
     
     subroutine read_staggered_frames
       open(unit,file="stagframes.conf",status="old",action="read",iostat = io)
       if (io==0) then
         call get(num_staggered_domains)

         do i = 1, num_staggered_domains
           read(unit,fmt=*)
           call get(domain_label)
           call get(range%min%x,range%max%x)
           call get(range%min%y,range%max%y)
           call get(range%min%z,range%max%z)
           call get(frame_times%nframes)
           call get(frame_times%start, frame_times%end)
           call get(frame_save_flags%U, frame_save_flags%V, frame_save_flags%W)
           call get(frame_save_flags%Pr)
           call get(frame_save_flags%Viscosity)
           call get(frame_save_flags%Temperature)
           if (.not.enable_buoyancy) frame_save_flags%Temperature = .false.
           call get(frame_save_flags%Moisture)
           if (.not.enable_moisture) frame_save_flags%Moisture = .false.
           call get(frame_save_flags%Scalar)
           if (num_of_scalars < 1) frame_save_flags%Scalar = .false.

           call AddDomain(TStaggeredFrameDomain(trim(domain_label), &
                                                range, &
                                                frame_times, &
                                                frame_save_flags))
         end do
         close(unit)
       else
         if (master) write (*,*) "stagframes.conf not found, no staggered frames will be saved."
       end if
     end subroutine read_staggered_frames

  end subroutine ReadConfiguration
  
  
  

  subroutine get_boundary_conditions_velocity(fname)
    use Strings
    use ParseTrees
    character(*), intent(in) :: fname
    type(tree_object), allocatable :: tree(:)
    logical :: ex
    integer :: iobj, stat
    
    real(knd), target :: Uinlet_conf(3)

    Uinlet_conf = 0
    Uinlet = -1

    Btype = 0
    
    inquire(file=fname, exist=ex)

    if (.not.ex) then
      write(*,*) "Error, '",trim(fname),"' does not found."
      call error_stop()
    end if

    call parse_file(tree, fname, stat)

    if (stat==0) then

      if (allocated(tree)) then
        do iobj = 1, size(tree)
          if (downcase(tree(iobj)%name)=="bc") then
            call get_bc(tree(iobj))
          else if (downcase(tree(iobj)%name)=="ic") then
            call get_ic(tree(iobj))
          end if
          call tree(iobj)%finalize
        end do
      end if

    else

      write(*,*) "Error parsing file " // fname
      call error_stop

    end if

    if (any(Btype<=0)) then
      write(*,*) "Error, non-positive boundary type value."
      write(*,*) "Btype:", Btype
      call error_stop
    end if
  
  contains
  
    subroutine get_bc(obj)
      type(tree_object), intent(in) :: obj
      
      character(char_len), target :: selected_side_str, type_str
      character(char_len), allocatable :: selected_sides_str(:)
      type(tree_object_ptr), target :: stg_properties_obj
      real(knd) :: u_vec(3)
      real(knd) :: z0
      
      character(*), dimension(6), parameter :: sides_str = &
        [character(len("bottom")) :: "west", "east", "south", "north", "bottom", "top"]
      logical :: side_selected, sides_selected
      integer, allocatable :: selected_sides(:)
      integer :: bc_type
      
      
      integer :: stat
      integer :: iside
      type(field_names) :: names(4)
      type(field_names_a) :: names_a(1)
      type(field_names_a_str_alloc) :: names_a_str_alloc(1)
    
      
      names = [field_names_init("side", selected_side_str), &
               field_names_init("type", type_str), &
               field_names_init("z0", z0), &
               field_names_init("stg", stg_properties_obj)]
                
      names_a = [field_names_a_init("u", u_vec)]
      
      names_a_str_alloc = [field_names_a_str_alloc("sides")]
      
      
      u_vec = 0
      z0 = 0
      selected_side_str = ""
      type_str = ""
      side_selected = .false.
      sides_selected = .false.
      bc_type = 0
      
      call get_object_field_values(obj, stat, names, fields_a_str_alloc=names_a_str_alloc)
      
      if (stat/=0) then
        write(*,*) "Error parsing a bc() object in " // fname
        write(*,*) "Error status", stat
        call error_stop()
      end if
      
      side_selected = len_trim(selected_side_str) > 0
      
      sides_selected = allocated(names_a_str_alloc(1)%var)
      
      if (side_selected.and.sides_selected) then
        write(*,*) "Error. Only one of `side=` and `sides=` can be specified in a bc object in " // fname
        call error_stop()
      else if (.not.(side_selected.or.sides_selected)) then
        write(*,*) "Error. One of `side=` or `sides=` must be specified in a bc object in " // fname
        call error_stop()
      else if (side_selected) then
        allocate(selected_sides(1))

        select case(downcase(selected_side_str(1:1)))
          case ("w")
            selected_sides(1) = 1
          case ("e")
            selected_sides(1) = 2
          case ("s")
            selected_sides(1) = 3
          case ("n")
            selected_sides(1) = 4
          case ("b")
            selected_sides(1) = 5
          case ("t")
            selected_sides(1) = 6
          case default
            write(*,*) "Error. Unknown boundary side in " // fname
            call error_stop()
        end select
      else if (sides_selected) then
        call move_alloc(names_a_str_alloc(1)%var, selected_sides_str)
        allocate(selected_sides(size(selected_sides_str)))
        if (size(selected_sides)<=0 .or. size(selected_sides)>6) then
          write(*,*) "Error. Selected_sizes array must contain from 1 to 6 elements in " // fname
          call error_stop()
        end if
        do iside = 1, size(selected_sides)
          select case(downcase(selected_sides_str(iside)(1:1)))
            case ("w")
              selected_sides(iside) = 1
            case ("e")
              selected_sides(iside) = 2
            case ("s")
              selected_sides(iside) = 3
            case ("n")
              selected_sides(iside) = 4
            case ("b")
              selected_sides(iside) = 5
            case ("t")
              selected_sides(iside) = 6
            case default
              write(*,*) "Error. Unknown boundary side in " // fname
              call error_stop()
          end select
        end do
      else
        write(*,*) "Internal error when selecting side in " // fname
        call error_stop()
      end if
      
      if (len_trim(type_str)==0) then
        write(*,*) "Error. The boundary type must be provided in " // fname
        call error_stop()
      end if
    
      select case(downcase(type_str))
        case ("periodic")
          bc_type = BC_PERIODIC
        case ("noslip")
          bc_type = BC_NOSLIP
        case ("freeslip")
          bc_type = BC_FREESLIP
        case ("dirichlet")
          bc_type = BC_DIRICHLET
        case ("neumann")
          bc_type = BC_NEUMANN
        case ("turbulent_inlet")
          bc_type = BC_TURBULENT_INLET
        case ("from_file")
          bc_type = BC_INLET_FROM_FILE
        case ("inlet_no_backflow")
          bc_type = BC_INLET_NO_BACKFLOW
        case ("outlet_no_backflow")
          bc_type = BC_OUTLET_NO_BACKFLOW
        case default
          write(*,*) "Error. Unknown boundary side in " // fname
          call error_stop()
      end select

      Btype(selected_sides) = bc_type
      
      if (bc_type == BC_DIRICHLET .or. bc_type == BC_INLET_NO_BACKFLOW) then
        do iside = 1, size(selected_sides)
          sideU(:,selected_sides(iside)) = u_vec
        end do
        
        if (norm2(u_vec)>0) then
          Uinlet_vec = u_vec
          Uinlet = norm2(Uinlet_vec)
        end if
      end if
      
      if (z0>0) then
        !TODO: refactor the z0 information to an array accross sides
        do iside = 1, size(selected_sides)
          select case (selected_sides(iside))
            case(1)
              z0W = z0
            case(2)
              z0E = z0
            case(3)
              z0S = z0
            case(4)
              z0N = z0
            case(5)
              z0B = z0
            case(6)
              z0T = z0
          end select
        end do
      end if
        
      if (bc_type == BC_TURBULENT_INLET) then
        if (size(selected_sides)>1) then
          write(*,*) "Error, multiple sides selected for the turbulent inlet boundary condition in " // fname
          call error_stop()
        end if
        if (associated(stg_properties_obj%ptr)) then
          call get_stg_properties(stg_properties_obj%ptr, selected_sides(1))
        end if
      end if
      
    end subroutine get_bc
    
    
    subroutine get_ic(obj)
      type(tree_object), intent(in) :: obj

      type(tree_object_ptr), target :: stg_properties_obj
      character(char_len), target :: inlet_type_str
      real(knd), target :: inlet_parameter
      
      type(field_names) :: names(2)
      type(field_names_a) :: names_a(1)

      integer :: is, side
      
      names = [field_names_init("inlet_type", inlet_type_str), &
               field_names_init("stg", stg_properties_obj)]

      names_a = [field_names_a_init("Uinlet", Uinlet_conf)]
      
      
      inlet_type_str = ""
      inlet_parameter = 0
      
      call get_object_field_values(obj, stat, names, names_a)
        
      if (stat/=0) then
        write(*,*) "Error parsing an ic() object in " // fname
        write(*,*) "Error status", stat
        call error_stop()
      end if

      if (associated(stg_properties_obj%ptr)) then
        side = 0
        !is there any side with the BC_TURBULENT_INLET condition?
        do is = We, No
          if (Btype(is)==BC_TURBULENT_INLET) then
            side = is
            exit
          end if
        end do

        !if not, get the direction of Uinlet if known
        if (side<=0) then
          if (norm2(Uinlet_conf)>0) then
            call u_vec_to_side(Uinlet_conf, side)

            if (Uinlet<0) then
              Uinlet_vec = Uinlet_conf
              Uinlet = norm2(Uinlet_vec)
            end if
          end if
        end if

        ! now side could still be 0
      
        call get_stg_properties(stg_properties_obj%ptr, side)
        inlettype = TurbulentInletType
        
      else if (len_trim(inlet_type_str)>0) then
      
        !TODO: Currently just setting the legacy configuration parameters. To be completely rewritten.
        select case (downcase(inlet_type_str))
          case ("zero")
            inlettype = ZeroInletType
          case ("shear")
            inlettype = ShearInletType
          case ("parabolic")
            inlettype = ParabolicInletType
          case ("turbulent")
            inlettype = TurbulentInletType
          case ("from_file")
            inlettype = FromFileInletType
          case ("geostrophic")
            inlettype = GeostrophicInletType
          case ("constant")
            inlettype = ConstantInletType
          case default
            write(*,*) "Error, unknown inlet_type '"// trim(inlet_type_str)//"' at "  // fname
            call error_stop()
        end select

        if (Uinlet<0) then
          Uinlet_vec = Uinlet_conf
          Uinlet = norm2(Uinlet_vec)
        end if

      else
      
        if (Uinlet<0) then
          Uinlet_vec = Uinlet_conf
          Uinlet = norm2(Uinlet_vec)
        end if
        inlettype = ConstantInletType

      end if
    end subroutine get_ic


    subroutine u_vec_to_side(u_vec, side)
      real(knd), intent(in) :: u_vec(3)
      integer, intent(out) :: side
      real(knd) :: theta
      theta = atan2(u_vec(2), u_vec(1))
      if (theta <= -3*pi/4 .or. theta >=3*pi/4) then
        side = Ea
      else if (theta < -pi/4) then
        side = No
      else if (theta > pi/4) then
        side = So
      else
        side = We
      end if
    end subroutine
    
    
      
    subroutine get_vec_profile(obj)
      type(tree_object), intent(in) :: obj

    end subroutine
    
    subroutine get_stg_properties(obj, side)
      use Turbinlet
      use, intrinsic :: ieee_arithmetic
      type(tree_object), intent(in) :: obj
      integer, intent(inout) :: side
      
      
      integer, target :: stg_method
      character(char_len), target :: stg_method_str
      
      character(char_len), target :: profile_type_str
      
      real(knd), target :: u_ref, z_ref, u_star, z0, power_exponent, stress_grad
      real(knd), target :: u_vec(3)
      
      real(knd), target :: T_Lag, L_y, L_z
      
      real(knd), target :: uu, vv, ww, uv, uw, vw
      real(knd), target :: stress(6)
      character(char_len), target :: stress_type_str
      
      integer, parameter :: STRESS_RELATIVE = 1, STRESS_ABSOLUTE = 2
      
      integer, target :: inlet_plane_i
      real(knd), target :: inlet_plane_x
      
      character(char_len), target :: mean_profile, turbulence_profile
      
      integer :: profile_type
      integer :: stress_type
      
      type(field_names), allocatable :: names(:)
      type(field_names_a), allocatable :: names_a(:)
      type(field_names_str), allocatable :: names_str(:)

      integer :: stat
      
      names = [ field_names_init("method", stg_method_str), &
                field_names_init("profile", profile_type_str), &
                field_names_init("u_ref", u_ref), &
                field_names_init("z_ref", z_ref), &
                field_names_init("u_star", u_star), &
                field_names_init("z0", z0), &
                field_names_init("exponent", power_exponent), &
                field_names_init("T_Lag", T_Lag), &
                field_names_init("L_y", L_y), &
                field_names_init("L_z", L_z), &
                field_names_init("uu", uu), &
                field_names_init("vv", vv), &
                field_names_init("ww", ww), &
                field_names_init("uv", uv), &
                field_names_init("uw", uw), &
                field_names_init("vw", vw), &
                field_names_init("stress_type", stress_type), &
                field_names_init("stress_gradeint", stress_grad), &
                field_names_init("inlet_plane_i", inlet_plane_i), &
                field_names_init("inlet_plane_x", inlet_plane_x) ]
              
      names_a = [ field_names_a_init("stress", stress), &
                  field_names_a_init("Uinlet", u_vec)]

      names_str = [ field_names_str("mean_profile", mean_profile), &
                    field_names_str("turbulence_profile", turbulence_profile)]

      
      
      stg_method = turbulent_inlet_XC
      stg_method_str = ""
      
      profile_type = 0
      profile_type_str = ""
      
      T_Lag = -1
      L_y = -1
      L_z = -1
      
      u_ref = -1
      z_ref = -1
      u_star = -1
      z0 = -1
      power_exponent = -1

      u_vec = 0
      
      uu = -1
      vv = -1
      uu = -1
      uv = 0
      uw = 0
      vw = 0
      stress = -1
      
      stress_type_str = ""
      stress_type = 0
      stress_grad = 0
      
      inlet_plane_i = huge(1)
      inlet_plane_x = huge(1._knd)

      mean_profile = ""
      turbulence_profile = ""
        
      call get_object_field_values(obj, stat, names, names_a, fields_str = names_str)
      
      if (stat/=0) then
        write(*,*) "Error parsing an stg() object in " // fname
        write(*,*) "Error status", stat
        call error_stop()
      end if

      if (len_trim(stg_method_str)>0) then
        select case (downcase(stg_method_str))
          case ("xc","xiecastro","default")
            stg_method = turbulent_inlet_XC
          case ("xcdf","xc_div_free","kcx")
            stg_method = turbulent_inlet_XCDF
          case default
            write(*,*) "Error, unknown stg method '" // trim(stg_method_str) // "' in " // fname
            call error_stop()
        end select
        
        turbulent_inlet_method = stg_method
      end if
      
      select case (profile_type_str)
        case ("log","logarithmic")
          profiletype = PROFILE_LOGARITHMIC
        case ("pow", "power", "power law", "power_law")
          profiletype = PROFILE_POWER_LAW
        case ("file", "from file", "from_file")
          profiletype = PROFILE_FROM_FILE
        case default
          profiletype = PROFILE_CONSTANT
      end select


      if (norm2(u_vec)>0) then
        if (Uinlet>0) then
          write(*,*) "Error, multiple definitions of Uinlet in " // fname
          call error_stop
        end if

        Uinlet_vec = u_vec
        Uinlet = norm2(u_vec)

        if (Uinlet==0) then
          write(*,*) "Error, Uinlet vector must be nonzero in " // fname
          call error_stop
        end if

        if (side<=0) then
          call u_vec_to_side(u_vec, side)
        end if
      else if (Uinlet < 0 .and. profiletype == PROFILE_CONSTANT) then
        write(*,*) "Error, Uinlet must be defined for the synthetic turbulence generator in" // fname
        write(*,*) "when the profile type is constant."
        call error_stop
      end if

        
      associate(g=>default_turbulence_generator)
        if (stg_method==turbulent_inlet_XCDF) then
          if (inlet_plane_x < huge(1._knd)) &
            g%inlet_plane_x = inlet_plane_x
          if (inlet_plane_i < huge(1)) &
            g%inlet_plane_i = inlet_plane_i

          if (.not. allocated(g%inlet_plane_x) .and. .not. allocated(g%inlet_plane_i)) then
            write(*,*) "Error, when method=XCDF in stg, then either inlet_plane_x or &
                       &inlet_plane_i must be provided in " // fname
            call error_stop
          end if
        end if
        
        g%Ustar_surf_inlet = u_star
        !TODO more options
        g%stress_gradient_inlet = stress_grad
        
        g%U_ref_inlet = u_ref
        g%z_ref_inlet = z_ref
        g%z0_inlet = z0
        g%power_exponent_inlet = power_exponent
        
        if (T_Lag>0) then
          g%T_Lag = T_Lag
        else
          write(*,*) "The Lagrangian time scale must be provided for the turbulence generator&
          & in the T_Lag variable in " // fname
          call error_stop()
        end if
        if (L_y>0 .and. L_z>0) then
          g%L_y = L_y
          g%L_z = L_z
        else
          write(*,*) "The integral length scales in the spanwise and vertical direction&
          & must be provided for the turbulence generator in the L_y and L_z variables in " // fname
          call error_stop()
        end if

        if (uu>0.and.vv>0.and.ww>0) then
          g%relative_stress(1,1) = uu
          g%relative_stress(2,2) = vv
          g%relative_stress(3,3) = ww
          g%relative_stress(1,2) = uv
          g%relative_stress(2,1) = uv
          g%relative_stress(1,3) = uw
          g%relative_stress(3,1) = uw
          g%relative_stress(2,3) = vw
          g%relative_stress(3,2) = vw
        else if (all(stress(1:3)>0)) then
          g%relative_stress(1,1) = stress(1)
          g%relative_stress(2,2) = stress(2)
          g%relative_stress(3,3) = stress(3)
          g%relative_stress(1,2) = stress(4)
          g%relative_stress(2,1) = stress(4)
          g%relative_stress(1,3) = stress(5)
          g%relative_stress(3,1) = stress(5)
          g%relative_stress(2,3) = stress(6)
          g%relative_stress(3,2) = stress(6)
        else if (.not. profiletype==PROFILE_FROM_FILE) then
          write(*,*) "Relative stress with positive diagonal terms must be provided for the turbulence generator in " // fname
          write(*,*) "Set either the uu, vv and ww variables or the stress array, ordered [uu, vvv, ww, uv, uw, vw]."
          call error_stop()          
        end if
        
        
        select case (side)
          case (We, Ea)
            g%direction = 1
          case (So, No)
            g%direction = 2
          case (0)
            !no information on the direction has been given
            !assuming +x
            g%direction = 1
          case default
            write(*,*) "Error, synthetic turbulence generation supported only on boundaries in the horizontal directions (W,E,S,N)."
            call error_stop()
        end select

        if (profiletype==PROFILE_FROM_FILE) then
          if (len_trim(mean_profile)>0) then
            g%mean_profile_file = trim(mean_profile)
          else
            g%mean_profile_file = "inlet_mean_profile.txt"
          end if
          if (len_trim(turbulence_profile)>0) then
            g%turbulence_profile_file = trim(turbulence_profile)
          else
            g%turbulence_profile_file = "inlet_turbulence_profile.txt"
          end if
        end if

      end associate
      
    end subroutine
    
  end subroutine get_boundary_conditions_velocity
  
  


  subroutine get_obstacles(fname)
    use Strings
    use ParseTrees
    use GeometricShapes
    use SolidBodies
    use ArrayUtilities, only: cross_product
    
    character(*), intent(in) :: fname

    type(tree_object), allocatable :: tree(:)

    type(field_names) :: names_bbox(6)
    type(field_names) :: names_base(4)
    type(field_names_a) :: names_plane_a(5)
    
    type(field_names_str) :: names_file_str(1)
    
    type obstacle_cfg
      real(knd) :: z0 = -1, z0H = -1
      real(knd) :: moisture_flux = 0, temperature_flux = 0
    end type
    
    type :: file_cfg
      character(char_len) :: file = ""
    end type
    
    type :: plane_cfg
      real(knd) :: point1(3) = 0, point2(3) = 0, point3(3) = 0
      real(knd) :: point(3) = 0, vec(3) = 0
    end type
    
    type :: bbox_cfg
      real(knd) :: xmin = -huge(1._knd)/2, xmax = huge(1._knd)
      real(knd) :: ymin = -huge(1._knd)/2, ymax = huge(1._knd)
      real(knd) :: zmin = -huge(1._knd)/2, zmax = huge(1._knd)
    end type
    
    type(bbox_cfg) :: o_bbox
    type(obstacle_cfg) :: base
    type(file_cfg), target :: o_file
    type(plane_cfg), target :: o_plane
    
    logical :: ex
    integer :: iobj, stat
    
    real(knd) :: vec(3)
    
    class(GeometricShape), allocatable :: gs
    
    names_bbox = [field_names_init("xmin", o_bbox%xmin), &
                  field_names_init("xmax", o_bbox%xmax), &
                  field_names_init("ymin", o_bbox%ymin), &
                  field_names_init("ymax", o_bbox%ymax), &
                  field_names_init("zmin", o_bbox%zmin), &
                  field_names_init("zmax", o_bbox%zmax)]

    names_base = [field_names_init("z0",  base%z0), &
                  field_names_init("z0H", base%z0H), &
                  field_names_init("temperature_flux", base%temperature_flux), &
                  field_names_init("moisture_flux",    base%moisture_flux)]

    names_file_str = [field_names_str("file", o_file%file)]

    names_plane_a = [field_names_a_init("point1", o_plane%point1), &
                     field_names_a_init("point2", o_plane%point2), &
                     field_names_a_init("point3", o_plane%point3), &
                     field_names_a_init("point",  o_plane%point), &
                     field_names_a_init("vec",    o_plane%vec)]

                 
    inquire(file=fname, exist=ex)

    if (.not.ex) return

    call parse_file(tree, fname, stat)

    if (stat/=0) then
      write(*,*) "Error parsing file " // fname
      call error_stop
    end if
  
    if (.not.allocated(tree)) then
      write(*,*) "Error, no content in " // fname
      call error_stop
    end if

    do iobj = 1, size(tree)
    
      base = obstacle_cfg()
    
      if (downcase(tree(iobj)%name)=="obstacles_bbox") then
      
         call get_object_field_values(tree(iobj), stat, &
                                     fields = names_bbox)
                                     
        if (stat/=0) then
          call error_stop("Error interpretting obstacle_file fields in " // trim(fname))
        end if
        
        call init_bbox
        
     else if (downcase(tree(iobj)%name)=="obstacle_file") then
      
        o_file = file_cfg()
        
        call get_object_field_values(tree(iobj), stat, &
                                     fields_str = names_file_str)
                                     
        if (stat/=0) then
          call error_stop("Error interpretting obstacle_file fields in " // trim(fname))
        end if
        
        inquire(file=o_file%file, exist=ex)
        if (.not.ex) then
          call error_stop("Error, file " // trim(o_file%file) // " does not exist when interpretting obstacle_file fields in " &
          // trim(fname))
        end if
        
        allocate(Polyhedron :: gs)
        select type (gs)
          type is (Polyhedron)
            gs = Polyhedron(o_file%file)
        end select        
        
      else if (downcase(tree(iobj)%name)=="plane") then
      
        o_plane = plane_cfg()
        
        call get_object_field_values(tree(iobj), stat, &
                                     fields_a = names_plane_a)
                                     
        if (stat/=0) then
          call error_stop("Error interpretting plane fields in " // trim(fname))
        end if
        
        
        if (any(o_plane%point1/=0) .or. &
            any(o_plane%point2/=0)   .or. &
            any(o_plane%point3/=0)) then
            
          vec = cross_product(o_plane%point3-o_plane%point2, o_plane%point1-o_plane%point2)
          
          if (norm2(vec)<epsilon(vec)) then
            write(*,*) "Error, plane points:"
            write(*,*) "point1:",o_plane%point1
            write(*,*) "point2:",o_plane%point2
            write(*,*) "point3:",o_plane%point3
            write(*,*) "defined in file '",trim(fname),"',"
            write(*,*) "lie on a single line." 
            call error_stop
          end if
            
          allocate(gs, source = Plane(point1 = o_plane%point1, &
                                      point2 = o_plane%point2, &
                                      point3 = o_plane%point3))
        else if (any(o_plane%vec>0)) then
          allocate(gs, source = Plane(point = o_plane%point, &
                                      vec   = o_plane%vec))
        else
          call error_stop("Error, unable to interpret plane orientation in file '"//trim(fname)//"'.")
        end if
        
      else 
        call error_stop("Error, unknown object type '"//trim(tree(iobj)%name)//"' in file '"//trim(fname)//"'.")
      end if
      
      if (allocated(gs)) then
        if (base%z0 >= 0 .and. base%z0H >= 0) then
          call AddSolidBody(SolidBody(gs, &
                                      z0 = base%z0, z0H = base%z0H, &
                                      temperature_flux = base%temperature_flux, &
                                      moisture_flux = base%moisture_flux))
        else if (base%z0 >= 0) then
          call AddSolidBody(SolidBody(gs, &
                                      z0 = base%z0, &
                                      temperature_flux = base%temperature_flux, &
                                      moisture_flux = base%moisture_flux))
        else
          call AddSolidBody(SolidBody(gs, &
                                      temperature_flux = base%temperature_flux, &
                                      moisture_flux = base%moisture_flux))
        end if
      
        deallocate(gs)
      end if
      
      call tree(iobj)%finalize
    end do
    
  contains
  
    subroutine init_bbox
      if (o_bbox%xmin > -huge(1._knd)/2) obstacles_bbox(We) = o_bbox%xmin
      if (o_bbox%xmax <  huge(1._knd)/2) obstacles_bbox(Ea) = o_bbox%xmax
      if (o_bbox%ymin > -huge(1._knd)/2) obstacles_bbox(So) = o_bbox%ymin
      if (o_bbox%ymax <  huge(1._knd)/2) obstacles_bbox(No) = o_bbox%ymax
      if (o_bbox%zmin > -huge(1._knd)/2) obstacles_bbox(Bo) = o_bbox%zmin
      if (o_bbox%zmax <  huge(1._knd)/2) obstacles_bbox(To) = o_bbox%zmax
    end subroutine
    
  end subroutine get_obstacles
  

    

  subroutine get_area_sources(fname)
    use Strings
    use ParseTrees
    use GeometricShapes2D
    use AreaSources
    character(*), intent(in) :: fname
    type(tree_object), allocatable :: tree(:)
    logical :: ex
    integer :: iobj, stat

    if (num_of_scalars==0) return

    inquire(file=fname, exist=ex)

    if (.not.ex) return

    call parse_file(tree, fname, stat)

    if (stat==0) then

      if (allocated(tree)) then
        do iobj = 1, size(tree)
          call get_area_source(tree(iobj))
          call tree(iobj)%finalize
        end do
      end if

    else

      write(*,*) "Error parsing file " // fname
      call error_stop

    end if

  contains

    subroutine get_area_source(obj)
      type(tree_object), intent(in) :: obj
      class(GeometricShape2D), allocatable :: shp
      type(ScalarAreaSource) :: src
      real(knd) :: flux
      integer :: scnum
      integer :: j

      if (downcase(obj%name)=='area_source') then

        if (allocated(obj%fields%array)) then

          associate(fields => obj%fields%array)

            do j = 1, size(fields)

              if (downcase(fields(j)%name)=='scalar_number') then
                read(fields(j)%value, *) scnum
              else if (downcase(fields(j)%name)=='flux') then
                read(fields(j)%value, *) flux
              else if (downcase(fields(j)%name)=='geometric_shape') then
                if (fields(j)%is_object .and. associated(fields(j)%object_value)) then
                  call get_geometric_shape(shp, fields(j)%object_value)
                else 
                  write(*,*) "Invalid geometric shape for area source in " // fname
                  call error_stop
                end if
              end if

            end do

          end associate

        else

          write(*,*) "No fields in the AreaSource object in " // fname
          call error_stop

        end if

        src = ScalarAreaSource(shp, scnum, flux)
        call add_element(ScalarAreaSources, src)

      else
        write(*,*) "Unknown object type " // downcase(obj%name) // " in " // fname
        call error_stop
      end if
    end subroutine

    subroutine get_geometric_shape(res, obj)
      class(GeometricShape2D), allocatable :: res
      type(tree_object), intent(in) :: obj
      real(knd) :: xc, yc, r
      integer :: j

      if (downcase(obj%name)=='circle') then

        if (allocated(obj%fields%array)) then

          associate(fields => obj%fields%array)

            do j = 1, size(fields)

              if (downcase(fields(j)%name)=='xc') then
                read(fields(j)%value, *) xc
              else if (downcase(fields(j)%name)=='yc') then
                read(fields(j)%value, *) yc
              else if (downcase(fields(j)%name)=='r') then
                read(fields(j)%value, *) r
              end if

            end do

          end associate

        else

          write(*,*) "No fields in the Circle object in " // fname
          call error_stop

        end if

        allocate(res, source = Circle(xc, yc, r))

      else

        write(*,*) "Invalid geometric shape for area source in " // fname
        write(*,*) "Supported variants: Circle"
        write(*,*) "Received:", obj%name

        call error_stop

      end if

    end subroutine

    subroutine add_element(a,e)
      type(ScalarAreaSource), allocatable, intent(inout) :: a(:)
      type(ScalarAreaSource), intent(inout) :: e
      type(ScalarAreaSource), allocatable :: tmp(:)
      integer :: i, n

      if (.not.allocated(a)) then
        a = [e]
      else
        n = size(a)
        call move_alloc(a,tmp)
        allocate(a(n+1))

        do i = 1, n
          call assign(a(i), tmp(i))
        end do
        call assign(a(n+1), e)
      end if
    end subroutine
    
    subroutine assign(l, r)
      type(ScalarAreaSource), intent(out) :: l
      type(ScalarAreaSource), intent(inout) :: r
      l%flux = r%flux
      l%scalar_number = r%scalar_number
      call move_alloc(r%GeometricShape, l%GeometricShape)
    end subroutine

  end subroutine get_area_sources

  
  subroutine get_buoyant_scalars(fname)
    use BuoyantGases
    use Strings
    use ParseTrees
    character(*), intent(in) :: fname
    type(tree_object), allocatable :: tree(:)
    logical :: ex
    integer :: iobj, stat
    real(knd), target :: molar_mass

    type(field_names) :: names(1)

    names = [field_names_init("m_g_mol", molar_mass)]

    inquire(file=fname, exist=ex)

    if (.not.ex) return

    call parse_file(tree, fname, stat)

    if (stat/=0) then
      write(*,*) "Error parsing file " // fname
      call error_stop
    end if
  
    if (.not.allocated(tree)) return

    do iobj = 1, size(tree)
         
      if (downcase(tree(iobj)%name)=="buoyant_scalar") then
         !m_mol_air is in kg/mol
         molar_mass = m_mol_air * 1000
         
         call get_object_field_values(tree(iobj), stat, &
                                     fields = names)
                                     
        if (stat/=0) then
          call error_stop("Error interpretting buoyant_scalar fields in " // trim(fname))
        end if
        
        call add
               
      else 
        call error_stop("Error, unknown object type '"//trim(tree(iobj)%name)//"' in file '"//trim(fname)//"'.")
      end if

      call tree(iobj)%finalize
      
    end do


  contains

    subroutine add
      if (.not.allocated(buoyant_scalars)) allocate(buoyant_scalars(0))
      
      enable_buoyant_scalars = .true.
        
      !g/mol -> kg/mol
      buoyant_scalars = [buoyant_scalars, buoyant_scalar(molar_mass / 1000)]
    end subroutine

  end subroutine get_buoyant_scalars


  subroutine get_geostrophic_wind(fname, g)
    use Interpolation
    character(*), intent(in) :: fname
    type(spline_coefs), intent(out) :: g
    character(256) :: line
    real(knd) :: r3(3)
    real(knd), allocatable :: ug(:), vg(:)
    integer :: unit, io, n, i, j

    open(newunit=unit,file=fname,status="old",action="read",iostat = io)
    if (io/=0) return

    n = 0
    do
      read(unit,'(a)',iostat=io) line
      if (io/=0) exit
      read(line,*,iostat=io) r3
      if (io/=0) exit
      n = n + 1
    end do
    
    rewind(unit)
    
    if (n>0) then
      allocate(g%z(n), ug(n), vg(n))
      allocate(g%cu(0:1,n), g%cv(0:1,n))
      do i = 1, n
        read(unit,'(a)',iostat=io) line
        read(line, *) g%z(i), ug(i), vg(i)
      end do
    else
      stop "Geostrophic profile empty."
    end if

    if (n > 1) then
      call linear_interpolation(g%z, ug, g%cu)
      call linear_interpolation(g%z, vg, g%cv)

      allocate(pr_gradient_profile_x(1:Prnz))
      allocate(pr_gradient_profile_y(1:Prnz))

      j = 1
      do i = 1, Prnz
        pr_gradient_profile_x(i) =   Coriolis_parameter * linear_interpolation_eval(zPr(i), g%z, g%cv, j)
      end do

      j = 1
      do i = 1, Prnz
        pr_gradient_profile_y(i) = - Coriolis_parameter * linear_interpolation_eval(zPr(i), g%z, g%cu, j)
      end do

      enable_pr_gradient_x_profile = any(pr_gradient_profile_x/=0)
      enable_pr_gradient_y_profile = any(pr_gradient_profile_y/=0)

      if (enable_pr_gradient_x_profile) enable_pr_gradient_x_uniform = .false.
      if (enable_pr_gradient_y_profile) enable_pr_gradient_y_uniform = .false.

    else

      pr_gradient_x =   Coriolis_parameter * vg(1)
      pr_gradient_y = - Coriolis_parameter * ug(1)

      enable_pr_gradient_x_uniform = pr_gradient_x /= 0
      enable_pr_gradient_y_uniform = pr_gradient_y /= 0

    end if
    

  end subroutine get_geostrophic_wind


  subroutine get_pressure_gradient(fname)
    use Interpolation
    character(*), intent(in) :: fname
    type(spline_coefs) :: g
    character(256) :: line
    real(knd) :: r3(3)
    real(knd), allocatable :: dpdx(:), dpdy(:)
    integer :: unit, io, n, i, j

    open(newunit=unit,file=fname,status="old",action="read",iostat = io)
    if (io/=0) return

    n = 0
    do
      read(unit,'(a)',iostat=io) line
      if (io/=0) exit
      read(line,*,iostat=io) r3
      if (io/=0) exit
      n = n + 1
    end do
    
    rewind(unit)
    
    if (n>0) then
      allocate(g%z(n), dpdx(n), dpdy(n))
      allocate(g%cu(0:1,n), g%cv(0:1,n))
      do i = 1, n
        read(unit,'(a)',iostat=io) line
        read(line, *) g%z(i), dpdx(i), dpdy(i)
      end do
    else
      stop "Pressure gradient profile empty."
    end if

    if (n > 1) then
      call linear_interpolation(g%z, dpdx, g%cu)
      call linear_interpolation(g%z, dpdy, g%cv)

      allocate(pr_gradient_profile_x(1:Prnz))
      allocate(pr_gradient_profile_y(1:Prnz))

      j = 1
      do i = 1, Prnz
        pr_gradient_profile_x(i) = linear_interpolation_eval(zPr(i), g%z, g%cu, j)
      end do

      j = 1
      do i = 1, Prnz
        pr_gradient_profile_y(i) = linear_interpolation_eval(zPr(i), g%z, g%cv, j)
      end do

      enable_pr_gradient_x_profile = any(pr_gradient_profile_x/=0)
      enable_pr_gradient_y_profile = any(pr_gradient_profile_y/=0)

    else

      pr_gradient_x = dpdx(1)
      pr_gradient_y = dpdy(1)

      enable_pr_gradient_x_uniform = pr_gradient_x /= 0
      enable_pr_gradient_y_uniform = pr_gradient_y /= 0

    end if
    

  end subroutine get_pressure_gradient


  subroutine get_forcing(fname)
    use Strings
    use ParseTrees
    use LinearForcing
    character(*), intent(in) :: fname
    real(knd), target :: constant = 0, tke0 = 0, epsilon0 = 0, Umean(3) = 0
    logical, target :: enable = .false., &
                       enable_tke0_epsilon0 = .false., &
                       variable_means = .false., &
                       filter = .false.
    type(tree_object), allocatable :: tree(:)
    logical :: ex
    integer :: i, stat
    
    type(field_names) :: fields(5)
    type(field_names_a) :: fields_a(1)
    
    fields = [field_names_init("variable_means",       variable_means), &
              field_names_init("filter",               filter), &
              field_names_init("forcing_constant",     constant), &
              field_names_init("tke",                  tke0), &
              field_names_init("epsilon",              epsilon0)]

    fields_a = [field_names_a_init("Umean",     Umean)]
    inquire(file=fname, exist=ex)

    if (.not.ex) return

    call parse_file(tree, fname, stat)

    if (stat==0) then

      if (allocated(tree)) then

        call find_object_get_field_values(tree, "linear_forcing", stat, &
                                     fields, fields_a)

        if (stat<=0) call init
        
        do i = 1, size(tree)
          call tree(i)%finalize
        end do
      end if

    else

      write(*,*) "Error parsing file " // fname
      call error_stop

    end if
    
  contains
  
    subroutine init
      enable = .true.
      
      if (tke0>0 .and. epsilon0>0) then
        enable_tke0_epsilon0 = .true.
      else if (tke0>0 .or. epsilon0>0) then
        call error_stop("Error in forcing.conf. Linear forcing requires both or &
                         & none of tke and epsilon to be specified and nonzero.")
      end if
      
      call init_linear_forcing(enable, enable_tke0_epsilon0, variable_means, filter, &
                               constant, tke0, epsilon0, Umean)
    end subroutine
    
  end subroutine get_forcing


  subroutine get_profiles(fname, profiles_config, enable_profiles)
    use Strings
    use ParseTrees
    character(*), intent(in) :: fname
    type(ProfileSwitches), intent(inout), target :: profiles_config
    character(char_len), target :: average_end_str, instant_end_str, running_end_str
    logical, intent(inout), target :: enable_profiles
    type(tree_object), allocatable :: tree(:)
    logical :: ex
    integer :: i, stat

    type(field_names) :: fields_avg(2), fields_inst(3), fields_running(3)


    fields_avg = [field_names_init("start",     profiles_config%average_start), &
                  field_names_init("end",       average_end_str)]

    fields_inst = [field_names_init("start",     profiles_config%instant_start), &
                   field_names_init("end",       instant_end_str), &
                   field_names_init("interval",  profiles_config%instant_interval)]

    fields_running = [field_names_init("start",     profiles_config%running_start), &
                      field_names_init("end",       running_end_str), &
                      field_names_init("interval",  profiles_config%running_interval)]

    inquire(file=fname, exist=ex)

    if (.not.ex) return

    call parse_file(tree, fname, stat)

    if (stat==0) then

      if (allocated(tree)) then

        call find_object_get_field_values(tree, "average_profiles", stat, &
                                     fields_avg)

        if (stat<=0) then
          enable_profiles = .true.
          
          if (downcase(average_end_str)=="end") then
            profiles_config%average_end = time_stepping%end_time
          else
            call real_value(average_end_str, profiles_config%average_end)
          end if
          
        end if

        call find_object_get_field_values(tree, "instantaneous_profiles", stat, &
                                     fields_inst)

        if (stat<=0) then
          enable_profiles = .true.
          
          if (downcase(instant_end_str)=="end") then
            profiles_config%instant_end = time_stepping%end_time
          else
            call real_value(instant_end_str, profiles_config%instant_end)
          end if
          
        end if

        call find_object_get_field_values(tree, "running_average_profiles", stat, &
                                     fields_running)

        if (stat<=0) then
          enable_profiles = .true.
          
          if (downcase(running_end_str)=="end") then
            profiles_config%running_end = time_stepping%end_time
          else
            call real_value(running_end_str, profiles_config%running_end)
          end if
          
        end if

        do i = 1, size(tree)
          call tree(i)%finalize
        end do
      end if

    else

      write(*,*) "Error parsing file " // fname
      call error_stop

    end if
    
  contains
    
    subroutine real_value(str, x)
      character(*), intent(in) :: str
      real(knd), intent(out) :: x
      integer :: ie
      read(str, *, iostat=ie) x
      if (ie/=0) call error_stop("Error interpretting '"// &
                                trim(str)// &
                                "' as a real value in '"// &
                                trim(fname)//"'.")
    end subroutine
  end subroutine get_profiles


  subroutine get_time_stepping(fname, t_s)
    use Strings
    use ParseTrees
    character(*), intent(in) :: fname
    type(time_step_control), intent(inout), target :: t_s
    type(tree_object), allocatable :: tree(:)
    logical :: ex
    integer :: iobj, stat

    logical, target :: constant_time_steps = .false.
    
    character(char_len), target :: clock_time_limit_str = ""

    type(field_names) :: names(13)
    type(field_names_a) :: names_a(3)
    type(field_names_str) :: names_str(1)

    names = [field_names_init("max_number_of_time_steps",   t_s%max_number_of_time_steps), &
             field_names_init("check_period",               t_s%check_period), &
             field_names_init("variable_time_steps",        t_s%variable_time_steps), &
             field_names_init("constant_time_steps",        constant_time_steps), &
             field_names_init("enable_U_scaling",           t_s%enable_U_scaling), &
             field_names_init("enable_CFL_check",           t_s%enable_CFL_check), &
             field_names_init("dt",                         t_s%dt_constant), &
             field_names_init("dt_max",                     t_s%dt_max), &
             field_names_init("dt_min",                     t_s%dt_min), &
             field_names_init("CFL",                        t_s%CFL), &
             field_names_init("CFL_max",                    t_s%CFL_max), &
             field_names_init("start_time",                 t_s%start_time), &
             field_names_init("end_time",                   t_s%end_time)]

    names_a = [field_names_a_init("U_scaling", t_s%U_scaling), &
               field_names_a_init("U_max",     t_s%U_max), &
               field_names_a_init("U_min",     t_s%U_min)]
               
    names_str = [field_names_str("clock_time_limit", clock_time_limit_str)]

    inquire(file=fname, exist=ex)

    if (.not.ex) return

    call parse_file(tree, fname, stat)

    if (stat==0) then

      if (allocated(tree)) then
        call find_object_get_field_values(tree, "time_stepping", stat, &
                                     fields = names, fields_a = names_a, fields_str = names_str)

        if (stat<=0) then
          call init
        else if (stat==1) then
          write(*,*) "Error, no time_stepping object in " // fname
          call error_stop
        else
          write(*,*) "Error, parsing time_stepping() in " // fname
          write(*,*) "Status:", stat
          call error_stop
        endif

        do iobj = 1, size(tree)
          call tree(iobj)%finalize
        end do
      else
        write(*,*) "Error, no content in " // fname
        call error_stop
      end if

    else

      write(*,*) "Error parsing file " // fname
      call error_stop

    end if

  contains

    subroutine init
      integer :: l, mult, stat

      if (t_s%dt_constant > 0) constant_time_steps = .true.

      if (constant_time_steps) t_s%variable_time_steps = .false.

      if (t_s%variable_time_steps) then

        t_s%U_max = abs(t_s%U_max)
        t_s%U_min = abs(t_s%U_min)

        if (maxval(t_s%U_max) <=  0) &
          call error_stop("Error, time_stepping%dt_min must be set or &
            &time_stepping%U_max must have at least one non-zero component.")

        if (t_s%dt_max <= 0 .and. maxval(t_s%U_min) <=  0) &
          call error_stop("Error, time_stepping%dt_max must be set or &
            &time_stepping%U_min must have at least one non-zero component.")

        if (t_s%CFL <= 0) &
          call error_stop("Error, time_stepping%CFL must be positive.")


        if (t_s%dt_min <= 0) &
          t_s%dt_min = t_s%CFL / (t_s%U_max(1) / dxmin + &
                                  t_s%U_max(2) / dymin + &
                                  t_s%U_max(3) / dzmin)
        
        if (t_s%dt_max <= 0) &
          t_s%dt_max = t_s%CFL / (t_s%U_min(1) / dxmin + &
                                  t_s%U_min(2) / dymin + &
                                  t_s%U_min(3) / dzmin)

        t_s%dt = t_s%dt_min

      else

        t_s%U_scaling = abs(t_s%U_scaling)

        !explicitly specified time-step length (dt=) has a priority
        if (t_s%dt_constant <=0 .and. maxval(t_s%U_scaling) > 0) then

          t_s%enable_U_scaling = .true.

          if (t_s%CFL<=0) &
            call error_stop("Error, time_stepping%CFL must be positive when using U_scaling.")

          t_s%dt_constant = t_s%CFL / (t_s%U_scaling(1) / dxmin + &
                                       t_s%U_scaling(2) / dymin + &
                                       t_s%U_scaling(3) / dzmin)
        else
          if (t_s%dt_constant <=  0) &
            call error_stop("Error, time_stepping%dt or t_s%U_scaling must be positive.")
        end if

        t_s%enable_CFL_check = (t_s%CFL_max > 0)

        t_s%dt = t_s%dt_constant

      end if
      
      l = len_trim(clock_time_limit_str)
      if (l > 0) then
        clock_time_limit_str = downcase(clock_time_limit_str)
        
        select case (clock_time_limit_str(l:l))
          case ('s')
            mult = 1
          case ('m')
            mult = 60
          case ('h')
            mult = 3600
          case ('d')
            mult = 86400
          case ('w')
            mult = 604800
          case default
            call error_stop("Error, clock_time_limit requires units: s, m, h, d or w.")
        end select
    
        read(clock_time_limit_str(1:l-1),*, iostat=stat) time_stepping%clock_time_limit
        
        if (stat/=0) then
          write(*,*) "Error interpretting the clock_time_limit value. Received: '",clock_time_limit_str(1:l-1),"'."
          call error_stop()
        end if

        time_stepping%clock_time_limit = time_stepping%clock_time_limit * mult
      end if

    end subroutine

  end subroutine get_time_stepping
  
  
  subroutine get_les(fname)
    use Strings
    use ParseTrees
    character(*), intent(in) :: fname
    type(tree_object), allocatable :: tree(:)
    logical :: ex
    integer :: iobj, stat
    
    sgstype = -1
    filtertype = -1
    wallmodeltype = -1

    inquire(file=fname, exist=ex)

    if (.not.ex) return

    call parse_file(tree, fname, stat)

    if (stat==0) then

      if (allocated(tree)) then
        do iobj = 1, size(tree)
          if (downcase(tree(iobj)%name)=="subgrid_model") then
            call get_subgrid_model(tree(iobj))
          else if (downcase(tree(iobj)%name)=="filter") then
            call get_filter(tree(iobj))
          else if (downcase(tree(iobj)%name)=="wall_model") then
            call get_filter(tree(iobj))
          end if
          call tree(iobj)%finalize
        end do
      end if

    else

      write(*,*) "Error parsing file " // fname
      call error_stop

    end if
    
    ! The default is laminar simulation.
    ! If a subgrid model is enabled, the default is the wall model on.
    
    if (sgstype == -1) sgstype = 0
    
    if (filtertype == -1) filtertype = 0
    
    if (wallmodeltype == -1) then
      if (sgstype>0) then
        wallmodeltype = 1
      else
        wallmodeltype = 0
      end if
    end if
    
  contains
  
    subroutine get_subgrid_model(obj)
      use Subgrid
      type(tree_object), intent(in) :: obj
      integer :: i, j
      real(knd) :: C_m
      
      sgstype = -1
      C_m = -1
      
      if (allocated(obj%fields%array)) then

        associate(fields => obj%fields%array)

          do j = 1, size(fields)

            if (downcase(fields(j)%name)=='type') then            
              do i = 0, ubound(subgrid_model_names,1)
                if (downcase(fields(j)%value)==downcase(subgrid_model_names(i))) then
                  sgstype = i
                  exit
                end if                
              end do
              if (sgstype==-1) then
                write(*,*) "Unknown subgrid model '"//trim(fields(j)%value)//"' in " // fname
                write(*,'(1x,g0,*(/,tr4,g0))') "Valid values:", (trim(subgrid_model_names(i)),i=0,ubound(subgrid_model_names,1))
                call error_stop
              end if
            else if (downcase(fields(j)%name)=='model_constant') then
              read(fields(j)%value, *) C_m
            else
              write(*,*) "Unknown field '"//trim(fields(j)%name)//"' in the subgrid_model object in " // fname
              call error_stop
            end if

          end do

        end associate
        
        if (C_m > 0 .and. sgstype > 0) then
          select case (sgstype)
            case (1)
              C_Smagorinsky = C_m
            case (2)
              C_sigma = C_m
            case (3)
              C_Vreman = C_m
            case (4)
              C_sigma = C_m
            case (5)
              C_MixedTimeScale = C_m
            case (6)
              C_WALE = C_m
          end select
        end if

      else

        write(*,*) "No fields in the subgrid_model object in " // fname
        call error_stop

      end if
      
    end subroutine
    
    subroutine get_filter(obj)
      use Filters
      type(tree_object), intent(in) :: obj
      integer :: j
      if (allocated(obj%fields%array)) then

        associate(fields => obj%fields%array)

          do j = 1, size(fields)
            if (downcase(fields(j)%name)=='type') then 
              select case (downcase(fields(j)%value))
                case ('none')
                  filtertype = 0
                case ('default')
                  filtertype = 1
                case default
                  write(*,*) "Unknown wall model type '"//trim(fields(j)%value)//"' in " // fname
                  write(*,'(1x,g0,*(/,tr4,g0))') "Valid values: ", "none", "default"
                  call error_stop
              end select
            else
              write(*,*) "Unknown field '"//trim(fields(j)%name)//"' in the wall_model object in " // fname
              call error_stop
            end if
          end do

        end associate

      else

        write(*,*) "No fields in the wall_model object in " // fname
        call error_stop

      end if
    end subroutine
    
    subroutine get_wall_model(obj)
      use Filters
      type(tree_object), intent(in) :: obj
      integer :: j
      if (allocated(obj%fields%array)) then

        associate(fields => obj%fields%array)

          do j = 1, size(fields)
            if (downcase(fields(j)%name)=='type') then 
              select case (downcase(fields(j)%value))
                case ('none')
                  filtertype = 0
                case ('default')
                  filtertype = 1
                case default
                  write(*,*) "Unknown filter type '"//trim(fields(j)%value)//"' in " // fname
                  write(*,'(1x,g0,*(/,tr4,g0))') "Valid values: ", "none", "default"
                  call error_stop
              end select
            else
              write(*,*) "Unknown field '"//trim(fields(j)%name)//"' in the filter object in " // fname
              call error_stop
            end if
          end do

        end associate

      else

        write(*,*) "No fields in the filter object in " // fname
        call error_stop

      end if
    end subroutine
  end subroutine
  

  subroutine get_pressure_solution(fname, p_s)
    use Strings
    use ParseTrees
    use Boundaries, only: InGlobalDomain, InDomain
    character(*), intent(in) :: fname
    type(pressure_solution_control), intent(inout), target :: p_s
    type(tree_object), allocatable :: tree(:)
    type(tree_object_ptr) :: reference_obj_ptr
    character(char_len), target :: boundary_index_str = "", reference_type_str = ""
    logical :: ex
    integer :: iobj, stat

    type(field_names) :: names(15)
    type(field_names_a) :: names_a(1)

    type(field_names) :: reference_names(2)
    type(field_names_a) :: reference_names_a(1)

    names = [field_names_init("check_mass_flux", p_s%check_mass_flux), &
             field_names_init("report_mass_flux", p_s%report_mass_flux), &
             field_names_init("report_total_mass_flux", p_s%report_total_mass_flux), &
             field_names_init("correct_mass_flux_west",   p_s%correct_mass_flux(We)), &
             field_names_init("correct_mass_flux_east",   p_s%correct_mass_flux(Ea)), &
             field_names_init("correct_mass_flux_south",  p_s%correct_mass_flux(So)), &
             field_names_init("correct_mass_flux_north",  p_s%correct_mass_flux(No)), &
             field_names_init("correct_mass_flux_bottom", p_s%correct_mass_flux(Bo)), &
             field_names_init("correct_mass_flux_top",    p_s%correct_mass_flux(To)), &
             field_names_init("poisson_solver",      p_s%poisson_solver), &
             field_names_init("projection_method",   p_s%projection_method), &
             field_names_init("check_divergence",    p_s%check_divergence), &
             field_names_init("check_poisson_residue", p_s%check_poisson_residue), &
             field_names_init("bottom_pressure",     p_s%bottom_pressure), &
             field_names_init("reference", reference_obj_ptr)]

    names_a = [field_names_a_init("correct_mass_flux", p_s%correct_mass_flux)]
    
    reference_names = [field_names_init("type", reference_type_str), &
                       field_names_init("boundary_index", boundary_index_str)]
                            
    reference_names_a = [field_names_a_init("reference_point", p_s%reference%reference_point_xyz)]

    inquire(file=fname, exist=ex)

    if (.not.ex) return

    call parse_file(tree, fname, stat)

    if (stat==0) then

      if (allocated(tree)) then

        call find_object_get_field_values(tree, "pressure_solution", stat, &
                                     fields = names, fields_a = names_a)

        if (stat<=0) then
          if (associated(reference_obj_ptr%ptr)) then
                      
            call get_object_field_values(reference_obj_ptr%ptr, stat, &
                                         fields = reference_names, &
                                         fields_a = reference_names_a)
            if (stat/=0) then
              write(*,*) "Error parsing the reference object inside pressure_solution() in pressure_solution.conf"
              call error_stop
            end if
          end if
          
          call init
        end if

        do iobj = 1, size(tree)
          call tree(iobj)%finalize
        end do
      else
        write(*,*) "Error, no content in " // fname
        call error_stop
      end if

    else

      write(*,*) "Error parsing file " // fname
      call error_stop

    end if

  contains

    subroutine init

      if (any(p_s%correct_mass_flux)) p_s%check_mass_flux = .true.

      where(Btype>=BC_MPI_BOUNDS_MIN .and. Btype<=BC_MPI_BOUNDS_MAX) p_s%correct_mass_flux = .false.
    
      where(Btype==BC_NOSLIP) p_s%correct_mass_flux = .false.

      where(Btype==BC_PERIODIC) p_s%correct_mass_flux = .false.
      
      associate(p_s_ref => p_s%reference)
        if (len_trim(reference_type_str)>0) then
          select case (downcase(reference_type_str))
            case ("boundary")
              p_s_ref%type = PRESSURE_REFERENCE_BOUNDARY
            case ("volume_average")
              p_s_ref%type = PRESSURE_REFERENCE_VOLUME_AVERAGE
            case ("point")
              p_s_ref%type = PRESSURE_REFERENCE_POINT
            case ("none")
              p_s_ref%type = PRESSURE_REFERENCE_NONE
            case default
              write(*,*) "Error, unknown reference type '", &
                        reference_type_str, &
                        "' in reference() in pressure_solution() in pressure_solution.conf"
              call error_stop
          end select
        end if
        
        if (p_s_ref%type ==PRESSURE_REFERENCE_BOUNDARY) then
          if (len_trim(boundary_index_str)>0) then
            select case (downcase(boundary_index_str(1:1)))
              case("w","1")
                p_s_ref%boundary_index = We
              case("e","2")
                p_s_ref%boundary_index = Ea
              case("s","3")
                p_s_ref%boundary_index = So
              case("n","4")
                p_s_ref%boundary_index = No
              case("b","5")
                p_s_ref%boundary_index = Bo
              case("t","6")
                p_s_ref%boundary_index = To
              case default
                write(*,*) "Error, invalid value '", &
                            trim(boundary_index_str), &
                            "' of boundary_index in reference() in pressure_solution() in pressure_solution.conf"
                call error_stop
            end select
          end if
        else if (p_s_ref%type == PRESSURE_REFERENCE_POINT) then 
          if (.not. InGlobalDomain(p_s_ref%reference_point_xyz)) then
            write(*,*) "Error, pressure reference point ", &
                      p_s_ref%reference_point_xyz, &
                      "outside of the grid domain."
            call error_stop          
          end if
#ifdef PAR        
          block
            integer :: num
            if (InDomain(p_s_ref%reference_point_xyz)) then
              num = 1
              num = par_co_sum(num)
              
              if (num>1) then
                write(*,*) "Error, reference point on an image boundary, not yet implemented"
                call error_stop
              else
                call GridCoords_interp(p_s_ref%reference_point(1), &
                                      p_s_ref%reference_point(2), &
                                      p_s_ref%reference_point(3), &
                                      p_s_ref%reference_point_xyz(1), &
                                      p_s_ref%reference_point_xyz(2), &
                                      p_s_ref%reference_point_xyz(3))
                p_s_ref%reference_point_im = myim
              end if
            else
              num = 0
              num = par_co_sum(num)
              p_s_ref%reference_point_im = 0
            end if
          end block
          
          ! instead of a broadcast, we do not know on each image
          ! which image will be broadcasting
          p_s_ref%reference_point_im = par_co_sum(p_s_ref%reference_point_im)
          
#else
          call GridCoords_interp(p_s_ref%reference_point(1), &
                                p_s_ref%reference_point(2), &
                                p_s_ref%reference_point(3), &
                                p_s_ref%reference_point_xyz(1), &
                                p_s_ref%reference_point_xyz(2), &
                                p_s_ref%reference_point_xyz(3))
          p_s_ref%reference_point_im = myim
#endif        
        end if
      end associate
    end subroutine

  end subroutine get_pressure_solution
  
  
  
  subroutine get_frames(fname)
    use Strings
    use ParseTrees
    use Frames_common
    use VTKFrames
#ifdef PAR
    use Frames_ParallelIO
#endif
    character(*), intent(in) :: fname
    type(TFrameTimes), target :: timing
    type(TFrameFlags), target :: flags
    integer, target :: dimension, direction    
    real(knd), target :: position
    real(knd), target :: interval
    character(char_len), target :: label, direction_ch
    type(tree_object_ptr), target :: flags_ptr, bbox_ptr
    logical, target :: distributed ! whether small distributed vtk files for each image should be forced

    
    type :: bbox_cfg
      real(knd) :: xmin = -huge(1._knd)/2, xmax = huge(1._knd)
      real(knd) :: ymin = -huge(1._knd)/2, ymax = huge(1._knd)
      real(knd) :: zmin = -huge(1._knd)/2, zmax = huge(1._knd)
    end type
    
    type(bounding_box) :: bbox


    type(tree_object), allocatable :: tree(:)
    logical :: ex
    integer :: iobj, ivtk, stat

    type(field_names) :: names_vtk(9), names_flags_vtk(11), names_bbox(6)
    
    type(field_names_str) :: names_str(2)
    
    names_bbox = [field_names_init("xmin", bbox%xmin), &
                  field_names_init("xmax", bbox%xmax), &
                  field_names_init("ymin", bbox%ymin), &
                  field_names_init("ymax", bbox%ymax), &
                  field_names_init("zmin", bbox%zmin), &
                  field_names_init("zmax", bbox%zmax)]

    names_str = [field_names_str("label", label), &
                 field_names_str("direction", direction_ch)]


    names_vtk = [field_names_init("dimension",   dimension), &
                 field_names_init("position",    position), &
                 field_names_init("start",       timing%start), &
                 field_names_init("end",         timing%end), &
                 field_names_init("n",           timing%nframes), &
                 field_names_init("interval",    interval), &
                 field_names_init("flags",       flags_ptr), &
                 field_names_init("distributed", distributed), &
                 field_names_init("bbox",        bbox_ptr)]
                 

    names_flags_vtk = [field_names_init("U", flags%U), &
                       field_names_init("vorticity",  flags%vorticity), &
                       field_names_init("Pr",  flags%Pr), &
                       field_names_init("lambda2",  flags%lambda2), &
                       field_names_init("scalars",  flags%scalars), &
                       field_names_init("sumscalars",  flags%sumscalars), &
                       field_names_init("temperature",  flags%temperature), &
                       field_names_init("moisture",  flags%moisture), &
                       field_names_init("temperature_flux",  flags%temperature_flux), &
                       field_names_init("moisture_flux",  flags%moisture_flux), &
                       field_names_init("scalar_flux",  flags%scalar_flux)]
                 
    inquire(file=fname, exist=ex)

    if (.not.ex) return

    call parse_file(tree, fname, stat)

    if (stat/=0) then
      write(*,*) "Error parsing file " // fname
      call error_stop
    end if
  
    if (.not.allocated(tree)) then
      write(*,*) "Error, no content in " // fname
      call error_stop
    end if
    
    ivtk = 0

    do iobj = 1, size(tree)
    
      if (downcase(tree(iobj)%name)=="vtk_frame_domain") then
      
        label = ""
        direction_ch = ""
        timing%start = 0; timing%end = 0; timing%nframes = 0
        interval = 0
        distributed = .false.
        flags_ptr%ptr => null()
        bbox_ptr%ptr => null()
        
        call get_object_field_values(tree(iobj), stat, &
                                     fields = names_vtk, fields_str = names_str)
                                     
        if (stat/=0) then
          call error_stop("Error interpretting fields in vtk_frame_domain '"//trim(label)//"'.")
        end if
        
        if (associated(flags_ptr%ptr)) then
          call get_object_field_values(flags_ptr%ptr, stat, &
                                       fields = names_flags_vtk)
          if (stat/=0) then
            call error_stop("Error interpretting flags object in vtk_frame_domain '"//trim(label)//"'.")
          end if
        end if
        
        if (associated(bbox_ptr%ptr)) then
          call get_object_field_values(bbox_ptr%ptr, stat, &
                                       fields = names_bbox)
          if (stat/=0) then
            call error_stop("Error interpretting bbox object in vtk_frame_domain '"//trim(label)//"'.")
          end if
        end if
        
        if (associated(bbox_ptr%ptr)) then
          call init_vtk_domain(bbox)
        else
          call init_vtk_domain
        end if
      end if
      
      call tree(iobj)%finalize
    end do
    

  contains

    subroutine init_vtk_domain(bbox)
      type(bounding_box), optional, intent(in) :: bbox
      
      ivtk = ivtk + 1
      
      if (label=="") label = achar(iachar('a')+ivtk-1)
      
      if (dimension==0) then
        dimension = 2
      else if (dimension < 2 .or. dimension > 3) then
        call error_stop("Error, dimension must be 2 or 3 in vtk_frame_domain '"//trim(label)//"'.")
      end if
      
      if (downcase(direction_ch(1:1))=="x") then
        direction = 1
      else if (downcase(direction_ch(1:1))=="y") then
        direction = 2
      else if (downcase(direction_ch(1:1))=="z") then
        direction = 3
      else if (len_trim(direction_ch) > 0) then
        read(direction_ch,*,iostat=stat) direction
        if (stat/=0) &
          call error_stop("Error, unrecognized value of direction in vtk_frame_domain '"//trim(label)// &
                          "'. Got:`"//trim(direction_ch)//"`.")
        if (direction < 1 .or. direction > 3) &
          call error_stop("Invalid value of direction in vtk_frame_domain '"//trim(label)//"'. Expected 1, 2 or 3.")
      else if (dimension==2) then
        call error_stop("Error, direction must be specified for 2D vtk_frame_domain '"//trim(label)//"'.")
      end if
      
      if (timing%nframes==0.and.interval>0) then     
        timing%nframes = int((timing%end - timing%start) / interval) + 1
        timing%end = timing%start + interval * (timing%nframes - 1)
      end if
    
      if (num_of_scalars < 1) flags%scalars = 0
      if (num_of_scalars < 1) flags%sumscalars = 0
      if (.not.enable_buoyancy) flags%temperature = 0
      if (.not.enable_moisture) flags%moisture = 0
      if (.not.enable_buoyancy) flags%temperature_flux = 0
      if (.not.enable_moisture) flags%moisture_flux = 0
      if (num_of_scalars < 1) flags%scalar_flux = 0
      
#ifdef PAR
      if (.not.(distributed.or.nims==1)) then
        call AddDomain(TFrameDomain_ParallelIO(label, &
                                    dimension, direction, position, &
                                    timing, flags, bbox))
      else
#endif
        call AddDomain(TFrameDomain(label, &
                                    dimension, direction, position, &
                                    timing, flags, bbox))
#ifdef PAR
      end if
#endif      
    end subroutine
    
  end subroutine get_frames
  
  
  
  subroutine get_scalar_zones(fname)
    use Strings
    use ParseTrees
    use Scalars

    character(*), intent(in) :: fname
    type(TScalarRemovalZone), target :: zone
    real(knd), target :: bounds(6)
    character(char_len), target :: type_str

    type(tree_object), allocatable :: tree(:)
    integer :: iobj, stat
    logical :: ex

    type(field_names) :: names(2)
    type(field_names_a) :: names_a(1)
    
    type(field_names_str) :: names_str(1)
    


    names = [field_names_init("relaxation_time", zone%relaxation_time), &
              field_names_init("scalar",  zone%scalar)]
                 
    names_a = [field_names_a_init("bounds", bounds)]
                 
    names_str = [field_names_str("type", type_str)]

    inquire(file=fname, exist=ex)

    if (.not.ex) return

    call parse_file(tree, fname, stat)

    if (stat/=0) then
      write(*,*) "Error parsing file " // fname
      call error_stop
    end if
  
    if (.not.allocated(tree)) then
      write(*,*) "Error, no content in " // fname
      call error_stop
    end if
   
    do iobj = 1, size(tree)
    
      if (downcase(tree(iobj)%name)=="scalar_removal_zone") then
      
        zone = TScalarRemovalZone()
        type_str = ""
        bounds = 0
        
        call get_object_field_values(tree(iobj), stat, &
                                     fields = names, fields_str = names_str, fields_a = names_a)
                                     
        if (stat/=0) then
          call error_stop("Error interpretting fields in scalar_removal_zone '"//itoa(iobj)//"'.")
        end if
               
        call init_zone

      else
        call error_stop("Error, unrecognized object '"//trim(tree(iobj)%name)//"' in " // fname)
      end if
      
      call tree(iobj)%finalize
    end do
    

  contains

    subroutine init_zone
      if (any(bounds/=0)) then
        zone%x1 = bounds(1)
        zone%x2 = bounds(2)
        zone%y1 = bounds(3)
        zone%y2 = bounds(4)
        zone%z1 = bounds(5)
        zone%z2 = bounds(6)
      end if

      if (zone%relaxation_time > 0 .and. zone%relaxation_time < huge(1._knd)) then
        zone%type = removal_continuous
      end if

      if (len_trim(type_str)>0) then
        select case (downcase(type_str(1:4)))
          case ("cont")
            zone%type = removal_continuous
          case ("inst")
            zone%type = removal_instantaneous
          case default
            call error_stop("Error, unrecognized type of scalar_removal_zone '"//trim(type_str)//"'.")
        end select
      end if

      if (zone%x1>zone%x2 .or. zone%y1>zone%y2 .or. zone%z1>zone%z2) then
        zone%active = .false.
      end if

      if (zone%scalar==0) then
        zone%active = .false.
      end if

      if (zone%active) then
        call AddScalarRemovalZone(zone)
      end if
    end subroutine
    
  end subroutine get_scalar_zones
  
  
  
  
#ifdef PAR
  subroutine get_domains(fname)
    use Strings
    use ParseTrees
    use custom_par
    character(*), intent(in) :: fname
    type(tree_object), allocatable :: tree(:)
    logical :: ex
    integer :: iobj, stat

    type(field_names) :: names(8)
    type(field_names_a) :: names_a(5)
    type(field_names_a_int_alloc) :: names_a_int_alloc(1)
    
    logical, target :: enable_multiple_domains_l = .false.
    logical, target :: is_two_way_nested_l = .false.
    logical, target :: receive_initial_conditions_from_parent_l = .false.
    integer, target :: number_of_domains_l = -99
    integer, target :: domain_index_l = -99
    integer, target :: parent_domain_l = -99
    integer, target :: spatial_ratio_l = -99
    integer, target :: time_step_ratio_l = -99

    logical, target :: is_domain_boundary_nested_l(6) = .true.
    logical, target :: has_domain_boundary_turbulence_generator_l(6) = .false.
    logical, target :: has_domain_boundary_relaxation_l(6) = .true.
    integer, target :: domain_boundary_relaxation_width_l(6) = 2 !in parent domain units
    real(knd), target :: domain_boundary_relaxation_factor_l(6) = 1

    names = [field_names_init("enable_multiple_domains", enable_multiple_domains_l), &
             field_names_init("is_two_way_nested",   is_two_way_nested_l), &
             field_names_init("receive_initial_conditions",   receive_initial_conditions_from_parent_l), &
             field_names_init("domain_index",   domain_index_l), &
             field_names_init("parent_domain",   parent_domain_l), &
             field_names_init("number_of_domains",   number_of_domains_l), &
             field_names_init("spatial_ratio",   spatial_ratio_l), &
             field_names_init("time_step_ratio",   time_step_ratio_l)]

    names_a = [field_names_a_init("is_boundary_nested", is_domain_boundary_nested_l), &
               field_names_a_init("has_boundary_turbulence_generator", has_domain_boundary_turbulence_generator_l), &
               field_names_a_init("has_boundary_relaxation", has_domain_boundary_relaxation), &
               field_names_a_init("boundary_relaxation_factor", domain_boundary_relaxation_factor), &
               field_names_a_init("boundary_relaxation_width", domain_boundary_relaxation_width)]

    names_a_int_alloc = [field_names_a_int_alloc_init("child_domains")]

    inquire(file=fname, exist=ex)

    if (.not.ex) return

    call parse_file(tree, fname, stat)

    if (stat==0) then

      if (allocated(tree)) then

        call find_object_get_field_values(tree, "domains", stat, &
                                     fields = names, fields_a = names_a, fields_a_int_alloc = names_a_int_alloc)

        if (stat<=0) call init

        do iobj = 1, size(tree)
          call tree(iobj)%finalize
        end do
      else
        write(*,*) "Error, no content in " // fname
        call error_stop
      end if

    else

      write(*,*) "Error parsing file " // fname
      call error_stop

    end if

  contains

    subroutine init

      enable_multiple_domains = enable_multiple_domains_l
      
      if (enable_multiple_domains) then
          
        number_of_domains = number_of_domains_l
        if (number_of_domains<1) &
          call error_stop("Error, positive number_of_domains must be specified in domains.")
    
        domain_index = domain_index_l
        if (domain_index<1) &
          call error_stop("Error, positive domain_index must be specified in domains.")
        if (number_of_domains<domain_index) &
          call error_stop("Error, domain_index must be smaller or equal to number_of_domains.")

        parent_domain = max(parent_domain_l, 0)
        if (domain_index>1 .and. parent_domain<=0) &
          call error_stop("Error, parent_domain shall be specified and positive if domain_index > 1.")

        if (allocated(names_a_int_alloc(1)%var)) then
          call move_alloc(names_a_int_alloc(1)%var, child_domains)
          if (any(child_domains<=domain_index) .or. &
              any(child_domains>number_of_domains)) then
            call error_stop("Error, child_domains shall be larger than domain_index &
                           &and smaller or equal to number_of_domains.")
          end if
        else if (domain_index<number_of_domains) then

          call error_stop("Error, child_domains shall be specified if domain_index < number_of_domains. &
                          & It may be empty.")
        end if
        
        receive_initial_conditions_from_parent = receive_initial_conditions_from_parent_l
          
        is_this_domain_two_way_nested = is_two_way_nested_l

        is_domain_boundary_nested = is_domain_boundary_nested_l

        where(Btype==BC_NOSLIP) is_domain_boundary_nested = .false.

        has_domain_boundary_turbulence_generator = has_domain_boundary_turbulence_generator_l
        where (.not.is_domain_boundary_nested) has_domain_boundary_turbulence_generator = .false.
        
        has_domain_boundary_relaxation = has_domain_boundary_relaxation_l
        where (.not.is_domain_boundary_nested) has_domain_boundary_relaxation = .false.
        
        domain_boundary_relaxation_factor = domain_boundary_relaxation_factor_l

        domain_boundary_relaxation_width = domain_boundary_relaxation_width_l
        where (.not.has_domain_boundary_relaxation) domain_boundary_relaxation_width = 0

        if (spatial_ratio_l>0) domain_spatial_ratio = spatial_ratio_l
        if (time_step_ratio_l>0) domain_time_step_ratio = time_step_ratio_l
      end if
      
    end subroutine

  end subroutine get_domains
#endif  
  
  
  

  subroutine ReadInitialConditions(U,V,W,Pr,Temperature,Moisture,Scalar,scalars_optional)
    use Endianness
    real(knd), intent(inout) :: U(-2:,-2:,-2:),V(-2:,-2:,-2:),W(-2:,-2:,-2:)
    real(knd), intent(inout) :: Pr(-1:,-1:,-1:)
    real(knd), intent(inout) :: Temperature(-2:,-2:,-2:)
    real(knd), intent(inout) :: Moisture(-2:,-2:,-2:)
    real(knd), intent(inout) :: Scalar(-2:,-2:,-2:,:)
    logical, intent(in) :: scalars_optional
    real(real32), allocatable :: buffer(:,:,:), UVWbuffer(:,:,:,:)
    integer :: i, unit, stat, file_stat
    logical :: exU(3)
    character(2) :: scalnum

    allocate(buffer(Prnx,Prny,Prnz))

    open(newunit=unit,file=input_dir//"out.vtk",access="stream",status="old",action="read",iostat=file_stat)

    if (file_stat/=0) call error_stop("Error opening "//input_dir//"out.vtk")

    call skip_to("SCALARS p float",stat)
    if (stat/=0) then
      Pr = 0
      rewind(unit)
    else
      call skip_line
      call skip_line
      read(unit) buffer
      Pr(1:Prnx,1:Prny,1:Prnz) =  real(BigEnd(buffer),knd)
    end if

    if (enable_buoyancy) then
      call skip_to("SCALARS temperature float",stat)
      if (stat/=0) then
        call error_stop("No temperature field found in the initial conditions file " // &
                        input_dir // "out.vtk")
      else
        call skip_line
        call skip_line
        read(unit) buffer
        Temperature(1:Prnx,1:Prny,1:Prnz) =  real(BigEnd(buffer),knd)
      end if
    end if

    if (enable_moisture) then
      call skip_to("SCALARS moisture float",stat)
      if (stat/=0) then
        call error_stop("No moisture field found in the initial conditions file " // &
                        input_dir // "out.vtk")
      else
        call skip_line
        call skip_line
        read(unit) buffer
        Moisture(1:Prnx,1:Prny,1:Prnz) =  real(BigEnd(buffer),knd)
      end if
    end if

    close(unit)


    if (num_of_scalars > 0) then
      open(newunit=unit,file=input_dir//"scalars.vtk",access="stream",status="old",action="read",iostat=file_stat)

      if (file_stat/=0 .and. scalars_optional) then
        !TODO: check consistency between images?
        if (master) write(*,*) "Warning, no scalar initial conditions found."
        if (master) write(*,*) "Scalar fields set to zero."
        
        Scalar = 0

      else if (file_stat/=0) then

        call error_stop("Error opening "//input_dir//"scalars.vtk")

      else

        do i = 1, num_of_scalars
          write(scalnum,"(I2.2)") i
          call skip_to("SCALARS scalar"//scalnum//" float",stat)
          call skip_line
          if (stat/=0) then
            call error_stop("scalar"//scalnum//" field not found in the initial conditions file " // &
                            input_dir // "scalars.vtk")
          else
            call skip_line
            read(unit) buffer
            Scalar(1:Prnx,1:Prny,1:Prnz,i) =  real(BigEnd(buffer),knd)
          end if
        end do

        close(unit)

      end if

    end if


    deallocate(buffer)


    inquire(file=input_dir//"U.vtk", exist=exU(1))
    inquire(file=input_dir//"V.vtk", exist=exU(2))
    inquire(file=input_dir//"W.vtk", exist=exU(3))
    
    if (all(exU)) then
    
      open(newunit=unit,file=input_dir//"U.vtk",access="stream",status="old",action="read")
      call skip_to("SCALARS U float",stat)
      call skip_line
      call skip_line
      allocate(buffer(1:Unx, 1:Uny, 1:Unz))
      read(unit) buffer
      U(1:Unx, 1:Uny, 1:Unz) = real(BigEnd(buffer), knd)
      deallocate(buffer)
      close(unit)

      open(newunit=unit,file=input_dir//"V.vtk",access="stream",status="old",action="read")
      call skip_to("SCALARS V float",stat)
      call skip_line
      call skip_line
      allocate(buffer(1:Vnx, 1:Vny, 1:Vnz))
      read(unit) buffer
      V(1:Vnx, 1:Vny, 1:Vnz) = real(BigEnd(buffer), knd)
      deallocate(buffer)
      close(unit)

      open(newunit=unit,file=input_dir//"W.vtk",access="stream",status="old",action="read")
      call skip_to("SCALARS W float",stat)
      call skip_line
      call skip_line
      allocate(buffer(1:Wnx, 1:Wny, 1:Wnz))
      read(unit) buffer
      W(1:Wnx, 1:Wny, 1:Wnz) = real(BigEnd(buffer), knd)
      deallocate(buffer)
      close(unit)
      
    else
    
      open(newunit=unit,file=input_dir//"out.vtk",access="stream",status="old",action="read")
      call skip_to("VECTORS u float",stat)
      call skip_line
      allocate(UVWbuffer(3, 1:Prnx, 1:Prny, 1:Prnz))
      read(unit) UVWbuffer
      UVWbuffer = BigEnd(UVWbuffer)
      close(unit)
      
      U = 0
      U(1:Prnx,1:Prny,1:Prnz) = real(UVWbuffer(1,1:Prnx,1:Prny,1:Prnz),knd)
      U(0:Prnx-1,1:Prny,1:Prnz) = U(0:Prnx-1,1:Prny,1:Prnz) + U(1:Prnx,1:Prny,1:Prnz)
    
      V = 0
      V(1:Prnx,1:Prny,1:Prnz) = real(UVWbuffer(2,1:Prnx,1:Prny,1:Prnz),knd)
      V(1:Prnx,0:Prny-1,1:Prnz) = V(1:Prnx,0:Prny-1,1:Prnz) + V(1:Prnx,1:Prny,1:Prnz)
    
      W = 0
      W(1:Prnx,1:Prny,1:Prnz) = real(UVWbuffer(3,1:Prnx,1:Prny,1:Prnz),knd)
      W(0:Prnx-1,1:Prny,1:Prnz) = W(0:Prnx-1,1:Prny,1:Prnz) + W(1:Prnx,1:Prny,1:Prnz)

#ifdef PAR
      call par_exchange_U_x(U, 1)
      call par_exchange_U_y(V, 2)
      call par_exchange_U_z(W, 3)
#endif
      if ((Btype(Ea)>=BC_MPI_BOUNDS_MIN .and. (Btype(Ea)<=BC_MPI_BOUNDS_MAX)) .or. &
          Btype(Ea)==BC_PERIODIC) &
        U(Prnx,1:Prny,1:Prnz) = U(Prnx,1:Prny,1:Prnz) + U(0,1:Prny,1:Prnz)
      if ((Btype(No)>=BC_MPI_BOUNDS_MIN .and. (Btype(No)<=BC_MPI_BOUNDS_MAX)) .or. &
          Btype(No)==BC_PERIODIC) &
        V(1:Prnx,Prny,1:Prnz) = V(1:Prnx,Prny,1:Prnz) + V(1:Prnx,0,1:Prnz)
      if ((Btype(To)>=BC_MPI_BOUNDS_MIN .and. (Btype(To)<=BC_MPI_BOUNDS_MAX)) .or. &
          Btype(To)==BC_PERIODIC) &
        W(1:Prnx,1:Prny,Prnz) = W(1:Prnx,1:Prny,Prnz) + W(1:Prnx,1:Prny,0)

      U = U / 2
      V = V / 2
      W = W / 2
    
    end if


  contains

    subroutine skip_line
      character :: ch
      do
        read(unit) ch
        if (ch==new_line("a")) return
      end do
    end subroutine

    subroutine skip_to(str, stat)
      character(*), intent(in) :: str
      integer, intent(out) :: stat
      character :: ch
      integer :: io

      do
        read(unit, iostat=io) ch

        if (io/=0) then
          stat = 1
          return
        end if

        if (ch==str(1:1)) then
          call check(str(2:), stat)
          if (stat == 0) return
        end if

      end do
    end subroutine

    subroutine check(str, stat)
      character(*), intent(in) :: str
      integer, intent(out) :: stat
      character :: ch
      integer :: i, io

      stat = 1
      i = 0

      do
        i = i + 1

        read(unit, iostat=io) ch

        if (io/=0) return

        if (ch/=str(i:i)) return

        if (i==len(str)) then
          stat = 0
          return
        end if
      end do
    end subroutine

  end subroutine ReadInitialConditions



  subroutine InitialConditions(U,V,W,Pr,Temperature,Moisture,Scalar,dt)
    use custom_par
#ifdef PAR
    use domains_bc_par, only: par_receive_initial_conditions, par_send_initial_conditions
#endif
    use ArrayUtilities
    real(knd),contiguous,intent(inout) :: U(-2:,-2:,-2:),V(-2:,-2:,-2:),W(-2:,-2:,-2:),Pr(-1:,-1:,-1:)
    real(knd),contiguous,intent(inout) :: Temperature(-2:,-2:,-2:)
    real(knd),contiguous,intent(inout) :: Moisture(-2:,-2:,-2:)
    real(knd),contiguous,intent(inout) :: Scalar(-2:,-2:,-2:,:)
    real(knd), intent(out) :: dt
    integer :: i,j,k
    real(knd) :: p,x,y,z,x1,x2,y1,y2,z1,z2
    real(knd) :: U_initial_scaling
    real(knd),allocatable :: Q(:,:,:)
    logical :: receive_ic

#ifdef CUSTOM_INITIAL_CONDITIONS
    interface
      subroutine CustomInitialConditions(U,V,W,Pr,Temperature,Moisture,Scalar)
        use Parameters
        real(knd),contiguous,intent(inout) :: U(-2:,-2:,-2:),V(-2:,-2:,-2:),W(-2:,-2:,-2:),Pr(-1:,-1:,-1:)
        real(knd),contiguous,intent(inout) :: Temperature(-2:,-2:,-2:)
        real(knd),contiguous,intent(inout) :: Moisture(-2:,-2:,-2:)
        real(knd),contiguous,intent(inout) :: Scalar(-2:,-2:,-2:,:)
      end subroutine
    end interface
#endif

    call par_sync_out("  ...setting initial dummy values.")

    if (abs(Uinlet)>0) then
      dt = min(abs(dxmin/Uinlet), abs(dymin/Uinlet), abs(dzmin/Uinlet))
    else if (maxval(abs(time_stepping%U_min))>0) then
      dt = min(abs(dxmin/time_stepping%U_min(1)), &
               abs(dymin/time_stepping%U_min(2)), &
               abs(dzmin/time_stepping%U_min(3)))
    else
      dt = dxmin
    end if
        
    Pr(1:Prnx,1:Prny,1:Prnz) = 0

    U = huge(1._knd)/2
    V = huge(1._knd)/2
    W = huge(1._knd)/2

    V(1:Vnx,1:Vny,1:Vnz) = 0
    W(1:Wnx,1:Wny,1:Wnz) = 0

    receive_ic = .false.

    if (initcondsfromfile>0) then

      call par_sync_out("  ...reading initial conditions from input files.")

      if (initcondsfromfile==2) then
        call ReadInitialConditions(U,V,W,Pr,Temperature,Moisture,Scalar,scalars_optional=.true.)
      else
        call ReadInitialConditions(U,V,W,Pr,Temperature,Moisture,Scalar,scalars_optional=.false.)
      end if
      Viscosity = molecular_viscosity

      if (enable_buoyancy.or. &
          enable_moisture.or. &
          num_of_scalars>0)        TDiff = molecular_diffusivity

      call BoundUVW(U, V, W)
      call Bound_Pr(Pr)

    else   !init conditions not from file

        call par_sync_out("  ...computing initial conditions.")

#ifdef PAR
        if (parent_domain>0.and.receive_initial_conditions_from_parent) then
          receive_ic = .true.
        else
#endif


#ifdef CUSTOM_INITIAL_CONDITIONS
        call CustomInitialConditions(U,V,W,Pr,Temperature,Moisture,Scalar)
#else

        if (task_type==2) then
           U(1:Unx,1:Uny,1:Unz) = 0
           do k = 1, Unz
            do j = 1, Uny
             do i = 1, Unx
                    x = xU(i)
                    y = yPr(j)
                    z = zPr(k)
                    U(i,j,k) = -cos(pi*x)*sin(pi*y)
             end do
            end do
           end do
           do k = 1, Vnz
            do j = 1, Vny
             do i = 1, Vnx
                    x = xPr(i)
                    y = yV(j)
                    z = zPr(k)
                    V(i,j,k) = sin(pi*x)*cos(pi*y)
             end do
            end do
           end do
           do k = 1, Wnz
            do j = 1, Wny
             do i = 1, Wnx
                    x = xPr(i)
                    y = yPr(j)
                    z = zW(k)
                    W(i,j,k) = 0
             end do
            end do
           end do
           do k = 1, Prnz
            do j = 1, Prny
             do i = 1, Prnx
                    x = xPr(i)
                    y = yPr(j)
                    z = zPr(k)
                    Pr(i,j,k) = -(1._knd/4._knd)*((cos(2*pi*y)+cos(2*pi*x)))
             end do
            end do
           end do

        elseif (task_type==3) then
           U(1:Unx,1:Uny,1:Unz) = 0
           do k = 1, Unz
            do j = 1, Uny
             do i = 1, Unx

                    x = xU(i)
                    y = yPr(j)
                    z = zPr(k)
                    U(i,j,k) = Uinlet*sin(x)*cos(z)*cos(-y)
             end do
            end do
           end do
           do k = 1, Vnz
            do j = 1, Vny
             do i = 1, Vnx
                    x = xPr(i)
                    y = yV(j)
                    z = zPr(k)
                    V(i,j,k) = 0
             end do
            end do
           end do
           do k = 1, Wnz
            do j = 1, Wny
             do i = 1, Wnx
                    x = xPr(i)
                    y = yPr(j)
                    z = zW(k)
                    W(i,j,k) = -Uinlet*cos(x)*sin(z)*cos(-y)
             end do
            end do
           end do
           do k = 1, Prnz
            do j = 1, Prny
             do i = 1, Prnx
                    x = xPr(i)
                    y = yPr(j)
                    z = zPr(k)
                    Pr(i,j,k) = (Uinlet/16._knd)*((2+cos(2*(-y)))*(cos(2*(z))+cos(2*(x)))-2)
             end do
            end do
           end do

        else if (InletType==TurbulentInletType) then

            call par_sync_out("  ...computing turbulent initial conditions.")
            
            U_initial_scaling = hypot(maxval(Uin),maxval(Vin))
            
            if (time_stepping%variable_time_steps) then
              U_initial_scaling = max(U_initial_scaling, maxval(time_stepping%U_min))
            end if
            
            dt = hypot(dxmin,dymin) / U_initial_scaling
            
            if (time_stepping%variable_time_steps) then
              dt = min(dt, time_stepping%dt)
            end if

            if (default_turbulence_generator%direction==2) then
              if (jim==1.and.Btype(So)==BC_TURBULENT_INLET) then
            
                do j = 1, Prny
                  
                  call default_turbulence_generator%step_xc(Uin, Vin, Win, time_stepping%dt)

                  !$omp parallel private(i,k)
                  !$omp do collapse(2)
                  do k = 1, Unz
                    do i = 1, Unx
                      if (Utype(i,j,k)<=0) then
                        U(i,j,k) = Uin(i,k)
                      else
                        U(i,j,k) = 0
                      end if
                    end do
                  end do
                  !$omp end do nowait
                  !$omp do collapse(2)
                  do k = 1, Vnz
                    do i = 1, Vnx
                      if (Vtype(i,j,k)<=0) then
                        V(i,j,k) = Vin(i,k)
                      else
                        V(i,j,k) = 0
                      end if
                    end do
                  end do
                  !$omp end do nowait
                  !$omp do collapse(2)
                  do k = 1, Wnz
                    do i = 1, Wnx
                      if (Wtype(i,j,k)<=0) then
                        W(i,j,k) = Win(i,k)
                      else
                        W(i,j,k) = 0
                      end if
                    end do
                  end do
                  !$omp end do
                  !$omp end parallel             
                end do
#ifdef PAR                
                block
                  use custom_par
                  integer :: ierr, im
                  do im = 2, nyims
                    do j = 1, domain_grids(domain_index)%nys(im)
                      call default_turbulence_generator%step_xc(Uin, Vin, Win, time_stepping%dt)
                      
                      call MPI_Send(Uin(1:Unx,1:Unz), Unx*Unz, PAR_KND, &
                                    ranks_grid(iim,im,kim), 111, domain_comm, ierr)
                      call MPI_Send(Vin(1:Vnx,1:Vnz), Vnx*Vnz, PAR_KND, &
                                    ranks_grid(iim,im,kim), 112, domain_comm, ierr)
                      call MPI_Send(Win(1:Wnx,1:Wnz), Wnx*Wnz, PAR_KND, &
                                    ranks_grid(iim,im,kim), 113, domain_comm, ierr)
                    end do
                  end do
                end block
#endif                
              else
#ifdef PAR              
                block
                  use custom_par
                  integer :: ierr, stat(MPI_STATUS_SIZE)
                  do j = 1, Prny
                    call MPI_Recv(U(1:Unx,j,1:Unz), Unx*Unz, PAR_KND, &
                                  ranks_grid(iim,1,kim), 111, domain_comm, stat, ierr)
                    call MPI_Recv(V(1:Vnx,j,1:Vnz), Vnx*Vnz, PAR_KND, &
                                  ranks_grid(iim,1,kim), 112, domain_comm, stat, ierr)
                    call MPI_Recv(W(1:Wnx,j,1:Wnz), Wnx*Wnz, PAR_KND, &
                                  ranks_grid(iim,1,kim), 113, domain_comm, stat, ierr)
                  end do
                end block
#endif                
              end if
              
            else

              do i = 1, Prnx
                
                call default_turbulence_generator%step_xc(Uin, Vin, Win, time_stepping%dt)
                
                !$omp parallel private(j,k)
                !$omp do collapse(2)
                do k = 1, Unz
                  do j = 1, Uny
                    if (Utype(i,j,k)<=0) then
                      U(i,j,k) = Uin(j,k)
                    else
                      U(i,j,k) = 0
                    end if
                  end do
                end do
                !$omp end do nowait
                !$omp do collapse(2)
                do k = 1, Vnz
                  do j = 1, Vny
                    if (Vtype(i,j,k)<=0) then
                      V(i,j,k) = Vin(j,k)
                    else
                      V(i,j,k) = 0
                    end if
                  end do
                end do
                !$omp end do nowait
                !$omp do collapse(2)
                do k = 1, Wnz
                  do j = 1, Wny
                    if (Wtype(i,j,k)<=0) then
                      W(i,j,k) = Win(j,k)
                    else
                      W(i,j,k) = 0
                    end if
                  end do
                end do
                !$omp end do
                !$omp end parallel
              end do
            end if
            
            !sets the Uin to mean values
            call default_turbulence_generator%Uin_reset(Uin, Vin, Win)

        else

           call par_sync_out("  ...setting initial conditions.")

           !$omp parallel private(i,j,k)
           !$omp do collapse(3)
           do k = 1, Unz
            do j = 1, Uny
             do i = 1, Unx
              if (Utype(i,j,k)<=0) then
                 U(i,j,k) = Uin(j,k)
               else
                 U(i,j,k) = 0
              end if
             end do
            end do
           end do
           !$omp end do nowait
           !$omp do collapse(3)
           do k = 1, Vnz
            do j = 1, Vny
             do i = 1, Vnx
              if (Vtype(i,j,k)<=0) then
                 V(i,j,k) = Vin(j,k)
               else
                 V(i,j,k) = 0
              end if
             end do
            end do
           end do
           !$omp end do nowait
           !$omp do collapse(3)
           do k = 1, Wnz
            do j = 1, Wny
             do i = 1, Wnx
              if (Wtype(i,j,k)<=0) then
                 W(i,j,k) = Win(j,k)
               else
                 W(i,j,k) = 0
              end if
             end do
            end do
           end do
           !$omp end do
           !$omp end parallel
        end if  !task_type


        if (num_of_scalars>0) then
          call par_sync_out("  ...setting initial scalar values.")
          !$omp parallel
          !$omp workshare
          SCALAR(1:Prnx,1:Prny,1:Prnz,:) = 0
          !$omp end workshare
          !$omp end parallel
        end if

        if (enable_buoyancy.and.task_type==2) then
          call par_sync_out("  ...setting initial temperature values.")

          do k = 0, Prnz+1
          do j = 0, Prny+1
            do i = 0, Prnx+1
            x = xPr(i)
            y = yPr(j)
            z = zPr(k)
            if ((x)**2+(y-0.5)**2<0.2_knd**2) then
              temperature(i,j,k) = cos(sqrt(x**2+(y-0.5)**2)*pi/2/0.2_knd)**2
            else
              temperature(i,j,k) = 0
            end if
            end do
          end do
          end do

        elseif (enable_buoyancy.and.task_type==3) then
          call par_sync_out("  ...setting initial temperature values.")

          do k = 0, Prnz+1
          do j = 0, Prny+1
            do i = 0, Prnx+1
            x = xPr(i)
            y = yPr(j)
            z = zPr(k)
            temperature(i,j,k) = temperature_ref + &
                (temperature_ref/100._knd) * ((2+cos(2*z))*(cos(2*(-y))+cos(2*(x)))-2)
            end do
          end do
          end do

        elseif (enable_buoyancy) then
          call par_sync_out("  ...setting initial temperature values.")

          call InitScalarProfile(TempIn,TemperatureProfileObj,temperature_ref)

          call InitScalar(TempIn,TemperatureProfileObj,Temperature)
          
          !only affects Dirichlet BC's without a wall model
          call UpdateSurfaceTemperatures(U, V, W, Pr, Temperature, Moisture)

        end if !buoyancy and task_type

        if (enable_moisture) then
          call par_sync_out("  ...setting initial moisture values.")

          call InitScalarProfile(MoistIn,MoistureProfileObj,moisture_ref)

          call InitScalar(MoistIn,MoistureProfileObj,Moisture)

        end if

!end not custom initial conditions
#endif


         if (enable_buoyancy) then
           call par_sync_out("  ...setting hydrostatic pressure.")

           call InitHydrostaticPressure(Pr,Temperature,Moisture)

         end if
       
#ifdef PAR
       end if !initial conditions not from parent
#endif
  
     end if !init conditions not from file


     call par_sync_out("  ...setting initial viscosity and diffusivity.")

     call set(Viscosity, molecular_viscosity)

     if (molecular_viscosity > 0 .and. &
           (enable_buoyancy .or. &
            enable_moisture .or. &
            num_of_scalars > 0))     then

       call set(TDiff, molecular_diffusivity)

     end if

#ifdef PAR
     if (enable_multiple_domains) then
       call par_sync_out("  ...getting initial conditions from parent (if there is one).")

       call par_receive_initial_conditions(receive_ic, U, V, W, Pr, Temperature, Moisture, Scalar)

       call par_sync_out("  ...getting boundary conditions from parent (if there is one).")

       call par_exchange_domain_bounds(U, V, W, Temperature, Moisture, Scalar, time_stepping%time, epsilon(1._knd), send=.false.)

       call par_update_domain_bounds(U, V, W, Temperature, Moisture, Scalar, time_stepping%start_time)
     end if
#endif

     call par_sync_out("  ...setting ghost cell values.")

     call BoundUVW(U, V, W)
     call Bound_Pr(Pr)

     call par_sync_out("  ...computing pressure correction.")
     call PressureCorrection(U,V,W,Pr,Q,1._knd)
     call par_sync_out("  ...computing initial eddy viscosity.")
     call SubgridModel(U, V, W)

     call par_sync_out("  ...setting viscosity in ghost cells.")
     call BoundViscosity(Viscosity)

     if (enable_buoyancy.or. &
         enable_moisture.or. &
         num_of_scalars>0)     then
       call par_sync_out("  ...setting subgrid diffusivity.")

       !$omp parallel do private(i,j,k)
       do k = 1, Prnz
         do j = 1, Prny
           do i = 1, Prnx
             TDiff(i,j,k) = (Viscosity(i,j,k) - molecular_viscosity) / constPr_sgs + &
                            molecular_diffusivity
           end do
         end do
       end do
       !$omp end parallel do

       call BoundViscosity(TDiff)
     end if

     if (enable_buoyancy) then
       call par_sync_out("  ...setting temperature in ghost cells.")
       call BoundTemperature(Temperature)
     end if

     if (enable_moisture) then
       call par_sync_out("  ...setting moisture in ghost cells.")
       call BoundMoisture(Moisture)
     end if

     if (num_of_scalars>0)  call par_sync_out("  ...setting scalars in ghost cells.")
     do i = 1, num_of_scalars
       call BoundScalar(Scalar(:,:,:,i))
     end do

     if (wallmodeltype>0) then
                    call par_sync_out("  ...computing wall model in scalar points.")
                    call ComputeViscsWM(U,V,W,Pr,Temperature,Moisture)
                    call par_sync_out("  ...computing wall model in velocity points.")
                    call ComputeUVWFluxesWM(U,V,W,Pr,Temperature,Moisture)
     end if

     call par_sync_out("  ...setting viscosity in ghost cells.")
     call BoundViscosity(Viscosity)

     call par_sync_out("  ...initializing fixed flow_rates.")
     call InitFlowRates(U, V, W)

#ifdef PAR
     if (enable_multiple_domains) then
       call par_sync_out("  ...sending initial conditions to children (if there are any).")

       call par_send_initial_conditions(U, V, W, Pr, Temperature, Moisture, Scalar)

       call par_sync_out("  ...sending boundary conditions to children (if there are any).")

       call par_exchange_domain_bounds(U, V, W, Temperature, Moisture, Scalar, time_stepping%time, epsilon(1._knd), receive=.false.)
     end if
#endif

    call par_sync_out("initial conditions set.")

  end subroutine InitialConditions






  subroutine InitBoundaryConditions
    use VTKFrames, only: InitVTKFrames
#ifdef PAR
    use Frames_ParallelIO, only: InitFrames_ParallelIO
#endif
    use SurfaceFrames, only: InitSurfaceFrames
    use custom_par
    
    real(knd), allocatable:: xU2(:), yV2(:), zW2(:)
    integer   :: i, j, k
    real(knd) :: dt
    integer   :: unit, io
    
    type(spline_coefs) :: geostrophic_wind
    
#ifdef CUSTOM_BOUNDARY_CONDITIONS
    interface
      subroutine CustomBoundaryConditions
      end subroutine
    end interface
#endif

    !Important to have some defined value before the first call to GetTurbInlet.
    !The value can be quite arbitrary.
    dt = min( min(dxmin,dymin,dzmin) / Uinlet, &
              min(dxmin / time_stepping%U_max(1), &
                  dymin / time_stepping%U_max(2), &
                  dzmin / time_stepping%U_max(3)))


    call par_sync_out("  ...computing grid coordinates.")


    allocate(xU2(-3:Prnx+3))
    allocate(yV2(-3:Prny+3))
    allocate(zW2(-3:Prnz+3))

    forall (i=-3:Prnx+3)
      xU2(i) = i * dxmin + im_xmin
    end forall
    
    forall (j=-3:Prny+3)
      yV2(j) = j * dymin + im_ymin
    end forall

    if (zgridfromfile) then

      zW2(0:Prnz) = gzW(offset_to_global_z:offset_to_global_z + Prnz)

      if (Btype(Bo)==BC_PERIODIC) then
        do j = -1, -3, -1
          zW2(j) = zW2(0) - (zW2(Prnz-1)-zW2(Prnz-1 + j))
        end do
      else if (Btype(Bo)>=BC_MPI_BOUNDS_MIN.and.Btype(Bo)<=BC_MPI_BOUNDS_MAX) then
        do j = -1, -3, -1
          zW2(j) = gzW(offset_to_global_z + j)
        end do
      else
        do j = -1, -3, -1
          zW2(j) = zW2(0) - (zW2(-j)-zW2(0))
        end do
      end if

      if (Btype(To)==BC_PERIODIC) then
        do j = Prnz+1, Prnz+3
          zW2(j) = zW2(Prnz-1) + (zW2(j-Prnz+1)-zW2(0))
        end do
      else if (Btype(To)>=BC_MPI_BOUNDS_MIN.and.Btype(Ea)<=BC_MPI_BOUNDS_MAX) then
        do j = Prnz+1, Prnz+3
          zW2(j) = gzW(offset_to_global_z + j)
        end do
      else
        do j = Prnz+1, Prnz+3
          zW2(j) = zW2(Prnz-1) + (zW2(Prnz-1)-zW2(Prnz-1 - (j-Prnz+1)))
        end do
      end if

      z0 = zW2(0)

    else

       forall (k=-3:Prnz+3)
         zW2(k) = k * dzmin + im_zmin
       end forall

    end if


    allocate(xU(-3:Prnx+3))
    allocate(yV(-3:Prny+3))
    allocate(zW(-3:Prnz+3))
    allocate(dxU(-2:Prnx+2))
    allocate(dyV(-2:Prny+2))
    allocate(dzW(-2:Prnz+2))
    allocate(xPr(-2:Prnx+3),dxPr(-2:Prnx+3))
    allocate(yPr(-2:Prny+3),dyPr(-2:Prny+3))
    allocate(zPr(-2:Prnz+3),dzPr(-2:Prnz+3))

    xU = xU2(-3:Prnx+3)
    yV = yV2(-3:Prny+3)
    zW = zW2(-3:Prnz+3)

    forall (i=-2:Prnx+3)
      xPr(i) = (xU(i-1)+xU(i))/2._knd
      dxPr(i) = xU(i)-xU(i-1)
    end forall

    forall (j=-2:Prny+3)
      yPr(j) = (yV(j-1)+yV(j))/2._knd
      dyPr(j) = yV(j)-yV(j-1)
    end forall

    forall (k=-2:Prnz+3)
      zPr(k) = (zW(k-1)+zW(k))/2._knd
      dzPr(k) = zW(k)-zW(k-1)
    end forall

    forall (i=-2:Prnx+2)
      dxU(i) = xPr(i+1)-xPr(i)
    end forall

    forall (j=-2:Prny+2)
      dyV(j) = yPr(j+1)-yPr(j)
    end forall

    forall (k=-2:Prnz+2)
      dzW(k) = zPr(k+1)-zPr(k)
    end forall
    
    if (zgridfromfile) then
            dzmin = minval(dzPr)
    end if


    deallocate(xU2)
    deallocate(yV2)
    deallocate(zW2)
  

    call par_sync_out("  ...creating grid cell type arrays.")


    allocate(Utype(-2:Unx+3,-2:Uny+3,-2:Unz+3))
    allocate(Vtype(-2:Vnx+3,-2:Vny+3,-2:Vnz+3))
    allocate(Wtype(-2:Wnx+3,-2:Wny+3,-2:Wnz+3))
    allocate(Prtype(-2:Prnx+3,-2:Prny+3,-2:Prnz+3))

    Utype = 0
    Vtype = 0
    Wtype = 0
    Prtype = 0


    call par_sync_out("  ...reading pressure gradient profile.")

    !Requires grid coordinates
    call get_pressure_gradient("pressure_gradient_profile.conf")
    
    call get_pressure_solution("pressure_solution.conf", pressure_solution)



    call par_sync_out("  ...reading geostrophic wind.")

    !Requires grid coordinates
    call get_geostrophic_wind("geostrophic_wind_profile.conf", geostrophic_wind)

    
    call par_sync_out("  ...reading forcing.conf")


    call get_forcing("forcing.conf")
    
    
    call par_sync_out("  ...getting inlet conditions.")

    if (default_turbulence_generator%direction==2) then
      allocate(Uin(-2:Unx+3,-2:Unz+3),Vin(-2:Vnx+3,-2:Vnz+3),Win(-2:Wnx+3,-2:Wnz+3))
    else
      allocate(Uin(-2:Uny+3,-2:Unz+3),Vin(-2:Vny+3,-2:Vnz+3),Win(-2:Wny+3,-2:Wnz+3))
    end if
    Uin = 0
    Vin = 0
    Win = 0

    if (enable_buoyancy) allocate(TempIn(-2:Prny+3,-2:Prnz+3))

    if (enable_moisture) allocate(MoistIn(-2:Prny+3,-2:Prnz+3))

    select case (inlettype)
      case (ZeroInletType)
        Uin = 0
        Vin = 0
        Win = 0
      case (ShearInletType)
        call ShearInlet(ShearInletTypeParameter)
      case (ParabolicInletType)
        call ParabolicInlet
      case (TurbulentInletType)
        call default_turbulence_generator%init()
        call default_turbulence_generator%step_xc(Uin, Vin, Win, time_stepping%dt)
      case (FromFileInletType)
        call GetInletFromFile(time_stepping%start_time)
      case (GeostrophicInletType)
        call GeostrophicWindInlet(geostrophic_wind)
      case default
        call ConstantInlet
    endselect


    if (enable_buoyancy) then
       
       call par_sync_out("  ...setting boundary temperature and temperature flux.")

       if (TempBtype(Bo)==BC_CONSTFLUX.or.TempBtype(Bo)==BC_DIRICHLET) then

         allocate(BsideTFlArr(-2:Prnx+3,-2:Prny+3))

         if (enable_radiation) then
           BsideTFlArr = 0
         else if (TempBtype(Bo)==BC_CONSTFLUX) then
           BsideTFlArr = sideTemp(Bo)
         else
           BsideTFlArr = 0
         end if

         if (TempBtype(Bo)==BC_DIRICHLET) then
           allocate(BsideTArr(-2:Prnx+3,-2:Prny+3))
           BsideTArr = sideTemp(Bo)
         end if

        end if
    end if

    if (.not.allocated(BsideTArr))  allocate(BsideTArr(0,0))
    if (.not.allocated(BsideTFlArr))  allocate(BsideTFlArr(0,0))


    if (enable_moisture) then

       call par_sync_out("  ...setting boundary moisture and moisture flux.")

       if (MoistBtype(Bo)==BC_CONSTFLUX.or.MoistBtype(Bo)==BC_DIRICHLET) then

         allocate(BsideMFlArr(-2:Prnx+3,-2:Prny+3))

         if (enable_radiation) then
           BsideMFlArr = 0
         else if (MoistBtype(Bo)==BC_CONSTFLUX) then
           BsideMFlArr = sideMoist(Bo)
         else
           BsideMFlArr = 0
         end if

         if (MoistBtype(Bo)==BC_DIRICHLET) then
           allocate(BsideMArr(-2:Prnx+3,-2:Prny+3))
           BsideMArr = sideMoist(Bo)
         end if

        end if
    end if

    if (.not.allocated(BsideMArr))  allocate(BsideMArr(0,0))
    if (.not.allocated(BsideMFlArr))  allocate(BsideMFlArr(0,0))



    call par_sync_out("  ...pressure correction.")
    call InitPressureCorrection

    call par_sync_out("  ...initializing subsidence profile.")
    call InitSubsidenceProfile

    call par_sync_out("  ...initializing loop tiles.")
    call InitTiles(Prnx,Prny,Prnz)

    if (enable_radiation) then
      call par_sync_out("  ...solar radiation.")
      call InitSolarRadiation
    end if

    call par_sync_out("  ...preparing solid bodies.")
    !read solid bodies in obstacles.conf
    call get_obstacles("obstacles.conf")
    !prepare the geometry of the solid bodies
    call InitSolidBodies

    call par_sync_out("  ...volume source bodies.")
    !prepare the geometry of the plant bodies
    call InitVolumeSourceBodies

    call par_sync_out("  ...preparing solid bodies boundary conditions.")
    !create actual immersed boundary and wall model points, i.m. points end up in an array
    call GetSolidBodiesBC

    call par_sync_out("  ...getting outside boundaries wall model points.")
    !add also the wall model points from domain boundaries to the list
    call GetOutsideBoundariesWM(num_of_scalars)

    call par_sync_out("  ...creating wall model points arrays.")
    !create arrays of w.m. points from the list
    call MoveWMPointsToArray

    call par_sync_out("  ...creating wall model mask arrays.")
    !creates masks for computation of diffusive fluxes
    call InitWMMasks

    if (enable_radiation) then
       call par_sync_out("  ...computing initial radiation balance.")
      !compute the radiation balance and prepare the wall fluxes 
      call InitIBPFluxes
      !set the immersed boundary values for fluxes
 !      call SetIBPFluxes
    end if

    call par_sync_out("  ...setting nullified points.")
    call SetNullifiedPoints

    call par_sync_out("  ...getting volume scalar sources.")
    !create actual arrays of the source points
    call InitVolumeSources

    call par_sync_out("  ...getting area scalar sources.")
    !add arrays of the line source points
    call InitAreaSources

    call par_sync_out("  ...getting line scalar sources.")
    !add arrays of the line source points
    call InitLineSources

    call par_sync_out("  ...getting point scalar sources.")
    !add arrays of the point source points
    call InitPointSources

    call par_sync_out("  ...getting puff scalar sources.")
    !add puff sources, each containing one or more points
    call InitPuffSources
    
    call par_sync_out("  ...setting scalar removal zones.")
    !add puff sources, each containing one or more points
    call InitScalarRemovalZones
    
    call par_sync_out("  ...creating output directories.")

    call CreateOutputDirectories
   
    !filter out frames outside the domain
    call par_sync_out("  ...preparing VTK frames.")
    call InitVTKFrames
#ifdef PAR    
    call par_sync_out("  ...preparing ParallelIO frames.")
    call InitFrames_ParallelIO
#endif    
    call par_sync_out("  ...preparing constant height frames.")
    call InitSurfaceFrames
   
#ifdef CUSTOM_BOUNDARY_CONDITIONS
    call par_sync_out("  ...executing custom boundary conditions.")
    call CustomBoundaryConditions
#endif

    call par_sync_out("boundary conditions set.")
    
  contains
  
    subroutine GeostrophicWindInlet(g)
      use Interpolation
      type(spline_coefs), intent(in) :: g
      real(knd) :: ug, vg
      integer :: j, k
      
      Win = 0
      
      j = 0
      do k = -2, Unz+3
        ug = linear_interpolation_eval(zPr(k), g%z, g%cu, j)
        Uin(:,k) = ug
      end do
      
      j = 0
      do k = -2, Vnz+3
        vg = linear_interpolation_eval(zPr(k), g%z, g%cv, j)
        Vin(:,k) = vg
      end do
    end subroutine
    
  end subroutine InitBoundaryConditions


  subroutine InitFlowRatesBC
#ifdef PAR
    use custom_par
#endif

    if (enable_fixed_flow_rate) then

      if (iim==nxims .and. Btype(Ea)==BC_PERIODIC) then
        flow_rate_x_fixed = .true.
      end if

#ifdef PAR        
      flow_rate_x_fixed = par_co_any(flow_rate_x_fixed)     
#endif


      if (jim==nyims .and. Btype(No)==BC_PERIODIC) then
        flow_rate_y_fixed = .true.
      end if

#ifdef PAR        
      flow_rate_y_fixed = par_co_any(flow_rate_y_fixed)     
#endif

      if (kim==nzims .and. Btype(To)==BC_PERIODIC) then
        flow_rate_z_fixed = .true.
      end if

#ifdef PAR        
      flow_rate_z_fixed = par_co_any(flow_rate_z_fixed)     
#endif

    end if

  end subroutine


  subroutine InitFlowRates(U, V, W)
#ifdef PAR
    use custom_par
#endif
    real(knd), intent(in), contiguous, dimension(-2:,-2:,-2:) :: U, V, W

    if (enable_fixed_flow_rate) then

      if (flow_rate_x_fixed) then
        if (set_U_bulk) then
          flow_rate_x = U_bulk_vec(1) * count(Utype(0,1:Uny,1:Unz)<=0) * dymin * dzmin
        else
          if (initcondsfromfile==0.and.inlettype==TurbulentInletType.and. &
              default_turbulence_generator%direction==1) then
            flow_rate_x = sum(default_turbulence_generator%Uinavg(1:Uny, 1:Unz)) &
                          * dymin * dzmin
          else
            flow_rate_x = sum(U(Unx,1:Uny, 1:Unz)) &
                          * dymin * dzmin
          end if
        end if
        
#ifdef PAR
        flow_rate_x = par_co_sum_plane_yz(flow_rate_x)
        call par_broadcast_from_last_x(flow_rate_x)
#endif
      end if


      if (flow_rate_y_fixed) then
        if (set_U_bulk) then
          flow_rate_y = U_bulk_vec(2) * count(Vtype(1:Vnx,0,1:Vnz)<=0) * dxmin * dzmin
        else
          if (initcondsfromfile==0.and.inlettype==TurbulentInletType.and. &
              default_turbulence_generator%direction==2) then
            !average V
            flow_rate_y = sum(default_turbulence_generator%Vinavg(1:Vny, 1:Vnz)) &
                          / (Vny*Vnz)
            flow_rate_y = flow_rate_y * Vnx * Vnz * dxmin * dzmin        
          else
            flow_rate_y = sum(V(1:Vnx, Vny, 1:Vnz)) * dxmin * dzmin
          end if
        end if

#ifdef PAR
        flow_rate_y = par_co_sum_plane_xz(flow_rate_y)
        call par_broadcast_from_last_y(flow_rate_y)
#endif
      end if
      

      if (flow_rate_z_fixed) then
        if (set_U_bulk) then
          flow_rate_z = U_bulk_vec(3) * count(Wtype(1:Wnx,1:Wny,0)<=0) * dxmin * dymin
        else
          flow_rate_z = sum(W(1:Wnx, 1:Wny, Wnz)) * dxmin * dymin
        end if

#ifdef PAR
        flow_rate_z = par_co_sum_plane_xy(flow_rate_z)
        call par_broadcast_from_last_z(flow_rate_z)
#endif
      end if

    end if
  end subroutine




  subroutine SetNullifiedPoints
    integer :: i,j,k,n
    
    !$omp parallel do reduction(+:nUnull)
    do k = 1, Unz
     do j = 1, Uny
      do i = 1, Unx        
       if (Utype(i,j,k)>0.and.Utype(i,j,k+1)>0.and.Utype(i,j,k-1)>0&
           .and.Utype(i,j-1,k)>0.and.Utype(i,j+1,k)>0&
           .and.Utype(i-1,j,k)>0.and.Utype(i+1,j,k)>0)  nUnull = nUnull+1
      end do
     end do
    end do
    !$omp end parallel do

    allocate(Unull(3,nUnull))

    n = 0

    do k = 1, Unz
     do j = 1, Uny
      do i = 1, Unx
       if (Utype(i,j,k)>0.and.Utype(i,j,k+1)>0.and.Utype(i,j,k-1)>0&
           .and.Utype(i,j-1,k)>0.and.Utype(i,j+1,k)>0&
           .and.Utype(i-1,j,k)>0.and.Utype(i+1,j,k)>0)  then

            n = n+1
            Unull(:,n) = [ i,j,k ]

       end if
      end do
     end do
    end do

    nVnull = 0

    !$omp parallel do reduction(+:nVnull)
    do k = 1, Vnz
     do j = 1, Vny
      do i = 1, Vnx
       if (Vtype(i,j,k)>0.and.Vtype(i,j,k+1)>0.and.Vtype(i,j,k-1)>0&
           .and.Vtype(i,j-1,k)>0.and.Vtype(i,j+1,k)>0&
           .and.Vtype(i-1,j,k)>0.and.Vtype(i+1,j,k)>0)  nVnull = nVnull+1
      end do
     end do
    end do
    !$omp end parallel do

    allocate(Vnull(3,nVnull))

    n = 0

    do k = 1, Vnz
     do j = 1, Vny
      do i = 1, Vnx
       if (Vtype(i,j,k)>0.and.Vtype(i,j,k+1)>0.and.Vtype(i,j,k-1)>0&
           .and.Vtype(i,j-1,k)>0.and.Vtype(i,j+1,k)>0&
           .and.Vtype(i-1,j,k)>0.and.Vtype(i+1,j,k)>0)  then

            n = n+1
            Vnull(:,n) = [ i,j,k ]

       end if
      end do
     end do
    end do

    nWnull = 0

    !$omp parallel do reduction(+:nWnull)
    do k = 1, Wnz
     do j = 1, Wny
      do i = 1, Wnx
       if (Wtype(i,j,k)>0.and.Wtype(i,j,k+1)>0.and.Wtype(i,j,k-1)>0&
           .and.Wtype(i,j-1,k)>0.and.Wtype(i,j+1,k)>0&
           .and.Wtype(i-1,j,k)>0.and.Wtype(i+1,j,k)>0)  nWnull = nWnull+1
      end do
     end do
    end do
    !$omp end parallel do

    allocate(Wnull(3,nWnull))

    n = 0


    do k = 1, Wnz
     do j = 1, Wny
      do i = 1, Wnx
       if (Wtype(i,j,k)>0.and.Wtype(i,j,k+1)>0.and.Wtype(i,j,k-1)>0&
           .and.Wtype(i,j-1,k)>0.and.Wtype(i,j+1,k)>0&
           .and.Wtype(i-1,j,k)>0.and.Wtype(i+1,j,k)>0)  then

            n = n+1
            Wnull(:,n) = [ i,j,k ]

       end if
      end do
     end do
    end do

  end subroutine SetNullifiedPoints



  subroutine get_io_dirs
    use strings, only: itoa
    integer :: l, stat

    call get_environment_variable(name="ELMM_OUTPUTDIR", length=l, status=stat)

    if (stat==0 .and. l>0 .and. l<=len(input_dir)) then
      call get_environment_variable(name="ELMM_OUTPUTDIR", value=output_dir)
    else if (stat==0 .and. l>len(input_dir)) then
      call error_stop("ELMM_OUTPUTDIR length exceeds variable length "//itoa(len(output_dir)))
    end if

    call get_environment_variable(name="ELMM_INPUTDIR")
    if (stat==0 .and. l>0 .and. l<=len(input_dir)) then
      call get_environment_variable(name="ELMM_INPUTDIR", value=input_dir)
    else if (stat==0 .and. l>len(input_dir)) then
      call error_stop("ELMM_INPUTDIR length exceeds variable length "//itoa(len(input_dir)))
    end if
  end subroutine


  subroutine init_random_seed()
    use rng_par_zig
    !$ use omp_lib
#ifdef PAR
    use custom_par
#endif
    integer :: i
    integer(int64) :: seed(2)
    integer :: nt
    
    ! Just some initial entropy for a reproducible random sequence
    integer(int64), parameter :: base(2) = [int(Z'1DADBEEFBAADD0D0', int64), &
                                            int(Z'5BADD0D0DEADBEEF', int64)]
    
    nt = 1
    !$omp parallel
    !$omp single
    !$ nt = omp_get_num_threads()
    !$omp end single
    !$omp end parallel

    seed = base
    
#ifdef PAR
    ! Jump nthread-times for each image so that each image
    ! has a disjunct set of random sequences for its threads.
    call rng_jump(seed, nt*(myim-1))
#endif

    call rng_init(nt, seed)

  end subroutine


endmodule Initial
