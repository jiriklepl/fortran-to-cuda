
  logical function ResidentRKSupported() result(supported)
    use Subgrid, only: sgstype, SigmaModel
    use Filters, only: filtertype
    use ImmersedBoundary, only: ScalFlIBPoints
    use BuoyantGases, only: enable_buoyant_scalars
    supported = gridtype==GRID_UNIFORM .and. explicit_diffusion .and. discretization_order==2
    supported = supported .and. sgstype==SigmaModel .and. (filtertype==0 .or. filtertype==1)
    supported = supported .and. (advection_method==2 .or. advection_method==4 .or. advection_method==5 .or. advection_method==6)
    supported = supported .and. min(Unx,Uny,Unz,Vnx,Vny,Vnz,Wnx,Wny,Wnz)>=3
    supported = supported .and. all(Btype([We,Ea,So,No])==BC_PERIODIC) .and. all(Btype([Bo,To])==BC_NOSLIP)
    supported = supported .and. .not. enable_buoyancy .and. .not. enable_moisture .and. .not. enable_liquid
    supported = supported .and. num_of_scalars==0 .and. abs(Coriolis_parameter)<=tiny(1._knd)
    ! Buoyant gases change W on the host even when no scalar is transported.
    supported = supported .and. .not. enable_buoyant_scalars
    supported = supported .and. .not. enable_top_sponge .and. .not. enable_out_sponge_x .and. .not. enable_out_sponge_y
    supported = supported .and. .not. enable_fixed_flow_rate
    supported = supported .and. debuglevel==0 .and. steady/=1 .and. size(ScalFlIBPoints)==0
    supported = supported .and. .not. any(pressure_solution%correct_mass_flux)
  end function
