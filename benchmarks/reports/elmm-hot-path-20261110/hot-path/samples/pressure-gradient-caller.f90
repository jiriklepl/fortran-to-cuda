    block
      use, intrinsic :: iso_c_binding
      use fort_scoped_memory, only: FORT_SCOPE_GPU
      use elmm_resident_pressure_state, only: resident_flow, pressure_require
      use fort_shared_80aa32c5945f, only: rk_device_run => run
      integer(c_int64_t) :: rk_handle_u
      if (resident_flow%active) then
        rk_handle_u=resident_flow%bind_real64(u,lbound(u),.true.)
        call pressure_require(rk_device_run( &
          resident_flow%context, &
          FORT_SCOPE_GPU, &
          rk_handle_u, &
          lbound(u,1), &
          lbound(u,2), &
          lbound(u,3), &
          a, &
          pr_gradient_x, &
          unx, &
          uny, &
          unz))
      else
      do k = 1, Unz
        do j = 1, Uny
          do i = 1, Unx
            U(i,j,k) = U(i,j,k) + A * pr_gradient_x
          end do
        end do
      end do
      endif
    end block
