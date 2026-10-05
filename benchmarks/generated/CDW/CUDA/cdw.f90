module MomentumAdvection
  use iso_c_binding, only: &
    fort_internal_c_integer => c_int, &
    fort_internal_c_real => c_double, &
    fort_internal_c_real32 => c_float, &
    fort_internal_c_extent => c_size_t, &
    fort_internal_c_logical => c_bool, &
    fort_internal_c_token => c_int64_t
  implicit none
  private
  public :: CDW, knd, start_hot, finish_hot
  integer, parameter :: knd = fort_internal_c_real

  public :: cdw_workspace
  public :: cdw_create
  public :: cdw_run
  public :: cdw_update_device
  public :: cdw_update_host
  public :: cdw_destroy
  type :: cdw_workspace
    private
    integer(fort_internal_c_token) :: token = 0
  end type cdw_workspace

  interface
    function fort_internal_create( &
        fort_v0_w2, &
        fort_v0_w2_dim1, &
        fort_v0_w2_dim2, &
        fort_v0_w2_dim3, &
        fort_v1_u, &
        fort_v1_u_dim1, &
        fort_v1_u_dim2, &
        fort_v1_u_dim3, &
        fort_v2_v, &
        fort_v2_v_dim1, &
        fort_v2_v_dim2, &
        fort_v2_v_dim3, &
        fort_v3_w, &
        fort_v3_w_dim1, &
        fort_v3_w_dim2, &
        fort_v3_w_dim3 &
    ) bind(C, name='cpp_cdw_create') result(fort_internal_result)
      import :: fort_internal_c_integer
      import :: fort_internal_c_real
      import :: fort_internal_c_real32
      import :: fort_internal_c_extent
      import :: fort_internal_c_logical
      import :: fort_internal_c_token
      integer(fort_internal_c_token) :: fort_internal_result
      real(fort_internal_c_real), intent(in) :: fort_v0_w2(*)
      integer(fort_internal_c_extent), value, intent(in) :: fort_v0_w2_dim1
      integer(fort_internal_c_extent), value, intent(in) :: fort_v0_w2_dim2
      integer(fort_internal_c_extent), value, intent(in) :: fort_v0_w2_dim3
      real(fort_internal_c_real), intent(in) :: fort_v1_u(*)
      integer(fort_internal_c_extent), value, intent(in) :: fort_v1_u_dim1
      integer(fort_internal_c_extent), value, intent(in) :: fort_v1_u_dim2
      integer(fort_internal_c_extent), value, intent(in) :: fort_v1_u_dim3
      real(fort_internal_c_real), intent(in) :: fort_v2_v(*)
      integer(fort_internal_c_extent), value, intent(in) :: fort_v2_v_dim1
      integer(fort_internal_c_extent), value, intent(in) :: fort_v2_v_dim2
      integer(fort_internal_c_extent), value, intent(in) :: fort_v2_v_dim3
      real(fort_internal_c_real), intent(in) :: fort_v3_w(*)
      integer(fort_internal_c_extent), value, intent(in) :: fort_v3_w_dim1
      integer(fort_internal_c_extent), value, intent(in) :: fort_v3_w_dim2
      integer(fort_internal_c_extent), value, intent(in) :: fort_v3_w_dim3
    end function fort_internal_create
    subroutine fort_internal_run( &
        fort_internal_token, &
        fort_v4_dxmin, &
        fort_v5_dymin, &
        fort_v6_dzmin, &
        fort_v7_wnx, &
        fort_v8_wny, &
        fort_v9_wnz &
    ) bind(C, name='cpp_cdw_run')
      import :: fort_internal_c_integer
      import :: fort_internal_c_real
      import :: fort_internal_c_real32
      import :: fort_internal_c_extent
      import :: fort_internal_c_logical
      import :: fort_internal_c_token
      integer(fort_internal_c_token), value, intent(in) :: fort_internal_token
      real(fort_internal_c_real), value, intent(in) :: fort_v4_dxmin
      real(fort_internal_c_real), value, intent(in) :: fort_v5_dymin
      real(fort_internal_c_real), value, intent(in) :: fort_v6_dzmin
      integer(fort_internal_c_integer), value, intent(in) :: fort_v7_wnx
      integer(fort_internal_c_integer), value, intent(in) :: fort_v8_wny
      integer(fort_internal_c_integer), value, intent(in) :: fort_v9_wnz
    end subroutine fort_internal_run
    subroutine fort_internal_destroy( &
        fort_internal_token &
    ) bind(C, name='cpp_cdw_destroy')
      import :: fort_internal_c_integer
      import :: fort_internal_c_real
      import :: fort_internal_c_real32
      import :: fort_internal_c_extent
      import :: fort_internal_c_logical
      import :: fort_internal_c_token
      integer(fort_internal_c_token), value, intent(in) :: fort_internal_token
    end subroutine fort_internal_destroy
    subroutine fort_internal_validate( &
        fort_internal_token &
    ) bind(C, name='cpp_cdw_workspace_validate')
      import :: fort_internal_c_integer
      import :: fort_internal_c_real
      import :: fort_internal_c_real32
      import :: fort_internal_c_extent
      import :: fort_internal_c_logical
      import :: fort_internal_c_token
      integer(fort_internal_c_token), value, intent(in) :: fort_internal_token
    end subroutine fort_internal_validate
    subroutine fort_internal_update_device_0( &
        fort_internal_token, &
        fort_v0_w2, &
        fort_v0_w2_dim1, &
        fort_v0_w2_dim2, &
        fort_v0_w2_dim3 &
    ) bind(C, name='cpp_cdw_update_device_0')
      import :: fort_internal_c_integer
      import :: fort_internal_c_real
      import :: fort_internal_c_real32
      import :: fort_internal_c_extent
      import :: fort_internal_c_logical
      import :: fort_internal_c_token
      integer(fort_internal_c_token), value, intent(in) :: fort_internal_token
      real(fort_internal_c_real), intent(in) :: fort_v0_w2(*)
      integer(fort_internal_c_extent), value, intent(in) :: fort_v0_w2_dim1
      integer(fort_internal_c_extent), value, intent(in) :: fort_v0_w2_dim2
      integer(fort_internal_c_extent), value, intent(in) :: fort_v0_w2_dim3
    end subroutine fort_internal_update_device_0
    subroutine fort_internal_update_device_1( &
        fort_internal_token, &
        fort_v1_u, &
        fort_v1_u_dim1, &
        fort_v1_u_dim2, &
        fort_v1_u_dim3 &
    ) bind(C, name='cpp_cdw_update_device_1')
      import :: fort_internal_c_integer
      import :: fort_internal_c_real
      import :: fort_internal_c_real32
      import :: fort_internal_c_extent
      import :: fort_internal_c_logical
      import :: fort_internal_c_token
      integer(fort_internal_c_token), value, intent(in) :: fort_internal_token
      real(fort_internal_c_real), intent(in) :: fort_v1_u(*)
      integer(fort_internal_c_extent), value, intent(in) :: fort_v1_u_dim1
      integer(fort_internal_c_extent), value, intent(in) :: fort_v1_u_dim2
      integer(fort_internal_c_extent), value, intent(in) :: fort_v1_u_dim3
    end subroutine fort_internal_update_device_1
    subroutine fort_internal_update_device_2( &
        fort_internal_token, &
        fort_v2_v, &
        fort_v2_v_dim1, &
        fort_v2_v_dim2, &
        fort_v2_v_dim3 &
    ) bind(C, name='cpp_cdw_update_device_2')
      import :: fort_internal_c_integer
      import :: fort_internal_c_real
      import :: fort_internal_c_real32
      import :: fort_internal_c_extent
      import :: fort_internal_c_logical
      import :: fort_internal_c_token
      integer(fort_internal_c_token), value, intent(in) :: fort_internal_token
      real(fort_internal_c_real), intent(in) :: fort_v2_v(*)
      integer(fort_internal_c_extent), value, intent(in) :: fort_v2_v_dim1
      integer(fort_internal_c_extent), value, intent(in) :: fort_v2_v_dim2
      integer(fort_internal_c_extent), value, intent(in) :: fort_v2_v_dim3
    end subroutine fort_internal_update_device_2
    subroutine fort_internal_update_device_3( &
        fort_internal_token, &
        fort_v3_w, &
        fort_v3_w_dim1, &
        fort_v3_w_dim2, &
        fort_v3_w_dim3 &
    ) bind(C, name='cpp_cdw_update_device_3')
      import :: fort_internal_c_integer
      import :: fort_internal_c_real
      import :: fort_internal_c_real32
      import :: fort_internal_c_extent
      import :: fort_internal_c_logical
      import :: fort_internal_c_token
      integer(fort_internal_c_token), value, intent(in) :: fort_internal_token
      real(fort_internal_c_real), intent(in) :: fort_v3_w(*)
      integer(fort_internal_c_extent), value, intent(in) :: fort_v3_w_dim1
      integer(fort_internal_c_extent), value, intent(in) :: fort_v3_w_dim2
      integer(fort_internal_c_extent), value, intent(in) :: fort_v3_w_dim3
    end subroutine fort_internal_update_device_3
    subroutine fort_internal_update_host_0( &
        fort_internal_token, &
        fort_v0_w2, &
        fort_v0_w2_dim1, &
        fort_v0_w2_dim2, &
        fort_v0_w2_dim3 &
    ) bind(C, name='cpp_cdw_update_host_0')
      import :: fort_internal_c_integer
      import :: fort_internal_c_real
      import :: fort_internal_c_real32
      import :: fort_internal_c_extent
      import :: fort_internal_c_logical
      import :: fort_internal_c_token
      integer(fort_internal_c_token), value, intent(in) :: fort_internal_token
      real(fort_internal_c_real), intent(out) :: fort_v0_w2(*)
      integer(fort_internal_c_extent), value, intent(in) :: fort_v0_w2_dim1
      integer(fort_internal_c_extent), value, intent(in) :: fort_v0_w2_dim2
      integer(fort_internal_c_extent), value, intent(in) :: fort_v0_w2_dim3
    end subroutine fort_internal_update_host_0
    subroutine fort_internal_update_host_1( &
        fort_internal_token, &
        fort_v1_u, &
        fort_v1_u_dim1, &
        fort_v1_u_dim2, &
        fort_v1_u_dim3 &
    ) bind(C, name='cpp_cdw_update_host_1')
      import :: fort_internal_c_integer
      import :: fort_internal_c_real
      import :: fort_internal_c_real32
      import :: fort_internal_c_extent
      import :: fort_internal_c_logical
      import :: fort_internal_c_token
      integer(fort_internal_c_token), value, intent(in) :: fort_internal_token
      real(fort_internal_c_real), intent(out) :: fort_v1_u(*)
      integer(fort_internal_c_extent), value, intent(in) :: fort_v1_u_dim1
      integer(fort_internal_c_extent), value, intent(in) :: fort_v1_u_dim2
      integer(fort_internal_c_extent), value, intent(in) :: fort_v1_u_dim3
    end subroutine fort_internal_update_host_1
    subroutine fort_internal_update_host_2( &
        fort_internal_token, &
        fort_v2_v, &
        fort_v2_v_dim1, &
        fort_v2_v_dim2, &
        fort_v2_v_dim3 &
    ) bind(C, name='cpp_cdw_update_host_2')
      import :: fort_internal_c_integer
      import :: fort_internal_c_real
      import :: fort_internal_c_real32
      import :: fort_internal_c_extent
      import :: fort_internal_c_logical
      import :: fort_internal_c_token
      integer(fort_internal_c_token), value, intent(in) :: fort_internal_token
      real(fort_internal_c_real), intent(out) :: fort_v2_v(*)
      integer(fort_internal_c_extent), value, intent(in) :: fort_v2_v_dim1
      integer(fort_internal_c_extent), value, intent(in) :: fort_v2_v_dim2
      integer(fort_internal_c_extent), value, intent(in) :: fort_v2_v_dim3
    end subroutine fort_internal_update_host_2
    subroutine fort_internal_update_host_3( &
        fort_internal_token, &
        fort_v3_w, &
        fort_v3_w_dim1, &
        fort_v3_w_dim2, &
        fort_v3_w_dim3 &
    ) bind(C, name='cpp_cdw_update_host_3')
      import :: fort_internal_c_integer
      import :: fort_internal_c_real
      import :: fort_internal_c_real32
      import :: fort_internal_c_extent
      import :: fort_internal_c_logical
      import :: fort_internal_c_token
      integer(fort_internal_c_token), value, intent(in) :: fort_internal_token
      real(fort_internal_c_real), intent(out) :: fort_v3_w(*)
      integer(fort_internal_c_extent), value, intent(in) :: fort_v3_w_dim1
      integer(fort_internal_c_extent), value, intent(in) :: fort_v3_w_dim2
      integer(fort_internal_c_extent), value, intent(in) :: fort_v3_w_dim3
    end subroutine fort_internal_update_host_3
    subroutine fort_internal_c_entry( &
        fort_v0_w2, &
        fort_v0_w2_dim1, &
        fort_v0_w2_dim2, &
        fort_v0_w2_dim3, &
        fort_v1_u, &
        fort_v1_u_dim1, &
        fort_v1_u_dim2, &
        fort_v1_u_dim3, &
        fort_v2_v, &
        fort_v2_v_dim1, &
        fort_v2_v_dim2, &
        fort_v2_v_dim3, &
        fort_v3_w, &
        fort_v3_w_dim1, &
        fort_v3_w_dim2, &
        fort_v3_w_dim3, &
        fort_v4_dxmin, &
        fort_v5_dymin, &
        fort_v6_dzmin, &
        fort_v7_wnx, &
        fort_v8_wny, &
        fort_v9_wnz &
    ) bind(C, name='cpp_CDW')
      import :: fort_internal_c_integer
      import :: fort_internal_c_real
      import :: fort_internal_c_real32
      import :: fort_internal_c_extent
      import :: fort_internal_c_logical
      real(fort_internal_c_real), intent(out) :: fort_v0_w2(*)
      integer(fort_internal_c_extent), value, intent(in) :: fort_v0_w2_dim1
      integer(fort_internal_c_extent), value, intent(in) :: fort_v0_w2_dim2
      integer(fort_internal_c_extent), value, intent(in) :: fort_v0_w2_dim3
      real(fort_internal_c_real), intent(in) :: fort_v1_u(*)
      integer(fort_internal_c_extent), value, intent(in) :: fort_v1_u_dim1
      integer(fort_internal_c_extent), value, intent(in) :: fort_v1_u_dim2
      integer(fort_internal_c_extent), value, intent(in) :: fort_v1_u_dim3
      real(fort_internal_c_real), intent(in) :: fort_v2_v(*)
      integer(fort_internal_c_extent), value, intent(in) :: fort_v2_v_dim1
      integer(fort_internal_c_extent), value, intent(in) :: fort_v2_v_dim2
      integer(fort_internal_c_extent), value, intent(in) :: fort_v2_v_dim3
      real(fort_internal_c_real), intent(in) :: fort_v3_w(*)
      integer(fort_internal_c_extent), value, intent(in) :: fort_v3_w_dim1
      integer(fort_internal_c_extent), value, intent(in) :: fort_v3_w_dim2
      integer(fort_internal_c_extent), value, intent(in) :: fort_v3_w_dim3
      real(fort_internal_c_real), value, intent(in) :: fort_v4_dxmin
      real(fort_internal_c_real), value, intent(in) :: fort_v5_dymin
      real(fort_internal_c_real), value, intent(in) :: fort_v6_dzmin
      integer(fort_internal_c_integer), value, intent(in) :: fort_v7_wnx
      integer(fort_internal_c_integer), value, intent(in) :: fort_v8_wny
      integer(fort_internal_c_integer), value, intent(in) :: fort_v9_wnz
    end subroutine fort_internal_c_entry
    subroutine fort_internal_c_start() bind(C, name='cpp_start_hot')
    end subroutine fort_internal_c_start
    subroutine fort_internal_c_finish() bind(C, name='cpp_finish_hot')
    end subroutine fort_internal_c_finish
  end interface

contains

  subroutine CDW( &
      w2, &
      u, &
      v, &
      w, &
      dxmin, &
      dymin, &
      dzmin, &
      wnx, &
      wny, &
      wnz &
  )
    real(knd), contiguous, intent(out) :: w2(:, :, :)
    real(knd), contiguous, intent(in) :: u(:, :, :)
    real(knd), contiguous, intent(in) :: v(:, :, :)
    real(knd), contiguous, intent(in) :: w(:, :, :)
    real(knd), intent(in) :: dxmin
    real(knd), intent(in) :: dymin
    real(knd), intent(in) :: dzmin
    integer, intent(in) :: wnx
    integer, intent(in) :: wny
    integer, intent(in) :: wnz
    call fort_internal_bridge( &
        w2, &
        u, &
        v, &
        w, &
        dxmin, &
        dymin, &
        dzmin, &
        wnx, &
        wny, &
        wnz &
    )
  end subroutine CDW

  subroutine fort_internal_bridge( &
      fort_v0_w2, &
      fort_v1_u, &
      fort_v2_v, &
      fort_v3_w, &
      fort_v4_dxmin, &
      fort_v5_dymin, &
      fort_v6_dzmin, &
      fort_v7_wnx, &
      fort_v8_wny, &
      fort_v9_wnz &
  )
    intrinsic :: size, int, real, logical
    real(knd), contiguous, intent(out) :: fort_v0_w2(:, :, :)
    real(knd), contiguous, intent(in) :: fort_v1_u(:, :, :)
    real(knd), contiguous, intent(in) :: fort_v2_v(:, :, :)
    real(knd), contiguous, intent(in) :: fort_v3_w(:, :, :)
    real(knd), intent(in) :: fort_v4_dxmin
    real(knd), intent(in) :: fort_v5_dymin
    real(knd), intent(in) :: fort_v6_dzmin
    integer, intent(in) :: fort_v7_wnx
    integer, intent(in) :: fort_v8_wny
    integer, intent(in) :: fort_v9_wnz
    call fort_internal_c_entry( &
        fort_v0_w2, &
        size(fort_v0_w2, 1, kind=fort_internal_c_extent), &
        size(fort_v0_w2, 2, kind=fort_internal_c_extent), &
        size(fort_v0_w2, 3, kind=fort_internal_c_extent), &
        fort_v1_u, &
        size(fort_v1_u, 1, kind=fort_internal_c_extent), &
        size(fort_v1_u, 2, kind=fort_internal_c_extent), &
        size(fort_v1_u, 3, kind=fort_internal_c_extent), &
        fort_v2_v, &
        size(fort_v2_v, 1, kind=fort_internal_c_extent), &
        size(fort_v2_v, 2, kind=fort_internal_c_extent), &
        size(fort_v2_v, 3, kind=fort_internal_c_extent), &
        fort_v3_w, &
        size(fort_v3_w, 1, kind=fort_internal_c_extent), &
        size(fort_v3_w, 2, kind=fort_internal_c_extent), &
        size(fort_v3_w, 3, kind=fort_internal_c_extent), &
        real(fort_v4_dxmin, kind=fort_internal_c_real), &
        real(fort_v5_dymin, kind=fort_internal_c_real), &
        real(fort_v6_dzmin, kind=fort_internal_c_real), &
        int(fort_v7_wnx, kind=fort_internal_c_integer), &
        int(fort_v8_wny, kind=fort_internal_c_integer), &
        int(fort_v9_wnz, kind=fort_internal_c_integer) &
    )
  end subroutine fort_internal_bridge

  subroutine start_hot()
    call fort_internal_c_start()
  end subroutine start_hot

  subroutine finish_hot()
    call fort_internal_c_finish()
  end subroutine finish_hot

  subroutine cdw_create( &
      work, &
      w2, &
      u, &
      v, &
      w &
  )
    type(cdw_workspace), intent(inout) :: work
    real(knd), contiguous, intent(in) :: w2(:, :, :)
    real(knd), contiguous, intent(in) :: u(:, :, :)
    real(knd), contiguous, intent(in) :: v(:, :, :)
    real(knd), contiguous, intent(in) :: w(:, :, :)
    call fort_internal_create_bridge( &
        work, &
        w2, &
        u, &
        v, &
        w &
    )
  end subroutine cdw_create

  subroutine fort_internal_create_bridge( &
      work, &
      fort_v0_w2, &
      fort_v1_u, &
      fort_v2_v, &
      fort_v3_w &
  )
    intrinsic :: size, int, real, logical, present
    type(cdw_workspace), intent(inout) :: work
    real(knd), contiguous, intent(in) :: fort_v0_w2(:, :, :)
    real(knd), contiguous, intent(in) :: fort_v1_u(:, :, :)
    real(knd), contiguous, intent(in) :: fort_v2_v(:, :, :)
    real(knd), contiguous, intent(in) :: fort_v3_w(:, :, :)
    if (work%token /= 0) error stop 'workspace already initialized'
    work%token = fort_internal_create( &
        fort_v0_w2, &
        size(fort_v0_w2, 1, kind=fort_internal_c_extent), &
        size(fort_v0_w2, 2, kind=fort_internal_c_extent), &
        size(fort_v0_w2, 3, kind=fort_internal_c_extent), &
        fort_v1_u, &
        size(fort_v1_u, 1, kind=fort_internal_c_extent), &
        size(fort_v1_u, 2, kind=fort_internal_c_extent), &
        size(fort_v1_u, 3, kind=fort_internal_c_extent), &
        fort_v2_v, &
        size(fort_v2_v, 1, kind=fort_internal_c_extent), &
        size(fort_v2_v, 2, kind=fort_internal_c_extent), &
        size(fort_v2_v, 3, kind=fort_internal_c_extent), &
        fort_v3_w, &
        size(fort_v3_w, 1, kind=fort_internal_c_extent), &
        size(fort_v3_w, 2, kind=fort_internal_c_extent), &
        size(fort_v3_w, 3, kind=fort_internal_c_extent) &
    )
  end subroutine fort_internal_create_bridge

  subroutine cdw_run( &
      work, &
      dxmin, &
      dymin, &
      dzmin, &
      wnx, &
      wny, &
      wnz &
  )
    type(cdw_workspace), intent(in) :: work
    real(knd), intent(in) :: dxmin
    real(knd), intent(in) :: dymin
    real(knd), intent(in) :: dzmin
    integer, intent(in) :: wnx
    integer, intent(in) :: wny
    integer, intent(in) :: wnz
    call fort_internal_run_bridge( &
        work, &
        dxmin, &
        dymin, &
        dzmin, &
        wnx, &
        wny, &
        wnz &
    )
  end subroutine cdw_run

  subroutine fort_internal_run_bridge( &
      work, &
      fort_v4_dxmin, &
      fort_v5_dymin, &
      fort_v6_dzmin, &
      fort_v7_wnx, &
      fort_v8_wny, &
      fort_v9_wnz &
  )
    intrinsic :: size, int, real, logical, present
    type(cdw_workspace), intent(in) :: work
    real(knd), intent(in) :: fort_v4_dxmin
    real(knd), intent(in) :: fort_v5_dymin
    real(knd), intent(in) :: fort_v6_dzmin
    integer, intent(in) :: fort_v7_wnx
    integer, intent(in) :: fort_v8_wny
    integer, intent(in) :: fort_v9_wnz
    call fort_internal_run( &
        work%token, &
        real(fort_v4_dxmin, kind=fort_internal_c_real), &
        real(fort_v5_dymin, kind=fort_internal_c_real), &
        real(fort_v6_dzmin, kind=fort_internal_c_real), &
        int(fort_v7_wnx, kind=fort_internal_c_integer), &
        int(fort_v8_wny, kind=fort_internal_c_integer), &
        int(fort_v9_wnz, kind=fort_internal_c_integer) &
    )
  end subroutine fort_internal_run_bridge

  subroutine cdw_update_device( &
      work, &
      w2, &
      u, &
      v, &
      w &
  )
    type(cdw_workspace), intent(in) :: work
    real(knd), contiguous, intent(in), optional :: w2(:, :, :)
    real(knd), contiguous, intent(in), optional :: u(:, :, :)
    real(knd), contiguous, intent(in), optional :: v(:, :, :)
    real(knd), contiguous, intent(in), optional :: w(:, :, :)
    call fort_internal_update_device_bridge( &
        work, &
        w2, &
        u, &
        v, &
        w &
    )
  end subroutine cdw_update_device

  subroutine fort_internal_update_device_bridge( &
      work, &
      fort_v0_w2, &
      fort_v1_u, &
      fort_v2_v, &
      fort_v3_w &
  )
    intrinsic :: size, int, real, logical, present
    type(cdw_workspace), intent(in) :: work
    real(knd), contiguous, intent(in), optional :: fort_v0_w2(:, :, :)
    real(knd), contiguous, intent(in), optional :: fort_v1_u(:, :, :)
    real(knd), contiguous, intent(in), optional :: fort_v2_v(:, :, :)
    real(knd), contiguous, intent(in), optional :: fort_v3_w(:, :, :)
    call fort_internal_validate(work%token)
    if (present(fort_v0_w2)) then
      call fort_internal_update_device_0( &
          work%token, &
          fort_v0_w2, &
          size(fort_v0_w2, 1, kind=fort_internal_c_extent), &
          size(fort_v0_w2, 2, kind=fort_internal_c_extent), &
          size(fort_v0_w2, 3, kind=fort_internal_c_extent) &
      )
    end if
    if (present(fort_v1_u)) then
      call fort_internal_update_device_1( &
          work%token, &
          fort_v1_u, &
          size(fort_v1_u, 1, kind=fort_internal_c_extent), &
          size(fort_v1_u, 2, kind=fort_internal_c_extent), &
          size(fort_v1_u, 3, kind=fort_internal_c_extent) &
      )
    end if
    if (present(fort_v2_v)) then
      call fort_internal_update_device_2( &
          work%token, &
          fort_v2_v, &
          size(fort_v2_v, 1, kind=fort_internal_c_extent), &
          size(fort_v2_v, 2, kind=fort_internal_c_extent), &
          size(fort_v2_v, 3, kind=fort_internal_c_extent) &
      )
    end if
    if (present(fort_v3_w)) then
      call fort_internal_update_device_3( &
          work%token, &
          fort_v3_w, &
          size(fort_v3_w, 1, kind=fort_internal_c_extent), &
          size(fort_v3_w, 2, kind=fort_internal_c_extent), &
          size(fort_v3_w, 3, kind=fort_internal_c_extent) &
      )
    end if
  end subroutine fort_internal_update_device_bridge

  subroutine cdw_update_host( &
      work, &
      w2, &
      u, &
      v, &
      w &
  )
    type(cdw_workspace), intent(in) :: work
    real(knd), contiguous, intent(out), optional :: w2(:, :, :)
    real(knd), contiguous, intent(out), optional :: u(:, :, :)
    real(knd), contiguous, intent(out), optional :: v(:, :, :)
    real(knd), contiguous, intent(out), optional :: w(:, :, :)
    call fort_internal_update_host_bridge( &
        work, &
        w2, &
        u, &
        v, &
        w &
    )
  end subroutine cdw_update_host

  subroutine fort_internal_update_host_bridge( &
      work, &
      fort_v0_w2, &
      fort_v1_u, &
      fort_v2_v, &
      fort_v3_w &
  )
    intrinsic :: size, int, real, logical, present
    type(cdw_workspace), intent(in) :: work
    real(knd), contiguous, intent(out), optional :: fort_v0_w2(:, :, :)
    real(knd), contiguous, intent(out), optional :: fort_v1_u(:, :, :)
    real(knd), contiguous, intent(out), optional :: fort_v2_v(:, :, :)
    real(knd), contiguous, intent(out), optional :: fort_v3_w(:, :, :)
    call fort_internal_validate(work%token)
    if (present(fort_v0_w2)) then
      call fort_internal_update_host_0( &
          work%token, &
          fort_v0_w2, &
          size(fort_v0_w2, 1, kind=fort_internal_c_extent), &
          size(fort_v0_w2, 2, kind=fort_internal_c_extent), &
          size(fort_v0_w2, 3, kind=fort_internal_c_extent) &
      )
    end if
    if (present(fort_v1_u)) then
      call fort_internal_update_host_1( &
          work%token, &
          fort_v1_u, &
          size(fort_v1_u, 1, kind=fort_internal_c_extent), &
          size(fort_v1_u, 2, kind=fort_internal_c_extent), &
          size(fort_v1_u, 3, kind=fort_internal_c_extent) &
      )
    end if
    if (present(fort_v2_v)) then
      call fort_internal_update_host_2( &
          work%token, &
          fort_v2_v, &
          size(fort_v2_v, 1, kind=fort_internal_c_extent), &
          size(fort_v2_v, 2, kind=fort_internal_c_extent), &
          size(fort_v2_v, 3, kind=fort_internal_c_extent) &
      )
    end if
    if (present(fort_v3_w)) then
      call fort_internal_update_host_3( &
          work%token, &
          fort_v3_w, &
          size(fort_v3_w, 1, kind=fort_internal_c_extent), &
          size(fort_v3_w, 2, kind=fort_internal_c_extent), &
          size(fort_v3_w, 3, kind=fort_internal_c_extent) &
      )
    end if
  end subroutine fort_internal_update_host_bridge

  subroutine cdw_destroy( &
      work &
  )
    type(cdw_workspace), intent(inout) :: work
    call fort_internal_destroy(work%token)
    work%token = 0
  end subroutine cdw_destroy

end module MomentumAdvection
