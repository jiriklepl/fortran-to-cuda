module MomentumAdvection
  use iso_c_binding, only: c_double, c_int, c_size_t, c_ptr, c_loc
  implicit none
  private
  public CDU, knd, start_hot, finish_hot

  integer, parameter :: knd = c_double

  interface
    subroutine cpp_CDU( &
            u, u_dim1, u_dim2, u_dim3, &
            u2, u2_dim1, u2_dim2, u2_dim3, &
            unx, &
            uny, &
            unz, &
            v, v_dim1, v_dim2, v_dim3, &
            w, w_dim1, w_dim2, w_dim3, &
            dxmin, &
            dymin, &
            dzmin &
        ) bind(C, name='cpp_CDU')
      import :: c_double, c_int, c_size_t, c_ptr, knd
        real(kind = knd), dimension(*) :: u
        integer(c_size_t), value, intent(in) :: u_dim1
        integer(c_size_t), value, intent(in) :: u_dim2
        integer(c_size_t), value, intent(in) :: u_dim3
        real(kind = knd), dimension(*) :: u2
        integer(c_size_t), value, intent(in) :: u2_dim1
        integer(c_size_t), value, intent(in) :: u2_dim2
        integer(c_size_t), value, intent(in) :: u2_dim3
        integer(c_int), value, intent(in) :: unx
        integer(c_int), value, intent(in) :: uny
        integer(c_int), value, intent(in) :: unz
        real(kind = knd), dimension(*) :: v
        integer(c_size_t), value, intent(in) :: v_dim1
        integer(c_size_t), value, intent(in) :: v_dim2
        integer(c_size_t), value, intent(in) :: v_dim3
        real(kind = knd), dimension(*) :: w
        integer(c_size_t), value, intent(in) :: w_dim1
        integer(c_size_t), value, intent(in) :: w_dim2
        integer(c_size_t), value, intent(in) :: w_dim3
        real(kind = knd), value, intent(in) :: dxmin
        real(kind = knd), value, intent(in) :: dymin
        real(kind = knd), value, intent(in) :: dzmin
    end subroutine cpp_CDU

    subroutine cpp_start_hot() bind(C, name='cpp_start_hot')
    end subroutine cpp_start_hot

    subroutine cpp_finish_hot() bind(C, name='cpp_finish_hot')
    end subroutine cpp_finish_hot
  end interface

contains

  ! ==========================================
  ! Main Entry Point
  ! ==========================================

  subroutine CDU( &
        u2, &
        u, &
        v, &
        w, &
        dxmin, &
        dymin, &
        dzmin, &
        unx, &
        uny, &
        unz &
    )
        real(kind = knd), dimension(:,:,:), intent(out) :: u2
        real(kind = knd), dimension(:,:,:), intent(in) :: u
        real(kind = knd), dimension(:,:,:), intent(in) :: v
        real(kind = knd), dimension(:,:,:), intent(in) :: w
        real(kind = knd), intent(in) :: dxmin
        real(kind = knd), intent(in) :: dymin
        real(kind = knd), intent(in) :: dzmin
        integer(c_int), intent(in) :: unx
        integer(c_int), intent(in) :: uny
        integer(c_int), intent(in) :: unz

    call cpp_CDU( &
        u, int(size(u, 1), kind=c_size_t), int(size(u, 2), kind=c_size_t), int(size(u, 3), kind=c_size_t), &
        u2, int(size(u2, 1), kind=c_size_t), int(size(u2, 2), kind=c_size_t), int(size(u2, 3), kind=c_size_t), &
        int(unx, kind=c_int), &
        int(uny, kind=c_int), &
        int(unz, kind=c_int), &
        v, int(size(v, 1), kind=c_size_t), int(size(v, 2), kind=c_size_t), int(size(v, 3), kind=c_size_t), &
        w, int(size(w, 1), kind=c_size_t), int(size(w, 2), kind=c_size_t), int(size(w, 3), kind=c_size_t), &
        real(dxmin, kind=c_double), &
        real(dymin, kind=c_double), &
        real(dzmin, kind=c_double) &
    )
    
  end subroutine CDU

  subroutine start_hot()
    call cpp_start_hot()
  end subroutine start_hot

  subroutine finish_hot()
    call cpp_finish_hot()
  end subroutine finish_hot

end module MomentumAdvection