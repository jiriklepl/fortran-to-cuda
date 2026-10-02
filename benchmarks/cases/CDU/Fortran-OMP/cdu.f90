! kernels
module MomentumAdvection
  implicit none
  private
  public CDU, knd, start_hot, finish_hot

  integer, parameter :: knd = kind(1.0d0)   ! double precision

contains

  ! kernel
  subroutine set(arr, val, Unx, Uny, Unz)
    real(knd), contiguous, intent(out) :: arr(:,:,:)
    real(knd), intent(in) :: val
    integer, intent(in) :: Unx, Uny, Unz
    integer :: i, j, k

    !$OMP PARALLEL DO PRIVATE(i, j)
    do k = 2, Unz + 1
      do j = 2, Uny + 1
        do i = 2, Unx + 1
          arr(i,j,k) = val
        end do
      end do
    end do
    !$OMP END PARALLEL DO
  end subroutine set

  ! kernel
  subroutine multiply(arr, val, Unx, Uny, Unz)
    real(knd), contiguous, intent(inout) :: arr(:,:,:)
    real(knd), intent(in) :: val
    integer, intent(in) :: Unx, Uny, Unz
    integer :: i, j, k

    !$OMP PARALLEL DO PRIVATE(i, j)
    do k = 2, Unz + 1
      do j = 2, Uny + 1
        do i = 2, Unx + 1
          arr(i,j,k) = arr(i,j,k) * val
        end do
      end do
    end do
    !$OMP END PARALLEL DO
  end subroutine multiply

  ! kernel
  subroutine CDUdiv(U2, U, V, W, dxmin, dymin, dzmin, Unx, Uny, Unz)
    real(knd), contiguous, intent(out) :: U2(:,:,:)
    real(knd), contiguous, intent(in)  :: U(:,:,:), V(:,:,:), W(:,:,:)
    real(knd), intent(in) :: dxmin, dymin, dzmin
    integer, intent(in) :: Unx, Uny, Unz
    
    real(knd) :: Ax, Ay, Az
    integer :: i, j, k
    
    Ax = 0.25_knd / dxmin
    Ay = 0.25_knd / dymin
    Az = 0.25_knd / dzmin
    
    !$OMP PARALLEL DO PRIVATE(i, j)
    do k = 2, Unz + 1
      do j = 2, Uny + 1
        do i = 2, Unx + 1
          
          U2(i,j,k) = - ((Ax*(U(i+1,j,k) + U(i,j,k)) * (U(i+1,j,k) + U(i,j,k)) &
                        - Ax*(U(i,j,k) + U(i-1,j,k)) * (U(i,j,k) + U(i-1,j,k))) &
                       + (Ay*(U(i,j+1,k) + U(i,j,k)) * (V(i+1,j,k) + V(i,j,k)) &
                        - Ay*(U(i,j,k) + U(i,j-1,k)) * (V(i+1,j-1,k) + V(i,j-1,k))) &
                       + (Az*(U(i,j,k+1) + U(i,j,k)) * (W(i+1,j,k) + W(i,j,k)) &
                        - Az*(U(i,j,k) + U(i,j,k-1)) * (W(i+1,j,k-1) + W(i,j,k-1))))
        end do
      end do
    end do
    !$OMP END PARALLEL DO
  end subroutine CDUdiv

  ! kernel
  subroutine CDUadv(U2, U, V, W, dxmin, dymin, dzmin, Unx, Uny, Unz)
    real(knd), contiguous, intent(inout) :: U2(:,:,:)
    real(knd), contiguous, intent(in)  :: U(:,:,:), V(:,:,:), W(:,:,:)
    real(knd), intent(in) :: dxmin, dymin, dzmin
    integer, intent(in) :: Unx, Uny, Unz
    
    real(knd) :: Ax, Ay, Az, Vadv, Wadv
    integer :: i, j, k
    
    Ax = 0.5_knd / dxmin
    Ay = 0.125_knd / dymin
    Az = 0.125_knd / dzmin
    
    !$OMP PARALLEL DO PRIVATE(i, j, Vadv, Wadv)
    do k = 2, Unz + 1
      do j = 2, Uny + 1
        do i = 2, Unx + 1
          Vadv = ( V(i,j,k) + V(i+1,j,k) + V(i,j-1,k) + V(i+1,j-1,k) )
          Wadv = ( W(i,j,k) + W(i+1,j,k) + W(i,j,k-1) + W(i+1,j,k-1) )
          U2(i,j,k) = U2(i,j,k) &
                     - (Ax*(U(i+1,j,k)-U(i-1,j,k)) * U(i,j,k) &
                     +  Ay*(U(i,j+1,k)-U(i,j-1,k)) * Vadv&
                     +  Az*(U(i,j,k+1)-U(i,j,k-1)) * Wadv )
        end do
      end do
    end do
    !$OMP END PARALLEL DO
  end subroutine CDUadv

  ! ==========================================
  ! Main Entry Point
  ! ==========================================

  ! kernel
  subroutine CDU(U2, U, V, W, dxmin, dymin, dzmin, Unx, Uny, Unz)
    real(knd), contiguous, intent(out) :: U2(:,:,:)
    real(knd), contiguous, intent(in)  :: U(:,:,:), V(:,:,:), W(:,:,:)
    real(knd), intent(in) :: dxmin, dymin, dzmin
    real(knd) :: zero, half
    integer, intent(in) :: Unx, Uny, Unz

    zero = 0.0_knd
    half = 0.5_knd

    ! Initialize the array
    call set(U2, zero, Unx, Uny, Unz)
    
    ! Core physics passes
    call CDUdiv(U2, U, V, W, dxmin, dymin, dzmin, Unx, Uny, Unz)
    call CDUadv(U2, U, V, W, dxmin, dymin, dzmin, Unx, Uny, Unz)
    
    ! Scale the array
    call multiply(U2, half, Unx, Uny, Unz)
    
  end subroutine CDU

  subroutine start_hot()
  end subroutine start_hot

  subroutine finish_hot()
  end subroutine finish_hot

end module MomentumAdvection
