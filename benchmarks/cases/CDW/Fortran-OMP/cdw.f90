! kernels
module MomentumAdvection
  implicit none
  private
  public CDW, knd, start_hot, finish_hot

  integer, parameter :: knd = kind(1.0d0)   ! double precision

contains

  ! kernel
  subroutine set(arr, val, Wnx, Wny, Wnz)
    real(knd), contiguous, intent(out) :: arr(:,:,:)
    real(knd), intent(in) :: val
    integer, intent(in) :: Wnx, Wny, Wnz
    integer :: i, j, k

    !$OMP PARALLEL DO PRIVATE(i, j)
    do k = 2, Wnz + 1
      do j = 2, Wny + 1
        do i = 2, Wnx + 1
          arr(i,j,k) = val
        end do
      end do
    end do
    !$OMP END PARALLEL DO
  end subroutine set

  ! kernel
  subroutine multiply(arr, val, Wnx, Wny, Wnz)
    real(knd), contiguous, intent(inout) :: arr(:,:,:)
    real(knd), intent(in) :: val
    integer, intent(in) :: Wnx, Wny, Wnz
    integer :: i, j, k

    !$OMP PARALLEL DO PRIVATE(i, j)
    do k = 2, Wnz + 1
      do j = 2, Wny + 1
        do i = 2, Wnx + 1
          arr(i,j,k) = arr(i,j,k) * val
        end do
      end do
    end do
    !$OMP END PARALLEL DO
  end subroutine multiply

  ! kernel
  subroutine CDWdiv(W2, U, V, W, dxmin, dymin, dzmin, Wnx, Wny, Wnz)
    real(knd), contiguous, intent(out) :: W2(:,:,:)
    real(knd), contiguous, intent(in)  :: U(:,:,:), V(:,:,:), W(:,:,:)
    real(knd), intent(in) :: dxmin, dymin, dzmin
    integer, intent(in) :: Wnx, Wny, Wnz
    
    real(knd) :: Ax, Ay, Az
    integer :: i, j, k
    
    Ax = 0.25_knd / dxmin
    Ay = 0.25_knd / dymin
    Az = 0.25_knd / dzmin
    
    !$OMP PARALLEL DO PRIVATE(i, j)
    do k = 2, Wnz + 1
      do j = 2, Wny + 1
        do i = 2, Wnx + 1
          
          W2(i,j,k) = - ((Az*(W(i,j,k+1) + W(i,j,k)) * (W(i,j,k+1) + W(i,j,k)) &
                        - Az*(W(i,j,k) + W(i,j,k-1)) * (W(i,j,k) + W(i,j,k-1))) &
                       + (Ay*(W(i,j+1,k) + W(i,j,k)) * (V(i,j,k+1) + V(i,j,k)) &
                        - Ay*(W(i,j,k) + W(i,j-1,k)) * (V(i,j-1,k) + V(i,j-1,k+1))) &
                       + (Ax*(W(i+1,j,k) + W(i,j,k)) * (U(i,j,k+1) + U(i,j,k)) &
                        - Ax*(W(i,j,k) + W(i-1,j,k)) * (U(i-1,j,k+1) + U(i-1,j,k))))
        end do
      end do
    end do
    !$OMP END PARALLEL DO
  end subroutine CDWdiv

  ! kernel
  subroutine CDWadv(W2, U, V, W, dxmin, dymin, dzmin, Wnx, Wny, Wnz)
    real(knd), contiguous, intent(inout) :: W2(:,:,:)
    real(knd), contiguous, intent(in)  :: U(:,:,:), V(:,:,:), W(:,:,:)
    real(knd), intent(in) :: dxmin, dymin, dzmin
    integer, intent(in) :: Wnx, Wny, Wnz
    
    real(knd) :: Ax, Ay, Az, Uadv, Vadv
    integer :: i, j, k
    
    Ax = 0.125_knd / dxmin
    Ay = 0.125_knd / dymin
    Az = 0.5_knd / dzmin
    
    !$OMP PARALLEL DO PRIVATE(i, j, Uadv, Vadv)
    do k = 2, Wnz + 1
      do j = 2, Wny + 1
        do i = 2, Wnx + 1
          Uadv = ( U(i,j,k) + U(i,j,k+1) + U(i-1,j,k) + U(i-1,j,k+1) )
          Vadv = ( V(i,j,k) + V(i,j,k+1) + V(i,j-1,k) + V(i,j-1,k+1) )
          W2(i,j,k) = W2(i,j,k) &
                     - (Ax*(W(i+1,j,k)-W(i-1,j,k)) * Uadv&
                     +  Ay*(W(i,j+1,k)-W(i,j-1,k)) * Vadv&
                     +  Az*(W(i,j,k+1)-W(i,j,k-1)) * W(i,j,k) )
        end do
      end do
    end do
    !$OMP END PARALLEL DO
  end subroutine CDWadv

  ! ==========================================
  ! Main Entry Point
  ! ==========================================

  ! kernel
  subroutine CDW(W2, U, V, W, dxmin, dymin, dzmin, Wnx, Wny, Wnz)
    real(knd), contiguous, intent(out) :: W2(:,:,:)
    real(knd), contiguous, intent(in)  :: U(:,:,:), V(:,:,:), W(:,:,:)
    real(knd), intent(in) :: dxmin, dymin, dzmin
    real(knd) :: zero, half
    integer, intent(in) :: Wnx, Wny, Wnz

    zero = 0.0_knd
    half = 0.5_knd

    ! Initialize the array
    call set(W2, zero, Wnx, Wny, Wnz)
    
    ! Core physics passes
    call CDWdiv(W2, U, V, W, dxmin, dymin, dzmin, Wnx, Wny, Wnz)
    call CDWadv(W2, U, V, W, dxmin, dymin, dzmin, Wnx, Wny, Wnz)
    
    ! Scale the array
    call multiply(W2, half, Wnx, Wny, Wnz)
    
  end subroutine CDW

  subroutine start_hot()
  end subroutine start_hot

  subroutine finish_hot()
  end subroutine finish_hot

end module MomentumAdvection
