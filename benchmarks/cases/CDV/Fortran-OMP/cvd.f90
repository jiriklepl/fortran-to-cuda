! kernels
module MomentumAdvection
  implicit none
  private
  public CDV, knd, start_hot, finish_hot

  integer, parameter :: knd = kind(1.0d0)   ! double precision

contains

  ! kernel
  subroutine set(arr, val, Vnx, Vny, Vnz)
    real(knd), contiguous, intent(out) :: arr(:,:,:)
    real(knd), intent(in) :: val
    integer, intent(in) :: Vnx, Vny, Vnz
    integer :: i, j, k

    !$OMP PARALLEL DO PRIVATE(i, j)
    do k = 2, Vnz + 1
      do j = 2, Vny + 1
        do i = 2, Vnx + 1
          arr(i,j,k) = val
        end do
      end do
    end do
    !$OMP END PARALLEL DO
  end subroutine set

  ! kernel
  subroutine multiply(arr, val, Vnx, Vny, Vnz)
    real(knd), contiguous, intent(inout) :: arr(:,:,:)
    real(knd), intent(in) :: val
    integer, intent(in) :: Vnx, Vny, Vnz
    integer :: i, j, k

    !$OMP PARALLEL DO PRIVATE(i, j)
    do k = 2, Vnz + 1
      do j = 2, Vny + 1
        do i = 2, Vnx + 1
          arr(i,j,k) = arr(i,j,k) * val
        end do
      end do
    end do
    !$OMP END PARALLEL DO
  end subroutine multiply

  ! kernel
  subroutine CDVdiv(V2, U, V, W, dxmin, dymin, dzmin, Vnx, Vny, Vnz)
    real(knd), contiguous, intent(out) :: V2(:,:,:)
    real(knd), contiguous, intent(in)  :: U(:,:,:), V(:,:,:), W(:,:,:)
    real(knd), intent(in) :: dxmin, dymin, dzmin
    integer, intent(in) :: Vnx, Vny, Vnz
    
    real(knd) :: Ax, Ay, Az
    integer :: i, j, k
    
    Ax = 0.25_knd / dxmin
    Ay = 0.25_knd / dymin
    Az = 0.25_knd / dzmin
    
    !$OMP PARALLEL DO PRIVATE(i, j)
    do k = 2, Vnz + 1
      do j = 2, Vny + 1
        do i = 2, Vnx + 1
          
          V2(i,j,k) = - ((Ay*(V(i,j+1,k) + V(i,j,k)) * (V(i,j+1,k) + V(i,j,k)) &
                         -Ay*(V(i,j,k) + V(i,j-1,k)) * (V(i,j,k) + V(i,j-1,k))) &
                        +(Ax*(V(i+1,j,k) + V(i,j,k)) * (U(i,j+1,k) + U(i,j,k)) &
                         -Ax*(V(i,j,k) + V(i-1,j,k)) * (U(i-1,j+1,k) + U(i-1,j,k))) &
                        +(Az*(V(i,j,k+1) + V(i,j,k)) * (W(i,j+1,k) + W(i,j,k)) &
                         -Az*(V(i,j,k) + V(i,j,k-1)) * (W(i,j+1,k-1) + W(i,j,k-1))))
        end do
      end do
    end do
    !$OMP END PARALLEL DO
  end subroutine CDVdiv

  ! kernel
  subroutine CDVadv(V2, U, V, W, dxmin, dymin, dzmin, Vnx, Vny, Vnz)
    real(knd), contiguous, intent(inout) :: V2(:,:,:)
    real(knd), contiguous, intent(in)  :: U(:,:,:), V(:,:,:), W(:,:,:)
    real(knd), intent(in) :: dxmin, dymin, dzmin
    integer, intent(in) :: Vnx, Vny, Vnz
    
    real(knd) :: Ax, Ay, Az, Uadv, Wadv
    integer :: i, j, k
    
    Ax = 0.125_knd / dxmin
    Ay = 0.5_knd / dymin
    Az = 0.125_knd / dzmin
    
    !$OMP PARALLEL DO PRIVATE(i, j, Uadv, Wadv)
    do k = 2, Vnz + 1
      do j = 2, Vny + 1
        do i = 2, Vnx + 1
          Uadv = ( U(i,j,k) + U(i,j+1,k) + U(i-1,j,k) + U(i-1,j+1,k) )
          Wadv = ( W(i,j,k) + W(i,j+1,k) + W(i,j,k-1) + W(i,j+1,k-1) )
          V2(i,j,k) = V2(i,j,k) &
                     - (Ax*(V(i+1,j,k)-V(i-1,j,k)) * Uadv&
                     +  Ay*(V(i,j+1,k)-V(i,j-1,k)) * V(i,j,k) &
                     +  Az*(V(i,j,k+1)-V(i,j,k-1)) * Wadv )
        end do
      end do
    end do
    !$OMP END PARALLEL DO
  end subroutine CDVadv

  ! ==========================================
  ! Main Entry Point
  ! ==========================================

  ! kernel
  subroutine CDV(V2, U, V, W, dxmin, dymin, dzmin, Vnx, Vny, Vnz)
    real(knd), contiguous, intent(out) :: V2(:,:,:)
    real(knd), contiguous, intent(in)  :: U(:,:,:), V(:,:,:), W(:,:,:)
    real(knd), intent(in) :: dxmin, dymin, dzmin
    real(knd) :: zero, half
    integer, intent(in) :: Vnx, Vny, Vnz

    zero = 0.0_knd
    half = 0.5_knd

    ! Initialize the array
    call set(V2, zero, Vnx, Vny, Vnz)
    
    ! Core physics passes
    call CDVdiv(V2, U, V, W, dxmin, dymin, dzmin, Vnx, Vny, Vnz)
    call CDVadv(V2, U, V, W, dxmin, dymin, dzmin, Vnx, Vny, Vnz)
    
    ! Scale the array
    call multiply(V2, half, Vnx, Vny, Vnz)
    
  end subroutine CDV

  subroutine start_hot()
  end subroutine start_hot

  subroutine finish_hot()
  end subroutine finish_hot

end module MomentumAdvection