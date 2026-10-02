! CDU correctness test
! Calls the kernel once on a deterministic input and dumps the interior
! of U2 to stdout (one value per line) for cross-variant comparison.
!
! Grid size is set by VAR_NX/VAR_NY/VAR_NZ (defaults: 16).

#ifndef VAR_NX
#  define VAR_NX 16
#endif
#ifndef VAR_NY
#  define VAR_NY 16
#endif
#ifndef VAR_NZ
#  define VAR_NZ 16
#endif

program cdu_test
    use MomentumAdvection
    implicit none

    integer, parameter :: NX = VAR_NX
    integer, parameter :: NY = VAR_NY
    integer, parameter :: NZ = VAR_NZ

    real(knd), allocatable :: U(:,:,:), V(:,:,:), W(:,:,:), U2(:,:,:)
    real(knd) :: dxmin, dymin, dzmin
    integer   :: i, j, k

    allocate(U (NX+2, NY+2, NZ+2))
    allocate(V (NX+2, NY+2, NZ+2))
    allocate(W (NX+2, NY+2, NZ+2))
    allocate(U2(NX+2, NY+2, NZ+2))

    dxmin = 1.0_knd / real(NX, knd)
    dymin = 1.0_knd / real(NY, knd)
    dzmin = 1.0_knd / real(NZ, knd)

    ! Deterministic, reproducible initialisation (no random_number)
    do k = 1, NZ+2
        do j = 1, NY+2
            do i = 1, NX+2
                U(i,j,k) = sin(real(i,knd)*0.3_knd) * cos(real(j,knd)*0.5_knd) &
                          * (1.0_knd + 0.1_knd*real(k,knd))
                V(i,j,k) = cos(real(i,knd)*0.7_knd) * sin(real(k,knd)*0.4_knd) &
                          * (1.0_knd + 0.1_knd*real(j,knd))
                W(i,j,k) = sin(real(j,knd)*0.6_knd + real(k,knd)*0.2_knd)
            end do
        end do
    end do
    U2 = 0.0_knd

    call CDU(U2, U, V, W, dxmin, dymin, dzmin, NX, NY, NZ)

    ! Dump interior (ghost-free) of U2, one value per line, full precision
    do k = 2, NZ+1
        do j = 2, NY+1
            do i = 2, NX+1
                write(*,'(g0.17)') U2(i,j,k)
            end do
        end do
    end do

    deallocate(U, V, W, U2)
end program cdu_test
