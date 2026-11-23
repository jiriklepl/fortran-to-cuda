! ============================================================
! CDU Benchmark — main driver
!
! Compile-time constants (override via -D flags):
!   VAR_NX, VAR_NY, VAR_NZ   : interior grid dimensions
!   VAR_NITER                 : number of timed iterations
!   VAR_NWARMUP               : number of warm-up iterations
! ============================================================

#ifndef VAR_NX
#  define VAR_NX 64
#endif
#ifndef VAR_NY
#  define VAR_NY 64
#endif
#ifndef VAR_NZ
#  define VAR_NZ 64
#endif
#ifndef VAR_NITER
#  define VAR_NITER 100
#endif
#ifndef VAR_NWARMUP
#  define VAR_NWARMUP 5
#endif

program cdu_benchmark
    use MomentumAdvection
    implicit none

    ! ---- compile-time parameters ----
    integer, parameter :: NX      = VAR_NX
    integer, parameter :: NY      = VAR_NY
    integer, parameter :: NZ      = VAR_NZ
    integer, parameter :: NITER   = VAR_NITER
    integer, parameter :: NWARMUP = VAR_NWARMUP

    ! arrays include one ghost cell on each side in every dimension
    real(knd), allocatable :: U(:,:,:), V(:,:,:), W(:,:,:), U2(:,:,:)
    real(knd) :: dxmin, dymin, dzmin

    integer :: iter
    integer(kind=8) :: t_start, t_end, count_rate
    real(kind=8)    :: total_ms

    ! ---- allocate ----
    allocate(U (NX+2, NY+2, NZ+2))
    allocate(V (NX+2, NY+2, NZ+2))
    allocate(W (NX+2, NY+2, NZ+2))
    allocate(U2(NX+2, NY+2, NZ+2))

    ! ---- initialise ----
    dxmin = 1.0_knd / real(NX, knd)
    dymin = 1.0_knd / real(NY, knd)
    dzmin = 1.0_knd / real(NZ, knd)

    call random_number(U)
    call random_number(V)
    call random_number(W)
    U2 = 0.0_knd

    ! ---- warm-up ----
    do iter = 1, NWARMUP
        call CDU(U2, U, V, W, dxmin, dymin, dzmin, NX, NY, NZ)
    end do

    call start_hot()

    ! ---- timed run ----
    call system_clock(t_start, count_rate)
    do iter = 1, NITER
        call CDU(U2, U, V, W, dxmin, dymin, dzmin, NX, NY, NZ)
    end do
    call system_clock(t_end)

    call finish_hot()

    total_ms = real(t_end - t_start, kind=8) / real(count_rate, kind=8) * 1.0d3

    ! ---- report ----
    write(*,'(A)')          "--- CDU benchmark ---"
    write(*,'(A,I0,A,I0,A,I0)') "grid:         ", NX, " x ", NY, " x ", NZ
    write(*,'(A,I0)')       "iterations:   ", NITER
    write(*,'(A,I0)')       "warmup_iters: ", NWARMUP
    write(*,'(A,F14.3,A)')  "total_ms:     ", total_ms, " ms"
    write(*,'(A,F14.3,A)')  "ms_per_iter:  ", total_ms / real(NITER, kind=8), " ms"

    deallocate(U, V, W, U2)
end program cdu_benchmark
