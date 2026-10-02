"""Common drivers for call-level and resident-data stencil comparisons."""

from .paths import CASES

OUTPUT_ARRAYS = {"CDU": "U2", "CDV": "V2", "CDW": "W2"}


def _replace_once(source: str, marker: str, replacement: str) -> str:
    """Fail if a canonical driver changed instead of silently changing timing."""
    if source.count(marker) != 1:
        raise ValueError(f"expected exactly one driver marker: {marker!r}")
    return source.replace(marker, replacement, 1)


def timing_driver(
    case: str,
    shape: tuple[int, int, int],
    iterations: int,
    warmup: int,
    *,
    resident: bool = False,
) -> str:
    """Time either individual calls or a batch with one data transfer round.

    Resident mode adds two OpenACC directives to the existing driver. Its
    measured region includes creating the device mappings and copying the
    final result back; only unchanged inputs and intermediate results stay
    on the device between calls. Warmup calls precede this measured region.
    """
    if case not in OUTPUT_ARRAYS:
        raise ValueError(f"unsupported benchmark: {case}")
    nx, ny, nz = shape
    source = f"""program comparison_timing
  use MomentumAdvection
  implicit none
  integer, parameter :: nx={nx}, ny={ny}, nz={nz}
  integer, parameter :: niter={iterations}, nwarmup={warmup}
  real(knd), allocatable :: u(:,:,:), v(:,:,:), w(:,:,:), result(:,:,:)
  real(knd) :: dxmin, dymin, dzmin, checksum
  integer :: i, j, k, iteration
  integer(kind=8) :: begin_count, end_count, rate
  real(kind=8) :: elapsed_ms
  allocate(u(nx+2,ny+2,nz+2), v(nx+2,ny+2,nz+2))
  allocate(w(nx+2,ny+2,nz+2), result(nx+2,ny+2,nz+2))
  dxmin = 1.0_knd / real(nx,knd)
  dymin = 1.0_knd / real(ny,knd)
  dzmin = 1.0_knd / real(nz,knd)
  do k = 1, nz+2
    do j = 1, ny+2
      do i = 1, nx+2
        u(i,j,k) = sin(real(i,knd)*0.3_knd) * cos(real(j,knd)*0.5_knd) &
                    * (1.0_knd + 0.1_knd*real(k,knd))
        v(i,j,k) = cos(real(i,knd)*0.7_knd) * sin(real(k,knd)*0.4_knd) &
                    * (1.0_knd + 0.1_knd*real(j,knd))
        w(i,j,k) = sin(real(j,knd)*0.6_knd + real(k,knd)*0.2_knd)
      end do
    end do
  end do
  result = 0.0_knd
  do iteration = 1, nwarmup
    call {case}(result,u,v,w,dxmin,dymin,dzmin,nx,ny,nz)
  end do
  call system_clock(begin_count,rate)
  do iteration = 1, niter
    call {case}(result,u,v,w,dxmin,dymin,dzmin,nx,ny,nz)
  end do
  call system_clock(end_count)
  elapsed_ms = real(end_count-begin_count,8) / real(rate,8) * 1000.0_8
  checksum = sum(abs(result(2:nx+1,2:ny+1,2:nz+1)))
  write(*,'(A,ES26.17E3)') 'total_ms ', elapsed_ms
  write(*,'(A,ES26.17E3)') 'checksum ', checksum
  deallocate(u,v,w,result)
end program comparison_timing
"""
    if not resident:
        return source
    source = _replace_once(
        source,
        "  call system_clock(begin_count,rate)",
        "  call system_clock(begin_count,rate)\n  !$acc data copyin(u,v,w) copyout(result)",
    )
    return _replace_once(
        source,
        "  call system_clock(end_count)",
        "  !$acc end data\n  call system_clock(end_count)",
    )


def correctness_driver(case: str, *, resident: bool = False) -> str:
    """Preserve canonical validation inputs and check repeated resident calls.

    Every benchmark resets its interior output within each call. Two calls
    with unchanged inputs therefore have the same expected result, while
    exercising nested data regions and reuse of an existing device mapping.
    """
    try:
        output = OUTPUT_ARRAYS[case]
    except KeyError as error:
        raise ValueError(f"unsupported benchmark: {case}") from error
    source = (CASES / case / "test_main.f90").read_text()
    if not resident:
        return source
    call = f"    call {case}({output}, U, V, W, dxmin, dymin, dzmin, NX, NY, NZ)"
    return _replace_once(
        source,
        call,
        f"    !$acc data copyin(U,V,W) copyout({output})\n{call}\n{call}\n    !$acc end data",
    )


GPU_PROBE = """program probe
  use openacc
  implicit none
  integer, parameter :: n=1031
  real(8) :: a(n), b(n)
  real(8) :: v
  integer :: i, on_device
  a=[(real(i,8)*0.25d0,i=1,n)]
  on_device=0
  if (acc_get_num_devices(acc_device_nvidia) < 1) stop 11
  call acc_set_device_type(acc_device_nvidia)
  !$acc parallel loop copyin(a) copyout(b) private(v) copy(on_device)
  do i=1,n
    v=a(i)*3.0d0
    b(i)=v+2.0d0
    if(i==1) on_device=merge(1,0,acc_on_device(acc_device_nvidia))
  end do
  !$acc end parallel loop
  if(on_device/=1) stop 12
  if(any(b /= a*3.0d0+2.0d0)) stop 13
  print *, 'OPENACC_GPU_CONFIRMED',acc_get_num_devices(acc_device_nvidia),maxval(abs(b-(a*3.0d0+2.0d0)))
end program
"""
