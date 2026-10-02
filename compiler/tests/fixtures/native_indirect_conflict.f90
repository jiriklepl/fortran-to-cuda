! kernels
module native_indirect_conflict_module
  implicit none
  integer, parameter :: knd = kind(1.0d0)
contains
  ! kernel
  subroutine native_indirect_conflict(arr, indices, n)
    real(knd), intent(inout) :: arr(:)
    integer, intent(in) :: indices(:), n
    integer :: i
    do i = 1, n
      arr(indices(i)) = arr(indices(i)) + 1.0_knd
    end do
  end subroutine native_indirect_conflict
end module native_indirect_conflict_module
