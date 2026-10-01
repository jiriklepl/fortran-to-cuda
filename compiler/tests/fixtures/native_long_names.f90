! kernels
module native_long_names_module
  implicit none
  integer, parameter :: knd = kind(1.0d0)
contains
  ! Procedure and array/value dummy names each use Fortran's 63-character limit.
  ! kernel
  subroutine native_long_procedure_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx( &
      long_array_argument_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, &
      long_value_argument_vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv, nx, ny, nz)
    real(knd), contiguous, intent(inout) :: long_array_argument_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(:,:,:)
    real(knd), intent(in) :: long_value_argument_vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    integer, intent(in) :: nx, ny, nz
    integer :: i, j, k

    do k = 1, nz
      do j = 1, ny
        do i = 1, nx
          long_array_argument_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(i,j,k) = &
              long_array_argument_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(i,j,k) * &
              long_value_argument_vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        end do
      end do
    end do
  end subroutine native_long_procedure_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
end module native_long_names_module
