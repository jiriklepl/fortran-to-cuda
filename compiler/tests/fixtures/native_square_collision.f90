! kernels
module native_square_collision_module
  implicit none
  integer, parameter :: knd = kind(1.0d0)
contains
  ! kernel
  subroutine native_square_collision(arr)
    real(knd), intent(inout) :: arr(:)
    integer :: i
    do i = -2, 2
      arr(i*i+1) = 0.5_knd*i
    end do
  end subroutine native_square_collision
end module native_square_collision_module
