module fort_inline_182d4562d805
implicit none
contains
subroutine region(u,fort_region_lb_u_1,fort_region_lb_u_2,fort_region_lb_u_3,a,pr_gradient_x,unx,uny,unz)
real(8), intent(inout) :: u(:,:,:)
integer, value, intent(in) :: fort_region_lb_u_1
integer, value, intent(in) :: fort_region_lb_u_2
integer, value, intent(in) :: fort_region_lb_u_3
real(8), intent(in) :: a
real(8), intent(in) :: pr_gradient_x
integer, value, intent(in) :: unx
integer, value, intent(in) :: uny
integer, value, intent(in) :: unz
integer :: i
integer :: j
integer :: k
DO k = 1, Unz
  DO j = 1, Uny
    DO i = 1, Unx
      u((i) - fort_region_lb_u_1 + 1, (j) - fort_region_lb_u_2 + 1, (k) - fort_region_lb_u_3 + 1) = u((i) - &
  & fort_region_lb_u_1 + 1, (j) - fort_region_lb_u_2 + 1, (k) - fort_region_lb_u_3 + 1) + A * pr_gradient_x
    END DO
  END DO
END DO
end subroutine
end module
