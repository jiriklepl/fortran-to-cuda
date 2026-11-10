      do k = 1, Unz
        do j = 1, Uny
          do i = 1, Unx
            U(i,j,k) = U(i,j,k) + A * pr_gradient_x
          end do
        end do
      end do
