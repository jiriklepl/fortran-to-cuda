        !$omp master
        !$fort reduction(+: Phi_ref)
        do j = 1, Prny
          do i = 1, Prnx
            Phi_ref = Phi_ref + Pr(i,j,k_ref)
          end do
        end do
        Phi_ref = Phi_ref / (Prnx*Prny)
        !$omp end master
        !$omp barrier
