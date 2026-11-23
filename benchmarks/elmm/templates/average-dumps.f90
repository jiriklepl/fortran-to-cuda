  block
    use OutputData, only: U_avg, V_avg, W_avg, Pr_avg, Scalar_avg
    integer :: comparison_unit
    if (allocated(U_avg)) then
      open(newunit=comparison_unit,file='fields-U_avg.bin', &
           access='stream',form='unformatted',status='new')
      write(comparison_unit) U_avg
      close(comparison_unit)
      open(newunit=comparison_unit,file='fields-U_avg.shape',status='new')
      write(comparison_unit,*) storage_size(U_avg), shape(U_avg), &
           lbound(U_avg), ubound(U_avg)
      close(comparison_unit)
    end if
    if (allocated(V_avg)) then
      open(newunit=comparison_unit,file='fields-V_avg.bin', &
           access='stream',form='unformatted',status='new')
      write(comparison_unit) V_avg
      close(comparison_unit)
      open(newunit=comparison_unit,file='fields-V_avg.shape',status='new')
      write(comparison_unit,*) storage_size(V_avg), shape(V_avg), &
           lbound(V_avg), ubound(V_avg)
      close(comparison_unit)
    end if
    if (allocated(W_avg)) then
      open(newunit=comparison_unit,file='fields-W_avg.bin', &
           access='stream',form='unformatted',status='new')
      write(comparison_unit) W_avg
      close(comparison_unit)
      open(newunit=comparison_unit,file='fields-W_avg.shape',status='new')
      write(comparison_unit,*) storage_size(W_avg), shape(W_avg), &
           lbound(W_avg), ubound(W_avg)
      close(comparison_unit)
    end if
    if (allocated(Pr_avg)) then
      open(newunit=comparison_unit,file='fields-Pr_avg.bin', &
           access='stream',form='unformatted',status='new')
      write(comparison_unit) Pr_avg
      close(comparison_unit)
      open(newunit=comparison_unit,file='fields-Pr_avg.shape',status='new')
      write(comparison_unit,*) storage_size(Pr_avg), shape(Pr_avg), &
           lbound(Pr_avg), ubound(Pr_avg)
      close(comparison_unit)
    end if
    if (allocated(Scalar_avg)) then
      open(newunit=comparison_unit,file='fields-Scalar_avg.bin', &
           access='stream',form='unformatted',status='new')
      write(comparison_unit) Scalar_avg
      close(comparison_unit)
      open(newunit=comparison_unit,file='fields-Scalar_avg.shape',status='new')
      write(comparison_unit,*) storage_size(Scalar_avg), shape(Scalar_avg), &
           lbound(Scalar_avg), ubound(Scalar_avg)
      close(comparison_unit)
    end if
  end block
