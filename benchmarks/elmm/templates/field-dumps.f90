  block
    use Dynamics, only: comparison_U2 => U2
    use Dynamics, only: comparison_V2 => V2
    use Dynamics, only: comparison_W2 => W2
    use Dynamics, only: comparison_Ustar => Ustar
    use Dynamics, only: comparison_Vstar => Vstar
    use Dynamics, only: comparison_Wstar => Wstar
    integer :: comparison_unit
    if (allocated(U)) then
      open(newunit=comparison_unit,file='fields-U.bin', &
           access='stream',form='unformatted',status='new')
      write(comparison_unit) U
      close(comparison_unit)
      open(newunit=comparison_unit,file='fields-U.shape',status='new')
      write(comparison_unit,*) storage_size(U), shape(U), &
           lbound(U), ubound(U)
      close(comparison_unit)
    end if
    if (allocated(V)) then
      open(newunit=comparison_unit,file='fields-V.bin', &
           access='stream',form='unformatted',status='new')
      write(comparison_unit) V
      close(comparison_unit)
      open(newunit=comparison_unit,file='fields-V.shape',status='new')
      write(comparison_unit,*) storage_size(V), shape(V), &
           lbound(V), ubound(V)
      close(comparison_unit)
    end if
    if (allocated(W)) then
      open(newunit=comparison_unit,file='fields-W.bin', &
           access='stream',form='unformatted',status='new')
      write(comparison_unit) W
      close(comparison_unit)
      open(newunit=comparison_unit,file='fields-W.shape',status='new')
      write(comparison_unit,*) storage_size(W), shape(W), &
           lbound(W), ubound(W)
      close(comparison_unit)
    end if
    if (allocated(Pr)) then
      open(newunit=comparison_unit,file='fields-Pr.bin', &
           access='stream',form='unformatted',status='new')
      write(comparison_unit) Pr
      close(comparison_unit)
      open(newunit=comparison_unit,file='fields-Pr.shape',status='new')
      write(comparison_unit,*) storage_size(Pr), shape(Pr), &
           lbound(Pr), ubound(Pr)
      close(comparison_unit)
    end if
    if (allocated(Viscosity)) then
      open(newunit=comparison_unit,file='fields-Viscosity.bin', &
           access='stream',form='unformatted',status='new')
      write(comparison_unit) Viscosity
      close(comparison_unit)
      open(newunit=comparison_unit,file='fields-Viscosity.shape',status='new')
      write(comparison_unit,*) storage_size(Viscosity), shape(Viscosity), &
           lbound(Viscosity), ubound(Viscosity)
      close(comparison_unit)
    end if
    if (allocated(comparison_U2)) then
      open(newunit=comparison_unit,file='fields-U2.bin', &
           access='stream',form='unformatted',status='new')
      write(comparison_unit) comparison_U2
      close(comparison_unit)
      open(newunit=comparison_unit,file='fields-U2.shape',status='new')
      write(comparison_unit,*) storage_size(comparison_U2), shape(comparison_U2), &
           lbound(comparison_U2), ubound(comparison_U2)
      close(comparison_unit)
    end if
    if (allocated(comparison_V2)) then
      open(newunit=comparison_unit,file='fields-V2.bin', &
           access='stream',form='unformatted',status='new')
      write(comparison_unit) comparison_V2
      close(comparison_unit)
      open(newunit=comparison_unit,file='fields-V2.shape',status='new')
      write(comparison_unit,*) storage_size(comparison_V2), shape(comparison_V2), &
           lbound(comparison_V2), ubound(comparison_V2)
      close(comparison_unit)
    end if
    if (allocated(comparison_W2)) then
      open(newunit=comparison_unit,file='fields-W2.bin', &
           access='stream',form='unformatted',status='new')
      write(comparison_unit) comparison_W2
      close(comparison_unit)
      open(newunit=comparison_unit,file='fields-W2.shape',status='new')
      write(comparison_unit,*) storage_size(comparison_W2), shape(comparison_W2), &
           lbound(comparison_W2), ubound(comparison_W2)
      close(comparison_unit)
    end if
    if (allocated(comparison_Ustar)) then
      open(newunit=comparison_unit,file='fields-Ustar.bin', &
           access='stream',form='unformatted',status='new')
      write(comparison_unit) comparison_Ustar
      close(comparison_unit)
      open(newunit=comparison_unit,file='fields-Ustar.shape',status='new')
      write(comparison_unit,*) storage_size(comparison_Ustar), shape(comparison_Ustar), &
           lbound(comparison_Ustar), ubound(comparison_Ustar)
      close(comparison_unit)
    end if
    if (allocated(comparison_Vstar)) then
      open(newunit=comparison_unit,file='fields-Vstar.bin', &
           access='stream',form='unformatted',status='new')
      write(comparison_unit) comparison_Vstar
      close(comparison_unit)
      open(newunit=comparison_unit,file='fields-Vstar.shape',status='new')
      write(comparison_unit,*) storage_size(comparison_Vstar), shape(comparison_Vstar), &
           lbound(comparison_Vstar), ubound(comparison_Vstar)
      close(comparison_unit)
    end if
    if (allocated(comparison_Wstar)) then
      open(newunit=comparison_unit,file='fields-Wstar.bin', &
           access='stream',form='unformatted',status='new')
      write(comparison_unit) comparison_Wstar
      close(comparison_unit)
      open(newunit=comparison_unit,file='fields-Wstar.shape',status='new')
      write(comparison_unit,*) storage_size(comparison_Wstar), shape(comparison_Wstar), &
           lbound(comparison_Wstar), ubound(comparison_Wstar)
      close(comparison_unit)
    end if
    if (allocated(Temperature)) then
      open(newunit=comparison_unit,file='fields-Temperature.bin', &
           access='stream',form='unformatted',status='new')
      write(comparison_unit) Temperature
      close(comparison_unit)
      open(newunit=comparison_unit,file='fields-Temperature.shape',status='new')
      write(comparison_unit,*) storage_size(Temperature), shape(Temperature), &
           lbound(Temperature), ubound(Temperature)
      close(comparison_unit)
    end if
    if (allocated(Moisture)) then
      open(newunit=comparison_unit,file='fields-Moisture.bin', &
           access='stream',form='unformatted',status='new')
      write(comparison_unit) Moisture
      close(comparison_unit)
      open(newunit=comparison_unit,file='fields-Moisture.shape',status='new')
      write(comparison_unit,*) storage_size(Moisture), shape(Moisture), &
           lbound(Moisture), ubound(Moisture)
      close(comparison_unit)
    end if
    if (allocated(Scalar)) then
      open(newunit=comparison_unit,file='fields-Scalar.bin', &
           access='stream',form='unformatted',status='new')
      write(comparison_unit) Scalar
      close(comparison_unit)
      open(newunit=comparison_unit,file='fields-Scalar.shape',status='new')
      write(comparison_unit,*) storage_size(Scalar), shape(Scalar), &
           lbound(Scalar), ubound(Scalar)
      close(comparison_unit)
    end if
    if (allocated(TDiff)) then
      open(newunit=comparison_unit,file='fields-TDiff.bin', &
           access='stream',form='unformatted',status='new')
      write(comparison_unit) TDiff
      close(comparison_unit)
      open(newunit=comparison_unit,file='fields-TDiff.shape',status='new')
      write(comparison_unit,*) storage_size(TDiff), shape(TDiff), &
           lbound(TDiff), ubound(TDiff)
      close(comparison_unit)
    end if
  end block
