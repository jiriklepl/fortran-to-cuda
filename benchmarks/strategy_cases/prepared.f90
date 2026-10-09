module prepared_fields
  implicit none
contains
  subroutine prepared_update(source,destination,nx,ny,nz)
    integer,intent(in)::nx,ny,nz
    real(8),intent(in)::source(:,:,:)
    real(8),intent(inout)::destination(:,:,:)
    integer::i,j,k,lower,upper
    real(8)::coefficient,term
    lower=2
    upper=nx+1
    coefficient=0.125d0+source(1,1,1)*0.01d0
    if(ny>0.and.nz>0) then
      !$omp parallel do collapse(3) private(i,j,k,term)
      do k=2,nz+1
        do j=2,ny+1
          do i=lower,upper
            term=source(i,j,k)
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            term=term*0.99995d0+source(i,j,k)*0.0001d0
            destination(i,j,k)=term+coefficient
          end do
        end do
      end do
      !$omp end parallel do
    end if
    coefficient=0.25d0+source(1,1,1)*0.02d0
    if(nx>0) then
      !$omp parallel do collapse(3) private(i,j,k)
      do k=2,nz+1
        do j=2,ny+1
          do i=lower,upper
            destination(i,j,k)=destination(i,j,k)+coefficient*(source(i-1,j,k)+source(i+1,j,k))
          end do
        end do
      end do
      !$omp end parallel do
    end if
  end subroutine
end module
