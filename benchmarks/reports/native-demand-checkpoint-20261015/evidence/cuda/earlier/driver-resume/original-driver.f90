program verify
use iso_fortran_env,only:int64,error_unit
use demand_owner
use reference_owner,only:native_advance=>advance,na=>a,nb=>b,ns=>star,no=>out,nm=>metadata
implicit none
integer::step,i,j,k,native_unit,candidate_unit
open(newunit=native_unit,file='native-fields.bin',access='stream',form='unformatted',status='replace')
open(newunit=candidate_unit,file='candidate-fields.bin',access='stream',form='unformatted',status='replace')
allocate(metadata%flux(-1:2),nm%flux(-1:2))
do step=1,2
do k=-1,6
do j=-1,6
do i=-1,6
a(i,j,k)=real(merge(1,-1,step==1)*(i*i+3*j+7*k+33),8)/16.d0
enddo
enddo
enddo
b=-111.d0
star=-113.d0
out=-117.d0
metadata%flux=[-9.d0,-3.d0,1.d0,7.d0]
metadata%visits=0
na=a
nb=b
ns=star
no=out
nm%flux=metadata%flux
nm%visits=0
call native_advance()
write(error_unit,*)'BEGIN_DEMAND_CALL',step
call advance()
write(error_unit,*)'END_DEMAND_CALL',step
if(metadata%visits/=1.or.nm%visits/=1) error stop 'native call did not execute exactly once'
if(any(lbound(a)/=-1).or.any(ubound(a)/=6)) error stop 'old bounds changed'
if(any(lbound(b)/=-1).or.any(ubound(b)/=6)) error stop 'producer bounds changed'
if(any(lbound(star)/=-1).or.any(ubound(star)/=6)) error stop 'diffusion bounds changed'
if(any(lbound(out)/=-1).or.any(ubound(out)/=6)) error stop 'accumulation bounds changed'
if(lbound(metadata%flux,1)/=-1.or.ubound(metadata%flux,1)/=2) error stop 'metadata bounds changed'
if(any(transfer(a,[0_int64],size(a))/=transfer(na,[0_int64],size(na)))) error stop 'old field differs'
if(any(transfer(b,[0_int64],size(b))/=transfer(nb,[0_int64],size(nb)))) error stop 'producer field differs'
if(any(transfer(star,[0_int64],size(star))/=transfer(ns,[0_int64],size(ns)))) error stop 'diffusion halos differ'
if(any(transfer(out,[0_int64],size(out))/=transfer(no,[0_int64],size(no)))) error stop 'accumulation halos differ'
if(any(transfer(metadata%flux,[0_int64],4)/=transfer(nm%flux,[0_int64],4))) error stop 'metadata differs'
write(native_unit)na,nb,ns,no,nm%flux,nm%visits
write(candidate_unit)a,b,star,out,metadata%flux,metadata%visits
enddo
close(native_unit)
close(candidate_unit)
print *,'NATIVE_DEMAND_COMPLETE_FIELDS_BITWISE_OK'
end program
