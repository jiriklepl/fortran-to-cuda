! Generic native numerical workloads. No application code or measurements.
! The producer specializes each family before its numerical DO and substitutes
! rk, without adding flags to the original compiler-options identity.
module numerical_native_v2
use iso_c_binding
use iso_fortran_env, only: compiler_version, compiler_options
implicit none
integer, parameter :: rk = c_double
contains
pure function primitive(operation,x) result(y)
integer,intent(in)::operation
real(rk),intent(in)::x
real(rk)::y
select case(operation)
case(0)
  y=0.25_rk+0.125_rk*sqrt(1.0_rk+0.25_rk*x*x)
case(1)
  y=0.25_rk+0.125_rk*acos(0.125_rk*x)
case default
  y=0.25_rk+0.125_rk*cos(x)
end select
end function

pure function numerical_work(family,input,other) result(y)
integer,intent(in)::family
real(rk),intent(in)::input,other
real(rk)::y,x,x1,x2,x3,x4,total,private_value
real(rk)::a(4,4),b(4,4),c(4,4),d(4,4)
integer::step,operation,row,col,k,roots,angles,cosines,width
x=input
private_value=0.0_rk
select case(family)
case(0)
  x1=x
  x2=x*0.5_rk
  x3=x*0.25_rk
  x4=x*0.125_rk
  do step=1,64
    x1=x1*1.000001_rk+0.000001_rk
    x2=x2*1.000002_rk+0.000002_rk
    x3=x3*1.000003_rk+0.000003_rk
    x4=x4*1.000004_rk+0.000004_rk
  enddo
  y=x1+x2+x3+x4
case(1)
  y=x+0.25_rk*other
case(2:4)
  do step=1,16
    x=primitive(family-2,x)
  enddo
  y=x
case(9)
  x1=x
  x2=x*0.5_rk
  x3=x*0.25_rk
  x4=x*0.125_rk
  do step=1,16
    x1=(x1+0.125_rk)/(other+x2*0.25_rk+1.0_rk)
    x2=(x2+0.25_rk)/(other+x3*0.125_rk+1.0_rk)
    x3=(x3+0.5_rk)/(other+x4*0.0625_rk+1.0_rk)
    x4=(x4+0.75_rk)/(other+x1*0.03125_rk+1.0_rk)
  enddo
  y=x1+x2+x3+x4
case(10)
  do step=1,64
    x=x*1.000001_rk+0.000001_rk
    x=max(-0.9_rk,min(-0.1_rk,-abs(x)))
    x=(x-0.1_rk)/(-1.0_rk-other)
  enddo
  y=x
case default
  if(family>=7) then
    width=merge(4,2,family==7)
    do col=1,width
      do row=1,width
        a(row,col)=x+real(row+col,rk)*0.01_rk
        b(row,col)=0.02_rk*x+merge(1.0_rk,0.0_rk,row==col)
      enddo
    enddo
    do col=1,width
      do row=1,width
        total=0.0_rk
        do k=1,width
          total=total+a(row,k)*b(k,col)
        enddo
        c(row,col)=total
        d(row,col)=c(row,col)+a(row,col)
      enddo
    enddo
    total=d(1,1)
    do k=2,width
      total=total+d(k,k)
    enddo
    private_value=0.125_rk+0.01_rk*total
    x=private_value
  endif
  if(family==5.or.family==7) then
    roots=6
    angles=2
    cosines=5
  else
    roots=2
    angles=4
    cosines=1
  endif
  do step=1,16
    do operation=1,roots
      x=primitive(0,x)
    enddo
    do operation=1,angles
      x=primitive(1,x)
    enddo
    do operation=1,cosines
      x=primitive(2,x)
    enddo
    ! Independent ordinary-expression holdout: ABS, unary minus, ordered
    ! MIN/MAX, subtraction and division; no application expression is used.
    x=max(-0.9_rk,min(-0.1_rk,-abs(x)))
    x=(x-0.1_rk)/(-1.0_rk-other)
  enddo
  y=x+0.001_rk*private_value
end select
end function

subroutine fort_numerical_native_v2(family,n,threads,fork_join,a,b,output) bind(c)
integer(c_int),value::family,threads,fork_join
integer(c_size_t),value::n
real(rk),intent(in)::a(*),b(*)
real(rk),intent(out)::output(*)
integer(c_size_t)::i
if(fork_join==0) then
  do i=1,n
    output(i)=numerical_work(family,a(i),b(i))
  enddo
else
  !$omp parallel do num_threads(threads) schedule(static) default(none) &
  !$omp shared(family,n,a,b,output) private(i)
  do i=1,n
    output(i)=numerical_work(family,a(i),b(i))
  enddo
  !$omp end parallel do
endif
end subroutine

subroutine fort_numerical_fortran_identity_v2(version,options,capacity) bind(c)
integer(c_int),value::capacity
character(c_char),intent(out)::version(*),options(*)
character(:),allocatable::v,o
integer::i
v=compiler_version()
o=compiler_options()
do i=1,min(len(v),capacity-1)
  version(i)=v(i:i)
enddo
version(min(len(v),capacity-1)+1)=c_null_char
do i=1,min(len(o),capacity-1)
  options(i)=o(i:i)
enddo
options(min(len(o),capacity-1)+1)=c_null_char
end subroutine
end module
