"""Generic host preparation, guarded arguments, and shared GPU intervals."""


SOURCE = """module prepared_case
contains
subroutine advance(a,b,configuration,n,scale,unused)
real(8),intent(inout)::a(:)
real(8),intent(in)::b(:)
integer,intent(in)::configuration(:),n,unused
real(8),intent(in)::scale
integer::i,limit
real(8)::coefficient
limit=configuration(1)
coefficient=scale/2.0_8
if(limit>0) then
 do i=1,n
  a(i+1)=b(i+1)*coefficient+real(i,8)
 end do
end if
limit=configuration(2)
coefficient=scale/4.0_8
if(limit>0) then
 do i=1,n
  a(i+1)=a(i+1)+b(i+1)*coefficient
 end do
end if
end subroutine
end module
"""
