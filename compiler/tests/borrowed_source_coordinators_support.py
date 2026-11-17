"""Reached source children share ownership without cloning persistent state."""



from compiler.tests.test_source_scopes import PROGRAM


def source():
    return PROGRAM.replace("subroutine step(a,b,out,n)", """subroutine child(a,b,out,n)
real(8),intent(in)::a(:)
real(8),intent(inout)::b(:),out(:)
integer,intent(inout)::n
logical::choose
call producer(a,b,n)
choose=n>2
if(choose) then
b(1)=b(1)+5.d0
else
call transform(b)
endif
n=n-1
call consumer(a,b,out,n)
end subroutine
subroutine step(a,b,out,n)""").replace(
        "integer,intent(in)::n\ncall producer(a,b,n)\ncall transform(b)\ncall consumer(a,b,out,n)",
        "integer,intent(inout)::n\ncall child(a,b,out,n)")
