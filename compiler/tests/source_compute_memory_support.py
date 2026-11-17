"""Generated source queries keep memory applicability separate from effects.

Only the emitted metadata, memory helpers and public query are compiled here.
The real reference runtime validates their effects; no numerical worker, CUDA
compiler, GPU initialization or device operation participates in these tests.
"""




ALIASES = """module distinct_inputs
contains
subroutine evaluate(a,b,out,n)
real(8),intent(in)::a(:),b(:)
real(8),intent(out)::out(:)
integer,intent(in)::n
integer::i
do i=1,n
out(i)=sqrt(a(i)*b(i)+1d0)+cos(a(i))
enddo
end subroutine
end module
"""

INDIRECT = """module unknown_read_footprint
contains
subroutine evaluate(out,b,n)
real(8),intent(out)::out(:)
real(8),intent(in)::b(:)
integer,intent(in)::n
integer::i
do i=1,n
out(i)=sqrt(b(i*i)*b(i*i)+1d0)+cos(b(i*i))
enddo
end subroutine
end module
"""

POINTWISE = """module total_memory_query
contains
subroutine evaluate(a,b,out,n)
real(8),intent(in)::a(:),b(:)
real(8),intent(out)::out(:)
integer,intent(in)::n
integer::i
do i=1,n
out(i)=2d0*a(i)+b(i)
enddo
end subroutine
end module
"""


def function_text(source, start):
    """Select complete compiler-emitted functions, including nested lambdas."""
    first = source.index("{", start)
    depth = 1
    end = first + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]
