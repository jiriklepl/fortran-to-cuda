program probe
use omp_lib
use iso_c_binding, only:c_int64_t,c_int
implicit none
interface
subroutine set_c() bind(C)
end subroutine
function get_c() bind(C) result(kind)
import c_int64_t
integer(c_int64_t)::kind
end function
subroutine get_abi(kind,chunk) bind(C,name='omp_get_schedule')
import c_int
integer(c_int),intent(out)::kind,chunk
end subroutine
end interface
integer(omp_sched_kind)::kind
integer::chunk
integer(c_int)::abi_kind,abi_chunk
call omp_set_schedule(omp_sched_static,0)
call omp_set_schedule(ibset(omp_sched_static,31),0)
call omp_get_schedule(kind,chunk)
print *, 'Fortran setter/getter',kind,chunk,'C getter',get_c()
call set_c()
call omp_get_schedule(kind,chunk)
call get_abi(abi_kind,abi_chunk)
print *, 'C setter/Fortran getter',kind,chunk,'C getter',get_c()
print *, 'C ABI getter',iand(int(abi_kind,c_int64_t),int(z'FFFFFFFF',c_int64_t)),abi_chunk
end program
