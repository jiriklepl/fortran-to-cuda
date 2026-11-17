"""Source-proved module allocations keep native effects in coherent scopes."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = """module original
implicit none
real(8),allocatable,target::field(:)
contains
subroutine produce(a,b,n)
real(8),contiguous,intent(in)::a(:)
real(8),contiguous,intent(inout)::b(:)
integer,intent(in)::n
integer::i
do i=2,n-1
b(i)=a(i-1)+2*a(i)+a(i+1)+real(i,8)
enddo
end subroutine
subroutine consume(a,b,out,n)
real(8),contiguous,intent(in)::a(:),b(:)
real(8),contiguous,intent(inout)::out(:)
integer,intent(in)::n
integer::i
do i=2,n-1
out(i)=a(i)+b(i)
enddo
end subroutine
subroutine native_hidden()
if(allocated(field)) then
 if(size(field)>2) then
  field(lbound(field,1)+1)=field(lbound(field,1)+1)+sum(field)+real(lbound(field,1),8)
 endif
endif
end subroutine
subroutine produce_team(a,b,n)
real(8),contiguous,intent(in)::a(:)
real(8),contiguous,intent(inout)::b(:)
integer,intent(in)::n
integer::i
!$omp do
do i=2,n-1
b(i)=a(i-1)+2*a(i)+a(i+1)+real(i,8)
enddo
!$omp end do
end subroutine
subroutine consume_team(a,b,out,n)
real(8),contiguous,intent(in)::a(:),b(:)
real(8),contiguous,intent(inout)::out(:)
integer,intent(in)::n
integer::i
!$omp do
do i=2,n-1
out(i)=a(i)+b(i)
enddo
!$omp end do
end subroutine
subroutine native_hidden_team(n)
integer,intent(in)::n
integer::i
!$omp do
do i=1,min(1,n)
if(n>2) field(lbound(field,1)+1)=field(lbound(field,1)+1)+sum(field)+real(lbound(field,1),8)
enddo
!$omp end do
end subroutine
subroutine step_serial(a,out,n)
real(8),contiguous,intent(in)::a(:)
real(8),contiguous,intent(inout)::out(:)
integer,intent(in)::n
if(allocated(field)) then
 call produce(a,field,n)
 call native_hidden()
 call consume(a,field,out,n)
else
 call native_hidden()
endif
end subroutine
subroutine step_team(a,out,n)
real(8),contiguous,intent(in)::a(:)
real(8),contiguous,intent(inout)::out(:)
integer,intent(in)::n
call produce_team(a,field,n)
call native_hidden_team(n)
call consume_team(a,field,out,n)
end subroutine
subroutine qualified(a,out,n)
real(8),contiguous,intent(in)::a(:)
real(8),contiguous,intent(inout)::out(:)
integer,intent(in)::n
if(.not.allocated(field)) then
 call native_hidden()
 return
endif
!$omp parallel default(none) shared(a,out,n,field) num_threads(4)
call step_team(a,out,n)
!$omp end parallel
end subroutine
end module
"""

PARTIAL_SOURCE = SOURCE.replace("""subroutine native_hidden()
if(allocated(field)) then
 if(size(field)>2) then
  field(lbound(field,1)+1)=field(lbound(field,1)+1)+sum(field)+real(lbound(field,1),8)
 endif
endif
end subroutine""", """subroutine native_hidden()
integer::i
if(allocated(field)) then
 if(size(field)>2) then
  do i=1,ubound(field,1)
   field(i)=100.d0+real(i,8)
  enddo
 endif
endif
end subroutine""")

DRIVER = """program verify
use original
use omp_lib,only:omp_set_dynamic
implicit none
real(8),allocatable::a(:),out(:),expected(:)
integer,parameter::sizes(3)=[0,19,27]
integer::shape,repeat,n,lower,i,index
real(8)::wanted
character(16)::path,mode
call get_command_argument(1,path)
call get_command_argument(2,mode)
call omp_set_dynamic(.false.)
if(mode=='unallocated') then
 allocate(a(-4:0),out(-4:0))
 a=3.d0
 out=-70.d0
 if(path=='serial'.or.path=='serial_partial') then
  call step_serial(a,out,5)
 else
  call qualified(a,out,5)
 endif
 if(allocated(field)) error stop 'native guard allocated its capture'
 if(any(a/=3.d0).or.any(out/=-70.d0)) error stop 'native guarded field mismatch'
 print *, 'UNALLOCATED_OK'
 stop
endif
do shape=1,size(sizes)
 n=sizes(shape)
 do repeat=1,2
  lower=-4-3*shape-repeat
  allocate(a(lower:lower+n-1),field(lower:lower+n-1),out(lower:lower+n-1),expected(lower:lower+n-1))
  do i=1,n
   a(lower+i-1)=real(i,8)*0.25d0+repeat
  enddo
  field=-50.d0
  out=-70.d0
  expected=field
  do i=2,n-1
   index=lower+i-1
   expected(index)=a(index-1)+2*a(index)+a(index+1)+real(i,8)
  enddo
  if(path=='serial_partial') then
   do i=1,ubound(expected,1)
    expected(i)=100.d0+real(i,8)
   enddo
  else if(n>2) then
   expected(lower+1)=expected(lower+1)+sum(expected)+real(lbound(field,1),8)
  endif
  if(path=='serial'.or.path=='serial_partial') then
   call step_serial(a,out,n)
  else
   call qualified(a,out,n)
  endif
  do i=1,n
   index=lower+i-1
   if(a(index)/=real(i,8)*0.25d0+repeat) error stop 'immutable input changed'
   if(field(index)/=expected(index)) error stop 'module field or halo disagreement'
   wanted=-70.d0
   if(i>=2.and.i<=n-1) wanted=a(index)+expected(index)
   if(out(index)/=wanted) error stop 'output or halo disagreement'
  enddo
  print *, 'CALL_OK',n,repeat,lbound(field,1)
  deallocate(a,field,out,expected)
 enddo
enddo
print *, 'FIELDS_OK'
end program
"""


def _guard():
    available = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
                     if line.startswith("MemAvailable:"))
    assert available >= 3 * 1024**3, "memory guard stopped module-allocation acceptance before a new process"


def _run(command, directory, *, env=None, timeout=180):
    _guard()
    result = subprocess.run(command, cwd=directory, env=env, capture_output=True, text=True, timeout=timeout)
    with (directory / "commands.jsonl").open("a") as stream:
        stream.write(json.dumps({"argv": list(map(str, command)), "returncode": result.returncode,
                                 "stdout": result.stdout, "stderr": result.stderr}) + "\n")
    assert result.returncode == 0, result.stdout + result.stderr
    return result
