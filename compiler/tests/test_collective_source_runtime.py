"""Public source companions preserve fields and use one coordinated context.

This opt-in acceptance fixture builds independent public compiler outputs. It
does not inspect compiler IR or generated CUDA to infer execution decisions.
"""

from __future__ import annotations

import json
import os
import subprocess
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SOURCE = """module operators
implicit none
contains
subroutine produce(a,b,n)
real(8),intent(in)::a(:)
real(8),intent(inout)::b(:)
integer,intent(in)::n
integer::i
!$omp do
do i=2,n-1
b(i)=2*a(i)+real(i,8)
enddo
!$omp end do
end subroutine
subroutine native_permute(b,permutation,n)
real(8),intent(inout)::b(:)
integer,intent(in)::permutation(:),n
integer::i
!$omp do
do i=2,n-1
b(permutation(i))=b(permutation(i))+1
enddo
!$omp end do
end subroutine
subroutine consume(a,b,c,n)
real(8),intent(in)::a(:),b(:)
real(8),intent(inout)::c(:)
integer,intent(in)::n
integer::i
!$omp do
do i=2,n-1
c(i)=a(i)+b(i)
enddo
!$omp end do
end subroutine
subroutine step(a,b,c,permutation,n)
real(8),intent(in)::a(:)
real(8),intent(inout)::b(:),c(:)
integer,intent(in)::permutation(:),n
call produce(a,b,n)
call consume(a,b,c,n)
end subroutine
subroutine step_native(a,b,c,permutation,n)
real(8),intent(in)::a(:)
real(8),intent(inout)::b(:),c(:)
integer,intent(in)::permutation(:),n
call produce(a,b,n)
call native_permute(b,permutation,n)
call consume(a,b,c,n)
end subroutine
end module
module callers
use operators,only:step,step_native
implicit none
contains
subroutine qualified(a,b,c,permutation,n)
real(8),allocatable,intent(in)::a(:)
real(8),allocatable,intent(inout)::b(:),c(:)
integer,allocatable,intent(in)::permutation(:)
integer,intent(in)::n
!$omp parallel default(none) shared(a,b,c,permutation,n) num_threads(4)
call step(a,b,c,permutation,n)
!$omp end parallel
end subroutine
subroutine qualified_native(a,b,c,permutation,n)
real(8),allocatable,intent(in)::a(:)
real(8),allocatable,intent(inout)::b(:),c(:)
integer,allocatable,intent(in)::permutation(:)
integer,intent(in)::n
!$omp parallel default(none) shared(a,b,c,permutation,n) num_threads(4)
call step_native(a,b,c,permutation,n)
!$omp end parallel
end subroutine
end module
"""

HIDDEN_SOURCE = SOURCE.replace("implicit none\ncontains", "implicit none\nreal(8),allocatable::scratch(:)\ncontains", 1).replace(
    "subroutine consume(", """subroutine native_guard(c,n)
real(8),intent(inout)::c(:)
integer,intent(in)::n
integer::i
!$omp do
do i=1,merge(n,0,allocated(scratch))
if(scratch(i)>0) c(i)=c(i)+scratch(i)
enddo
!$omp end do
end subroutine
subroutine consume(""", 1).replace(
    "call produce(a,b,n)\ncall consume(a,b,c,n)",
    "call produce(a,b,n)\ncall native_guard(c,n)\ncall consume(a,b,c,n)", 1).replace(
    "use operators,only:step,step_native", "use operators,only:step,step_native,scratch").replace(
    "shared(a,b,c,permutation,n)", "shared(a,b,c,permutation,n,scratch)")

DRIVER = """program verify_fields
use callers
use iso_c_binding
use omp_lib,only:omp_set_dynamic
implicit none
interface
subroutine counter_snapshot(created,registered,closed) bind(C)
import c_int64_t
integer(c_int64_t),intent(out)::created,registered,closed
end subroutine
end interface
real(8),allocatable::a(:),b(:),c(:)
integer,allocatable::permutation(:)
integer::n,repetition,shape,k,i,checks
integer,parameter::sizes(3)=[0,19,27]
integer(c_int64_t)::created,registered,closed
real(8)::wanted_a,wanted_b,wanted_c
character(16)::mode
call get_command_argument(1,mode)
call omp_set_dynamic(.false.)
checks=0
do shape=1,size(sizes)
n=sizes(shape)
do repetition=1,2
allocate(a(-2:n-3),b(-2:n-3),c(-2:n-3),permutation(-2:n-3))
do k=-2,n-3
i=k+3
a(k)=real(3*i+repetition,8)
b(k)=-50.d0
c(k)=-70.d0
permutation(k)=n-i+1
enddo
if(mode=='native') then
call qualified_native(a,b,c,permutation,n)
else if(mode=='serial') then
call step(a,b,c,permutation,n)
else
call qualified(a,b,c,permutation,n)
endif
do k=-2,n-3
i=k+3
wanted_a=real(3*i+repetition,8)
wanted_b=-50.d0
wanted_c=-70.d0
if(i>=2.and.i<=n-1) then
wanted_b=2*wanted_a+real(i,8)
if(mode=='native') wanted_b=wanted_b+1
wanted_c=wanted_a+wanted_b
endif
if(a(k)/=wanted_a.or.b(k)/=wanted_b.or.c(k)/=wanted_c) error stop 'full field or halo mismatch'
if(permutation(k)/=n-i+1) error stop 'immutable permutation changed'
enddo
deallocate(a,b,c,permutation)
checks=checks+1
enddo
enddo
call counter_snapshot(created,registered,closed)
print '(a,a,a,i0)', 'FIELDS_OK ',trim(mode),' checks=',checks
print '(a,i0,a,i0,a,i0)', 'COUNTERS created=',created,' registered=',registered,' closed=',closed
end program
"""

COUNTERS = r'''#include <atomic>
#include <cstdint>
#include "scoped_runtime.h"
static std::atomic<std::int64_t> creates{0},registrations{0},closes{0};
extern "C" int __real_fort_scope_create(int,fort_scope_t *);
extern "C" int __real_fort_scope_register(fort_scope_t,std::uint64_t,std::uint64_t,
                                         const fort_scope_layout *,int,fort_buffer_t *);
extern "C" int __real_fort_scope_register_sections(fort_scope_t,std::uint64_t,std::uint64_t,
                                                  const fort_scope_layout *,const fort_scope_section *,
                                                  std::size_t,fort_buffer_t *);
extern "C" int __real_fort_scope_close(fort_scope_t);
extern "C" int __wrap_fort_scope_create(int device,fort_scope_t *scope) {
    const int status=__real_fort_scope_create(device,scope);if(!status) ++creates;return status;
}
extern "C" int __wrap_fort_scope_register(fort_scope_t scope,std::uint64_t identity,std::uint64_t generation,
                                         const fort_scope_layout *layout,int initialized,fort_buffer_t *buffer) {
    const int status=__real_fort_scope_register(scope,identity,generation,layout,initialized,buffer);
    if(!status) ++registrations;return status;
}
extern "C" int __wrap_fort_scope_register_sections(fort_scope_t scope,std::uint64_t identity,std::uint64_t generation,
                                                  const fort_scope_layout *layout,const fort_scope_section *sections,
                                                  std::size_t count,fort_buffer_t *buffer) {
    const int status=__real_fort_scope_register_sections(scope,identity,generation,layout,sections,count,buffer);
    if(!status) ++registrations;return status;
}
extern "C" int __wrap_fort_scope_close(fort_scope_t scope) {
    const int status=__real_fort_scope_close(scope);if(!status) ++closes;return status;
}
extern "C" void counter_snapshot(std::int64_t *created,std::int64_t *registered,std::int64_t *closed) {
    *created=creates.load();*registered=registrations.load();*closed=closes.load();
}
'''

NATIVE_COUNTERS = r'''#include <cstdint>
extern "C" void counter_snapshot(std::int64_t *created,std::int64_t *registered,std::int64_t *closed) {
    *created=*registered=*closed=0;
}
'''


def _guard():
    minimum = int(os.environ.get("FORT_TEST_MIN_AVAILABLE_BYTES", "0"))
    if minimum:
        available = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
                         if line.startswith("MemAvailable:"))
        assert available >= minimum, "memory guard stopped source-team acceptance before a new process"


def _run(command, directory, *, env=None, timeout=180, expected_returncode=0):
    _guard()
    result = subprocess.run(command, cwd=directory, env=env, capture_output=True, text=True, timeout=timeout)
    if Path(directory).resolve() != ROOT:  # keep command logs in test directories, never in the checkout
        with (directory / "command-log.jsonl").open("a") as stream:
            stream.write(json.dumps({"argv": list(map(str, command)), "returncode": result.returncode,
                                     "stdout": result.stdout, "stderr": result.stderr}) + "\n")
    assert result.returncode == expected_returncode, result.stdout + result.stderr
    return result


def _facts(original, *, native=False, budget=None):
    text = original.read_text().splitlines(keepends=True)
    entry = "step_native" if native else "step"
    first = next(i for i, line in enumerate(text, 1) if line.strip() == f"call {entry}(a,b,c,permutation,n)"
                 and text[i-2].startswith("!$omp parallel"))
    stable = {"storage": "stable", "initialized": "whole", "allocation_changes": False, "escapes": False}
    facts = {"schema_version": 2, "sources": {str(original): sha256(original.read_bytes()).hexdigest()},
             "captures": {**{"argument::" + name: {**stable, "association": "shared_whole_storage",
                                                    "descriptor_uniform": True}
                              for name in ("a", "b", "c", "permutation")},
                          "argument::n": {**stable, "association": "shared_immutable_control"}},
             "participation": {"kind": "omp_full_team", "dispatch": "qualified_companion",
                               "entry": "operators::" + entry, "host_threads": 4, "expected_omp_level": 1,
                               "call_sites": [{"source": str(original),
                                               "caller": "callers::qualified_native" if native else "callers::qualified",
                                               "first_line": first, "last_line": first,
                                               "span_sha256": sha256(text[first-1].encode()).hexdigest(),
                                               "team_first_line": first-1, "team_last_line": first+1,
                                               "uniform_guard": "unconditional"}]}}
    if budget is not None:
        facts["device_budget_bytes"] = budget
    return facts
