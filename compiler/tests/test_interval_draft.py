"""One source inventory precedes cross-procedure ownership and placement."""

from hashlib import sha256

from compiler.driver.options import CompilerOptions
from compiler.offload.config import OffloadConfig
from compiler.scopes.interval_draft import prepare_interval_draft
from compiler.scopes.lexical import LexicalOwner
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_source_scopes import FACT

SOURCE = """module interval_fields
implicit none
real(8)::old(-1:16),viscosity(-1:16),star(-1:16),out(-1:16)
integer::visits=0
contains
subroutine advance(n,correcting)
integer,intent(in)::n
logical,intent(in)::correcting
call producer(old,viscosity,n)
call momentum(old,viscosity,star,out,n,correcting)
end subroutine
subroutine producer(a,v,n)
real(8),intent(in)::a(-1:)
real(8),intent(inout)::v(-1:)
integer,intent(in)::n
integer::i
!$omp parallel do private(i) schedule(runtime)
do i=0,n-1
v(i)=2.d0*a(i)
enddo
!$omp end parallel do
end subroutine
subroutine momentum(a,v,s,o,n,correcting)
real(8),intent(in)::a(-1:)
real(8),intent(inout)::v(-1:),s(-1:),o(-1:)
integer,intent(in)::n
logical,intent(in)::correcting
integer::i
visits=visits+1
if(correcting) then
v(0)=v(0)+a(0)
endif
!$omp parallel do private(i) schedule(runtime)
do i=0,n-1
s(i)=v(i)+a(i)
enddo
!$omp end parallel do
!$omp parallel do private(i) schedule(runtime)
do i=0,n-1
o(i)=o(i)+s(i)
enddo
!$omp end parallel do
end subroutine
end module
"""


def generate(directory, *, source=SOURCE, config=None):
    path = directory / "interval.f90"
    path.write_text(source)
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"interval_fields::" + name: dict(FACT)
                          for name in ("old", "viscosity", "star", "out")}}
    config = config or OffloadConfig(policy="sections", scope_execution="reached")
    builder = ScopeBuilder([path], "interval_fields::advance", facts=facts,
                           options=CompilerOptions(gpu_policy=config.policy), config=config)
    outputs, report = builder.run()
    return path, builder, outputs, report


def draft_only(directory, source):
    path = directory / "interval.f90"
    path.write_text(source)
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"interval_fields::" + name: dict(FACT)
                          for name in ("old", "viscosity", "star", "out")}}
    builder = ScopeBuilder([path], "interval_fields::advance", facts=facts,
        options=CompilerOptions(gpu_policy="sections"),
        config=OffloadConfig(policy="sections", scope_execution="reached"))
    owner = LexicalOwner(builder)
    return prepare_interval_draft(owner, tuple(builder.entry.execution.content))


def routing_source():
    helpers = """subroutine native_touch()
visits=visits+1
end subroutine
subroutine declined(a,s,n)
real(8),intent(in)::a(-1:)
real(8),intent(inout)::s(-1:)
integer,intent(in)::n
integer::i
!$omp parallel do private(i) schedule(runtime)
do i=0,n-1
s(i)=s(i)+sum(a)
enddo
!$omp end parallel do
end subroutine
"""
    return SOURCE.replace("call momentum(old,viscosity,star,out,n,correcting)",
        "call native_touch()\nif(correcting) call declined(old,star,n)\n"
        "call unknown()\ncall momentum(old,viscosity,star,out,n,correcting)").replace(
            "end module", helpers + "end module")
