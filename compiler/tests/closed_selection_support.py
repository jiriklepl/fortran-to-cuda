"""Complete original numerical groups use one checked reached owner path."""

from hashlib import sha256

from compiler.driver.options import CompilerOptions
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_source_scopes import FACT

SOURCE = """module selected_groups
implicit none
integer:: tiles(3)=[4,3,2], nx=15, ny=7, nz=3, order=2
contains
subroutine advance(a,b)
real(8),intent(in)::a(0:,0:,0:)
real(8),intent(out)::b(0:,0:,0:)
integer:: tx,ty,tz,i,j,k,bi,bj,bk
real(8):: coefficient
tx=tiles(1)
ty=tiles(2)
tz=tiles(3)
coefficient=2.0_8
if(order==4) then
!$omp parallel do private(i,j,k,bi,bj,bk) collapse(3) schedule(runtime)
do bk=0,nz,tz
do bj=0,ny,ty
do bi=0,nx,tx
do k=bk,min(bk+tz-1,nz)
do j=bj,min(bj+ty-1,ny)
do i=bi,min(bi+tx-1,nx)
b(i,j,k)=coefficient*a(i,j,k)+4.0_8
enddo
enddo
enddo
enddo
enddo
enddo
!$omp end parallel do
else
!$omp parallel do private(i,j,k,bi,bj,bk) collapse(3) schedule(runtime)
do bk=0,nz,tz
do bj=0,ny,ty
do bi=0,nx,tx
do k=bk,min(bk+tz-1,nz)
do j=bj,min(bj+ty-1,ny)
do i=bi,min(bi+tx-1,nx)
b(i,j,k)=coefficient*a(i,j,k)
enddo
enddo
enddo
enddo
enddo
enddo
!$omp end parallel do
endif
end subroutine
end module
"""


def builder(path, source=SOURCE, *, config=None, options=None, captures=None):
    path.write_text(source)
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"argument::a": FACT, "argument::b": {**FACT, "initialized": "none"},
                          "selected_groups::tiles": FACT}}
    facts["captures"].update(captures or {})
    return ScopeBuilder([path], "selected_groups::advance", facts=facts,
                        options=options or CompilerOptions(gpu_policy="sections"),
                        config=config or OffloadConfig(policy="sections"))
