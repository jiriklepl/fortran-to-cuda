"""Original source control continues once after a reached resident escape."""

from hashlib import sha256

from compiler.driver.options import CompilerOptions
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_source_scopes import FACT

SOURCE = """module local_owner
implicit none
integer::visits=0
contains
subroutine step(a,b,out,n,escape)
real(8),intent(in)::a(-2:)
real(8),intent(inout)::b(-2:),out(-2:)
integer,intent(in)::n
logical,intent(in)::escape
integer::i
visits=visits+1
do i=-2,n-3
b(i)=2*a(i)+real(i,8)
enddo
if(escape) then
call opaque(b,n)
endif
do i=-2,n-3
out(i)=a(i)+b(i)
enddo
end subroutine
end module
"""

INTERNAL_SOURCE = SOURCE.replace('if(escape) then\ncall opaque(b,n)\nendif',
    'call manage()\ncall manage()').replace('end subroutine\nend module', '''contains
subroutine manage()
integer,save::times=0
integer::j
times=times+1
visits=visits+times
if(escape) then
call opaque(b,n)
endif
do j=-2,n-3
b(j)=b(j)+a(j)
enddo
end subroutine
end subroutine
end module''')

ASSOCIATE_SOURCE = SOURCE.replace('visits=visits+1',
    'associate(count=>visits)\ncount=count+1').replace('end subroutine\nend module',
    'end associate\nend subroutine\nend module')

GUARDED_CHILD_SOURCE = SOURCE.replace('subroutine step', '''subroutine adjust(x,n)
real(8),intent(inout)::x(-2:)
integer,intent(in)::n
integer::j
visits=visits+11
do j=-2,n-3
x(j)=x(j)+7
enddo
end subroutine
subroutine step''', 1).replace('if(escape) then\ncall opaque(b,n)\nendif', 'if(escape) call adjust(b,n)')

PROTECTED_SOURCE = '''module readonly_controls
integer,protected::coefficients(3)=[1,2,3]
end module
''' + SOURCE.replace('module local_owner\nimplicit none',
    'module local_owner\nuse readonly_controls,only:coefficients\nimplicit none').replace(
    '2*a(i)+real(i,8)', '2*a(i)+real(i,8)+real(coefficients(2),8)')

MODULE_SOURCE = '''module stage_child
implicit none
integer::child_visits=0
contains
subroutine adjust(x,n,escape)
real(8),intent(inout)::x(-4:)
integer,intent(in)::n
logical,intent(in)::escape
integer::j
child_visits=child_visits+1
do j=-4,n-5
x(j)=x(j)+real(j,8)
enddo
if(escape) then
call opaque(x,n)
endif
end subroutine
end module
''' + SOURCE.replace('module local_owner\nimplicit none',
    'module local_owner\nuse stage_child, only: adjust\nimplicit none').replace(
        'if(escape) then\ncall opaque(b,n)\nendif', 'call adjust(b,n,escape)\ncall adjust(b,n,.false.)')

PAYLOAD_SOURCE = SOURCE.replace('if(escape) then', 'if(b(-2)>10.d0) then')

STRIDED_MODULE_SOURCE = MODULE_SOURCE.replace('intent(inout)::x(-4:)',
    'contiguous,intent(inout)::x(-4:)').replace('b(i)=2*a(i)+real(i,8)', 'out(i)=2*a(i)+real(i,8)')

RETURN_MODULE_SOURCE = MODULE_SOURCE.replace('child_visits=child_visits+1',
    'child_visits=child_visits+1\nif (.not.escape) return').replace('call opaque(x,n)', 'return')

FIRST_GUARD_MODULE_SOURCE = MODULE_SOURCE.replace('child_visits=child_visits+1',
    'if(x(-4)>0.d0) child_visits=child_visits+1')

FIRST_GUARD_SOURCE = SOURCE.replace('visits=visits+1', 'if(b(-2)<0.d0) visits=visits+1')

ALLOCATION_GUARD_SOURCE = SOURCE.replace('integer::visits=0',
    'integer::visits=0\nreal(8),allocatable::spare(:)').replace('if(escape) then', 'if(allocated(spare)) then')

NATIVE_ATOMIC_SOURCE = SOURCE.replace('if(escape) then\ncall opaque(b,n)\nendif',
    '!$omp parallel private(i)\n' + ('!$omp do\ndo i=-2,n-3\n'
    'b(i)=b(i)+a(i)+b(i)+a(i)+sum(a)\nenddo\n!$omp end do\n')*24 + '!$omp end parallel')

GUARDED_NATIVE_SOURCE = SOURCE.replace('integer::visits=0', '''integer::visits=0
type cell
integer::coordinate
end type
type(cell),allocatable::metadata(:)''').replace('if(escape) then\ncall opaque(b,n)\nendif',
    'call guarded(b,metadata,transform)\ncall guarded(out,metadata,transform)').replace(
    'end subroutine\nend module', '''contains
pure real(8) function transform(x)
real(8),intent(in)::x
transform=x+2
end function
subroutine guarded(x,items,fun)
real(8),intent(inout)::x(:)
type(cell),allocatable,intent(in)::items(:)
procedure(transform)::fun
integer::j
if(allocated(items)) then
do j=1,size(items)
x(items(j)%coordinate)=fun(x(items(j)%coordinate))
enddo
endif
end subroutine
end subroutine
end module''')

NATIVE_METADATA_SOURCE = SOURCE.replace('integer::visits=0', '''integer::visits=0
type cell
integer::coordinate
real(8)::weight
end type
type(cell),allocatable,target::metadata(:)''').replace('if(escape) then\ncall opaque(b,n)\nendif', '''if(allocated(metadata)) then
do i=1,size(metadata)
b(metadata(i)%coordinate)=b(metadata(i)%coordinate)+metadata(i)%weight
enddo
endif''')

METADATA_ALIAS_SOURCE = NATIVE_METADATA_SOURCE.replace('type cell\ninteger::coordinate',
    'type,bind(C)::cell').replace('do i=1,size(metadata)', 'do i=1,n').replace('metadata(i)%coordinate', 'i-3')

NESTED_MODULE_SOURCE = '''module deep_child
implicit none
real(8)::weights(8)=1.d0
contains
subroutine leaf(y,n,escape)
real(8),intent(inout)::y(-6:)
integer,intent(in)::n
logical,intent(in)::escape
integer::k
do k=-6,n-7
y(k)=y(k)+weights(1)
enddo
if(escape) then
call opaque(y,n)
endif
end subroutine
end module
''' + MODULE_SOURCE.replace('module stage_child\nimplicit none',
    'module stage_child\nuse deep_child,only:leaf\nimplicit none').replace(
        'call opaque(x,n)', 'call leaf(x,n,escape)')


def emit(directory, source=SOURCE, *, captures=None, policy='sections', profile=None):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/'owner.f90'
    path.write_text(source)
    facts = {'schema_version':1, 'participation':'serial',
             'sources':{str(path):sha256(path.read_bytes()).hexdigest()},
             'captures':{'argument::'+name:FACT for name in ('a','b','out')}}
    facts['captures'].update(captures or {})
    builder = ScopeBuilder([path], 'local_owner::step', facts=facts, options=CompilerOptions(),
                           config=OffloadConfig(policy=policy,profile=profile,scope_execution='reached'))
    outputs, report = builder.run()
    return path, outputs, report
