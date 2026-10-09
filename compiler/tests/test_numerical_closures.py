"""Bounded numerical helper closures lower into the existing scalar IR."""

import os
import shutil
import subprocess

import pytest

from compiler.analysis import build_execution_plan
from compiler.frontend import discover_file, lower_file
from compiler.ir import Assignment, Binary, CompilationError, If, Literal, Loop, Reference, walk_expr
from compiler.emission import generate_sources, read_common_header
from compiler.tests.test_native import cuda_device as cuda_device


def source(tmp_path, routines, module="renamed_math"):
    path = tmp_path / "closures.f90"
    path.write_text(f"module {module}\nimplicit none\ncontains\n{routines}\nend module\n")
    return path


CLOSURE = """
subroutine advance(a,b,n,shift)
real(8),intent(in)::a(:),shift
real(8),intent(out)::b(:)
integer,intent(in)::n
integer::i
real(8)::p(-1:1,2),q
do i=1,n
call tensor(p,i)
call measure(q,p)
b(i)=q
enddo
contains
pure subroutine tensor(g,k)
real(8),intent(out)::g(0:2,2)
integer,intent(in)::k
g(0,1)=a(k)+shift
g(1,1)=a(k)-shift
g(2,1)=a(k)*shift
g(0,2)=2.0_8
g(1,2)=3.0_8
g(2,2)=4.0_8
end subroutine
pure subroutine measure(value,g)
real(8),intent(out)::value
real(8),intent(in)::g(-1:1,2)
real(8),parameter::factor=1.0_8/2
value=dot_product(g(:,1),g(:,2))*factor+determinant(g)
end subroutine
pure function determinant(g) result(value)
real(8),intent(in)::g(3,2)
real(8)::value
value=g(1,1)**2-g(2,1)*g(3,1)
end function
end subroutine
"""


def test_internal_closure_private_arrays_and_formal_rebasing(tmp_path):
    function = lower_file(source(tmp_path, CLOSURE), "advance")
    assert all(symbol.rank == 0 for symbol in function.symbols if not symbol.parameter)
    loop, = function.body.statements
    assert isinstance(loop, Loop)
    plan = build_execution_plan(function)
    assert len(plan.regions) == 1
    assert {symbol.name for symbol in plan.regions[0].write_symbols} == {"b"}
    assert {symbol.name for symbol in plan.regions[0].read_symbols} == {"a"}
    assert any("determinant" in " ".join(statement.location.call_stack)
               for statement in loop.body.statements)
    records = discover_file(source(tmp_path, CLOSURE))
    assert [(item.name, item.lowerable) for item in records] == [("advance", True)]


def test_pure_module_functions_return_values_and_scalar_expression_actuals(tmp_path):
    path = source(tmp_path, """
pure real(8) function square(x)
real(8),intent(in)::x
square=x*x
end function
subroutine advance(a,n)
real(8),intent(inout)::a(:)
integer,intent(in)::n
integer::i
do i=1,n
a(i)=square(a(i)+1.0_8)+square(a(i)-1.0_8)
enddo
end subroutine
""")
    function = lower_file(path, "advance")
    assert len(build_execution_plan(function).regions) == 1
    calls = [statement for statement in function.body.statements[0].body.statements
             if statement.location.call_stack]
    assert len(calls) == 2
    assert calls[0].target.symbol != calls[1].target.symbol


def test_function_preludes_remain_inside_protected_else_branch(tmp_path):
    path = source(tmp_path, """
subroutine advance(a,n)
real(8),intent(inout)::a(:)
integer,intent(in)::n
if(n<0)then
a(1)=0.0_8
else if(positive(a(1)))then
a(1)=1.0_8
endif
contains
pure logical function positive(x)
real(8),intent(in)::x
positive=x>0.0_8
end function
end subroutine
""")
    function = lower_file(path, "advance")
    outer, = function.body.statements
    assert isinstance(outer, If)
    assert len(outer.else_body.statements) == 3
    assert isinstance(outer.else_body.statements[-1], If)
    build_execution_plan(function)


def test_negative_private_bounds_inquiries_and_strided_vector_sections(tmp_path):
    path = source(tmp_path, """
subroutine advance(a,n)
real(8),intent(out)::a(:)
integer,intent(in)::n
integer::i
real(8)::g(-2:1)
do i=1,n
g(-2)=1.0_8
g(-1)=2.0_8
g(0)=3.0_8
g(1)=4.0_8
a(i)=dot_product(g(-2:1:2),g(1:-2:-2))+lbound(g,1)+ubound(g,1)+size(g)
enddo
end subroutine
""")
    function = lower_file(path, "advance")
    value = function.body.statements[0].body.statements[-1].value
    assert all(not getattr(getattr(node, "symbol", None), "rank", 0) for node in walk_expr(value))
    assert Literal("-2", function.parameters[-1].dtype) in tuple(walk_expr(value))
    assert len(build_execution_plan(function).regions) == 1


@pytest.mark.parametrize("change,message", [
    (lambda text: text.replace("g(2,2)=4.0_8", ""), "complete ordered definitions"),
    (lambda text: text.replace("g(2,2)=4.0_8", "g(2,2)=g(2,2)+1.0_8"), "read before"),
    (lambda text: text.replace("g(0,1)=a(k)+shift", "a(k)=1.0_8"), "cannot write INTENT"),
    (lambda text: text.replace("value=g(1,1)**2-g(2,1)*g(3,1)", "value=determinant(g)"), "recursive"),
    (lambda text: text.replace("g(0,1)=a(k)+shift", "print *, k"), "Print_Stmt"),
    (lambda text: text.replace("g(0,1)=a(k)+shift", "g(k,1)=a(k)+shift"), "constant subscripts"),
])
def test_unsafe_or_unproved_helper_closures_are_rejected(tmp_path, change, message):
    with pytest.raises(CompilationError, match=message):
        build_execution_plan(lower_file(source(tmp_path, change(CLOSURE)), "advance"))


def test_pure_writable_arguments_cannot_alias(tmp_path):
    path = source(tmp_path, """
subroutine advance(a,n)
real(8),intent(out)::a(:)
integer,intent(in)::n
integer::i
real(8)::t
do i=1,n
t=1.0_8
call mutate(t,t)
a(i)=t
enddo
contains
pure subroutine mutate(x,y)
real(8),intent(inout)::x,y
x=y+1.0_8
y=x+1.0_8
end subroutine
end subroutine
""")
    with pytest.raises(CompilationError, match="overlapping writable storage"):
        lower_file(path, "advance")


def test_private_fixed_array_budget_is_bounded(tmp_path):
    path = source(tmp_path, """
subroutine advance(a)
real(8),intent(out)::a(:)
real(8)::huge(257)
a(1)=0.0_8
end subroutine
""")
    with pytest.raises(CompilationError, match="256-element"):
        lower_file(path, "advance")


def test_bounded_private_index_loops_unroll_in_fortran_order(tmp_path):
    path = source(tmp_path, """
subroutine advance(a,n)
real(8),intent(out)::a(:)
integer,intent(in)::n
integer::i,j,k
real(8)::g(-1:1,2)
do i=1,n
do k=2,1,-1
do j=-1,1
g(j,k)=real(j+3*k,8)
enddo
enddo
a(i)=dot_product(g(:,1),g(:,2))+j+k
enddo
end subroutine
""")
    function = lower_file(path, "advance")
    loop, = function.body.statements
    assert not any(isinstance(statement, Loop) for statement in loop.body.statements)
    assert len(build_execution_plan(function).regions) == 1


def test_empty_private_vector_dot_product_is_zero(tmp_path):
    path = source(tmp_path, """
subroutine advance(a)
real(8),intent(out)::a(:)
real(8)::empty(-2:-3)
a(1)=dot_product(empty,empty)+lbound(empty,1)+ubound(empty,1)+size(empty)
end subroutine
""")
    function = lower_file(path, "advance")
    value = function.body.statements[0].value
    assert not any(isinstance(node, Reference) for node in walk_expr(value))
    build_execution_plan(function)


SPECTRAL = """
subroutine advance(a,b,n)
real(WP),intent(in)::a(:,:)
real(WP),intent(out)::b(:,:)
integer,intent(in)::n
integer::i
real(WP)::g(-1:1,-1:1),x,y,z,d
do i=1,n
call gradient(g,i)
call spectrum(x,y,z,g)
d=z*(x-y)*(y-z)/(x*x)
b(1,i)=x
b(2,i)=y
b(3,i)=z
b(4,i)=max(0.0_WP,d)
enddo
contains
pure subroutine gradient(g,k)
real(WP),intent(out)::g(0:2,0:2)
integer,intent(in)::k
g(0,0)=a(1,k)
g(1,0)=a(2,k)
g(2,0)=a(3,k)
g(0,1)=a(4,k)
g(1,1)=a(5,k)
g(2,1)=a(6,k)
g(0,2)=a(7,k)
g(1,2)=a(8,k)
g(2,2)=a(9,k)
end subroutine
pure subroutine spectrum(x,y,z,t)
real(WP),intent(out)::x,y,z
real(WP),intent(in)::t(3,3)
real(WP)::h(-1:1,-1:1),trace,trace2,c1,c2,c3,p,q,angle,c
real(WP),parameter::pi=3.1415926535897932384626433832795_WP
h(-1,-1)=dot_product(t(:,1),t(:,1))
h(-1,0)=dot_product(t(:,1),t(:,2))
h(-1,1)=dot_product(t(:,1),t(:,3))
h(0,-1)=h(-1,0)
h(0,0)=dot_product(t(:,2),t(:,2))
h(0,1)=dot_product(t(:,2),t(:,3))
h(1,-1)=h(-1,1)
h(1,0)=h(0,1)
h(1,1)=dot_product(t(:,3),t(:,3))
trace=h(-1,-1)+h(0,0)+h(1,1)
trace2=dot_product(h(:,-1),h(:,-1))+dot_product(h(:,0),h(:,0))+dot_product(h(:,1),h(:,1))
c1=trace
c2=(trace**2-trace2)/2
c3=det(h)
p=max((c1**2)/9-c2/3,0.0_WP)
q=(c1**3)/27-c1*c2/6+c3/2
c=q/sqrt(p**3)
c=max(-1.0_WP,min(1.0_WP,c))
angle=acos(c)/3
c=2*sqrt(p)
x=sqrt(c1/3+c*cos(angle))
y=sqrt(max(c1/3-c*cos(pi/3+angle),0.0_WP))
z=sqrt(max(c1/3-c*cos(pi/3-angle),0.0_WP))
end subroutine
pure function det(t) result(value)
real(WP),intent(in)::t(3,3)
real(WP)::value
value=t(1,1)*t(2,2)*t(3,3)-t(1,1)*t(2,3)*t(3,2)-t(1,2)*t(2,1)*t(3,3)&
      +t(1,2)*t(2,3)*t(3,1)+t(1,3)*t(2,1)*t(3,2)-t(1,3)*t(2,2)*t(3,1)
end function
end subroutine
"""


@pytest.mark.native
@pytest.mark.parametrize("precision", [4, 8])
@pytest.mark.parametrize("backend", ["cpp", pytest.param("cuda", marks=[pytest.mark.cuda])])
def test_spectral_helper_closure_executes_like_native_fortran(tmp_path, precision, backend, request):
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not fortran or not shutil.which("g++"):
        pytest.skip("native compilers are unavailable")
    if backend == "cuda":
        request.getfixturevalue("cuda_device")
    path = source(tmp_path, SPECTRAL.replace("WP", str(precision)))
    function = lower_file(path, "advance")
    plan = build_execution_plan(function)
    assert len(plan.regions) == 1
    generated = generate_sources(function, plan)
    (tmp_path / "implementation.cpp").write_text(generated.cpp)
    (tmp_path / "interface.f90").write_text(generated.fortran)
    (tmp_path / "common_functions.cuh").write_text(read_common_header())
    driver = f"""program check
use renamed_math
use ieee_arithmetic
use ieee_exceptions
implicit none
real({precision})::a(9,21),b(4,21)
integer::i,j
logical::before,after
call ieee_get_halting_mode(ieee_invalid,before)
call ieee_set_halting_mode(ieee_invalid,.false.)
call ieee_set_halting_mode(ieee_divide_by_zero,.false.)
a=0
a(1,2)=1
a(5,2)=1
a(9,2)=1
a(1,3)=-0.0_{precision}
do i=4,19
do j=1,9
a(j,i)=sin(real(11*i+7*j+j*j*i,{precision}))*real(0.4,{precision})
enddo
enddo
a(1,20)=ieee_value(0.0_{precision},ieee_positive_inf)
a(1,21)=ieee_value(0.0_{precision},ieee_negative_inf)
call advance(a,b,21)
print *,b
call ieee_set_halting_mode(ieee_invalid,before)
call ieee_get_halting_mode(ieee_invalid,after)
print *,merge(1,0,before.eqv.after)
end program
"""
    (tmp_path / "driver.f90").write_text(driver)

    def run(args):
        completed = subprocess.run(args, cwd=tmp_path, text=True, capture_output=True, timeout=45,
                                   env={**os.environ, "OMP_NUM_THREADS": "4"})
        assert completed.returncode == 0, completed.stdout + completed.stderr
        return completed.stdout

    run([fortran, "-O3", "-ffp-contract=off", str(path), "driver.f90", "-o", "native"])
    expected = [float(value) for value in run([str(tmp_path / "native")]).split()]
    if backend == "cpp":
        run(["g++", "-O3", "-ffp-contract=off", "-std=c++17", "-fopenmp", "-c", "implementation.cpp"])
        run([fortran, "-O3", "-ffp-contract=off", "-fopenmp", "interface.f90", "driver.f90",
             "implementation.o", "-lstdc++", "-o", "generated"])
    else:
        (tmp_path / "implementation.cu").write_text(generated.cuda)
        run(["nvcc", "-O3", "--fmad=false", "-std=c++17", "-c", "implementation.cu", "-o", "implementation.o"])
        run([fortran, "-O3", "-ffp-contract=off", "-c", "interface.f90", "driver.f90"])
        run(["nvcc", "implementation.o", "interface.o", "driver.o", "-lgfortran", "-o", "generated"])
    actual = [float(value) for value in run([str(tmp_path / "generated")]).split()]
    assert actual[-1] == expected[-1] == 1
    tolerance = 2e-5 if precision == 4 else 2e-12
    assert actual[:-1] == pytest.approx(expected[:-1], rel=tolerance, abs=tolerance, nan_ok=True)
