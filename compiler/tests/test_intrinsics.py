"""Intrinsic typing and execution compared with native Fortran on each backend."""

import re
import shutil
import subprocess

import pytest

from compiler.analysis import ParallelizationError, build_execution_plan
from compiler.analysis.semantics import expression_type
from compiler.driver.options import CompilerOptions
from compiler.driver.pipeline import prepare_function
from compiler.emission import read_common_header
from compiler.frontend import lower_file
from compiler.ir import CompilationError, IntrinsicCall, Literal, ScalarType, SourceLocation
from compiler.ir.intrinsics import INTRINSIC_ARGUMENTS
from compiler.tests.test_language import lower
from compiler.tests.test_native import cuda_device as cuda_device

REQUESTED = {
    "abs",
    "min",
    "max",
    "sqrt",
    "exp",
    "log",
    "log10",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "atan2",
    "sinh",
    "cosh",
    "tanh",
    "real",
    "int",
    "nint",
    "floor",
    "ceiling",
    "mod",
    "modulo",
    "sign",
    "dim",
    "dble",
    "kind",
    "size",
    "lbound",
    "ubound",
    "epsilon",
    "tiny",
    "huge",
    "merge",
}

REAL_FUNCTIONS = (
    "abs",
    "sqrt",
    "exp",
    "log",
    "log10",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
)


def test_requested_intrinsics_are_registered():
    assert INTRINSIC_ARGUMENTS.keys() >= REQUESTED


@pytest.mark.parametrize(
    ("call", "message"),
    [
        ("exp(1)", "invalid argument types"),
        ("atan2(1.,1.d0)", "invalid argument types"),
        ("real(.true.)", "invalid argument types"),
        ("dble(.true.)", "invalid argument types"),
        ("int(.false.)", "invalid argument types"),
        ("floor(1)", "invalid argument types"),
        ("ceiling(1)", "invalid argument types"),
        ("nint(1)", "invalid argument types"),
        ("mod(1,2.)", "invalid argument types"),
        ("modulo(1.,2.d0)", "invalid argument types"),
        ("sign(1,2.)", "invalid argument types"),
        ("dim(1.,2.d0)", "invalid argument types"),
        ("merge(1,2,1)", "invalid argument types"),
        ("merge(1.,2.d0,.true.)", "invalid argument types"),
        ("real(1,kind=16)", "unsupported REAL result kind"),
        ("int(1.,kind=8)", "unsupported INT result kind"),
        ("nint(1.,kind=8)", "unsupported NINT result kind"),
        ("real(1,kind=n)", "KIND must be a constant INTEGER"),
        ("real(1,kind=4.)", "KIND must be a constant INTEGER"),
        ("epsilon(1)", "invalid argument types"),
        ("tiny(1)", "invalid argument types"),
        ("huge(.true.)", "invalid argument types"),
        ("size(n)", "requires an array"),
        ("lbound(a)", "array-valued results are unsupported"),
        ("ubound(a)", "array-valued results are unsupported"),
        ("ubound(a,dim=0)", "outside rank"),
        ("lbound(a,dim=2)", "outside rank"),
        ("size(a,dim=1.)", "dimension must be INTEGER"),
        ("size(a,kind=8)", "only default INTEGER result kind 4"),
        ("sin(a=1.)", "unknown keyword"),
        ("real(1,a=2)", "duplicate argument"),
        ("merge(1,mask=.true.)", "expects 3 arg"),
        ("real(kind=8)", "missing required argument"),
        ("mod(n,0)", "division by zero"),
        ("modulo(1,0)", "division by zero"),
        ("sign((-2147483647-1),1)", "constant expression overflows"),
        ("dim(2147483647,-1)", "constant expression overflows"),
    ],
)
@pytest.mark.parametrize("fallback", ["error", "host"])
def test_invalid_intrinsics_reject_before_execution_policy(tmp_path, call, message, fallback):
    with pytest.raises(CompilationError, match=message) as failure:
        prepare_function(lower(tmp_path, "a(1)=" + call), options=CompilerOptions(fallback=fallback))
    assert failure.value.location.path.endswith("language.f90")


def test_ir_validation_checks_conversion_kind_and_result_type():
    location = SourceLocation("intrinsics.f90", 10)
    args = (Literal("1", ScalarType.INTEGER), Literal("8", ScalarType.INTEGER))
    assert expression_type(IntrinsicCall("real", args, ScalarType.REAL), location) is ScalarType.REAL
    with pytest.raises(CompilationError, match="result type does not match"):
        expression_type(IntrinsicCall("real", args, ScalarType.REAL32), location)
    with pytest.raises(CompilationError, match="unsupported INT result kind"):
        expression_type(IntrinsicCall("int", args, ScalarType.INTEGER), location)


def test_merge_sources_still_participate_in_dependence_and_definition_checks(tmp_path):
    function = lower(tmp_path, "do i=2,n\na(i)=merge(a(i-1),a(i),i>2)\nenddo")
    with pytest.raises(ParallelizationError, match="RAW"):
        build_execution_plan(function)
    function = lower(tmp_path, "a(1)=merge(t,1.d0,n>0)", "real(8)::t")
    with pytest.raises(CompilationError, match="read before definition"):
        build_execution_plan(function)


def intrinsic_case(tmp_path):
    """Use runtime inputs, including both signs, ties, signed zero, and INT_MIN."""
    real_calls = [f"{name}(abs(x))" for name in REAL_FUNCTIONS]
    real_calls += [
        "min(x,y,0.25_KIND)",
        "max(x,y,0.25_KIND)",
        "atan2(x,y)",
        "mod(x,y)",
        "modulo(x,y)",
        "sign(x,y)",
        "dim(x,y)",
        "merge(x,y,x>y)",
        "real(x)",
        "real(x,kind=8)",
        "int(x)",
        "nint(x)",
        "floor(x)",
        "ceiling(x)",
        "dble(x)",
        "kind(x)",
        "epsilon(x)",
        "tiny(x)",
        "huge(x)",
        "sign(1._KIND,mod(x,y))",
        "sign(1._KIND,modulo(x,y))",
        "sign(1._KIND,sign(x,y))",
    ]
    integer_calls = [
        "min(j,k,2)",
        "max(j,k,2)",
        "mod(j,k)",
        "modulo(j,k)",
        "sign(j,-1)",
        "dim(-huge(j),huge(k))",
        "merge(j,k,j>k)",
        "int(j)",
        "kind(j)",
        "huge(j)",
    ]
    body = ["do i=1,size(x4)", "x=x4(i)", "y=y4(i)", "u=x8(i)", "v=y8(i)", "j=ints(i)", "k=divs(i)"]
    expressions = []
    for kind, x, y in [(4, "x", "y"), (8, "u", "v")]:
        expressions.extend(
            re.sub(r"\by\b", y, re.sub(r"\bx\b", x, call.replace("KIND", str(kind)))) for call in real_calls
        )
    expressions += integer_calls + [
        "real(j)",
        "real(j,kind=8)",
        "dble(j)",
        "merge(.true.,.false.,j>k) .eqv. (j>k)",
        "size(grid)",
        "size(grid,dim=1,kind=4)",
        "size(array=grid,dim=2)",
        "size(grid,3)",
        "lbound(grid,1)",
        "ubound(grid,2)",
        "ubound(grid,3)",
        "size(grid,dim=1+mod(i-1,3))",
        "lbound(grid,dim=1+mod(i-1,3))",
        "ubound(grid,dim=1+mod(i-1,3))",
        "ubound(empty,1)",
        "lbound(empty,1)",
        "size(empty)",
        "real(j,kind=kind(u))",
        "real(j,kind=wp)",
    ]
    logical_column = len(real_calls) * 2 + len(integer_calls) + 4
    for column, expression in enumerate(expressions, 1):
        if column == logical_column:
            expression = f"merge(1,0,{expression})"
        body.append(f"out(i,{column})={expression}")
    body += ["enddo"]
    path = tmp_path / "intrinsics.f90"
    path.write_text(
        "! kernels\nmodule intrinsic_case\ninteger,parameter::wp=kind(0.d0),ik=kind(0)\ncontains\n! kernel\n"
        "subroutine entry(out,x4,y4,x8,y8,ints,divs,grid,empty)\n"
        "real(8),intent(out)::out(:,:)\nreal,intent(in)::x4(:),y4(:)\n"
        "real(8),intent(in)::x8(:),y8(:),grid(:,:,:),empty(:)\n"
        "integer,intent(in)::ints(:),divs(:)\ninteger::i,j,k\nreal::x,y\nreal(8)::u,v\n"
        + "\n".join(body)
        + "\nend subroutine\nend module\n"
    )
    count = len(expressions)
    driver = f"""program main
use intrinsic_case
real::x(8),y(8)
real(8)::out(8,{count}),grid(-1:0,3:5,0:3),empty(0)
integer::j(8),k(8)
x=[-0.75,-0.5,-0.25,0.25,0.5,0.75,0.5,-0.5]
y=[0.5,0.25,-0.5,0.5,-0.25,-0.5,0.25,-0.25]
j=[-7,7,-7,7,0,(-2147483647-1),(-2147483647-1),2147483647]
k=[3,-3,-3,3,-1,-2,2147483647,(-2147483647-1)]
call entry(out,x,y,dble(x),dble(y),j,k,grid,empty)
print '(ES26.17E3)',out
end program
"""
    return lower_file(path, "entry"), driver, len(real_calls)


def check_results(actual, expected, real_count):
    assert len(actual) == len(expected)
    for index, (value, reference) in enumerate(zip(actual, expected, strict=True)):
        column = index // 8
        # Small relative-only tolerances also check TINY and EPSILON correctly.
        tolerance = 3e-7 if column < real_count else 3e-14
        if column >= real_count * 2:
            assert value == reference, (column + 1, index % 8 + 1)
        else:
            assert value == pytest.approx(reference, rel=tolerance, abs=0), (column + 1, index % 8 + 1)


@pytest.mark.native
def test_integer_remainder_boundary_and_argument_evaluation(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("native C++ compiler is unavailable")
    (tmp_path / "common_functions.cuh").write_text(read_common_header())
    (tmp_path / "numeric.cpp").write_text(
        r"""
#include "common_functions.cuh"
#include <cassert>
#include <limits>
int main() {
    using namespace generated_kernels::numeric;
    volatile int minimum = std::numeric_limits<int>::min();
    // Native integer division can trap on MIN / -1 even though its remainder is 0.
    assert(mod(minimum, -1) == 0);
    assert(modulo(minimum, -1) == 0);
    assert(sign(minimum, -1) == minimum);
    assert(dim(minimum, 1) == 0);
    int a = 0, p = 0, mask = 0;
    assert(modulo(++a, ++p) == 0);
    assert(a == 1 && p == 1);
    assert(merge(++a, ++p, ++mask) == 2);
    assert(a == 2 && p == 2 && mask == 1);
    assert(nint(::nextafter(0.5, 0.0)) == 0);
    assert(nint(::nextafter(-0.5, 0.0)) == 0);
    assert(nint(1.5) == 2 && nint(-1.5) == -2);
    assert(std::signbit(sign(1.0, -0.0)));
    assert(std::signbit(modulo(0.0, -1.0)));
}
"""
    )
    for command in (
        [
            compiler,
            "-std=c++17",
            "-fsanitize=undefined",
            "-fno-sanitize-recover=undefined",
            "numeric.cpp",
            "-o",
            "numeric",
        ],
        [str(tmp_path / "numeric")],
    ):
        result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=60, check=False)
        assert result.returncode == 0, result.stdout + result.stderr
