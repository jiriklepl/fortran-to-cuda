"""Intrinsic typing and execution compared with native Fortran on each backend."""

import shutil
import subprocess

import pytest

from compiler.analysis import ParallelizationError, build_execution_plan
from compiler.analysis.semantics import expression_type
from compiler.driver.options import CompilerOptions
from compiler.driver.pipeline import prepare_function
from compiler.emission import read_common_header
from compiler.ir import CompilationError, IntrinsicCall, Literal, ScalarType, SourceLocation
from compiler.ir.intrinsics import INTRINSIC_ARGUMENTS
from compiler.tests.test_language import lower

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
