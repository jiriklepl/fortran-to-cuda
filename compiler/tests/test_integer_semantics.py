"""The default INTEGER ABI has checked constants and unchanged runtime arithmetic."""

import os
import re
import shutil
import subprocess
from dataclasses import replace

import pytest

from compiler.analysis import build_execution_plan
from compiler.analysis.semantics import constant_integer as stage_constant_integer
from compiler.driver.options import CompilerOptions
from compiler.driver.pipeline import prepare_function
from compiler.emission import generate_sources, read_common_header
from compiler.emission.common.c_family import render_expression
from compiler.frontend import lower_file
from compiler.ir import (
    ArrayAccess,
    Assignment,
    Binary,
    Block,
    CompilationError,
    FunctionIR,
    IntrinsicCall,
    Literal,
    Reference,
    ScalarType,
    SourceLocation,
    Symbol,
    Unary,
)
from compiler.ir.integers import INTEGER_MAX, INTEGER_MIN, constant_integer
from compiler.tests.test_language import compile_cuda_sources
from compiler.tests.test_language_cuda import CUDA_RUNTIME

INTEGER = ScalarType.INTEGER
LOCATION = SourceLocation("integer_case.f90", 17)
N = Symbol(0, "n", INTEGER, intent="in", parameter=True)
T = Symbol(1, "t", INTEGER)
A = Symbol(2, "a", INTEGER, intent="inout", rank=1, parameter=True)


def literal(value):
    return Literal(str(value), INTEGER)


def source(tmp_path, body):
    path = tmp_path / "integers.f90"
    path.write_text(
        "! kernels\nmodule integer_case\ncontains\n! kernel\nsubroutine entry(a,n)\n"
        "integer,intent(inout)::a(:)\ninteger,intent(in)::n\ninteger::i,t\n" + body + "\nend subroutine\nend module\n"
    )
    return path


def function(expression):
    return FunctionIR(
        "entry",
        "integer_case",
        (A, N),
        (A, N, T),
        Block((Assignment(Reference(T), expression, LOCATION),)),
        LOCATION.path,
    )


@pytest.mark.parametrize("token", ["2147483648", "-2147483648", "+2147483648", "-2147483649", "99999999999999999999"])
@pytest.mark.parametrize("context", ["a(1)={}", "a({})=1", "do i=1,{}\na(1)=1\nenddo", "a(1)=min(n,{})"])
def test_source_literal_tokens_must_fit_before_unary_signs(tmp_path, token, context):
    path = source(tmp_path, context.format(token))
    with pytest.raises(CompilationError, match="default INTEGER literal is out of range") as error:
        lower_file(path, "entry")
    assert error.value.location == SourceLocation(str(path), 9)


@pytest.mark.parametrize(
    "expression",
    [
        "2147483647+1",
        "(-2147483647-1)-1",
        "2147483647*2",
        "-(-2147483647-1)",
        "(-2147483647-1)/(-1)",
        "abs(-2147483647-1)",
        "(2147483647+1)-1",
        "min(n,2147483647+1)",
        "max(n,abs(-2147483647-1))",
        "n+(2147483647*2)",
        "min(0,2147483647+1)",
        "1/0",
        "n/0",
        "max(n,1/0)",
        "n+(1/(2-2))",
    ],
)
@pytest.mark.parametrize("fallback", ["error", "host"])
def test_invalid_constant_subexpressions_cannot_become_fallback(tmp_path, expression, fallback):
    path = source(tmp_path, "a(1)=" + expression)
    with pytest.raises(
        CompilationError, match="default INTEGER (constant expression overflows|division by zero)"
    ) as error:
        build_execution_plan(lower_file(path, "entry"), options=CompilerOptions(fallback=fallback))
    assert error.value.location == SourceLocation(str(path), 9)


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        (literal(INTEGER_MIN), INTEGER_MIN),
        (literal(INTEGER_MAX), INTEGER_MAX),
        (literal("0008"), 8),
        (Binary("-", Unary("-", literal(INTEGER_MAX)), literal(1)), INTEGER_MIN),
        (Binary("/", literal(-7), literal(3)), -2),
        (Binary("/", literal(7), literal(-3)), -2),
        (Binary("/", literal(-7), literal(-3)), 2),
        (Binary("/", literal(INTEGER_MIN), literal(1)), INTEGER_MIN),
        (IntrinsicCall("min", (literal(INTEGER_MAX), literal(INTEGER_MIN), literal(0)), INTEGER), INTEGER_MIN),
        (IntrinsicCall("max", (literal(INTEGER_MIN), literal(0), literal(INTEGER_MAX)), INTEGER), INTEGER_MAX),
        (IntrinsicCall("abs", (literal(-INTEGER_MAX),), INTEGER), INTEGER_MAX),
        (Reference(N), None),
        (Binary("/", Reference(N), literal(-1)), None),
        (IntrinsicCall("min", (Reference(N), literal(INTEGER_MIN)), INTEGER), None),
        (Binary("/", Literal("1.0", ScalarType.REAL32), literal(0)), None),
    ],
)
def test_checked_evaluation_keeps_valid_values_and_unknown_runtime_values(expression, expected):
    assert constant_integer(expression) == expected
    assert stage_constant_integer(expression) == expected
    build_execution_plan(function(expression))


@pytest.mark.parametrize(
    "expression",
    [
        literal(INTEGER_MAX + 1),
        literal(INTEGER_MIN - 1),
        Unary("-", literal(INTEGER_MIN)),
        Binary("*", literal(INTEGER_MAX), literal(2)),
        Binary("/", literal(INTEGER_MIN), literal(-1)),
        Binary("/", Reference(N), literal(0)),
        IntrinsicCall("abs", (literal(INTEGER_MIN),), INTEGER),
        IntrinsicCall("min", (Reference(N), Binary("+", literal(INTEGER_MAX), literal(1))), INTEGER),
        Binary("+", Reference(N), Binary("/", literal(1), literal(0))),
        Binary("+", Literal("1.0", ScalarType.REAL32), Binary("+", literal(INTEGER_MAX), literal(1))),
        ArrayAccess(A, (Binary("+", Reference(N), literal(INTEGER_MAX + 1)),)),
    ],
)
@pytest.mark.parametrize("fallback", ["error", "host"])
def test_manually_constructed_ir_has_identical_integer_validation(expression, fallback):
    with pytest.raises(CompilationError, match="default INTEGER") as error:
        build_execution_plan(function(expression), options=CompilerOptions(fallback=fallback))
    assert error.value.location == LOCATION


@pytest.mark.native
def test_manually_constructed_int_min_has_cpp_int_type(tmp_path):
    executable = shutil.which("g++")
    if executable is None:
        pytest.skip("C++ compiler is unavailable")
    rendered = render_expression(literal(INTEGER_MIN))
    path = tmp_path / "minimum.cpp"
    path.write_text(
        "#include <type_traits>\n#include <limits>\n"
        f"static_assert(std::is_same_v<decltype({rendered}), int>);\n"
        f"static_assert({rendered} == std::numeric_limits<int>::min());\n"
    )
    result = subprocess.run(
        [executable, "-std=c++17", "-c", str(path), "-o", str(tmp_path / "minimum.o")],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


BOUNDARY_BODY = """a(1)=2147483647
a(2)=(-2147483647-1)
a(3)=abs(-2147483647)
a(4)=min(2147483647,(-2147483647-1),n)
a(5)=max((-2147483647-1),n,2147483647)
a(6)=(-7)/3
a(7)=7/(-3)
a(8)=(-7)/(-3)
a(9)=(-2147483647-1)/1
a(10)=0008
do i=2147483646,2147483644,-1
 a(2147483647-i+10)=i
enddo
do i=(-2147483647-1),(-2147483646)
 a(i+2147483647+15)=i
enddo
do i=1,0
 a(1)=0
enddo"""


@pytest.mark.native
@pytest.mark.parametrize("backend", ["serial", "openmp", "cuda-sim"])
@pytest.mark.parametrize("opt_level", [0, 1])
def test_boundary_values_execute_exactly_like_fortran(tmp_path, backend, opt_level):
    if not shutil.which("gfortran") or not shutil.which("g++"):
        pytest.skip("native Fortran and C++ compilers are unavailable")
    path = source(tmp_path, BOUNDARY_BODY)
    prepared, plan = prepare_function(lower_file(path, "entry"), options=CompilerOptions(opt_level=opt_level))
    generated = generate_sources(prepared, plan)
    if backend == "cuda-sim":
        simulated = generated.cuda.replace("#include <cuda_runtime.h>", '#include "cuda_runtime.h"')
        simulated = re.sub(r"(kernel_region_\d+_device)<<<([^>]+)>>>\(", r"fort_test_launch(\1, \2, ", simulated)
        generated = replace(generated, cpp=simulated)
        (tmp_path / "cuda_runtime.h").write_text(CUDA_RUNTIME)
    (tmp_path / "implementation.cpp").write_text(generated.cpp)
    (tmp_path / "interface.f90").write_text(generated.fortran)
    (tmp_path / "common_functions.cuh").write_text(read_common_header())
    (tmp_path / "driver.f90").write_text(
        "program main\nuse integer_case\ninteger::a(16)\na=0\ncall entry(a,7)\nprint *,a\nend program\n"
    )

    def run(command):
        result = subprocess.run(
            command,
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, "OMP_NUM_THREADS": "2"},
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout.split()

    run(["gfortran", str(path), "driver.f90", "-o", "reference"])
    expected = run([str(tmp_path / "reference")])
    flags = ["-fopenmp"] if backend == "openmp" else []
    run(
        [
            "g++",
            "-std=c++17",
            *flags,
            "-fsanitize=undefined",
            "-fno-sanitize-recover=undefined",
            "-c",
            "implementation.cpp",
            "-o",
            "implementation.o",
        ]
    )
    run(
        [
            "gfortran",
            *flags,
            "-fsanitize=undefined",
            "-fno-sanitize-recover=undefined",
            "interface.f90",
            "driver.f90",
            "implementation.o",
            "-lstdc++",
            "-o",
            "generated",
        ]
    )
    actual = run([str(tmp_path / "generated")])
    assert actual == expected


@pytest.mark.cuda
@pytest.mark.native
def test_boundary_values_compile_on_cuda(tmp_path):
    function, plan = prepare_function(lower_file(source(tmp_path, BOUNDARY_BODY), "entry"))
    compile_cuda_sources(tmp_path, function, plan)
