"""Structured conditions, scalar logical values, and typed scalar intrinsics."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from compiler.analysis import ParallelizationError, build_execution_plan
from compiler.emission import generate_sources, read_common_header
from compiler.frontend import lower_file
from compiler.ir import CompilationError, ConditionalRegion, If, IntrinsicCall, ScalarType, walk_expr
from compiler.tests.test_native import cuda_device as cuda_device


def lower(tmp_path, body, declarations="", arguments="a,n"):
    path = tmp_path / "language.f90"
    path.write_text(
        "! kernels\nmodule language_case\ninteger,parameter::knd=kind(1.d0)\ncontains\n! kernel\n"
        f"subroutine entry({arguments})\nreal(knd),intent(inout)::a(:)\ninteger,intent(in)::n\n"
        f"integer::i\n{declarations}\n{body}\nend subroutine\nend module\n"
    )
    return lower_file(path, "entry")


def test_block_elseif_and_single_line_conditions_are_typed(tmp_path):
    function = lower(
        tmp_path,
        "if (n > 2) then\na(1)=1\nelse if(n==2)then\na(1)=2\nelse\na(1)=3\nendif\nif(n/=0) a(1)=a(1)+1",
    )
    first, second = function.body.statements
    assert isinstance(first, If)
    assert isinstance(first.else_body.statements[0], If)
    assert isinstance(second, If)
    plan = build_execution_plan(function)
    assert all(isinstance(step, ConditionalRegion) for step in plan.steps)


def test_logical_expressions_and_scalar_arguments(tmp_path):
    function = lower(
        tmp_path,
        "flag = (.not.enabled .or. n>0) .eqv. .true.\nif(flag .neqv. .false.) a(1)=1",
        "logical,intent(in)::enabled\nlogical::flag",
        "a,n,enabled",
    )
    assert function.parameters[-1].dtype is ScalarType.LOGICAL
    assert len(build_execution_plan(function).steps) == 2


@pytest.mark.parametrize(
    "call", ["abs(-2)", "abs(-2.0)", "abs(-2.d0)", "min(1,2,3)", "max(1.d0,2.d0)", "sqrt(4.0)", "sqrt(4.d0)"]
)
def test_intrinsics_have_resolved_numeric_result_types(tmp_path, call):
    function = lower(tmp_path, "a(1)=" + call)
    node = function.body.statements[0].value
    assert isinstance(node, IntrinsicCall)
    assert node.dtype in (ScalarType.INTEGER, ScalarType.REAL32, ScalarType.REAL)
    assert tuple(walk_expr(node))[0] is node
    build_execution_plan(function)


@pytest.mark.parametrize(
    ("body", "declarations", "message"),
    [
        ("if(n) a(1)=1", "", "IF condition"),
        ("a(1)=.true.", "", "compatible logical or numeric"),
        ("a(1)=.true.+1", "", "invalid operand"),
        ("a(1)=sqrt(1)", "", "invalid argument types"),
        ("a(1)=min(1,2.d0)", "", "invalid argument types"),
        ("a(1)=abs(.true.)", "", "invalid argument types"),
        ("a(1)=1", "logical::flags(:)", "LOGICAL arrays"),
    ],
)
def test_invalid_types_are_rejected(tmp_path, body, declarations, message):
    with pytest.raises(CompilationError, match=message):
        build_execution_plan(lower(tmp_path, body, declarations))


def test_branch_definition_join_uses_intersection(tmp_path):
    function = lower(tmp_path, "if(n>0) t=1\na(1)=t", "real(knd)::t")
    with pytest.raises(CompilationError, match="read before definition"):
        build_execution_plan(function)
    function = lower(tmp_path, "if(n>0) then\nt=1\nelse\nt=2\nendif\na(1)=t", "real(knd)::t")
    assert len(build_execution_plan(function).steps) == 2


def test_private_branch_definitions_and_condition_reads(tmp_path):
    function = lower(
        tmp_path,
        "do i=1,n\nif(a(i)>0)then\nt=1\nelse\nt=2\nendif\na(i)=t\nenddo",
        "real(knd)::t",
    )
    region = build_execution_plan(function).regions[0]
    assert [symbol.name for symbol in region.private_symbols] == ["t"]
    assert [symbol.name for symbol in region.read_symbols] == ["a"]


def test_predicate_array_access_participates_in_dependence_proof(tmp_path):
    function = lower(tmp_path, "do i=2,n\nif(a(i-1)>0) a(i)=1\nenddo")
    with pytest.raises(ParallelizationError, match="RAW"):
        build_execution_plan(function)


def test_recursive_host_plan_retains_parallel_regions(tmp_path):
    function = lower(tmp_path, "if(n>0)then\ndo i=1,n\na(i)=1\nenddo\nelse\na(1)=2\nendif")
    plan = build_execution_plan(function)
    assert isinstance(plan.steps[0], ConditionalRegion)
    assert len(plan.regions) == 1
    assert plan.regions[0] == plan.steps[0].then_plan.regions[0]


def run_reference_and_cpp(
    tmp_path: Path, function, plan, driver, *, openmp=False, backend="cpp", generated_driver=None
):
    """Compare generated execution with the same program using original Fortran."""
    if not shutil.which("gfortran") or not shutil.which("g++"):
        pytest.skip("native Fortran and C++ compilers are unavailable")
    source = Path(function.source)
    generated = generate_sources(function, plan)
    (tmp_path / "implementation.cpp").write_text(generated.cpp)
    (tmp_path / "interface.f90").write_text(generated.fortran)
    (tmp_path / "common_functions.cuh").write_text(read_common_header())
    (tmp_path / "driver.f90").write_text(driver)

    def run(command):
        result = subprocess.run(
            command,
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=45,
            env={**os.environ, "OMP_NUM_THREADS": "2"},
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout.split()

    reference = tmp_path / "reference"
    target = tmp_path / "generated"
    run(["gfortran", str(source), "driver.f90", "-o", str(reference)])
    expected = run([str(reference)])
    if generated_driver is not None:
        (tmp_path / "driver.f90").write_text(generated_driver)
    flags = ["-fopenmp"] if openmp else []
    if backend == "cuda":
        compile_cuda_sources(tmp_path, function, plan)
        run(["gfortran", "-c", "interface.f90", "driver.f90"])
        run(["nvcc", "implementation.o", "interface.o", "driver.o", "-lgfortran", "-o", str(target)])
        actual = run([str(target)])
        assert [float(value) for value in actual] == pytest.approx([float(value) for value in expected])
        return
    run(["g++", "-std=c++17", *flags, "-c", "implementation.cpp", "-o", "implementation.o"])
    run(["gfortran", *flags, "interface.f90", "driver.f90", "implementation.o", "-lstdc++", "-o", str(target)])
    actual = run([str(target)])
    assert [float(value) for value in actual] == pytest.approx([float(value) for value in expected])


def structured_case(tmp_path):
    function = lower(
        tmp_path,
        "iv=max(abs(-3),min(5,2))\nrv=sqrt(abs(-4.0))\n"
        "flag=(.not.(.not.enabled) .and. n>0) .eqv. .true.\nif(flag)then\ndo i=1,n\n"
        "pick=(a(i)<0) .eqv. .true.\nif(pick)then\nt=abs(a(i))\n"
        "elseif((a(i)>2) .neqv. .false.)then\nt=min(a(i),4.d0,3.d0)\n"
        "else\nt=max(a(i),2.d0)\nendif\na(i)=sqrt(t*t)+rv+iv\nenddo\n"
        "else\ndo i=1,n\na(i)=-1\nenddo\nendif\nif(.not.enabled) a(1)=a(1)+2",
        "logical,intent(in)::enabled\nlogical::flag,pick\nreal(knd)::t\nreal::rv\ninteger::iv",
        "a,n,enabled",
    )
    driver = """program main
use language_case
real(knd)::a(4)
a=[-4.d0,1.d0,3.d0,9.d0]
call entry(a,4,.true.)
print *,a
call entry(a,4,.false.)
print *,a
end program
"""
    return function, driver


@pytest.mark.native
@pytest.mark.parametrize("openmp", [False, True])
def test_conditions_intrinsics_and_logical_bridge_execute_like_fortran(tmp_path, openmp):
    function, driver = structured_case(tmp_path)
    run_reference_and_cpp(tmp_path, function, build_execution_plan(function), driver, openmp=openmp)


def compile_cuda_sources(tmp_path, function, plan):
    executable = shutil.which("nvcc")
    if executable is None:
        pytest.skip("CUDA toolkit is unavailable")
    sources = generate_sources(function, plan)
    (tmp_path / "implementation.cu").write_text(sources.cuda)
    (tmp_path / "interface.f90").write_text(sources.fortran)
    (tmp_path / "common_functions.cuh").write_text(read_common_header())
    result = subprocess.run(
        [executable, "-std=c++17", "--fmad=false", "-c", "implementation.cu", "-o", "implementation.o"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.cuda
def test_conditions_intrinsics_and_logical_values_compile_for_cuda(tmp_path):
    function, _ = structured_case(tmp_path)
    compile_cuda_sources(tmp_path, function, build_execution_plan(function))


@pytest.mark.native
@pytest.mark.cuda
@pytest.mark.usefixtures("cuda_device")
def test_conditions_intrinsics_and_logical_values_match_fortran_on_cuda(tmp_path):
    function, driver = structured_case(tmp_path)
    run_reference_and_cpp(tmp_path, function, build_execution_plan(function), driver, backend="cuda")


def test_host_branch_snapshots_have_distinct_symbolic_parameters(tmp_path):
    function = lower(
        tmp_path,
        "m=index(1)\nold=m\nif(n>0)then\nm=index(2)\ndo i=old,m\na(i)=1\nenddo\nendif",
        "integer,intent(in)::index(:)\ninteger::m,old",
        "a,n,index",
    )
    region = build_execution_plan(function).regions[0]
    import re

    snapshots = set(re.findall(r"h[0-9]+_[0-9]+", region.report.domain))
    assert len(snapshots) == 2


def test_many_argument_minmax_rendering_grows_linearly():
    from compiler.emission.common.c_family import render_expression
    from compiler.ir import Binary, Literal

    def rendered(count):
        arguments = tuple(
            Binary("-", Literal(str(i) + ".0", ScalarType.REAL32), Literal("40.0", ScalarType.REAL32))
            for i in range(1, count + 1)
        )
        return render_expression(IntrinsicCall("max", arguments, ScalarType.REAL32))

    assert len(rendered(64)) < 10 * len(rendered(8))
    assert rendered(64).count("- 40.0f") == 64


@pytest.mark.native
def test_many_argument_minmax_preserves_typed_grouped_values(tmp_path):
    arguments = ", &\n &".join(f"({i}.0-40.0)" for i in range(1, 65))
    function = lower(
        tmp_path,
        f"a(1)=min({arguments})\na(2)=max({arguments})\na(3)=max(-1,min(abs(-2),3))\na(4)=max(1.d0,min(-1.d0,3.d0))",
    )
    driver = """program main
use language_case
real(knd)::a(4)
a=0
call entry(a,4)
print *,a
end program
"""
    run_reference_and_cpp(tmp_path, function, build_execution_plan(function), driver)
