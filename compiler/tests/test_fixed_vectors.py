"""Bounded vectors disappear into ordered scalar numerical IR."""

import math
import os
import shutil
import subprocess

import pytest

from compiler.analysis import build_execution_plan
from compiler.emission import generate_sources, read_common_header
from compiler.frontend import lower_file
from compiler.ir import (
    ArrayAccess,
    Assignment,
    CompilationError,
    IntrinsicCall,
    Literal,
    Reference,
    ScalarType,
    walk_expr,
)
from compiler.tests.test_native import cuda_device as cuda_device


def source(tmp_path, declarations, body, *, helpers="", precision=8):
    path = tmp_path / "bounded_vectors.f90"
    path.write_text(f"""module renamed_vectors
implicit none
contains
subroutine evaluate(a,b,n,origin)
real({precision}),intent(in)::a(:)
real({precision}),intent(inout)::b(:)
integer,intent(in)::n,origin
integer::i
{declarations}
{body}
{helpers}
end subroutine
end module
""")
    return path


def statements(function):
    def visit(block):
        for item in block.statements:
            if isinstance(item, Assignment):
                yield item
            elif hasattr(item, "body"):
                yield from visit(item.body)
            else:
                yield from visit(item.then_body)
                yield from visit(item.else_body)
    return tuple(visit(function.body))


@pytest.mark.parametrize(("precision", "dtype"), [(4, ScalarType.REAL32), (8, ScalarType.REAL)])
def test_parameter_vectors_are_typed_constants_without_private_storage(tmp_path, precision, dtype):
    path = source(tmp_path, f"""real({precision}),parameter::factor=0.5_{precision}
real({precision}),parameter::weights(-1:2)=[1._{precision},2._{precision},3._{precision},4._{precision}]
real({precision}),parameter::shifted(0:3)=weights*factor+1._{precision}
real({precision})::g(-2:1)""", """do i=1,n
g=shifted*a(i)
b(i)=sum(g)+real(lbound(weights,1)+ubound(weights,1)+size(weights),kind=8)
enddo""", precision=precision)
    function = lower_file(path, "evaluate")
    plan = build_execution_plan(function)
    assert len(plan.regions) == 1
    assert not any(symbol.name.startswith(("weights", "shifted")) for symbol in function.symbols)
    private = [symbol for symbol in function.symbols if symbol.private_array_origin]
    assert len(private) == 4
    assert {symbol.private_array_origin.bounds for symbol in private} == {((-2, 1),)}
    assert all(symbol.dtype is dtype for symbol in private)
    assert all(not getattr(item.value, "elements", None) for item in statements(function))


def test_typed_constructor_converts_after_original_kind_arithmetic(tmp_path):
    function = lower_file(source(tmp_path,
        "real(8),parameter::weights(3)=[real(8)::1._4/10._4,1._8/10._8,1]",
        "do i=1,n\nb(i)=sum(weights*a(i))\nenddo"), "evaluate")
    expressions = [value for item in statements(function) for value in walk_expr(item.value)]
    conversions = [value for value in expressions if isinstance(value, IntrinsicCall) and value.name == "real"]
    assert any(value.dtype is ScalarType.REAL and
               getattr(value.arguments[0], "operator", None) == "/" and
               value.arguments[0].left.dtype is ScalarType.REAL32 for value in conversions)
    assert any(value.dtype is ScalarType.REAL and isinstance(value.arguments[0], Literal)
               and value.arguments[0].dtype is ScalarType.INTEGER for value in conversions)


def test_negative_private_reverse_and_overlapping_assignment_snapshot(tmp_path):
    function = lower_file(source(tmp_path, "real(8)::g(-2:1)", """do i=1,n
g=[1.d0,2.d0,3.d0,4.d0]
g(-1:1)=g(-2:0)
g(1:-2:-1)=g
b(i)=dot_product(g,g(1:-2:-1))
enddo"""), "evaluate")
    loop, = function.body.statements
    writes = [(index, item) for index, item in enumerate(loop.body.statements)
              if isinstance(item.target, Reference) and item.target.symbol.private_array_origin]
    # The three overlapping RHS elements are all snapshotted before their
    # original private destinations are modified.
    shifted = writes[4:7]
    snapshot_ids = {item.value.symbol for _, item in shifted}
    assert len(snapshot_ids) == 3
    snapshot_positions = [index for index, item in enumerate(loop.body.statements)
                          if isinstance(item.target, Reference) and item.target.symbol in snapshot_ids]
    assert max(snapshot_positions) < min(index for index, _ in shifted)
    assert [item.target.symbol.private_array_origin.element_offset for _, item in writes[7:11]] == [3, 2, 1, 0]
    assert len(build_execution_plan(function).regions) == 1


def test_captured_sections_preserve_runtime_origins_and_constant_strides(tmp_path):
    function = lower_file(source(tmp_path, "real(8)::g(3)", """do i=3,n-1
g=a(i-2:i+2:2)
b(i)=sum(g)+sum(a(i+2:i-2:-2))
enddo"""), "evaluate")
    accesses = [value for item in statements(function) for value in walk_expr(item.value)
                if isinstance(value, ArrayAccess) and value.symbol is function.parameters[0]]
    assert len(accesses) == 6
    assert all(any(isinstance(value, Reference) and value.symbol.name == "i" for value in walk_expr(access.indices[0]))
               for access in accesses)
    build_execution_plan(function)


def test_scalar_helper_broadcast_is_evaluated_once(tmp_path):
    function = lower_file(source(tmp_path, "real(8)::g(3)", """do i=1,n
g=[1.d0,2.d0,3.d0]+twice(a(i))
b(i)=sum(g)
enddo""", helpers="""contains
pure function twice(x) result(value)
real(8),intent(in)::x
real(8)::value
value=x+x
end function"""), "evaluate")
    calls = [item for item in statements(function) if item.location.call_stack]
    assert len(calls) == 1
    broadcasts = [item for item in statements(function) if item.target.symbol.name.startswith("fort_broadcast")]
    assert len(broadcasts) == 1
    build_execution_plan(function)


def test_parameter_vector_readonly_helper_forwarding_preserves_rebased_bounds(tmp_path):
    function = lower_file(source(tmp_path,
        "real(8),parameter::weights(-1:1)=[1.d0,2.d0,3.d0]",
        "do i=1,n\nb(i)=weight(weights)*a(i)\nenddo",
        helpers="""contains
pure function weight(g) result(value)
real(8),intent(in)::g(0:2)
real(8)::value
value=sum(g)+g(0)+lbound(g,1)
end function"""), "evaluate")
    assert not any(symbol.private_array_origin for symbol in function.symbols)
    build_execution_plan(function)


@pytest.mark.parametrize("precision", [4, 8])
@pytest.mark.parametrize("expression", ["sum(g)", "dot_product(g,g)"])
def test_real_vector_reductions_retain_environment_requirement(tmp_path, precision, expression):
    function = lower_file(source(tmp_path, f"real({precision})::g(3)",
        f"do i=1,n\ng=a(i:i+2)\nb(i)={expression}\nenddo", precision=precision), "evaluate")
    assert function.requires_numerical_environment


@pytest.mark.parametrize("expression", ["sum(g)", "dot_product(g,g)"])
def test_integer_vector_reductions_do_not_require_floating_environment(tmp_path, expression):
    function = lower_file(source(tmp_path, "integer::g(3)",
        f"g=[1,2,3]\nb(1)={expression}"), "evaluate")
    assert not function.requires_numerical_environment


def test_real_vector_work_without_reduction_retains_environment_contract(tmp_path):
    function = lower_file(source(tmp_path, "real(8)::g(3)",
        "g=a(1:3)+1.d0\nb(1:3)=g"), "evaluate")
    assert function.requires_numerical_environment


@pytest.mark.parametrize("expression", ["sqrt(a(1:3))", "int(a(1:3))", "int(a(1))"])
def test_vector_math_and_float_to_integer_conversions_retain_environment_contract(tmp_path, expression):
    function = lower_file(source(tmp_path, "integer::g(3)",
        f"g={expression}\nb(1)=g(1)"), "evaluate")
    assert function.requires_numerical_environment


def test_integer_vector_metadata_inquiries_do_not_read_real_array_values(tmp_path):
    function = lower_file(source(tmp_path, "integer::g(3)",
        "g=[1,2,3]+size(a)\nb(1)=g(1)"), "evaluate")
    assert not function.requires_numerical_environment


@pytest.mark.parametrize("expression", ["convert_value(a(1))", "[1,2,3]+convert_value(a(1))"])
def test_integer_vector_helper_broadcast_retains_transitive_float_environment(tmp_path, expression):
    function = lower_file(source(tmp_path, "integer::g(3)",
        f"g={expression}\nb(1)=g(1)", helpers="""contains
pure function convert_value(x) result(value)
real(8),intent(in)::x
integer::value
value=int(sqrt(x))
end function"""), "evaluate")
    assert function.requires_numerical_environment


@pytest.mark.parametrize("precision", [4, 8])
@pytest.mark.parametrize("intrinsic", ["min", "max"])
def test_real_vector_varargs_require_proven_native_nan_ordering(tmp_path, precision, intrinsic):
    with pytest.raises(CompilationError, match="real vector MIN/MAX supports only two operands"):
        lower_file(source(tmp_path, f"real({precision})::g(3)",
            f"g={intrinsic}(a(1:3),a(2:4),a(3:5))\nb(1:3)=g", precision=precision), "evaluate")


@pytest.mark.parametrize("intrinsic", ["min", "max"])
def test_real_two_operand_vectors_and_integer_varargs_remain_supported(tmp_path, intrinsic):
    function = lower_file(source(tmp_path, "real(8)::g(3)",
        f"g={intrinsic}(a(1:3),a(2:4))\nb(1:3)=g"), "evaluate")
    assert function.requires_numerical_environment
    function = lower_file(source(tmp_path, "integer::g(3),h(3)",
        f"g=[1,2,3]\nh={intrinsic}(g,g+1,g+2)\nb(1)=h(1)"), "evaluate")
    assert not function.requires_numerical_environment


def test_existing_scalar_real_varargs_behavior_is_unchanged(tmp_path):
    function = lower_file(source(tmp_path, "", "b(1)=max(a(1),a(2),a(3))"), "evaluate")
    assert isinstance(function.body.statements[0].value, IntrinsicCall)
    assert len(function.body.statements[0].value.arguments) == 3


@pytest.mark.parametrize(("declarations", "body", "reason"), [
    ("real(8)::g(3)", "g=a(1:n)", "constant cardinality"),
    ("real(8)::g(3)", "g=a(1:3:n)", "constant nonzero stride"),
    ("real(8)::g(3)", "g=a(1:3:0)", "constant nonzero stride"),
    ("real(8)::g(3)", "g=[1.d0,2.d0]", "extents differ"),
    ("real(8)::g(3)", "b(1)=g", "scalar assignment"),
    ("real(8)::g(3,3)", "g(:,:)=1.d0", "one varying axis"),
    ("real(8),parameter::w(2)=[1.,2.d0]", "b(1)=sum(w)", "matching kinds"),
    ("real(8),parameter::w(2)=[1.d0,2.d0]", "w(1)=3.d0", "assignment targets|INTENT"),
    ("real(8),parameter::w(2)=[a(1),a(2)]", "b(1)=sum(w)", "immutable constant"),
    ("real(8)::g(3)", "b(1)=sum(g,mask=.true.)", "without DIM or MASK"),
    ("real(8)::g(257)", "b(1)=sum(g)", "256-element"),
    ("integer::g(3)", "g=[1,2,3]/0", "division by zero"),
    ("integer::g(3)", "g=[1,2,3]+(1/0)", "division by zero"),
    ("integer::g(3)", "g=[2147483647,2,3]+1", "constant expression overflows"),
])
def test_unproved_or_unbounded_vectors_remain_frontend_errors(tmp_path, declarations, body, reason):
    with pytest.raises(CompilationError, match=reason):
        lower_file(source(tmp_path, declarations, body), "evaluate")


def test_empty_typed_vectors_and_sum_keep_positive_zero_seed(tmp_path):
    function = lower_file(source(tmp_path, "real(8),parameter::empty(-2:-3)=[real(8)::]",
        "b(1)=sum(empty)"), "evaluate")
    value = function.body.statements[0].value
    assert value == Literal("0.0", ScalarType.REAL)


def test_omitted_reverse_bounds_retain_declared_bounds_instead_of_reversing(tmp_path):
    function = lower_file(source(tmp_path, "real(8)::g(-2:1)",
        "b(1)=sum(g(::-1))"), "evaluate")
    assert function.body.statements[0].value == Literal("0.0", ScalarType.REAL)
    function = lower_file(source(tmp_path, "real(8)::g(-2:1)",
        "g=[1.d0,2.d0,3.d0,4.d0]\nb(1)=sum(g(:-2:-1))+sum(g(1::-1))"), "evaluate")
    last = function.body.statements[-1]
    used = [value.symbol.private_array_origin.element_offset for value in walk_expr(last.value)
            if isinstance(value, Reference) and value.symbol.private_array_origin]
    assert used == [0, 3]


@pytest.mark.native
@pytest.mark.parametrize("precision", [4, 8])
@pytest.mark.parametrize("backend", ["cpp", pytest.param("cuda", marks=pytest.mark.cuda)])
def test_vector_snapshots_and_short_sum_match_native_complete_fields(tmp_path, precision, backend, request):
    fc = shutil.which("gfortran-15") or shutil.which("gfortran")
    cxx = shutil.which("g++")
    if not fc or not cxx:
        pytest.skip("native Fortran and C++ compilers required")
    if backend == "cuda":
        request.getfixturevalue("cuda_device")
    path = source(tmp_path,
        f"""real({precision}),parameter::w(-1:2)=[real({precision})::1._4/10._4,1._8/10._8,1,-0._{precision}]
real({precision})::g(-2:1),t(4)""",
        """do i=3,n-1
g=a(i-2:i+1)
g(-1:1)=g(-2:0)
g(1:-2:-1)=g
t=abs(g)*w+half(a(i))
b(i)=sum(t)+dot_product(g(1:-2:-1),w)+sum(a(i+1:i-2:-1))
enddo""",
        helpers=f"""contains
pure function half(x) result(value)
real({precision}),intent(in)::x
real({precision})::value
value=x/2
end function""", precision=precision)
    function = lower_file(path, "evaluate")
    generated = generate_sources(function, build_execution_plan(function))
    (tmp_path / "implementation.cpp").write_text(generated.cpp)
    (tmp_path / "interface.f90").write_text(generated.fortran)
    (tmp_path / "common_functions.cuh").write_text(read_common_header())
    (tmp_path / "driver.f90").write_text(f"""program check
use renamed_vectors
use ieee_arithmetic
implicit none
real({precision})::a(12),b(12),large,nan,inf
integer::i,which
large=2._{precision}**{30 if precision == 4 else 60}
nan=ieee_value(0._{precision},ieee_quiet_nan)
inf=ieee_value(0._{precision},ieee_positive_inf)
do which=1,6
do i=1,12
 select case(which)
 case(1)
  a(i)=real(i-7,{precision})/8
 case(2)
  a(i)=merge(large,-large,mod(i,2)==0)
 case(3)
  a(i)=-0._{precision}
 case(4)
  a(i)=nan
 case(5)
  a(i)=inf
 case(6)
  a(i)=merge(inf,-inf,mod(i,2)==0)
 end select
enddo
b=-321._{precision}
call evaluate(a,b,12,3)
print '(ES26.17E3)',b
enddo
end program
""")
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE"}

    def run(command):
        result = subprocess.run(command, cwd=tmp_path, env=environment, text=True, capture_output=True, timeout=45)
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout

    # Match supported native flags; this test never enables fast math or
    # substitutes a parallel-reduction contract for serial short SUM.
    run([fc, "-O3", "-fopenmp", str(path), "driver.f90", "-o", "reference"])
    expected = [float(value) for value in run([str(tmp_path / "reference")]).split()]
    if backend == "cpp":
        run([cxx, "-O3", "-std=c++17", "-fopenmp", "-c", "implementation.cpp"])
        run([fc, "-O3", "-fopenmp", "interface.f90", "driver.f90", "implementation.o", "-lstdc++", "-o", "generated"])
    else:
        nvcc = shutil.which("nvcc")
        if not nvcc:
            pytest.skip("CUDA toolkit required")
        host = shutil.which("g++-14") or cxx
        (tmp_path / "implementation.cu").write_text(generated.cuda)
        # Compile for the selected visible device (or an explicitly selected
        # test target). nvcc supplies its own host-target CUDA library paths.
        run([nvcc, "-O3", "--fmad=false", "-std=c++17", "-arch=" + os.environ.get("FORT_TEST_CUDA_ARCH", "native"),
             "-ccbin", host, "-Xcompiler=-fopenmp", "-c", "implementation.cu", "-o", "implementation.o"])
        run([fc, "-O3", "-fopenmp", "-c", "interface.f90", "driver.f90"])
        run([nvcc, "-ccbin", host, "-Xcompiler=-fopenmp", "implementation.o", "interface.o", "driver.o",
             "-lgfortran", "-o", "generated"])
    actual = [float(value) for value in run([str(tmp_path / "generated")]).split()]
    assert len(actual) == len(expected) == 72
    for value, reference in zip(actual, expected, strict=True):
        if math.isnan(reference):
            assert math.isnan(value)
        else:
            assert value == reference
            if reference == 0:
                assert math.copysign(1, value) == math.copysign(1, reference)
