"""Address plans change subscript arithmetic without changing source value types."""

import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from compiler.emission.c.generator import _point_body as cpu_point_body
from compiler.emission.common.c_family import render_assignment, render_expression, wide_iterator_name
from compiler.emission.common.loops import iteration_value, mapped_coordinates, mapped_snapshots, sequential_block
from compiler.emission.cuda.kernels import _point_body as cuda_point_body
from compiler.ir import (
    ArrayAccess,
    Assignment,
    Binary,
    Block,
    If,
    IntegerRange,
    IntrinsicCall,
    Literal,
    Loop,
    ParallelRegion,
    Reference,
    RegionAddressing,
    RegionReport,
    ScalarType,
    Size,
    SourceLocation,
    SubscriptDecision,
    Symbol,
    Unary,
)
from compiler.ir.integers import INTEGER_MAX, INTEGER_MIN

ITERATOR = Symbol(0, "i", ScalarType.INTEGER)
J = Symbol(1, "j", ScalarType.INTEGER)
N = Symbol(2, "n", ScalarType.INTEGER, intent="in", parameter=True)
A = Symbol(3, "a", ScalarType.INTEGER, rank=1, intent="inout", parameter=True)
INDICES = Symbol(4, "indices", ScalarType.INTEGER, rank=1, intent="in", parameter=True)
LOCATION = SourceLocation("addressing.f90", 10)
REPORT = RegionReport("", "", "", "", "", "", "")


def integer(value):
    return Literal(str(value), ScalarType.INTEGER)


def address_plan(*wide, source=(), iterators=(ITERATOR,)):
    interval = IntegerRange(INTEGER_MIN, INTEGER_MAX)
    decisions = tuple(SubscriptDecision(expr, "wide", interval, "proved fixture") for expr in wide)
    decisions += tuple(SubscriptDecision(expr, "source", None, "uncertain fixture") for expr in source)
    return RegionAddressing(decisions, tuple((symbol, interval) for symbol in iterators), iterators)


def region(body, addressing=None):
    loop = Loop(ITERATOR, integer(2), integer(8), body, LOCATION)
    return ParallelRegion(0, (loop,), (), (J,), (A, INDICES, N), REPORT, body, addressing=addressing)


def test_selected_subscript_preserves_identical_scalar_expression():
    expression = Binary("+", Reference(ITERATOR), integer(1))
    assignment = Assignment(ArrayAccess(A, (expression,)), expression, LOCATION)
    rendered = render_assignment(assignment, addressing=address_plan(expression))
    assert (
        rendered
        == f"{A.cpp_name}[F_IDX(({wide_iterator_name(ITERATOR)} + 1LL), {A.cpp_name}_dim1)] = ({ITERATOR.cpp_name} + 1);"
    )
    assert render_expression(expression, addressing=address_plan(expression)) == render_expression(expression)


def test_selected_and_declined_subscripts_coexist():
    selected = Binary("-", Reference(ITERATOR), integer(1))
    declined = Binary("+", Reference(ITERATOR), Reference(N))
    matrix = replace(A, rank=2)
    addressing = address_plan(selected, source=(declined,))
    rendered = render_expression(ArrayAccess(matrix, (selected, declined)), addressing=addressing)
    assert f"({wide_iterator_name(ITERATOR)} - 1LL)" in rendered
    assert f"({ITERATOR.cpp_name} + {N.cpp_name})" in rendered
    assert render_expression(ArrayAccess(A, (declined,)), addressing=addressing) == render_expression(
        ArrayAccess(A, (declined,))
    )


def test_nested_array_load_can_keep_source_value_with_independently_wide_address():
    inner = ArrayAccess(INDICES, (Reference(ITERATOR),))
    rendered = render_expression(
        ArrayAccess(A, (inner,)), addressing=address_plan(Reference(ITERATOR), source=(inner,))
    )
    assert rendered == (
        f"{A.cpp_name}[F_IDX({INDICES.cpp_name}[F_IDX({wide_iterator_name(ITERATOR)}, {INDICES.cpp_name}_dim1)], "
        f"{A.cpp_name}_dim1)]"
    )
    assert "static_cast<long long>" not in rendered


def test_wide_outer_arithmetic_preserves_load_and_size_source_conversions():
    load = ArrayAccess(INDICES, (Reference(ITERATOR),))
    size = Size(A, 1)
    index = IntrinsicCall("min", (Reference(ITERATOR), load, size, Reference(N)), ScalarType.INTEGER)
    rendered = render_expression(ArrayAccess(A, (index,)), addressing=address_plan(index, Reference(ITERATOR)))
    assert f"static_cast<long long>({INDICES.cpp_name}[F_IDX({wide_iterator_name(ITERATOR)}," in rendered
    assert f"static_cast<long long>(static_cast<int>({A.cpp_name}_dim1))" in rendered
    assert f"static_cast<long long>({N.cpp_name})" in rendered
    assert rendered.count(INDICES.cpp_name + "[") == 1


def test_predicate_and_nested_body_accesses_promote_but_retained_headers_stay_source():
    index = Binary("+", Reference(ITERATOR), integer(1))
    access = ArrayAccess(INDICES, (index,))
    assignment = Assignment(ArrayAccess(A, (index,)), Reference(ITERATOR), LOCATION)
    conditional = If(Binary(">", access, integer(0)), Block((assignment,)), Block((assignment,)), LOCATION)
    retained = Loop(J, access, access, Block((conditional,)), LOCATION, access)
    addressing = address_plan(index)
    rendered = sequential_block(Block((retained,)), 0, [0], addressing=addressing)
    snapshots = [line.strip() for line in rendered if line.strip().startswith("const int fort_internal_")]
    assert len(snapshots) == 3
    assert all(render_expression(access) in line for line in snapshots)
    assert all("fort_internal_wide" not in line for line in snapshots)
    assert f"if ({render_expression(conditional.condition, addressing=addressing)}) {{" in "\n".join(rendered)
    assert "\n".join(rendered).count(render_assignment(assignment, addressing=addressing)) == 2


def test_source_and_missing_metadata_preserve_coordinate_and_body_text():
    index = Binary("+", Reference(ITERATOR), integer(1))
    assignment = Assignment(ArrayAccess(A, (index,)), Reference(ITERATOR), LOCATION)
    original = region(Block((assignment,)))
    source = replace(original, addressing=address_plan(source=(index,), iterators=()))
    assert cpu_point_body(original, 1) == cpu_point_body(source, 1)
    assert cuda_point_body(original, 1) == cuda_point_body(source, 1)
    assert mapped_snapshots(original) == mapped_snapshots(source)
    assert mapped_coordinates(original) == [
        f"const int {ITERATOR.cpp_name} = static_cast<int>(static_cast<long long>(fort_internal_lower0)"
        " + static_cast<long long>(fort_internal_ordinal0) * fort_internal_stride0);"
    ]
    assert iteration_value("_serial0", "ordinal") == (
        "static_cast<int>(static_cast<long long>(fort_internal_lower_serial0)"
        " + static_cast<long long>(ordinal) * fort_internal_stride_serial0)"
    )


def test_cpu_cuda_share_direct_wide_coordinate_reconstruction():
    index = Binary("+", Reference(ITERATOR), integer(1))
    assignment = Assignment(ArrayAccess(A, (index,)), Reference(ITERATOR), LOCATION)
    planned = region(Block((assignment,)), address_plan(index))
    aliases = mapped_coordinates(planned)
    assert aliases[1] == (
        f"const long long {wide_iterator_name(ITERATOR)} = static_cast<long long>(fort_internal_lower0)"
        " + static_cast<long long>(fort_internal_ordinal0) * fort_internal_stride0;"
    )
    assert "static_cast<int>" not in aliases[1]
    assert cpu_point_body(planned, 2) == cuda_point_body(planned, 2)
    assert mapped_snapshots(planned) == mapped_snapshots(replace(planned, addressing=None))


@pytest.mark.native
def test_promoted_integer_intrinsics_and_grouping_have_signed_wide_types(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("Native capability unavailable: g++")
    point = Reference(ITERATOR)
    loaded = ArrayAccess(INDICES, (point,))
    expressions = (
        (Binary("/", Unary("-", point), integer(3)), -2),
        (Binary("+", point, Unary("-", integer(-1))), 8),
        (Binary("*", Binary("+", point, integer(1)), integer(2)), 16),
        (IntrinsicCall("min", (integer(2), point), ScalarType.INTEGER), 2),
        (IntrinsicCall("max", (loaded, point), ScalarType.INTEGER), 7),
        (IntrinsicCall("abs", (Unary("-", point),), ScalarType.INTEGER), 7),
        (IntrinsicCall("min", (Size(A, 1), point), ScalarType.INTEGER), 4),
        (IntrinsicCall("min", (Reference(N), point), ScalarType.INTEGER), 5),
        (IntrinsicCall("min", (integer(INTEGER_MIN), point), ScalarType.INTEGER), INTEGER_MIN),
    )
    addressing = address_plan(point, *(expression for expression, _ in expressions))
    numeric = Path(__file__).parents[1] / "runtime" / "numeric.hpp"
    source = [
        "#include <cassert>",
        "#include <cstddef>",
        "#include <type_traits>",
        "#define CUDA_CALLABLE",
        numeric.read_text(),
        "long long recorded;",
        "template <typename Index, typename Extent> std::size_t F_IDX(Index index, Extent) {",
        "    static_assert(std::is_same_v<Index, long long>);",
        "    recorded = index; return 0;",
        "}",
        "int main() {",
        f"    const long long {wide_iterator_name(ITERATOR)} = 7;",
        f"    const int {N.cpp_name} = 5;",
        f"    int {A.cpp_name}[4] = {{}}, {INDICES.cpp_name}[4] = {{3, 3, 3, 3}};",
        f"    const std::size_t {A.cpp_name}_dim1 = 4, {INDICES.cpp_name}_dim1 = 4;",
    ]
    for expression, expected in expressions:
        access = render_expression(ArrayAccess(A, (expression,)), addressing=addressing)
        source.extend((f"    (void){access};", f"    assert(recorded == {expected}LL);"))
    source.append("}")
    (tmp_path / "test.cpp").write_text("\n".join(source))
    result = subprocess.run(
        [compiler, "-std=c++17", "-Wall", "-Werror", "test.cpp", "-o", "run"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run(["./run"], cwd=tmp_path, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
