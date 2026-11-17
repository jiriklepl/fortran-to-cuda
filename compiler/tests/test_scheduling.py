"""Check schedule selection and execute emitted coordinate maps without a GPU."""

from __future__ import annotations

import ctypes
import itertools
import math
import os
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from compiler.driver.options import CompilerOptions
from compiler.emission.common.resources import read_common_header
from compiler.emission.cuda.kernels import generate_kernel, generate_launch
from compiler.ir import (
    ArrayAccess,
    Assignment,
    Binary,
    Block,
    CompilationError,
    ConditionalRegion,
    ExecutionPlan,
    FunctionIR,
    Literal,
    Loop,
    ParallelRegion,
    Reference,
    RegionReport,
    ScalarType,
    SourceLocation,
    Symbol,
)
from compiler.scheduling import schedule_plan

LOCATION = SourceLocation("schedules.f90", 8)


def _integer(value: int) -> Literal:
    return Literal(str(value), ScalarType.INTEGER)


def _example(rank: int = 2, bounds: tuple[tuple[int, int, int], ...] | None = None) -> tuple[FunctionIR, ExecutionPlan]:
    array = Symbol(0, "a", ScalarType.INTEGER, rank, "inout", True)
    iterators = tuple(Symbol(axis + 1, f"i{axis}", ScalarType.INTEGER) for axis in range(rank))
    value = _integer(0)
    for axis, iterator in enumerate(iterators):
        value = Binary("+", value, Binary("*", _integer(10**axis), Reference(iterator)))
    assignment = Assignment(ArrayAccess(array, tuple(Reference(iterator) for iterator in iterators)), value, LOCATION)
    body = Block((assignment,))
    loops = []
    bounds = bounds or tuple((1, 3, 1) for _ in iterators)
    for iterator, (lower, upper, stride) in reversed(tuple(zip(iterators, bounds, strict=True))):
        loop = Loop(iterator, _integer(lower), _integer(upper), body, LOCATION, stride)
        loops.insert(0, loop)
        body = Block((loop,))
    function = FunctionIR("example", "test_schedule", (array,), (array, *iterators), body, "schedules.f90")
    region = ParallelRegion(
        0, tuple(loops), (assignment,), (), (array,), RegionReport("", "", "", "", "", "", ""), Block((assignment,))
    )
    return function, ExecutionPlan((region,))


def _scheduled(rank: int = 2, **kwargs) -> tuple[FunctionIR, ExecutionPlan]:
    function, plan = _example(rank)
    return function, schedule_plan(function, plan, options=CompilerOptions(**kwargs))


def test_auto_schedule_favors_fortran_contiguity_and_source_option_preserves_order() -> None:
    _, auto = _scheduled()
    _, source = _scheduled(schedule="source")
    _, level_zero = _scheduled(opt_level=0)
    _, explicit_auto = _scheduled(opt_level=0, schedule="auto")
    assert auto.regions[0].schedule.axis_order == (0, 1)
    assert source.regions[0].schedule.axis_order == (1, 0)
    assert level_zero.regions[0].schedule == source.regions[0].schedule
    assert explicit_auto.regions[0].schedule == auto.regions[0].schedule
    assert auto.regions[0].loops == source.regions[0].loops
    assert "fastest-first [i0,i1]" in auto.reports[0]


def test_unscheduled_plan_api_keeps_source_order() -> None:
    function, plan = _example()
    scheduled = schedule_plan(function, plan, options=CompilerOptions(schedule="source"))
    # The shared package records explicit plan provenance. Compare the actual
    # coordinate mapping and launch policy, independently of that metadata.
    assert generate_kernel(plan.regions[0]) == generate_kernel(scheduled.regions[0])
    assert generate_launch(plan.regions[0]) == generate_launch(scheduled.regions[0])


def test_writes_have_twice_the_locality_weight_of_reads() -> None:
    function, plan = _example()
    region = plan.regions[0]
    assignment = region.assignments[0]
    transposed = ArrayAccess(assignment.target.symbol, tuple(reversed(assignment.target.indices)))
    assignment = replace(assignment, value=transposed)
    region = replace(region, assignments=(assignment,), body=Block((assignment,)))
    scheduled = schedule_plan(function, ExecutionPlan((region,)), options=CompilerOptions())
    assert scheduled.regions[0].schedule.axis_order == (0, 1)
    assert "((2, 1), (1, 2))" in scheduled.reports[0]


def test_unknown_accesses_and_runtime_strides_receive_no_locality_credit() -> None:
    function, plan = _example()
    region = plan.regions[0]
    assignment = region.assignments[0]
    i, j = assignment.target.indices
    indirect = replace(assignment.target, indices=(Binary("*", i, i), j))
    assignment = replace(assignment, target=indirect)
    region = replace(region, assignments=(assignment,), body=Block((assignment,)))
    scheduled = schedule_plan(function, ExecutionPlan((region,)), options=CompilerOptions())
    assert scheduled.regions[0].schedule.axis_order == (1, 0)
    stride = Symbol(99, "stride", ScalarType.INTEGER, parameter=True)
    region = replace(plan.regions[0], loops=tuple(replace(loop, step=Reference(stride)) for loop in region.loops))
    scheduled = schedule_plan(function, ExecutionPlan((region,)), options=CompilerOptions())
    assert scheduled.regions[0].schedule.axis_order == (1, 0)


def test_tile_prefix_is_mapped_to_scheduled_axes() -> None:
    _, auto = _scheduled(3, tile_sizes=(7, 2))
    _, source = _scheduled(3, schedule="source", tile_sizes=(7, 2))
    assert auto.regions[0].schedule.tile_sizes == (7, 2, 1)
    assert source.regions[0].schedule.tile_sizes == (1, 2, 7)


def test_schedule_handles_conditional_branches_and_mixed_ranks() -> None:
    function, plan = _example(3)
    _, branch = _example(1)
    conditional = ConditionalRegion(_integer(1), branch, ExecutionPlan(()), LOCATION)
    nested = ExecutionPlan((*plan.steps, conditional))
    scheduled = schedule_plan(function, nested, options=CompilerOptions(tile_sizes=(7, 2)))
    assert len(scheduled.regions) == 2
    assert scheduled.regions[1].schedule.tile_sizes == (7,)


@pytest.mark.parametrize("sizes", [(1, 2, 3, 4), (2**64,)])
def test_invalid_tile_specification_is_rejected(sizes: tuple[int, ...]) -> None:
    function, plan = _example(3)
    with pytest.raises(CompilationError, match="Tile"):
        schedule_plan(function, plan, options=CompilerOptions(tile_sizes=sizes))


def test_cuda_launch_is_bounded_and_products_are_checked() -> None:
    for tiles in ((), (32, 16)):
        _, plan = _scheduled(tile_sizes=tiles)
        launch = "\n".join(generate_launch(plan.regions[0]))
        kernel = "\n".join(generate_kernel(plan.regions[0]))
        assert "fort_internal_needed_blocks > 65535 ? 65535" in launch
        assert "overflows size_t" in launch
        assert "while (fort_internal_" in kernel
        assert "gridDim.x" in kernel
        if tiles:
            assert "fort_internal_point += blockDim.x" in kernel
            assert "fort_internal_ordinal0 < fort_internal_extent0" in kernel


def _compile_shared(tmp_path: Path, source: str, *, openmp: bool = False) -> ctypes.CDLL:
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("Native capability unavailable: g++ is not installed")
    (tmp_path / "support.cuh").write_text(read_common_header())
    (tmp_path / "test.cpp").write_text(source)
    library = tmp_path / "test.so"
    result = subprocess.run(
        [compiler, "-std=c++17", "-O0", "-shared", "-fPIC", *("-fopenmp",) * openmp, "test.cpp", "-o", str(library)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    return ctypes.CDLL(str(library))


def _expected(shape: tuple[int, ...], bounds: tuple[tuple[int, int, int], ...]) -> list[int]:
    result = [-1] * math.prod(shape)
    ranges = [range(lower, upper + (1 if stride > 0 else -1), stride) for lower, upper, stride in bounds]
    for coordinates in itertools.product(*ranges):
        offset = sum((index - 1) * math.prod(shape[:axis]) for axis, index in enumerate(coordinates))
        result[offset] = sum(10**axis * index for axis, index in enumerate(coordinates))
    return result


def test_constant_integer_division_stride_receives_locality_credit() -> None:
    function, plan = _example()
    region = plan.regions[0]
    region = replace(
        region,
        loops=tuple(replace(loop, step=Binary("/", _integer(4), _integer(4))) for loop in region.loops),
    )
    scheduled = schedule_plan(function, ExecutionPlan((region,)), options=CompilerOptions())
    assert scheduled.regions[0].schedule.axis_order == (0, 1)
    assert "((2, 0), (0, 2))" in scheduled.reports[0]
