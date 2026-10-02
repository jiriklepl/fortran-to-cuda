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
from compiler.emission.c.generator import generate_cpp
from compiler.emission.common.abi import abi_arguments
from compiler.emission.common.resources import read_common_header
from compiler.emission.cuda.kernels import generate_kernel, generate_launch, kernel_name
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
    from compiler.emission import generate_sources

    function, plan = _example()
    scheduled = schedule_plan(function, plan, options=CompilerOptions(schedule="source"))
    assert generate_sources(function, plan) == generate_sources(function, scheduled)


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


@pytest.mark.native
@pytest.mark.parametrize("backend", ["serial", "openmp", "cuda-coordinate-map"])
@pytest.mark.parametrize("tiles", [(), (2,), (2, 4, 3, 2, 3), (512, 8)])
def test_native_tails_signed_strides_arbitrary_rank(tmp_path: Path, backend: str, tiles: tuple[int, ...]) -> None:
    shape = (9, 5, 3, 2, 3)
    bounds = ((9, 1, -2), (1, 5, 1), (3, 1, -1), (1, 2, 1), (1, 3, 1))
    function, plan = _example(5, bounds)
    plan = schedule_plan(function, plan, options=CompilerOptions(tile_sizes=tiles))
    if backend == "cuda-coordinate-map":
        region = plan.regions[0]
        kernel = "\n".join(generate_kernel(region))
        extents = tuple(len(range(lower, upper + (1 if step > 0 else -1), step)) for lower, upper, step in bounds)
        schedule = region.schedule
        tile_sizes = schedule.tile_sizes
        total = (
            math.prod((extent - 1) // size + 1 for extent, size in zip(extents, tile_sizes, strict=True))
            if tile_sizes
            else math.prod(extents)
        )
        arguments = ["a", *(str(size) for size in shape)]
        for (lower, _, stride), extent in zip(bounds, extents, strict=True):
            arguments.extend((str(lower), str(stride), str(extent)))
        arguments.append(str(total))
        if tile_sizes:
            arguments.append(
                str(math.prod(min(size, extent) for size, extent in zip(tile_sizes, extents, strict=True)))
            )
        source = (
            '#include <cstdlib>\n#include "support.cuh"\n#define __global__\n'
            "struct dimension { unsigned x; };\n"
            "dimension blockIdx, threadIdx, blockDim{3}, gridDim{2};\n"
            "using namespace generated_kernels::indexing;\n" + kernel + '\nextern "C" void test(int* a) {\n'
            "  for (blockIdx.x = 0; blockIdx.x < gridDim.x; ++blockIdx.x)\n"
            "    for (threadIdx.x = 0; threadIdx.x < blockDim.x; ++threadIdx.x)\n"
            f"      {kernel_name(region)}({', '.join(arguments)});\n"
            "}\n"
        )
    else:
        source = generate_cpp(function, plan, abi_arguments(function.parameters), "support.cuh")
    library = _compile_shared(tmp_path, source, openmp=backend == "openmp")
    values = (ctypes.c_int * math.prod(shape))(*([-1] * math.prod(shape)))
    if backend == "cuda-coordinate-map":
        procedure = library.test
        procedure.argtypes = [ctypes.POINTER(ctypes.c_int)]
        procedure(values)
    else:
        procedure = library.cpp_example
        procedure.argtypes = [ctypes.POINTER(ctypes.c_int), *([ctypes.c_size_t] * len(shape))]
        procedure(values, *shape)
    assert list(values) == _expected(shape, bounds)


@pytest.mark.native
@pytest.mark.parametrize("tiles", [(), (3, 2)])
def test_empty_outer_loop_suppresses_invalid_inner_stride(tmp_path: Path, tiles: tuple[int, ...]) -> None:
    function, plan = _example(2, ((2, 1, 1), (1, 3, 0)))
    plan = schedule_plan(function, plan, options=CompilerOptions(tile_sizes=tiles))
    source = generate_cpp(function, plan, abi_arguments(function.parameters), "support.cuh")
    library = _compile_shared(tmp_path, source)
    values = (ctypes.c_int * 9)(*([-1] * 9))
    library.cpp_example.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_size_t, ctypes.c_size_t]
    library.cpp_example(values, 3, 3)
    assert list(values) == [-1] * 9


@pytest.mark.native
@pytest.mark.cuda
def test_tiled_cuda_launch_compiles(tmp_path: Path) -> None:
    from compiler.emission.cuda.generator import generate_cuda

    compiler = shutil.which("nvcc")
    if compiler is None:
        pytest.skip("Native capability unavailable: nvcc is not installed")
    function, plan = _scheduled(5, tile_sizes=(512, 8, 2, 3, 4))
    (tmp_path / "support.cuh").write_text(read_common_header())
    (tmp_path / "test.cu").write_text(generate_cuda(function, plan, abi_arguments(function.parameters), "support.cuh"))
    result = subprocess.run(
        [compiler, "-std=c++17", "-c", "test.cu", "-o", "test.o"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


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
