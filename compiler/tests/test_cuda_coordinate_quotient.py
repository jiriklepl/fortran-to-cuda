"""Execute the emitted untiled decoder, including wide products, without CUDA."""

from __future__ import annotations

import ctypes
import itertools
import math
import subprocess
import sys
from dataclasses import replace

import pytest

from compiler.emission.common.schedules import checked_product
from compiler.emission.cuda.kernels import generate_kernel, generate_launch, kernel_name
from compiler.ir import ArrayAccess, Assignment, Binary, Block, Reference, RegionSchedule, ScalarType, Symbol
from compiler.tests.test_scheduling import LOCATION, _compile_shared, _example, _integer


def _cases():
    cases = []
    for rank in range(1, 6):
        natural = tuple(range(rank))
        orders = dict.fromkeys((natural, tuple(reversed(natural)), natural[1:] + natural[:1]))
        for order in orders:
            for reverse in (False, True):
                cases.append(pytest.param(rank, order, reverse, id=f"rank-{rank}-{order}-{reverse}"))
    return cases


CASES = _cases()
COUNTS = (5, 3, 4, 2, 3)
LOWERS = (-9, 8, -11, 6, -17)
SENTINEL = -87654321


def _case(rank, order, reverse, identifier):
    counts = COUNTS[:rank]
    steps = tuple((2 if axis % 2 == 0 else -3) * (-1 if reverse else 1) for axis in range(rank))
    bounds = tuple(
        (lower, lower + (count - 1) * step, step)
        for lower, count, step in zip(LOWERS[:rank], counts, steps, strict=True)
    )
    _function, plan = _example(rank, bounds)
    original = plan.regions[0]
    assignment = original.assignments[0]
    # Store by dense ordinal while retaining original signed logical indices
    # as the computed values. This checks coordinates separately from layout.
    indices = tuple(
        Binary("+", Binary("/", Binary("-", Reference(loop.iterator), _integer(lower)), _integer(step)), _integer(1))
        for loop, (lower, _upper, step) in zip(original.loops, bounds, strict=True)
    )
    assignment = replace(assignment, target=replace(assignment.target, indices=indices))
    visits = Symbol(100, "visits", ScalarType.INTEGER, rank, "inout", True)
    cell = ArrayAccess(visits, indices)
    count_visit = Assignment(cell, Binary("+", cell, _integer(1)), LOCATION)
    region = replace(
        original,
        id=identifier,
        assignments=(assignment, count_visit),
        body=Block((assignment, count_visit)),
        captured_symbols=(*original.captured_symbols, visits),
        schedule=RegionSchedule(order),
    )
    return region, bounds, counts


@pytest.fixture(scope="module")
def coordinate_program(tmp_path_factory):
    directory = tmp_path_factory.mktemp("coordinate_quotient")
    source = [
        '#include <cstdlib>\n#include "support.cuh"\n#define __global__',
        "struct dimension { unsigned x; };",
        "dimension blockIdx, threadIdx, blockDim, gridDim;",
        "using namespace generated_kernels::indexing;",
    ]
    cases = {}
    for identifier, parameter in enumerate(CASES):
        rank, order, reverse = parameter.values
        region, bounds, counts = _case(rank, order, reverse, identifier)
        cases[(rank, order, reverse)] = (identifier, bounds, counts)
        kernel = generate_kernel(region)
        source.extend(kernel)
        arguments = ["out+1", *(str(count) for count in counts), "visits+1", *(str(count) for count in counts)]
        for axis, ((lower, _upper, step), count) in enumerate(zip(bounds, counts, strict=True)):
            arguments.extend((str(lower), str(step), f"empty_axis == {axis} ? 0ULL : {count}ULL"))
        arguments.append(f"empty_axis < 0 ? {math.prod(counts)}ULL : 0ULL")
        source.extend(
            [
                f'extern "C" void case_{identifier}(int *out,int *visits,unsigned blocks,unsigned threads,int empty_axis) {{',
                "    gridDim.x=blocks; blockDim.x=threads;",
                "    for (blockIdx.x=0; blockIdx.x<blocks; ++blockIdx.x)",
                "        for (threadIdx.x=0; threadIdx.x<threads; ++threadIdx.x)",
                f"            {kernel_name(region)}({', '.join(arguments)});",
                "}",
            ]
        )
        # Compile the actual emitted decoder separately to sample huge domains
        # without allocating their arrays or iterating billions of work items.
        start = next(
            index
            for index, line in enumerate(kernel)
            if "std::size_t fort_internal_index = fort_internal_point;" in line
        )
        end = next(index for index in range(start + 1, len(kernel)) if "const int " in kernel[index])
        source.append(
            f'extern "C" void decode_{identifier}(std::size_t flat,const std::size_t *extents,std::size_t *out) {{'
        )
        source.append("    const std::size_t fort_internal_point=flat;")
        source.extend(f"    const std::size_t fort_internal_extent{axis}=extents[{axis}];" for axis in range(rank))
        source.extend(kernel[start:end])
        source.extend(f"    out[{axis}]=fort_internal_ordinal{axis};" for axis in range(rank))
        source.append("}")
    source.append('extern "C" std::size_t product(std::size_t a,std::size_t b,std::size_t c) {')
    source.extend(checked_product("total", ("a", "b", "c")))
    source.extend(["return total;", "}"])
    return _compile_shared(directory, "\n".join(source)), cases


@pytest.mark.native
@pytest.mark.parametrize(("rank", "order", "reverse"), CASES)
@pytest.mark.parametrize("grid", [(1, 1), (2, 3), (4, 7), (2, 64)])
def test_generated_coordinates_cover_each_signed_work_item_once(coordinate_program, rank, order, reverse, grid):
    library, cases = coordinate_program
    identifier, bounds, counts = cases[(rank, order, reverse)]
    total = math.prod(counts)
    values = (ctypes.c_int * (total + 2))(*([SENTINEL] * (total + 2)))
    visits = (ctypes.c_int * (total + 2))()
    function = getattr(library, f"case_{identifier}")
    function.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_int,
    ]
    function(values, visits, *grid, -1)
    expected = [SENTINEL] * (total + 2)
    for ordinals in itertools.product(*(range(count) for count in counts)):
        offset = sum(ordinal * math.prod(counts[:axis]) for axis, ordinal in enumerate(ordinals))
        expected[offset + 1] = sum(
            10**axis * (lower + ordinal * step)
            for axis, (ordinal, (lower, _upper, step)) in enumerate(zip(ordinals, bounds, strict=True))
        )
    assert list(values) == expected
    assert list(visits) == [0, *([1] * total), 0]


@pytest.mark.native
@pytest.mark.parametrize(("rank", "order", "reverse"), CASES)
def test_empty_extent_at_each_schedule_position_skips_the_decoder(coordinate_program, rank, order, reverse):
    library, cases = coordinate_program
    identifier, _bounds, counts = cases[(rank, order, reverse)]
    total = math.prod(counts)
    function = getattr(library, f"case_{identifier}")
    function.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_int,
    ]
    for axis in order:
        values = (ctypes.c_int * (total + 2))(*([SENTINEL] * (total + 2)))
        visits = (ctypes.c_int * (total + 2))()
        function(values, visits, 2, 3, axis)
        assert list(values) == [SENTINEL] * (total + 2)
        assert not any(visits)


WIDE_EXTENTS = {
    1: (2**32,),
    2: (2**32, 2**32 - 1),
    3: (65535, 65537, 2**32),
    4: (65535, 65537, 65536, 65536),
    5: (65535, 65537, 256, 256, 65536),
}


@pytest.mark.native
@pytest.mark.parametrize(("rank", "order", "reverse"), CASES)
def test_emitted_decoder_matches_divmod_near_size_t_limits(coordinate_program, rank, order, reverse):
    if ctypes.sizeof(ctypes.c_size_t) != 8:
        pytest.skip("64-bit size_t required for the wide-domain fixture")
    library, cases = coordinate_program
    identifier, _bounds, _counts = cases[(rank, order, reverse)]
    extents = WIDE_EXTENTS[rank]
    total = math.prod(extents)
    assert total <= ctypes.c_size_t(-1).value
    function = getattr(library, f"decode_{identifier}")
    function.argtypes = [ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    physical = (ctypes.c_size_t * rank)(*extents)
    points = {0, 1, extents[order[0]] - 1, extents[order[0]], 2**32 - 1, 2**32, total // 2, total - 2, total - 1}
    for flat in sorted(point for point in points if point < total):
        observed = (ctypes.c_size_t * rank)()
        function(flat, physical, observed)
        quotient = flat
        expected = [0] * rank
        for axis in order:
            quotient, expected[axis] = divmod(quotient, extents[axis])
        assert list(observed) == expected


@pytest.mark.native
@pytest.mark.parametrize(
    ("factors", "valid"), [((2**64 - 1, 1, 1), True), ((2**64 - 1, 2, 0), True), ((2**64 - 1, 2, 1), False)]
)
def test_launch_product_keeps_empty_first_and_overflow_rejection(coordinate_program, factors, valid):
    if ctypes.sizeof(ctypes.c_size_t) != 8:
        pytest.skip("64-bit size_t required for the checked-product fixture")
    library, _cases = coordinate_program
    # An overflow intentionally aborts in the existing launch guard. Isolate it
    # from pytest and suppress core files; no CUDA or large allocations occur.
    probe = """import ctypes,resource,sys
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
lib=ctypes.CDLL(sys.argv[1])
lib.product.argtypes=[ctypes.c_size_t]*3
lib.product.restype=ctypes.c_size_t
print(lib.product(*(int(value) for value in sys.argv[2:])))
"""
    result = subprocess.run(
        [sys.executable, "-c", probe, library._name, *(str(value) for value in factors)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if valid:
        assert result.returncode == 0, result.stderr
        assert int(result.stdout) == math.prod(factors)
    else:
        assert result.returncode != 0
        assert "Iteration size product overflows size_t" in result.stderr


def test_only_the_final_axis_uses_the_quotient_and_launch_policy_is_preserved():
    for identifier, parameter in enumerate(CASES):
        rank, order, reverse = parameter.values
        region, _bounds, _counts = _case(rank, order, reverse, identifier)
        kernel = "\n".join(generate_kernel(region))
        final = order[-1]
        assert f"fort_internal_ordinal{final} = fort_internal_index;" in kernel
        assert f"fort_internal_index % fort_internal_extent{final}" not in kernel
        for axis in order[:-1]:
            assert f"fort_internal_index % fort_internal_extent{axis}" in kernel
        launch = "\n".join(generate_launch(region))
        assert "constexpr unsigned int fort_internal_threads = 256;" in launch
        assert "fort_internal_needed_blocks > 65535 ? 65535" in launch
        assert "Iteration size product overflows size_t" in launch
