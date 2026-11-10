"""Execute emitted coordinates on native simulation and CUDA, including wide products."""

from __future__ import annotations

import ctypes
import itertools
import math
import shutil
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
        source.append("    const std::size_t fort_internal_total=" + "*".join(f"extents[{a}]" for a in range(rank)) + ";")
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


@pytest.mark.cuda
def test_actual_device_decoder_matches_full_width_divmod(tmp_path):
    nvcc, host = shutil.which("nvcc"), shutil.which("g++-14") or shutil.which("g++")
    if not nvcc or not host:
        pytest.skip("CUDA toolkit required")
    order = (4, 2, 0, 3, 1)
    region, _, _ = _case(5, order, False, 0)
    lines = generate_kernel(region)
    start = next(i for i, line in enumerate(lines) if "std::size_t fort_internal_index = fort_internal_point;" in line)
    end = next(i for i in range(start + 1, len(lines)) if "const int " in lines[i])
    decoder = "\n".join(lines[start:end])
    declarations = "\n".join(f"const std::size_t fort_internal_extent{a}=r.extents[{a}];" for a in range(5))
    stores = "\n".join(f"r.actual[{a}]=fort_internal_ordinal{a};" for a in range(5))
    source = DEVICE_DECODER.replace("// DECLARATIONS", declarations).replace("// DECODER", decoder).replace("// STORES", stores)
    path = tmp_path / "decoder.cu"
    path.write_text(source)
    binary = tmp_path / "decoder"
    compiled = subprocess.run([nvcc, "-O3", "-std=c++17", "-arch=sm_86", "-ccbin", host, str(path), "-o", str(binary)],
                              capture_output=True, text=True, timeout=120)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=60)
    if run.returncode == 77:
        pytest.skip("CUDA device unavailable")
    assert run.returncode == 0, run.stdout + run.stderr
    assert "DEVICE_DECODER_PASS" in run.stdout


DEVICE_DECODER = r'''
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <vector>
struct Row { std::size_t point,total,extents[5],actual[5]; };
__global__ void decode(Row *rows,std::size_t count) {
    for(std::size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=gridDim.x*blockDim.x) {
        auto &r=rows[i];const auto fort_internal_point=r.point,fort_internal_total=r.total;
        // DECLARATIONS
        // DECODER
        // STORES
    }
}
void check(cudaError_t status) { if(status!=cudaSuccess) { std::puts(cudaGetErrorString(status));std::abort(); } }
int main() {
    int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices)return 77;
    static_assert(sizeof(std::size_t)==8);
    std::vector<Row> rows;
    for(int a=0;a<5;++a)for(unsigned bit=0;bit<64;++bit)for(int delta=-1;delta<=1;++delta) {
        const auto extent=(std::size_t(1)<<bit)+delta;if(!extent)continue;
        for(auto point:{std::size_t(0),extent/2,extent-1}) {
            Row r{point,extent,{1,1,1,1,1},{}};r.extents[a]=extent;rows.push_back(r);
        }
    }
    std::uint64_t random=0x87356adfde071d33ULL;
    for(unsigned i=0;i<131072;++i) {
        random^=random<<13;random^=random>>7;random^=random<<17;
        std::size_t a=(random & 0xffffffffULL)|1,b=((random>>32)&0xffffffffULL)|1;
        if(i%3==0) { a=1+i%127;b=1+i%257; }
        Row r{random%(a*b),a*b,{1,1,1,1,1},{}};
        r.extents[i%5]=a;r.extents[(i+2)%5]=b;rows.push_back(r);
    }
    Row *device=nullptr;const auto bytes=rows.size()*sizeof(Row);
    check(cudaMalloc(&device,bytes));check(cudaMemcpy(device,rows.data(),bytes,cudaMemcpyHostToDevice));
    decode<<<71,128>>>(device,rows.size());check(cudaGetLastError());
    check(cudaMemcpy(rows.data(),device,bytes,cudaMemcpyDeviceToHost));check(cudaFree(device));
    for(const auto &r:rows) {
        auto quotient=r.point;
        for(int axis:{4,2,0,3,1}) {
            assert(r.actual[axis]==quotient%r.extents[axis]);quotient/=r.extents[axis];
        }
        assert(!quotient);
    }
    std::printf("DEVICE_DECODER_PASS %zu rows\n",rows.size());
}
'''
