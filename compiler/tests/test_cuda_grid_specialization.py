"""Check covering-grid dispatch against the unchanged bounded grid-stride path."""

from __future__ import annotations

import ctypes
import json
import re
import shutil
import subprocess
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

from compiler.emission.common.resources import read_common_header
from compiler.emission.cuda.kernels import generate_kernel, generate_launch, kernel_name
from compiler.ir import RegionSchedule
from compiler.numerical_contract import cuda_compile_options
from compiler.offload.numerical_gpu_execution import gpu_generator_identity, gpu_renderer_sources
from compiler.tests.test_cuda_coordinate_quotient import SENTINEL, _case
from compiler.tests.test_scheduling import _compile_shared


def _source(*, cuda, threads):
    region, _, _ = _case(1, (0,), False, 777)
    region = replace(region, schedule=RegionSchedule((0,), cuda_threads=threads))
    launch = "\n".join(generate_launch(
        region, profile=False, prepared_bounds=True,
        error_check="check(cudaGetLastError()); ++errors;" if cuda else "++errors;",
        launch_record="++records;",
    ))
    support = [
        '#include <cstdlib>\n#include <cstdio>\n#include "support.cuh"',
        "using namespace generated_kernels::indexing;",
    ]
    if cuda:
        support += [
            "void check(cudaError_t status) {",
            '    if(status!=cudaSuccess) { std::fprintf(stderr,"%s\\n",cudaGetErrorString(status)); std::exit(1); }',
            "}",
        ]
    else:
        support += [
            "#define __global__",
            "struct dimension { unsigned x; };",
            "dimension blockIdx, threadIdx, blockDim, gridDim;",
        ]
    support.extend(generate_kernel(region))
    support.append("unsigned records=0, errors=0, selected=0, launches=0, blocks_seen=0, threads_seen=0;")
    if not cuda:
        support += [
            "template<bool GridStride, typename... Args>",
            "void dispatch(unsigned blocks,unsigned threads,Args... args) {",
            "    ++launches; selected=GridStride ? 2 : 1; blocks_seen=blocks; threads_seen=threads;",
            "    gridDim.x=blocks; blockDim.x=threads;",
            "    for(blockIdx.x=0;blockIdx.x<blocks;++blockIdx.x)",
            "        for(threadIdx.x=0;threadIdx.x<threads;++threadIdx.x)",
            f"            {kernel_name(region)}<GridStride>(args...);",
            "}",
        ]
        launch = re.sub(
            r"kernel_region_777_device<(true|false)><<<([^>]+)>>>\(", r"dispatch<\1>(\2, ", launch,
        )
    support += [
        'extern "C" void run(int *out,int *visits,std::size_t n) {',
        "    records=errors=selected=launches=blocks_seen=threads_seen=0;",
        "    int *fort_v0_a_device=out+1,*fort_v100_visits_device=visits+1;",
        "    std::size_t fort_v0_a_dim1=n,fort_v100_visits_dim1=n;",
        "    int fort_internal_lower0=-9,fort_internal_stride0=2;",
        "    std::size_t fort_internal_extent0=n,fort_internal_total=n;",
        launch,
        "}",
        'extern "C" unsigned status(unsigned index) {',
        "    unsigned values[]={records,errors,selected,launches,blocks_seen,threads_seen};",
        "    return values[index];",
        "}",
    ]
    return "\n".join(support)


@pytest.fixture(scope="module")
def dispatch_program(tmp_path_factory):
    return _compile_shared(tmp_path_factory.mktemp("grid_dispatch"), _source(cuda=False, threads=1))


@pytest.mark.native
@pytest.mark.parametrize("count", [0, 1, 2, 65534, 65535, 65536, 65537, 131073])
def test_actual_launch_dispatch_covers_both_sides_of_block_cap(dispatch_program, count):
    values = (ctypes.c_int * (count + 2))(*([SENTINEL] * (count + 2)))
    visits = (ctypes.c_int * (count + 2))()
    dispatch_program.run.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_size_t]
    dispatch_program.status.argtypes = [ctypes.c_uint]
    dispatch_program.status.restype = ctypes.c_uint
    dispatch_program.run(values, visits, count)
    assert list(values) == [SENTINEL, *(-9 + 2 * point for point in range(count)), SENTINEL]
    assert list(visits) == [0, *([1] * count), 0]
    expected = [1, 1, 1 if count <= 65535 else 2, 1, min(count, 65535), 1] if count else [0] * 6
    assert [dispatch_program.status(index) for index in range(6)] == expected


def test_numerical_body_is_shared_and_tiled_kernels_do_not_gain_variants():
    region, _, _ = _case(2, (1, 0), False, 777)
    kernel = "\n".join(generate_kernel(region))
    assert kernel.count("__global__ void") == 1
    assert kernel.count("fort_v0_a[F_IDX(") == 1
    assert "if constexpr (!fort_internal_grid_stride_enabled) break;" in kernel
    launch = "\n".join(generate_launch(region, stream="stream", profile=True))
    assert launch.count("timing::measure_kernel_executions") == 1
    assert launch.count("cudaGetLastError()") == 1
    assert launch.count('storage::trace("kernel")') == 1
    assert launch.count("<<<fort_internal_blocks, fort_internal_threads, 0, stream>>>") == 2
    tiled = replace(region, schedule=RegionSchedule((1, 0), (3, 2)))
    assert "template<" not in "\n".join(generate_kernel(tiled))
    assert "if constexpr" not in "\n".join(generate_kernel(tiled))
    assert "<false>" not in "\n".join(generate_launch(tiled))
    assert "<true>" not in "\n".join(generate_launch(tiled))


def test_kernel_renderer_change_invalidates_numerical_gpu_identity():
    renderer = gpu_renderer_sources()
    path = "emission/cuda/kernels.py"
    assert renderer[path] == sha256((Path(__file__).parents[1] / path).read_bytes()).hexdigest()
    altered = {**renderer, path: "0" * 64}
    assert gpu_generator_identity("a" * 64, "b" * 64, renderer) != gpu_generator_identity("a" * 64, "b" * 64, altered)


@pytest.mark.cuda
def test_covering_and_grid_stride_specializations_execute_on_cuda(tmp_path):
    nvcc = shutil.which("nvcc")
    host = shutil.which("g++-14") or shutil.which("g++")
    if not nvcc or not host:
        pytest.skip("CUDA and C++ toolchains required")
    (tmp_path / "support.cuh").write_text(read_common_header())
    main = r"""
int main() {
    int devices=0;
    if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices) return 77;
    constexpr std::size_t cap=65535*32, maximum=cap+33;
    int *out=nullptr,*visits=nullptr;
    check(cudaMalloc(&out,(maximum+2)*sizeof(int)));
    check(cudaMalloc(&visits,(maximum+2)*sizeof(int)));
    int *actual=new int[maximum+2],*seen=new int[maximum+2];
    for(std::size_t n : {std::size_t(0),std::size_t(1),std::size_t(31),std::size_t(32),
                        std::size_t(33),cap,cap+1,cap+33,std::size_t(7)}) {
        for(std::size_t i=0;i<n+2;++i) actual[i]=-87654321;
        check(cudaMemcpy(out,actual,(n+2)*sizeof(int),cudaMemcpyHostToDevice));
        check(cudaMemset(visits,0,(n+2)*sizeof(int)));
        run(out,visits,n);
        check(cudaDeviceSynchronize());
        check(cudaMemcpy(actual,out,(n+2)*sizeof(int),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(seen,visits,(n+2)*sizeof(int),cudaMemcpyDeviceToHost));
        if(records!=(n?1:0) || errors!=(n?1:0)) return 2;
        for(std::size_t i=0;i<n+2;++i) {
            int expected=(i && i<=n) ? -9+2*int(i-1) : -87654321;
            if(actual[i]!=expected || seen[i]!=(i && i<=n ? 1 : 0)) return 3;
        }
    }
    check(cudaFree(out));check(cudaFree(visits));delete[] actual;delete[] seen;
    std::puts("COVERING_AND_GRID_STRIDE_OK");
}
"""
    source = tmp_path / "grid.cu"
    source.write_text('#include <cuda_runtime.h>\n' + _source(cuda=True, threads=32) + main)
    executable = tmp_path / "grid"
    command = [nvcc, "-O3", "-std=c++17", *cuda_compile_options(), "-arch=native", "-ccbin", host,
               str(source), "-o", str(executable)]
    build = subprocess.run(command, capture_output=True, text=True, timeout=120)
    commands = [{"argv": command, "exit_code": build.returncode, "stdout": build.stdout, "stderr": build.stderr}]
    (tmp_path / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    assert build.returncode == 0, build.stdout + build.stderr
    result = subprocess.run([str(executable)], capture_output=True, text=True, timeout=60)
    commands.append({"argv": [str(executable)], "exit_code": result.returncode,
                     "stdout": result.stdout, "stderr": result.stderr})
    (tmp_path / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    if result.returncode == 77:
        pytest.skip("CUDA device required")
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "COVERING_AND_GRID_STRIDE_OK"
