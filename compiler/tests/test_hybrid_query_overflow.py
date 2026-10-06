"""Oversized hybrid queries decline before touching captured storage."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from compiler.driver.options import CompilerOptions
from compiler.driver.pipeline import prepare_function
from compiler.emission import generate_sources, read_common_header
from compiler.frontend import lower_file
from compiler.offload.config import OffloadConfig

SOURCE = """module enormous_domain
contains
subroutine advance(a,b,n,m,l)
real(8),intent(inout)::a(:,:,:)
real(8),intent(in)::b(:,:,:)
integer,intent(in)::n,m,l
integer::i,j,k
do k=1,l
 do j=1,m
  do i=1,n
   a(i,j,k)=b(i,j,k)*2.0_8
  end do
 end do
end do
end subroutine
end module
"""


def generate(tmp_path, policy="chunked", collective=False):
    source = tmp_path / "input.f90"
    source.write_text(SOURCE)
    function, plan = prepare_function(lower_file(source, "advance"), options=CompilerOptions(gpu_policy=policy))
    return generate_sources(
        function, plan, common_header="runtime.hpp",
        offload_config=OffloadConfig(policy, collective=collective),
    )


def test_only_query_product_overflow_returns_native(tmp_path):
    generated = generate(tmp_path)
    selector = generated.cuda.split("_select(", 1)[1].split("_dispatch(", 1)[0]
    assert "Iteration size product overflows size_t" not in selector
    assert "std::abort()" not in selector
    assert "static_cast<std::size_t>(-1) / (fort_internal_extent" in selector
    assert "return {};" in selector
    dispatcher = generated.cuda.split("_dispatch(", 1)[1]
    assert "Iteration size product overflows size_t" in dispatcher
    assert "std::abort()" in dispatcher


@pytest.mark.native
@pytest.mark.cuda
@pytest.mark.parametrize("policy", ["chunked", "hybrid"])
@pytest.mark.parametrize("collective", [False, True])
def test_cuda_hybrid_query_rejects_int_max_cubed_without_reading_arrays(tmp_path, policy, collective):
    nvcc = shutil.which(os.environ.get("NVCC", "/usr/local/cuda/bin/nvcc"))
    host = shutil.which(os.environ.get("CUDAHOSTCXX", "g++-14"))
    if not nvcc or not host:
        pytest.skip("requires CUDA and GNU C++")
    generated = generate(tmp_path, policy, collective)
    assert generated.offload["hybrid_available"], generated.offload
    query = "cpp_" + generated.offload["native_fallback_query"]
    (tmp_path / "runtime.hpp").write_text(read_common_header())
    (tmp_path / "generated.cu").write_text(generated.cuda)
    parallel = "#pragma omp parallel num_threads(4)" if collective else ""
    (tmp_path / "driver.cpp").write_text(f"""
#include <sys/mman.h>
#include <climits>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
extern "C" int {query}(double*, std::size_t, std::size_t, std::size_t,
                        const double*, std::size_t, std::size_t, std::size_t,
                        const int&, const int&, const int&);
int main() {{
    static_assert(sizeof(std::size_t) >= 8);
    void *page = mmap(nullptr, 4096, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (page == MAP_FAILED) std::abort();
    auto *storage = static_cast<double*>(page);
    const int limit = INT_MAX;
    const std::size_t extent = INT_MAX;
    {parallel}
    {{
        if ({query}(storage, extent, extent, extent, storage, extent, extent, extent,
                    limit, limit, limit)) std::abort();
    }}
    if (munmap(page, 4096)) std::abort();
    std::puts("HYBRID_OVERFLOW_QUERY_PASS");
}}
""")
    environment = dict(os.environ, OMP_DYNAMIC="FALSE", FORT_OFFLOAD_TRACE="1", FORT_RUNTIME_TRACE="1")
    commands = [
        [nvcc, "-O2", "-std=c++17", "-Xcompiler=-fopenmp", "-arch=sm_86", "-ccbin", host,
         "-c", "generated.cu", "-o", "generated.o"],
        [host, "-O2", "-std=c++17", "-fopenmp", "driver.cpp", "generated.o",
         "-L" + str(Path(nvcc).resolve().parents[1] / "lib64"), "-lcudart", "-o", "verify"],
        [str(tmp_path / "verify")],
    ]
    (tmp_path / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    for command in commands:
        result = subprocess.run(command, cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
    log = result.stdout + result.stderr
    (tmp_path / "query-run.log").write_text(log)
    assert "HYBRID_OVERFLOW_QUERY_PASS" in result.stdout
    assert log.count("FORT_OFFLOAD entry=") == 1, log
    assert "mode=native" in result.stderr
    assert "FORT_RUNTIME" not in log
    assert "Iteration size product overflows" not in log
