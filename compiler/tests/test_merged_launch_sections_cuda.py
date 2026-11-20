"""Two regions that write one array share a launch while their actual sections stay apart."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from compiler.driver.options import CompilerOptions
from compiler.driver.pipeline import prepare_function
from compiler.emission import read_common_header
from compiler.emission.common.resources import read_scoped_runtime
from compiler.emission.cuda.scoped import generate_scoped
from compiler.frontend import lower_file
from compiler.offload.config import OffloadConfig

# Whether [1,3] and [n-1,n] meet depends on n, so the compiler leaves it to the runtime.
SOURCE = """module faces
contains
subroutine entry(u,x,n)
real(8),intent(inout)::u(:)
real(8),intent(in)::x(:)
integer,intent(in)::n
integer::i
do i=1,3
u(i)=x(i)
enddo
do i=n-1,n
u(i)=2*x(i)
enddo
end subroutine
end module
"""

DRIVER = r"""
#include "scoped_runtime.h"
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
extern "C" int ENTRY(fort_scope_t,int,fort_buffer_t,fort_buffer_t,const int*);
void check(int status) {
    if(status) { std::fprintf(stderr,"status=%d: %s\n",status,fort_scope_error()); std::abort(); }
}
int main() {
    int count=0; if(cudaGetDeviceCount(&count)!=cudaSuccess || !count) return 77;
    double u[12],x[12];
    for(int i=0;i<12;++i) { u[i]=-1; x[i]=i+1; }
    size_t extent[1]={12}; int64_t lower[1]={1};
    fort_scope_t context=0; fort_buffer_t handles[2]{};
    check(fort_scope_create(0,&context));
    double *values[2]={u,x};
    for(int i=0;i<2;++i) {
        fort_scope_layout layout{1,FORT_SCOPE_REAL64,8,values[i],extent,lower,1};
        check(fort_scope_register(context,i+1,1,&layout,1,&handles[i]));
    }
    fort_scope_graph_queue_stats_v1 queued{};
    // The sections [1,3] and [9,10] are apart: one launch runs both regions.
    int n=10;
    check(ENTRY(context,1,handles[0],handles[1],&n));
    check(fort_scope_graph_queue_stats_get_v1(context,&queued));
    assert(queued.accepted_tasks==1);
    check(fort_scope_observe_v1(context,&handles[0],1));
    for(int i=0;i<12;++i) assert(u[i]==(i<3 ? x[i] : i==8 || i==9 ? 2*x[i] : -1));
    // The sections [1,3] and [3,4] meet: the regions run one after the other, and the second one's value stays.
    n=4;
    check(ENTRY(context,1,handles[0],handles[1],&n));
    check(fort_scope_graph_queue_stats_get_v1(context,&queued));
    assert(queued.accepted_tasks==3);
    check(fort_scope_observe_v1(context,&handles[0],1));
    for(int i=0;i<12;++i) assert(u[i]==(i<2 ? x[i] : i==2 || i==3 || i==8 || i==9 ? 2*x[i] : -1));
    check(fort_scope_close(context));
    return 0;
}
"""


@pytest.mark.cuda
def test_the_runtime_decides_whether_sections_of_one_array_share_a_launch(tmp_path):
    nvcc = shutil.which("nvcc")
    host = shutil.which("g++-14") or shutil.which("g++")
    if not nvcc or not host:
        pytest.skip("CUDA toolkit and host C++ compiler required")
    source = tmp_path / "original.f90"
    source.write_text(SOURCE)
    function, plan = prepare_function(lower_file(source, "entry"), options=CompilerOptions(gpu_policy="sections"))
    generated = generate_scoped(function, plan, OffloadConfig(), "common_functions.cuh")
    assert [group["regions"] for group in generated.report["graph_execution"]["groups"]
            if len(group["regions"]) > 1] == [[0, 1]]
    artifacts, _manifest = read_scoped_runtime()
    for name, content in artifacts.items():
        (tmp_path / name).write_text(content)
    (tmp_path / "common_functions.cuh").write_text(read_common_header())
    (tmp_path / "entry.cu").write_text(generated.cuda)
    (tmp_path / "driver.cu").write_text(DRIVER.replace("ENTRY", generated.report["entry"]))
    target = tmp_path / "test"
    build = subprocess.run(
        [nvcc, "-std=c++17", "-ccbin", host, "-Xcompiler=-fopenmp", "-Xcompiler=-ffp-contract=off", "--fmad=false",
         "entry.cu", "scoped_runtime.cu", "driver.cu", "-o", str(target)],
        cwd=tmp_path, capture_output=True, text=True, timeout=300)
    assert build.returncode == 0, build.stdout + build.stderr
    result = subprocess.run([str(target)], capture_output=True, text=True, timeout=60)
    if result.returncode == 77:
        pytest.skip("CUDA device unavailable")
    assert result.returncode == 0, result.stdout + result.stderr
